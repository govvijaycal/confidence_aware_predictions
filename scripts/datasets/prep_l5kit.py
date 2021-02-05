import os
import glob
import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
from zarr import convenience 
from prettytable import PrettyTable
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset, select_agents
from l5kit.dataset.select_agents import TH_YAW_DEGREE, TH_EXTENT_RATIO, TH_DISTANCE_AV # 30, 1.1, 50
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from l5kit.data import PERCEPTION_LABELS

from tfrecord_utils import write_tfrecord, visualize_tfrecords, shuffle_test
##########################################
# Utils below for downsampling full l5kit dataset to a final set for TFRecord generation.
MIN_EGO_TRAJ_LENGTH = 5.0 # m, could add this to config file if desired.

def downsample_zarr_agents(zarr_dataset, frame_skip_interval = 10):
	""" Given a zarr dataset, downsamples frames by frame_skip_interval
	    in each scene and returns a mask with agents contained in
	    the downsampled frames.  E.g. frame_skip_interval=10 roughly
	    results in 1/10 of the agents being selected.

	    NOTE: This function was not used finally, since non-ego agent data was
	    not found to be reliable.  Extent can vary and positions can jump, resulting
	    in bad velocity estimates.  I think this is since the l5kit agent data
	    is raw perception output and not from annotations.
	"""
	agents_mask = np.zeros( len(zarr_dataset.agents), dtype=np.bool )
	frame_intervals = zarr_dataset.scenes['frame_index_interval']
	agent_intervals = zarr_dataset.frames['agent_index_interval']

	# First iterate over the frame intervals (i.e. the scenes).
	for (frame_st, frame_end) in frame_intervals:		
		# Then iterate over a downsampled set of frames in the current scene / frame interval.
		for frame_selected in range(frame_st, frame_end, frame_skip_interval):			
			# Mark agents in the selected frame as active.
			agent_active_st, agent_active_end = agent_intervals[frame_selected]
			agents_mask[agent_active_st:agent_active_end] = True

	return agents_mask

def downsample_zarr_frames(zarr_dataset, 
	                  frame_skip_interval = 10,
	                  num_history_frames=10,
	                  num_future_frames=50):
	""" Given a zarr dataset, downsamples frames by frame_skip_interval
	    in each scene and returns a mask with the selected frames.
	    E.g. frame_skip_interval=10 reduces sampling frequency by about 1/10.
	    num_history_frames and num_future_frames are used to avoid sampling
	    partial trajectories (i.e. all history/future data should be "available" in l5kit terms).
	"""
	frames_mask = np.zeros( len(zarr_dataset.frames), dtype=np.bool )
	frame_intervals = zarr_dataset.scenes['frame_index_interval']

	for (frame_st, frame_end) in frame_intervals:
		for frame_selected in range(frame_st + 10, frame_end - 50, frame_skip_interval):
			frames_mask[frame_selected] = True

	return frames_mask

def downsample_torch_length(torch_dataset, length_thresh=MIN_EGO_TRAJ_LENGTH, batch_size=64, num_workers=16):
	""" Given a torch EgoDataset/AgentDataset, compute the trajectory length for every data
		instance and return a mask for which ones exceed length_thresh.  This is slow-ish
		(approx. 1 hour on train_full after it has been downsampled by 10x), so results should
		be saved.  batch_size / num_workers used to speed up by using torch dataloader.
	"""
	dataloader = DataLoader(torch_dataset, 
		                    batch_size = batch_size,
		                    shuffle = False,
		                    num_workers=num_workers)
	num_instances = len(torch_dataset)
	lengths_mask = np.zeros( num_instances, dtype=np.bool)

	print(f'Computing the lengths mask - approx. {int(num_instances / batch_size)} iterations.')
	for bs_ind, instances in tqdm(enumerate(dataloader)):
		st_ind = bs_ind * batch_size
		end_ind = min(num_instances, (bs_ind + 1) * batch_size)
		diffs = instances['target_positions'][:, 1:, :] - instances['target_positions'][:, :-1, :]
		length = torch.sum(torch.norm(diffs, dim=2), dim=1) + torch.norm(instances['target_positions'][:, 0, :], dim=1)
		lengths_mask[st_ind:end_ind] = length.numpy() > length_thresh	
	return lengths_mask	
	
def stratified_sampling_length_curvature(torch_dataset, samples_per_part=8000, batch_size=64, num_workers=16):
	""" Given a torch EgoDataset/AgentDataset, compute the trajectory length and curvature for every data
		instance.  Then try to get a balanced dataset by equally sampling from partitions defined below.
		This tries to address overrepresentation of straight trajectorie and balance slow vs. fast speeds.
	"""
	dataloader = DataLoader(torch_dataset, batch_size = 64, shuffle = False, num_workers=16)
	num_instances = len(torch_dataset)

	lengths = np.ones( num_instances ) * np.nan
	curvs   = np.ones( num_instances ) * np.nan
	
	print(f'Computing lengths/curvatures - approx. {int(num_instances / batch_size)} iterations.')
	for bs_ind, instances in tqdm(enumerate(dataloader)):
		st_ind = bs_ind * batch_size
		end_ind = min( num_instances , (bs_ind + 1) * batch_size)
		
		# Length is taken by finding the cumulative Euclidean distance along the future trajectory.
		# We add the norm of the first target position as well as it it 1 timestep from the current one.
		diffs = instances['target_positions'][:, 1:, :] - instances['target_positions'][:, :-1, :]
		length = torch.sum(torch.norm(diffs, dim=2), dim=1) + torch.norm(instances['target_positions'][:, 0, :], dim=1)

		# Curvature crudely estimated by estimating as (final_yaw - 0) / length.
		# We assume this dataset has been prefiltered to avoid very short length trajectories.
		heading_final = torch.atan2(diffs[:, -1, 1], diffs[:, -1, 0])
		curv = heading_final / length
		
		curvs[st_ind:end_ind]   = curv.numpy()
		lengths[st_ind:end_ind] = length.numpy()

	# Hard-coded partitions and sampling based on examining the data.
	# Can make this more adaptive / user-friendly in the future.
	# One issue is if samples_per_part exceeds the number of samples in that partition - see try/except block.
	under_50 = lengths < 50	  # under 50 m long future trajectory
	above_50 = lengths >= 50  # over 50 m ""

	# 0.02 / 0.002 is approximately 1 std of curvature for that partition.
	# For train/val set, min partition size is 27236/3365 (part6).
	part1 = np.logical_and( under_50, np.abs(curvs) <= 0.02)
	part2 = np.logical_and( under_50, curvs > 0.02)
	part3 = np.logical_and( under_50, curvs < -0.02)
	part4 = np.logical_and( above_50, np.abs(curvs) <= 0.002)
	part5 = np.logical_and( above_50, curvs > 0.002)
	part6 = np.logical_and( above_50, curvs < -0.002)

	parts  = [part1, part2, part3, part4, part5, part6]
	# Check every instance has been allocated a partition and only once (disjoint).
	assert np.all( np.sum(np.array([p for p in parts]).astype(np.int), axis=0) == 1 ) 

	selected = np.zeros( num_instances, dtype=np.bool )
	np.random.seed(0)

	# Randomly sample samples_per_part in each partition
	# to make a balanced dataset across partitions.
	# This will not address imbalance within the partition, however.
	try:
		for ind_part, part in enumerate(parts):		
			print(f"Partition {ind_part+1} has {np.sum(part)} elements.")	
			samples = np.random.choice( np.ravel(np.argwhere(part > 0)), replace=False, size=samples_per_part)
			selected[samples] = True
	except Exception as e:
		print(e)
		import pdb; pdb.set_trace() # will be triggered if samples_per_part is too high

	return selected, lengths, curvs

##########################################
FINAL_DT_TFRECORD = 0.2
# Main function needed to extract a dictionary representation of a single element from a dataset.
def get_data_dict(torch_dataset,  # prefiltered/downsampled dataset that provides l5kit dataset instances
	              rasterizer,     # rasterizer to combine raw images from torch_dataset	              
                  dataset_index): # index (int) for which we want to get a dataset dict to write.
	if FINAL_DT_TFRECORD == 0.1:
		frame_skip = 1 # 10 Hz
	elif FINAL_DT_TFRECORD == 0.2:
		frame_skip = 2 # 5 Hz
	else:
		raise NotImplementedError("Only handling 0.1 s (10Hz) or 0.2 s (5 Hz) at the moment.")


	element = torch_dataset[dataset_index]
	# Unused keys: host_id, timestamp, extent, history_extents, future_extents
	#              *_velocities can be reconstructed using np.diff(..., axis=0) / 0.1
	#			   raster_from_agent: straightforward given cfg params, just need to flip y-axis
	#			   world_from_agent: similar functionality implemented in pose_utils.py
	#              speed - dropped since it involves peeking into the future poses

	# Sample in nuscenes terms = identifier for the timestamp in the dataset.
	sample = f"scene_{element['scene_index']}_frame_{element['frame_index']}"

	# Instance in nuscenes terms = identifier for the agent in that sample.
	instance = f"track_{element['track_id']}"
	if instance == f"track_{-1}":
		agent_type = 'ego'
	else:
		raise NotImplementedError("Not set up yet for arbitrary AgentDataset instances.")

	current_pose = np.append( element['centroid'], element['yaw'] )
	assert np.all(element['target_availabilities'] == 1), "Invalid future frame detected!"
	assert np.all(element['history_availabilities'] == 1), "Invalid past frame detected!"
	
	past_local_poses    = np.concatenate( (element['history_positions'], element['history_yaws']), axis=-1 )
	past_local_poses    = past_local_poses[frame_skip::frame_skip, :] # sample every frame_skip frame without the current pose
	future_local_poses  = np.concatenate( (element['target_positions'], element['target_yaws']), axis=-1 )
	future_local_poses  = future_local_poses[(frame_skip-1)::frame_skip, :] # sample every frame skip frame
	
	vel = np.linalg.norm( past_local_poses[0, :2] ) / FINAL_DT_TFRECORD
	vel_prev = np.linalg.norm( past_local_poses[1, :2] - past_local_poses[0, :2] ) / FINAL_DT_TFRECORD
	yaw_rate = -past_local_poses[0, 2] / FINAL_DT_TFRECORD
	accel = (vel - vel_prev) / FINAL_DT_TFRECORD

	past_tms   = np.array( [-FINAL_DT_TFRECORD * x for x in range(1, len(past_local_poses)+1)] )
	future_tms = np.array( [FINAL_DT_TFRECORD * x for x in range(1, len(future_local_poses)+1)] )

	img = rasterizer.to_rgb( element['image'].transpose(1,2,0) )

	return {'instance': instance,
	        'sample': sample,
	        'type': agent_type,
	        'pose': current_pose,
	        'velocity': vel,
	        'acceleration': accel,
	        'yaw_rate': yaw_rate,
	        'past_poses_local': past_local_poses,
	        'future_poses_local': future_local_poses,
	        'past_tms': past_tms,
	        'future_tms': future_tms,
	        'image': img}

##########################################	
if __name__ == '__main__':		
	parser = argparse.ArgumentParser('Read/Write L5Kit prediction instances in TFRecord format.')
	parser.add_argument('--mode', choices=['read', 'write', 'batch_test'], type=str, required=True, help='Write or read TFRecords.')
	parser.add_argument('--datadir', type=str, help='Where the TFRecords are located or should be saved.', \
		                    default=os.path.abspath(__file__).split('scripts')[0] + 'data')
	parser.add_argument('--dataroot', type=str, help='Location of the L5Kit dataset.', \
		                    default='/media/data/l5kit-data/')	
	args = parser.parse_args()
	datadir = args.datadir
	mode = args.mode
	dataroot = args.dataroot

	if mode == "write":
		dm = LocalDataManager(local_data_folder=dataroot)

		cfg = load_config_data('/'.join( os.path.abspath(__file__).split('/')[:-1]) + \
		                       '/l5kit_prediction_config.yaml')
		frames_history    = cfg['model_params']['history_num_frames']
		frames_future     = cfg['model_params']['future_num_frames']
		rasterizer = build_rasterizer(cfg, dm)

		for split in ['val', 'train']:
			zarr = ChunkedDataset(dm.require(cfg[f'{split}_data_loader']['key'])).open()
			ego_dataset = EgoDataset(cfg, zarr, None) # rasterizer not used to improve speed

			# 1st downsampling
			print(f"Dataset {split}: downsampling and removing trajectories " \
				  f"with length < {MIN_EGO_TRAJ_LENGTH} m.")
			frames_mask_path = Path( f"{datadir}/l5kit_{split}_ego_frames_mask.npy" )
			if not frames_mask_path.exists():
				frames_mask = downsample_zarr_frames(zarr,
					                                 frame_skip_interval=10,
					                                 num_history_frames=frames_history,
					                                 num_future_frames=frames_future)
				frame_ds_inds = np.nonzero(frames_mask)[0]
				frame_subset = torch.utils.data.Subset(ego_dataset, frame_ds_inds)
				lengths_mask = downsample_torch_length(frame_subset)	
				frames_mask[ frame_ds_inds ] = lengths_mask
				np.save(str(frames_mask_path), frames_mask)
			else:
				frames_mask = np.load(str(frames_mask_path))

			# 2nd downsampling: perform "stratified sampling" to resample length/curvature
			#                   distribution for reduced dataset imbalance.
			print(f"Dataset {split}: performing stratified sampling over length and curvature.")
			subset_mask_path = Path( f"{datadir}/l5kit_{split}_ego_frames_mask_subset.npy" )
			if not subset_mask_path.exists():
				frame_selected_inds = np.nonzero(frames_mask)[0]
				frame_subset = torch.utils.data.Subset(ego_dataset, frame_selected_inds)
				samples_per_part = 7500 if split is 'train' else 1500
				subsample_mask, lengths, curvs = \
				    stratified_sampling_length_curvature(frame_subset, samples_per_part=samples_per_part)
				frames_mask[ frame_selected_inds ] = subsample_mask
				subset_mask = frames_mask
				
				# We also save length / curvature to visualize stratified sampling selections later on.
				np.save( str(subset_mask_path), subset_mask )
				np.save( str(subset_mask_path).replace('subset', 'lengths'), lengths )
				np.save( str(subset_mask_path).replace('subset', 'curvs'), curvs )
			else:
				subset_mask = np.load(str(subset_mask_path))
			
			final_ego_dataset = torch.utils.data.Subset(EgoDataset(cfg, zarr, rasterizer), np.nonzero(subset_mask)[0])
			print(f"Dataset {split} has {len(final_ego_dataset)} instances out of {len(ego_dataset)}.")
			del ego_dataset

			data_dict_function = partial(get_data_dict, final_ego_dataset, rasterizer) 
			dataset_inds = np.arange(len(final_ego_dataset)).astype(np.int)
			file_prefix = f"{datadir}/l5kit_{split}"
			write_tfrecord(file_prefix,         
			               dataset_inds,             
			               data_dict_function,  
			               shuffle = True,      
			               shuffle_seed = 0,    
			               max_per_record = 1000)
	elif mode == 'read':
		train_set = glob.glob(datadir + '/l5kit_train*.record')
		visualize_tfrecords(train_set, max_batches=100)

	elif mode == 'batch_test':
		train_set = glob.glob(datadir + '/l5kit_train*.record')
		shuffle_test(train_set, batch_size=32)

		# Results: shows good shuffling (dataset size 32000)
		# Shuffle min range and std dev: 8923, 3914.881497426057
		# Shuffle max range and std dev: 31900, 12547.59303168894
		# Shuffle mean range and std dev: 23114.5870625, 8466.038283025504

	else:
		raise ValueError("Invalid mode: {}".format(mode))