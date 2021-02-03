import os
import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
from zarr import convenience 
from prettytable import PrettyTable
from pathlib import Path

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

def downsample_agents(zarr_dataset, frame_skip_interval = 10):

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

def downsample_frames(zarr_dataset, 
	                  frame_skip_interval = 10,
	                  num_history_frames=10,
	                  num_future_frames=50):
	frames_mask = np.zeros( len(zarr_dataset.frames), dtype=np.bool )
	frame_intervals = zarr_dataset.scenes['frame_index_interval']

	for (frame_st, frame_end) in frame_intervals:
		for frame_selected in range(frame_st + 10, frame_end - 50, frame_skip_interval):
			frames_mask[frame_selected] = True

	return frames_mask
	
if __name__ == '__main__':		
	parser = argparse.ArgumentParser('Read/Write L5Kit prediction instances in TFRecord format.')
	parser.add_argument('--mode', choices=['write', 'read'], type=str, required=True, help='Write or read TFRecords.')
	parser.add_argument('--datadir', type=str, help='Where the TFRecords are located or should be saved.', \
		                    default=os.path.abspath(__file__).split('scripts')[0] + 'data')
	parser.add_argument('--dataroot', type=str, help='Location of the L5Kit dataset.', \
		                    default='/media/data/l5kit-data/')	
	args = parser.parse_args()
	datadir = args.datadir
	mode = args.mode
	dataroot = args.dataroot

	dm = LocalDataManager(local_data_folder=dataroot)

	cfg = load_config_data('/'.join( os.path.abspath(__file__).split('/')[:-1]) + \
	                       '/l5kit_prediction_config.yaml')

	agent_prob        = cfg['raster_params']['filter_agents_threshold']
	frames_history    = cfg['model_params']['history_num_frames']
	frames_future     = cfg['model_params']['future_num_frames']

	filter_type       = cfg['agent_select_params']['filter_type']
	th_speed          = cfg['agent_select_params']['speed_thresh']

	rasterizer = build_rasterizer(cfg, dm)

	train_zarr = ChunkedDataset(dm.require(cfg['train_data_loader']['key'])).open()
	val_zarr   = ChunkedDataset(dm.require(cfg['val_data_loader']['key'])).open()

	rasterizer = None
	train_dataset = EgoDataset(cfg, train_zarr, rasterizer)
	frames_mask_path = Path( datadir + '/train_ego_frames_mask.npy' )
	if not frames_mask_path.exists():
		frames_mask = downsample_frames(train_zarr,
		                                frame_skip_interval=10,
		                                num_history_frames=frames_history,
		                                num_future_frames=frames_future)
		train_subset = torch.utils.data.Subset(train_dataset, np.nonzero(frames_mask)[0])
		train_dataloader = DataLoader(train_subset, batch_size = 16, shuffle = False, num_workers=16)
		lengths_mask = np.zeros( len(train_subset), dtype=np.bool)

		for bs_ind, instances in tqdm(enumerate(train_dataloader)):
			st_ind = bs_ind * 16
			end_ind = min(len(train_subset), (bs_ind + 1) * 16)
			diffs = instances['target_positions'][:, 1:, :] - instances['target_positions'][:, :-1, :]
			length = torch.sum(torch.norm(diffs, dim=2), dim=1) + torch.norm(instances['target_positions'][:, 0, :], dim=1)
			lengths_mask[st_ind:end_ind] = length.numpy() > 5.0 # LENGTH_MIN		
		frames_mask[ np.nonzero(frames_mask)[0] ] = lengths_mask
		np.save(str(frames_mask_path), frames_mask)
	else:
		frames_mask = np.load(str(frames_mask_path))

# Results of using a custom vehicle filter_type, 0.8 prob thresh, and 2.0 m/s speed thresh:
# ==============================
# Writing to /media/data/l5kit-data/scenes/train.zarr/agents_mask/0.8_vehicle_sp2.0
# start report for /media/data/l5kit-data/scenes/train.zarr
# {   'reject_th_AV_distance': 12772912,
#     'reject_th_agent_filter_probability_threshold': 289927706,
#     'reject_th_extent': 3206991,
#     'reject_th_yaw': 31244,
#     'th_agent_filter_probability_threshold': 0.8,
#     'th_distance_av': 50,
#     'th_extent_ratio': 1.1,
#     'th_yaw_degree': 30,
#     'total_agent_frames': 320124624,
#     'total_reject': 305938853}
# computing past/future table:
# +-------------+-----------+---------+---------+---------+
# | past/future |     0     |    10   |    30   |    50   |
# +-------------+-----------+---------+---------+---------+
# |      0      | 320124624 | 7047605 | 3379755 | 1836965 |
# |      10     |  7047605  | 4766174 | 2466861 | 1384948 |
# |      30     |  3379755  | 2466861 | 1384948 |  826962 |
# |      50     |  1836965  | 1384948 |  826962 |  518052 |
# +-------------+-----------+---------+---------+---------+
# end report for /media/data/l5kit-data/scenes/train.zarr
# Saved mask: 1384948 of 320124624
# Downsampling 31980917 of 320124624
# Combined: 133891  of 320124624
# 133891 0.000418246488904896
# ==============================
# Writing to /media/data/l5kit-data/scenes/validate.zarr/agents_mask/0.8_vehicle_sp2.0
# start report for /media/data/l5kit-data/scenes/validate.zarr
# {   'reject_th_AV_distance': 12900207,
#     'reject_th_agent_filter_probability_threshold': 282912089,
#     'reject_th_extent': 3158362,
#     'reject_th_yaw': 30026,
#     'th_agent_filter_probability_threshold': 0.8,
#     'th_distance_av': 50,
#     'th_extent_ratio': 1.1,
#     'th_yaw_degree': 30,
#     'total_agent_frames': 312617887,
#     'total_reject': 299000684}
# computing past/future table:
# +-------------+-----------+---------+---------+---------+
# | past/future |     0     |    10   |    30   |    50   |
# +-------------+-----------+---------+---------+---------+
# |      0      | 312617887 | 6679230 | 3170220 | 1704359 |
# |      10     |  6679230  | 4494943 | 2299981 | 1278493 |
# |      30     |  3170220  | 2299981 | 1278493 |  756509 |
# |      50     |  1704359  | 1278493 |  756509 |  469667 |
# +-------------+-----------+---------+---------+---------+
# end report for /media/data/l5kit-data/scenes/validate.zarr
# ==============================
# Saved mask: 1278493 of 312617887
# Downsampling 31214236 of 312617887
# Combined: 123273  of 312617887
# 123273 0.00039432484552619343

