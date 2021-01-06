import os
import numpy as np
from pyquaternion import Quaternion
from functools import partial

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.eval.common.utils import quaternion_yaw

from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.input_representation.interface import InputRepresentation 
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.combinators import Rasterizer

from tfrecord_utils import write_tfrecord, _parse_function

##########################################
# Helper functions to handle poses and angles.
def extract_pose_from_annotation(annotation):
	x, y, _ = annotation['translation']
	yaw     = quaternion_yaw( Quaternion(annotation['rotation']) )
	return x, y, yaw

def extract_poses(annotation_list):
	return np.array([extract_pose_from_annotation(ann) for ann in annotation_list])

def rotation_global_to_local(yaw):
	return np.array([[ np.cos(yaw), np.sin(yaw)], \
		             [-np.sin(yaw), np.cos(yaw)]])

def angle_mod_2pi(angle):
	return (angle + np.pi) % (2.0 * np.pi) - np.pi

def pose_diff_norm(pose_diff):
	# Not exactly a traditional norm but just meant to ensure no pose differences.
	xy_norm    = np.linalg.norm(pose_diff[:,:2], ord=np.inf)
	angle_norm = np.max( [angle_mod_2pi(x) for x in pose_diff[:,2]] )
	return xy_norm + angle_norm

def convert_global_to_local(global_pose_origin, global_poses):
	R_global_to_local = rotation_global_to_local(global_pose_origin[2])
	t_global_to_local = - R_global_to_local @ global_pose_origin[:2]

	local_xy  = np.array([ R_global_to_local @ pose[:2] + t_global_to_local 
	                         for pose in global_poses])

	local_yaw = np.array([ angle_mod_2pi(pose[2] - global_pose_origin[2])
		                     for pose in global_poses])
	
	return np.column_stack((local_xy, local_yaw))

def convert_local_to_global(global_pose_origin, local_poses):
	R_local_to_global = rotation_global_to_local(global_pose_origin[2]).T
	t_local_to_global = global_pose_origin[:2]

	global_xy  = np.array([ R_local_to_global @ pose[:2] + t_local_to_global 
		                      for pose in local_poses])

	global_yaw = np.array([ angle_mod_2pi( pose[2] + global_pose_origin[2]) 
		                     for pose in local_poses])
	
	return np.column_stack((global_xy, global_yaw))

##########################################
# Main function needed to extract a dictionary representation of a single element from a dataset.
def get_data_dict(input_representation, # an InputRepresentation object to do image rendering
	              helper,               # nuScenes prediction helper object
                  element,              # a dataset element, i.e. a {instance}_{sample} string
	              past_secs=1.0,        # seconds before current time to include in history
	              future_secs=6.0):     # seconds ahead of current time to predict

	instance, sample = element.split('_')
	annotation = helper.get_sample_annotation(instance, sample)
	
	# Get the state of the agent.
	agent_type   = annotation['category_name']	
	current_pose = np.array( extract_pose_from_annotation(annotation) )
	vel          = helper.get_velocity_for_agent(instance, sample)
	accel        = helper.get_acceleration_for_agent(instance, sample)
	yaw_rate     = helper.get_heading_change_rate_for_agent(instance, sample)

	# Get pose history/future.  False/False used to get raw annotations rather than local xy coords.
	past_poses   = \
	  extract_poses( helper.get_past_for_agent(instance, sample, past_secs, False, False) ) 			           
	future_poses = \
	  extract_poses( helper.get_future_for_agent(instance, sample, future_secs, False, False) )

	# Return None if we have an invalid data entry.
	if np.isnan(vel) or np.isnan(accel) or np.isnan(yaw_rate):
		return None # invalid motion state at current time for this agent
	if past_poses.shape[0] != 2 or future_poses.shape[0] != 12:			
		return None # insufficient past or future information for this agent
	
	# Convert poses to the agent frame (local) and ensure consistency when converted back to global.
	past_local_poses  = convert_global_to_local(current_pose, past_poses)
	past_global_poses = convert_local_to_global(current_pose, past_local_poses)
	pnorm = pose_diff_norm( past_global_poses - past_poses )
	
	future_local_poses  = convert_global_to_local(current_pose, future_poses)
	future_global_poses = convert_local_to_global(current_pose, future_local_poses)
	fnorm = pose_diff_norm( future_global_poses - future_poses )
	
	assert (pnorm < 1e-6 and fnorm < 1e-6) # make sure local <-> global is consistent.

	# Get the rasterized semantic image representation of this dataset instance.
	img = input_representation.make_input_representation(instance, sample)

	return {'instance': instance,
	        'sample': sample,
	        'type': agent_type,
	        'pose': current_pose,
	        'velocity': vel,
	        'acceleration': accel,
	        'yaw_rate': yaw_rate,
	        'past_poses_local': past_local_poses,
	        'future_poses_local': future_local_poses,
	        'image': img}

if __name__ == '__main__':	
	mode = 'write'

	datadir = os.path.abspath(__file__).split('scripts')[0] + 'data'	
	if not os.path.exists(datadir):
		raise Exception("Data directory does not exist: {}".format(datadir))
	print('Saving to: {}'.format(datadir))

	if mode == 'write':
		DATAROOT = '/media/data/nuscenes-data/' # location of NuScenes dataset
		nusc = NuScenes('v1.0-trainval', dataroot=DATAROOT) # 850 scenes, 700 train and 150 val
		helper = PredictHelper(nusc)

		PAST_SECS   = 1.0
		FUTURE_SECS = 6.0 

		# Using the CoverNet/MTP input representation taken from tutorials/prediction_tutorial.ipynb.
		static_layer_rasterizer = StaticLayerRasterizer(helper)
		agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=PAST_SECS)
		mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, \
			                                           Rasterizer())

		data_dict_function = partial(get_data_dict, mtp_input_representation, helper, 
			                         past_secs=PAST_SECS, future_secs=FUTURE_SECS) 

		for split in ['train_val', 'val', 'train']:
			dataset    = get_prediction_challenge_split(split, dataroot=DATAROOT)     
			file_prefix = '{}/nuscenes_{}'.format(datadir, split)
			write_tfrecord(file_prefix,         
			               dataset,             
			               data_dict_function,  
			               shuffle = True,      
			               shuffle_seed = 0,    
			               max_per_record = 1000)
	elif mode == 'read':
		import tensorflow as tf
		import glob
		import matplotlib.pyplot as plt
		plt.ion()

		# TODO: shuffling -> shuffle the order of the tfrecords ("chunk"-level randomness)
		#                    but also within ("local" randomness in the "chunk").
		dataset = tf.data.TFRecordDataset(glob.glob(datadir + '/*.record'))
		dataset = dataset.map(_parse_function)
		dataset = dataset.batch(2)

		f1 = plt.figure()
		f2 = plt.figure()

		for batch_ind, entry in enumerate(dataset):
			# This returns a dictionary which maps to a batch_size x data_shape tensor.
			
			if batch_ind == 0:
				for key in entry.keys():
					if 'image' in key:
						pass
					else:
						print(key, entry[key])
						print()

			plt.figure(f1.number)
			plt.plot(entry['future_poses_local'][0][:,0], entry['future_poses_local'][0][:,1])
			plt.plot(entry['future_poses_local'][1][:,0], entry['future_poses_local'][1][:,1])

			plt.figure(f2.number); 
			plt.clf()
			plt.subplot(211); plt.imshow(entry['image'][0])
			plt.subplot(212); plt.imshow(entry['image'][1])
			plt.draw(); plt.pause(0.01)
		
		plt.ioff()
		plt.show()

	else:
		raise ValueError("Invalid mode: {}".format(mode))
