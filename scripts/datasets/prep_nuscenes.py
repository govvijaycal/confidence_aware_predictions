import os
import glob
import numpy as np
from pyquaternion import Quaternion
from functools import partial
import argparse

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.eval.common.utils import quaternion_yaw

from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.input_representation.interface import InputRepresentation 
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.combinators import Rasterizer

from tfrecord_utils import write_tfrecord, visualize_tfrecords, shuffle_test
from pose_utils import convert_global_to_local, convert_local_to_global, pose_diff_norm

##########################################
# Helper functions to convert annotations into a list of poses.
def extract_pose_from_annotation(annotation):
	x, y, _ = annotation['translation']
	yaw     = quaternion_yaw( Quaternion(annotation['rotation']) )
	return x, y, yaw

def extract_poses(annotation_list):
	return np.array([extract_pose_from_annotation(ann) for ann in annotation_list])

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
	past   = helper.get_past_for_agent(instance, sample, past_secs, False, False)
	future = helper.get_future_for_agent(instance, sample, future_secs, False, False)
	past_poses   = extract_poses(past) 			           
	future_poses = extract_poses(future)

	# Get the relative time difference (seconds) from the current sample time.
	# This is required since nuscenes does not have a constant sampling time (i.e. can be 0.4 - 0.6 s).
	current_tm = helper._timestamp_for_sample(sample)

	past_tms   = [helper._timestamp_for_sample(p['sample_token']) for p in past]

	future_tms = [helper._timestamp_for_sample(f['sample_token']) for f in future]

	# Microseconds -> seconds.
	past_tms   = np.array([10**(-6) * (x - current_tm)  for x in past_tms])
	future_tms = np.array([10**(-6) * (x - current_tm)  for x in future_tms])

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
	        'past_tms': past_tms,
	        'future_tms': future_tms,
	        'image': img}

if __name__ == '__main__':	
	parser = argparse.ArgumentParser('Read/write NuScenes prediction instances in TFRecord format.')
	parser.add_argument('--mode', choices=['read', 'write', 'batch_test'], type=str, required=True, \
		                    help='Read/write tfrecords or check batch shuffling.')
	parser.add_argument('--datadir', type=str, help='Where the TFRecords are located or should be saved.', \
		                    default=os.path.abspath(__file__).split('scripts')[0] + 'data')
	parser.add_argument('--dataroot', type=str, help='Location of the NuScenes dataset.', \
		                    default='/media/data/nuscenes-data/')
	args = parser.parse_args()
	datadir = args.datadir
	mode = args.mode
	dataroot = args.dataroot

	if not os.path.exists(datadir):
		raise Exception("Data directory does not exist: {}".format(datadir))
	print('Saving to: {}'.format(datadir))

	if mode == 'write':		
		nusc = NuScenes('v1.0-trainval', dataroot=dataroot) # 850 scenes, 700 train and 150 val
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
			dataset    = get_prediction_challenge_split(split, dataroot=dataroot)     
			file_prefix = '{}/nuscenes_{}'.format(datadir, split)
			write_tfrecord(file_prefix,         
			               dataset,             
			               data_dict_function,  
			               shuffle = True,      
			               shuffle_seed = 0,    
			               max_per_record = 1000)
	elif mode == 'read':
		train_set = glob.glob(datadir + '/nuscenes_train*.record')
		train_set = [x for x in train_set if 'val' not in x]
		visualize_tfrecords(train_set, max_batches=100)

	elif mode == 'batch_test':
		train_set = glob.glob(datadir + '/nuscenes_train*.record')
		train_set = [x for x in train_set if 'val' not in x]
		shuffle_test(train_set, batch_size=32)

		# Results: shows good shuffling (dataset size approx. 32000)
		# Shuffle min range and std dev: 9375, 3678.020500969993
		# Shuffle max range and std dev: 29670, 11924.862244720294
		# Shuffle mean range and std dev: 21017.645061728395, 7663.28797797656


	else:
		raise ValueError("Invalid mode: {}".format(mode))
