''' This file is essentially the same as the Physics oracle baseline in nuscenes located here:
    nuscenes-devkit/python-sdk/nuscenes/prediction/physics.py

    One minor change is that the integration is done with a finer timestep to reduce numerical errors.
    The point of this is for sanity checking that ADE/FDE/etc. metrics match up as expected.
'''
import sys
import os
import glob
import json
import numpy as np
import tensorflow as tf
from nuscenes.eval.prediction.data_classes import Prediction

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)
from datasets.tfrecord_utils import _parse_function
from datasets.pose_utils import convert_local_to_global

def constant_velocity_heading(state_dict, dt=0.5, nsteps=12):
	preds = []

	xcurr   = float(state_dict['x'])
	ycurr   = float(state_dict['y'])
	yaw     = float(state_dict['yaw'])
	v       = float(state_dict['v'])

	for t in range(nsteps):
		xcurr += v * np.cos(yaw) * dt
		ycurr += v * np.sin(yaw) * dt
		preds.append([xcurr, ycurr])
	return np.array(preds).reshape(nsteps, 2)

def constant_velocity_yaw_rate(state_dict, dt=0.5, nsteps=12):
	preds = []

	xcurr   = float(state_dict['x'])
	ycurr   = float(state_dict['y'])
	yawcurr = float(state_dict['yaw'])
	v       = float(state_dict['v'])
	yawrate = float(state_dict['yawrate'])

	for t in range(nsteps):
		for i in range(10):
			# Use a smaller timestep to avoid errors due to
			# forward Euler integration.
			xcurr   += v * np.cos(yawcurr) * (dt/10.)
			ycurr   += v * np.sin(yawcurr) * (dt/10.)
			yawcurr += yawrate * (dt/10.)
		preds.append([xcurr, ycurr])
	return np.array(preds).reshape(nsteps, 2)

def constant_acceleration_heading(state_dict, dt=0.5, nsteps=12):
	preds = []

	xcurr   = float(state_dict['x'])
	ycurr   = float(state_dict['y'])
	yaw     = float(state_dict['yaw'])
	vcurr   = float(state_dict['v'])
	acc     = float(state_dict['acc'])

	for t in range(nsteps):
		for i in range(10):
			# Use a smaller timestep to avoid errors due to
			# forward Euler integration.
			xcurr += vcurr * np.cos(yaw) * (dt/10.)
			ycurr += vcurr * np.sin(yaw) * (dt/10.)
			vcurr += acc * (dt/10.)
			vcurr = max(0., vcurr) # assume no reverse driving
		preds.append([xcurr, ycurr])
	return np.array(preds).reshape(nsteps, 2)

def constant_acceleration_yaw_rate(state_dict, dt=0.5, nsteps=12):
	preds = []

	xcurr   = float(state_dict['x'])
	ycurr   = float(state_dict['y'])
	yawcurr = float(state_dict['yaw'])
	vcurr   = float(state_dict['v'])
	yawrate = float(state_dict['yawrate'])
	acc     = float(state_dict['acc'])

	for t in range(nsteps):
		for i in range(10):
			# Use a smaller timestep to avoid errors due to
			# forward Euler integration.
			xcurr   += vcurr * np.cos(yawcurr) * (dt/10.)
			ycurr   += vcurr * np.sin(yawcurr) * (dt/10.)
			yawcurr += yawrate * (dt/10.)
			vcurr   += acc * (dt/10.)
			vcurr = max(0., vcurr) # assume no reverse driving
		preds.append([xcurr, ycurr])
	return np.array(preds).reshape(nsteps, 2)

def make_predictions(dataset, savefile):
	dataset = tf.data.TFRecordDataset(dataset)
	dataset = dataset.map(_parse_function)

	pred_funcs = [constant_velocity_heading, \
	              constant_acceleration_heading, \
	              constant_velocity_yaw_rate, \
	              constant_acceleration_yaw_rate]

	oracle_preds = []

	for entry in dataset:
		instance = tf.compat.as_str(entry['instance'].numpy())
		sample   = tf.compat.as_str(entry['sample'].numpy())
		pose     = entry['pose'].numpy()
		vel      = entry['velocity'].numpy()
		yawrate  = entry['yaw_rate'].numpy()
		acc      = entry['acceleration'].numpy()

		state_dict = {}
		state_dict['x']       = pose[0]
		state_dict['y']       = pose[1]
		state_dict['yaw']     = pose[2]
		state_dict['v']       = vel
		state_dict['yawrate'] = yawrate
		state_dict['acc']     = acc

		pred_trajs = [pf(state_dict) for pf in pred_funcs]

		actual_traj = convert_local_to_global(entry['pose'].numpy(), entry['future_poses_local'].numpy())[:,:2]

		best_traj_index = np.argmin([ np.sum(np.linalg.norm(actual_traj - pred_traj, axis=-1), axis=-1)
		                              for pred_traj in pred_trajs ])

		best_traj = pred_trajs[best_traj_index]

		oracle_preds.append( Prediction(instance, sample, np.expand_dims(best_traj, 0), np.array([1.])).serialize() )

	json.dump(oracle_preds, open(savefile, 'w'))


if __name__ == '__main__':
	repo_path = os.path.abspath(__file__).split('scripts')[0]
	datadir = repo_path + 'data'
	logdir = repo_path + 'log/physics/'

	os.makedirs(logdir, exist_ok=True)

	train_set = glob.glob(datadir + '/nuscenes_train*.record')
	train_set = [x for x in train_set if 'val' not in x]

	val_set   = glob.glob(datadir + '/nuscenes_train_val*.record')

	print('Predicting on the training set.')
	make_predictions(train_set, logdir + 'nuscenes_train.json')
	print('Predicting on the validation (train_val) set.')
	make_predictions(val_set, logdir + 'nuscenes_train_val.json')

	# To evaluate the produced json file:
	# python ../nuscenes-devkit/python-sdk/nuscenes/eval/prediction/compute_metrics.py \
	#        --version 'v1.0-trainval' --data_root /media/data/nuscenes-data/ --submission_path log/physics/nuscenes_train.json

