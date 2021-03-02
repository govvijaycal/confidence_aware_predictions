import os
import sys
import json
import numpy as np

from models.regression import Regression
from models.multipath import MultiPath
from models.ekf import EKFKinematicFull, EKFKinematicCAH, EKFKinematicCVH
from datasets.splits import NUSCENES_TRAIN, NUSCENES_VAL

if __name__ == '__main__':
	repo_path = os.path.abspath(__file__).split('scripts')[0]

	# Full list of experiments for reference.  Can pick a subset to run on.
	name_model_list = []
	name_model_list.append(['nuscenes_regression_lstm', Regression])
	name_model_list.append(['nuscenes_multipath_lstm', MultiPath])
	name_model_list.append(['nuscenes_ekf', EKFKinematicFull])
	name_model_list.append(['nuscenes_ekf_cah', EKFKinematicCAH])
	name_model_list.append(['nuscenes_ekf_cvh', EKFKinematicCVH])

	nuscenes_anchors = np.load(repo_path + 'data/nuscenes_clusters_16.npy')
	nuscenes_weights = np.load(repo_path + 'data/nuscenes_clusters_16_weights.npy')

	for name_model in name_model_list:
		name, model = name_model
		# Construct the model.
		if issubclass(model, EKFKinematicFull):
			m = model()
		elif model == Regression:
			m = model(num_timesteps=12, num_hist_timesteps=2)
		elif model == MultiPath:
			m = model(num_timesteps=12, num_hist_timesteps=2, \
				      anchors=nuscenes_anchors, weights=nuscenes_weights)
		else:
			raise ValueError(f"Invalid model: {model}")
		
		logdir = f"{repo_path}log/{name}/"

		# Train the model.
		m.fit(NUSCENES_TRAIN, 
			  NUSCENES_VAL,
			  logdir=logdir,
			  log_epoch_freq=2,
			  save_epoch_freq=10,
			  num_epochs=100,
			  batch_size=32)
