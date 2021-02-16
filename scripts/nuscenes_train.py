import os
import sys
import json
import numpy as np

from models.regression import Regression
from models.multipath import MultiPath
from datasets.splits import NUSCENES_TRAIN, NUSCENES_VAL

if __name__ == '__main__':
	repo_path = os.path.abspath(__file__).split('scripts')[0]

	# Full list of experiments for reference.  Can pick a subset to run on.
	names  = ['nuscenes_regression_lstm', 'nuscenes_multipath_lstm']
	models = [Regression, MultiPath]

	nuscenes_anchors = np.load(repo_path + 'data/nuscenes_clusters_16.npy')
	nuscenes_weights = np.load(repo_path + 'data/nuscenes_clusters_16_weights.npy')

	for name, model in zip(names, models):
		# Construct the model.
		if model == Regression:
			m = model(num_timesteps=12, num_hist_timesteps=2)
		else:
			m = model(num_timesteps=12, num_hist_timesteps=2, \
				      anchors=nuscenes_anchors, weights=nuscenes_weights)
		
		logdir = f"{repo_path}log/{name}/"

		# Train the model.
		m.fit(NUSCENES_TRAIN, 
			  NUSCENES_VAL,
			  logdir=logdir,
			  log_epoch_freq=2,
			  save_epoch_freq=10,
			  num_epochs=100,
			  batch_size=32)
