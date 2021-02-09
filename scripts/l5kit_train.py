import os
import sys
import json
import numpy as np

from models.regression import Regression
from models.multipath import MultiPath
from datasets.splits import L5KIT_TRAIN, L5KIT_VAL

if __name__ == '__main__':
	repo_path = os.path.abspath(__file__).split('scripts')[0]

	# Full list of experiments for reference.  Can pick a subset to run on.
	#names  = ['l5kit_regression', 'l5kit_multipath']
	#models = [Regression, MultiPath]
	names  = ['l5kit_multipath']
	models = [MultiPath]

	l5kit_anchors = np.load(repo_path + 'data/l5kit_clusters_16.npy')
	l5kit_weights = np.load(repo_path + 'data/l5kit_clusters_16_weights.npy')

	for name, model in zip(names, models):
		# Construct the model.
		if type(model) == Regression:
			m = model(num_timesteps=25)
		else:
			m = model(num_timesteps=25, anchors=l5kit_anchors, weights=l5kit_weights)
		
		logdir = f"{repo_path}log/{name}/"

		# Train the model.
		m.fit(L5KIT_TRAIN, 
			  L5KIT_VAL,
			  logdir=logdir,
			  log_epoch_freq=2,
			  save_epoch_freq=10,
			  num_epochs=100,
			  batch_size=32)

		# TODO: Predictions.  Test Cases.  MultiPath full updated.
		# Variant with KF and IMM.
		# Multipath Dynamic.