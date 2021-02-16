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
	names   = ['l5kit_regression', 'l5kit_multipath']
	models  = [Regression, MultiPath]
	weights = ['00030_epochs.h5', '00080_epochs.h5']

	l5kit_anchors = np.load(repo_path + 'data/l5kit_clusters_16.npy')
	l5kit_weights = np.load(repo_path + 'data/l5kit_clusters_16_weights.npy')

	for name, model, weight in zip(names, models, weights):
		# Construct the model.
		if model == Regression:
			m = model(num_timesteps=25)
		else:
			m = model(num_timesteps=25, anchors=l5kit_anchors, weights=l5kit_weights)
		
		logdir = f"{repo_path}log/{name}/"

		m.load_weights(f"{logdir}{weight}")

		res_dict = m.predict(L5KIT_VAL)
		import pdb; pdb.set_trace()

		"""
		key is scene_instance
		res_dict[key] => pred_dict
		pred_dict has keys: state, traj, gmm_pred
		gmm_pred has keys for each mode (int), each of which map to a unimodal Gaussian prediction.

		This Gaussian prediction has keys: mode_probability, mus, sigmas
		"""

		# TODO: Prediction Code.  Pandas dataframe analysis + metrics reporting.
		#       ID examples to plot and plot it (aka MultiPath, Fig 3)
		#       Wrap up EKF impl. and plan out IMM impl process.
		#       Plan out MultiPath Dynamic impl.
		#       Closed-loop eval: Carla + val set resimulation.