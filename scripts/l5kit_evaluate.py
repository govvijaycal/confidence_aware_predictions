import os
import sys
import json
import numpy as np
import pickle
import pandas as pd
import argparse
from tqdm import tqdm

from models.regression import Regression
from models.multipath import MultiPath
from datasets.splits import L5KIT_TRAIN, L5KIT_VAL
from evaluation.gmm_prediction import GMMPrediction

"""
TODO: modes -> predict, evaluate, visualize.
Prediction Code.  Pandas dataframe analysis + metrics reporting.
ID examples to plot and plot it (aka MultiPath, Fig 3)
Wrap up EKF impl. and plan out IMM impl process.
Plan out MultiPath Dynamic impl.
Closed-loop eval: Carla + val set resimulation.
"""


if __name__ == '__main__':
	repo_path = os.path.abspath(__file__).split('scripts')[0]

	mode = "evaluate" #"predict", "evaluate", "visualize"
	
	# Full list of experiments for reference.  Can pick a subset to run on.
	names   = ['l5kit_regression_lstm', 'l5kit_multipath_lstm']
	models  = [Regression, MultiPath]
	weights = ['00040_epochs.h5', '00080_epochs.h5']
	l5kit_anchors = np.load(repo_path + 'data/l5kit_clusters_16.npy')
	l5kit_weights = np.load(repo_path + 'data/l5kit_clusters_16_weights.npy')
	
	if mode == 'predict':
		for name, model, weight in zip(names, models, weights):
			# Construct the model.
			if model == Regression:
				m = model(num_timesteps=25, num_hist_timesteps=5)
			else:
				m = model(num_timesteps=25, num_hist_timesteps=5,
				          anchors=l5kit_anchors, weights=l5kit_weights)
			
			logdir = f"{repo_path}log/{name}/"

			m.load_weights(f"{logdir}{weight}")

			predict_dict = m.predict(L5KIT_VAL)
			epoch_label = weight.split('_')[0]
			pkl_name = f"{repo_path}log/{name}/predictions_{epoch_label}.pkl"
			pickle.dump( predict_dict, open(pkl_name, "wb") )

	elif mode == 'evaluate':
		data_list = []
		ks_eval = [1, 3, 5]		

		columns   = ["sample", "instance", "model"]
		columns.extend([f"traj_LL_{k}" for k in ks_eval])
		columns.extend([f"class_top_{k}" for k in ks_eval])
		columns.extend([f"min_ade_{k}" for k in ks_eval])
		columns.extend([f"min_fde_{k}" for k in ks_eval])
		columns.extend([f"minmax_dist_{k}" for k in ks_eval])

		for name, weight in zip(names, weights):			
			epoch_label = weight.split('_')[0]
			pkl_name = f"{repo_path}log/{name}/predictions_{epoch_label}.pkl"
			model_name = f"{name}_{epoch_label}"
			predict_dict = pickle.load( open(pkl_name, "rb"))

			print(f"Evaluating model: {model_name}")

			for key in tqdm(predict_dict.keys()):
				future_traj_gt = predict_dict[key]['future_traj']
				future_xy_gt = future_traj_gt[:, 1:3]
				gmm_pred       = predict_dict[key]['gmm_pred']				

				n_modes     = len(gmm_pred.keys())
				n_timesteps = future_traj_gt.shape[0]
				mode_probabilities = np.array( [gmm_pred[mode]['mode_probability'] for mode in range(n_modes)] )
				mus                = np.array( [gmm_pred[mode]['mus'] for mode in range(n_modes)] )
				sigmas             = np.array( [gmm_pred[mode]['sigmas'] for mode in range(n_modes)] )

				gmm_pred     = GMMPrediction(n_modes, n_timesteps, mode_probabilities, mus, sigmas)
				
				sample_token   = '_'.join( key.split('_')[:-2] )
				instance_token = '_'.join( key.split('_')[-2:] )
				
				data_list_entry = [sample_token, instance_token, model_name]
				if n_modes == 1:
					num_ks = len(ks_eval)
					data_list_entry.extend([gmm_pred.compute_trajectory_log_likelihood(future_xy_gt)]*num_ks)
					data_list_entry.extend([1]*num_ks) # unimodal
					data_list_entry.extend([gmm_pred.compute_min_ADE(future_xy_gt)]*num_ks)
					data_list_entry.extend([gmm_pred.compute_min_FDE(future_xy_gt)]*num_ks)
					data_list_entry.extend([gmm_pred.compute_minmax_d(future_xy_gt)]*num_ks)
				else:
					gmm_pred_ks  = [gmm_pred.get_top_k_GMM(k) for k in ks_eval]

					data_list_entry.extend([gmm_pred_k.compute_trajectory_log_likelihood(future_xy_gt) \
						                    for gmm_pred_k in gmm_pred_ks])
					data_list_entry.extend(gmm_pred.get_class_top_k_scores(future_xy_gt, l5kit_anchors, ks_eval))
					data_list_entry.extend([gmm_pred_k.compute_min_ADE(future_xy_gt) \
					                        for gmm_pred_k in gmm_pred_ks])
					data_list_entry.extend([gmm_pred_k.compute_min_FDE(future_xy_gt) \
					                        for gmm_pred_k in gmm_pred_ks])
					data_list_entry.extend([gmm_pred_k.compute_minmax_d(future_xy_gt) \
					                        for gmm_pred_k in gmm_pred_ks])

				data_list.append(data_list_entry)	
		metrics_df = pd.DataFrame(data_list, columns=columns)	
		metrics_df.to_pickle(f"{repo_path}metrics_df.pkl")

	elif mode == 'visualize':
		pass
		# make aggregate plots
		# save results
		# use metrics to identify some interesting examples
		# use opencv to overlay GMM distribution and GT
		# save to pngs in logdir
	else:
		raise ValueError("Invalid mode!")
