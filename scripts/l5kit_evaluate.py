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
from models.ekf import EKFKinematicBase, EKFKinematicCATR, EKFKinematicCAH, \
                       EKFKinematicCVTR, EKFKinematicCVH
from models.static_multiple_model import StaticMultipleModel
from datasets.splits import L5KIT_TRAIN, L5KIT_VAL
from evaluation.gmm_prediction import GMMPrediction

if __name__ == '__main__':
	repo_path = os.path.abspath(__file__).split('scripts')[0]

	# TODO: clean up interface, argparse, etc.
	mode = "visualize" #"predict", "evaluate", "visualize"

	# Full list of experiments for reference.  Can pick a subset to run on.
	name_model_weight_list = []
	# TODO: update name_model_weight_list.
	raise NotImplementedError("Need to fill this in with updated values.")

	l5kit_anchors = np.load(repo_path + 'data/l5kit_clusters_16.npy')
	l5kit_weights = np.load(repo_path + 'data/l5kit_clusters_16_weights.npy')

	if mode == 'predict':
		for name_model_weight in name_model_weight_list:
			name, model, weight = name_model_weight
			# Construct the model.
			if issubclass(model, EKFKinematicBase) or model == StaticMultipleModel:
				m = model()
				epoch_label = name.split('_')[-1]
			elif model == Regression:
				m = model(num_timesteps=25, num_hist_timesteps=5)
				epoch_label = weight.split('_')[0]
			elif model == MultiPath:
				m = model(num_timesteps=25, num_hist_timesteps=5,
				          anchors=l5kit_anchors, weights=l5kit_weights)
				epoch_label = weight.split('_')[0]
			else:
				raise ValueError(f"Invalid model: {model}")

			logdir = f"{repo_path}log/{name}/"

			m.load_weights(f"{logdir}{weight}")

			predict_dict = m.predict(L5KIT_VAL)

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

		for name_model_weight in name_model_weight_list:
			name, _, weight = name_model_weight

			if 'ekf' in name:
				epoch_label = name.split('_')[-1]
			else:
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
		metrics_df.to_pickle(f"{repo_path}l5kit_metrics_df.pkl")

	elif mode == 'visualize':
		metrics_df = pd.read_pickle(f"{repo_path}l5kit_metrics_df.pkl")

		model_names = set(metrics_df.model)

		ks_eval = [1, 3, 5]
		metrics = []
		metrics.extend([f"traj_LL_{k}" for k in ks_eval])
		metrics.extend([f"class_top_{k}" for k in ks_eval])
		metrics.extend([f"min_ade_{k}" for k in ks_eval])
		metrics.extend([f"min_fde_{k}" for k in ks_eval])
		metrics.extend([f"minmax_dist_{k}" for k in ks_eval])

		for model_name in model_names:
			model_df = metrics_df[metrics_df.model==model_name]
			print(model_name)

			for metric in metrics:
				print(f"\t{metric}")
				print(f"\t\tMean: {model_df[metric].mean()}")
				print(f"\t\tStd: {model_df[metric].std()}")
				print(f"\t\tMin: {model_df[metric].min()}")
				print(f"\t\tMax: {model_df[metric].max()}")

		# make aggregate plots
		# save results
		# use metrics to identify some interesting examples
		# use opencv to overlay GMM distribution and GT
		# save to pngs in logdir
	else:
		raise ValueError("Invalid mode!")
