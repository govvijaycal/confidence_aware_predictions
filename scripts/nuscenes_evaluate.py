import os
import sys
import json
import numpy as np
import pickle
import pandas as pd
import argparse
from tqdm import tqdm
import cv2

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper

from nuscenes.prediction.input_representation.interface import InputRepresentation 
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.combinators import Rasterizer

from models.regression import Regression
from models.multipath import MultiPath
from models.ekf import EKFKinematicFull, EKFKinematicCAH, EKFKinematicCVH
from datasets.splits import NUSCENES_TRAIN, NUSCENES_VAL
from evaluation.gmm_prediction import GMMPrediction

if __name__ == '__main__':
	repo_path = os.path.abspath(__file__).split('scripts')[0]

	mode = "visualize" #"predict", "evaluate", "visualize"
	
	# Full list of experiments for reference.  Can pick a subset to run on.
	name_model_weight_list = []
	name_model_weight_list.append(['nuscenes_regression_lstm', Regression, '00040_epochs.h5'])
	name_model_weight_list.append(['nuscenes_multipath_lstm', MultiPath, '00080_epochs.h5'])
	name_model_weight_list.append(['nuscenes_ekf', EKFKinematicFull, 'params.pkl'])
	name_model_weight_list.append(['nuscenes_ekf_cah', EKFKinematicCAH, 'params.pkl'])
	name_model_weight_list.append(['nuscenes_ekf_cvh', EKFKinematicCVH, 'params.pkl'])

	nuscenes_anchors = np.load(repo_path + 'data/nuscenes_clusters_16.npy')
	nuscenes_weights = np.load(repo_path + 'data/nuscenes_clusters_16_weights.npy')
	
	if mode == 'predict':
		for name_model_weight in name_model_weight_list:
			name, model, weight = name_model_weight
			# Construct the model.
			if issubclass(model, EKFKinematicFull):
				m = model()
				epoch_label = name.split('_')[-1]
			elif model == Regression:
				m = model(num_timesteps=12, num_hist_timesteps=2)
				epoch_label = weight.split('_')[0]
			elif model == MultiPath:
				m = model(num_timesteps=12, num_hist_timesteps=2,
				          anchors=nuscenes_anchors, weights=nuscenes_weights)
				epoch_label = weight.split('_')[0]
			else:
				raise ValueError(f"Invalid model: {model}")
			
			logdir = f"{repo_path}log/{name}/"

			m.load_weights(f"{logdir}{weight}")

			predict_dict = m.predict(NUSCENES_VAL)
			
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
					data_list_entry.extend(gmm_pred.get_class_top_k_scores(future_xy_gt, nuscenes_anchors, ks_eval))
					data_list_entry.extend([gmm_pred_k.compute_min_ADE(future_xy_gt) \
					                        for gmm_pred_k in gmm_pred_ks])
					data_list_entry.extend([gmm_pred_k.compute_min_FDE(future_xy_gt) \
					                        for gmm_pred_k in gmm_pred_ks])
					data_list_entry.extend([gmm_pred_k.compute_minmax_d(future_xy_gt) \
					                        for gmm_pred_k in gmm_pred_ks])

				data_list.append(data_list_entry)	
		metrics_df = pd.DataFrame(data_list, columns=columns)	
		metrics_df.to_pickle(f"{repo_path}nuscenes_metrics_df.pkl")
	elif mode == 'aggregate_metrics':
		metrics_df = pd.read_pickle(f"{repo_path}nuscenes_metrics_df.pkl")
		
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
				# print(f"\t\tMin: {model_df[metric].min()}")
				# print(f"\t\tMax: {model_df[metric].max()}")
	elif mode == 'visualize':
		# NuScenes setup to use rasterizer without need for reading tfrecords.
		# Same as the code in prep_nuscenes.
		nusc = NuScenes('v1.0-trainval', dataroot="/media/data/nuscenes-data/")
		helper = PredictHelper(nusc)
		static_layer_rasterizer = StaticLayerRasterizer(helper)
		agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=1.0)
		mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, \
			                                           Rasterizer())

		def local_coords_to_pixels(X_local, Y_local):
			pix_y = 400 - int(X_local / 0.1) # in the vertical axis (height)
			pix_x = 250 - int(Y_local / 0.1) # in the horizontal axis (width)

			return pix_x, pix_y

		def plot_trajectory(img, traj, radius=5, color=(255, 0, 0)):
			# expect traj to be a 12 by 2 numpy array in the local frame
			max_y, max_x, _ = img.shape

			for xy in traj:
				pix_x, pix_y = local_coords_to_pixels(*xy)
				if pix_x >= 0 and pix_x < max_x and pix_y >=0 and pix_y < max_y:
					cv2.circle(img, (pix_x, pix_y), radius, color, -1)

		def plot_covariance_ellipse(img, mean, sigma, mdist_sq_thresh=5.991, color=(0, 128, 0), thickness=2):
			# Reference: https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
			evals, evecs = np.linalg.eig(sigma)
			# Note: mdist_sq_thresh = scaling parameter equal to the Mahalanobis distance squared.
			length_ax1 = int( np.sqrt(mdist_sq_thresh * evals[0]) / 0.1 ) # half the first axis diameter in pixels
			length_ax2 = int( np.sqrt(mdist_sq_thresh * evals[1]) / 0.1 ) # half the second axis diameter in pixels
			ang_1 = -90.-np.degrees( np.arctan2(evecs[1,0], evecs[0,0]) ) # -90 due to pixel axes orientation, -ang since cv2.ellipse uses clockwise angle

			center_px, center_py = local_coords_to_pixels(*mean)

			cv2.ellipse( img, (center_px, center_py), (length_ax1, length_ax2), ang_1, 0, 360, color, thickness=thickness  )

		def plot_GMM(img, gmm_pred, num_modes_to_plot=3):
			max_y, max_x, _ = img.shape

			num_modes = len(gmm_pred.keys())
			probs = [ gmm_pred[k]['mode_probability'] for k in range(num_modes) ]
			
			if num_modes > 1:
				modes_to_plot = np.argsort(probs)[-num_modes_to_plot:]
				color_list = [(128, 128, 0), (0, 128, 128), (128, 0, 128)]
			else:
				modes_to_plot = [0]
				color_list = [(128, 128, 0)]

			for ind_mode, mode in enumerate(modes_to_plot):
				mean_traj  = gmm_pred[mode]['mus']
				covar_traj = gmm_pred[mode]['sigmas']
				color = color_list[ind_mode]

				for tmstep in range(mean_traj.shape[0]):
					plot_covariance_ellipse(img, mean_traj[tmstep], covar_traj[tmstep], color=color)

		# Load the predictions made by each model.  Make into a nested dict for easy lookup.
		predictions_dict = {}
		for name_model_weight in name_model_weight_list:
			name, _, weight = name_model_weight
			
			if 'ekf' in name:
				epoch_label = name.split('_')[-1]
			else:
				epoch_label = weight.split('_')[0]

			pkl_name = f"{repo_path}log/{name}/predictions_{epoch_label}.pkl"
			model_name = f"{name}_{epoch_label}"
			predictions_dict[model_name] = pickle.load( open(pkl_name, "rb"))

		# Identify which cases to visualize based on variation in min_ade_5 across all models.
		metrics_df = pd.read_pickle(f"{repo_path}nuscenes_metrics_df.pkl")
		metrics_df = metrics_df[metrics_df.model != 'nuscenes_ekf_cah_cah']
		metrics_df = metrics_df[metrics_df.model != 'nuscenes_ekf_ekf']
		
		instances = set(metrics_df.instance) 
		instance_data_list = []

		for instance in instances:
			instance_df = metrics_df[metrics_df.instance == instance] # TODO: fix the sample_instance parse issue above.
			avg_ade_5   = instance_df['min_ade_5'].mean()			
			best_ade_5  = instance_df['min_ade_5'].min()
			std_ade_5   = instance_df['min_ade_5'].std()
			range_ade_5 = instance_df['min_ade_5'].max() - instance_df['min_ade_5'].min()
			instance_data_list.append([instance, avg_ade_5, best_ade_5, std_ade_5, range_ade_5])
		instance_df = pd.DataFrame(instance_data_list, \
			columns=["instance", "avg_ade_5", "best_ade_5", "std_ade_5", "range_ade_5"])
		
		# code: l = low, h = high; b = "bias", v = "variance"
		# using "" above since my choice of metrics are crude proxies for actual bias/variance
		lb_lv = np.logical_and(instance_df.best_ade_5 < 5.  , instance_df.range_ade_5 < 5.)		
		hb_lv = np.logical_and(instance_df.best_ade_5 >= 5. , instance_df.range_ade_5 < 5.)
		lb_hv = np.logical_and(instance_df.best_ade_5 < 5.  , instance_df.range_ade_5 >= 5.)
		hb_hv = np.logical_and(instance_df.best_ade_5 >= 5. , instance_df.range_ade_5 >= 5.)

		cases_to_visualize = []

		np.random.seed(0)
		cases_to_visualize.extend( np.random.choice(instance_df[lb_lv].instance, 10))
		cases_to_visualize.extend( np.random.choice(instance_df[hb_lv].instance, 10))
		cases_to_visualize.extend( np.random.choice(instance_df[lb_hv].instance, 10))
		cases_to_visualize.extend( np.random.choice(instance_df[hb_hv].instance, 10))

		model_names = list( predictions_dict.keys() )
		model_names.remove('nuscenes_ekf_ekf')
		model_names.remove('nuscenes_ekf_cah_cah')
		
		for case in cases_to_visualize:
			mosaic_array = np.zeros((500, 500 * len(model_names), 3), dtype=np.uint8)

			sample, instance = case.split('_')
			img_base = mtp_input_representation.make_input_representation(instance, sample)
			
			for ind_model, model in enumerate(model_names):
				img_annotated = np.copy(img_base)

				pred = predictions_dict[model][case]
				
				plot_trajectory(img_annotated, pred['future_traj'][:, 1:3])
				plot_GMM(img_annotated, pred['gmm_pred'])

				text_to_show = model.split('nuscenes_')[-1]
				cv2.putText(img_annotated, text_to_show, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
				
				mosaic_array[:, ind_model*500:(ind_model+1)*500, :] = img_annotated

			mosaic_array = cv2.cvtColor(mosaic_array, cv2.COLOR_RGB2BGR)
			savedir = f"{repo_path}log/{name}/"
			cv2.imwrite(savedir + '_{}_{}.png'.format(instance, sample) , mosaic_array)

		with open(savedir + 'examples.txt', 'w') as f:
			for case in cases_to_visualize:
				sample, instance = case.split('_')
				f.write("{}_{}\n".format(instance, sample))
		



		



		# multipath_df = metrics_df[metrics_df.model == 'nuscenes_multipath_lstm_00080']

		# pkl_name = f"{repo_path}log/{name}/predictions_{epoch_label}.pkl"
		# model_name = f"{name}_{epoch_label}"
		# predict_dict = pickle.load( open(pkl_name, "rb"))

		# # SUCCESSES
		# success_df = multipath_df[ multipath_df.min_ade_5 < 2. ]
		
		# # OKAY BUT NOT GREAT
		# okay_df = multipath_df[ multipath_df.min_ade_5 < 5. ]

		# # FAILURES
		# failure_df = multipath_df[ multipath_df.min_ade_5 >= 5. ]



		# TODO: set up the rasterizer and predict helper.
		# TODO: figure out the correct transform to overlay actual + GMM trajectories.
		# TODO: figure out how to plot ellipses.
		import pdb; pdb.set_trace()
	else:
		raise ValueError("Invalid mode!")
