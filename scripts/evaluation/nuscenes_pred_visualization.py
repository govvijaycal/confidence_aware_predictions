import numpy as np
import cv2

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.interface import InputRepresentation 
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.combinators import Rasterizer

class NuscenesPredictionVisualizer:
	def __init__(self, dataroot="/media/data/nuscenes-data/"):
		# We only need the image generation (mtp_input_representation) since
		# the prediction_dict is assumed to have all other relevant information.
		nusc = NuScenes('v1.0-trainval', dataroot=dataroot)
		helper = PredictHelper(nusc)
		static_layer_rasterizer = StaticLayerRasterizer(helper)
		agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=1.0)
		self.mtp_input_representation = InputRepresentation(static_layer_rasterizer,
		                                                    agent_rasterizer,
		                                                    Rasterizer())

	def visualize_prediction(self, prediction_dict, sample_instance_key):
		# Given a set of predictions (prediction_dict) and a given example (sample_instance_key),
		# overlay the real trajectory and multimodal predictions on the scene image.
		sample, instance = sample_instance_key.split('_')		
		img = self.mtp_input_representation.make_input_representation(instance, sample)

		pred = prediction_dict[sample_instance_key]

		img_gt  = np.copy(img)
		img_gmm = np.copy(img)

		self.plot_trajectory(img_gt, pred['future_traj'][:, 1:3])
		top_probs = self.plot_GMM(img_gmm, pred['gmm_pred'])

		return img, img_gt, img_gmm, top_probs

	@staticmethod
	def local_coords_to_pixels(X_local, Y_local):
		# Vehicle -> pixel transformation assuming default rasterization parameters:
		# i.e. 500 px x 500 px at 0.1m / px resolution, ego centered at pix_x = 250, pix_y = 400.
		pix_y = 400 - int(X_local / 0.1) # X_local points upward.
		pix_x = 250 - int(Y_local / 0.1) # Y_local points left.

		return pix_x, pix_y

	@staticmethod
	def plot_trajectory(img, traj, radius=5, color=(255, 0, 0)):
		# Plots a single trajectory, which is expected to be
		# a 12 x 2 matrix in the vehicle frame.				
		for xy in traj:
			pix_x, pix_y = NuscenesPredictionVisualizer.local_coords_to_pixels(*xy)
			cv2.circle(img, (pix_x, pix_y), radius, color, -1)

	@staticmethod
	def plot_covariance_ellipse(img, mean, sigma, mdist_sq_thresh=5.991, color=(0, 128, 0), thickness=2):
		# Given a mean / sigma in the vehicle frame, plot a covariance ellipse of desired Mahalanobis distance
		# in the image.  Note that mdist_sq_thresh is the Mahalanobis distance squared, and 5.991 corresponds to 95% 
		# confidence.  See reference below for details on the chi-squared distribution derivation and plotting details.
		# Reference: https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
		evals, evecs = np.linalg.eig(sigma) # relative to multipath covariance estimate, evecs are the variances and evecs are the columns of the rotation matrix.
		
		length_ax1 = int( np.sqrt(mdist_sq_thresh * evals[0]) / 0.1 ) # half the first axis diameter in pixels
		length_ax2 = int( np.sqrt(mdist_sq_thresh * evals[1]) / 0.1 ) # half the second axis diameter in pixels
		ang_1 = -90.-np.degrees( np.arctan2(evecs[1,0], evecs[0,0]) ) # -90 due to pixel axes orientation, -ang since cv2.ellipse uses clockwise angle

		center_px, center_py = NuscenesPredictionVisualizer.local_coords_to_pixels(*mean)

		cv2.ellipse( img, (center_px, center_py), (length_ax1, length_ax2), ang_1, 0, 360, color, thickness=thickness  )

	@staticmethod
	def plot_GMM(img, gmm_pred, num_modes_to_plot=3):	
		# Given a GMM prediction dictionary, plot the top num_modes_to_plot covariance ellipses.
		# Currently at most 3 modes can be visualized as of now.  Can update in future.
		num_modes = len(gmm_pred.keys())
		probs = [ gmm_pred[k]['mode_probability'] for k in range(num_modes) ]
		
		color_list = [(128, 128, 0), # yellow
		              (0, 128, 128), # cyan
		              (128, 0, 128)] # magenta

		if num_modes_to_plot > 3:
			raise NotImplementedError("The color_list is too short for more than 3 modes. Update code.")

		if num_modes > 1:
			modes_to_plot = np.argsort(probs)[-num_modes_to_plot:]
			modes_to_plot = modes_to_plot[::-1] # reverse so we start with most likely mode first.
		else:
			modes_to_plot = [0]

		for ind_mode, mode in enumerate(modes_to_plot):			
			mean_traj  = gmm_pred[mode]['mus']
			covar_traj = gmm_pred[mode]['sigmas']
			color = color_list[ind_mode]

			for tmstep in range(mean_traj.shape[0]):
				NuscenesPredictionVisualizer.plot_covariance_ellipse(img, mean_traj[tmstep], covar_traj[tmstep], color=color)

		top_mode_probabilities = [probs[k] for k in modes_to_plot]

		return top_mode_probabilities
