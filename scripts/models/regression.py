import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)
from models.multipath_base import MultiPathBase

class Regression(MultiPathBase):
	'''Regression Baseline as used by the MultiPath architecture.
	   Paper Link: https://arxiv.org/pdf/1910.05449.pdf
	'''

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def _create_model(self):
		image_input = Input(shape=self.image_shape, name='image_input')
		state_input = Input(shape=self.past_state_shape, name='state_input')

		backbone = self.resnet_backbone(image_input, state_input)

		# Output: 5 * T trajectory (mu_x, mu_y, std_1, std_2, theta).
		pred = Dense(self.num_timesteps * 5,
			         activation=None)(backbone)

		model = Model(inputs=[image_input, state_input], outputs=pred, name='Regression')
		loss_function = self.likelihood_loss_1()
		metric_function = self.ade_1()

		return model, loss_function, metric_function

	def _extract_gmm_params(self, gmm_pred):
		""" This returns a list of GMM params for each dataset entry.
		    Each GMM param dictionary is generated from the raw model preds. """

		gmm_dicts = []

		for entry in gmm_pred: # iterate over batch_size
			mode_dict = {}

			trajectory = tf.reshape(entry, (self.num_timesteps, 5)) # N_T x 5
			traj_xy = trajectory[:, :2].numpy()
			std1 = tf.math.exp(tf.math.abs(trajectory[:, 2])).numpy()
			std2 = tf.math.exp(tf.math.abs(trajectory[:, 3])).numpy()
			cos_th = tf.math.cos(trajectory[:,4]).numpy()
			sin_th = tf.math.sin(trajectory[:,4]).numpy()

			sigmas = np.ones((self.num_timesteps, 2, 2), dtype=traj_xy.dtype) * np.nan
			for tm, (s1, s2, ct, st) in enumerate(zip(std1, std2, cos_th, sin_th)):
				R_t = np.array([[ct, -st],[st, ct]])
				D   = np.diag([s1**2, s2**2])
				sigmas[tm] = R_t @ D @ R_t.T
			assert np.all(~np.isnan(sigmas))

			mode_dict['mode_probability'] = 1.
			mode_dict['mus'] = traj_xy
			mode_dict['sigmas'] = sigmas

			gmm_dicts.append({0: mode_dict})

		return gmm_dicts

	def ade_1(self):
		""" Returns the average displacement error for a unimodal regression model. """

		def metric(y_true, y_pred):
			# N_B = batch_size, N_T = num_timesteps
			batch_size = y_true.shape[0]
			trajectories = tf.reshape(y_pred, (batch_size, self.num_timesteps, 5))

			residual_trajs = trajectories[:,:,:2] - y_true # N_B x N_T x 2
			displacements = tf.norm(residual_trajs, axis=-1) # N_B x N_T

			avg_disp_error   = tf.reduce_mean(displacements, axis=-1) # N_B
			#final_disp_error = displacements[:, -1] # N_B

			return tf.reduce_mean( avg_disp_error )

		return metric

	def likelihood_loss_1(self):
		""" Simple negative log likelihood loss for a single set of Gaussians per timestep. """

		def loss(y_true, y_pred):
			# loss = -sum_{t=1}^T log N(z_t; mu_t, sigma_t)
			# The covariance has the form: sigma_t = R(theta) @ diag( std1**2, std2**2 ) @ R(theta).T
			# This can be decomposed into log-det-covariance and Mahalanobis distance terms.

			# Tensor dimensions are as follows.  Let us define:
			# N_B = batch_size; N_T = number of timesteps.
			# y_true: N_B x N_T x 2, actual XY trajectory taken
			# y_pred: N_B x (5*N_T), trajectory parameters [mux, muy, log(std1), log(std2), theta]
			batch_size         = y_true.shape[0]
			trajectories = tf.reshape(y_pred, (batch_size, self.num_timesteps, 5))

			# residual_trajs is the difference between the predicted  and the actual trajectory.
			residual_trajs  = y_true - trajectories[:,:,:2]

			# All variables in this code block have shape N_B x N_T.
			# They include the differences to the active mean (dx, dy) and
			# the covariance parameters (log_std1, log_std2, theta).
			dx = residual_trajs[:, :, 0]
			dy = residual_trajs[:, :, 1]
			log_std1 = tf.math.abs( trajectories[:,:,2] )
			log_std2 = tf.math.abs( trajectories[:,:,3] )
			std1     = tf.math.exp(log_std1)
			std2     = tf.math.exp(log_std2)
			cos_th   = tf.math.cos(trajectories[:,:,4])
			sin_th   = tf.math.sin(trajectories[:,:,4])

			# NLL regression loss includes a log-det-covariance term:
			reg_log_det_cov_loss = tf.reduce_sum(log_std1 + log_std2, axis=-1)
			# and a Mahalanobis distance loss, with a scalar expression of the quadratic form:
			reg_mahalanobis_loss = tf.reduce_sum( 0.5 * \
				(tf.square( dx*cos_th + dy*sin_th) / tf.square(std1) + \
				 tf.square(-dx*sin_th + dy*cos_th) / tf.square(std2)),
				 axis=-1)

			# The full loss is a sum of classification and regression losses.
			total_loss = tf.reduce_mean(reg_log_det_cov_loss + reg_mahalanobis_loss)

			return total_loss

		return loss

if __name__ == '__main__':
	mdl = Regression()
