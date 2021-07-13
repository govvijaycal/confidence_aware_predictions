import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input,  Dense
from tensorflow.keras import Model

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)
from models.multipath_base import MultiPathBase

class MultiPath(MultiPathBase):
	'''Implementation of the MultiPath model by Waymo.
	   Paper Link: https://arxiv.org/pdf/1910.05449.pdf
	'''
	def __init__(self,
		         anchors,
		         weights=None,
		         **kwargs
		         ):

		# (1) Load the anchors and check size consistency.
		self.anchors = tf.constant(anchors, dtype=tf.float32)

		# Check shape: should be N_A x N_T x 2.
		assert (len(self.anchors.shape) == 3 and self.anchors.shape[-1] == 2)
		self.num_anchors, self.num_timesteps, _ = self.anchors.shape

		# Check num_timesteps passed to super is consistent with anchors.
		# self.num_timesteps will get overwritten in super.init but shouldn't be an issue.
		assert self.num_timesteps == kwargs.get('num_timesteps')

		print(f'Anchors Shape: {self.anchors.shape}')

		# (2) Load anchor weights for classification loss, checking size consistency.
		if weights is None:
			self.weights = tf.ones((self.num_anchors), dtype=tf.float32)
			print('Using uniform weights.')
		else:
			self.weights = tf.constant(weights, dtype=tf.float32)
			assert self.weights.shape == (self.num_anchors)
			print('Using custom weights.')

		# (3) Initialize common params using super's init.
		super().__init__(**kwargs)

	def _create_model(self):
		image_input = Input(shape=self.image_shape, name='image_input')
		state_input = Input(shape=self.past_state_shape, name='state_input')

		backbone = self.resnet_backbone(image_input, state_input)

		# Output: K mode probabilities, K * 5 * T trajectory (mu_x, mu_y, std_1, std_2, theta).
		pred = Dense(self.num_anchors * (1 + self.num_timesteps * 5),
			         activation=None)(backbone)

		model = Model(inputs=[image_input, state_input], outputs=pred, name='MultiPath')
		loss_function = self.likelihood_loss_mm()
		metric_function = self.ade_mm()

		return model, loss_function, metric_function

	def _extract_gmm_params(self, gmm_pred):
		""" This returns a list of GMM params per mode for each dataset entry.
			Each GMM param dictionary is generated from the raw model preds. """

		gmm_dicts = []

		for entry in gmm_pred: # iterate over batch_size
			gmm_dict = {}

			trajectories = tf.reshape(entry[:-self.num_anchors],
				                      (self.num_anchors, self.num_timesteps, 5))
			anchor_probs = tf.nn.softmax( entry[-self.num_anchors:] ).numpy()
			anchors = self.anchors.numpy()

			for mode_id in range(self.num_anchors):
				traj_xy = (trajectories[mode_id, :, :2].numpy() + anchors[mode_id])

				std1   = tf.math.exp( tf.clip_by_value(tf.math.abs(trajectories[mode_id, :, 2]), 0.0, 5.0) ).numpy()
				std2   = tf.math.exp( tf.clip_by_value(tf.math.abs(trajectories[mode_id, :, 3]), 0.0, 5.0) ).numpy()
				cos_th = tf.math.cos(trajectories[mode_id, :, 4]).numpy()
				sin_th = tf.math.sin(trajectories[mode_id, :, 4]).numpy()

				sigmas = np.ones((self.num_timesteps, 2, 2), dtype=traj_xy.dtype) * np.nan
				for tm, (s1, s2, ct, st) in enumerate(zip(std1, std2, cos_th, sin_th)):
					R_t = np.array([[ct, -st],[st, ct]])
					D   = np.diag([s1**2, s2**2])
					sigmas[tm] = R_t @ D @ R_t.T
				assert np.all(~np.isnan(sigmas))

				mode_dict = {}
				mode_dict['mode_probability'] = anchor_probs[mode_id]
				mode_dict['mus']    = traj_xy
				mode_dict['sigmas'] = sigmas

				gmm_dict[mode_id] = mode_dict

			gmm_dicts.append(gmm_dict)

		return gmm_dicts

	def ade_mm(self):
		""" Returns the average displacement error for a multimodal regression model.
		    It does this by picking the highest probability mode (i.e. this is not min ADE!). """

		def metric(y_true, y_pred):
			# N_B = batch_size, N_T = num_timesteps
			batch_size = y_true.shape[0]
			trajectories = tf.reshape(y_pred[:,:-self.num_anchors],
				                      (batch_size, self.num_anchors, self.num_timesteps, 5))
			anchor_probs = tf.nn.softmax( y_pred[:,-self.num_anchors:] )

			active_modes = tf.math.top_k(anchor_probs, k=1).indices
			active_indices = tf.concat( (tf.reshape(tf.range(batch_size), (batch_size, 1)), \
				                         active_modes), axis=1) # N_B x 2

			trajectories_xy = trajectories[:, :, :, :2] + self.anchors

			active_trajs = tf.gather_nd(trajectories_xy, active_indices) # N_B x N_T x 2
			residual_trajs = active_trajs - y_true # N_B x N_T x 2
			displacements = tf.norm(residual_trajs, axis=-1) # N_B x N_T

			avg_disp_error   = tf.reduce_mean( displacements, axis=-1) # N_B
			#final_disp_error = displacements[:, -1] # N_B

			return tf.reduce_mean( avg_disp_error )

		return metric

	def likelihood_loss_mm(self):

		def loss(y_true, y_pred):
			# The loss involves first finding the mode (i.e. the anchor trajectory) nearest
			# the demonstrated trajectory in y_true.  Call this anchor/mode index k_cl.
			# Then the negative log likelihood loss for one instance is found by summing:
			# (1) classification loss = -log P(mode = k_cl)
			# (2) regression loss = -sum_{t=1}^T log N(z_t; a_t^{k_cl}+mu_t^{k_cl}, sigma_t^{k_cl})
			#     item 2 can be decomposed into log-det-covariance and Mahalanobis distance terms.
			# The covariance has the form: sigma_t = R(theta) @ diag( std1**2, std2**2 ) @ R(theta).T

			# Tensor dimensions are as follows.  Let us define:
			# N_B = batch_size; N_A = number of anchors; N_T = number of timesteps.
			# y_true: N_B x N_T x 2, actual XY trajectory taken
			# y_pred: N_B x (N_A x (1 + 5*N_T)), GMM mode probabilities (anchor probs)
			#         and offset trajectory parameters [mux, muy, log(std1), log(std2), theta]
			batch_size         = y_true.shape[0]
			trajectories       = tf.reshape(y_pred[:,:-self.num_anchors],
				                            (batch_size, self.num_anchors, self.num_timesteps, 5))
			anchor_probs       = tf.nn.softmax( y_pred[:,-self.num_anchors:] )

			# Find the nearest anchor mode (k_cl) by using Euclidean distance between trajectories.
			# The difference involves tensors with shapes N_A x N_T x 2 and N_B x 1 x N_T x 2 after
			# the latter passes through expand_dims.  The broadcasting seems to work, as the diff.
			# has shape N_B x N_A x N_T x 2 before the norm/sum operations are performed.  Tried
			# with y_true being composed of anchor elements -> works fine.
			distance_to_anchors = tf.math.reduce_sum(tf.norm(
				                   self.anchors - tf.expand_dims(y_true, axis=1),
				                   axis=-1), axis=-1)

			nearest_mode         = tf.argmin(distance_to_anchors, axis=-1) # shape N_B
			nearest_mode_indices = tf.stack([ tf.range(batch_size, dtype=tf.int64), nearest_mode ],
			                                axis=-1) # used to extract the correct mode over a batch

			# Classification loss is a simple NLL with only the closest mode being penalized.
			# This is based on the assumption that only a single mode is active at a time.
			loss_weights = tf.gather(self.weights, nearest_mode)
			class_loss = -tf.math.log(tf.gather_nd(anchor_probs, nearest_mode_indices))

			# trajectories_xy contains the mean xy trajectory for all modes; N_B x N_A x N_T x 2.
			# nearest_trajs is the mean xy trajectory for the closest mode;  N_B x N_T x 2.
			# residual_trajs is the difference between the nearest_trajs and the actual trajectory.
			trajectories_xy = trajectories[:, :, :, :2] + self.anchors
			nearest_trajs   = tf.gather_nd(trajectories_xy, nearest_mode_indices)
			residual_trajs  = y_true - nearest_trajs

			# All variables in this code block have shape N_B x N_T.
			# They include the differences to the active mean (dx, dy) and
			# the covariance parameters (log_std1, log_std2, theta).
			dx = residual_trajs[:, :, 0]
			dy = residual_trajs[:, :, 1]
			log_std1 = tf.clip_by_value( tf.math.abs(tf.gather_nd(trajectories[:,:,:,2], nearest_mode_indices)),
				                         0.0, 5.0 )
			log_std2 = tf.clip_by_value( tf.math.abs(tf.gather_nd(trajectories[:,:,:,3], nearest_mode_indices)),
				                         0.0, 5.0 )
			std1     = tf.math.exp(log_std1)
			std2     = tf.math.exp(log_std2)
			cos_th   = tf.math.cos( tf.gather_nd(trajectories[:,:,:,4], nearest_mode_indices) )
			sin_th   = tf.math.sin( tf.gather_nd(trajectories[:,:,:,4], nearest_mode_indices) )

			# NLL regression loss includes a log-det-covariance term:
			reg_log_det_cov_loss = tf.reduce_sum(log_std1 + log_std2, axis=-1)
			# and a Mahalanobis distance loss, with a scalar expression of the quadratic form:
			reg_mahalanobis_loss = tf.reduce_sum( 0.5 * \
				(tf.square( dx*cos_th + dy*sin_th) / tf.square(std1) + \
				 tf.square(-dx*sin_th + dy*cos_th) / tf.square(std2)),
				 axis=-1)

			# The full loss is a sum of classification and regression losses.
			total_loss = tf.reduce_mean(loss_weights * ( class_loss + reg_log_det_cov_loss + reg_mahalanobis_loss ))

			return total_loss

		return loss

if __name__ == '__main__':
	repo_path = os.path.abspath(__file__).split('scripts')[0]
	datadir = repo_path + 'data'

	anchors = np.load(datadir + '/nuscenes_clusters_16.npy')
	weights = np.load(datadir + '/nuscenes_clusters_16_weights.npy')
	m = MultiPath(anchors=anchors, weights=weights, num_timesteps=12, num_hist_timesteps=2)