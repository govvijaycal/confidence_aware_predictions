import sys
import os
import glob
import numpy as np
from datetime import datetime

import tensorflow as tf 
import tensorflow.keras.backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, LeakyReLU, Conv2D, Dense, \
                                    BatchNormalization, Flatten, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)
from datasets.tfrecord_utils import _parse_function

class MultiPath(object):
	'''Implementation of the MultiPath model by Waymo.  
	   Paper Link: https://arxiv.org/pdf/1910.05449.pdf
	'''
	def __init__(self, anchors):
		self.num_anchors = 16
		self.num_timesteps = 12

		self.anchors = tf.constant(anchors, dtype=tf.float32)
		assert self.anchors.shape == (self.num_anchors, self.num_timesteps, 2)
		
		self.init_lr = 1e-3
		self.model = self._create_model()
		self.trained = False

	def _create_model(self):
		image_input = Input(shape=(500,500,3), name='image_input')
		state_input = Input(shape=(3,), name='state_input')

		# Reference to save LeakyReLU layer: 
		# https://www.gitmemory.com/issue/keras-team/keras/6532/481666883
		my_leakyrelu = LeakyReLU(alpha=0.3)
		my_leakyrelu.__name__ = 'lrelu'
	
		base_model = ResNet50(include_top=False, 
			                  weights='imagenet',
			                  input_shape=(500,500,3),
			                  pooling=False)
		
		for layer in base_model.layers:
			if 'conv5' in layer.name:
				layer.trainable = False
			else:
				layer.trainable = False
			# conv2: 125 x 125
			# conv3: 63 x 63
			# conv4: 32 x 32
			# conv5: 16 x 16		

		x = base_model(image_input)
		x = Conv2D(8, (1,1), strides=1, 
			       kernel_regularizer=l2(1e-3), 
			       bias_regularizer=l2(1e-3))(x)
		x = BatchNormalization()(x)
		x = Flatten()(x)

		x = Dense(1024,
			      activation=my_leakyrelu,
			      kernel_regularizer=l2(1e-3),
			      bias_regularizer=l2(1e-3))(x)

		y = concatenate([x, state_input])

		for _ in range(2):
			y = BatchNormalization()(y)
			y = Dense(1024,
			      activation=my_leakyrelu,
			      kernel_regularizer=l2(1e-3),
			      bias_regularizer=l2(1e-3))(y)

		# Output: K mode probabilities, K * 5 * T trajectory (mu_x, mu_y, sigma_x, sigma_y, rho)
		pred = Dense(self.num_anchors * (1 + self.num_timesteps * 5), 
			         activation=None,
			         kernel_regularizer=l2(1e-3),
			         bias_regularizer=l2(1e-3))(y)

		model = Model(inputs=[image_input, state_input], outputs=pred)
		import pdb; pdb.set_trace()

		model.compile(loss=self.likelihood_loss(), 
			          metrics=self.ade(),
			          optimizer=SGD(lr=self.init_lr, momentum=0.9, nesterov=True, clipnorm=10.))
		return model

	def ade(self):

		def metric(y_true, y_pred):
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

	def likelihood_loss(self):

		def loss(y_true, y_pred):
			# The loss involves first finding the mode (i.e. the anchor trajectory) nearest
			# the demonstrated trajectory in y_true.  Call this anchor/mode index k_cl. 
			# Then the negative log likelihood loss for one instance is found by summing:
			# (1) classification loss = -log P(mode = k_cl)
			# (2) regression loss = -sum_{t=1}^T log N(z_t; a_t^{k_cl}+mu_t^{k_cl}, sigma_t^{k_cl})
			#     item 2 can be decomposed into log-det-covariance and Mahalanobis distance terms.

			# Tensor dimensions are as follows.  Let us define:
			# N_B = batch_size; N_A = number of anchors; N_T = number of timesteps.
			# y_true: N_B x N_T x 2, actual XY trajectory taken
			# y_pred: N_B x N_A x (1 + 5*N_T), GMM mode probabilities (anchor probs) 
			#         and offset trajectory parameters [mux, muy, log(stdx), log(stdy), rho_xy]
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
			class_loss = -tf.math.log(tf.gather_nd(anchor_probs, nearest_mode_indices))

			# trajectories_xy contains the mean xy trajectory for all modes; N_B x N_A x N_T x 2.
			# nearest_trajs is the mean xy trajectory for the closest mode;  N_B x N_T x 2.
			# residual_trajs is the difference between the nearest_trajs and the actual trajectory.
			trajectories_xy = trajectories[:, :, :, :2] + self.anchors
			nearest_trajs   = tf.gather_nd(trajectories_xy, nearest_mode_indices)						
			residual_trajs  = nearest_trajs - y_true

			# All variables in this code block have shape N_B x N_T.
			# They include the differences to the active mean (dx, dy) and
			# the covariance parameters (log_stdx, log_stdy, rho_xy).
			dx = residual_trajs[:, :, 0]
			dy = residual_trajs[:, :, 1]
			log_stdx = tf.gather_nd(trajectories[:,:,:,2], nearest_mode_indices)
			log_stdy = tf.gather_nd(trajectories[:,:,:,3], nearest_mode_indices)
			stdx     = tf.math.exp(log_stdx)
			stdy     = tf.math.exp(log_stdy)
			rho_xy   = tf.math.tanh( 
			               tf.gather_nd(trajectories[:,:,:,4], nearest_mode_indices) 
			               ) # rho in [-1, 1]
		
			# NLL regression loss includes a log-det-covariance term:
			reg_log_det_cov_loss = tf.reduce_sum( 0.5 * tf.math.log(1. - tf.square(rho_xy))
				                              + log_stdx + log_stdy, axis=-1)
			# and a Mahalanobis distance loss, with a scalar expression of the quadratic form:
			reg_mahalanobis_loss = tf.reduce_sum( 0.5 / (1 - tf.square(rho_xy)) * \
			    ( tf.square(dx)/tf.square(stdx) + tf.square(dy)/tf.square(stdy) - \
			      2.*rho_xy*dx*dy/(stdx*stdy) ), axis=-1)

			# The full loss is a sum of classification and regression losses.
			total_loss = tf.reduce_mean( class_loss + reg_log_det_cov_loss + reg_mahalanobis_loss )
			
			return total_loss

		return loss

	def fit(self, 
		    train_set, 
		    val_set, 
		    logdir=None, 
		    log_epoch_freq=10,
		    save_epoch_freq=20,
		    num_epochs=100, 
		    batch_size=32):

		if logdir is None:
			raise ValueError("Need to provide a logdir for TensorBoard logging and model saving.")
		os.makedirs(logdir, exist_ok=True)

        # Note: the following pipeline does two levels of shuffling:
        # file order shuffling ("global"/"macro" level)
        # dataset instance shuffling ("local" level)
		files   = tf.data.Dataset.from_tensor_slices(train_set)
		files   = files.shuffle(buffer_size=len(train_set), reshuffle_each_iteration=True) 
		dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x), 
			                       cycle_length=2, block_length=16)
		dataset = dataset.map(_parse_function)
		dataset = dataset.shuffle(10*batch_size, reshuffle_each_iteration=True)
		dataset = dataset.batch(batch_size)
		dataset = dataset.prefetch(2)

		# Val dataset only used to evaluate loss/metrics so we don't random or large batches.		
		val_dataset = tf.data.TFRecordDataset(val_set)
		val_dataset = val_dataset.map(_parse_function)
		val_dataset = val_dataset.batch(2) # making this smaller to reduce potential OOM issues

		init_lr = K.get_value(self.model.optimizer.lr)

		tensorboard = TensorBoard(log_dir=logdir,
			                      histogram_freq=0,
			                      write_graph=False)
		tensorboard.set_model(self.model)

		for epoch in range(1, num_epochs+1):
			print('Epoch {} started at {}'.format(epoch, datetime.now()))

			losses = []
			ades   = []			

			for entry in dataset:
				# Note: this is equivalent to the 'tf' style of image preprocessing.
				# Technically, Resnet50 uses the 'caffe' style but it should not matter too much
				# since anyway the domain is very different from ImageNet.  We just want to get
				# features out so the exact color ordering (BGR vs RGB) and mean pixels are not 
				# so relevant.
				img = tf.cast(entry['image'], dtype=tf.float32) / 127.5 - 1.0
				
				state = tf.cast( 
					        tf.concat([entry['velocity'], entry['acceleration'], entry['yaw_rate']], -1),
					        dtype=tf.float32)
				future_xy = tf.cast(entry['future_poses_local'][:,:,:2],
					                dtype=tf.float32)
								
				batch_loss, batch_ade = self.model.train_on_batch([img, state], future_xy)
				losses.append(batch_loss)
				ades.append(batch_ade)				

			epoch_loss = np.mean(losses)
			epoch_ade  = np.mean(ades)			
			print('\t Train Loss: {}, Train ADE: {}'.format(epoch_loss, epoch_ade))

			# TODO: learning rate decay/ cyclical learning rate.
	
			if log_epoch_freq and epoch % log_epoch_freq == 0:
				val_losses = []
				val_ades   = []

				for entry in val_dataset:
					img = tf.cast(entry['image'], dtype=tf.float32) / 127.5 - 1.0
					state = tf.cast( 
						        tf.concat([entry['velocity'], entry['acceleration'], entry['yaw_rate']], -1),
						        dtype=tf.float32)
					future_xy = tf.cast(entry['future_poses_local'][:,:,:2],
						                dtype=tf.float32)
					
					batch_loss, batch_ade = self.model.test_on_batch([img, state], future_xy)
					val_losses.append(batch_loss)
					val_ades.append(batch_ade)					

				val_epoch_loss = np.mean(val_losses)
				val_epoch_ade  = np.mean(val_ades)				
				print('\t Val Loss: {}, Val ADE: {}'.format(
					  val_epoch_loss, val_epoch_ade))

				tensorboard.on_epoch_end(epoch, {'lr'       : K.get_value(self.model.optimizer.lr),
					                             'loss'     : epoch_loss,
					                             'ADE'      : epoch_ade,
					                             'val_loss' : val_epoch_loss,
					                             'val_ADE'  : val_epoch_ade})

			if save_epoch_freq and epoch % save_epoch_freq == 0:
				filename = logdir + 'multipath_{0:05d}_epochs'.format(epoch)
				self.save_weights(filename)
				print('Saving model weights at epoch {} to {}.'.format(epoch, filename))

		tensorboard.on_train_end(None)
		self.trained = True

	# def predict(self, test_set, batch_size=1):
	# 	dataset = tf.data.TFRecordDataset(test_set)
	# 	dataset = dataset.map(_parse_function)
	# 	dataset = dataset.batch(batch_size)

	# 	for entry in dataset:
	# 		# do stuff
	# 		out = self.model.predict(...)
	# 		# make a prediction type object or list of dicts?

	def save_weights(self, path):
		self.model.save_weights(path + '.h5')

	def load_weights(self, path):
		self.model.load_weights(path + '.h5')

	def save_model(self, model_dir):
		# Save the entire the model for deployment. Default format is a Tensorflow SavedModel.
		# Note we don't implement the load_model here, since it would just overwrite self.model.
		# This function is meant to be used more at the end after all training is done.
		self.model.save(model_dir)

if __name__ == '__main__':
	repo_path = os.path.abspath(__file__).split('scripts')[0]
	datadir = repo_path + 'data'
	logdir = repo_path + 'log/multipath_nuscenes_16_full/'

	anchors = np.load(datadir + '/nuscenes_clusters_16.npy')

	train_set = glob.glob(datadir + '/nuscenes_train*.record')
	train_set = [x for x in train_set if 'val' not in x]

	val_set   = glob.glob(datadir + '/nuscenes_train_val*.record')

	m = MultiPath(anchors=anchors)
	m.fit(train_set, val_set, logdir=logdir, num_epochs=10, batch_size=32)