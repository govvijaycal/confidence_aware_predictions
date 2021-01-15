import os
import sys
import glob
import numpy as np
np.random.seed(0)
import tensorflow as tf

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)

from datasets.tfrecord_utils import _parse_function
from models.multipath import MultiPath

def numpy_ade(anchors, y_true, y_pred):
	# anchors: num_anchors x num_timesteps x 2
	# y_true: batch_size x num_timesteps x 2
	# y_pred: batch_size x (num_anchors x (1 + 5 * num_timesteps))

	batch_size, num_timesteps, _ = y_true.shape
	num_anchors = len(anchors)

	trajectories = np.reshape(y_pred[:,:-num_anchors], 
		                      (batch_size, num_anchors, num_timesteps, 5))
	anchor_probs = tf.nn.softmax( y_pred[:,-num_anchors:], axis=1)

	active_modes = np.argmax(anchor_probs, axis=1)

	ades = []

	for batch_ind, active_mode in enumerate(active_modes):
		active_traj_xy = trajectories[batch_ind, active_mode, :, :2] + anchors[active_mode]

		residual_traj = y_true[batch_ind] - active_traj_xy

		displacements = np.linalg.norm(residual_traj, axis=-1)

		ades.append(np.mean(displacements))

	return np.mean(ades)

def numpy_nll(anchors, y_true, y_pred):
	# anchors: num_anchors x num_timesteps x 2
	# y_true: batch_size x num_timesteps x 2
	# y_pred: batch_size x (num_anchors x (1 + 5 * num_timesteps))

	batch_size, num_timesteps, _ = y_true.shape
	num_anchors = len(anchors)

	neg_log_likelihoods = []
	active_modes = []

	for batch_ind in range(batch_size):
		current_traj = y_true[batch_ind]
		dists_to_anchor = [np.sum(np.linalg.norm(current_traj - anc, axis=-1), axis=-1) for anc in anchors]
		active_modes.append(np.argmin(dists_to_anchor))

	for batch_ind, active_mode in enumerate(active_modes):
		current_pred = y_pred[batch_ind]
		active_pred_traj = current_pred[:-num_anchors].reshape(num_anchors, num_timesteps, 5)[active_mode]
		active_pred_prob = tf.nn.softmax(current_pred[-num_anchors:])[active_mode]

		log_det_loss = 0.
		mahalanobis_loss = 0.

		for t in range(num_timesteps):
			mean_xy = active_pred_traj[t, :2] + anchors[active_mode, t, :]
			residual_xy = y_true[batch_ind, t, :] - mean_xy
			std_x = np.exp( max(0., active_pred_traj[t, 2]) )
			std_y = np.exp( max(0., active_pred_traj[t, 3]) )
			rho   = 0.9*np.tanh(active_pred_traj[t, 4])
			cov_xy  = np.array([[std_x**2, rho*std_x*std_y], \
				                [rho*std_x*std_y, std_y**2]])

			log_det_loss += 0.5 * np.log( np.linalg.det(cov_xy) )
			mahalanobis_loss += 0.5 * residual_xy.T @ np.linalg.inv(cov_xy) @ residual_xy

		nll = -np.log(active_pred_prob) + log_det_loss + mahalanobis_loss

		neg_log_likelihoods.append(nll)

	return np.mean(neg_log_likelihoods)

def make_correct_mode_predictions(anchors, gt_indices = [0, 5, 10]):
	# y_true: batch_size x num_timesteps x 2
	# y_pred: batch_size x (num_anchors x (1 + 5 * num_timesteps))
	
	batch_size    = len(gt_indices)
	num_anchors   = anchors.shape[0]
	num_timesteps = anchors.shape[1]

	y_true = np.array( [anchors[ind] for ind in gt_indices] )

	y_pred = np.random.rand(batch_size, num_anchors * (1 + 5*num_timesteps))

	for batch_ind, gt_anchor in enumerate(gt_indices):
		y_pred[batch_ind, -num_anchors:] = [i == gt_anchor for i in range(num_anchors)]

		pred_xy = np.random.rand(*y_true.shape[1:])
		pred_cov = np.ones((num_timesteps, 3)) * [np.log(1.), np.log(1.), 0.01]
		pred = np.concatenate( (pred_xy, pred_cov), axis=1)

		start = gt_anchor * 5 * num_timesteps
		stop  = start + 5 * num_timesteps
		y_pred[batch_ind, start:stop] = pred.flatten()

	return y_true.astype(np.float32), y_pred.astype(np.float32)

def test1_single_batch(multipath):
	tf_ade_metric = multipath.ade()
	tf_ll_loss = multipath.likelihood_loss()
	anchors = multipath.anchors.numpy()

	y_true, y_pred = make_correct_mode_predictions(anchors)

	print('Test1: ADE/NLL on selected correct mode predictions.')
	np_ade = numpy_ade(anchors, y_true, y_pred)
	np_nll = numpy_nll(anchors, y_true, y_pred)

	tf_ade = tf_ade_metric(y_true, y_pred).numpy()
	tf_nll = tf_ll_loss(y_true, y_pred).numpy()

	ade_diff = np.abs(np_ade - tf_ade)
	nll_diff = np.abs(np_nll - tf_nll)

	print( 'ADE Diff: {}'.format(ade_diff) )
	print( 'NLL Diff: {}'.format(nll_diff) )

	assert (ade_diff < 1e-6) and (nll_diff < 1e-6)

def test2_full_dataset(multipath, dataset):
	np_ades = []
	np_nlls = []
	tf_ades = []
	tf_nlls = []

	anchors = multipath.anchors.numpy()
	tf_ade_metric = multipath.ade()
	tf_ll_loss = multipath.likelihood_loss()

	for entry in dataset:
		img = tf.cast(entry['image'], dtype=tf.float32) / 127.5 - 1.0
		state = tf.cast( 
			        tf.concat([entry['velocity'], entry['acceleration'], entry['yaw_rate']], -1),
			        dtype=tf.float32)
		future_xy = tf.cast(entry['future_poses_local'][:,:,:2],
			                dtype=tf.float32)
		
		pred = multipath.model.predict_on_batch([img, state])

		tf_ade = tf_ade_metric(future_xy, pred).numpy()
		tf_nll = tf_ll_loss(future_xy, pred).numpy()

		tf_ades.append(tf_ade)
		tf_nlls.append(tf_nll)

		np_ade = numpy_ade(anchors, future_xy.numpy(), pred)
		np_nll = numpy_nll(anchors, future_xy.numpy(), pred)

		np_ades.append(np_ade)
		np_nlls.append(np_nll)

	import matplotlib.pyplot as plt
	plt.subplot(211)
	ade_diffs = [x-y for (x,y) in zip(np_ades, tf_ades)]
	plt.hist( ade_diffs )
	plt.ylabel('ADE Diff')

	plt.subplot(212)
	nll_diffs = [x-y for (x,y) in zip(np_nlls, tf_nlls)]
	plt.hist( nll_diffs )
	plt.ylabel('NLL Diff')

	print('Difference stats:')
	print( '\tade_mean: {}, ade_max:{}'.format(np.mean(ade_diffs), np.max(ade_diffs)) )
	print( '\tnll_mean: {}, nll_max:{}'.format(np.mean(nll_diffs), np.max(nll_diffs)) )
	plt.show()

	assert (np.mean(ade_diffs) < 1e-3) and (np.mean(nll_diffs) < 1e-3)

if __name__ == '__main__':
	repo_path = os.path.abspath(__file__).split('scripts')[0]
	datadir = repo_path + 'data'

	dataset   = glob.glob(datadir + '/nuscenes_train_val*.record')
	dataset = tf.data.TFRecordDataset(dataset)
	dataset = dataset.map(_parse_function)
	dataset = dataset.batch(32)

	m = MultiPath(np.load(datadir + '/nuscenes_clusters_16.npy'))
	
	test1_single_batch(m)
	test2_full_dataset(m, dataset)
