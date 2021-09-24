import os
import sys
import glob
import numpy as np
import tensorflow as tf

import pytest

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)

from datasets.tfrecord_utils import _parse_function
from datasets.splits import NUSCENES_VAL, L5KIT_VAL
from models.multipath import MultiPath

# To address some CuDNN initialization errors, idk why needed only here.
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

'''
Test Fixture Parametrized by Dataset.
'''
@pytest.fixture(scope="module", params=['nuscenes', 'l5kit'])
def multipath_and_dataset(request):
	repo_path = os.path.abspath(__file__).split('scripts')[0]
	datadir = repo_path + 'data'

	if request.param == 'nuscenes':
		num_timesteps = 12
		num_hist_timesteps = 2
		anchors = np.load(datadir + '/nuscenes_clusters_16.npy')
		tfrecords = NUSCENES_VAL
	elif request.param == 'l5kit':
		num_timesteps = 25
		num_hist_timesteps = 5
		anchors = np.load(datadir + '/l5kit_clusters_16.npy')
		tfrecords = L5KIT_VAL

	multipath = MultiPath(num_timesteps=num_timesteps,
		                  num_hist_timesteps=num_hist_timesteps,
		                  anchors=anchors)

	dataset = tf.data.TFRecordDataset(tfrecords)
	dataset = dataset.map(_parse_function)
	dataset = dataset.batch(8)

	return multipath, dataset

'''
Util Test Functions: Metrics/Loss/Fake Predictions
'''
def numpy_ade(anchors, y_true, y_pred):
	# Returns the average displacement error (ADE) for a batch of data.
	# Use of Numpy to compare vs. complicted TF logic (e.g. tf.gather_nd).
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
	# Returns Negative Log Likelihood Loss for a batch of data.
	# Implemented in Numpy to compare against complicated TF logic to get the
	# active trajectories (closest mode).
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

	LOG_STD_MIN = np.float32(0.)
	LOG_STD_MAX = np.float32(5.)

	for batch_ind, active_mode in enumerate(active_modes):
		current_pred = y_pred[batch_ind]
		active_pred_traj = current_pred[:-num_anchors].reshape(num_anchors, num_timesteps, 5)[active_mode]
		active_pred_prob = tf.nn.softmax(current_pred[-num_anchors:])[active_mode]

		log_det_loss = np.float32(0.)
		mahalanobis_loss = np.float32(0.)

		for t in range(num_timesteps):
			mean_xy = active_pred_traj[t, :2] + anchors[active_mode, t, :]
			residual_xy = y_true[batch_ind, t, :] - mean_xy
			std_1 = np.exp( np.clip(np.abs(active_pred_traj[t, 2]), LOG_STD_MIN, LOG_STD_MAX) )
			std_2 = np.exp( np.clip(np.abs(active_pred_traj[t, 3]), LOG_STD_MIN, LOG_STD_MAX) )
			cos_th = np.cos(active_pred_traj[t, 4])
			sin_th = np.sin(active_pred_traj[t, 4])
			R_th   = np.array([[cos_th, -sin_th],
				               [sin_th,  cos_th]])
			cov_xy = R_th @ np.diag([std_1**2, std_2**2]) @ R_th.T
			log_det_loss += np.float32(0.5) * np.log( np.linalg.det(cov_xy) )
			mahalanobis_loss += np.float32(0.5) * residual_xy.T @ np.linalg.inv(cov_xy) @ residual_xy

		nll = -np.log(active_pred_prob) + log_det_loss + mahalanobis_loss

		neg_log_likelihoods.append(nll)

	return np.mean(neg_log_likelihoods)

def make_correct_mode_predictions(anchors, gt_indices = [0, 5, 10]):
	# This function makes fake predictions according to the following scheme:
	# The batch_size is len(gt_indices) and classification error is 0.
	# The true trajectories are simply the anchors identified by gt_indices.
	# y_pred is a bunch of random floats, aside from the values given for the
	# gt_index which are random mean and fixed covariance parameters.
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
		pred_cov = np.ones((num_timesteps, 3)) * [np.log(1.), np.log(5.), np.radians(30.)]
		pred = np.concatenate( (pred_xy, pred_cov), axis=1)

		start = num_anchors + gt_anchor * 5 * num_timesteps
		stop  = start + 5 * num_timesteps
		y_pred[batch_ind, start:stop] = pred.flatten() # Only assign these custom values for the gt_anchor trajectory.

	return y_true.astype(np.float32), y_pred.astype(np.float32)

'''
Test Suite
'''
def test_fake_predictions(multipath_and_dataset):
	# Check that ADE / LL match for a set of fake predictions.
	multipath, _ = multipath_and_dataset
	tf_ade_metric = multipath.ade_mm()
	tf_ll_loss = multipath.likelihood_loss_mm()
	anchors = multipath.anchors.numpy()

	y_true, y_pred = make_correct_mode_predictions(anchors)

	np_ade = numpy_ade(anchors, y_true, y_pred)
	np_nll = numpy_nll(anchors, y_true, y_pred)

	tf_ade = tf_ade_metric(y_true, y_pred).numpy()
	tf_nll = tf_ll_loss(y_true, y_pred).numpy()

	assert np.isclose(np_ade, tf_ade) and np.isclose(np_nll, tf_nll)

def test_full_dataset(multipath_and_dataset, max_iters=10):
	# Check that ADE / LL match across max_iters of a tfrecord dataset.
	multipath, dataset = multipath_and_dataset
	anchors = multipath.anchors.numpy()
	tf_ade_metric = multipath.ade_mm()
	tf_ll_loss = multipath.likelihood_loss_mm()

	for ind_entry, entry in enumerate(dataset):
		img, past_states, future_xy = multipath.preprocess_entry(entry)
		pred = multipath.model.predict_on_batch([img, past_states])

		tf_ade = tf_ade_metric(future_xy, pred).numpy()
		tf_nll = tf_ll_loss(future_xy, pred).numpy()

		np_ade = numpy_ade(anchors, future_xy.numpy(), pred)
		np_nll = numpy_nll(anchors, future_xy.numpy(), pred)

		assert np.isclose(np_ade, tf_ade) and np.isclose(np_nll, tf_nll)

		if max_iters is not None and ind_entry > max_iters:
			break
