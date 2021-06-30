import os
import sys
import glob
import numpy as np
import tensorflow as tf

import pytest

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)

from datasets.tfrecord_utils import _parse_function
from models.regression import Regression

# To address some CuDNN initialization errors, idk why needed only here.
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

'''
Test Fixture Parametrized by Dataset.
'''
@pytest.fixture(scope="module", params=['nuscenes', 'l5kit'])
def regression_and_dataset(request):
	repo_path = os.path.abspath(__file__).split('scripts')[0]
	datadir = repo_path + 'data'

	if request.param == 'nuscenes':
		num_timesteps = 12
		num_hist_timesteps = 2
		tfrecords = glob.glob(datadir + '/nuscenes_train_val*.record')
	elif request.param == 'l5kit':
		num_timesteps = 25
		num_hist_timesteps = 5
		tfrecords = glob.glob(datadir + '/l5kit_val*0.record')

	regression = Regression(num_timesteps=num_timesteps,
		                    num_hist_timesteps=num_hist_timesteps)

	dataset = tf.data.TFRecordDataset(tfrecords)
	dataset = dataset.map(_parse_function)
	dataset = dataset.batch(8)

	return regression, dataset

'''
Util Test Functions: Metrics/Loss/Fake Predictions
'''
def numpy_ade(y_true, y_pred):
	# Returns the average displacement error (ADE) for a batch of data.
	# y_true: batch_size x num_timesteps x 2
	# y_pred: batch_size x (5 * num_timesteps)

	batch_size, num_timesteps, _ = y_true.shape
	trajectories = np.reshape(y_pred, (batch_size, num_timesteps, 5))

	ades = []

	for batch_ind in range(batch_size):
		pred_traj = trajectories[batch_ind, :, :2] # num_timesteps x 2

		residual_traj = y_true[batch_ind] - pred_traj

		displacements = np.linalg.norm(residual_traj, axis=-1)

		ades.append(np.mean(displacements))

	return np.mean(ades)

def numpy_nll(y_true, y_pred):
	# Returns Negative Log Likelihood Loss for a batch of data.
	# y_true: batch_size x num_timesteps x 2
	# y_pred: batch_size x (5 * num_timesteps)

	batch_size, num_timesteps, _ = y_true.shape
	trajectories = np.reshape(y_pred, (batch_size, num_timesteps, 5))

	neg_log_likelihoods = []

	for batch_ind in range(batch_size):
		pred_traj = trajectories[batch_ind]
		true_traj = y_true[batch_ind]

		log_det_loss     = np.float32(0.)
		mahalanobis_loss = np.float32(0.)

		for t in range(num_timesteps):
			residual_xy = true_traj[t, :] - pred_traj[t, :2]
			std_1 = np.exp( np.abs(pred_traj[t, 2]) )
			std_2 = np.exp( np.abs(pred_traj[t, 3]) )
			cos_th = np.cos(pred_traj[t, 4])
			sin_th = np.sin(pred_traj[t, 4])
			R_th   = np.array([[cos_th, -sin_th],
				               [sin_th,  cos_th]])
			cov_xy = R_th @ np.diag([std_1**2, std_2**2]).astype(R_th.dtype) @ R_th.T
			log_det_loss += np.float32(0.5) * np.log( np.linalg.det(cov_xy) )
			mahalanobis_loss += np.float32(0.5) * residual_xy.T @ np.linalg.inv(cov_xy) @ residual_xy

		nll = log_det_loss + mahalanobis_loss
		neg_log_likelihoods.append(nll)

	return np.mean(neg_log_likelihoods)

def make_fake_predictions(num_timesteps):
	# This function makes fake predictions with XY noise about the true trajectory.
	# y_true: batch_size x num_timesteps x 2
	# y_pred: batch_size x (5 * num_timesteps)

	y_true = np.ones((4, num_timesteps, 2)) * np.nan
	y_true[0] = np.zeros((num_timesteps, 2))                                     # stationary
	y_true[1] = np.array([[5*t, 0.] for t in range(num_timesteps)])              # constant velocity
	y_true[2] = np.array([[5*t + -0.1*t**2, 0.] for t in range(num_timesteps)])  # constant acceleration
	y_true[3] = np.array([[np.cos(t), np.sin(t)] for t in range(num_timesteps)]) # circular motion

	batch_size = y_true.shape[0]

	random_mean  = np.random.rand(*y_true.shape) + y_true
	fixed_covar  = np.ones((batch_size, num_timesteps, 3)) * \
	                   [np.log(1.), np.log(5.), np.radians(30.)]
	y_pred = np.concatenate((random_mean, fixed_covar), axis=-1)
	y_pred = y_pred.reshape((batch_size, 5 * num_timesteps))

	return y_true.astype(np.float32), y_pred.astype(np.float32)

'''
Test Suite
'''
def test_fake_predictions(regression_and_dataset):
	# Check that ADE / LL match for a set of fake predictions.
	regression, _ = regression_and_dataset
	tf_ade_metric = regression.ade_1()
	tf_ll_loss = regression.likelihood_loss_1()

	y_true, y_pred = make_fake_predictions(regression.num_timesteps)

	np_ade = numpy_ade(y_true, y_pred)
	np_nll = numpy_nll(y_true, y_pred)

	tf_ade = tf_ade_metric(y_true, y_pred).numpy()
	tf_nll = tf_ll_loss(y_true, y_pred).numpy()

	assert np.isclose(np_ade, tf_ade) and np.isclose(np_nll, tf_nll)

def test_full_dataset(regression_and_dataset, max_iters=10):
	# Check that ADE / LL match across max_iters of a tfrecord dataset.
	regression, dataset = regression_and_dataset
	tf_ade_metric = regression.ade_1()
	tf_ll_loss = regression.likelihood_loss_1()

	for ind_entry, entry in enumerate(dataset):
		img, past_states, future_xy = regression.preprocess_entry(entry)
		pred = regression.model.predict_on_batch([img, past_states])

		tf_ade = tf_ade_metric(future_xy, pred).numpy()
		tf_nll = tf_ll_loss(future_xy, pred).numpy()

		np_ade = numpy_ade(future_xy.numpy(), pred)
		np_nll = numpy_nll(future_xy.numpy(), pred)

		assert np.isclose(np_ade, tf_ade) and np.isclose(np_nll, tf_nll)

		if max_iters is not None and ind_entry > max_iters:
			break
