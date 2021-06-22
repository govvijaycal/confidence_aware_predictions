import os
import sys
import numpy as np
import heapq
from scipy.stats import multivariate_normal
import pytest
import cv2

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)

from evaluation.gmm_prediction import GMMPrediction

'''
Test Fixture Parametrized by Dataset.
'''
def get_straight_traj(vel, num_timesteps, dt):
	xs = vel * dt * np.arange(1, 1+num_timesteps)
	ys = np.zeros(len(xs))
	return np.column_stack((xs, ys))

def get_const_curv_traj(vel, ang_vel, num_timesteps, dt):
	assert np.abs(ang_vel) > 0.
	xs = vel / ang_vel * np.sin(ang_vel * dt * np.arange(1, 1 + num_timesteps))
	ys = vel / ang_vel * (1 - np.cos(ang_vel * dt * np.arange(1, 1 + num_timesteps)))
	return np.column_stack((xs, ys))

def generate_perturbation_traj(n_steps, max_noise_radius):
	# Returns a trajectory with radius-bounded noise about the origin.
	noise_angles = np.random.rand(n_steps) * ( 2 * np.pi )
	noise_radii  = np.random.rand(n_steps) * max_noise_radius
	noise_vectors = np.array([rad * np.array([np.cos(ang), np.sin(ang)])
		                      for ang, rad in zip(noise_angles, noise_radii)])
	return noise_vectors

def make_test_predictions(N, dt):
	# 3 mode GMM with left, straight, and right turns.
	vel     = 10 # m/s
	ang_vel = 0.2 # rad/s

	mode_probs = np.array([0.25, 0.5, 0.25])
	traj_l = get_const_curv_traj(vel, ang_vel, N, dt)
	traj_c = get_straight_traj(vel, N, dt)
	traj_r = get_const_curv_traj(vel, -ang_vel, N, dt)
	num_modes = len(mode_probs)

	mus    = np.array([traj_l, traj_c, traj_r])
	sigmas = np.tile( np.diag([2., 5.]), (num_modes, N, 1, 1) )

	return GMMPrediction(num_modes, N, mode_probs, mus, sigmas)

def get_area_upper_bound(sigmas, beta):
	# Integrates the ellipsoid area over all sigmas.
	# This is an upper bound because it doesn't consider
	# area overlaps and will overcount such regions.
	area = 0.
	for mode in range(sigmas.shape[0]):
		for covar in sigmas[mode]:
			evals, evecs = np.linalg.eig( covar ) # rotation/unitary matrices and evecs = diag(sigma_1**2, sigma_2**2)

			length_ax1 = np.sqrt(beta * evals[0]) # semi-major axis length, sigma_1 * sqrt(beta)
			length_ax2 = np.sqrt(beta * evals[1]) # semi-minor axis length, sigma_2 * sqrt(beta)

			area += (np.pi * length_ax1 * length_ax2)

	return area

@pytest.fixture(scope="module", params=['nuscenes', 'l5kit'])
def gmm_preds(request):
	# Identify prediction horizon (N) and discretization time (dt).
	# Build the GMM example given in make_test_predictions.
	if request.param == 'nuscenes':
		N = 12; dt = 0.5
	elif request.param == 'l5kit':
		N = 25; dt = 0.2
	else:
		raise NotImplementedError

	return make_test_predictions(N, dt)

'''
Test Suite
'''
def test_get_top_k_GMM(gmm_preds):
	# Check that the correct modes get selected for the truncated GMM.
	mode_probs = gmm_preds.mode_probabilities
	for k in range( 1, max(1, gmm_preds.n_modes) ):
		truncated_gmm = gmm_preds.get_top_k_GMM(k)

		# Check truncated probablity selection with heapq + renormalization match.
		# We need to reverse heapq to maintain consistent ordering with argsort.
		expected_mode_probs = np.array( heapq.nlargest(k, mode_probs)[::-1] )
		expected_mode_probs = expected_mode_probs / np.sum(expected_mode_probs)
		assert np.allclose(expected_mode_probs, truncated_gmm.mode_probabilities)

		# Check that the mean/covariance are the ones we expected.
		inds_k_largest = np.argsort(mode_probs)[-k:]
		for trunc_inc, ind in enumerate(inds_k_largest):
			mu_expected    = gmm_preds.mus[ind, :]
			sigma_expected = gmm_preds.sigmas[ind, :]

			assert np.allclose(mu_expected, truncated_gmm.mus[trunc_inc])
			assert np.allclose(sigma_expected, truncated_gmm.sigmas[trunc_inc])

def test_compute_trajectory_log_likelihood(gmm_preds):
	# Check if naive calculation matches logsumexp (for numerically simple case).
	for test_traj_idx in range(gmm_preds.n_modes):
		traj = np.copy(gmm_preds.mus[test_traj_idx])
		traj += generate_perturbation_traj(traj.shape[0], 5.)

		naive_prob_calc = 0.
		for mode_idx in range(gmm_preds.n_modes):
			gaussian_prod = 1.
			for t in range(gmm_preds.n_timesteps):
				prob = multivariate_normal.pdf( traj[t],
					                            mean=gmm_preds.mus[mode_idx][t],
					                            cov=gmm_preds.sigmas[mode_idx][t])
				gaussian_prod *= prob
			naive_prob_calc += gmm_preds.mode_probabilities[mode_idx] * gaussian_prod
		naive_ll = np.log( naive_prob_calc )

		lse_ll   = gmm_preds.compute_trajectory_log_likelihood(traj)

		assert np.isclose(naive_ll, lse_ll)

def test_compute_min_ADE(gmm_preds):
	# Add noise within a norm ball of a true trajectory.
	# The minADE should be upper-bounded by the max radius.
	for max_noise_radius in np.arange(0., 21., 5.):
		for selected_traj in range(gmm_preds.n_modes):
			test_traj = np.copy(gmm_preds.mus[selected_traj])
			test_traj += generate_perturbation_traj(test_traj.shape[0],
				                                    max_noise_radius)

			ade = gmm_preds.compute_min_ADE(test_traj)

			assert ade >= 0
			assert ade <= max_noise_radius

def test_compute_min_FDE(gmm_preds):
	# Similar to ADE, the max radius should upper bound the FDE here.
	for max_noise_radius in np.arange(0., 21., 5.):
		for selected_traj in range(gmm_preds.n_modes):
			test_traj  = np.copy(gmm_preds.mus[selected_traj])
			test_traj += generate_perturbation_traj(test_traj.shape[0],
				                                    max_noise_radius)
			fde = gmm_preds.compute_min_FDE(test_traj)
			assert fde >= 0
			assert fde <= max_noise_radius

	# FDE should be zero if the last state is left unchanged.
	for selected_traj in range(gmm_preds.n_modes):
		test_traj = np.copy(gmm_preds.mus[selected_traj])
		test_traj[:-1] += generate_perturbation_traj(test_traj.shape[0] - 1,
			                                         5.)
		assert np.isclose(gmm_preds.compute_min_FDE(test_traj), 0.)

	# FDE should be impacted even if all prior states are noise-free.
	for selected_traj in range(gmm_preds.n_modes):
		test_traj = np.copy(gmm_preds.mus[selected_traj])
		test_traj[-1] += np.ravel(generate_perturbation_traj(1, 5.))
		fde = gmm_preds.compute_min_FDE(test_traj)
		assert fde >= 0
		assert fde <= 5.

def test_compute_set_accuracy(gmm_preds):
	# This test checks if a perturbed trajectory lies within
	# a given Mahalanobis distance of the mean trajectory.

	debug = False # for plotting the ellipse sets and trajectories

	for max_noise_radius in np.array([1., 10., 50.]):
		mdist_sq_upper_bound = max_noise_radius**2
		beta_inc = mdist_sq_upper_bound / 5
		betas = np.arange(1., mdist_sq_upper_bound + beta_inc, beta_inc)

		for selected_traj in range(gmm_preds.n_modes):
			test_traj  = np.copy(gmm_preds.mus[selected_traj])
			test_traj += generate_perturbation_traj(test_traj.shape[0],
				                                    max_noise_radius)

			accs = [gmm_preds.compute_set_accuracy(test_traj, beta, debug=debug) for beta in betas]

			# accs should be 0 or 1 and should be in ascending order
			# i.e. enlarging the set can only improve accuracy
			assert np.all( np.arange(len(betas)) == np.argsort(accs) )
			assert np.min(accs) >= 0
			assert np.max(accs) <= 1

def test_compute_set_area(gmm_preds):
	# This test checks that set area results are reasonable:
	# (1) Area should be positive.
	# (2) Increasing beta (conf. threshold) should increase area.
	# (3) It should be upper bounded by the areas of all ellipsoids added together
	#     up to some discretization error of OpenCV.
	betas     = np.linspace(1., 50., 4)
	areas     = [] # area computed via OpenCV numerical integration
	areas_ub  = [] # upper bound area computed as a sum of ellipse areas
	occ_grids = [] # the "occupancy grids" made with OpenCV to find areas above

	debug = False # for plotting the occupancy grids

	for beta in betas:
		area, occ_grid = gmm_preds.compute_set_area(beta)
		areas.append(area)
		occ_grids.append(occ_grid)

		area_ub = get_area_upper_bound(gmm_preds.sigmas, beta)
		areas_ub.append(area_ub)

	areas       = np.array(areas)
	areas_ub    = np.array(areas_ub)
	areas_ratio = areas / areas_ub

	if debug:
		n_rows, n_cols = occ_grids[0].shape
		n_rows_ds = int(n_rows / 10)
		n_cols_ds = int(n_cols / 10)

		combined_img = np.zeros( (n_rows_ds * len(occ_grids), n_cols_ds), dtype=np.uint8)

		for ind, grid in enumerate(occ_grids):
			combined_img[ind*n_rows_ds:(ind+1)*n_rows_ds, :] = cv2.resize(grid, (n_cols_ds, n_rows_ds))

		del occ_grids

		print(f"Betas: {betas}")
		print(f"Areas: {areas}")
		cv2.imshow("occ_grid", combined_img); cv2.waitKey(0)

	assert np.all(areas > 0)
	assert np.all( np.arange(len(areas)) == np.argsort(areas) )
	assert np.max(areas_ratio) < 1.05
