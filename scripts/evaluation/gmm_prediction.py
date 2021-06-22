import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
from colorsys import hsv_to_rgb
import cv2

class GMMPrediction:
	""" This class allows for metric evaluation of GMM-based predictions. """

	def __init__(self, n_modes, n_timesteps, mode_probabilities, mus, sigmas):
		self.n_modes = n_modes
		self.n_timesteps = n_timesteps

		assert mode_probabilities.shape == (self.n_modes,)
		assert np.isclose(np.sum(mode_probabilities), 1.) and np.all(mode_probabilities >= 0.)

		self.mode_probabilities = mode_probabilities

		assert mus.shape == (self.n_modes, self.n_timesteps, 2)
		self.mus = mus

		assert sigmas.shape == (self.n_modes, self.n_timesteps, 2, 2)
		self.sigmas = sigmas

	def get_top_k_mode_labels(self, k=1):
		return np.argsort(self.mode_probabilities)[-k:]

	def get_top_k_GMM(self, k):
		assert k > 0 and k < self.n_modes

		# Find the top-k most likely modes.
		top_k_inds = self.get_top_k_mode_labels(k=k)

		# Normalize the probabilities based on the subset of modes.
		mode_probs_k = self.mode_probabilities[top_k_inds]
		mode_probs_k = mode_probs_k / np.sum(mode_probs_k)

		mus_k    = self.mus[top_k_inds, :, :]

		sigmas_k = self.sigmas[top_k_inds, :, :, :]

		return GMMPrediction(k, self.n_timesteps, mode_probs_k, mus_k, sigmas_k)

	def get_mode_ADEs(self, traj_xy):
		# Returns the average displacement error (ADE) across all modes in the GMM.
		ades = []

		for mode in range(self.n_modes):
			traj_xy_pred = self.mus[mode]
			displacements = np.linalg.norm(traj_xy_pred - traj_xy, axis=-1)
			ades.append(np.mean(displacements))

		return ades

	##################################################################
	#################### LIKELIHOOD METRIC ###########################
	def compute_trajectory_log_likelihood(self, traj_xy):
		# LL = log {sum_{j=1:num_modes} P(mode j) * product_{t=1:T} N[traj_xy(t); mu_j(t), sigma_j(t)] }
		# Some clever manipulation of the inner log argument to allow logsumexp to be used, which
		# is more numerically stable since the multiplication of the Gaussians becomes small very fast.

		assert traj_xy.shape == (self.n_timesteps, 2)

		# First check if the covariances are fine to use.
		# They are expected to be positive semi-definite.
		with np.errstate(all='raise'):
			for mode in range(self.n_modes):
				for covar in self.sigmas[mode]:
					try:
						np.log(np.linalg.det(covar))
					except Exception as e:
						print(f"Unable to compute log-det!: {e}")
						import pdb; pdb.set_trace()

		# Construct the array argument to the logsumexp function.
		# Each element corresponds to the log probability component
		# contributed to the final result by that specific mode.
		exp_arr = []
		const_term = self.n_timesteps*np.log(2*np.pi)
		for mode in range(self.n_modes):
			weight = self.mode_probabilities[mode]
			residual_traj = traj_xy - self.mus[mode]

			exp_term  = np.log(weight) - const_term
			exp_term -= 0.5 * np.sum( [np.log(np.linalg.det(covar)) for covar in self.sigmas[mode]] )
			exp_term -= 0.5 * np.sum( [residual_traj[tm_step].T @ \
				                       np.linalg.pinv(self.sigmas[mode][tm_step]) @ \
				                       residual_traj[tm_step] \
				                       for tm_step in range(self.n_timesteps)])
			exp_arr.append(exp_term)

		return logsumexp(exp_arr)

	##################################################################
	################## CLASSIFICATION METRICS ########################
	def get_class_top_k_scores(self, traj_xy, ks = [1, 3, 5]):
		scores = []
		mode_label = np.argmin(self.get_mode_ADEs(traj_xy))

		for k in ks:
			if k > self.n_modes:
				scores.append(1)
			else:
				acc_top_k = (mode_label in self.get_top_k_mode_labels(k=k))
				scores.append(1 if acc_top_k else 0)

		return scores

	##################################################################
	#################### TRAJECTORY METRICS ##########################
	def compute_min_ADE(self, traj_xy):
		# min_ADE = minimum average displacement error among all modes in the GMM.
		return np.amin(self.get_mode_ADEs(traj_xy))

	def compute_min_FDE(self, traj_xy):
		# min FDE = minimum final displacement error among all modes in the GMM.
		fdes = []

		for mode in range(self.n_modes):
			traj_xy_pred = self.mus[mode]
			displacement_final = np.linalg.norm(traj_xy_pred[-1, :] - traj_xy[-1, :])
			fdes.append(displacement_final)

		return np.amin(fdes)

	##################################################################
	####################### SET METRICS ##############################
	def compute_set_accuracy(self, traj_xy, beta, debug=False):
		# Determines if traj_xy is contained in the set prediction
		# formed by finding the union of beta-sublevel sets from the GMM.

		smallest_mdist_sq_per_timestep = np.ones(self.n_timesteps, dtype=np.float64) * np.finfo(dtype=np.float64).max

		if debug:
			# plot the sublevel sets of each mode
			ax = plt.gca()
			plt.plot(0, 0, 'kx')

		for mode in range(self.n_modes):
			for tm_step, (mean, covar) in enumerate(zip(self.mus[mode], self.sigmas[mode])):
				residual = traj_xy[tm_step] - mean
				mdist_sq = residual.T @ np.linalg.pinv(covar) @ residual

				smallest_mdist_sq_per_timestep[tm_step] = min(smallest_mdist_sq_per_timestep[tm_step], mdist_sq)

				if debug:
					hue = mode / self.n_modes
					rgb = hsv_to_rgb(hue, 1., 1.)

					evals, evecs = np.linalg.eig( covar ) # rotation/unitary matrices and evecs = diag(sigma_1**2, sigma_2**2)

					length_ax1 = np.sqrt(beta * evals[0]) # semi-major axis length, sigma_1 * sqrt(beta)
					length_ax2 = np.sqrt(beta * evals[1]) # semi-minor axis length, sigma_2 * sqrt(beta)
					ang = np.degrees( np.arctan2(evecs[1,0], evecs[0,0]) ) # ccw

					alpha = np.clip( 0.8 * (tm_step + 1) / self.n_timesteps , 0., 1.)

					# Matplotlib wants diameters, not semi-axes lengths like OpenCV:
					el = Ellipse(tuple(mean), 2.*length_ax1, 2.*length_ax2, ang, color=rgb, alpha=alpha)
					ax.add_patch(el)

		maxmin_mdist_sq = np.max(smallest_mdist_sq_per_timestep)
		containment = maxmin_mdist_sq <= beta

		if debug:
			# plot this trajectory
			colors = cm.rainbow(np.linspace(0, 1, self.n_timesteps))
			for tm_step in range(self.n_timesteps):
				if smallest_mdist_sq_per_timestep[tm_step] <= beta:
					plt.scatter(traj_xy[tm_step,0], traj_xy[tm_step,1], marker='x', color=colors[tm_step])
				else:
					plt.scatter(traj_xy[tm_step,0], traj_xy[tm_step,1], marker='o', s=100, color='k')

			# plot the containment result
			plt.title(f"Contained: {containment}, beta: {beta}, maxmin: {np.round(maxmin_mdist_sq, 3)}")
			plt.axis("equal")
			plt.show()

		return containment

	def compute_set_area(self, beta):
		# Numerically computes the area of the set prediction
		# formed by the union of beta-sublevel sets from the GMM.
		side_length = 200 # meters
		resolution  = 0.1 # meters
		n_pts       = int( side_length / resolution )

		occ_grid = np.zeros((n_pts, n_pts), dtype=np.uint8)
		color = (255)
		thickness = -1

		def local_coords_to_pixels(X_local, Y_local):
			# Given points in the vehicle local coordinate system,
			# find the corresponding pixels.
			pix_x = int(n_pts/2) + int(X_local / resolution) # X_local points right
			pix_y = int(n_pts/2) - int(Y_local / resolution) # Y_local points upward.
			return pix_x, pix_y

		for mode in range(self.n_modes):
			for tm_step in range(self.n_timesteps):
				center_px, center_py = local_coords_to_pixels(*self.mus[mode][tm_step])

				evals, evecs = np.linalg.eig( self.sigmas[mode][tm_step] )

				length_ax1 = int( np.sqrt(beta * evals[0]) / resolution )
				length_ax2 = int( np.sqrt(beta * evals[1]) / resolution )

				ang_1 = -np.degrees( np.arctan2(evecs[1,0], evecs[0,0]) ) # negative sign since cv2.ellipse needs cw angle.

				cv2.ellipse( occ_grid, (center_px, center_py), (length_ax1, length_ax2), ang_1, 0., 360., color, thickness=thickness  )

		return (np.count_nonzero(occ_grid) * resolution * resolution), occ_grid
