import numpy as np

class ConfidenceLevelManager:

	def __init__(self,
		         n_gmm_modes,
		         n_steps_sliding_window,
		         conf_level_init = 4.61,
		         conf_level_min = 3.22,
		         alpha=0.1):
		# For a 2-degree-of-freedom chi-square distribution,
		# here are critical values:
		# 0.8  -> 3.22
		# 0.9  -> 4.61
		# 0.95 -> 5.99

		self.n_modes        = n_gmm_modes
		self.conf_level_min = conf_level_min
		self.alpha          = alpha

		self.mode_conf_levels = np.ones(self.n_modes) * conf_level_init

	def update(self, traj, gmm_pred):
		# TODO: deal with the sliding window stuff.
		# Find the closest mode in the GMM to the traj.
		# sigma level for the closest mode
		# lpf for all modes

		"""
		Pseudocode

		compute d_M per mode for traj
		let act_idx = argmin_{1...n_modes} d_M(mode j, traj)
		let sigma* = min_{sigma > 0} sigma s.t. traj contained in ellipsoidal chance constraints for all timesteps

		for all modes j:
			if(j == act_idx):
				u_beta = max(conf_level_min, sigma*)
			else:
				u_beta = conf_level_min

		    beta_j = alpha * beta_j + (1 - alpha) * u_beta
		"""

	def get_confidence_level(self, mode_idx):
		return self.mode_conf_levels[mode_idx]



	"""
	RANDOM THOUGHTS:
	Another approach is to have a single beta, chosen as the maxmin Mahalanobis distance across all relevant timesteps.
	This is basically the smallest value needed to have the observed trajectory be contained by any of the relevant modes.
	"""