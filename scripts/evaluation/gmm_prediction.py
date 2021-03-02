import numpy as np
from scipy.special import logsumexp

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

	def get_top_k_anchor_predictions(self, k=1):
		return np.argsort(self.mode_probabilities)[-k:]

	def get_top_k_GMM(self, k):
		assert k > 0 and k < self.n_modes

		# Find the top-k most likely modes. 
		top_k_inds = self.get_top_k_anchor_predictions(k=k)

		# Normalize the probabilities based on the subset of modes.
		mode_probs_k = self.mode_probabilities[top_k_inds]
		mode_probs_k = mode_probs_k / np.sum(mode_probs_k)

		mus_k    = self.mus[top_k_inds, :, :]

		sigmas_k = self.sigmas[top_k_inds, :, :, :]

		return GMMPrediction(k, self.n_timesteps, mode_probs_k, mus_k, sigmas_k)

	# TODO: Maybe useful to add a get_mode_label for IMM.

	@staticmethod
	def get_anchor_label(traj_xy, anchors):
		# IMPORTANT: We assume that anchors are ordered and correspond
		# 1-1 with self.mus!  So the label/prediction are consistent.
		# This can be used to generate a confusion matrix or evaluate
		# mode classification accuracies.
		dists_to_anchor = [np.sum(np.linalg.norm(traj_xy - anc, axis=-1), axis=-1) for anc in anchors]
		anchor_label = np.argmin(dists_to_anchor)

		return anchor_label

	##################################################################
	#################### LIKELIHOOD METRIC ###########################
	def compute_trajectory_log_likelihood(self, traj_xy):
		# LL = log {sum_{k=1:num_modes} P(mode k) * product_{t=1:T} N[traj_xy(t); mus(t), sigmas(t)] }
		# Some clever manipulation of the inner log argument to allow logsumexp to be used, which
		# is more numerical stable (avoids -np.inf).

		assert traj_xy.shape == (self.n_timesteps, 2)

		exp_arr = []
		for mode in range(self.n_modes):
			weight = self.mode_probabilities[mode]
			residual_traj = traj_xy - self.mus[mode]

			# # debugging:
			# # triggered by negative det in CVH model, worth checking that out.
			# with np.errstate(all='raise'):
			# 	for covar in self.sigmas[mode]:
			# 		try:
			# 			np.log(np.linalg.det(covar))
			# 		except Exception as e:
			# 			print(e)
			# 			import pdb; pdb.set_trace()

			exp_term  = np.log(weight) - self.n_timesteps*np.log(2*np.pi)
			exp_term += -0.5 * np.sum( [np.log(np.linalg.det(covar)) for covar in self.sigmas[mode]] )
			exp_term += -0.5 * np.sum( [residual_traj[tm_step].T @ \
				                        np.linalg.inv(self.sigmas[mode][tm_step]) @ \
				                        residual_traj[tm_step] \
				                        for tm_step in range(self.n_timesteps)])
			exp_arr.append(exp_term)

		return logsumexp(exp_arr)

	##################################################################
	################## CLASSIFICATION METRICS ########################
	def get_class_top_k_scores(self, traj_xy, anchors, ks = [1, 3, 5]):
		scores = []
		anchor_label = self.get_anchor_label(traj_xy, anchors)

		for k in ks:
			acc_top_k = (anchor_label in self.get_top_k_anchor_predictions(k=k))
			scores.append(1 if acc_top_k else 0)
		
		return scores
	
	# TODO: confusion matrix handled externally using get_anchor_label and get_top_k_anchor_predictions.

	##################################################################
	#################### TRAJECTORY METRICS ##########################
	def compute_min_ADE(self, traj_xy):
		# min_ADE = minimum average displacement error among all modes in the GMM.
		ades = []

		for mode in range(self.n_modes):
			traj_xy_pred = self.mus[mode]
			displacements = np.linalg.norm(traj_xy_pred - traj_xy, axis=-1)
			ades.append(np.mean(displacements))

		return np.amin(ades)

	def compute_min_FDE(self, traj_xy):
		# min FDE = minimum final displacement error among all modes in the GMM.
		fdes = []

		for mode in range(self.n_modes):
			traj_xy_pred = self.mus[mode]
			displacement_final = np.linalg.norm(traj_xy_pred[-1, :] - traj_xy[-1, :])
			fdes.append(displacement_final)
		
		return np.amin(fdes)

	def compute_minmax_d(self, traj_xy):
		# Finds the minimum of the maximum displacements of a trajectory with 
		# respect to a mode mean trajectory.  This is the minimum radius required
		# such that the observed trajectory falls within the union of tubes with this 
		# radius centered about each mode mean trajectory.

		traj_max_displacements = []

		for mode in range(self.n_modes):
			traj_xy_pred = self.mus[mode]
			displacements = np.linalg.norm(traj_xy_pred - traj_xy, axis=-1)
			traj_max_displacements.append(np.amax(displacements))

		return np.amin(traj_max_displacements)
