import numpy as np
from collections import deque

class ConfidenceThreshManager:

    def __init__(self,
                 n_buffer_size,
                 is_adaptive,
                 conf_thresh_init,
                 conf_thresh_min  = 0.211,
                 alpha=0.9):

        # For a 2-degree-of-freedom chi-square distribution,
        # here are critical values:
        # 0.10 -> 0.211
        # 0.20 -> 0.446
        # 0.40 -> 1.00
        # 0.63 -> 2.00
        # 0.80 -> 3.22
        # 0.90 -> 4.61
        # 0.95 -> 5.99

        # Assumption is that the user properly chooses when to update.
        # It should be at the time discretization of the prediction/MPC model.
        # The pred_buffer is used to maintain a sliding window to
        # retrospectively see how well we predicted the partial observed
        # trajectory since that timestep n_buffer_size updates ago.

        self.pred_buffer     = deque(maxlen=n_buffer_size)
        self.is_adaptive     = is_adaptive
        self.conf_thresh_min = conf_thresh_min
        self.alpha           = alpha

        self.conf_thresh     = conf_thresh_init

    def update(self, pred_dict):
        if not self.is_adaptive:
            # Confidence threshold fixed for all time.
            return

        if len(self.pred_buffer) < self.pred_buffer.maxlen:
            # Not enough time has passed for a new confidence update.
            pass
        else:

            pred_dict_sw = self.pred_buffer[0]
            mus          = pred_dict_sw["tvs_mode_dists"][0][0] # Prediction made before.
            sigmas       = pred_dict_sw["tvs_mode_dists"][1][0]
            tv_traj_sw   = pred_dict["tvs_traj_hists"][0]       # History right now.

            n_modes, n_timesteps, n_state = mus.shape
            assert n_state == 2
            assert sigmas.shape == (n_modes, n_timesteps, 2, 2)

            # tv_traj_sw should iterate over states at the following timesteps:
            # t - buff_len + 1, ..., t, where t is the current timestep.
            assert tv_traj_sw.shape == (self.pred_buffer.maxlen, 2)

            # Iterate over modes and find lowest conf thresh required to capture tv_traj over the buffer window.
            smallest_mdist_sq_per_timestep = np.ones(self.pred_buffer.maxlen, dtype=np.float64) * np.finfo(dtype=np.float64).max

            for mode in range(n_modes):
                for tm_step in range(self.pred_buffer.maxlen):
                    curr_tv_xy = tv_traj_sw[tm_step, :]
                    mean       = mus[mode][tm_step]
                    covar      = sigmas[mode][tm_step]

                    residual = curr_tv_xy - mean
                    mdist_sq = residual.T @ np.linalg.pinv(covar) @ residual

                    smallest_mdist_sq_per_timestep[tm_step] = min(smallest_mdist_sq_per_timestep[tm_step], mdist_sq)

            maxmin_mdist_sq = np.max(smallest_mdist_sq_per_timestep)

            # Low pass filter update for conf thresh.
            self.conf_thresh = self.alpha * max(self.conf_thresh_min, maxmin_mdist_sq) + (1. - self.alpha) * self.conf_thresh

        self.pred_buffer.append(pred_dict)
