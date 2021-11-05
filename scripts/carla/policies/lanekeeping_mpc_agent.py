import os
import sys
import numpy as np
import time
from collections import deque

scriptdir = os.path.abspath(__file__).split('carla')[0] + 'carla/'
sys.path.append(scriptdir)
from policies.dynamic_agent import DynamicAgent
from utils import frenet_trajectory_handler as fth
from utils.low_level_control import LowLevelControl
from policies.lanekeeping_mpc import LanekeepingMPC
from policies.confidence_thresh_manager import ConfidenceThreshManager
from policies.lane_interval_manager import LaneIntervalManager

class LanekeepingMPCAgent(DynamicAgent):
    """ A lanekeeping agent using MPC and implementing optional stopline constraints. """

    def __init__(self,
                 vehicle,
                 goal_location,
                 conf_params_dict,        # Constructor params for confidence level manager.
                 N = 10,                  # MPC horizon
                 DT = 0.2,                # s, discretization time for MPC
                 dt = 0.05,               # s, control timestep
                 nominal_speed_mps = 8.0, # sets desired speed (m/s) to track
                 lat_accel_max = 2.0):    # sets the maximum lateral acceleration (m/s^2)

        super().__init__(vehicle=vehicle,
                         goal_location=goal_location,
                         dt=dt)

        self.confidence_thresh_manager   = ConfidenceThreshManager(**conf_params_dict)
        self.lane_interval_manager       = LaneIntervalManager(self._frenet_traj, DT, self.A_MIN, self.A_MAX)

        # Underlying class used to solve/update MPC problem for lanekeeping.
        self.lk_mpc = LanekeepingMPC( N       = N,
                                      DT      = DT,
                                      DT_CTRL = dt,
                                      L_F     = self.lf,
                                      L_R     = self.lr,
                                      A_MIN   = self.A_MIN,
                                      A_MAX   = self.A_MAX,
                                      V_MIN   = self.V_MIN,
                                      V_MAX   = self.V_MAX,
                                      DF_MIN  = self.DF_MIN,
                                      DF_MAX  = self.DF_MAX)

        # Speed/curvature profile params and generation.
        self.nominal_speed = nominal_speed_mps # m/s
        self.lat_accel_max = lat_accel_max     # m/s^2
        self.preview = 20                      # number of measurements ahead to compute v_des and local linear fit of curvature
        self._generate_speed_profile()

        # Tuning of low level control with "good" parameters and low actuation delay.
        self.acc_prev = 0.
        self.df_prev  = 0.
        self._low_level_control = LowLevelControl(vehicle,
                                                  dt_control     = self.DT,
                                                  tau_delay_long = self.DT/5.,
                                                  tau_delay_lat  = self.DT/5.,
                                                  k_v=0.5,
                                                  k_i=0.01)

        # Use of elapsed time to decide when to update confidence_thresh_manager.
        self.elapsed_time = 0.

    def run_step(self, pred_dict):
        state_dict = self.get_current_state()

        # Initialize variables to be returned.
        z0=np.array([state_dict["x"],
                     state_dict["y"],
                     state_dict["psi"],
                     state_dict["speed"]])
        u0=np.array([self.A_MIN, 0.])
        is_opt=False
        solve_time=np.nan

        self.update_completion(state_dict["s"])

        if self.done():
            v_des = 0. # we should remain stopped until the end of the simulation.
        else:
            # We use preview to slow down early before upcoming turns and approximate local curvature.
            buffer_idx_st  = state_dict["ft_idx"]
            buffer_idx_end = min(len(self.speed_profile) - 1, buffer_idx_st + self.preview)
            v_des          = np.amin(self.speed_profile[buffer_idx_st:buffer_idx_end]) # desired speed with preview

            curv_lin_fit   = self._fit_local_curvature(buffer_idx_st, buffer_idx_end)

            # Update obstacle avoidance (i.e. interval) constraints at prediction time discretization.
            if pred_dict["tvs_valid_pred"][0] and np.isclose(self.elapsed_time % self.lk_mpc.DT, 0.):
              self.confidence_thresh_manager.update(pred_dict)

            # Get MPC initial conditions and parameters.
            if pred_dict["tvs_valid_pred"][0]:
              s0_ev   = state_dict["s"]
              v0_ev   = state_dict["speed"]
              x0_tv   = pred_dict["tvs_poses"][0][0]
              y0_tv   = pred_dict["tvs_poses"][0][1]
              mus     = pred_dict["tvs_mode_dists"][0][0]
              sigmas  = pred_dict["tvs_mode_dists"][1][0]
              tv_dims = pred_dict["tvs_dimensions"][0]
              conf_threshs = self.confidence_thresh_manager.conf_thresh * np.ones(len(mus))
              s_lb, s_ub   = self.lane_interval_manager.generate_interval_constraints(s0_ev, v0_ev, v_des,
                                                                                      x0_tv, y0_tv,
                                                                                      conf_threshs,
                                                                                      mus, sigmas, tv_dims)
            else:
              s_lb = np.zeros(self.lk_mpc.N)
              s_ub = np.ones((self.lk_mpc.N)) * self._frenet_traj.trajectory[-1, 0]

            # Update MPC initial conditions and parameters.
            update_dict = {}
            update_dict["s"]            = state_dict["s"]
            update_dict["ey"]           = state_dict["ey"]
            update_dict["epsi"]         = state_dict["epsi"]
            update_dict["v"]            = state_dict["speed"]
            update_dict["curv_lin_fit"] = curv_lin_fit
            update_dict["v_ref"]        = v_des
            update_dict["acc_prev"]     = self.acc_prev
            update_dict["df_prev"]      = self.df_prev
            update_dict["s_lb"]         = s_lb
            update_dict["s_ub"]         = s_ub
            self.lk_mpc.update(update_dict)

            # Solve the MPC problem.
            sol_dict = self.lk_mpc.solve()

            # Extract solution and log results.
            # NOTE: we don't worry about infeasibility due to slack
            # variables being active, just focusing on the resultant motion.
            u0         = sol_dict["u_control"]
            is_opt     = sol_dict["optimal"]
            solve_time = sol_dict["solve_time"]
            v_des      = sol_dict["z_mpc"][1, 3] # v one step ahead
            self.acc_prev = u0[0]
            self.df_prev  = u0[1]

        # Get low level control -> key things are v_des and df_des for setpoints.
        control =  self._low_level_control.update(state_dict["speed"], # v_curr
                                                  u0[0],               # a_des
                                                  v_des,               # v_mpc
                                                  u0[1])               # df_des

        self.elapsed_time = self.elapsed_time + self.lk_mpc.DT_CTRL
        return control, z0, u0, is_opt, solve_time

    ################################################################################################
    ########################## Helper / Update Functions ###########################################
    ################################################################################################
    def _generate_speed_profile(self):
        curv_profile  = self._frenet_traj.trajectory[:, 4]
        self.speed_profile = np.minimum( self.nominal_speed,
                                         np.sqrt( self.lat_accel_max / np.maximum(0.01, np.abs(curv_profile)) )
                                       )

    def _fit_local_curvature(self, start_idx, end_idx):
        s_fit    = self._frenet_traj.trajectory[start_idx:end_idx, 0]
        curv_fit = self._frenet_traj.trajectory[start_idx:end_idx, 4]

        A = np.column_stack((s_fit, np.ones(len(s_fit))))
        y = curv_fit

        if(len(s_fit) == self.preview):
          return np.linalg.lstsq(A, y, rcond=None)[0]
        else:
          return np.array([0., 0.])

