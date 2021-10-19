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

class LanekeepingPIAgent(DynamicAgent):
    """ A lanekeeping agent with rational/irrational speed + lane tracking. """

    def __init__(self,
                 vehicle,
                 goal_location,
                 dt = 0.05,               # s, control timestep
                 is_rational = True,      # whether to use a rational/irrational driving policy
                 nominal_speed_mps = 8.0, # sets desired speed (m/s) to track
                 lat_accel_max = 2.0):    # sets the maximum lateral acceleration (m/s^2)

        super().__init__(vehicle=vehicle,
                         goal_location=goal_location,
                         dt=dt)

        self.is_rational = is_rational
        self.nominal_speed = nominal_speed_mps # m/s
        self.lat_accel_max = lat_accel_max     # m/s^2

        self._generate_speed_profile()

        if self.is_rational:
            self.k_ey          = 0.0538    # ey proportional gain, rad/m
            self.x_LA          = 14.2      # lookahead distance, m
            self.curv_delay    = 0         # number of timesteps delay in curvature measurement / speed setpoint
            self.speed_preview = 20        # number of discretized lane measurements to consider for slowing down

            # Tuning of low level control with "good" parameters and low actuation delay.
            self._low_level_control = LowLevelControl(vehicle,
                                                      dt_control     = self.DT,
                                                      tau_delay_long = self.DT/5.,
                                                      tau_delay_lat  = self.DT/5.,
                                                      k_v=0.5,
                                                      k_i=0.01)
        else:
            self.k_ey = 0.5                # ey proportional gain, rad/m
            self.x_LA = 5.0                # lookahead distance, m
            self.curv_delay = 2            # number of timesteps delay in curvature measurement / speed setpoint
            self.ey_noise_magnitude = 1.5  # amplitude of lateral error sinusoidal noise
            self.v_noise_magnitude  = 2.0  # amplitude of speed setpoint sinusoidal noise
            self.n_calls = 0.              # current timestep % period
            self.n_calls_period = 80       # period for sinusoidal noise

            # Tuning of low level control with "bad" parameters and high actuation delay.
            self._low_level_control = LowLevelControl(vehicle,
                                                      dt_control     = self.DT,
                                                      tau_delay_long = self.DT,
                                                      tau_delay_lat  = self.DT,
                                                      k_v=0.9,
                                                      k_i=0.0)

        # Buffer of speed + curvature measurements to impose delayed feedback.
        self.speed_buffer     = deque(maxlen=(1+self.curv_delay))
        self.curvature_buffer = deque(maxlen=(1+self.curv_delay))

    def run_step(self, pred_dict):
        state_dict = self.get_current_state()

        self.curvature_buffer.append(state_dict["curv"])
        self.speed_buffer.append(self.speed_profile[state_dict["ft_idx"]])

        # Get the delayed speed setpoint and curvature measurement.
        v_des     = self.speed_buffer[0]
        curv_des  = self.curvature_buffer[0]

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
            if self.is_rational:
                # We have a speed preview to slow down early before upcoming turns.
                speed_buffer_idx_st  = state_dict["ft_idx"]
                speed_buffer_idx_end = min(len(self.speed_profile) - 1, speed_buffer_idx_st + self.speed_preview)
                v_des                = np.amin(self.speed_profile[speed_buffer_idx_st:speed_buffer_idx_end])
            else:
                # We have a sinusoidal noise component added to lateral error / velocity in order
                # to bring about swerving + poor speed following behaviors.
                state_dict["ey"] += self.ey_noise_magnitude * np.cos( 2 * np.pi * self.n_calls / self.n_calls_period)
                v_des            += self.v_noise_magnitude  * np.cos( 2 * np.pi * self.n_calls / self.n_calls_period)
                self.n_calls     += 1
                self.n_calls     %= self.n_calls_period

            st = time.time()

            # Compute acceleration based on a simplified IDM (this is mostly used to determine when to brake).
            u0[0]  = self._compute_desired_acceleration(state_dict["speed"], v_des)

            # Compute desired steer angle based on a FF/FB policy.
            u0[1]  = self._compute_desired_steer_angle(curv_des, state_dict["ey"], state_dict["epsi"])

            solve_time = time.time() - st
            is_opt = self.is_rational

        # Get low level control -> key things are v_des and df_des for setpoints.
        control =  self._low_level_control.update(state_dict["speed"], # v_curr
                                                  u0[0],               # a_des
                                                  v_des,               # v_des
                                                  u0[1])               # df_des

        return control, z0, u0, is_opt, solve_time

    ################################################################################################
    ########################## Helper / Update Functions ###########################################
    ################################################################################################
    def _generate_speed_profile(self):
        curv_profile  = self._frenet_traj.trajectory[:, 4]
        self.speed_profile = np.minimum( self.nominal_speed,
                                         np.sqrt( self.lat_accel_max / np.maximum(0.01, np.abs(curv_profile)) )
                                       )

    def _compute_desired_acceleration(self, v_curr, v_target):
        a_des = self.A_MAX * (1 - (v_curr / v_target)**4)
        a_des = np.clip(a_des, self.A_MIN, self.A_MAX)
        return a_des

    def _compute_desired_steer_angle(self, curv, ey, epsi):
        # Feedback/feedforward approach.
        # Adapted from https://github.com/nkapania/Wolverine/blob/9a9efbdc98c7820268039544082002874ac67007/utils/control.py#L16
        df_des = curv * (self.lf + self.lr) - self.k_ey * (ey + self.x_LA * epsi)
        df_des = np.clip(df_des, self.DF_MIN, self.DF_MAX)
        return df_des
