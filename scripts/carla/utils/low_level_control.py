import carla
import numpy as np

class LowLevelControl:
    def __init__(self, vehicle,
                 dt_control = 0.05,     # s, control timestep
                 tau_delay_long = 0.01, # s, throttle/brake 1st order time delay
                 tau_delay_lat  = 0.01, # s, steering angle 1st order time delay
                 k_v = 0.5,             # P gain on velocity tracking error (throttle)
                 k_i = 0.01,            # I gain on velocity tracking error (throttle)
                 i_max = 10.0           # I max value (throttle)
                ):
        # Control setup and parameters.
        self.control_prev = carla.VehicleControl()
        self.max_steer_angle = np.radians( vehicle.get_physics_control().wheels[0].max_steer_angle )

        # Low pass filter gains on actuation to simulate first order delay.
        self.dt_control = dt_control
        self.lat_alpha  = np.exp(-dt_control / tau_delay_lat)
        self.long_alpha = np.exp(-dt_control / tau_delay_long)

        # Throttle Parameters
        self.k_v    = k_v
        self.k_i    = k_i

        self.i_curr = 0.0    # I accumulated value
        self.i_max  = i_max
        self.thr_ff_map  = np.column_stack(([  2.5,  7.5,  12.5,  17.5],        # speed (m/s) -> steady state throttle
                                            [0.325, 0.45, 0.525, 0.625]))

        # Brake Parameters
        self.brake_accel_thresh = -2.0 # m/s^2, value below which the brake is activated
        self.brake_decel_map  = np.column_stack(([ 1.6,  3.9, 6.8,  7.1, 7.9],  # deceleration (m/s^2) -> steady state throttle (at 12 m/s^2)
                                                 [  0., 0.25, 0.5, 0.75, 1.0]))

        self.v_prev       = np.nan
        self.acc_est_lp   = 0.
        self.acc_alpha_lp = np.exp(-1.0)

    def update(self, v_curr, a_des, v_des, df_des):
        control = carla.VehicleControl()
        control.hand_brake = False
        control.manual_gear_shift = False

        # Update acceleration estimate.
        if not np.isnan(self.v_prev):
            self.acc_est_lp = (1 - self.acc_alpha_lp) * (v_curr - self.v_prev) / self.dt_control + self.acc_alpha_lp * self.acc_est_lp

        # Handling integral windup.
        if np.abs(self.i_curr) > self.i_max:
            self.i_curr = 0.

        if a_des > self.brake_accel_thresh:
            # Speed related logic.
            control.throttle = self.k_v * (v_des - v_curr) + self.k_i * self.i_curr
            control.throttle += np.interp(v_des, self.thr_ff_map[:,0], self.thr_ff_map[:,1])
            self.i_curr += (v_des - v_curr) * self.dt_control

            # Acceleration-related logic.
            if not np.isnan(self.v_prev):
                k_a = 1.0 if v_curr <= 1.0 else 0.0
                control.throttle += k_a * (a_des - self.acc_est_lp)

        else:
            control.brake    = np.interp( -a_des, self.brake_decel_map[:,0], self.brake_decel_map[:,1])
            self.i_curr = 0.

        # Simulated actuation delay, also used to avoid high frequency control inputs.
        if control.throttle > 0.0:
            control.throttle = (1 - self.long_alpha) * control.throttle + self.long_alpha * self.control_prev.throttle

        elif control.brake > 0.0:
            control.brake    = (1 - self.long_alpha) * control.brake    + self.long_alpha * self.control_prev.brake

        # Steering control.  Flipped sign due to Carla LHS convention.
        control.steer    = -df_des / self.max_steer_angle
        control.steer    = (1 - self.lat_alpha) * control.steer + self.lat_alpha * self.control_prev.steer

        # Clip Carla control to limits.
        control.throttle = np.clip(control.throttle, 0.0, 1.0)
        control.brake    = np.clip(control.brake, 0.0, 1.0)
        control.steer    = np.clip(control.steer, -1.0, 1.0)

        self.control_prev = control
        self.v_prev = v_curr

        return control
