import numpy as np
from scipy.signal import filtfilt

class LaneLocalizer():
    def __init__(self, lane_xs, lane_ys, lane_yaws, lane_vs, s_resolution=0.5):
        # Make sure yaw angles are within bounds:
        lane_ss    = self._get_cumulative_distances(lane_xs, lane_ys)
        lane_yaws = self._bound_angle_within_pi(lane_yaws)

        s_interp    = np.arange(0., lane_ss[-1] + s_resolution/2., s_resolution)
        x_interp    = np.interp(s_interp, lane_ss, lane_xs)
        y_interp    = np.interp(s_interp, lane_ss, lane_ys)
        yaw_interp  = np.interp(s_interp, lane_ss, np.unwrap(lane_yaws))
        v_interp    = np.interp(s_interp, lane_ss, lane_vs)

        curv_interp = self._get_curvatures(s_interp, yaw_interp)
        yaw_interp  = self._bound_angle_within_pi(yaw_interp)

        self.lane_arr = np.column_stack((s_interp, x_interp, y_interp, yaw_interp, v_interp, curv_interp))
        self.lane_length = s_interp[-1]

    @staticmethod
    def _bound_angle_within_pi(angle):
        """ Given an angle, adjusts it to lie within a +/- PI range """
        return (angle + np.pi) % (2 * np.pi) - np.pi # https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap

    @staticmethod
    def _get_cumulative_distances(xs, ys):
        # Arclength/progress estimation.
        lane_xy = np.column_stack((xs, ys))
        lane_ss = np.cumsum( np.linalg.norm( np.diff(lane_xy, axis=0), axis=1 ) )
        lane_ss = np.insert(lane_ss, 0, [0.0])
        return lane_ss

    @staticmethod
    def _get_curvatures(ss, yaws):
        # Curvature estimation.
        curv_raw = LaneLocalizer._bound_angle_within_pi(np.diff(yaws)) / np.diff(ss)

        if len(curv_raw) < 10:
            curv_filt = curv_raw
        else:
            curv_filt = filtfilt(np.ones((3,))/3, 1, curv_raw)

        curv_filt = np.append(curv_filt, curv_filt[-1])
        return curv_filt

    def get_reference_speed_and_curvature(self, s):
        closest_index = np.argmin( np.abs(self.lane_arr[:,0] - s) )

        v_waypoint    = self.lane_arr[closest_index, 4]
        curv_waypoint = self.lane_arr[closest_index, 5]

        return v_waypoint, curv_waypoint

    def convert_global_to_frenet_coords(self, x, y, psi, extrapolate_s = False):
        xy_traj = self.lane_arr[:,1:3]
        xy_query = np.array([x, y])

        closest_index = np.argmin( np.linalg.norm(xy_traj - xy_query, axis=1) )

        # Note: Can do some smarter things here, like linear interpolation.
        # If s_K+1 - s_k is reasonably small, we can assume s of the waypoint
        # and s of the query point are the same for simplicity.
        s_waypoint    = self.lane_arr[closest_index, 0]
        xy_waypoint   = self.lane_arr[closest_index, 1:3]
        psi_waypoint  = self.lane_arr[closest_index, 3]


        rot_global_to_frenet = np.array([[ np.cos(psi_waypoint), np.sin(psi_waypoint)], \
                                         [-np.sin(psi_waypoint), np.cos(psi_waypoint)]])

        # Error_xy     = xy deviation (global frame)
        # Error_frenet = e_s, e_y deviation (Frenet frame)
        error_xy = xy_query - xy_waypoint
        error_frenet = rot_global_to_frenet @ error_xy

        # e_psi
        error_psi = self._bound_angle_within_pi(psi - psi_waypoint)

        if extrapolate_s:
            if closest_index == 0 or closest_index == self.lane_arr.shape[0]-1:
                s_waypoint += error_frenet[0] # Add "e_s" at the endpoints to extrapolate the lane.

        return s_waypoint, error_frenet[1], error_psi # s, ey, epsi

    def convert_frenet_to_global_coords(self, s, ey, epsi):
        s_traj  = self.lane_arr[:,0]

        # Handle "closest waypoint" differently based on s_query:
        if s < s_traj[0]:
            # NOTE: This can be problematic if s_query is really far away from the start.
            # Not handling this but intuitively, need to do some extrapolation.
            x_waypoint   = self.lane_arr[0, 1]
            y_waypoint   = self.lane_arr[0, 2]
            psi_waypoint = self.lane_arr[0, 3]
        elif s > s_traj[-1]:
            # NOTE: This can be problematic if s_query is really far away from the end.
            # Not handling this but intuitively, need to do some extrapolation.
            x_waypoint   = self.lane_arr[-1, 1]
            y_waypoint   = self.lane_arr[-1, 2]
            psi_waypoint = self.lane_arr[-1, 3]
        else:
            # NOTE: keeping this simple and using the closest waypoint, in place of more
            # complex and possibly error-prone interpolation strategies.

            closest_index = np.argmin( np.abs( s_traj - s) )
            x_waypoint   = self.lane_arr[closest_index, 1]
            y_waypoint   = self.lane_arr[closest_index, 2]
            psi_waypoint = self.lane_arr[closest_index, 3]

        rot_frenet_to_global = np.array([[np.cos(psi_waypoint), -np.sin(psi_waypoint)], \
                                         [np.sin(psi_waypoint),  np.cos(psi_waypoint)]])

        error_global = rot_frenet_to_global @ np.array([0, ey]) # assuming "e_s" is 0.

        x_global   = x_waypoint + error_global[0]
        y_global   = y_waypoint + error_global[1]
        psi_global = self._bound_angle_within_pi(psi_waypoint + epsi)

        return x_global, y_global, psi_global

    def get_lane_measurement(self, x, y):
        # Similar to conversion to Frenet coords but getting the actual waypoint / local rotation matrix.
        xy_traj = self.lane_arr[:,1:3]
        xy_query = np.array([x, y])

        closest_index = np.argmin( np.linalg.norm(xy_traj - xy_query, axis=1) )

        xy_waypoint   = self.lane_arr[closest_index, 1:3]
        psi_waypoint  = self.lane_arr[closest_index, 3]

        pose_waypoint = np.append(xy_waypoint, psi_waypoint)

        rot_frenet_to_global = np.array([[np.cos(psi_waypoint), -np.sin(psi_waypoint)], \
                                         [np.sin(psi_waypoint),  np.cos(psi_waypoint)]])

        return pose_waypoint, rot_frenet_to_global