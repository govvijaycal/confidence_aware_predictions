import numpy as np
import pytope

class LaneIntervalManager:
    def __init__(self,
                 frenet_traj_h,   # frenet_trajectory_handler object used to handle lane "localization"
                 dt_mpc,          # time discretization for both MPC + predictions, s
                 A_MIN,           # min acceleration, m/s^2
                 A_MAX,           # max (longitudinal) acceleration, m/s^2
                 lane_width = 3.7 # lane width used to generate lane boundaries, m
                 ):

        self.dt_mpc = dt_mpc
        self.A_MIN  = A_MIN
        self.A_MAX  = A_MAX

        self.frenet_traj_h       = frenet_traj_h
        self.lane_width          = lane_width

        self.ev_half_length      = 2.5 # meters, estimate of CoG to bumper distance
        s_resolution             = self.frenet_traj_h.trajectory[1,0] - self.frenet_traj_h.trajectory[0,0]
        self.obs_buff            = int(self.ev_half_length / s_resolution)
        self.left_lane_boundary  = self.frenet_traj_h.get_left_boundary(offset=lane_width/2.)
        self.right_lane_boundary = self.frenet_traj_h.get_right_boundary(offset=lane_width/2.)
        self.s_lane              = self.frenet_traj_h.trajectory[:,0]

    def generate_interval_constraints(self, s0_ev, v0_ev, v_des, x0_tv, y0_tv, conf_threshs, mus, sigmas, tv_dims):
        # Check inputs.
        n_modes, n_timesteps, n_state = mus.shape

        assert n_state            == 2
        assert conf_threshs.shape == (n_modes,)
        assert sigmas.shape       == (n_modes, n_timesteps, n_state, n_state)

        # Convert the tuple (conf_threshs, mus, sigmas) into a collection of polytopes for each mode.
        tv_conf_sets_by_mode       = self._get_conf_sets(conf_threshs, mus, sigmas)

        # Estimate TV poses based on arctan(dy, dx) for each mode.
        tv_poses_by_mode           = self._get_mean_tv_poses(x0_tv, y0_tv, mus)

        # Get the occupancy sets per mode for the TV, accounting for the TV's shape.
        tv_occupancy_sets_by_mode  = self._get_tv_occupancy_sets(tv_conf_sets_by_mode, tv_poses_by_mode, tv_dims)

        # Project the TV occupancy sets onto the lane to get occupied lane intervals over all modes per timestep.
        s_occupied_intervals       = self._get_occupied_intervals(tv_occupancy_sets_by_mode)

        # Find a feasible set of "interval traversals" for the EV s.t. that avoids the TV by remaining in free intervals.
        s_lb, s_ub                 = self._find_feasible_interval_traversal(s0_ev, v0_ev, v_des, s_occupied_intervals)

        return s_lb, s_ub

    @staticmethod
    def _get_conf_sets(conf_threshs, mus, sigmas):
        # Returns a collection of polytopes as [[P^(j)_t]_{t=1}^{N_H}]_{j=1}^{N_modes}.
        # These polytopes are outer-approximation of the corresponding ellipsoidal level set with X^2 critical value conf_threshs.
        tv_conf_sets_by_mode = []

        for ct, mu_traj, sigma_traj in zip(conf_threshs, mus, sigmas):
            tv_conf_sets_by_timestep = []
            for mu, sigma in zip(mu_traj, sigma_traj):
                evals, evecs = np.linalg.eigh(sigma)

                length_ax1 = np.sqrt(ct * evals[0]) # half the first axis diameter
                length_ax2 = np.sqrt(ct * evals[1]) # half the second axis diameter

                ang_1 = np.arctan2(evecs[1,0], evecs[0,0])
                ang_2 = np.arctan2(evecs[1,1], evecs[0,1])

                a = length_ax1 * np.array([np.cos(ang_1), np.sin(ang_1)])
                b = length_ax2 * np.array([np.cos(ang_2), np.sin(ang_2)])

                vertices = np.vstack((mu + a + b, mu + a - b,\
                                      mu - a + b, mu - a - b))

                conf_set_polytope = pytope.Polytope(vertices)

                tv_conf_sets_by_timestep.append(conf_set_polytope)

            tv_conf_sets_by_mode.append(tv_conf_sets_by_timestep)

        return tv_conf_sets_by_mode

    @staticmethod
    def _get_mean_tv_poses(x0_tv, y0_tv, mus):
        # Returns a collection of pose trajectories as [ np.ndarray([yaw traj in mode j]) ]_{j=1}^{N_modes}.
        # These pose trajectories correspond to the mean behavior (i.e. following mus exactly).
        init_tv_position   = np.array([x0_tv, y0_tv]).reshape((1,2))

        tv_poses_by_mode = []
        for mu in mus:
            tv_mean_traj = np.concatenate((init_tv_position, mu), axis=0)
            tv_diff_traj = np.diff(tv_mean_traj, axis=0)
            tv_pose_traj = np.arctan2(tv_diff_traj[:,1], tv_diff_traj[:,0])
            tv_poses_by_mode.append(tv_pose_traj)

        return tv_poses_by_mode

    @staticmethod
    def _get_tv_occupancy_sets(tv_conf_sets_by_mode, tv_poses_by_mode, tv_dims):
        # Returns a collection of polytopes as [[P^(j)_t]_{t=1}^{N_H}]_{j=1}^{N_modes}.
        # This represents the space + time occupancy of the TV per mode, accounting for the confidence sets (on the vehicle center)
        # as well as the TV vehicle's dimensions.  This is done by a Minkowski sum, where we assume the vehicle's shape is oriented
        # by the mean trajectory pose.

        tv_length = tv_dims["length"] + 1.
        tv_width  = tv_dims["width"]  + 0.5
        tv_shape_polytope = pytope.Polytope(lb=(-tv_length/2., -tv_width/2.),
                                            ub=( tv_length/2.,  tv_width/2.))

        def rotate_by(ang_rad):
            return np.array([[np.cos(ang_rad), -np.sin(ang_rad)],
                             [np.sin(ang_rad),  np.cos(ang_rad)]])

        tv_occupancy_sets_by_mode = []
        for (tv_conf_set_traj, tv_pose_traj) in zip(tv_conf_sets_by_mode, tv_poses_by_mode):
            tv_occupancy_sets_by_timestep = []
            for (tv_conf_set, tv_pose) in zip(tv_conf_set_traj, tv_pose_traj):
                tv_bbox_polytope      = rotate_by(tv_pose) * tv_shape_polytope
                tv_occupancy_polytope = tv_bbox_polytope + tv_conf_set
                tv_occupancy_sets_by_timestep.append(tv_occupancy_polytope)
            tv_occupancy_sets_by_mode.append(tv_occupancy_sets_by_timestep)

        return tv_occupancy_sets_by_mode

    def _get_occupied_intervals(self, tv_occupancy_sets_by_mode):
        # This returns the portion of the lane (i.e. an interval from s=0 to s=lane.length) that the TV may occupy over space + time.
        n_modes = len(tv_occupancy_sets_by_mode)
        n_steps = len(tv_occupancy_sets_by_mode[0])

        def polytope_containment(A, b, pts):
            # A is a N_halfspace by 2 array
            # b is a N_halfspace array
            # pts is a N_pts by 2 array
            # We return a N_pts boolean array which indicates if the
            # corresponding point is contained in the (A,b)-polytope.

            b_pts = A @ pts.T # N_halfspace by N_pts
            return np.all(b_pts <= b, axis=0)

        s_occupied_intervals = []
        for t in range(n_steps):
            s_occupied = np.zeros((len(self.left_lane_boundary)), dtype=np.bool)

            for m in range(n_modes):
                tv_occ_set = tv_occupancy_sets_by_mode[m][t]
                A, b = tv_occ_set.A, tv_occ_set.b


                s_occupied = np.logical_or(s_occupied,
                                           polytope_containment(A, b, self.left_lane_boundary))
                s_occupied = np.logical_or(s_occupied,
                                           polytope_containment(A, b, self.right_lane_boundary))

            if len(s_occupied_intervals) > 0:
                s_occupied = np.logical_or(s_occupied, s_occupied_intervals[-1])

            s_occupied_intervals.append(s_occupied)

        return s_occupied_intervals


    def _find_feasible_interval_traversal(self, s0, v0, v_des, s_occupied_intervals):
        # Greedy approach to identify free intervals along the lane for the EV to travel to over time.

        def should_brake(s, v, s_occ):
            if s < 0 or s > self.s_lane[-1]:
                raise ValueError(f"Invalid value s out of lane bounds: {s} in interval [0, {self.s_lane[-1]}")

            # Compute braking distance and determine where we'd stop.
            # https://traffic-simulation.de/info/info_IDM.html
            A_CMFT = self.A_MIN / 2.0
            s_stopped = s +  v**2 / (2. * abs(A_CMFT)) + 5.0 + 2.0 * v

            # Consider the region within the braking distance. We care about front collisions
            # and assume that rear collisions are not our responsibility.
            s_st_ind  = np.argmin(np.abs(self.s_lane - s))
            s_end_ind = np.argmin(np.abs(self.s_lane - s_stopped))
            s_end_ind = min(s_end_ind + self.obs_buff, len(self.s_lane) - 1)

            if np.any(s_occ[s_st_ind:s_end_ind]):
                # There is a portion of our braking distance which is occupied.
                # So we should brake now.
                return True
            else:
                # Else we can come to a safe stop whenever and don't need to brake yet.
                return False

        def get_free_interval_about(s, s_occ):
            if s < 0 or s > self.s_lane[-1]:
                raise ValueError(f"Invalid value s out of lane bounds: {s} in interval [0, {self.s_lane[-1]}")

            s_idx = np.argmin(np.abs(self.s_lane - s))

            # Identify the start of the interval containing s_idx.
            s_inv_st = s_idx
            while not s_occ[s_inv_st] and s_inv_st > 0:
                s_inv_st = s_inv_st - 1

            # If the interval starts at a non-zero entry, that means we have something behind us.
            # Add a buffer for safety.
            if s_inv_st == 0:
                pass
            else:
                s_inv_st = min(s_inv_st+self.obs_buff, len(self.s_lane) - 1)

            # Identify the end of the interval containing s_idx.
            s_inv_end = s_idx
            while not s_occ[s_inv_end] and s_inv_end < len(self.s_lane) - 1:
                s_inv_end = s_inv_end + 1

            # If the interval ends before the last entry, we have something in front of us.
            # Add a buffer for safety.
            if s_inv_end == len(self.s_lane) -1:
                pass
            else:
                s_inv_end = max(s_inv_end-self.obs_buff, 0)

            if s_inv_st > s_inv_end:
                # Infeasible interval, return an empty one.
                s_inv_st  = s_idx
                s_inv_end = s_idx

            return s_inv_st, s_inv_end


        def get_acceleration(v_curr, k_v = 0.5):
            return np.clip(-k_v * (v_curr - v_des), self.A_MIN, self.A_MAX)


        # Fill in the intervals for the EV over time with greedy approach on accel input.
        # Basically, proceed to track v_des while there is no braking threat.
        # Once a braking threat is identified, decelerate with self.A_MIN.
        s_ind_lb = []
        s_ind_ub = []
        n_steps = len(s_occupied_intervals)
        s_curr = s0
        v_curr = v0
        is_stopped = False

        for t, s_occ in enumerate(s_occupied_intervals):

            if is_stopped or should_brake(s_curr, v_curr, s_occ):
                is_stopped = True
                a_curr = self.A_MIN
            else:
                a_curr = get_acceleration(v_curr)

            s_curr = s_curr + v_curr*self.dt_mpc
            v_curr = v_curr + a_curr*self.dt_mpc

            s_curr = min(s_curr, self.s_lane[-1]) # stay within lane length
            v_curr = max(0., v_curr)              # don't move in reverse

            # Identify the smallest free interval containing s_curr.
            # If it's empty, then return a small interval about s_curr
            # that encloses the EV geometry as the least-harmful solution.
            s_inv_st, s_inv_end = get_free_interval_about(s_curr, s_occ)
            if s_inv_st != s_inv_end:
                # Nonempty interval -> nominal case.
                pass
            else:
                # Empty interval -> give the least-bad solution.
                # We should stop since we detected an infeasible interval.
                is_stopped = True
                s_inv_mid = np.argmin(np.abs(self.s_lane - s_curr))
                s_inv_st  = max(0, s_inv_mid - self.obs_buff)
                s_inv_end = min(len(self.s_lane) - 1, s_inv_mid + self.obs_buff)

            assert s_inv_st < s_inv_end

            s_ind_lb.append(s_inv_st)
            s_ind_ub.append(s_inv_end)

        s_lb = [self.s_lane[ind] for ind in s_ind_lb]
        s_ub = [self.s_lane[ind] for ind in s_ind_ub]

        return s_lb, s_ub