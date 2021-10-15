import os
import sys
from tqdm import tqdm
import numpy as np
import pickle
import tensorflow as tf
from dataclasses import dataclass
from typing import List
import heapq

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)

from models.context_providers.l5kit_context_provider import L5KitContextProvider
from models.context_providers.nuscenes_context_provider import NuScenesContextProvider
from models.lane_utils.lane_ekf import LaneEKF
from models.lane_utils.lane_localizer import LaneLocalizer
from models.ekf import EKFKinematicBase, EKFKinematicCVTR

from datasets.tfrecord_utils import _parse_no_img_function
from datasets.pose_utils import angle_mod_2pi as bound_angle_within_pi
from datasets.splits import NUSCENES_TRAIN, NUSCENES_VAL, NUSCENES_TEST, \
                            L5KIT_TRAIN, L5KIT_VAL, L5KIT_TEST

from evaluation.prediction_metrics import compute_trajectory_metrics

@dataclass(frozen=True)
class LaneMotionHypothesis():
    # Relative timestamps (secs) for each timestep.
    ts: np.ndarray

    # Selected mean control actions per timestep.
    u_accs  : List[float] # acceleration input, m/s^2
    u_curvs : List[float] # curvature input, rad / m

    # Constant input covariance matrix used for uncertainty propagation.
    # This is a 2 x 2 matrix where Q_u[0,0] = variance(u_accs[i]) and
    #                              Q_u[1,1] = variance(u_curvs[i]).
    Q_u     : np.ndarray

    # Resultant state distributions per timestep.
    # zs[i] is the mean 4-dimensional state (x, y, theta, v).
    # Ps[i] is the corresponding 4-dimensional state covariance.
    zs : List[np.ndarray]
    Ps : List[np.ndarray]

class LaneFollower():

    def __init__(self,
                 dataset_name,
                 Q_ctrl = np.eye(2),
                 R_cost = np.eye(2),
                 n_max_modes = 16,
                 ekf_cvtr_weights_path=None,
                 **kwargs):
        if dataset_name == "nuscenes":
            self.context_provider = NuScenesContextProvider()
        elif dataset_name == "l5kit":
            self.context_provider = L5KitContextProvider()
        else:
            raise ValueError(f"{dataset_name} is not a valid dataset selection.")
        self.dataset_name = dataset_name

        self.n_max_modes = n_max_modes
        self.R_cost = R_cost
        self._init_fixed_params()

        self.ekf_cvtr = EKFKinematicCVTR()
        if ekf_cvtr_weights_path is not None:
            try:
                self.ekf_cvtr_path = ekf_cvtr_weights_path
                self.ekf_cvtr.load_weights(self.ekf_cvtr_path)
                print(f"Using trained covariance params for EKF CVTR from {self.ekf_cvtr_path}")
            except Exception as e:
                raise e
        else:
            self.ekf_cvtr_path = ""
            print("Using default covariance params for EKF CVTR")

        self.lane_ekf = LaneEKF(Q_u = Q_ctrl,
                                R_lane_frame = self.lane_projection_covar)

    def _init_fixed_params(self):
        if self.dataset_name == "l5kit":
            self.n_assoc_pred_timesteps = 2 # 0.4 seconds
        elif self.dataset_name == "nuscenes":
            self.n_assoc_pred_timesteps = 1 # 0.5 seconds
        else:
            raise NotImplementedError(f"{self.dataset_name} not implemented.")

        self.lane_projection_covar = np.diag([0.3, 1.5, 0.5]) # Taken from Eqn 33, https://doi.org/10.1109/ITSC.2013.6728549
        self.lane_width    = self.context_provider.lane_association_radius # width to be considered in the same lane

        self.lat_accel_max = 4.0 # m/s^2, based on https://doi.org/10.3390/electronics8090943 (used to limit v_des)

        # IDM params, picking from Table 11.2 of Traffic Flow Dynamics book.  These correspond to typical
        # parameters in urban traffic environments.
        self.min_gap = 2.0 # m
        self.T_gap   = 1.0 # s
        self.a_max   = 1.0 # m/s^2
        self.b_decel = 1.5 # m/s^2

        # Curvature FF/FB Control Params, selected based on Nitin Kapania's code here:
        # https://github.com/nkapania/Wolverine/blob/9a9efbdc98c7820268039544082002874ac67007/utils/control.py#L16
        # This was originally applied with a full dynamic bicycle model, so arguable if params work without tuning.
        # Leaving this alone for now for simplicity.
        self.k_curv_fb  = 0.0538 # proportional gain, rad/m
        self.x_la       = 14.2   # lookahead distance, m

    def save_weights(self, path):
        model_dict = {}
        model_dict["n_max_modes"]   = self.n_max_modes
        model_dict["ekf_cvtr_path"] = self.ekf_cvtr_path
        model_dict["Q_ctrl"]        = self.lane_ekf.Q_u
        model_dict["R_cost"]        = self.R_cost

        pickle.dump(model_dict, open(path, 'wb'))

    def load_weights(self, path):
        path = path if '.pkl' in path else (path + '.pkl')
        model_dict = pickle.load(open(path, 'rb'))

        assert model_dict["n_max_modes"] == self.n_max_modes

        self.ekf_cvtr_path = model_dict["ekf_cvtr_path"]
        self.ekf_cvtr.load_weights(self.ekf_cvtr_path)

        self.lane_ekf.update_Q_u(model_dict["Q_ctrl"])

        self.R_cost = model_dict["R_cost"]

    def _preprocess_entry(self, entry, split_name, mode="predict", debug=False):
        """ Given a dataset entry from a tfrecord, returns the motion history and associated scene context.

            Note that split_name is one of "train", "val", and "test",
            matching the split suffixes in datasets/splits.py
        """
        entry_proc = {}

        prior_tms, prior_poses, future_tms = EKFKinematicBase.preprocess_entry_prediction(entry)
        entry_proc["prior_tms"]    = prior_tms
        entry_proc["prior_poses"]  = prior_poses
        entry_proc["future_tms"]   = future_tms

        if mode == "train":
            entry_proc["future_poses"] =  np.array(entry['future_poses_local'], dtype=np.float32)
        elif mode == "predict":
            pass
        else:
            raise ValueError(f"{mode} is not a valid mode for tfrecord preprocessing.")

        sample   = tf.compat.as_str(entry["sample"].numpy())
        instance = tf.compat.as_str(entry["instance"].numpy())
        entry_proc["scene_context"] = self.context_provider.get_context(sample, instance, split_name)

        if debug:
            # Sanity check that scene context matches the tfrecord entry.
            pose_record = entry["pose"].numpy()
            sc     = entry_proc["scene_context"]
            pose_sc = np.array([sc.x, sc.y, sc.yaw])

            assert np.allclose(pose_record, pose_sc)

        return entry_proc

    """
    ===========================================================================================================
    Helper (Static) Methods
    ===========================================================================================================
    """
    @staticmethod
    def _identify_split_name(dataset_name, dataset):
        # Check the split and ensure consistency with our context provider.

        def is_contained(subset, full_set):
            return np.all([x in full_set for x in subset])

        split_name = ""
        if dataset_name == "nuscenes":
            if is_contained(dataset, NUSCENES_TRAIN):
                split_name = "train"
            elif is_contained(dataset, NUSCENES_VAL):
                split_name = "val"
            elif is_contained(dataset, NUSCENES_TEST):
                split_name = "test"
            else:
                pass # This is an error, handled below.
        elif dataset_name == "l5kit":
            if is_contained(dataset, L5KIT_TRAIN):
                split_name = "train"
            elif is_contained(dataset, L5KIT_VAL):
                split_name = "val"
            elif is_contained(dataset, L5KIT_TEST):
                split_name = "test"
            else:
                pass # This is an error, handled below.
        else:
            raise NotImplementedError

        if split_name == "":
            raise RuntimeError(f"Mismatch between configured dataset choice: {self.dataset_name} vs. "
                               f"the tfrecord set: {dataset}")
        return split_name

    @staticmethod
    def _extrapolate_pose_trajs(step_fn, pose_traj, dts):
        # This function provides basic predictions based on step_fn over the horizon given by dts.
        # step_fn defines the one-step integration (e.g., CVH or CVTR) function
        # pose_traj is M by 4, each row containing [t, x, y, theta].
        # dts is a vector containing seconds between timesteps.

        def get_pose_at_timestep_0(step_fn, t_last, pose_last, v, w, dt):
            assert dt > 0.
            assert t_last <= 0.
            assert v >= 0.

            x, y, th = pose_last
            # Handle coarse jumps with fixed time discretization dt.
            while t_last <= -dt:
                xn, yn, thn = step_fn(x, y, th, v, w, dt)
                t_last += dt
                x, y, th = xn, yn, thn

            # Handle fine jump with variable timestep based on abs(t_last).
            if t_last < 0.:
                xn, yn, thn = step_fn(x, y, th, v, w, abs(t_last))
                t_last += abs(t_last)
                assert np.allclose(t_last, 0.)
                x, y, th = xn, yn, thn

            return np.array([x, y, th])

        def get_future_poses(step_fn, pose, v, w, dts):
            assert np.all(dts > 0.)
            assert v >= 0.

            x, y, th = pose

            poses = []
            for dt in dts:
                xn, yn, thn = step_fn(x, y, th, v, w, dt)
                poses.append([xn, yn, thn])
                x, y, th = xn, yn, thn

            N = len(poses)
            poses_vel = np.concatenate( (np.array(poses),
                                         v * np.ones((N,1)),
                                         w * np.ones((N,1))),
                                         axis=1)

            return poses_vel

        N = len(dts)
        if pose_traj.shape[0] == 1:
            # We assume a constant pose / zero velocity trajectory if only given a single pose.
            poses     =  np.tile(pose_traj[:, 1:], (N, 1))
            poses_vel = np.concatenate( ( poses, np.zeros((N,2)) ), axis=1 )
        else:
            # Use last two poses to make a simple guess for velocity / turn rate.
            diff_pose = np.diff(pose_traj[-2:, :], axis=0)[0]
            v_est     = np.linalg.norm(diff_pose[1:3]) / diff_pose[0]
            w_est     = diff_pose[3] / diff_pose[0]

            # Get the pose at timestep 0.
            t_last    = pose_traj[-1, 0]
            pose_last = pose_traj[-1, 1:]
            pose_last = get_pose_at_timestep_0(step_fn, t_last, pose_last, v_est, w_est, dts[0])

            # Extrapolate for N steps with time discretization dt.
            poses_vel = get_future_poses(step_fn, pose_last, v_est, w_est, dts)

        assert poses_vel.shape == (N, 5)
        return poses_vel # [x_t, y_t, theta_t, v_t, w_t] for t in [1, N]

    @staticmethod
    def _cvtr_step_fn(x, y, th, v, w, dt):
        # The CVTR 1-step integration function.
        xn  =  x + dt * (v * np.cos(th))
        yn  =  y + dt * (v * np.sin(th))
        thn = th + dt * (w)
        return xn, yn, thn

    @staticmethod
    def _cvh_step_fn(x, y, th, v, w, dt):
        # The CVH 1-step integration function.
        xn  =  x + dt * (v * np.cos(th))
        yn  =  y + dt * (v * np.sin(th))
        thn = th
        return xn, yn, thn

    @staticmethod
    def _identify_lead_agent(step_ind, s_curr, lane_width, lane_localizer, veh_preds, other_agent_preds):
        # This function determines a single agent (if it exists) we should consider for the IDM model.
        # step_ind indicates the timestep to consider in the trajectories given byveh_preds/other_agent_preds.
        # s_curr is the location of the "ego" agent -> s values greater than this are considered in "front".
        # lane_width is used to filter out agents that are not in the same lane as the "ego" agent.
        # lane_localizer is a helper class used to identify lane coordinate projections.
        # *_preds are lists of N by 5 trajectories (see _extrapolate_pose_trajs) for nearby agents to consider.

        def get_lane_projection(x, y, th, v, lane_localizer):
            # Returns the lane (error) coordinates and lane-aligned velocity.
            s, ey, epsi = lane_localizer.convert_global_to_frenet_coords(x, y, th, extrapolate_s = True)
            v_lane = v * np.cos(epsi) # projection of the agent's velocity along the lane direction.
            return s, ey, epsi, v_lane

        s_lead, v_lead = np.nan, np.nan # np.nan used to indicate lack of a lead agent
        all_agt_preds = veh_preds + other_agent_preds  # combined predictions for all agents
        agent_pq = [] # priority queue to rank relevant agents and pick the "closest" one in front

        for agt_pred in all_agt_preds:
            agt_state = agt_pred[step_ind, :] # [x_t, y_t, theta_t, v_t, w_t]
            agt_x, agt_y, agt_th, agt_v, _ = agt_state

            s_agt, ey_agt, epsi_agt, v_lane = get_lane_projection(agt_x, agt_y, agt_th, agt_v, lane_localizer)

            if s_agt > s_curr and np.abs(ey_agt) < 0.5*lane_width:
                # If the agent is in front of us and in the same lane, it's relevant to us.
                heapq.heappush(agent_pq, (s_agt, v_lane))

        if len(agent_pq) > 0:
            # If any relevant agents exist, choose the agent that's the closest in s in front of us.
            s_lead, v_lead = agent_pq[0]

        return s_lead, v_lead

    """
    ===========================================================================================================
    Lane Follower Model Implementation
    ===========================================================================================================
    """
    def get_prior_lane_association(self, entry_proc):
        """ The purpose of this function is to get a prior probability distribution over
            lanes by using distance of short-term predicted poses (using prior motion) to
            the lanes.  This portion doesn't consider any control policies / lane-following behavior.
        """

        # Filter the prior motion.
        filter_dict = self.ekf_cvtr.filter(entry_proc["prior_tms"], entry_proc["prior_poses"])

        # Do short-term prediction to guess the vehicle's pose in n_assoc_pred_timesteps.
        future_dts = np.append([entry_proc["future_tms"][0]],
                                np.diff(entry_proc["future_tms"]))

        for k in range(self.n_assoc_pred_timesteps):
            z, P, _ = self.ekf_cvtr.time_update(future_dts[k])

        # This is in local frame (vehicle coordinate system at current timestep).
        z_local_pose = z[:3]
        P_local_pose = P[:3, :3]

        # Project to closest point on each lane, get squared Mahalanobis distance,
        # and decide whether to keep/prune this lane candidate.
        lane_assoc_priors = []
        sc = entry_proc["scene_context"]

        for lane in sc.lanes:

            # Get the lane coordinates in vehicle local frame.
            lane_poses_local = self.context_provider._transform_poses_to_local_frame(sc.x, sc.y, sc.yaw, lane[:, :3])
            lane_xy_local  = lane_poses_local[:, :2]
            lane_yaw_local = lane_poses_local[:,  2]

            # Find the nearest lane point ("active lane point").
            lane_dists = np.linalg.norm(z_local_pose[:2] - lane_xy_local , axis=1)
            closest_lane_ind = np.argmin(lane_dists)

            # Residual between vehicle and lane active point (in vehicle local frame).
            xy_residual_local  = z_local_pose[:2] - lane_xy_local[closest_lane_ind]
            yaw_residual_local = self.context_provider._bound_angle_within_pi(z_local_pose[2] - lane_yaw_local[closest_lane_ind])
            pose_residual_local = np.append(xy_residual_local, yaw_residual_local)

            # Get the lane projection covariance in vehicle frame (accounting for yaw rotation).
            lane_projection_covar_local = np.copy(self.lane_projection_covar)
            lane_alp_yaw = lane_yaw_local[closest_lane_ind]
            R = np.array([[np.cos(lane_alp_yaw), -np.sin(lane_alp_yaw)],
                          [np.sin(lane_alp_yaw),  np.cos(lane_alp_yaw)]])
            lane_projection_covar_local[:2, :2] = R @ lane_projection_covar_local[:2, :2] @ R.T

            # Get the residual covariance (pose measurement + lane measurement errors combined).
            pose_residual_covar_local = P_local_pose + lane_projection_covar_local

            # Find Mahalanobis distance squared of pose residual according to our specified distribution.
            d_M_sq = pose_residual_local.T @ \
                     np.linalg.pinv(pose_residual_covar_local) @ \
                     pose_residual_local

            # Prior probability based on Mahalanobis distance squared (closer to zero = high prior prob).
            lane_assoc_priors.append( np.exp(-d_M_sq) )

        # Return normalized lane probabilities.
        lane_assoc_priors = np.array(lane_assoc_priors)
        assert np.sum(lane_assoc_priors) > 0.
        return lane_assoc_priors / np.sum(lane_assoc_priors)

    def get_lane_motion_hypotheses(self, entry_proc, prior_lane_probs=None):
        # Filter this agent's motion to get initial state.
        filter_dict = self.ekf_cvtr.filter(entry_proc["prior_tms"], entry_proc["prior_poses"])
        z_cvtr_init = filter_dict["states_ms"][-1] # z_{0|0} where 0 = current time
        P_cvtr_init = filter_dict["covars_ms"][-1] # P_{0|0} ""

        sc = entry_proc["scene_context"]

        future_tms = entry_proc["future_tms"]
        future_dts = np.append([future_tms[0]], np.diff(future_tms))

        # Handle vehicles with simple CVTR predictions, after transforming into this vehicle's local frame.
        veh_agent_preds = []
        for veh_arr in sc.vehicles:
            # [t, x, y, theta]
            veh_arr_local = np.copy(veh_arr)
            veh_poses_local = self.context_provider._transform_poses_to_local_frame(sc.x, sc.y, sc.yaw, veh_arr[:, 1:4])
            veh_arr_local[:, 1:4] = veh_poses_local
            veh_agent_preds.append( self._extrapolate_pose_trajs(self._cvtr_step_fn, veh_arr_local, future_dts) )

        # Handle non-vehicle agents with simple CVH predictions, after transforming into this vehicle's local frame.
        other_agent_preds = []
        for agt_arr in sc.other_agents:
            # [t, x, y, theta]
            agt_arr_local = np.copy(agt_arr)
            agt_poses_local = self.context_provider._transform_poses_to_local_frame(sc.x, sc.y, sc.yaw, agt_arr[:, 1:4])
            agt_arr_local[:, 1:4] = agt_poses_local
            other_agent_preds.append( self._extrapolate_pose_trajs(self._cvh_step_fn, agt_arr_local, future_dts) )

        # If given prior_lane_probs, we can save time by not computing rollouts for pruned lanes (P = 0).
        if prior_lane_probs is None:
            lanes_to_consider = range(len(sc.lanes))
        else:
            lanes_to_consider = [ind for (ind, prob) in enumerate(prior_lane_probs) if prob > 0]

        lane_motion_hypotheses = []
        for lane_idx in range(len(sc.lanes)):
            if lane_idx not in lanes_to_consider:
                lmh = None
            else:
                lane   = np.copy(sc.lanes[lane_idx])
                red_tl = np.copy(sc.red_traffic_lights[lane_idx])

                # Convert lane into vehicle local frame for consistency with z/P.
                # [x, y, theta, v]
                lane_poses_local = self.context_provider._transform_poses_to_local_frame(sc.x, sc.y, sc.yaw, lane[:, :3])
                lane[:, :3]    = lane_poses_local
                lmh = self._get_lane_rollout(z_cvtr_init, P_cvtr_init, future_tms, lane, red_tl, veh_agent_preds, other_agent_preds)

            lane_motion_hypotheses.append(lmh)

        return lane_motion_hypotheses

    def get_posterior_lane_association(self, prior_lane_probs, lane_motion_hypotheses):
        # Evaluate the tracking costs.
        cost_likelihoods = []

        for lmh in lane_motion_hypotheses:
            if lmh is None:
                cost_likelihoods.append(0.)
            else:
                u_comb  = np.column_stack((lmh.u_accs, lmh.u_curvs))
                costs   = np.sum([u.T @ self.R_cost @ u for u in u_comb])
                cost_likelihoods.append( np.exp(-np.sum(costs)) )

        # Compute posterior lane probabilities.
        cost_likelihoods = np.array(cost_likelihoods)
        posterior_lane_probs = cost_likelihoods * prior_lane_probs / np.dot(cost_likelihoods, prior_lane_probs)
        return posterior_lane_probs

    def truncate_num_modes(self, probs, lmhs):
        top_mode_inds = np.argsort(probs)[-self.n_max_modes:]
        top_mode_inds = [x for x in top_mode_inds if probs[x] > 0.]

        probs_final = np.array([probs[x] for x in top_mode_inds])
        probs_final = probs_final / np.sum(probs_final)
        lmhs_final  = [lmhs[x] for x in top_mode_inds]

        return probs_final, lmhs_final

    def _get_acceleration_idm(self, s_curr, v_curr, v_des, s_lead=np.nan, v_lead=np.nan):
            # Applies the Intelligent Driver Model to get the next acceleration input.
            # Reference: Traffic Flow Dynamics, Trieber and Kesting, 2013.  Ch 11.3.
            vel_ratio = v_curr / max(0.1, v_des)

            if np.isnan(v_lead) or np.isnan(s_lead):
                # Free driving case, nothing to worry about braking for.
                a_idm = self.a_max * (1 - vel_ratio**4)
            else:
                # Need to maintain a safe gap since something's in front.
                delta_v   = v_curr - v_lead # called the approaching rate
                gap_des   = self.min_gap + v_curr * max(0, self.T_gap + delta_v / (2 * np.sqrt(self.a_max * self.b_decel)))

                gap_curr  = s_lead - s_curr
                gap_ratio = gap_des / max(0.1, gap_curr)

                a_idm = self.a_max * (1 - (vel_ratio)**4 - gap_ratio**2)

            a_idm = np.clip(a_idm, -self.b_decel, self.a_max) # Limit with a threshold on max deceleration + acceleration.

            return a_idm

    def _get_curv_ff_fb(self, curv_lane, e_y, e_psi):
        # Use a feedforward/feedback curvature policy for a point-mass,
        # inspired by the vehicle version located here:
        # https://ddl.stanford.edu/publications/design-feedback-feedforward-steering-controller-accurate-path-tracking-and-stability
        curv_ff = curv_lane
        curv_fb = -self.k_curv_fb * (e_y + self.x_la * e_psi)
        return curv_ff + curv_fb

    def _get_lane_rollout(self, z_cvtr_init, P_cvtr_init, future_tms, lane, red_tl, veh_agent_preds, other_agent_preds):
        # Given a specified lane and processed scene context, returns a single Gaussian trajectory for lane following behavior.

        # If we don't have speed limit info, best guess of the reference speed is the vehicle's current filtered speed.
        inds_no_speed_limit = np.argwhere( np.isnan(lane[:, 3]) )
        if len(inds_no_speed_limit) > 0:
            lane[inds_no_speed_limit, 3] = z_cvtr_init[3]

        # Handle red traffic light info by setting corresponding lane points (and those following) to 0 speed limit.
        lane_inds_with_red_tl = np.argwhere(red_tl)
        if len(lane_inds_with_red_tl) > 0:
            tl_active_ind = np.amin(lane_inds_with_red_tl)
            lane[tl_active_ind:, 3] = 0.

        # Lane localizer used to handle projections of agents to lane coordinates.
        lane_localizer = LaneLocalizer(lane[:,0], lane[:,1], lane[:,2], lane[:,3])

        u_accs  = [] # acceleration control trajectory
        u_curvs = [] # curvature control trajectory
        zs      = [] # state mean trajectory
        Ps      = [] # state covariance trajectory

        # Get initial kinematic state for our context-aware lane rollout.
        # We ignore angular velocity since curvature is an input in the LaneEKF model.
        z_curr = z_cvtr_init[:4]    # [x, y, th, v]
        P_curr = P_cvtr_init[:4,:4] # covariance associated with z_curr

        dts = np.append([future_tms[0]], np.diff(future_tms))

        self.lane_ekf._reset(z_curr, P_curr)

        for step_ind, dt in enumerate(dts):
            s, ey, epsi       = lane_localizer.convert_global_to_frenet_coords( z_curr[0], z_curr[1], z_curr[2] )
            v_lane, curv_lane = lane_localizer.get_reference_speed_and_curvature(s)

            if s < lane_localizer.lane_length:
                # If we are still within the defined lane region, use an input model based on IDM + FF/FB policies.

                # Use lateral acceleration constraints to limit v_lane on turns.
                if np.abs(curv_lane) >= 0.01:
                    v_lane = min( v_lane, np.sqrt(self.lat_accel_max / np.abs(curv_lane)) )

                s_lead, v_lead = self._identify_lead_agent(step_ind,
                                                           s,
                                                           self.lane_width,
                                                           lane_localizer,
                                                           veh_agent_preds,
                                                           other_agent_preds)

                u_acc  = self._get_acceleration_idm(s, z_curr[3], v_lane, s_lead=s_lead, v_lead=v_lead)
                u_curv = self._get_curv_ff_fb(curv_lane, ey, epsi)

            else:
                # Else we are at the end of the defined lane, let's just assume zero inputs (CVH) due to lack of further context.
                u_acc  = 0.
                u_curv = 0.

            u_accs.append(u_acc)
            u_curvs.append(u_curv)
            u = [u_acc, u_curv]

            z_curr, P_curr, _, _ = self.lane_ekf.time_update(u, dt)
            if s < lane_localizer.lane_length:
                # Use the lane pseudo-measurement if available.
                z_curr, P_curr, _, _ = self.lane_ekf.measurement_update(lane_localizer)

            zs.append(z_curr)
            Ps.append(P_curr)

        return LaneMotionHypothesis(ts=future_tms,
                                    u_accs=u_accs,
                                    u_curvs=u_curvs,
                                    Q_u=self.lane_ekf.Q_u,
                                    zs=zs,
                                    Ps=Ps)

    """
    ===========================================================================================================
    Prediction
    ===========================================================================================================
    """
    def predict(self, dataset):
        ''' Returns a dictionary of predictions given a set of tfrecords. '''
        predict_dict = {}
        split_name = self._identify_split_name(self.dataset_name, dataset)
        dataset = tf.data.TFRecordDataset(dataset)
        dataset = dataset.map(_parse_no_img_function)

        num_instances_without_context = 0

        for entry in tqdm(dataset):
            entry_proc  = self._preprocess_entry(entry, split_name, mode="predict", debug=True)

            if len(entry_proc["scene_context"].lanes) > 0:
                # Make lane context-aware predictions using a Bayesian framework.
                prior_lane_probs     = self.get_prior_lane_association(entry_proc)
                lmhs                 = self.get_lane_motion_hypotheses(entry_proc, prior_lane_probs)
                posterior_lane_probs = self.get_posterior_lane_association(prior_lane_probs, lmhs)

                # Truncate based on number of modes we are allowed to consider.
                final_probs, final_lmhs  = self.truncate_num_modes(posterior_lane_probs, lmhs)

                # Extract the GMM out.
                gmm_pred = {}
                for mode_ind, (prob, lmh) in enumerate( zip(final_probs, final_lmhs) ):
                    mode_dict={}
                    mode_dict['mode_probability'] = prob
                    mode_dict['mus'] = np.array([state[:2] for state in lmh.zs])
                    mode_dict['sigmas'] = np.array([covar[:2, :2] for covar in lmh.Ps])

                    gmm_pred[mode_ind] = mode_dict
            else:
                # We don't have any lane context, just revert to constant velocity/turn rate base model.
                num_instances_without_context += 1
                 # Filter the prior motion.
                filter_dict = self.ekf_cvtr.filter(entry_proc["prior_tms"], entry_proc["prior_poses"])

                # Do short-term prediction to guess the vehicle's pose in n_assoc_pred_timesteps.
                future_dts = np.append([entry_proc["future_tms"][0]],
                                        np.diff(entry_proc["future_tms"]))

                states = []
                covars = []

                for dt in future_dts:
                    z, P, _ = self.ekf_cvtr.time_update(dt)
                    states.append(z)
                    covars.append(P)

                mode_dict={}
                mode_dict['mode_probability'] = 1.
                mode_dict['mus'] = np.array([state[:2] for state in states])
                mode_dict['sigmas'] = np.array([covar[:2, :2] for covar in covars])

                gmm_pred = {0: mode_dict}

            # Log results to dictionary.
            key = f"{tf.compat.as_str(entry['sample'].numpy())}_{tf.compat.as_str(entry['instance'].numpy())}"
            future_states = tf.cast(tf.concat([tf.expand_dims(entry['future_tms'], -1),
                                         entry['future_poses_local']], -1), dtype=tf.float32)
            prior_tms   = entry_proc["prior_tms"]
            prior_poses = entry_proc["prior_poses"]

            predict_dict[key] = {'type': tf.compat.as_str(entry['type'].numpy()),
                                 'velocity': tf.cast(entry['velocity'], dtype=tf.float32).numpy().item(),
                                 'yaw_rate': tf.cast(entry['yaw_rate'], dtype=tf.float32).numpy().item(),
                                 'acceleration': tf.cast(entry['acceleration'], dtype=tf.float32).numpy().item(),
                                 'pose': tf.cast(entry['pose'], dtype=tf.float32).numpy(),
                                 'past_traj': np.concatenate((np.expand_dims(prior_tms[:-1], axis=1), prior_poses[:-1]), axis=1),
                                 'future_traj': future_states.numpy(),
                                 'gmm_pred': gmm_pred}

        print(f"There were {num_instances_without_context} for which no lane info was available.")
        return predict_dict

    def predict_instance(self, image_raw, past_states, future_tms=np.arange(0.2, 5.1,0.2)):
        raise NotImplementedError

    """
    ===========================================================================================================
    Training
    ===========================================================================================================
    """
    def fit(self, train_set, val_set, logdir=None, **kwargs):
        ''' Fit params (self.lane_ekf.Q_u and self.R_cost) based on a subset of the train_set. '''

        # Deterministically pick out a subset of the training set to fit on.
        np.random.seed(0)
        np.random.shuffle(train_set)
        train_set = train_set[:20]

        # Step 1.  Fit Q_u based on the log-likelihood of the most likely mode (with R_cost fixed).
        sigma_acc_sq_cands  = [1e-2, 1e-1, 1e0]
        sigma_curv_sq_cands = [1e-4, 1e-3, 1e-2]
        ll_fit_list = []

        for sigma_acc_sq in sigma_acc_sq_cands:
            for sigma_curv_sq in sigma_curv_sq_cands:
                Q_eval = np.diag([sigma_acc_sq, sigma_curv_sq])
                self.lane_ekf.update_Q_u(Q_eval)
                predict_dict = self.predict(train_set)
                metrics_df = compute_trajectory_metrics(predict_dict, ks_eval=[1])

                ll_result = np.mean(metrics_df.traj_LL_1)
                ll_fit_list.append( [ll_result, sigma_acc_sq, sigma_curv_sq] )

        ll_fit_list = np.array(ll_fit_list)
        print("Q", ll_fit_list)
        best_fit_ind       = np.argmax( ll_fit_list[:, 0] )
        best_sigma_acc_sq  = ll_fit_list[best_fit_ind, 1]
        best_sigma_curv_sq = ll_fit_list[best_fit_ind, 2]
        self.lane_ekf.update_Q_u( np.diag([best_sigma_acc_sq,
                                           best_sigma_curv_sq]) )

        print(f"BEST Q_u: {self.lane_ekf.Q_u}")

        # Step 2.  If we have a multimodal model, fit R_cost based on 5 (max) modes log-likelihood.
        if self.n_max_modes > 1:
            R_cost_accs  = [1e-4, 1e-2, 1]
            R_cost_curvs = [1e-2,  1e0, 1e2]

            ll_fit_list = []
            for R_acc in R_cost_accs:
                for R_curv in R_cost_curvs:
                    self.R_cost = np.diag([R_acc, R_curv])
                    predict_dict = self.predict(train_set)
                    metrics_df = compute_trajectory_metrics(predict_dict, ks_eval=[5])

                    ll_result = np.mean(metrics_df.traj_LL_5)
                    ll_fit_list.append( [ll_result, R_acc, R_curv] )

            ll_fit_list = np.array(ll_fit_list)
            print("R", ll_fit_list)
            best_fit_ind = np.argmax( ll_fit_list[:, 0] )
            R_acc_best   = ll_fit_list[best_fit_ind, 1]
            R_curv_best  = ll_fit_list[best_fit_ind, 2]
            self.R_cost = np.diag( [R_acc_best, R_curv_best] )

        print(f"BEST R_cost: {self.R_cost}")

        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)
            filename = logdir + 'params.pkl'
            self.save_weights(filename)
