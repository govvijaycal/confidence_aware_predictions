import os
import sys
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import pickle
import tensorflow as tf

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)
from datasets.tfrecord_utils import _parse_no_img_function
from datasets.pose_utils import angle_mod_2pi as bound_angle_within_pi

# Note: Based on Fig. 4 of "Synthetic 2D LIDAR for precise vehicle localization in 3D urban environment", ICRA 2013,
# I am setting a 3sigma bound of 20 cm position error and 1 degree orientation error.
# I am guessing a reasonable max range of acceleration is +/- 1 g.
# But less confident about this estimate so use this as a 1-sigma bound.
# Similar story for velocity (assume 0-20 m/s range) and angular velocity (assume -1 to 1 rad/s range).
ORI_STD_ERROR   = np.radians(10.) / 3.         # std error of approx. 5.8e-2 radians for yaw angle
POS_STD_ERROR   = 0.2 / 3.                     # std error of approx. 6.7 cm for position (radius of 1-std error circle)
XY_STD_ERROR    = POS_STD_ERROR / np.sqrt(2.)  # std error of approx. 4.7 cm error along x or y axis
VEL_INIT_ERROR  = 10.                          # guesstimate std error (m/s) for initial velocity state
ACC_INIT_ERROR  = 10.                          # guesstimate std error (m/s^2) for initial acceleration state
YR_INIT_ERROR   = 1.                           # guesstimate std error (rad/s) for initial yawrate state

# Midpoint Guesses for Kinematic State Initialization.
VEL_INIT_GUESS = 10. # [0, 20] m/s range
ACC_INIT_GUESS = 0.  # [-10, 10] m/s^2 range
YR_INIT_GUESS  = 0.  # [-1, 1] rad/s range

class EKFKinematicBase(ABC):
    """ Base Class for Extended Kalman Filter variants.  Contains the
        main time/measurement updates and prediction API, letting children
        classes provide the custom dynamics models and covariances.

        Reference Notes Used For EM/Smoother: CS287 Lecture 13,
        https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/slides/Lec13-KalmanSmoother-MAP-ML-EM.pdf
        """

    ##################### Model Varying  #############################
    ##################################################################
    @abstractmethod
    def __init__(self, **kwargs):
        """ Initializes covariance / model params for each EKF.
            This is model-dependent due to varying states.
        """
        raise NotImplementedError

    @abstractmethod
    def _dynamics_model(self, state):
        """ Returns the next state given the current state.
            Note that the discretization time, dt, is appended to the input
            argument but not to the return value.  In other words,
            the state passed in has one more entry than the state returned.
        """
        raise NotImplementedError

    @abstractmethod
    def get_init_state(init_pose):
        """ Provides an initial state guess given the initial pose.
            Basically, this just fills in the kinematic state elements
            with a reasonable guess (with uncertainty captured by self.P_init).
        """
        raise NotImplementedError

    ##################### Weight Utils  ##############################
    ##################################################################
    def save_weights(self, path):
        path = path if '.pkl' in path else (path + '.pkl')
        model_dict = {'P': self.P_init,
                      'Q': self.Q,
                      'R': self.R}

        pickle.dump(model_dict, open(path, 'wb'))

    def load_weights(self, path):
        path = path if '.pkl' in path else (path + '.pkl')
        model_dict = pickle.load(open(path, 'rb'))

        # I assume the _reset function will be used to update self.P.
        self.P_init = model_dict['P']
        self.Q      = model_dict['Q']
        self.R      = model_dict['R']

    ################### EKF Implementation ###########################
    ##################################################################
    def filter(self, tms, poses):
        """ Runs the forward update equations for the EKF.

        Notation: *_tm = estimate after a time update
                  *_ms = estimate after a measurement update
        We are lumping the initial state estimate / covariance in the _ms fields.
        """
        filter_dict = {}
        filter_dict['states_tm']         = [] # 1->N
        filter_dict['covars_tm']         = [] # 1->N
        filter_dict['A_tm']              = [] # 0->N-1 (taken wrt the state before the time update!)
        filter_dict['states_ms']         = [] # 0->N
        filter_dict['covars_ms']         = [] # 0->N
        filter_dict['residual_ms']       = [] # 1->N
        filter_dict['residual_covar_ms'] = [] # 1->N
        filter_dict['residual_log_likelihood'] = 0.

        dts = np.diff(tms)

         # Initial State: Set z_{0 | 0}, P_{0 | 0} and include it as a "measurement" estimate.
        full_init_state = self.get_init_state(poses[0])
        self._reset(full_init_state)
        filter_dict['states_ms'].append(self.z)
        filter_dict['covars_ms'].append(self.P)

        # Recursions:
        for dt, next_pose in zip(dts, poses[1:]):
            z, P, A = self.time_update(dt)
            filter_dict['states_tm'].append(z)
            filter_dict['covars_tm'].append(P)
            filter_dict['A_tm'].append(A)

            z2, P2, res, res_covar = self.measurement_update(next_pose)
            filter_dict['states_ms'].append(z2)
            filter_dict['covars_ms'].append(P2)
            filter_dict['residual_ms'].append(res)
            filter_dict['residual_covar_ms'].append(res_covar)

        # Compute residual log-likelihood.
        res_dim = filter_dict['residual_ms'][0].size
        const_term = res_dim / 2. * np.log(2 * np.pi)

        for res, res_covar in zip(filter_dict['residual_ms'], \
                                  filter_dict['residual_covar_ms']):

            log_det_term = 1./2. * np.log( np.linalg.det(res_covar) )
            dist_term    = 1./2. * res.T @ np.linalg.pinv(res_covar) @ res

            filter_dict['residual_log_likelihood'] -= (const_term + log_det_term + dist_term)

        return filter_dict

    @staticmethod
    def smoother(filter_dict):
        """ Runs the backward update equations for the RTS Smoother.

        Notation: *_sm = estimate after applying a smoother update

        Unlike the filter, this is a static function so we are not modifying
        the object properties (i.e., self.z, self.P).

        Note that we are applying the linear KF smoother equations
        to a nonlinear system, so there is a slight abuse of the assumptions here.

        In particular, the linearization point would really change after
        smoothening a state estimate, but we are ignoring this for simplicity.
        This linearization error is exacerbated for the first couple states where the
        velocity/acceleration/yaw rate are more uncertain.

        See slide 63-64 of https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/slides/Lec13-KalmanSmoother-MAP-ML-EM.pdf.
        """
        smoother_dict = {}
        smoother_dict['states_sm'] = [] # 0, ..., N; z_{t | 0:N}
        smoother_dict['covars_sm'] = [] # 0, ..., N; P_{t | 0:N}
        smoother_dict['L_sm']      = [] # 0, ..., N-1; L_t (smoother gain matrix)

        N = len(filter_dict['A_tm'])

        # Final State: Initialize from z_{N | 0:N}, P_{N | 0:N}.
        smoother_dict['states_sm'].append(filter_dict['states_ms'][N])
        smoother_dict['covars_sm'].append(filter_dict['covars_ms'][N])

        for t in range(N-1, -1, -1): # starting from N-1, proceeding to 0
            z_sm_next = smoother_dict['states_sm'][-1]
            P_sm_next = smoother_dict['covars_sm'][-1]

            z_ms = filter_dict['states_ms'][t]      # z_{t | 0:t}
            P_ms = filter_dict['covars_ms'][t]      # P_{t | 0:t}

            z_tm_next = filter_dict['states_tm'][t] # z_{t+1 | 0:t}
            P_tm_next = filter_dict['covars_tm'][t] # P_{t+1 | 0:t}
            A         = filter_dict['A_tm'][t]      # A_t

            L = P_ms @ A.T @ np.linalg.pinv(P_tm_next)
            smoother_residual = z_sm_next - z_tm_next
            smoother_residual[2] = bound_angle_within_pi(smoother_residual[2])

            z_sm = z_ms + L @ (smoother_residual)
            z_sm[2] = bound_angle_within_pi(z_sm[2])
            P_sm = P_ms + L @ (P_sm_next - P_tm_next) @ L.T

            # TODO: REMOVE THIS
            assert np.allclose(P_sm, P_sm.T)

            smoother_dict['states_sm'].append(z_sm) # z_{t | 0:N}
            smoother_dict['covars_sm'].append(P_sm) # P_{t | 0:N}
            smoother_dict['L_sm'].append(L)         # L_t

        # Change the data ordering to be ascending in time index (0, 1, ...).
        for key in smoother_dict.keys():
            smoother_dict[key].reverse()

        return smoother_dict

    def time_update(self, dt):
        ''' Performs the time/dynamics update step, changing the state estimate (z, P). '''
        state = np.append(self.z, dt)
        self.z = self._dynamics_model(state)
        self.z[2] = bound_angle_within_pi(self.z[2])
        A      = self._dynamics_jacobian(state)
        self.P = A @ self.P @ A.T + self.Q

        # TODO: REMOVE THIS.
        assert np.allclose(self.P, self.P.T)

        # Return values:
        # z_{t+1 | 0:t}, state estimate at time t+1 given measurements to t.
        # P_{t+1 | 0:t}, state covariance at time t+1 given measurements to t.
        # A_t, linearization about state z at time t.
        return self.z, self.P, A

    def measurement_update(self, measurement):
        ''' Performs the measurement update step, changing the state estimate (z, P).
            We assume but do not check that a time_update step was done first. '''
        residual  = self._obs_model(self.z, measurement)
        H         = self._obs_jacobian(self.z)

        res_covar = H @ self.P @ H.T + self.R
        K         = self.P @ H.T @ np.linalg.pinv(res_covar)

        self.z    = self.z + K @ residual
        self.z[2] = bound_angle_within_pi(self.z[2])
        self.P    = (np.eye(self.nx) - K @ H) @ self.P

        # TODO: REMOVE THIS
        assert np.allclose(res_covar, res_covar.T)
        assert np.allclose(self.P, self.P.T)

        # Return values:
        # z_{t+1 | 0:t+1}, state estimate at time t+1 given measurements to t+1.
        # P_{t+1 | 0:t+1}, state covariance at time t+1 given measurements to t+1.
        # o_{t+1} - H @ z_{t+1: 0:t}, measurement residual (aka innovation) at time t+1.
        # covariance corresponding to the residual/innovation
        return self.z, self.P, residual, res_covar

    def _reset(self, z_init):
        ''' Resets the initial state estimate for a new track. '''
        self.P = self.P_init
        self.z = z_init

    def _dynamics_jacobian(self, state, eps=1e-8):
        ''' Numerical jacobian computed via finite differences for the dynamics.
            We could do this symbolically but this is less error prone, especially given
            factors like variable dts, potentially using more complex integration approaches,
            and constrained positive speed.
        '''
        jac = np.zeros((self.nx, self.nx))

        for i in range(self.nx):
            # nx + 1 here is a bit confusing.  We're including it because
            # the _dynamics_model needs to know the "dt", so its argument
            # is of size nx  + 1.
            splus  = state + eps * np.array([int(ind==i) for ind in range(self.nx+1)])
            sminus = state - eps * np.array([int(ind==i) for ind in range(self.nx+1)])

            f_plus  = self._dynamics_model(splus)
            f_minus = self._dynamics_model(sminus)

            diff = f_plus - f_minus
            diff[2] = bound_angle_within_pi(diff[2])

            jac[:,i] = diff / (2.*eps)
        return jac

    def _obs_model(self, state, measurement):
        ''' Returns the residual using the state estimate, measurement, and observation model. '''
        expected    = self._obs_jacobian(state) @ state # linear measurement model
        residual    = measurement - expected
        residual[2] = bound_angle_within_pi(residual[2])
        return residual

    def _obs_jacobian(self, state):
        ''' Returns the observation jacobian given the current state. '''
        H        = np.zeros((3, self.nx))
        H[:, :3] = np.eye(3)
        return H

    ##################### Prediction  ################################
    ##################################################################
    @staticmethod
    def preprocess_entry_prediction(entry):
        ''' Prepares a batch of states (past trajectory) and future timestamps given an entry from a TF Dataset.
            This is meant to predict future states using the predict function.
        '''
        prior_tms = np.array(entry['past_tms'][::-1], dtype=np.float32)
        prior_tms = np.append( prior_tms, np.float32(0.) )

        prior_poses = np.array(entry['past_poses_local'][::-1, :], dtype=np.float32)
        prior_poses = np.concatenate( (prior_poses, np.zeros((1,3), dtype=np.float32)), axis=0)

        future_tms =  np.array(entry['future_tms'], dtype=np.float32)

        return prior_tms, prior_poses, future_tms

    def predict(self, dataset):
        ''' Returns a dictionary of predictions given a set of tfrecords. '''
        predict_dict = {}

        dataset = tf.data.TFRecordDataset(dataset)
        dataset = dataset.map(_parse_no_img_function)

        for entry in tqdm(dataset):
            prior_tms, prior_poses, future_tms  = self.preprocess_entry_prediction(entry)

            # Filter the previous pose history.
            filter_dict = self.filter(prior_tms, prior_poses)

            # Predict via extrapolating the motion model.
            future_dts = np.append([future_tms[0]], np.diff(future_tms))
            states = []
            covars = []

            for dt in future_dts:
                z, P, _ = self.time_update(dt)
                states.append(z)
                covars.append(P)

            mode_dict={}
            mode_dict['mode_probability'] = 1.
            mode_dict['mus'] = np.array([state[:2] for state in states])
            mode_dict['sigmas'] = np.array([covar[:2, :2] for covar in covars])

            key = f"{tf.compat.as_str(entry['sample'].numpy())}_{tf.compat.as_str(entry['instance'].numpy())}"
            future_states = tf.cast(tf.concat([tf.expand_dims(entry['future_tms'], -1),
                                         entry['future_poses_local']], -1), dtype=tf.float32)

            predict_dict[key] = {'type': tf.compat.as_str(entry['type'].numpy()),
                                 'velocity': tf.cast(entry['velocity'], dtype=tf.float32).numpy().item(),
                                 'yaw_rate': tf.cast(entry['yaw_rate'], dtype=tf.float32).numpy().item(),
                                 'acceleration': tf.cast(entry['acceleration'], dtype=tf.float32).numpy().item(),
                                 'pose': tf.cast(entry['pose'], dtype=tf.float32).numpy(),
                                 'past_traj': np.concatenate((np.expand_dims(prior_tms[:-1], axis=1), prior_poses[:-1]), axis=1),
                                 'future_traj': future_states.numpy(),
                                 'gmm_pred': {0: mode_dict}}

        return predict_dict

    def predict_instance(self, image_raw, past_states, future_tms=np.arange(0.2, 5.1, 0.2)):
        """ Runs prediction on a single instance for real-time implementation.
            Future times are required to know how many states ahead to predict and at what time interval.
        """
        import pdb; pdb.set_trace() # TODO: remove this.
        past_states = np.concatenate( (past_states, np.zeros((1,4)).astype(np.float32)), axis=0 ) # Add the zero time/pose.
        prior_tms = past_states[:, 0]
        prior_poses = past_states[:, 1:]

        # Filter the previous pose history.
        filter_dict = self.filter(prior_tms, prior_poses)

        # Predict via extrapolating the motion model.
        future_dts = np.append([future_tms[0]], np.diff(future_tms))
        states = []
        covars = []

        for dt in future_dts:
            z, P, _ = self.time_update(dt)
            states.append(z)
            covars.append(P)

        mode_dict={}
        mode_dict['mode_probability'] = 1.
        mode_dict['mus'] = np.array([state[:2] for state in states])
        mode_dict['sigmas'] = np.array([covar[:2, :2] for covar in covars])

        gmm_pred = {0: mode_dict}

        return gmm_pred

    ##################### Model Fitting  #############################
    ##################################################################
    @staticmethod
    def preprocess_entry(entry):
        """ Prepares a batch of states (past + future trajectory) given an entry from a TF Dataset.
            This is meant to estimate the disturbance covariance using the fit function.
        """
        tms = np.array(entry['past_tms'][::-1], dtype=np.float32)
        tms = np.append( tms, np.float32(0.) )
        tms = np.append( tms, np.array(entry['future_tms'], dtype=np.float32) )

        poses = np.array(entry['past_poses_local'][::-1, :], dtype=np.float32)
        poses = np.concatenate( (poses, np.zeros((1,3), dtype=np.float32)), axis=0)
        poses = np.concatenate( (poses, np.array(entry['future_poses_local'], dtype=np.float32)), axis=0)

        return tms, poses

    def fit(self, train_set, val_set, logdir=None, **kwargs):
        """ Identifies Q covariance matrix from the data.

        We use a EM algorithm on the smoothed state estimates.
        We assume the linearization error is not too large so the EM procedure for linear KFs can be applied.

        Essentially, run filter/smoother to get smoothed state estimates.
        Then find the Q_MLE in terms of the smoothed estimates, do a "coarse" backtracking line search
        with the residual log likelihood to choose the next Q.
        """

        # First check if we need to make a cached_train_dataset (numpy array)
        # from tfrecords.  If so, then use the tf.data pipeline for the first
        # call to evaluate_candidate_Q and stick with numpy array after that.
        train_dataset, cached_train_dataset = None, None
        if type(train_set) == np.ndarray:
            # Already a cached numpy array, use it as is.  No extra work
            # to be done in the first call of evaluate_candidate_Q.
            cached_train_dataset = train_set
            del train_dataset
        else:
            # A tfrecord dataset, used only in the first function call
            # of evaluate_candidate_Q.  We explicitly del this
            # after caching, so we don't need to use the TFRecordDataset.
            train_dataset = tf.data.TFRecordDataset(train_set)
            train_dataset = train_dataset.map(_parse_no_img_function)

            # A numpy array we'll build.  This is a cached version of
            # the trajectories in train_dataset, used for subsequent
            # function calls of evaluate_candidate_Q.
            cached_train_dataset = []

        def evaluate_candidate_Q(Q_candidate, get_MLE = True):
            # This inner function simply uses the current Q_candidate to run the KF filter and smoother,
            # allowing us to (1) compute residual_log_likelihood at Q_candidate and (2) get a MLE guess
            # of the next Q to try out (if get_MLE = True).

            self.Q = Q_candidate # Set the KF's Q matrix to our Q_candidate.

            Q_mle_trajs  = [] # Stores the fitted Q_MLE for a single trajectory (dataset instance)
            loglik_trajs = [] # Stores the residual log likelihood for a single trajectory (dataset instance).

            # We access the datasets from the outer scope.
            nonlocal cached_train_dataset
            if len(cached_train_dataset) == 0:
                # this time, make the cache dataset
                nonlocal train_dataset
                generate_cache_dataset = True
                dataset = train_dataset
            else:
                # this time, use the cache dataset
                generate_cache_dataset = False
                dataset = cached_train_dataset

            for entry in tqdm(dataset):
                if generate_cache_dataset:
                    # Incrementally add the tfrecord processed entries to a cached dataset.
                    tms, poses  = self.preprocess_entry(entry)
                    cached_train_dataset.append(np.column_stack((tms, poses)))
                else:
                    # Just access the entry from the cached (numpy) dataset.
                    tms = entry[:, 0]
                    poses = entry[:, 1:]

                dts = np.diff(tms)
                filter_dict   = self.filter(tms, poses)

                # Note down log likelihood of the current Q_candidate.
                loglik_trajs.append( filter_dict['residual_log_likelihood'] )

                if get_MLE:
                    smoother_dict = self.smoother(filter_dict)

                    # Compute update term for Q_MLE:
                    N = len(filter_dict['A_tm'])
                    Q_MLE = np.zeros_like(self.Q)

                    for t in range(N): # 0, ..., N-1
                        A         = filter_dict['A_tm'][t]           # A_t
                        L         = smoother_dict['L_sm'][t]         # L_t

                        z_sm      = smoother_dict['states_sm'][t+1]  # z_{t+1 | 0:N}
                        P_sm      = smoother_dict['covars_sm'][t+1]  # P_{t+1 | 0:N}

                        z_sm_prev = smoother_dict['states_sm'][t]    # z_{t | 0:N}
                        P_sm_prev = smoother_dict['covars_sm'][t]    # P_{t | 0:N}

                        dt = dts[t] # not the greatest notation, but means time interval betweeen t and t+1 timestamps

                        res_model = z_sm - self._dynamics_model( np.append(z_sm_prev, dt) )
                        res_model[2] = bound_angle_within_pi(res_model[2])

                        Q_MLE += 1./N * (res_model @ res_model.T + \
                                         A @ P_sm_prev @ A.T + P_sm - \
                                         P_sm @ L.T @ A.T - A @ L @ P_sm)

                    Q_mle_trajs.append( Q_MLE )

            if generate_cache_dataset:
                # Convert the cached dataset from a list to numpy array.
                # Free the memory occupied by the TFRecord Dataset.
                cached_train_dataset = np.array(cached_train_dataset)
                del train_dataset

            loglik = np.mean(loglik_trajs)

            if not get_MLE:
                return loglik
            else:
                Q_MLE  = np.mean(Q_mle_trajs, axis=0)

                # TODO: Remove this later, for debugging.
                assert np.allclose(Q_MLE, Q_MLE.T)

                return loglik, Q_MLE

        # EM algorithm implementation follows.
        Q_by_iter       = [self.Q]
        log_lik_by_iter = []

        max_iters = 10
        frob_norm_eps = 1e-2

        print('Starting EM Fit:')
        for em_iter in range(max_iters):
            Q_current = Q_by_iter[-1]

            print(f"\n\tEM Q Fit: Iter {em_iter+1} of {max_iters}")
            print(f"\t{np.diag(Q_current)}")

            loglik, Q_MLE = evaluate_candidate_Q(Q_current, get_MLE = True)
            log_lik_by_iter.append(loglik)

            print(f"\tLL: {loglik}")

            # Decision point: converged, continue, or do a line search?
            if np.linalg.norm(Q_current - Q_MLE, ord='fro') < frob_norm_eps:
                # If the new Q "guess" is sufficiently close to the current Q guess,
                # no point doing a line search update.  Just use the current Q guess.
                break

            # A very simplified/coarse attempt at backtracking line search to handle cases
            # where Q_MLE is not actually improving the residual log-likelihood.
            # Essentially, using Q_MLE as a "descent direction" guide and choosing
            # a stepsize from a set of options, rather than computing the stepsize.
            improved_LL = False
            for s in [0., 0.25, 0.5, 0.75]:
                Q_test  = s * Q_current + (1. - s) * Q_MLE # Note s starts from 0, so we first try Q_MLE.
                ll_test = evaluate_candidate_Q(Q_test, get_MLE = False)
                # TODO: for very large datasets, we may opt to only evaluate
                # residual log likelhood on a subsample to speed this up.

                if ll_test > loglik:
                    # Could find a better direction to proceed. Proceed to next iter of EM.
                    improved_LL = True
                    Q_by_iter.append(Q_test)
                    break

            if not improved_LL:
                break # Could not find a better direction to proceed.  Terminate early.

        # Use the final Q matrix as the fitted result.

        print('EM Fit Results: ')
        Q_final = Q_by_iter[-1]
        print(f"\t{np.diag(Q_final)}")
        loglik_final = evaluate_candidate_Q(Q_final, get_MLE = False)
        print(f"\tLL: {loglik_final}")

        self.Q = Q_final

        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)
            filename = logdir + 'params.pkl'
            self.save_weights(filename)

##################################################################
################ Full Order EKF with 6 states ####################
##################################################################
class EKFKinematicCATR(EKFKinematicBase):
    ''' 6-state Kinematic Extended Kalman Filter Implementation.

        The kinematic state (z) used here has the following elements:
        x        = z[0], X coordinate (m)
        y        = z[1], Y coordinate (m)
        theta    = z[2], yaw angle (rad) counterclockwise wrt X-axis
        v        = z[3], longitudinal speed (m/s)
        w        = z[4], yaw_rate (rad/s)
        acc      = z[5], longitudinal acceleration, i.e. dv/dt (m/s^2)

        The measurement model is a pose measurement, i.e. o = [x, y, theta].
    '''

    def __init__(self,
                 z_init=np.zeros(6),
                 P_init=np.diag(np.square([XY_STD_ERROR, XY_STD_ERROR, ORI_STD_ERROR, \
                                           VEL_INIT_ERROR, YR_INIT_ERROR, ACC_INIT_ERROR])),
                 Q=np.eye(6),
                 R=np.diag(np.square([XY_STD_ERROR, XY_STD_ERROR, ORI_STD_ERROR]))):

        self.nx = 6     # state dimension

        self.P_init = P_init # state covariance initial value, kept to reset the filter for different tracks.
        self.Q = Q           # disturbance covariance, identified from data
        self.R = R           # measurement noise covariance

        self._reset(z_init)

    def _dynamics_model(self, state):
        ''' Given the current state at timestep t, returns the next state at timestep t+1. '''
        # Note: we assume the vehicle is always moving forward and not in reverse.
        x, y, th, v, w, acc, dt = state

        staten = np.zeros((self.nx), dtype=state.dtype)

        staten[5] = acc
        staten[4] = w
        staten[3] = max(v + acc*dt, 0.)
        staten[2] = th    + w*dt
        staten[1] = y     + v*np.sin(th)*dt
        staten[0] = x     + v*np.cos(th)*dt

        return staten

    @staticmethod
    def get_init_state(init_pose):
        x_init, y_init, th_init = init_pose
        return np.array([x_init, y_init, th_init, VEL_INIT_GUESS, YR_INIT_GUESS, ACC_INIT_GUESS])


##################################################################
################ Reduced EKF - Zero Yaw Rate #####################
##################################################################
class EKFKinematicCAH(EKFKinematicBase):
    ''' 5-state Kinematic Extended Kalman Filter Constant Acceleration + Heading Implementation.

        The kinematic state (z) used here has the following elements:
        x        = z[0], X coordinate (m)
        y        = z[1], Y coordinate (m)
        theta    = z[2], yaw angle (rad) counterclockwise wrt X-axis
        v        = z[3], longitudinal speed (m/s)
        acc      = z[4], longitudinal acceleration, i.e. dv/dt (m/s^2)

        The measurement model is a pose measurement, i.e. o = [x, y, theta].
    '''

    def __init__(self,
                 z_init=np.zeros(5),
                 P_init=np.diag(np.square([XY_STD_ERROR, XY_STD_ERROR, ORI_STD_ERROR, \
                                           VEL_INIT_ERROR, ACC_INIT_ERROR])),
                 Q=np.eye(5),
                 R=np.diag(np.square([XY_STD_ERROR, XY_STD_ERROR, ORI_STD_ERROR]))):
        self.nx = 5     # state dimension

        self.P_init = P_init # state covariance initial value, kept to reset the filter for different tracks.
        self.Q = Q           # disturbance covariance, identified from data
        self.R = R           # measurement noise covariance

        self._reset(z_init)

    def _dynamics_model(self, state):
        ''' Given the current state at timestep t, returns the next state at timestep t+1. '''
        # Note: we assume the vehicle is always moving forward and not in reverse.
        x, y, th, v, acc, dt = state

        staten = np.zeros((self.nx), dtype=state.dtype)

        staten[4] = acc
        staten[3] = max(v + acc*dt, 0.)
        staten[2] = th
        staten[1] = y     + v*np.sin(th)*dt
        staten[0] = x     + v*np.cos(th)*dt

        return staten

    @staticmethod
    def get_init_state(init_pose):
        x_init, y_init, th_init = init_pose
        return np.array([x_init, y_init, th_init, VEL_INIT_GUESS, ACC_INIT_GUESS])


##################################################################
############# Reduced EKF - Zero Acceleration  ###################
##################################################################
class EKFKinematicCVTR(EKFKinematicBase):
    ''' 5-state Kinematic Extended Kalman Filter Constant Velocity + Turn Rate Implementation.

        The kinematic state (z) used here has the following elements:
        x        = z[0], X coordinate (m)
        y        = z[1], Y coordinate (m)
        theta    = z[2], yaw angle (rad) counterclockwise wrt X-axis
        v        = z[3], longitudinal speed (m/s)
        w        = z[4], yaw_rate (rad/s)

        The measurement model is a pose measurement, i.e. o = [x, y, theta].
    '''

    def __init__(self,
                 z_init=np.zeros(5),
                 P_init=np.diag(np.square([XY_STD_ERROR, XY_STD_ERROR, ORI_STD_ERROR, \
                                           VEL_INIT_ERROR, YR_INIT_ERROR])),
                 Q=np.eye(5),
                 R=np.diag(np.square([XY_STD_ERROR, XY_STD_ERROR, ORI_STD_ERROR]))):
        self.nx = 5     # state dimension

        self.P_init = P_init # state covariance initial value, kept to reset the filter for different tracks.
        self.Q = Q           # disturbance covariance, identified from data
        self.R = R           # measurement noise covariance

        self._reset(z_init)

    def _dynamics_model(self, state):
        ''' Given the current state at timestep t, returns the next state at timestep t+1. '''
        # Note: we assume the vehicle is always moving forward and not in reverse.
        x, y, th, v, w, dt = state

        staten = np.zeros((self.nx), dtype=state.dtype)

        staten[4] = w
        staten[3] = max(v, 0.)
        staten[2] = th + w*dt
        staten[1] = y  + v*np.sin(th)*dt
        staten[0] = x  + v*np.cos(th)*dt

        return staten

    @staticmethod
    def get_init_state(init_pose):
        x_init, y_init, th_init = init_pose
        return np.array([x_init, y_init, th_init, VEL_INIT_GUESS, YR_INIT_GUESS])


##################################################################
########### Reduced EKF - Zero Acceleration and Yaw Rate  ########
##################################################################
class EKFKinematicCVH(EKFKinematicBase):
    ''' 4-state Kinematic Extended Kalman Filter Constant Velocity + Heading Implementation.

        The kinematic state (z) used here has the following elements:
        x        = z[0], X coordinate (m)
        y        = z[1], Y coordinate (m)
        theta    = z[2], yaw angle (rad) counterclockwise wrt X-axis
        v        = z[3], longitudinal speed (m/s)

        The measurement model is a pose measurement, i.e. o = [x, y, theta].
    '''

    def __init__(self,
                 z_init=np.zeros(4),
                 P_init=np.diag(np.square([XY_STD_ERROR, XY_STD_ERROR, ORI_STD_ERROR, \
                                           VEL_INIT_ERROR])),
                 Q=np.eye(4),
                 R=np.diag(np.square([XY_STD_ERROR, XY_STD_ERROR, ORI_STD_ERROR]))):
        self.nx = 4     # state dimension

        self.P_init = P_init # state covariance initial value, kept to reset the filter for different tracks.
        self.Q = Q           # disturbance covariance, identified from data
        self.R = R           # measurement noise covariance

        self._reset(z_init)

    def _dynamics_model(self, state):
        ''' Given the current state at timestep t, returns the next state at timestep t+1. '''
        # Note: we assume the vehicle is always moving forward and not in reverse.
        x, y, th, v, dt = state

        staten = np.zeros((self.nx), dtype=state.dtype)

        staten[3] = max(v, 0.)
        staten[2] = th
        staten[1] = y    + v*np.sin(th)*dt
        staten[0] = x    + v*np.cos(th)*dt

        return staten

    @staticmethod
    def get_init_state(init_pose):
        x_init, y_init, th_init = init_pose
        return np.array([x_init, y_init, th_init, VEL_INIT_GUESS])


if __name__ == '__main__':
    test_pose = np.array([1.0, 0.1, -0.001])
    for mdl in [EKFKinematicCATR, EKFKinematicCAH, EKFKinematicCVTR, EKFKinematicCVH]:
        print( issubclass(mdl, EKFKinematicBase) )
        m = mdl()
        print(f"{mdl} with dimension: {m.nx}")
        print(f"Init state for this model:\n{m.get_init_state(test_pose)}")
        print(f"Init covar for this model:\n{m.P_init}")
        print(f"Dist covar for this model:\n{m.Q}")
        print(f"Meas covar for this model:\n{m.R}\n")
