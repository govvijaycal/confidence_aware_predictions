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

from models.ekf import bound_angle_within_pi, EKFKinematicBase, \
                       EKFKinematicCATR, EKFKinematicCAH, \
                       EKFKinematicCVTR, EKFKinematicCVH

from models.physics_oracle import constant_velocity_heading, \
                                  constant_velocity_yaw_rate, \
                                  constant_acceleration_heading, \
                                  constant_acceleration_yaw_rate

VEL_RANGE_THRESH = 3.0  # m/s range (~7 mph) for CV vs. CA assignment
YR_MIN_THRESH    = 0.07 # rad/s (~4 deg/sec) rate for CH vs. CTR assignment

class StaticMultipleModel():
    """ Implementation of a Static Multiple Model (Bank of EKF filters).

        Reference: Estimation with Applications to Tracking and Navigation,
                   Bar-Shalom et al., Ch 11.6.2.
    """

    def __init__(self, **kwargs):
        self.filters = [EKFKinematicCVH(),  \
                        EKFKinematicCVTR(), \
                        EKFKinematicCAH(),  \
                        EKFKinematicCATR()]
        self.filter_names = [type(filt).__name__.split('EKFKinematic')[-1] \
                             for filt in self.filters]

    ##################### Weight Utils  ##############################
    ##################################################################
    def save_weights(self, path):
        path = path if '.pkl' in path else (path + '.pkl')
        model_dict = {}

        for filt, filt_name in zip(self.filters, self.filter_names):
            filter_dict = {'P': filt.P_init,
                           'Q': filt.Q,
                           'R': filt.R}

            model_dict[filt_name] = filter_dict

        pickle.dump(model_dict, open(path, 'wb'))

    def load_weights(self, path):
        path = path if '.pkl' in path else (path + '.pkl')
        model_dict = pickle.load(open(path, 'rb'))

        for filt, filt_name in zip(self.filters, self.filter_names):

            if filt_name not in model_dict.keys():
                raise ValueError(f"Could not find the saved EKF model for {filt_name}")

            filt.P_init = model[filt_name]['P']
            filt.Q      = model[filt_name]['Q']
            filt.R      = model[filt_name]['R']

    ##################### Prediction  ################################
    ##################################################################
    @staticmethod
    def compute_residual_log_likelihood(filter_dict, t):
        res   = filter_dict['residual_ms'][t]
        covar = filter_dict['residual_covar_ms'][t]

        # Approach: compute the log-likelihood and then exponentiate the result.
        const_term = res.size / 2. * np.log(2 * np.pi)
        log_det_term = 1./2. * np.log( np.linalg.det(covar) )
        dist_term    = 1./2. * res.T @ np.linalg.pinv(covar) @ res

        rll = -(const_term + log_det_term + dist_term)

        return rll

    @staticmethod
    def compute_mode_probs(filter_dicts):
        ''' Determines the probability of each filter (i.e. mode probability)
            in a static multiple model using recursive updates of the residual
            likelihood.  See p442 of Bar-Shalom for details.
        '''
        n_filters   = len(filter_dicts)
        n_timesteps = len(filter_dicts[0]['residual_ms'])

        mode_probs = np.ones(n_filters) / n_filters

        for t in range(n_timesteps):
            # TODO: remove the asserts.
            assert np.allclose( np.sum(mode_probs), 1. )
            assert np.all( mode_probs >= 0.)
            assert np.all( mode_probs <= 1.)

            rls = np.array([compute_residual_log_likelihood(filt_dict, t) for filt_dict in filter_dicts])
            rls = np.exp(rls)

            next_mode_probs = rlls * mode_probs / np.dot(rll, mode_probs)
            mode_probs = next_mode_probs

        assert np.allclose( np.sum(mode_probs), 1. )
        assert np.all( mode_probs >= 0.)
        assert np.all( mode_probs <= 1.)

        return mode_probs

    def predict(self, dataset):
        ''' Returns a dictionary of predictions given a set of tfrecords. '''
        predict_dict = {}

        dataset = tf.data.TFRecordDataset(dataset)
        dataset = dataset.map(_parse_no_img_function)

        import pdb; pdb.set_trace() # TODO: remove this.
        for ind_entry, entry in enumerate(dataset):
            prior_tms, prior_poses, future_tms  = EKFKinematicBase.preprocess_entry_prediction(entry)

            # Filter the previous pose history.
            filter_dicts = [ filt.filter(prior_tms, prior_poses) for filt in self.filters ]

            # Compute the mode posterior distribution.
            mode_probs   = self.compute_mode_probs(filter_dicts)

            # Predict, per filter, via extrapolation of the underlying motion model.
            gmm_dict = {}
            future_dts = np.append([future_tms[0]], np.diff(future_tms))

            for mode_id, (filt, mode_prob) in enumerate(zip(self.filters, mode_probs)):
                states = []
                covars = []

                for dt in future_dts:
                    z, P, _ = filt.time_update(dt)
                    states.append(z)
                    covars.append(P)

                mode_dict={}
                mode_dict['mode_probability'] = mode_prob
                mode_dict['mus'] = np.array([state[:2] for state in states])
                mode_dict['sigmas'] = np.array([covar[:2, :2] for covar in covars])

                gmm_dict[mode_id] = mode_dict

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
                                 'gmm_pred': gmm_dict}

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
        filter_dicts = [ filt.filter(prior_tms, prior_poses) for filt in self.filters ]

        # Compute the mode posterior distribution.
        mode_probs   = self.compute_mode_probs(filter_dicts)

        # Predict, per filter, via extrapolation of the underlying motion model.
        gmm_pred = {}
        future_dts = np.append([future_tms[0]], np.diff(future_tms))

        for mode_id, (filt, mode_prob) in enumerate(zip(self.filters, mode_probs)):
            states = []
            covars = []

            for dt in future_dts:
                z, P, _ = filt.time_update(dt)
                states.append(z)
                covars.append(P)

            mode_dict={}
            mode_dict['mode_probability'] = mode_prob
            mode_dict['mus'] = np.array([state[:2] for state in states])
            mode_dict['sigmas'] = np.array([covar[:2, :2] for covar in covars])

            gmm_pred[mode_id] = mode_dict

        return gmm_pred

    ##################### Model Fitting  #############################
    ##################################################################
    @staticmethod
    def vel_yawrate_finite_difference(tms, poses):
        dts = np.diff(tms)

        displacements     = np.linalg.norm( np.diff(poses[:, :2], axis=0), axis=-1 )
        ang_displacements = np.diff(poses[:, 2])

        vels      = displacements / dts
        yawrates  = ang_displacements / dts

        return vels, yawrates

    def fit(self, train_set, val_set, logdir=None, **kwargs):
        """ This function fits the underlying process models (EKFs)
            disturbance covariance (Q) based on a heuristic assignment
            and EM algorithm approach.

            Basic Idea:
                - Initialize all filters with Q = I_nx, where nx = state dim of that model.
                - For each dataset instance,
                    - Find the closest process model, measured using residual_log_likelihood
                    - Assign this dataset instance to its "nearest" model
                - For each model,
                    - Fit the covariance Q for this model using EM, but only on the subset of
                      dataset instances assigned to this model.

            We could futher iterate between assignment of dataset instances + fitting the process models,
            but this would take a very long time.  One pass should be enough, assuming changing Q's have
            low impact on the residual log likelihood as compared to the process model itself.
        """

        train_dataset = tf.data.TFRecordDataset(train_set)
        train_dataset = train_dataset.map(_parse_no_img_function)

        cached_train_dataset      = []
        filter_assignment_dataset = []

        for ind_entry, entry in tqdm(enumerate(train_dataset)):

            # Incrementally add the tfrecord processed entries to a cached dataset.
            # Useful since we want to do random access and partition this later on.
            tms, poses = EKFKinematicBase.preprocess_entry(entry)
            cached_train_dataset.append(np.column_stack((tms, poses)))

            # This assignment step is a very simple heuristic.
            # Simply use the variation in velocity (min/max) and the
            # maximum magnitude of the yaw rate to determine if it
            # should be labeled as (1) constant velocity or acceleration,
            # and (2) constant heading or yawrate.
            vels, yrs = self.vel_yawrate_finite_difference(tms, poses)

            model_prefix = 'CA'
            if np.amax(vels) - np.amin(vels) <= VEL_RANGE_THRESH:
                model_prefix = 'CV'

            model_suffix = 'TR'
            if np.amax(np.abs(yrs)) <= YR_MIN_THRESH:
                model_suffix = 'H'

            filter_name = model_prefix + model_suffix
            filter_idx = [idx for (idx, x) in enumerate(self.filter_names) \
                          if x == filter_name]

            assert len(filter_idx) == 1
            filter_assignment_dataset.append(filter_idx[0])

        cached_train_dataset      = np.array(cached_train_dataset)
        filter_assignment_dataset = np.array(filter_assignment_dataset)
        del train_dataset

        for ind_filt, (filt, filt_name) in enumerate(zip(self.filters, self.filter_names)):
            # Train each filter on the partition of the full dataset identified in the first pass.
            print(f"Training Filter: {filt_name} on Dataset Partition: "
                  f"{np.sum(filter_assignment_dataset == ind_filt)} of "
                  f"{len(filter_assignment_dataset)} instances")

            part_dataset = cached_train_dataset[filter_assignment_dataset == ind_filt]
            filt.fit(part_dataset, val_set, logdir = None)

        os.makedirs(logdir, exist_ok=True)
        filename = logdir + 'params.pkl'
        self.save_weights(filename)

if __name__ == '__main__':
    smm = StaticMultipleModel()

    for filt in smm.filters:
        print( issubclass(type(filt), EKFKinematicBase) )
        print(f"{type(filt)} with dimension: {filt.nx}")
