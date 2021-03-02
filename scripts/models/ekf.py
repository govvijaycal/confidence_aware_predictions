import os
import sys
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import pickle

import tensorflow as tf

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)
from datasets.tfrecord_utils import _parse_function

def bound_angle_within_pi(angle):
	# Refer to https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
	return (angle + np.pi) % (2.0 * np.pi) - np.pi 

# Note: Based on Fig. 4 of "Synthetic 2D LIDAR for precise vehicle localization in 3D urban environment", ICRA 2013,
# I am setting a 3sigma bound of 20 cm position error and 1 degree orientation error.
ORI_STD_ERROR   = np.radians(1.) / 3.          # std error of approx. 5.8e-3 radians for yaw angle
POS_STD_ERROR   = 0.2 / 3.                     # std error of approx. 6.7 cm for position (radius of 1-std error circle)
XY_STD_ERROR    = POS_STD_ERROR / np.sqrt(2.)  # std error of approx. 4.7 cm error along x or y axis
VEL_INIT_ERROR  = 1.                           # guesstimate std error (m/s) for initial velocity state
ACC_INIT_ERROR  = 1.                           # guesstimate std error (m/s^2) for initial acceleration state
YR_INIT_ERROR   = 0.5                          # guesstimate std error (rad/s) for initial yawrate state

class EKFKinematicFull():
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
		         R=np.square([XY_STD_ERROR, XY_STD_ERROR, ORI_STD_ERROR])):

		self.nx = 6     # state dimension
	
		self.P_init = P_init # state covariance initial value, kept to reset the filter for different tracks.	
		self.Q = Q           # disturbance covariance, identified from data
		self.R = R           # measurement noise covariance, assuming <=10 cm std error and <= 1 degree std error

		self._reset(z_init)

	def _reset(self, z_init):
		''' Resets the initial state estimate for a new track. '''
		self.P = self.P_init
		self.z = z_init

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
	def time_update(self, dt):
		''' Performs the time/dynamics update step, changing the state estimate (z, P). '''
		state = np.zeros((self.nx+1))
		state[:self.nx] = self.z 
		state[-1] = dt
		self.z = self._dynamics_model(state)
		A      = self._dynamics_jacobian(state)
		self.P = A @ self.P @ A.T + self.Q * dt**2

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
		return residual, res_covar # return for innovation probability calculation

	def _dynamics_model(self, state):
		''' Given the current state at timestep t, returns the next state at timestep t+1. '''
		# Note: we assume the vehicle is always moving forward and not in reverse.
		x, y, th, v, w, acc, dt = state
		
		staten = np.zeros((self.nx), dtype=state.dtype)
		
		staten[5] = acc
		staten[4] = w
		staten[3] = max(v    + acc*dt, 0.)		
		staten[2] = th   + w*dt		
		staten[1] = y    + v*np.sin(th)*dt
		staten[0] = x    + v*np.cos(th)*dt		

		return staten	

	def _dynamics_jacobian(self, state, eps=1e-8):
		''' Numerical jacobian computed via finite differences for the dynamics. '''
		jac = np.zeros((self.nx, self.nx))

		for i in range(self.nx):
			splus  = state + eps * np.array([int(ind==i) for ind in range(self.nx+1)])
			sminus = state - eps * np.array([int(ind==i) for ind in range(self.nx+1)])

			f_plus  = self._dynamics_model(splus)
			f_minus = self._dynamics_model(sminus)

			jac[:,i] = (f_plus - f_minus) / (2.*eps)
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
	def predict(self, dataset):
		predict_dict = {}

		dataset = tf.data.TFRecordDataset(dataset)
		dataset = dataset.map(_parse_function)

		for entry in dataset:
			prior_tms, prior_poses, future_tms  = self.preprocess_entry_prediction(entry)
			prior_dts = np.diff(prior_tms)

			full_prior_states = self.state_completion(prior_tms, prior_poses)
			
			self._reset(full_prior_states[0])

			for dt, next_state in zip(prior_dts, full_prior_states[1:]):
				self.time_update(dt)
				self.measurement_update(next_state[:3])

			future_dts = np.append([future_tms[0]], np.diff(future_tms))
			states = []
			covars = []

			for dt in future_dts:
				self.time_update(dt)
				states.append(self.z)
				covars.append(self.P)

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

	def predict_instance(self, image_raw, past_states):
		raise NotImplementedError

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

	@staticmethod
	def preprocess_entry_prediction(entry):
		""" Prepares a batch of states (past trajectory) and future timestamps given an entry from a TF Dataset.
		    This is meant to predict future states using the predict function.
		"""
		prior_tms = np.array(entry['past_tms'][::-1], dtype=np.float32)
		prior_tms = np.append( prior_tms, np.float32(0.) )

		prior_poses = np.array(entry['past_poses_local'][::-1, :], dtype=np.float32)
		prior_poses = np.concatenate( (prior_poses, np.zeros((1,3), dtype=np.float32)), axis=0)

		future_tms =  np.array(entry['future_tms'], dtype=np.float32)

		return prior_tms, prior_poses, future_tms

	@staticmethod
	def state_completion(tms, poses):
		""" Estimate the velocity, yawrate, and acceleration profile given a pose trajectory. """
		dts = np.diff(tms)
		assert np.all(dts > 1e-3) # make sure dt is positive and not very small

		dposes = np.diff(poses, axis=0)

		""" Step 1: Estimate velocity and acceleration profile. """
		displacements     = np.linalg.norm( dposes[:, :2], axis=-1)
		v_est = displacements / dts
		num_inputs = len(v_est)

		acc_est = np.diff(v_est) / dts[:-1]
		acc_est = np.append(acc_est, acc_est[-1])

		# Step 2: Estimate yawrate profile.
		ang_displacements = dposes[:,2]
		yr_est = ang_displacements / dts
		
		# Handle the final state by just assuming it's the same as the N-1 input for acceleration
		# and yawrate and extrapolating only the velocity state.
		yr_est  = np.append(yr_est, [yr_est[-1]])
		acc_est = np.append(acc_est, acc_est[-1])
		v_est   = np.append(v_est, v_est[-1] + acc_est[-1] * dts[-1])
			
		full_states = np.column_stack((poses, v_est, yr_est, acc_est))

		return full_states

	def fit(self, train_set, val_set, logdir=None, **kwargs):
		""" Identifies Q covariance matrix from the data """
		train_dataset = tf.data.TFRecordDataset(train_set)
		train_dataset = train_dataset.map(_parse_function)		
		Q_trajs = []

		for entry in tqdm(train_dataset):
			tms, poses  = self.preprocess_entry(entry)
			full_states = self.state_completion(tms, poses)

			# Estimate the disturbance covariance for this trajectory.
			# Ignore the initial and final steps of full_states where we have less information about inputs.
			omegas = []
			start_index = np.argwhere(tms == 0.)[0][0] # start at the current state (0)
			end_index   = full_states.shape[0] - 3     # end at the state N-2
			
			dts = np.diff(tms)

			for ind in range(start_index, end_index):
				state_curr = full_states[ind]
				state_next = full_states[ind+1]
				dt = dts[ind]

				state_next_est = self._dynamics_model(np.append(state_curr, [dt]))

				# We divide by dt here since we assume a w_k * dt term
				# appears in the dynamics function.
				omega = (state_next - state_next_est) / dt
				omegas.append(omega)
			Q_trajs.append( np.mean([np.outer(w, w) for w in omegas], axis=0) )

		Q_fit = np.mean(Q_trajs, axis=0)
		self.Q = Q_fit
		print(np.diag(self.Q))

		os.makedirs(logdir, exist_ok=True)
		filename = logdir + 'params.pkl'
		self.save_weights(filename)

class EKFKinematicCAH(EKFKinematicFull):
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
		         R=np.square([XY_STD_ERROR, XY_STD_ERROR, ORI_STD_ERROR])):
		self.nx = 5     # state dimension
	
		self.P_init = P_init # state covariance initial value, kept to reset the filter for different tracks.	
		self.Q = Q           # disturbance covariance, identified from data
		self.R = R           # measurement noise covariance, assuming <=10 cm std error and <= 1 degree std error

		self._reset(z_init)

	def _dynamics_model(self, state):
		''' Given the current state at timestep t, returns the next state at timestep t+1. '''
		# Note: we assume the vehicle is always moving forward and not in reverse.
		x, y, th, v, acc, dt = state
		
		staten = np.zeros((self.nx), dtype=state.dtype)
		
		staten[4] = acc
		staten[3] = max(v    + acc*dt, 0.)		
		staten[2] = th
		staten[1] = y    + v*np.sin(th)*dt
		staten[0] = x    + v*np.cos(th)*dt

		return staten	

	@staticmethod
	def state_completion(tms, poses):
		""" Estimate the velocity and acceleration profile given a pose trajectory. """
		dts = np.diff(tms)
		assert np.all(dts > 1e-3) # make sure dt is positive and not very small

		dposes = np.diff(poses, axis=0)

		""" Step 1: Estimate velocity and acceleration profile. """
		displacements = np.linalg.norm( dposes[:, :2], axis=-1)
		v_est = displacements / dts
		num_inputs = len(v_est)

		acc_est = np.diff(v_est) / dts[:-1]
		acc_est = np.append(acc_est, acc_est[-1])

		# Handle the final state by just assuming it's the same as the N-1 input for acceleration
		# and extrapolating only the velocity state.		
		acc_est = np.append(acc_est, acc_est[-1])
		v_est   = np.append(v_est, v_est[-1] + acc_est[-1] * dts[-1])
			
		full_states = np.column_stack((poses, v_est, acc_est))

		return full_states

class EKFKinematicCVH(EKFKinematicFull):
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
		         R=np.square([XY_STD_ERROR, XY_STD_ERROR, ORI_STD_ERROR])):
		self.nx = 4     # state dimension
	
		self.P_init = P_init # state covariance initial value, kept to reset the filter for different tracks.	
		self.Q = Q           # disturbance covariance, identified from data
		self.R = R           # measurement noise covariance, assuming <=10 cm std error and <= 1 degree std error

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
	def state_completion(tms, poses):
		""" Estimate the velocity profile given a pose trajectory. """
		dts = np.diff(tms)
		assert np.all(dts > 1e-3) # make sure dt is positive and not very small

		dposes = np.diff(poses, axis=0)

		""" Step 1: Estimate velocity and acceleration profile. """
		displacements = np.linalg.norm( dposes[:, :2], axis=-1)
		v_est = displacements / dts
		
		# Handle the final state by just assuming it's the same as the N-1 input for velocity.
		v_est   = np.append(v_est, v_est[-1])
			
		full_states = np.column_stack((poses, v_est))

		return full_states

if __name__ == '__main__':
	for mdl in [EKFKinematicFull, EKFKinematicCAH, EKFKinematicCVH]:
		print( issubclass(mdl, EKFKinematicFull) )
		m = mdl()
		print(f"{mdl} with dimension: {m.nx}")
