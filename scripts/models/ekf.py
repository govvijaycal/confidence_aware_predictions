import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# TODO: move to a common utils folder.
def bound_angle_within_pi(angle):
	# Refer to https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
	return (angle + np.pi) % (2.0 * np.pi) - np.pi 

def full_kinematic_model(state, dt):
	x, y, th, v, curv, acc, curv_dot = state
	staten = np.zeros_like(state)

	staten[0] = x    + v * np.cos(th) * dt
	staten[1] = y    + v * np.sin(th) * dt
	staten[2] = th   + v * curv * dt  # TODO: need to wrap angle
	staten[3] = v    + acc*dt
	staten[4] = curv + curv_dot * dt
	staten[5] = acc
	staten[6] = curv_dot

	return staten		

def full_kinematic_jacobian(state, dt):
	return np.array([[1., 0., -v*np.sin(th)*dt, np.cos(th)*dt,   0., 0., 0.],
		             [0., 1.,  v*np.cos(th)*dt, np.sin(th)*dt,   0., 0., 0.],
		             [0., 0.,                1.,      curv*dt, v*dt, 0., 0.],
		             [0., 0.,                0.,            1.,  0., dt, 0.],
		             [0., 0.,                0.,            0.,  1., 0., dt],
		             [0., 0.,                0.,            0.,  0., 1., 0.],
		             [0., 0.,                0.,            0.,  0., 0., 1.]])

class KinematicEKFBase(ABC):
	''' Base Class for Kinematic Extended Kalman Filter Implementations.

	    The kinematic state (z) used here has the following elements:
	    x        = z[0], X coordinate (m)
	    y        = z[1], Y coordinate (m)
	    theta    = z[2], yaw angle (rad) counterclockwise wrt X-axis
	    v        = z[3], longitudinal speed (m/s)
	    curv     = z[4], curvature (rad/m)
	    acc      = z[5], longitudinal acceleration, i.e. dv/dt (m/s^2)
	    curv_dot = z[6]. time derivative of curvature, i.e. dcurv/dt (rad / m *s )
	     '''

	def __init__(self,
		         z_init,
		         P_init,
		         Q,
		         R,
		         dt):
		self.nx = 7     # state dimension
		self.z = z_init # state estimate
		self.P = P_init # state estimate covariance
		self.Q = Q      # disturbance covariance
		self.R = R      # measurement noise covariance
		self.dt = dt    # discretization time (s) between timesteps

	def time_update(self):
		''' Performs the time/dynamics update step, changing the state estimate (z, P). '''
		self.z = self._dynamics_model(self.z)		
		A      = self._dynamics_jacobian(self.z)
		self.P = A @ self.P @ A.T + self.Q

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

	def _obs_model(self, state, measurement):
		''' Returns the residual using the state estimate, measurement, and observation model. '''
		expected    = self._obs_jacobian(state) @ state # linear measurement model
		residual    = measurement - expected
		residual[2] = bound_angle_within_pi(residual[2])
		return residual

	def _obs_jacobian(self, state):
		''' Returns the observation jacobian given the current state. '''
		H        = np.zeros((3, len(state))) # n_obs by n_x
		H[:, :3] = np.eye(3)
		return H
	
	@abstractmethod
	def _dynamics_model(self, state):
		''' Given the current state at timestep t, returns the next state at timestep t+1. '''
		raise NotImplementedError

	@abstractmethod
	def _dynamics_jacobian(self, state):
		''' Returns the jacobian of the dynamics_model, computed at the current state. '''
		raise NotImplementedError

	def _dynamics_jacobian_num(self, state, eps=1e-8):
		# Jacobian of motion model with finite differences (for gradient checking).
		jac = np.zeros([self.nx, self.nx])
		
		for i in range(self.nx):
			stateplus  = state + eps * np.array([int(ind==i) for ind in range(self.nx)])
			stateminus = state - eps * np.array([int(ind==i) for ind in range(self.nx)])

			f_plus  = self._dynamics_model(stateplus)
			f_minus = self._dynamics_model(stateminus)

			jac[:,i] = (f_plus - f_minus) / (2.*eps)
		return jac 

''' 
Straight Modes 
'''
class KinematicEKF_Stationary(KinematicEKFBase):
	def _dynamics_model(self, state):		
		staten = np.zeros_like(state)
		staten[:3] = state[:3]
		return staten

	def _dynamics_jacobian(self, state):
		A = np.zeros((self.nx, self.nx))
		A[:3, :3] = np.eye(3)
		return A

class KinematicEKF_ConstantVelocityHeading(KinematicEKFBase):
	def _dynamics_model(self, state):		
		staten = np.zeros_like(state)

		x, y, th, v, _, _, _ = state
		dt = self.dt
		staten[0] = x + v*np.cos(th)*dt
		staten[1] = y + v*np.sin(th)*dt
		staten[2] = th
		staten[3] = v

		return staten

	def _dynamics_jacobian(self, state):		
		A = np.zeros((self.nx, self.nx))
		x, y, th, v, _, _, _ = state
		dt = self.dt
		A[:4, :4] = np.array([[1., 0., -v*np.sin(th)*dt, np.cos(th)*dt],
			                  [0., 1.,  v*np.cos(th)*dt, np.sin(th)*dt],
			                  [0., 0.,               1.,            0.],
			                  [0., 0.,               0.,            1.]])
		return A

class KinematicEKF_ConstantAccelerationHeading(KinematicEKFBase):
	def _dynamics_model(self, state):		
		staten = np.zeros_like(state)

		x, y, th, v, _, acc, _ = state
		dt = self.dt
		staten[0] = x + v*np.cos(th)*dt
		staten[1] = y + v*np.sin(th)*dt
		staten[2] = th
		staten[3] = v + acc*dt
		staten[5] = acc

		return staten

	def _dynamics_jacobian(self, state):	
		x, y, th, v, _, acc, _ = state
		dt = self.dt	
		return np.array([[1., 0., -v*np.sin(th)*dt, np.cos(th)*dt,   0., 0., 0.],
			             [0., 1.,  v*np.cos(th)*dt, np.sin(th)*dt,   0., 0., 0.],
			             [0., 0.,                1.,            0.,  0., 0., 0.],
			             [0., 0.,                0.,            1.,  0., dt, 0.],
			             [0., 0.,                0.,            0.,  0., 0., 0.],
			             [0., 0.,                0.,            0.,  0., 1., 0.],
			             [0., 0.,                0.,            0.,  0., 0., 0.]])

''' 
Constant Curvature Modes
'''
class KinematicEKF_ConstantVelocityCurvature(KinematicEKFBase):
	def _dynamics_model(self, state):
		x, y, th, v, curv, _,_ = state
		dt = self.dt
		staten = np.zeros_like(state)

		staten[0] = x    + v*np.cos(th)*dt
		staten[1] = y    + v*np.sin(th)*dt
		staten[2] = bound_angle_within_pi(th   + v*curv*dt)
		staten[3] = v
		staten[4] = curv
		staten[5] = 0
		staten[6] = 0

		return staten		

	def _dynamics_jacobian(self, state):
		x, y, th, v, curv, _, _ = state
		dt = self.dt
		return np.array([[1., 0., -v*np.sin(th)*dt, np.cos(th)*dt,   0., 0., 0.],
			             [0., 1.,  v*np.cos(th)*dt, np.sin(th)*dt,   0., 0., 0.],
			             [0., 0.,                1.,      curv*dt, v*dt, 0., 0.],
			             [0., 0.,                0.,            1.,  0., 0., 0.],
			             [0., 0.,                0.,            0.,  1., 0., 0.],
			             [0., 0.,                0.,            0.,  0., 0., 0.],
			             [0., 0.,                0.,            0.,  0., 0., 0.]])

class KinematicEKF_ConstantAccelerationCurvature(KinematicEKFBase):
	def _dynamics_model(self, state):
		x, y, th, v, curv, acc, _ = state
		dt = self.dt
		staten = np.zeros_like(state)

		staten[0] = x    + v*np.cos(th)*dt
		staten[1] = y    + v*np.sin(th)*dt
		staten[2] = bound_angle_within_pi(th   + v*curv*dt)
		staten[3] = v    + acc*dt
		staten[4] = curv
		staten[5] = acc
		staten[6] = 0

		return staten		

	def _dynamics_jacobian(self, state):
		x, y, th, v, curv, acc, _ = state
		dt = self.dt
		return np.array([[1., 0., -v*np.sin(th)*dt, np.cos(th)*dt,   0., 0., 0.],
			             [0., 1.,  v*np.cos(th)*dt, np.sin(th)*dt,   0., 0., 0.],
			             [0., 0.,                1.,      curv*dt, v*dt, 0., 0.],
			             [0., 0.,                0.,            1.,  0., dt, 0.],
			             [0., 0.,                0.,            0.,  1., 0., 0.],
			             [0., 0.,                0.,            0.,  0., 1., 0.],
			             [0., 0.,                0.,            0.,  0., 0., 0.]])

'''
Constant Curvature Rate Modes
'''
class KinematicEKF_ConstantVelocityCurvatureRate(KinematicEKFBase):
	def _dynamics_model(self, state):
		x, y, th, v, curv, _, curv_dot = state
		staten = np.zeros_like(state)

		staten[0] = x    + v*np.cos(th)*dt
		staten[1] = y    + v*np.sin(th)*dt
		staten[2] = bound_angle_within_pi(th   + v*curv*dt)
		staten[3] = v  
		staten[4] = curv + curv_dot * dt
		staten[5] = 0
		staten[6] = curv_dot

		return staten		

	def _dynamics_jacobian(self, state):
		x, y, th, v, curv, _, curv_dot = state
		dt = self.dt
		return np.array([[1., 0., -v*np.sin(th)*dt, np.cos(th)*dt,   0., 0., 0.],
			             [0., 1.,  v*np.cos(th)*dt, np.sin(th)*dt,   0., 0., 0.],
			             [0., 0.,                1.,      curv*dt, v*dt, 0., 0.],
			             [0., 0.,                0.,            1.,  0., 0,  0.],
			             [0., 0.,                0.,            0.,  1., 0., dt],
			             [0., 0.,                0.,            0.,  0., 0., 0.],
			             [0., 0.,                0.,            0.,  0., 0., 1.]])


class KinematicEKF_ConstantAccelerationCurvatureRate(KinematicEKFBase):
	def _dynamics_model(self, state):
		x, y, th, v, curv, acc, curv_dot = state
		dt = self.dt
		staten = np.zeros_like(state)

		staten[0] = x    + v*np.cos(th)*dt
		staten[1] = y    + v*np.sin(th)*dt
		staten[2] = bound_angle_within_pi(th   + v*curv*dt)
		staten[3] = v    + acc*dt
		staten[4] = curv + curv_dot * dt
		staten[5] = acc
		staten[6] = curv_dot

		return staten		

	def _dynamics_jacobian(self, state):
		x, y, th, v, curv, acc, curv_dot = state
		dt = self.dt
		return np.array([[1., 0., -v*np.sin(th)*dt, np.cos(th)*dt,   0., 0., 0.],
			             [0., 1.,  v*np.cos(th)*dt, np.sin(th)*dt,   0., 0., 0.],
			             [0., 0.,                1.,      curv*dt, v*dt, 0., 0.],
			             [0., 0.,                0.,            1.,  0., dt, 0.],
			             [0., 0.,                0.,            0.,  1., 0., dt],
			             [0., 0.,                0.,            0.,  0., 1., 0.],
			             [0., 0.,                0.,            0.,  0., 0., 1.]])


if __name__ == '__main__':
	class_prefix = 'KinematicEKF_'
	class_suffix_list = ['Stationary', 'ConstantVelocityHeading', 'ConstantAccelerationHeading', \
	                     'ConstantVelocityCurvature', 'ConstantAccelerationCurvature', \
	                     'ConstantVelocityCurvatureRate', 'ConstantAccelerationCurvatureRate']

	z_init = np.random.random(7) * [50., 50., np.pi, 10., 0.05, 3.0, 0.001]
	measurement = z_init[:3] + np.random.normal(size = 3)
	P_init = np.eye(7)
	Q      = np.diag([1., 1., 0.1, 5., 0.01, 1., 0.01])
	R      = np.diag([1., 1., 0.1])
	dt = 0.2	

	for suffix in class_suffix_list:
		classname = class_prefix + suffix
		print(classname)

		ekf = globals()[classname](z_init, P_init, Q, R, dt)			
		ekf.time_update()
		ekf.measurement_update(measurement)	
	
		print('\tTime and Measurement Updates Ran Successfully.')


		diffs = []
		for i in range(100):
			z_test = np.random.random(7) * [100., 100., 100, 100., 100., 100., 100.]
			jac_computed =  ekf._dynamics_jacobian(z_test)
			jac_numerical = ekf._dynamics_jacobian_num(z_test, eps=1e-6)
			diffs.append( np.linalg.norm(jac_computed - jac_numerical, ord =np.inf) )
		print('\tDynamics Jacobian Norm Difference: ', max(diffs) )
