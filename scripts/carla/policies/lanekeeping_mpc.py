"""
Frenet Nonlinear Kinematic MPC Module.
  Follows a specified reference speed (self.v_ref)
  subject to actuation constraints (with allowable emergency braking)
  while trying (with softened constraints) to remain
  within lane bounds (e_y) and safe intervals (s).
"""

import time
import casadi
import numpy as np

class LanekeepingMPC:
	##
	def __init__(self,
		         N          = 10,            # timesteps in MPC Horizon
		         DT       	= 0.2,           # discretization time between timesteps (s)
		         DT_CTRL    = 0.05,          # control input period (s)
		         L_F        = 1.5213,        # distance from CoG to front axle (m)
		         L_R        = 1.4987,        # distance from CoG to rear axle (m)
				 A_MIN     = -3.0,           # min/max longitudinal acceleration constraint (m/s^2)
				 A_MAX     =  2.0,
		         DF_MIN     = -0.5,          # min/max front steer angle constraint (rad)
		         DF_MAX     =  0.5,
		         A_DOT_MIN  = -1.5,          # min/max longitudinal jerk constraint (m/s^3)
		         A_DOT_MAX  =  1.5,
		         DF_DOT_MIN = -0.5,          # min/max front steer angle rate constraint (rad/s)
		         DF_DOT_MAX =  0.5,
		         EY_MIN     = -0.8,          # min/max lateral error (m)
		         EY_MAX     =  0.8,
		         EPSI_MIN   = -0.75,
		         EPSI_MAX   =  0.75,
		         V_MIN      =  0.0,          # min/max speed (m/s)
		         V_MAX      = 20.0,
				 Q = [0., 1., 100., 0.1],     # weights on s, ey, epsi, v
				 R = [10., 1000.]):          # input rate weights on ax, df

		for key in list(locals()):
			if key == 'self':
				pass
			elif key == 'Q':
				self.Q = casadi.diag(Q)
			elif key == 'R':
				self.R = casadi.diag(R)
			else:
				setattr(self, '%s' % key, locals()[key])

		self.opti = casadi.Opti()

		'''
		(1) Parameters
		'''
		self.u_prev  = self.opti.parameter(2) # previous input: [u_{acc, -1}, u_{df, -1}]
		self.z_curr  = self.opti.parameter(4) # current state:  [s0, ey_0, epsi_0, v_0]

		# Velocity and Curvature Profile.
		self.curv_ref_params = self.opti.parameter(2) # (m,b) for a linear interpolation s.t. curv(s) = m * s + b
		self.v_ref           = self.opti.parameter()  # constant speed reference
		self.z_ref           = casadi.horzcat(casadi.MX.zeros(1,3), self.v_ref)

		# Constraints on "progress" variable s.  These intervals define free space for the EV from timestep 1 onwards,
		# i.e.  s_lb[0] <= s_dv[1] <= s_ub[0] - so it's one ahead in index compared to s_dv below.
		self.s_lb = self.opti.parameter(self.N)
		self.s_ub = self.opti.parameter(self.N)

		'''
		(2) Decision Variables
		'''
		## First index is the timestep k, i.e. self.z_dv[0,:] is z_0.
		## It has self.N+1 timesteps since we go from z_0, ..., z_N
		## Second index is the state element, as detailed below.
		self.z_dv = self.opti.variable(self.N+1, 4) # s, ey, epsi, v

		self.s_dv    = self.z_dv[:, 0]
		self.ey_dv   = self.z_dv[:, 1]
		self.epsi_dv = self.z_dv[:, 2]
		self.v_dv    = self.z_dv[:, 3]

		## Control inputs used to achieve self.z_dv according to dynamics.
		## First index is the timestep k, i.e. self.u_dv[0,:] is u_0.
		## Second index is the input element as detailed below.
		self.u_dv = self.opti.variable(self.N, 2)

		self.acc_dv = self.u_dv[:,0]
		self.df_dv  = self.u_dv[:,1]

		self.slacks           = self.opti.variable(5)
		self.slack_ey         = self.slacks[0]
		self.slack_epsi       = self.slacks[1]
		self.slack_decel_jerk = self.slacks[2]
		self.slack_s          = self.slacks[3]
		self.slack_slew_rate  = self.slacks[4]

		'''
		(3) Problem Setup: Constraints, Cost, Initial Solve
		'''
		self._add_constraints()
		self._add_interval_constraints()

		self._add_cost()

		self._update_initial_condition(0., 0., 0., 1.)

		self._update_reference([0., 0.], 5.)

		self._update_previous_input(0., 0.)
		self._update_interval_constraints([0.]*self.N,
			                              [1000.]*self.N )

		# Ipopt with custom options: https://web.casadi.org/docs/ -> see sec 9.1 on Opti stack.
		p_opts = {'expand': True} # http://casadi.sourceforge.net/api/internal/d4/d89/group__nlpsol.html
		s_opts = {'print_level': 0} # https://coin-or.github.io/Ipopt/OPTIONS.html
		self.opti.solver('ipopt', p_opts, s_opts)
		sol = self.solve()

	def _add_constraints(self):
		## State Bound Constraints
		self.opti.subject_to( self.opti.bounded(self.EY_MIN - self.slack_ey, self.ey_dv, self.EY_MAX + self.slack_ey) )
		self.opti.subject_to( self.opti.bounded(self.EPSI_MIN - self.slack_epsi, self.epsi_dv, self.EPSI_MAX + self.slack_epsi))
		self.opti.subject_to( self.opti.bounded( self.V_MIN,  self.v_dv, self.V_MAX) )

		## Initial State Constraint
		self.opti.subject_to( self.s_dv[0]    == self.z_curr[0] )
		self.opti.subject_to( self.ey_dv[0]   == self.z_curr[1] )
		self.opti.subject_to( self.epsi_dv[0] == self.z_curr[2] )
		self.opti.subject_to( self.v_dv[0]    == self.z_curr[3] )

		## State Dynamics Constraints
		for i in range(self.N):
			beta   = casadi.atan( self.L_R / (self.L_F + self.L_R) * casadi.tan(self.df_dv[i]) )
			dyawdt = self.v_dv[i] / self.L_R * casadi.sin(beta)
			curv   = self.curv_ref_params[0] * self.s_dv[i] + self.curv_ref_params[1]
			dsdt   = self.v_dv[i] * casadi.cos(self.epsi_dv[i]+beta) / (1 - self.ey_dv[i] * curv )

			self.opti.subject_to( self.s_dv[i+1]    == self.s_dv[i]    + self.DT * (dsdt) )
			self.opti.subject_to( self.ey_dv[i+1]   == self.ey_dv[i]   + self.DT * (self.v_dv[i] * casadi.sin(self.epsi_dv[i] + beta)) )
			self.opti.subject_to( self.epsi_dv[i+1] == self.epsi_dv[i] + self.DT * (dyawdt - dsdt * curv) )
			self.opti.subject_to( self.v_dv[i+1]    == self.v_dv[i]    + self.DT * (self.acc_dv[i]) )

		## Input Bound Constraints
		self.opti.subject_to( self.opti.bounded(self.A_MIN, self.acc_dv, self.A_MAX) )
		self.opti.subject_to( self.opti.bounded(self.DF_MIN, self.df_dv,  self.DF_MAX) )

		# Input Rate Bound Constraints
		self.opti.subject_to( self.opti.bounded( self.A_DOT_MIN*self.DT_CTRL - self.slack_decel_jerk,
			                                     self.acc_dv[0] - self.u_prev[0],
			                                     self.A_DOT_MAX*self.DT_CTRL) )

		self.opti.subject_to( self.opti.bounded( self.DF_DOT_MIN*self.DT_CTRL - self.slack_slew_rate,
			                                     self.df_dv[0] - self.u_prev[1],
			                                     self.DF_DOT_MAX*self.DT_CTRL + self.slack_slew_rate) )

		for i in range(self.N - 1):
			self.opti.subject_to( self.opti.bounded( self.A_DOT_MIN*self.DT,
				                                     self.acc_dv[i+1] - self.acc_dv[i],
				                                     self.A_DOT_MAX*self.DT) )
			self.opti.subject_to( self.opti.bounded( self.DF_DOT_MIN*self.DT,
				                                     self.df_dv[i+1]  - self.df_dv[i],
				                                     self.DF_DOT_MAX*self.DT) )
		# Other Constraints
		self.opti.subject_to( 0 <= self.slacks )

	def _add_interval_constraints(self):
		for i in range(self.N):
			self.opti.subject_to( self.opti.bounded( self.s_lb[i] - self.slack_s,
				                                     self.s_dv[i+1],
				                                     self.s_ub[i] + self.slack_s) )

	@staticmethod
	def _quad_form(z, Q):
		return casadi.mtimes(z, casadi.mtimes(Q, z.T))

	## Cost function
	def _add_cost(self):
		cost = 0
		for i in range(self.N):
			cost += self._quad_form(self.z_dv[i+1, :]-self.z_ref, self.Q)

		for i in range(self.N - 1):
			cost += self._quad_form(self.u_dv[i+1, :] - self.u_dv[i,:], self.R)

		cost += (1e2  * self.slack_decel_jerk)
		cost += (1e3  * self.slack_slew_rate)
		cost += (1e6  * self.slack_ey)
		cost += (1e9  * self.slack_epsi)
		cost += (1e3  * self.slack_s)

		self.opti.minimize( cost )

	def solve(self):
		st = time.time()
		try:
			sol = self.opti.solve()
			# Optimal solution.
			u_mpc    = sol.value(self.u_dv)
			z_mpc    = sol.value(self.z_dv)
			is_opt = True
		except:
			# Suboptimal solution (e.g. timed out).
			u_mpc    = self.opti.debug.value(self.u_dv)
			z_mpc    = self.opti.debug.value(self.z_dv)
			is_opt   = False

		solve_time = time.time() - st

		sol_dict = {}
		sol_dict['u_control']    = u_mpc[0,:]      # control input to apply based on solution
		sol_dict['optimal']      = is_opt          # whether the solution is optimal or not
		sol_dict['solve_time']   = solve_time      # how long the solver took in seconds
		sol_dict['u_mpc']        = u_mpc           # solution inputs (N by 2, see self.u_dv above)
		sol_dict['z_mpc']        = z_mpc           # solution states (N+1 by 4, see self.z_dv above)

		# NOTE: the notion of is_opt is a bit fuzzy here since we are dealing with soft constraints.
		# It could be compared with the hard constraints by checking the slack variables.
		# For now, this will be ignored -> focus will be on the ensuing motion rather than optimality/feasibility.

		return sol_dict

	def get_global_trajectory(self, u_mpc, z_global_init):
		z_global = np.ones((self.N+1, 4)) * np.nan
		z_global[0, :] = z_global_init

		for ind, u in enumerate(u_mpc):
			x, y, p, v = z_global[ind, :]
			u_acc, u_df = u

			beta = np.arctan( self.L_R / (self.L_F + self.L_R) * np.tan(u_df) )

			xn = x + self.DT * (v * np.cos(p + beta))
			yn = y + self.DT * (v * np.sin(p + beta))
			pn = p + self.DT * (v / self.L_R * np.sin(beta))
			vn = v + self.DT * (u_acc)

			z_global[ind+1, :] = [xn, yn, pn, vn]

		return z_global

	def update(self, update_dict):
		self._update_initial_condition(*[update_dict[key] for key in ['s', 'ey', 'epsi', 'v']])
		self._update_reference(*[update_dict[key] for key in ['curv_lin_fit', 'v_ref']])
		self._update_previous_input(*[update_dict[key] for key in ['acc_prev', 'df_prev']])
		self._update_interval_constraints(*[update_dict[key] for key in ['s_lb', 's_ub']])

	def _update_initial_condition(self, s0, ey0, epsi0, vel0):
		self.opti.set_value(self.z_curr, [s0, ey0, epsi0, vel0])

	def _update_reference(self, curv_ref_linear_fit, v_ref):
		self.opti.set_value(self.curv_ref_params, curv_ref_linear_fit)
		self.opti.set_value(self.v_ref, v_ref)

	def _update_previous_input(self, acc_prev, df_prev):
		self.opti.set_value(self.u_prev, [acc_prev, df_prev])

	def _update_interval_constraints(self, s_lb, s_ub):
		self.opti.set_value(self.s_lb, s_lb)
		self.opti.set_value(self.s_ub, s_ub)

if __name__ == "__main__":
	lk_mpc = LanekeepingMPC()