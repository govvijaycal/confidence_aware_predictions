import numpy as np

class LaneEKF():
    def __init__(self, Q_u, R_lane_frame):
        # Kinematic state distribution
        # We deal with a 4-state: [x, y, theta, v] and
        # 2-input: [u_acc, u_curv] model.

        self.nx = 4 # state dimension
        self.nu = 2 # input dimension
        self.nm = 3 # measurement dimension

        self.z = np.zeros(self.nx) # mean
        self.P = np.eye(self.nx)   # covariance

        # Input covariance
        assert Q_u.shape == (self.nu, self.nu)
        assert np.allclose(Q_u, Q_u.T)
        assert np.linalg.det(Q_u) > 0.
        self.Q_u = Q_u

        # Measurement covariance (lane waypoint ID)
        assert R_lane_frame.shape == (self.nm, self.nm)
        assert np.allclose(R_lane_frame, R_lane_frame.T)
        assert np.linalg.det(R_lane_frame) > 0.
        self.R_lane_frame = R_lane_frame

    def time_update(self, u, dt):
        A = self._dynamics_state_jacobian(self.z, u, dt)
        B = self._dynamics_input_jacobian(self.z, u, dt)
        z_next = self._dynamics_model(self.z, u, dt)

        self.z = z_next
        self.P = A @ self.P @ A.T + B @ self.Q_u @ B.T

        return self.z, self.P, A, B

    def measurement_update(self, lane_localizer):
        x = self.z[0]
        y = self.z[1]
        lane_pose, rot_local_to_global = lane_localizer.get_lane_measurement(x, y)

        H = np.zeros((self.nm, self.nx))
        H[:, :self.nm] = np.eye(self.nm)

        residual = lane_pose - H @ self.z
        residual[2] = self._bound_angle_within_pi(residual[2])

        R_lane_global = np.copy(self.R_lane_frame)
        R_lane_global[:2, :2] = rot_local_to_global @ R_lane_global[:2, :2] @ rot_local_to_global.T

        res_covar = H @ self.P @ H.T + R_lane_global
        K         = self.P @ H.T @ np.linalg.pinv(res_covar)

        self.z    = self.z + K @ residual
        self.z[2] = self._bound_angle_within_pi(self.z[2])
        self.P    = (np.eye(self.nx) - K @ H) @ self.P

        return self.z, self.P, residual, res_covar

    def _reset(self, z_init, P_init):
        assert z_init.shape == (self.nx,)
        assert P_init.shape == (self.nx, self.nx)
        self.z = z_init
        self.P = P_init

    @staticmethod
    def _bound_angle_within_pi(angle):
        """ Given an angle, adjusts it to lie within a +/- PI range """
        return (angle + np.pi) % (2 * np.pi) - np.pi # https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap

    @staticmethod
    def _dynamics_model(z, u, dt):
        u_acc, u_curv = u
        x, y, th, v   = z

        xn  = x  + dt*(v * np.cos(th))
        yn  = y  + dt*(v * np.sin(th))
        thn = th + dt*(v*u_curv)
        vn  = v  + dt*(u_acc)

        vn = max(0, vn)
        thn = LaneEKF._bound_angle_within_pi(thn)

        return np.array([xn, yn, thn, vn])

    @staticmethod
    def _dynamics_state_jacobian(z, u, dt):
        u_acc, u_curv  = u
        x, y, th, v    = z

        A = np.eye(4) + dt * \
            np.array([[0, 0, -v*np.sin(th), np.cos(th)],
                      [0, 0,  v*np.cos(th), np.sin(th)],
                      [0, 0,             0,     u_curv],
                      [0, 0,             0,          0]])
        return A

    @staticmethod
    def _dynamics_input_jacobian(z, u, dt):
        u_acc, u_curv  = u
        x, y, th, v    = z

        B = np.array([[ 0,       0],
                      [ 0,       0],
                      [ 0,    v*dt],
                      [dt,       0]])
        return B

if __name__ == '__main__':
    nz, nu = 4, 2
    z  = np.array([10., 5., 0.25, 8.0])
    u  = np.array([1.2, -0.2])
    dt = 0.1
    eps = 1e-4

    # TEST STATE JACOBIAN
    A = LaneEKF._dynamics_state_jacobian(z, u, dt)
    A_num_jac = np.zeros((nz, nz))
    for i in range(nz):
        z_plus  = z + eps * np.array([int(ind==i) for ind in range(nz)])
        z_minus = z - eps * np.array([int(ind==i) for ind in range(nz)])

        f_plus  = LaneEKF._dynamics_model(z_plus, u, dt)
        f_minus = LaneEKF._dynamics_model(z_minus, u, dt)
        diff    = f_plus - f_minus
        diff[2] = LaneEKF._bound_angle_within_pi(diff[2])

        A_num_jac[:, i] = diff/(2.*eps)

    print("STATE JACOBIAN")
    print(A)
    print(f"STATE JACOBIAN ERROR: {np.linalg.norm(A - A_num_jac, ord=np.inf)}")


    # TEST INPUT JACOBIAN
    B = LaneEKF._dynamics_input_jacobian(z, u, dt)
    B_num_jac = np.zeros((nz, nu))
    for i in range(nu):
        u_plus  = u + eps * np.array([int(ind==i) for ind in range(nu)])
        u_minus = u - eps * np.array([int(ind==i) for ind in range(nu)])

        f_plus  = LaneEKF._dynamics_model(z, u_plus, dt)
        f_minus = LaneEKF._dynamics_model(z, u_minus, dt)
        diff    = f_plus - f_minus
        diff[2] = LaneEKF._bound_angle_within_pi(diff[2])

        B_num_jac[:, i] = diff/(2.*eps)

    print("INPUT JACOBIAN")
    print(B)
    print(f"INPUT JACOBIAN ERROR: {np.linalg.norm(B - B_num_jac, ord=np.inf)}")