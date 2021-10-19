import numpy as np
from scipy.signal import filtfilt

def fix_angle( angle ):
	""" Given an angle, adjusts it to lie within a +/- PI range """
	return (angle + np.pi) % (2 * np.pi) - np.pi # https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap

def extract_path_from_waypoints( waypoints ):
	""" Given a list of Carla waypoints, returns the corresponding path (x, y, yaw) in global frame """

	# Pose extraction
	extract_xy_arr  = lambda waypoints: np.array( [[way[0].transform.location.x, -way[0].transform.location.y]for way in waypoints] )
	extract_yaw_arr = lambda waypoints: np.array( [[-fix_angle(np.radians(way[0].transform.rotation.yaw))] for way in waypoints] )

	way_xy  = extract_xy_arr(waypoints)
	way_yaw = np.ravel( extract_yaw_arr(waypoints) )
	diff_xy = np.diff(way_xy, axis=0)
	way_s   = np.cumsum( np.sqrt( np.sum( np.square(diff_xy), axis=1) ) )
	way_s   = np.insert( way_s, 0, [0.0] )

	return way_s, way_xy, way_yaw  # s, x, y, yaw

class FrenetTrajectoryHandler(object):
	def __init__(self, way_s, way_xy, way_yaw, s_resolution=0.5):
		self.update(way_s, way_xy, way_yaw, s_resolution=s_resolution)
		self.get_curvatures_at_s = np.vectorize(self.get_curvature_at_s)

	def update(self, way_s, way_xy, way_yaw, s_resolution=0.5):
		s_frenet, x_frenet, y_frenet, yaw_frenet, curv_frenet = \
		    self._generate_frenet_reference_trajectory(way_s, way_xy, way_yaw, s_resolution=s_resolution)

		self.trajectory = np.column_stack((s_frenet, x_frenet, y_frenet, yaw_frenet, curv_frenet))

	def reached_trajectory_end(self, s_query, resolution=2.):
		return np.abs(self.trajectory[-1, 0] - s_query) < resolution

	def _generate_frenet_reference_trajectory(self, way_s, way_xy, way_yaw, s_resolution=0.5):
		"""
		Returns an interpolated trajectory x(s), y(s), yaw(s), curvature(s) with resolution
		given by s_resolution.  Here s is the arclength (meters).
		"""
		s_frenet = np.arange(way_s[0], way_s[-1] + s_resolution/2, s_resolution)
		x_frenet = np.interp(s_frenet, way_s, way_xy[:,0])
		y_frenet = np.interp(s_frenet, way_s, way_xy[:,1])

		yaw_frenet = np.interp(s_frenet, way_s, np.unwrap(way_yaw))        # unwrap to avoid jumps when interpolating.
		curv_frenet_raw = np.diff(yaw_frenet) / np.diff(s_frenet)          # hope this stays low due to use of unwrap above
		if len(curv_frenet_raw) > 10:
			curv_frenet = filtfilt(np.ones((3,))/3, 1, curv_frenet_raw)   # curvature filtering
		else:
			curv_frenet = curv_frenet_raw
		curv_frenet = np.insert(curv_frenet, len(curv_frenet), 0.0)

		yaw_frenet = np.array([fix_angle(ang) for ang in yaw_frenet])

		return s_frenet, x_frenet, y_frenet, yaw_frenet, curv_frenet

	def convert_global_to_frenet_frame(self, x_query, y_query, psi_query):
		xy_traj = self.trajectory[:,1:3] # N by 2
		xy_query = np.array([[x_query, y_query]]) # 1 by 2

		closest_index = np.argmin( np.linalg.norm(xy_traj - xy_query, axis=1) )

		# Note: Can do some smarter things here, like linear interpolation.
		# If s_K+1 - s_k is reasonably small, we can assume s of the waypoint
		# and s of the query point are the same for simplicity.
		s_waypoint   = self.trajectory[closest_index, 0]
		xy_waypoint  = self.trajectory[closest_index, 1:3]

		psi_waypoint = self.trajectory[closest_index, 3]

		rot_global_to_frenet = np.array([[ np.cos(psi_waypoint), np.sin(psi_waypoint)], \
			                             [-np.sin(psi_waypoint), np.cos(psi_waypoint)]])

		# Error_xy     = xy deviation (global frame)
		# Error_frenet = e_s, e_y deviation (Frenet frame)
		error_xy = xy_query - xy_waypoint
		error_frenet = np.dot(rot_global_to_frenet, error_xy[0,:])

		psi_error = fix_angle(psi_query - psi_waypoint)

		# Note: handling e_s can be ugly at kinks in the path (not smooth curvature)
		# so for simplicity, we can just ignore the e_s component and assume it's small.
		# Also, this approach is not well suited for cases like u-turns where the closest point
		# is not well defined.

		return s_waypoint, error_frenet[1], psi_error, closest_index # s, e_y, e_psi, idx

	def convert_frenet_to_global_frame(self, s_query, e_y_query, e_psi_query):
		s_traj  = self.trajectory[:,0]
		xy_traj = self.trajectory[:,1:3]

		# Handle "closest waypoint" differently based on s_query:
		if s_query < s_traj[0]:
			# NOTE: This can be problematic if s_query is really far away from the start.
			# Not handling this but intuitively, need to do some extrapolation.
			x_waypoint   = self.trajectory[0, 1]
			y_waypoint   = self.trajectory[0, 2]
			psi_waypoint = self.trajectory[0, 3]
		elif s_query > s_traj[-1]:
			# NOTE: This can be problematic if s_query is really far away from the end.
			# Not handling this but intuitively, need to do some extrapolation.
			x_waypoint   = self.trajectory[-1, 1]
			y_waypoint   = self.trajectory[-1, 2]
			psi_waypoint = self.trajectory[-1, 3]
		else:
			# NOTE: we can discuss this more in depth, but it's ambiguous whether to
			# linearly interpolate between waypoints or just using the closest one.
			# I think this is due to the lack of curvature constraints possibly on waypoints
			# such that the Frenet <-> Global transformation isn't truly invertible (see e_s discussion above).
			# Just going to keep this simple and use the closest waypoint.

			closest_index = np.argmin( np.abs( s_traj - s_query) )
			x_waypoint   = self.trajectory[closest_index, 1]
			y_waypoint   = self.trajectory[closest_index, 2]
			psi_waypoint = self.trajectory[closest_index, 3]

		rot_frenet_to_global = np.array([[np.cos(psi_waypoint), -np.sin(psi_waypoint)], \
			                             [np.sin(psi_waypoint),  np.cos(psi_waypoint)]])

		error_global = np.dot(rot_frenet_to_global, np.array([0, e_y_query])) # again assuming "e_s" is 0.

		x_global   = x_waypoint + error_global[0]
		y_global   = y_waypoint + error_global[1]
		psi_global = psi_waypoint + e_psi_query


		return x_global, y_global, psi_global

	def get_curvature_at_s(self, s_query):
		s_traj    = self.trajectory[:,0]
		closest_index = np.argmin( np.abs( s_traj - s_query) )
		return self.trajectory[closest_index,4]
