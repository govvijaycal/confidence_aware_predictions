import numpy as np

# Helper functions to handle poses and angles.
def rotation_global_to_local(yaw):
	return np.array([[ np.cos(yaw), np.sin(yaw)], \
		             [-np.sin(yaw), np.cos(yaw)]])

def angle_mod_2pi(angle):
	return (angle + np.pi) % (2.0 * np.pi) - np.pi

def pose_diff_norm(pose_diff):
	# Not exactly a traditional norm but just meant to ensure no pose differences.
	xy_norm    = np.linalg.norm(pose_diff[:,:2], ord=np.inf)
	angle_norm = np.max( [angle_mod_2pi(x) for x in pose_diff[:,2]] )
	return xy_norm + angle_norm

def convert_global_to_local(global_pose_origin, global_poses):
	R_global_to_local = rotation_global_to_local(global_pose_origin[2])
	t_global_to_local = - R_global_to_local @ global_pose_origin[:2]

	local_xy  = np.array([ R_global_to_local @ pose[:2] + t_global_to_local 
	                         for pose in global_poses])

	local_yaw = np.array([ angle_mod_2pi(pose[2] - global_pose_origin[2])
		                     for pose in global_poses])
	
	return np.column_stack((local_xy, local_yaw))

def convert_local_to_global(global_pose_origin, local_poses):
	R_local_to_global = rotation_global_to_local(global_pose_origin[2]).T
	t_local_to_global = global_pose_origin[:2]

	global_xy  = np.array([ R_local_to_global @ pose[:2] + t_local_to_global 
		                      for pose in local_poses])

	global_yaw = np.array([ angle_mod_2pi( pose[2] + global_pose_origin[2]) 
		                     for pose in local_poses])
	
	return np.column_stack((global_xy, global_yaw))