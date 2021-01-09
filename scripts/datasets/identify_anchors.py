import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from sklearn.cluster import KMeans
from tfrecord_utils import _parse_function
from colorsys import hsv_to_rgb

def load_trajectory_dataset(tfrecord_files, init_size=(30000, 12, 3)):
	trajectory_dataset = np.ones(init_size) * np.nan

	dataset = tf.data.TFRecordDataset(tfrecord_files)
	dataset = dataset.map(_parse_function)

	for ind, entry in enumerate(dataset):
		trajectory_dataset[ind,:,:] = entry['future_poses_local'] # x, y, theta
	
	num_elements = ind + 1
	return trajectory_dataset[:num_elements, :, :]

def plot_trajectory_dataset(trajectory_dataset):
	num_elements = trajectory_dataset.shape[0]
	for ind in range(num_elements):
		plt.plot(trajectory_dataset[ind, :, 0], trajectory_dataset[ind, :, 1])
	plt.title('{} Trajectories'.format(num_elements))
	plt.xlabel('X (m)')
	plt.ylabel('Y (m)')
	plt.show()

def length_curv_to_trajectories(length_curv_list, num_timesteps=12):
	trajectories = []
	for length_curv in length_curv_list:
		length, curv = length_curv
		assert np.abs(curv) > 1e-6

		# Integration of the following ODE:
		# dx/ds = cos(theta(s))
		# dy/ds = sin(theta(s))
		# dtheta/ds = curv
		# x(0) = y(0) = theta(0) = 0.
		# assume constant "speed" and curvature to generate ss.
		ss = np.linspace(0.0, length, num_timesteps)
		xs = 1 / curv * np.sin(curv * np.array(ss))
		ys = 1 / curv * ( 1 - np.cos(curv * np.array(ss)) )

		trajectories.append(np.column_stack((xs, ys)))	
	return np.array(trajectories)

def identify_clusters_length_curvature(trajectory_dataset, n_clusters=16):
	# Approximate curvature just with endpoints yaw difference / length for simplicity.
	length = np.sum( np.linalg.norm(trajectory_dataset[:,1:,:2] - trajectory_dataset[:,:-1,:2], 
		                            axis=-1), axis=-1)
	curv   = (trajectory_dataset[:, -1, 2] - trajectory_dataset[:, 0, 2]) / length

	# Partition by length into four equally long bins.
	part_1 = np.argwhere( length < 25. )
	part_2 = np.argwhere( np.logical_and( length >= 25.,  length < 50.) )
	part_3 = np.argwhere( np.logical_and( length >= 50.,  length < 75.) )
	part_4 = np.argwhere( length >= 75. )

	# Used to "normalize" the length curvature dataset with customizability
	# as to how much to emphasize curvature vs. length.
	scale_factor = np.array([[100., 0.1]])
	length_curv_scaled = np.column_stack( (length, curv) ) / scale_factor
	
	# Find the clusters in the scaled length-curvature space, then scale
	# back to the original length/curvature.  Silhouette score used to
	# determine clustering performance (closer to +1 preferred).
	# With this approach, getting scores in [0.1, 0.15].
	kmeans = KMeans(n_clusters=n_clusters).fit(length_curv_scaled)
	clusters = kmeans.cluster_centers_ * scale_factor
	print(silhouette_score(trajectory_dataset[:,:,:2], kmeans.labels_, metric='euclidean'))
	
	# Plot all the trajectories in black and then the clusters
	# using colors chosen by partitioning the hue in HSV space.
	plt.figure()
	for traj in trajectory_dataset:
		plt.plot(traj[:,0], traj[:,1], 'k', lw=1)

	cluster_trajs = length_curv_to_trajectories(clusters)
	mean_colors = [hsv_to_rgb(h, 1.0, 1.0) for h in np.linspace(0.0, 1.0, cluster_trajs.shape[0]+1)]
	mean_colors = mean_colors[:cluster_trajs.shape[0]]
	for ind_cl, cluster_traj in enumerate(cluster_trajs):
		plt.plot(cluster_traj[:,0], cluster_traj[:,1], color=mean_colors[ind_cl], lw=3)
	plt.axis('equal')
	
	# Plot the length vs. curvature for all trajectories based on the partition
	# and then show how the clusters were allocated to each partition.
	# We are not manually specifying the number of cluster per partition with KMeans above.
	plt.figure()
	plt.scatter(length[part_1], curv[part_1], color='r')
	plt.scatter(length[part_2], curv[part_2], color='g')
	plt.scatter(length[part_3], curv[part_3], color='b')
	plt.scatter(length[part_4], curv[part_4], color='c')
	plt.scatter(clusters[:,0], clusters[:,1], color='k')	

	plt.show()	

	return cluster_trajs

def identify_clusters(trajectory_dataset, n_clusters=16):	
	kmeans = TimeSeriesKMeans(n_clusters=n_clusters,
		                      max_iter=100,
		                      tol=1e-6,
		                      n_init=10,
		                      metric='euclidean',
		                      init='k-means++',
		                      verbose=1)

	# We fit clusters without theta, as the euclidean distance metric makes sense only with XY.
	kmeans.fit(trajectory_dataset[:,:,:2]) 

	# Silhouette score used to determine clustering performance (closer to +1 preferred).
	# With this approach, getting scores around 0.3, which is better than the length/curv approach.
	print('Silhouette Score: {}'.format( \
		  silhouette_score(trajectory_dataset[:,:,:2], kmeans.labels_, metric='euclidean')))
	
	# Plot all the trajectories in black and then the clusters
	# using colors chosen by partitioning the hue in HSV space.
	plt.figure()	
	for traj in trajectory_dataset:
		plt.plot(traj[:,0], traj[:,1], 'k', lw=1)
	
	traj_clusters  = kmeans.cluster_centers_
	mean_colors = [hsv_to_rgb(h, 1.0, 1.0) for h in np.linspace(0.0, 1.0, n_clusters+1)]
	mean_colors = mean_colors[:n_clusters]

	for cl in range(n_clusters):				
		plt.plot(traj_clusters[cl,:,0], traj_clusters[cl,:,1], color=mean_colors[cl], linewidth=5)	
	plt.axis('equal')

	# Plot the length vs. curvature for all trajectories based on the partition
	# and then show how the clusters were allocated to each partition.
	# We are not manually specifying the number of cluster per partition with KMeans above.
	plt.figure()
	length = np.sum( np.linalg.norm(trajectory_dataset[:,1:,:2] - trajectory_dataset[:,:-1,:2], 
		                            axis=-1), axis=-1)
	curv   = (trajectory_dataset[:, -1, 2] - trajectory_dataset[:, 0, 2]) / length
	part_1 = np.argwhere( length < 25. )
	part_2 = np.argwhere( np.logical_and( length >= 25.,  length < 50.) )
	part_3 = np.argwhere( np.logical_and( length >= 50.,  length < 75.) )
	part_4 = np.argwhere( length >= 75. )
	plt.scatter(length[part_1], curv[part_1], color='r')
	plt.scatter(length[part_2], curv[part_2], color='g')
	plt.scatter(length[part_3], curv[part_3], color='b')
	plt.scatter(length[part_4], curv[part_4], color='c')

	length_cluster = np.sum( np.linalg.norm(traj_clusters[:,1:,:2] - traj_clusters[:,:-1,:2], 
		                            axis=-1), axis=-1)
	xy_diff_end = traj_clusters[:,-1,:] - traj_clusters[:,-2,:]
	heading_end = np.arctan( xy_diff_end[:,1] / xy_diff_end[:,0] )
	curv_cluster = heading_end / length_cluster
	plt.scatter(length_cluster, curv_cluster, color='k')
	plt.show()		

	return traj_clusters

if __name__ == '__main__':
	save_clusters = False

	datadir = os.path.abspath(__file__).split('scripts')[0] + 'data'
	train_set = glob.glob(datadir + '/nuscenes_train*.record')
	train_set = [x for x in train_set if 'val' not in x]	

	trajectory_dataset = load_trajectory_dataset(train_set)
	#plot_trajectory_dataset(trajectory_dataset)
	#identify_clusters_length_curvature(trajectory_dataset, n_clusters=16)
	cluster_trajs = identify_clusters(trajectory_dataset, n_clusters=32)

	if save_clusters:
		np.save( datadir + '/nuscenes_clusters_16.npy', cluster_trajs)


	

	

	
	

	
	

	


