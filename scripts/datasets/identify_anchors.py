import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from sklearn.cluster import KMeans
from colorsys import hsv_to_rgb

from tfrecord_utils import _parse_function
from splits import L5KIT_TRAIN, L5KIT_VAL, L5KIT_TEST, \
                   NUSCENES_TRAIN, NUSCENES_VAL, NUSCENES_TEST

##########################################
# Loading and visualizing a trajectory dataset.
def load_trajectory_dataset(tfrecord_files, batch_size=64, init_size=(30000, 12, 3)):
	""" Given a set of tfrecords, assembles a np.array containing all future trajectories. """

	# The size is M x N x 3, where:
	# M = number of trajectories
	# N = number of timesteps in a trajectory
	# 3 = state dim, [x, y, yaw] in a local frame s.t. state = [0,0,0] at current timestep
	trajectory_dataset = np.ones(init_size) * np.nan

	dataset = tf.data.TFRecordDataset(tfrecord_files)
	dataset = dataset.map(_parse_function)
	dataset = dataset.batch(batch_size)

	print("Loading data: ")
	num_elements = 0

	for ind_batch, entry_batch in tqdm(enumerate(dataset)):
		poses = entry_batch['future_poses_local'] # x, y, theta
		num_elements += poses.shape[0]

		st_ind  = batch_size * ind_batch
		end_ind = st_ind + poses.shape[0]
		trajectory_dataset[st_ind:end_ind,:,:] = poses

	return trajectory_dataset[:num_elements, :, :]

def plot_trajectory_dataset(trajectory_dataset):
	""" Given a trajectory dataset (np.array),plots all future trajectories. """
	num_elements = trajectory_dataset.shape[0]

	plt.figure()
	for ind in range(num_elements):
		plt.plot(trajectory_dataset[ind, :, 0], trajectory_dataset[ind, :, 1])
	plt.title('{} Trajectories'.format(num_elements))
	plt.xlabel('X (m)')
	plt.ylabel('Y (m)')
	
	plt.show()

def compute_length_curvature(trajectory_dataset):
	# Approximate curvature just with endpoints yaw difference / length for simplicity.
	length = np.linalg.norm(trajectory_dataset[:,0, :2], axis=-1) +  \
	         np.sum( np.linalg.norm(trajectory_dataset[:,1:,:2] - trajectory_dataset[:,:-1,:2], 
		                            axis=-1), axis=-1)
	curv   = trajectory_dataset[:, -1, 2] / length

	return length, curv

##########################################
# Utils/prototyping to try out various ideas based on length /curvature.
# Unused as of now.
def constant_length_curv_to_trajectories(length_curv_list, num_timesteps=12):
	""" Convert a constant length and curvature description to a trajectory based on
	    numerical integration of ODE below.
	"""
	trajectories = []
	for length_curv in length_curv_list:
		length, curv = length_curv # curv/length are scalars
		
		# Integration of the following ODE:
		# dx/ds = cos(theta(s))
		# dy/ds = sin(theta(s))
		# dtheta/ds = curv
		# x(0) = y(0) = theta(0) = 0.
		# assume constant "speed" and curvature to generate ss.
		ss = np.linspace(0.0, length, num_timesteps)

		if np.abs(curv) > 1e-6:
			# Nonzero curvature
			xs = 1 / curv * np.sin(curv * np.array(ss))
			ys = 1 / curv * ( 1 - np.cos(curv * np.array(ss)) )
		else:
			# Zero curvature
			xs = ss
			ys = np.zeros_like(ss)

		trajectories.append(np.column_stack((xs, ys)))	
	return np.array(trajectories)

def check_stratified_coverage(trajectory_dataset):	
	""" Breaks down dataset into bins of length/curvature and sees the
		distribution among these strata.  L5kit dataset is balanced
		manually but nuscenes is taken as is without rebalancing.
	"""
	lengths, curvs = compute_length_curvature(trajectory_dataset)

	under_50 = lengths < 50	  # under 50 m long future trajectory
	above_50 = lengths >= 50  # over 50 m ""

	# Strata used for l5kit.  For nuscenes, there was insufficient data
	# to use this effectively but there is a heavy bias to short trajectories
	# with low curvature (>50% straight with length < 50 m).
	part1 = np.logical_and( under_50, np.abs(curvs) <= 0.02)
	part2 = np.logical_and( under_50, curvs > 0.02)
	part3 = np.logical_and( under_50, curvs < -0.02)
	part4 = np.logical_and( above_50, np.abs(curvs) <= 0.002)
	part5 = np.logical_and( above_50, curvs > 0.002)
	part6 = np.logical_and( above_50, curvs < -0.002)

	for ind_part, part in enumerate([part1, part2, part3, part4, part5, part6]):
		print(f"Partition {ind_part+1} has {np.sum(part)} elements.")	

##########################################
def identify_clusters_length_curvature(trajectory_dataset, n_clusters=16):
	""" KMeans performed in length vs. curvature space """
	length, curv = compute_length_curvature(trajectory_dataset)

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

	length, curv = compute_length_curvature(trajectory_dataset)
	trajectories_xy = trajectory_dataset[:,:,:2]
	# We fit clusters without theta, as the euclidean distance metric makes sense only with XY.
	kmeans.fit(trajectories_xy) 

	# Silhouette score used to determine clustering performance (closer to +1 preferred).
	# With this approach, getting scores around 0.3, which is better than the length/curv approach.
	print('Silhouette Score: {}'.format( \
		  silhouette_score(trajectories_xy, kmeans.labels_, metric='euclidean')))
	
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
	
	part_1 = np.argwhere( length < 25. )
	part_2 = np.argwhere( np.logical_and( length >= 25.,  length < 50.) )
	part_3 = np.argwhere( np.logical_and( length >= 50.,  length < 75.) )
	part_4 = np.argwhere( length >= 75. )
	plt.scatter(length[part_1], curv[part_1], color='r')
	plt.scatter(length[part_2], curv[part_2], color='g')
	plt.scatter(length[part_3], curv[part_3], color='b')
	plt.scatter(length[part_4], curv[part_4], color='c')

	length_cluster = np.sum( traj_clusters[:,0,:2], axis=-1 ) + \
	                 np.sum( np.linalg.norm(traj_clusters[:,1:,:2] - traj_clusters[:,:-1,:2], 
		                            axis=-1), axis=-1)
	xy_diff_end = traj_clusters[:,-1,:] - traj_clusters[:,-2,:]
	heading_end = np.arctan2( xy_diff_end[:,1] , xy_diff_end[:,0] )
	curv_cluster = heading_end / length_cluster
	plt.scatter(length_cluster, curv_cluster, color='k')
	plt.show()		

	return traj_clusters

def get_anchor_weights(anchors, trajectory_dataset):
	""" Given identified anchors and a trajectory dataset, finds anchor classification weights 
	    to address unbalanced datasets with some rare classes. 
	"""
	# Identify the classification labels (anchor id) for each trajectory.
	trajectories_xy = trajectory_dataset[:, :, :2]

	anchor_dists = np.column_stack([np.sum(np.linalg.norm(trajectories_xy - anc, axis=-1), axis=-1)
	                                for anc in anchors])

	anchor_closest = np.argmin(anchor_dists, axis=-1)

	# Determine the frequency of each class/anchor and compute weights to emphasize
	# the less frequent classes.  Roughly, this is inversely proportional to the relative
	# frequency of that anchor.
	num_anchors = anchors.shape[0]
	freq = [np.sum(anchor_closest == ind_anc) for ind_anc in range(num_anchors)] 
	weights = np.sum(freq) / freq
	weights /= np.max(weights)

	# Visualize results.
	plt.subplot(211)
	plt.bar( np.arange(num_anchors), freq)
	plt.ylabel('Frequency')
	plt.subplot(212)
	plt.bar(np.arange(num_anchors), weights)
	plt.ylabel('Weight')
	plt.suptitle('Anchor Weighting by Inverse Frequency')
	plt.show()

	return weights

##########################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser('Analyze trajectories in TFRecord and determine clusters using time-series k-means.')
	parser.add_argument('--mode', choices=['visualize', 'cluster'], type=str, required=True, help='What task to perform: visualizing or clustering trajectories.')
	parser.add_argument('--dataset', choices=['l5kit', 'nuscenes'], type=str, required=True, help='Which TFRecord dataset to analyze.')
	parser.add_argument('--n_clusters', type=int, default=16, help='Number of k-means clusters if using cluster mode.')
	parser.add_argument('--datadir', type=str, help='Where to load datasets and save results..', \
		                    default=os.path.abspath(__file__).split('scripts')[0] + 'data')
	args = parser.parse_args()
	mode = args.mode
	dataset = args.dataset
	n_clusters = args.n_clusters
	datadir = args.datadir
	
	if dataset == 'nuscenes':
		train_set = NUSCENES_TRAIN
		init_size = (32000, 12, 3)
	elif dataset == 'l5kit':
		train_set = L5KIT_TRAIN
		init_size = (36000, 25, 3)
	else:
		raise ValueError(f"Dataset {dataset} not supported.")

	trajectory_dataset = load_trajectory_dataset(train_set, init_size=init_size)	

	if mode == 'visualize':
		plot_trajectory_dataset(trajectory_dataset)
		check_stratified_coverage(trajectory_dataset)
	elif mode == 'cluster':		
		cluster_trajs = identify_clusters(trajectory_dataset, n_clusters=n_clusters)

		weights = get_anchor_weights(cluster_trajs, trajectory_dataset)
		print("Anchor ID: Weight")
		[print(w_id, w) for (w_id, w) in zip(range(n_clusters), weights)]

		np.save( f"{datadir}/{dataset}_clusters_{n_clusters}.npy", cluster_trajs)	
		np.save( f"{datadir}/{dataset}_clusters_{n_clusters}_weights.npy", weights)
	else:
		# Unused util functions related to clustering in length/curv space:
		# identify_clusters_length_curvature, constant_length_curv_to_trajectories
		raise ValueError(f"Mode {mode} not valid.")
