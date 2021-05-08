import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tslearn.clustering import TimeSeriesKMeans
from colorsys import hsv_to_rgb

from tfrecord_utils import _parse_no_img_function
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
	dataset = dataset.map(_parse_no_img_function, num_parallel_calls=8)
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(2)

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
# Clustering using kmeans over trajectories.
def identify_clusters(trajectory_dataset, n_clusters=16):
	kmeans = TimeSeriesKMeans(n_clusters=n_clusters,
		                      max_iter=100,
		                      tol=1e-6,
		                      n_init=5,
		                      metric='euclidean',
		                      init='k-means++',
		                      verbose=1)

	length, curv = compute_length_curvature(trajectory_dataset)
	trajectories_xy = trajectory_dataset[:,:,:2]
	# We fit clusters without theta, as the euclidean distance metric makes sense only with XY.
	kmeans.fit(trajectories_xy)

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
	plt.figure()
	plt.subplot(211)
	plt.bar( np.arange(num_anchors), freq)
	plt.ylabel('Frequency')
	plt.subplot(212)
	plt.bar(np.arange(num_anchors), weights)
	plt.ylabel('Weight')
	plt.suptitle('Anchor Weighting by Inverse Frequency')

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
		init_size = (400000, 25, 3)
	else:
		raise ValueError(f"Dataset {dataset} not supported.")

	trajectory_dataset = load_trajectory_dataset(train_set, init_size=init_size)

	if mode == 'visualize':
		plot_trajectory_dataset(trajectory_dataset)
	elif mode == 'cluster':
		cluster_trajs = identify_clusters(trajectory_dataset, n_clusters=n_clusters)

		weights = get_anchor_weights(cluster_trajs, trajectory_dataset)
		print("Anchor ID: Weight")
		[print(w_id, w) for (w_id, w) in zip(range(n_clusters), weights)]
		plt.show()
		import pdb; pdb.set_trace()

		np.save( f"{datadir}/{dataset}_clusters_{n_clusters}.npy", cluster_trajs)
		np.save( f"{datadir}/{dataset}_clusters_{n_clusters}_weights.npy", weights)
	else:
		raise ValueError(f"Mode {mode} not valid.")
