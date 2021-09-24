import os
import sys
import glob
import numpy as np
import tensorflow as tf

import pytest

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)

from datasets.tfrecord_utils import _parse_no_img_function
from datasets.splits import NUSCENES_VAL, L5KIT_VAL
from models.context_providers.context_provider_base import ContextProviderBase

NUSCENES_TIME_THRESH = 0.25 # secs, half a sample period of NuScenes.
L5KIT_TIME_THRESH    = 0.1  # secs, half a sample period of L5Kit (after downsampling).

'''
Test Fixture Parametrized by Dataset.
'''
SPLIT_NAME = "val"
@pytest.fixture(scope="module", params=['nuscenes', 'l5kit'])
def context_provider_and_dataset(request):
	if request.param == 'nuscenes':
		tfrecords = NUSCENES_VAL
		from models.context_providers.nuscenes_context_provider import NuScenesContextProvider as NCP
		context_provider = NCP()
	elif request.param == 'l5kit':
		tfrecords = L5KIT_VAL
		from models.context_providers.l5kit_context_provider import L5KitContextProvider as LCP
		context_provider = LCP()

	dataset = tf.data.TFRecordDataset(tfrecords)
	dataset = dataset.map(_parse_no_img_function)

	return context_provider, dataset

'''
Test Suite
'''
def test_transform_to_local_frame():
    # Test the world -> local by making sure it's "invertible".
    transform_func = ContextProviderBase._transform_to_local_frame

    test_x, test_y, test_yaw             = 10., 5., 1.0
    test_inv_x   = -( test_x * np.cos(test_yaw) + test_y * np.sin(test_yaw))
    test_inv_y   = -(-test_x * np.sin(test_yaw) + test_y * np.cos(test_yaw))
    test_inv_yaw = -test_yaw

    test_points = np.array([test_x, test_y]) + \
                  np.array([
                             [0., 0.],
                             [5., 0.],
                             [-5., 0.],
                             [0., 5.],
                             [0., -5.],
                             [1., 1.],
                             [1., -1.],
                             [-1., 1.],
                             [-1., -1.],
                           ])

    test_points_local = transform_func(test_x, test_y, test_yaw, test_points)
    test_points_recon = transform_func(test_inv_x, test_inv_y, test_inv_yaw, test_points_local)
    diff = np.linalg.norm(test_points - test_points_recon, axis=-1)
    assert np.allclose(diff, 0)

def test_lane_associations(context_provider_and_dataset, max_iters=10):
	context_provider, dataset = context_provider_and_dataset

	for ind_entry, entry in enumerate(dataset):
		sc = context_provider.get_context(tf.compat.as_str(entry["sample"].numpy()),
		                                  tf.compat.as_str(entry["instance"].numpy()),
		                                  split_name=SPLIT_NAME)
		pose = np.array([[sc.x, sc.y, sc.yaw]])

		for (lane_arr, red_tl_arr) in zip(sc.lanes, sc.red_traffic_lights):
			# Make sure traffic lights + lanes are consistent shapes.
			assert len(lane_arr.shape)   == 2
			assert len(red_tl_arr.shape) == 1
			assert lane_arr.shape[0] == red_tl_arr.shape[0]

			# Make sure the lane is sufficiently close to the pose.
			norms =  np.linalg.norm( lane_arr[:, :2] - pose[:, :2], axis=1)
			closest_idx = np.argmin(norms)
			assert norms[closest_idx] <= context_provider.lane_association_radius
			assert ContextProviderBase._bound_angle_within_pi( lane_arr[closest_idx, 2] - pose[0,2] ) < np.pi/2.

		if max_iters is not None and ind_entry > max_iters:
			break

def test_scene_validity(context_provider_and_dataset, max_iters=10):
	context_provider, dataset = context_provider_and_dataset

	if "nuscenes" in context_provider.__repr__():
		tm_thresh = NUSCENES_TIME_THRESH
	else:
		tm_thresh = L5KIT_TIME_THRESH


	for ind_entry, entry in enumerate(dataset):
		sc = context_provider.get_context(tf.compat.as_str(entry["sample"].numpy()),
			                              tf.compat.as_str(entry["instance"].numpy()),
			                              split_name=SPLIT_NAME)

		xmin, ymin, xmax, ymax = context_provider.in_view_bounds

		for lane_arr in sc.lanes:
			lane_arr_xy = lane_arr[:, :2]
			lane_arr_rel_xy = context_provider._transform_to_local_frame(sc.x, sc.y, sc.yaw, lane_arr_xy)
			assert (np.amin(lane_arr_rel_xy[:,0]) >= xmin and \
			       np.amax(lane_arr_rel_xy[:,0]) <= xmax and \
			       np.amin(lane_arr_rel_xy[:,1]) >= ymin and \
			       np.amax(lane_arr_rel_xy[:,1]) <= ymax)

		for agent_arr in sc.vehicles+sc.other_agents:
			agent_t  = agent_arr[:, 0]
			agent_xy = agent_arr[:, 1:3]
			agent_rel_xy = context_provider._transform_to_local_frame(sc.x, sc.y, sc.yaw, agent_xy)
			assert (np.amin(agent_rel_xy[:,0]) >= xmin and \
			       np.amax(agent_rel_xy[:,0]) <= xmax and \
			       np.amin(agent_rel_xy[:,1]) >= ymin and \
			       np.amax(agent_rel_xy[:,1]) <= ymax)

			assert (np.amin(agent_t) >= -context_provider.secs_of_hist-tm_thresh and \
			       np.amax(agent_t) <= 0.)

		if max_iters is not None and ind_entry > max_iters:
			break
