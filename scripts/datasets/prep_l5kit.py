import os
import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer

from tfrecord_utils import write_tfrecord, visualize_tfrecords, shuffle_test
from splits import L5KIT_TRAIN
##########################################
# Utils below for downsampling full l5kit dataset to a final set for TFRecord generation.
MIN_EGO_TRAJ_LENGTH = 1.0 # m, assumption is below this a vehicle is essentially "static".

def downsample_zarr_frames(zarr_dataset,
                      frame_skip_interval=10,
                      num_history_frames=10,
                      num_future_frames=50):
    """ Given a zarr dataset, downsamples frames by frame_skip_interval
        in each scene and returns a mask with the selected frames.
        E.g. frame_skip_interval=10 reduces sampling frequency by about 1/10.
        num_history_frames and num_future_frames are used to avoid sampling
        partial trajectories (i.e. all history/future data should be "available" in l5kit terms).
    """
    frames_mask = np.zeros( len(zarr_dataset.frames), dtype=np.bool )
    frame_intervals = zarr_dataset.scenes['frame_index_interval']

    for (frame_st, frame_end) in frame_intervals:
        for frame_selected in range(frame_st + num_history_frames,\
                                    frame_end - num_future_frames,\
                                    frame_skip_interval):
            frames_mask[frame_selected] = True

    return frames_mask

def downsample_torch_length(torch_dataset, length_thresh=MIN_EGO_TRAJ_LENGTH, batch_size=64, num_workers=16):
    """ Given a torch EgoDataset/AgentDataset, compute the trajectory length for every data
        instance and return a mask for which ones exceed length_thresh.  This is slow-ish
        (approx. 1 hour on train_full after it has been downsampled by 10x), so results should
        be saved.  batch_size / num_workers used to speed up by using torch dataloader.
    """
    dataloader = DataLoader(torch_dataset,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers=num_workers)
    num_instances = len(torch_dataset)
    lengths_mask = np.zeros( num_instances, dtype=np.bool)

    print(f'Computing the lengths mask - approx. {int(num_instances / batch_size)} iterations.')
    for bs_ind, instances in tqdm(enumerate(dataloader)):
        st_ind = bs_ind * batch_size
        end_ind = min(num_instances, (bs_ind + 1) * batch_size)
        diffs = instances['target_positions'][:, 1:, :] - instances['target_positions'][:, :-1, :]
        length = torch.sum(torch.norm(diffs, dim=2), dim=1) + torch.norm(instances['target_positions'][:, 0, :], dim=1)
        lengths_mask[st_ind:end_ind] = length.numpy() > length_thresh
    return lengths_mask

def compute_traj_length_curv_vel_alat(torch_dataset, frames_dt, batch_size = 64, num_workers = 16):
    """ Given a torch dataset, computes trajectory length, (avg.) curvature, (avg.) velocity,
        (avg.) lateral acceleration in a simple way.  Returns these stats for further processing and
        usage in a stratified sampling scheme.
    """
    dataloader = DataLoader(torch_dataset, batch_size = 64, shuffle = False, num_workers=16)
    num_instances = len(torch_dataset)

    lengths = np.ones( num_instances ) * np.nan # total displacement (m), aggregated over time
    curvs   = np.ones( num_instances ) * np.nan # crude estimate of constant curvature (rad / m)
    vels    = np.ones( num_instances ) * np.nan # crude estimate of average velocity (m / s)
    alats   = np.ones( num_instances ) * np.nan # crude estimate of average lateral acceleration (m/s^2)

    print(f'Computing trajectory statistics - approx. {int(num_instances / batch_size)} iterations.')
    for bs_ind, instances in tqdm(enumerate(dataloader)):
        st_ind = bs_ind * batch_size
        end_ind = min( num_instances , (bs_ind + 1) * batch_size)

        # Length is taken by finding the cumulative Euclidean distance along the future trajectory.
        # We add the norm of the first target position as well as it is 1 timestep ahead of the current one.
        diffs = instances['target_positions'][:, 1:, :] - instances['target_positions'][:, :-1, :]
        length = torch.sum(torch.norm(diffs, dim=2), dim=1) + torch.norm(instances['target_positions'][:, 0, :], dim=1)

        # Curvature crudely estimated by estimating as (final_yaw - 0) / length.
        # We assume this dataset has been prefiltered to avoid very short length trajectories.
        heading_final = torch.atan2(diffs[:, -1, 1], diffs[:, -1, 0])
        curv = heading_final / length

        # Crudely estimate velocity + lateral acceleration.
        vel = length / (instances['target_positions'].shape[1] * frames_dt) # essentialy dist / time
        alat = torch.square(vel) * curv

        lengths[st_ind:end_ind] = length.numpy()
        curvs[st_ind:end_ind]   = curv.numpy()
        vels[st_ind:end_ind]    = vel.numpy()
        alats[st_ind:end_ind]   = alat.numpy()

    return lengths, curvs, vels, alats

def stratified_sample_by_vel_alat(traj_stats,              # N by 4 matrix with non-nan rows for "masked" instances
                                  final_sample_size=12000, # desired sample size after stratified sampling, not guaranteed
                                  max_partition_freq=1.):  # maximum proportion allowed per partition for downsampling majority classes
    """ Attempts to produced a balanced, stratified random sample by using
        strata/partitions based on velocity and curvature.
    """

    # Which instances to include in a final stratified random sample.
    chosen_inds     = np.zeros( traj_stats.shape[0], dtype=np.bool )

    # Edges of the strata/partitions used.
    v_part_edges    = [0., 2., 4., 6., 8., 10., 12., 14., np.inf]
    alat_part_edges = [-np.inf, -1.5, -1., -0.5, 0., 0.5, 1., 1.5, np.inf]

    # valid_inds       = frames we are downsampling from.  Needed since we have nan rows in traj_stats.
    # num_instances    = number of full instances of valid data before downsampling.
    # valid_traj_stats = trajectory statistics for the valid indices only (the non-nan part).
    valid_inds        = np.ravel( np.argwhere( ~np.isnan(np.sum(traj_stats, axis=1)) ) ) # non nan rows
    num_instances     = len(valid_inds)
    valid_traj_stats  = traj_stats[valid_inds, :]

    np.random.seed(0) # want this sampling to be repeatable

    for ind_v, (vmin, vmax) in enumerate(zip(v_part_edges[:-1], v_part_edges[1:])):
        for ind_alat, (alatmin, alatmax) in enumerate(zip(alat_part_edges[:-1], alat_part_edges[1:])):

            # Identify which rows correspond to this partition.
            in_part = \
            np.logical_and.reduce( (valid_traj_stats[:,2] > vmin, \
                                    valid_traj_stats[:,3] > alatmin, \
                                    valid_traj_stats[:,2] <= vmax, \
                                    valid_traj_stats[:,3] <= alatmax) )
            indices_in_part = valid_inds[in_part]

            # Determine random sample size for this partition.  Upper bound with max_partition_freq.
            part_freq = min( np.sum(in_part) / num_instances, max_partition_freq )

            part_sample_size = np.ceil(final_sample_size * part_freq).astype(np.int)

            # Get the random samples for this partition only.  And mark it in our subset selection.
            samples = np.random.choice( indices_in_part, replace=False, size=part_sample_size)
            chosen_inds[samples] = True

            print(f"\tSelected {part_sample_size} of {np.sum(in_part)} samples from strata:"
                  f" v in [{vmin}, {vmax}], a_lat in [{alatmin}, {alatmax}]")

    # Make sure the final result is a proper subset of the set of downsampled frames.
    assert set(np.ravel(np.argwhere(chosen_inds))).issubset(valid_inds)

    # Note: This may not exactly match up due to use of np.ceil and max_partition_freq.
    #       But should be close.
    print(f"\tIn total, selected {np.sum(chosen_inds)} samples with {final_sample_size} desired.")

    return chosen_inds

##########################################
FINAL_DT_TFRECORD = 0.2
# Main function needed to extract a dictionary representation of a single element from a dataset.
def get_data_dict(torch_dataset,  # prefiltered/downsampled dataset that provides l5kit dataset instances
                  rasterizer,     # rasterizer to combine raw images from torch_dataset
                  dataset_index): # index (int) for which we want to get a dataset dict to write.
    if FINAL_DT_TFRECORD == 0.1:
        frame_skip = 1 # 10 Hz
    elif FINAL_DT_TFRECORD == 0.2:
        frame_skip = 2 # 5 Hz
    else:
        raise NotImplementedError("Only handling 0.1 s (10Hz) or 0.2 s (5 Hz) at the moment.")


    element = torch_dataset[dataset_index]
    # Unused keys: host_id, timestamp, extent, history_extents, future_extents
    #              *_velocities can be reconstructed using np.diff(..., axis=0) / 0.1
    #              raster_from_agent: straightforward given cfg params, just need to flip y-axis
    #              world_from_agent: similar functionality implemented in pose_utils.py
    #              speed - dropped since it involves peeking into the future poses

    # Sample in nuscenes terms = identifier for the timestamp in the dataset.
    sample = f"scene_{element['scene_index']}_frame_{element['frame_index']}"

    # Instance in nuscenes terms = identifier for the agent in that sample.
    instance = f"track_{element['track_id']}"
    if instance == f"track_{-1}":
        agent_type = 'ego'
    else:
        raise NotImplementedError("Not set up yet for arbitrary AgentDataset instances.")

    current_pose = np.append( element['centroid'], element['yaw'] )
    assert np.all(element['target_availabilities'] == 1), "Invalid future frame detected!"
    assert np.all(element['history_availabilities'] == 1), "Invalid past frame detected!"

    past_local_poses    = np.concatenate( (element['history_positions'], element['history_yaws']), axis=-1 )
    past_local_poses    = past_local_poses[frame_skip::frame_skip, :] # sample every frame_skip frame without the current pose
    future_local_poses  = np.concatenate( (element['target_positions'], element['target_yaws']), axis=-1 )
    future_local_poses  = future_local_poses[(frame_skip-1)::frame_skip, :] # sample every frame skip frame

    vel = np.linalg.norm( past_local_poses[0, :2] ) / FINAL_DT_TFRECORD
    vel_prev = np.linalg.norm( past_local_poses[1, :2] - past_local_poses[0, :2] ) / FINAL_DT_TFRECORD
    yaw_rate = -past_local_poses[0, 2] / FINAL_DT_TFRECORD
    accel = (vel - vel_prev) / FINAL_DT_TFRECORD

    past_tms   = np.array( [-FINAL_DT_TFRECORD * x for x in range(1, len(past_local_poses)+1)] )
    future_tms = np.array( [FINAL_DT_TFRECORD * x for x in range(1, len(future_local_poses)+1)] )

    img = rasterizer.to_rgb( element['image'].transpose(1,2,0) )

    return {'instance': instance,
            'sample': sample,
            'type': agent_type,
            'pose': current_pose,
            'velocity': vel,
            'acceleration': accel,
            'yaw_rate': yaw_rate,
            'past_poses_local': past_local_poses,
            'future_poses_local': future_local_poses,
            'past_tms': past_tms,
            'future_tms': future_tms,
            'image': img}

##########################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Read/Write L5Kit prediction instances in TFRecord format.')
    parser.add_argument('--mode', choices=['read', 'write', 'batch_test'], type=str, required=True, help='Write or read TFRecords.')
    parser.add_argument('--datadir', type=str, help='Where the TFRecords are located or should be saved.', \
                            default=os.path.abspath(__file__).split('scripts')[0] + 'data')
    parser.add_argument('--dataroot', type=str, help='Location of the L5Kit dataset.', \
                            default='/media/data/l5kit-data/')
    args = parser.parse_args()
    datadir = args.datadir
    mode = args.mode
    dataroot = args.dataroot

    if mode == "write":
        dm = LocalDataManager(local_data_folder=dataroot)

        cfg = load_config_data('/'.join( os.path.abspath(__file__).split('/')[:-1]) + \
                               '/l5kit_prediction_config.yaml')
        frames_dt         = cfg['model_params']['step_time']
        frames_history    = cfg['model_params']['history_num_frames']
        frames_future     = cfg['model_params']['future_num_frames']
        rasterizer = build_rasterizer(cfg, dm)

        for split in ['val', 'train']:
            zarr = ChunkedDataset(dm.require(cfg[f'{split}_data_loader']['key'])).open()
            ego_dataset = EgoDataset(cfg, zarr, None) # rasterizer not used to improve speed

            # Downsample dataset based on desired frames_history/frames_future
            # so that we have full dataset instances.  Prune trajectories that
            # are near-stationary. as judged by MIN_EGO_TRAJ_LENGTH.
            print(f"Dataset {split}: downsampling and removing trajectories " \
                  f"with length < {MIN_EGO_TRAJ_LENGTH} m.")
            frames_mask_path = Path( f"{datadir}/l5kit_{split}_ego_frames_mask.npy" )
            if not frames_mask_path.exists():
                frames_mask = downsample_zarr_frames(zarr,
                                                     frame_skip_interval=frames_history, # avoid overlapping samples
                                                     num_history_frames=frames_history,
                                                     num_future_frames=frames_future)
                frame_ds_inds = np.nonzero(frames_mask)[0]
                frame_subset = torch.utils.data.Subset(ego_dataset, frame_ds_inds)
                lengths_mask = downsample_torch_length(frame_subset)
                frames_mask[ frame_ds_inds ] = lengths_mask
                np.save(str(frames_mask_path), frames_mask)
            else:
                frames_mask = np.load(str(frames_mask_path))

            # Compute statistics needed for sampling the datasets.
            # In particular, we are storing length, curvature, velocity
            # and lateral acceleration.
            print(f"Dataset {split}: computing trajectory statistics.")
            traj_stats_path = Path( f"{datadir}/l5kit_{split}_ego_frames_stats.npy" )
            if not traj_stats_path.exists():
                frame_selected_inds = np.nonzero(frames_mask)[0]
                frame_subset = torch.utils.data.Subset(ego_dataset, frame_selected_inds)

                lengths, curvs, vels, alats = compute_traj_length_curv_vel_alat(frame_subset, frames_dt)

                # Traj stats will be a N by 4 matrix.  The row is full of np.nan if it's not part of our frame_subset.
                # Else it is a nonzero row with values length, curvature, velocity, and lateral acceleration.
                traj_stats = np.ones( (len(frames_mask), 4), dtype=np.float32) * np.nan
                traj_stats[ frame_selected_inds, : ] = np.column_stack((lengths, curvs, vels, alats)).astype(np.float32)
                np.save(str(traj_stats_path), traj_stats)
            else:
                traj_stats = np.load(str(traj_stats_path))

            # Perform stratified sampling by velocity and lateral acceleration.
            print(f"Dataset {split}: performing stratified sampling over velocity and lateral acceleration.")
            if split == 'val':
                # Naming is a bit confusing, but the 'val' split is the test set.
                # Downsample to 100,000 examples to keep space usage bounded (~70 GB).
                val_mask = stratified_sample_by_vel_alat(traj_stats, final_sample_size=100000)

                assert np.all( np.logical_or(val_mask, frames_mask) == frames_mask )

                masks_to_write = {'val': val_mask}
            elif split == 'train':
                # The large train split is used to construct training and training_validation sets.

                # First, we use stratified random sampling to pick out training_validation examples.
                # 100,000 examples is approx 70 GB.
                trainval_mask = stratified_sample_by_vel_alat(traj_stats, final_sample_size=100000)

                # Then we remove the selected indices from consideration for training examples.
                traj_stats[trainval_mask, :] = np.nan

                # Finally, we apply stratified random sampling for the training set with reduced
                # candidate set.  We can impose a max_partition_freq bound here to undersample
                # the majority strata.  Looking to get about 300,000 examples (210 GB).
                train_mask = stratified_sample_by_vel_alat(traj_stats, final_sample_size=350000,
                                                           max_partition_freq=0.1)

                # Double check that we have selected disjoint subsets for train and trainval.
                # Also check that train/trainval are subsets of frames_mask (we didn't screw up indexing).
                assert np.all( np.logical_and(trainval_mask, train_mask) == False )
                assert np.all( np.logical_or(trainval_mask, frames_mask) == frames_mask )
                assert np.all( np.logical_or(train_mask, frames_mask) == frames_mask )

                masks_to_write = {'trainval': trainval_mask, 'train': train_mask}
            else:
                raise ValueError(f"Invalid split: {split}")

            # Finally write the tfrecords.
            len_ego_dataset = len(ego_dataset)
            del ego_dataset

            for (mask_name, mask) in masks_to_write.items():

                final_mask_path = Path( f"{datadir}/l5kit_{mask_name}_ego_frames_mask_final.npy" )
                np.save(str(final_mask_path), mask)

                final_ego_dataset = torch.utils.data.Subset(EgoDataset(cfg, zarr, rasterizer), np.nonzero(mask)[0])
                print(f"Dataset {mask_name} has {len(final_ego_dataset)} instances out of {len_ego_dataset}.")

                data_dict_function = partial(get_data_dict, final_ego_dataset, rasterizer)
                dataset_inds = np.arange(len(final_ego_dataset)).astype(np.int)
                file_prefix = f"{datadir}/l5kit_{mask_name}"
                write_tfrecord(file_prefix,
                               dataset_inds,
                               data_dict_function,
                               shuffle = True,
                               shuffle_seed = 0,
                               max_per_record = 1000)
    elif mode == 'read':
        visualize_tfrecords(L5KIT_TRAIN, max_batches=100)

    elif mode == 'batch_test':
        shuffle_test(L5KIT_TRAIN, batch_size=32)

        # Results: shows good shuffling (dataset size 306186)
        # Shuffle min range and std dev:  4555, 2024
        # Shuffle max range and std dev:  298685, 132786
        # Shuffle mean range and std dev: 152352, 65311

    else:
        raise ValueError("Invalid mode: {}".format(mode))
