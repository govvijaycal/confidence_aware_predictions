import os
import sys
import numpy as np
import time
import pickle
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from functools import partial

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset, get_frames_slice_from_scenes
from l5kit.data.filter import filter_vehicle_agents_by_labels, filter_nonvehicle_agents_by_labels, \
                              filter_tl_faces_by_status
from l5kit.geometry import transform_points, ecef_to_geodetic, rotation33_as_yaw
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.data.map_api import InterpolationMethod, MapAPI, TLFacesColors
from l5kit.rasterization import RenderContext
from l5kit.rasterization.semantic_rasterizer import INTERPOLATION_POINTS, RasterEls, indices_in_bounds
from l5kit.rasterization.sem_box_rasterizer_vg import SemBoxRasterizerVG
from l5kit.sampling import get_agent_context

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)
from models.context_providers.context_provider_base import ContextProviderBase, SceneContext

####################################################################################################
class L5KitContextProvider(ContextProviderBase):
    def __init__(self, dataroot="/media/data/l5kit-data/", overpass_url=None):
        # Note: change overpass_url to http://localhost/api/interpreter for local server
        super().__init__()

        dm  = LocalDataManager(local_data_folder=dataroot)
        cfg = load_config_data( os.path.abspath(__file__).split("models")[0] + \
                                "datasets/l5kit_prediction_config.yaml" )

        semantic_map_path = dm.require(cfg["raster_params"]["semantic_map_key"])
        metadata_path = dm.require(cfg["raster_params"]["dataset_meta_key"])
        with open(metadata_path, "r") as f:
            dataset_meta  = json.load(f)
            self.world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

        self.mapAPI = MapAPI(semantic_map_path, self.world_to_ecef)

        centerline_cache_path = Path( f"{dataroot}/semantic_map/centerline_cache.pkl" )
        if not centerline_cache_path.exists():
            print('Generating and saving L5Kit centerlines cache.')
            self._init_overpass(overpass_url)
            self._generate_map_cache(centerline_cache_path)

        print('Loading L5Kit centerlines cache.')
        self._load_map_cache(centerline_cache_path)

        self.train_zarr = ChunkedDataset(dm.require(cfg['train_data_loader']['key'])).open()
        self.val_zarr   = ChunkedDataset(dm.require(cfg['val_data_loader']['key'])).open()

        # Make the sample function, adapted from the constructor of l5kit.dataset.EgoDataset.
        self.sample_function = partial(
            get_agent_context,
            history_num_frames=cfg["model_params"]["history_num_frames"],
            future_num_frames=cfg["model_params"]["future_num_frames"]
        )
        self.filter_agents_threshold = cfg["raster_params"]["filter_agents_threshold"]
        self.frames_to_use           = [0] + cfg["raster_params"]["frames_to_plot"]

    def _generate_map_cache(self, centerline_cache_path):
        centerline_dict = {}
        for lane_id in tqdm(self.mapAPI.bounds_info["lanes"]["ids"]):
            lane_element = self.mapAPI[lane_id].element.lane

            # Use the same interpolation method as the semantic_rasterizer for consistency.
            lane_coords = self.mapAPI.get_lane_as_interpolation(
                lane_id, INTERPOLATION_POINTS, InterpolationMethod.INTER_ENSURE_LEN
            )
            centerline_xy     = lane_coords["xyz_midlane"][:, :2]
            centerline_dxy    = np.diff(centerline_xy, axis=0)
            centerline_yaw    = np.arctan2(centerline_dxy[:,1], centerline_dxy[:,0])
            centerline_yaw    = np.append(centerline_yaw, centerline_yaw[-1])

            centerline_ecef   = transform_points(lane_coords["xyz_midlane"], self.world_to_ecef)
            centerline_latlon = np.array([ecef_to_geodetic(pt) for pt in centerline_ecef])[:, :2]

            # Note: I found that only ~1000/8000 of the lanes had speed limit annotations
            #       in the map.  So using Overpass API instead.

            latlon_queries = centerline_latlon[[0, INTERPOLATION_POINTS//2, INTERPOLATION_POINTS-1], :]

            v_avg_mps = self._get_average_speed_overpass(latlon_queries)

            lane_arr = np.column_stack( (centerline_xy,
                                         centerline_yaw,
                                         v_avg_mps * np.ones(centerline_xy.shape[0])
                                        )
                                      )
            centerline_dict[lane_id] = lane_arr

        pickle.dump(centerline_dict, open(str(centerline_cache_path), "wb"))

    @staticmethod
    def _get_ego_pose_from_frame(frame):
        x, y, _ = frame["ego_translation"]
        yaw     = rotation33_as_yaw(frame["ego_rotation"])
        return x, y, yaw

    def _get_frame(self, scene_idx, state_idx, split_name):
        # TODO: check this out.
        if split_name   == "train" or split_name == "val":
            dataset = self.train_zarr
        elif split_name == "test":
            dataset = self.val_zarr
        else:
            raise ValueError(f"Invalid split: {split_name}")

        # This is taken from get_frame method of l5kit.dataset.EgoDatset.
        frames = dataset.frames[get_frames_slice_from_scenes(dataset.scenes[scene_idx])]

        return self.sample_function(state_idx, frames, dataset.agents, dataset.tl_faces)

    def _get_candidate_lanes(self, x, y, yaw):
        # (1) Get the lane graph in the local vicinity (upper-bounding the actual view region).
        lane_indices = indices_in_bounds( np.array([x, y]), self.mapAPI.bounds_info["lanes"]["bounds"], self.view_radius )
        lane_indices = [self.mapAPI.bounds_info["lanes"]["ids"][lid] for lid in lane_indices]
        lane_adjacency = {lane_id : [] for lane_id in lane_indices} # lane_id : [lane_id_next_1, lane_id_next_2, ...]
        lane_indegree  = {lane_id : 0 for lane_id in lane_indices}  # lane_id : indegree, i.e. # of incoming lanes

        for lane_idx in lane_indices:
            lane = self.mapAPI[lane_idx].element.lane
            lanes_next = lane.lanes_ahead

            for lane_n in lanes_next:
                lane_n_idx = self.mapAPI.id_as_str(lane_n)

                if lane_n_idx not in lane_indices:
                    continue # lane_n_idx is out of the view

                # lane_n_idx is in view, mark the edge lane_idx -> lane_n_idx
                lane_indegree[lane_n_idx] += 1
                lane_adjacency[lane_idx].append(lane_n_idx)

        # (2) Get the possible lane traversals (as concatenated strings of lane tokens).
        cand_lane_traversal_str = self._identify_candidate_lane_traversals(lane_adjacency, lane_indegree)

        # (3) For each traversal, get the corresponding array of states and lane segment lengths (per lane token).
        cand_lane_traversal_arr = []
        cand_lane_traversal_seg_lens = []
        for l in cand_lane_traversal_str:
            out = self._get_lane_array_from_traversal(l, self.centerline_dict)
            cand_lane_traversal_arr.append(out[0])
            cand_lane_traversal_seg_lens.append(out[1])

        # (4) Cut the lane traversals to those in the view region and sufficiently "close" to the query pose.
        final_lane_traversal_str, \
        final_lane_traversal_arr, \
        final_lane_traversal_seg_lens = \
        self._truncate_lanes_in_view(x, y, yaw, self.lane_association_radius,
                                     cand_lane_traversal_str,
                                     cand_lane_traversal_arr,
                                     cand_lane_traversal_seg_lens)

        return (final_lane_traversal_str,     # List with each entry including constituent, ordered lane tokens in that traversal
               final_lane_traversal_arr,      # List with each entry including states (x, y, theta, v) along that traversal
               final_lane_traversal_seg_lens) # List with each entry including number of states from a lane token, matching _str / _arr.

    def _get_nearby_agents(self, history_frames, history_agents):
        ego_x, ego_y, ego_yaw = self._get_ego_pose_from_frame(history_frames[0])
        curr_time             = history_frames[0]["timestamp"]
        vehicle_dict     = {}
        other_agent_dict = {}

        def add_agent_info_to_dict(agent, rel_time, agent_dict):
            agent_pose    = np.append(agent["centroid"], agent["yaw"])
            agent_tm_pose = np.insert(agent_pose, 0, rel_time)
            if agent["track_id"] not in agent_dict.keys():
                agent_dict[agent["track_id"]] = [agent_tm_pose]
            else:
                agent_dict[agent["track_id"]].append(agent_tm_pose)

        for frame_ind in reversed(self.frames_to_use):
            rel_time = np.round( 1e-9*(history_frames[frame_ind]["timestamp"] - curr_time), 2 ) # seconds
            agents = history_agents[frame_ind]

            vehicle_agents    = filter_vehicle_agents_by_labels(agents, self.filter_agents_threshold)
            for agent in vehicle_agents:
                add_agent_info_to_dict(agent, rel_time, vehicle_dict)

            nonvehicle_agents = filter_nonvehicle_agents_by_labels(agents, self.filter_agents_threshold)
            for agent in nonvehicle_agents:
                add_agent_info_to_dict(agent, rel_time, other_agent_dict)

        vehicles     = []
        for traj in vehicle_dict.values():
            traj = np.array(traj)
            states_in_view = self._in_view( self._transform_to_local_frame(ego_x, ego_y, ego_yaw, traj[:, 1:3]) ,
                                            *self.in_view_bounds)
            if np.any(states_in_view):
                vehicles.append(traj[states_in_view, :])

        other_agents = []
        for traj in other_agent_dict.values():
            traj = np.array(traj)
            states_in_view = self._in_view( self._transform_to_local_frame(ego_x, ego_y, ego_yaw, traj[:, 1:3]) ,
                                            *self.in_view_bounds)
            if np.any(states_in_view):
                other_agents.append(traj[states_in_view, :])

        return vehicles, other_agents

    def _associate_traffic_lights(self, history_tl_faces, lane_strs, lane_seg_lens):
        # List of traffic lights for which we have annotations at this time.
        active_tl_ids = set(filter_tl_faces_by_status(history_tl_faces[0], "ACTIVE")["face_id"].tolist())

         # List of boolean mask arrays, where tl_red_arr[i] corresponds to lane[i].
         # The mask indicates, for each state in the lane, whether the traffic light is red.
        tl_red_arr = []

        for (lane_str, lane_seg_len) in zip(lane_strs, lane_seg_lens):
            tokens = lane_str.split("_")
            is_red_tl_by_state = [] # boolean mask for this specific lane

            for ind_token, token in enumerate(tokens):
                lane_tl_inds = set(self.mapAPI.get_lane_traffic_control_ids(token)) # which TLs affect this lane

                # Iterate over TLs that are both active and affect this lane, checking if any are red.
                is_red_tl = False # whether the red TL applies to this lane token
                for tl_id in lane_tl_inds.intersection(active_tl_ids):
                    tl_color = self.mapAPI.get_color_for_face(tl_id)
                    if tl_color == TLFacesColors.RED.name:
                        is_red_tl = True
                is_red_tl_by_state.extend([is_red_tl] * lane_seg_len[ind_token])

            tl_red_arr.append(np.array(is_red_tl_by_state, dtype=np.bool))

        return tl_red_arr

    def get_context(self, sample_token, instance_token, split_name):
        if instance_token != "track_-1":
            raise NotImplementedError(f"Only handling ego vehicle (-1) at the moment.  Saw unexpected track_id: {instance_token}")

        # sample token has the form "scene_{idx}_frame_{idx}"
        scene_idx = int( sample_token.split("scene_")[-1].split("_frame")[0] )
        frame_idx = int( sample_token.split("frame_")[-1] )

        # Get history information for this scene and frame.
        # Format: index 0 is the current timestamp, -1 is the earliest timestamp.
        # history_frames[i]   -> single item with timestamp, ego pose, etc.
        # history_agents[i]   -> array with variable size, each entry with track id, pose, extent, velocity, label, etc.
        # history_tl_faces[i] -> array with variable size, each entry with traffic light id and status.
        (history_frames, _, history_agents, _, history_tl_faces, _) = \
            self._get_frame(scene_idx, frame_idx, split_name)

        if len(history_frames) != 11:
            raise ValueError(f"Expected to get 11 frames of history (1 second) but got {len(history_frames)} frames.")
        elif len(history_frames) != len(history_agents) != len(history_tl_faces):
            raise ValueError(f"Inconsistency in frames vs. agents vs. tl_faces : {len(history_frames), len(history_agents), len(history_tl_faces)}")
        else:
            pass

        x, y, yaw = self._get_ego_pose_from_frame(history_frames[0])

        # Lanes in the region:
        (lane_strs, lane_arrs, lane_seg_lens)  = \
         self._get_candidate_lanes(x, y, yaw)

        # Traffic lights
        red_traffic_lights = self._associate_traffic_lights(history_tl_faces, lane_strs, lane_seg_lens)

        # Agents (based off prediction/input_representation/agents.py in nuscenes-devkit)
        vehicles, other_agents = self._get_nearby_agents(history_frames, history_agents)

        sc = SceneContext(x=x,
                          y=y,
                          yaw=yaw,
                          lanes=lane_arrs,
                          vehicles=vehicles,
                          other_agents=other_agents,
                          red_traffic_lights=red_traffic_lights)
        return sc

if __name__ == "__main__":
    lcp = L5KitContextProvider()
    test_inst_str = "track_-1"

    # Scene selection
    #scene_num = 0    # no red TL present
    scene_num = 4    # red TL present

    # Frame range (for animation below)
    frame_range = range(50, 240, 10)

    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure()

    for frame_num in frame_range:
        test_samp_str = f"scene_{scene_num}_frame_{frame_num}"
        sc = lcp.get_context(test_samp_str, test_inst_str, "test")

        plt.clf()
        sc.plot()
        lcp.plot_view(sc.x, sc.y, sc.yaw)
        plt.axis('equal')
        plt.draw(); plt.pause(1.)

    plt.ioff()
    plt.show()
