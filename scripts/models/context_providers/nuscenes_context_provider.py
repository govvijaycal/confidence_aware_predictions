import os
import sys
import numpy as np
import time
import pickle
from pathlib import Path
from tqdm import tqdm
from pyquaternion import Quaternion
import pyproj

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import load_all_maps
from nuscenes.prediction.input_representation.agents import reverse_history, add_present_time_to_history

from context_provider_base import ContextProviderBase, SceneContext

# Map coordinates in WGS84 (EPSG:3857) were taken from the link:
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/map_expansion/map_api.py#L46
MAP_LATLON_DICT = {}
MAP_LATLON_DICT['boston-seaport']           = [42.336849169438615, -71.05785369873047]
MAP_LATLON_DICT['singapore-onenorth']       = [1.2882100868743724, 103.78475189208984]
MAP_LATLON_DICT['singapore-hollandvillage'] = [1.2993652317780957, 103.78217697143555]
MAP_LATLON_DICT['singapore-queenstown']     = [1.2782562240223188, 103.76741409301758]
####################################################################################################
class NuScenesContextProvider(ContextProviderBase):
    def __init__(self, dataroot="/media/data/nuscenes-data/"):
        super().__init__(overpass_url="http://localhost/api/interpreter")
        nusc = NuScenes('v1.0-trainval', dataroot=dataroot)
        self.helper = PredictHelper(nusc)
        self.maps = load_all_maps(self.helper)

        centerline_cache_path = Path( f"{dataroot}/maps/centerline_cache.pkl" )
        if not centerline_cache_path.exists():
            print('Generating and saving Nuscenes centerlines cache.')
            self._generate_map_cache(centerline_cache_path)

        print('Loading Nuscenes centerlines cache.')
        self._load_map_cache(centerline_cache_path)

    def _xy_to_latlon(self, x, y, lat0, lon0):
        # Modified based on code snippets given here:
        # https://gis.stackexchange.com/questions/212723/how-can-i-convert-lon-lat-coordinates-to-x-y

        # Convert latlon origin (EPSG:4326) to XY in WGS84 Mercator projection (EPSG:3857)
        x0, y0 = self.trans_wgs_to_mer.transform(lat0, lon0)

        # Convert local XY to "global" XY in WGS84 Mercator projection (EPSG:3857)
        x_m, y_m = x + x0, y + y0

        # Convert "global" XY back to latlon (EPSG:4326).
        lat, lon = self.trans_mer_to_wgs.transform(x_m, y_m)

        return lat, lon

    def _generate_map_cache(self, centerline_cache_path):
        map_keys = self.maps.keys() # Use all maps (if using the remote server).
        # map_singapore_keys = [k for k in map_keys if 'singapore' in k]  # local server with Singapore map
        # map_boston_keys = [k for k in map_keys if 'boston' in k]        # local server with Boston map

        # WGS84 Pseudo-Mercator Easting(X)/Northing(Y) <-> WGS84 (lat/lon) projections.
        self.trans_wgs_to_mer = pyproj.Transformer.from_crs(4326, 3857)
        self.trans_mer_to_wgs = pyproj.Transformer.from_crs(3857, 4326)

        # Get centerline/speed limit information per map.
        for k in map_keys:
            lanes = self.maps[k].lane + self.maps[k].lane_connector
            lane_ids = [l['token'] for l in lanes]
            curr_map_dict = self.maps[k].discretize_lanes(lane_ids, resolution_meters=0.5)

            print(f"\tMap:{k}")
            # Get the latlon for this map.
            curr_latlon = MAP_LATLON_DICT[k]

            for lane_id in tqdm(curr_map_dict.keys()):
                # Get the speed limit average and add to the lane entry.
                curr_lane_array = curr_map_dict[lane_id]

                wpt_queries = [ curr_lane_array[0],                           # first lane waypoint
                                curr_lane_array[ len(curr_lane_array) // 2],  # middle lane waypoint
                                curr_lane_array[-1]                           # last lane waypoint
                              ]
                latlon_queries = [self._xy_to_latlon(xq, yq, *curr_latlon) for (xq, yq, _) in wpt_queries]

                v_avg_mps = self._get_average_speed_overpass(latlon_queries)

                curr_map_dict[lane_id] = np.concatenate((np.array(curr_map_dict[lane_id]), \
                                                         v_avg_mps*np.ones( (len(curr_map_dict[lane_id]), 1) ) \
                                                        ), axis = 1 \
                                                       )
            savepath = str(centerline_cache_path).replace(".pkl", f"_{k}.pkl")
            pickle.dump( curr_map_dict, open(savepath, "wb") )

        # Combine all cached maps into one for easier later use.
        centerline_dict = {}
        for k in map_keys:
            path_map = str(centerline_cache_path).replace(".pkl", f"_{k}.pkl")
            centerline_dict[k] = pickle.load(open(path_map, "rb"))
        pickle.dump(centerline_dict, open(str(centerline_cache_path), "wb"))

    @staticmethod
    def _get_pose_from_annotation(annotation):
        x, y, _ = annotation['translation']
        yaw     = quaternion_yaw( Quaternion(annotation['rotation']) )
        return x, y, yaw

    def _get_candidate_lanes(self, x, y, yaw, radius, mapname):
        # (1) Get the lane graph in the local vicinity (upper-bounding the actual view region).
        lanes = self.maps[mapname].get_records_in_radius(x, y, self.view_radius, ["lane", "lane_connector"])
        lanes = lanes["lane"] + lanes["lane_connector"]

        lane_adjacency = {lane_tkn : [] for lane_tkn in lanes} # lane_id : [lane_id_next_1, lane_id_next_2, ...]
        lane_indegree  = {lane_tkn : 0  for lane_tkn in lanes}  # lane_id : indegree, i.e. # of incoming lanes

        for lane_tkn in lanes:
            next_lane_tokens = self.maps[mapname].get_outgoing_lane_ids(lane_tkn)

            for lane_n_tkn in next_lane_tokens:
                if lane_n_tkn not in lanes:
                    continue # lane_n_tkn is out of the view

                # lane_n_tkn is in view, mark the edge lane_tkn -> lane_n_tkn
                lane_indegree[lane_n_tkn] += 1
                lane_adjacency[lane_tkn].append(lane_n_tkn)

         # (2) Get the possible lane traversals (as concatenated strings of lane tokens).
        cand_lane_traversal_str = self._identify_candidate_lane_traversals(lane_adjacency, lane_indegree)

        # (3) For each traversal, get the corresponding array of states and lane segment lengths (per lane token).
        cand_lane_traversal_arr = []
        cand_lane_traversal_seg_lens = []
        for l in cand_lane_traversal_str:
            out = self._get_lane_array_from_traversal(l, self.centerline_dict[mapname])
            cand_lane_traversal_arr.append(out[0])
            cand_lane_traversal_seg_lens.append(out[1])

        # (4) Cut the lane traversals to those in the view region and sufficiently "close" to the query pose.
        final_lane_traversal_str, \
        final_lane_traversal_arr, \
        final_lane_traversal_seg_lens = \
        self._truncate_lanes_in_view(x, y, yaw, radius,
                                     cand_lane_traversal_str,
                                     cand_lane_traversal_arr,
                                     cand_lane_traversal_seg_lens)

        return final_lane_traversal_arr # List with each entry including states (x, y, theta, v) along that traversal

    def _get_nearby_agents(self, sample_token, instance_token):
        vehicles       = []
        other_agents   = []

        present_time = self.helper.get_annotations_for_sample(sample_token)
        history = self.helper.get_past_for_sample(sample_token, self.secs_of_hist, in_agent_frame=False, just_xy=False)
        history = reverse_history(history)
        history = add_present_time_to_history(present_time, history)

        annotation = self.helper.get_sample_annotation(instance_token, sample_token)
        x, y, yaw = self._get_pose_from_annotation(annotation)
        current_tm = self.helper._timestamp_for_sample(sample_token)

        for agent_token in history.keys():
            if agent_token == instance_token:
                continue # don't double count the ego vehicle

            agent_entries = history[agent_token]
            tms   = [self.helper._timestamp_for_sample(ann['sample_token']) for ann in agent_entries]
            tms   = np.array([1e-6 * (x - current_tm)  for x in tms])
            poses = np.array([ self._get_pose_from_annotation(ann) for ann in agent_entries ])
            traj = np.column_stack((tms, poses))

            states_in_view = self._in_view( self._transform_to_local_frame(x, y, yaw, traj[:, 1:3]) ,
                                            *self.in_view_bounds)
            if ~np.any(states_in_view):
                continue

            category = agent_entries[0]["category_name"]
            if "vehicle" in category:
                vehicles.append( traj[states_in_view, :] )
            else:
                other_agents.append( traj[states_in_view, :] )

        return vehicles, other_agents

    def get_context(self, sample_token, instance_token, split_name=None):
        # Specialization of the get_context method for the NuScenes dataset.

        annotation = self.helper.get_sample_annotation(instance_token, sample_token)
        x, y, yaw = self._get_pose_from_annotation(annotation)
        mapname    = self.helper.get_map_name_from_sample_token(sample_token)

        # Lanes in the region:
        lane_arrs = self._get_candidate_lanes(x, y, yaw, self.lane_association_radius, mapname)

        red_traffic_lights = [np.array([False]*len(x)) for x in lane_arrs]

        # Agents (based off prediction/input_representation/agents.py in nuscenes-devkit)
        vehicles, other_agents = self._get_nearby_agents(sample_token, instance_token)

        sc = SceneContext(x=x,
                          y=y,
                          yaw=yaw,
                          lanes=lane_arrs,
                          vehicles=vehicles,
                          other_agents=other_agents,
                          red_traffic_lights=red_traffic_lights)
        return sc

if __name__ == "__main__":
    ncp = NuScenesContextProvider()
    test_inst_str = "4d87aaf2d82549969f1550607ef46a63"
    test_samp_str = "faf2ea71b30941329a3c3f3866cec714"
    sc = ncp.get_context(test_samp_str, test_inst_str)

    import matplotlib.pyplot as plt
    plt.figure()
    sc.plot()
    ncp.plot_view(sc.x, sc.y, sc.yaw)
    plt.axis('equal')
    plt.show()
