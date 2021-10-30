import numpy as np
import pickle
import overpy
from abc import abstractmethod
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter

MPH_TO_MPS = 0.44704 # mps per mph
KPH_TO_MPS = 0.27778 # mps per kph

@dataclass(frozen=True)
class SceneContext:
    # Current pose of the predicted vehicle in world frame.
    x              : float
    y              : float
    yaw            : float

    # List of lane arrays ([x, y, theta, v])
    lanes          : List[np.ndarray]

    # List of pose trajectories ([t, x, y, theta]) of nearby vehicle and other relevant agents.
    # t is given in relative time (i.e. t = 0 means at the current prediction timestamp).
    vehicles        : List[np.ndarray]
    other_agents    : List[np.ndarray] # catch-all for pedestrians, animals, non-vehicle objects

    # List that describes how lanes are impacted by red traffic lights.
    # Each entry indicates which indices of the lane array are impacted by
    # red traffic lights (i.e. where one should come to a stop).
    red_traffic_lights : List[np.ndarray]

    def plot(self):
        lane_colors = [x for x in mcolors.TABLEAU_COLORS.keys()]
        lane_colors.remove("tab:red")

        zip_obj = zip(self.lanes, self.red_traffic_lights)
        for ind_lane, (curr_lane, curr_red_tl) in enumerate(zip_obj):
            inds_red_tl    = np.ravel(np.argwhere(curr_red_tl))
            inds_no_red_tl = np.ravel(np.argwhere(~curr_red_tl))

            plt.plot(curr_lane[inds_red_tl,0], curr_lane[inds_red_tl,1], linestyle='dotted', color="tab:red")
            plt.plot(curr_lane[inds_no_red_tl,0], curr_lane[inds_no_red_tl,1], linestyle='solid',
                     color=lane_colors[ind_lane % len(lane_colors)])

            plt.plot(curr_lane[ 0,0], curr_lane[ 0,1], 'bx')
            plt.plot(curr_lane[-1,0], curr_lane[-1,1], 'gx')

        # Plot other agents in the scene.
        alpha_fn = lambda t_rel: np.clip(1.0 + t_rel, 0., 1.) # t_rel assumed to lay within -1.0 to 0.0 s
        for veh in self.vehicles:
            for pose_ind in range(veh.shape[0]):
                plt.plot(veh[pose_ind, 1], veh[pose_ind, 2], 'go', alpha=alpha_fn(veh[pose_ind, 0]))
        for agt in self.other_agents:
            for pose_ind in range(agt.shape[0]):
                plt.plot(agt[pose_ind, 1], agt[pose_ind, 2], 'bo', alpha=alpha_fn(agt[pose_ind, 0]))

        # Plot the pose information of the agent being predicted.
        plt.plot(self.x, self.y, 'ro')
        dx, dy = 2*np.cos(self.yaw), 2*np.sin(self.yaw)
        plt.arrow(self.x, self.y, dx, dy, color='r')

class ContextProviderBase:
    # This base class provides a basic API to query the scene context, which include
    # lanes, agents, and traffic lights in view,for a given dataset instance.

    def __init__(self):
        # Bounds in local frame of the predicted agent.  Matches rasterization settings.
        self.in_view_bounds = [-10., -25., 40., 25.] # x_min, y_min, x_max, y_max (meters)
        self.view_radius = 50. # TODO
        self.lane_association_radius = 4. # a value greater than the typical lane width (meters)
        self.max_lane_yaw_deviation   = np.pi/2.
        self.secs_of_hist = 1.0           # how many seconds back for which to provide agent/TL info

    def _init_overpass(overpass_url):
        # Overpass API used to query speed limits (only once, for _generate_map_cache).
        if overpass_url is None:
            # Remote server (default but times out if you make too many queries.)
            self.overpass_api = overpy.Overpass()
        else:
            # Local server (unlimited number of requests but requires a Docker container - see below).
            self.overpass_api = overpy.Overpass(url=overpass_url)

            # Note: The default Overpass server kept timing out, so I followed the suggestion
            # to download a local OSM map copy and use Docker.
            # The Docker image/container must be constructed from scratch per region/map.
            # Docker git repo: https://github.com/mediasuitenz/docker-overpass-api
            # url if this is used: "http://localhost/api/interpreter"

    @abstractmethod
    def _generate_map_cache(self, centerline_cache_path):
        # This function generates a pickle file at centerline_cache_path (if not already generated).
        # The pickle file contains a dictionary of discretized centerlines with speed limits.
        raise NotImplementedError

    def _load_map_cache(self, centerline_cache_path):
        # Load the aforementioned discretized centerline dictionary from the cache pickle file.
        self.centerline_dict = pickle.load(open(str(centerline_cache_path), "rb"))

    @abstractmethod
    def get_context(self, sample_token, instance_token, split_name):
        """ Main function to provide scene context information for a dataset instance.

            The sample_token and instance_token are specified to identify a timestamp and agent.
            The split_name is required for the L5Kit dataset where the dataset source (zarr)
            must be identified.

            Returns a SceneContext object as detailed above.
        """
        raise NotImplementedError

    @staticmethod
    def _identify_candidate_lane_traversals(lane_adjacency, lane_indegree):
        """ This function uses an adjacency list and indegree measure of lane tokens.
            Using this information, it returns the valid lane traversals in this graph using DFS.

            lane_adjacency = Dict: lane_token -> [one-step reachable lane_tokens]
            lane_indegree  = Dict: lane_token -> indegree (0 indicates this is a "starting" lane_token in the graph)
        """

        lane_traversals = []
        def lane_dfs_helper(curr_lane_id, lane_prefix):
            nonlocal lane_traversals

            if len(lane_adjacency[curr_lane_id]) == 0:
                # Base case: reached end of a lane traversal.
                lane_traversals.append(lane_prefix)
                return

            for lane_n in lane_adjacency[curr_lane_id]:
                if lane_n in lane_prefix:
                    # Cyclical case: we've looped back to a prior lane_token.
                    # To handle this, we opt to add the traversal before the repeating node
                    # and end the recursion here as another base case.
                    lane_traversals.append(lane_prefix)
                else:
                    # Recursive case: continue exploring lane traversals.
                    lane_dfs_helper(lane_n, f"{lane_prefix}_{lane_n}")

        # We explore starting from all lane_tokens with indegree 0.
        # This may backfire in the case of cyclic graphs where some components may not have
        # any nodes with indegree 0.
        for (lane_id, indegree) in lane_indegree.items():
            if indegree == 0: # lane traversal start
                lane_dfs_helper(lane_id, f"{lane_id}")

        # NOTE: for now, we ignore cyclic lane traversals.
        # In other words, we only consider lane traversals such that the starting lane token
        # does not have a predecessor, which is usually the case with typical intersections.
        # My guess is that cyclical cases are usually at places like u-turns / roundabouts,
        # which are relatively rare (and only seem to pop up in the L5 dataset).
        """
        # This sanity check is to handle cases where entire section of the graph
        # are ignored due to cyclic issue above.
        for lane_token in lane_adjacency.keys():
            token_found = False
            for traversal in lane_traversals:
                if lane_token in traversal:
                    token_found = True
                    break
            if not token_found:
                print(f"Did not explore lane token {lane_token} using DFS!  Check it out.")
                import pdb; pdb.set_trace()
        """

        return lane_traversals

    @staticmethod
    def _get_lane_array_from_traversal(traversal_str, cline_dict):
        """ This function returns the actual lane traversal array ([x, y, theta, v]_{i=1}^{N_{lane}}) given
            a string encoding the lane_tokens making up the traversal (traversal_str).

            cline_dict is the centerline_dict that maps lane_tokens to their corresponding arrays,
            passed as an argument here to avoid dealing with the map location key used by NuScenes.
        """
        return np.concatenate(([cline_dict[lt] for lt in traversal_str.split("_")]), axis=0), \
               np.array([cline_dict[lt].shape[0] for lt in traversal_str.split("_")])

    def _truncate_lanes_in_view(self,
                                x, y, yaw,
                                radius,
                                cand_lane_traversal_str,
                                cand_lane_traversal_arr,
                                cand_lane_traversal_seg_lens):
        """ This function takes in a pose and candidate lane traversals and cuts down to a
            final set of lane traversals in view and sufficiently close to the query pose.

            (x, y, yaw): query pose about which the view region is centered
            radius     : lane association radius for sufficient proximity to the query pose

            The candidate lane traversals are indicated by the following 3-tuple:
            cand_lane_traversal_str      = List[strings]    , indicates the lane tokens in the traversal
            cand_lane_traversal_arr      = List[np.ndarrays], contains the states (x,y,theta,v) in the traversal
            cand_lane_traversal_seg_lens = List[np.ndarrays], indicates the # states (in arr) per lane token (in str)
        """
        # Final set of truncated traversals.
        final_lane_traversal_str = []
        final_lane_traversal_arr = []
        final_lane_traversal_seg_lens = []

        # Proximity checking helper function.  Essentially a min "pose distance" approach.
        def check_lane_proximity(x, y, yaw, radius, lane_arr):
            query_position = np.array([[x, y]])
            closest_idx = np.argmin( np.linalg.norm(lane_arr[:, :2] - query_position, axis=-1) )
            is_proximal = True

            # Don't consider lanes that are going in the opposite direction.
            # Threshold: keep if abs(yaw error) < 90 deg.
            yaw_diff = self._bound_angle_within_pi(lane_arr[closest_idx, 2] - yaw)
            if np.abs(yaw_diff) >= self.max_lane_yaw_deviation:
                is_proximal = False

            # Skip lanes for which the projected closest point exceeds the queried radius.
            error_xy = lane_arr[closest_idx, :2] - query_position
            if np.linalg.norm(error_xy) > radius:
                is_proximal = False

            return is_proximal, closest_idx

        zip_obj = zip(cand_lane_traversal_str, cand_lane_traversal_arr, cand_lane_traversal_seg_lens)
        for (lane_str, lane_arr, lane_seg_lens) in zip_obj:
            is_proximal, closest_idx = check_lane_proximity(x, y, yaw, radius, lane_arr)

            if not is_proximal:
                continue

            # Use closest_idx to cut lane_str, lane_arr, lane_seg_lens
            lane_arr_rel_xy = self._transform_to_local_frame(x, y, yaw, lane_arr[:, :2])
            lane_arr_in_view = self._in_view(lane_arr_rel_xy, *self.in_view_bounds)

            # Sanity check: this should never trigger in theory.
            if not lane_arr_in_view[closest_idx]:
                print("Closest point is not in view, yet is_proximal is True.  Check this out.")
                import pdb; pdb.set_trace()

            # Identify start and end indices of lane traversal in view.
            # We pick start_idx = closest_idx since we don't really care about variation
            # in prior lane segments, only variation in future lane segments.
            start_idx = closest_idx
            end_idx = np.nan
            if closest_idx == lane_arr.shape[0] - 1:
                # We are next to the end of the lane, this is useless for prediction.
                continue
            else:
                end_idx = closest_idx
                while(end_idx < lane_arr.shape[0] - 1):
                    if(lane_arr_in_view[end_idx+1]):
                        end_idx += 1
                    else:
                        break

            # Use start_idx and end_idx to get final lane traversal in view after truncation.
            lane_str, lane_arr, lane_seg_lens = \
                self._truncate_by_idx(lane_str, lane_arr, lane_seg_lens, start_idx, end_idx)

            # Only add a truncated lane traversal if it's a new one.
            if lane_str not in final_lane_traversal_str:
                final_lane_traversal_str.append(lane_str)
                final_lane_traversal_arr.append(lane_arr)
                final_lane_traversal_seg_lens.append(lane_seg_lens)

        return final_lane_traversal_str, final_lane_traversal_arr, final_lane_traversal_seg_lens

    @staticmethod
    def _truncate_by_idx(lane_str, lane_arr, lane_seg_lens, start_idx, end_idx):

        tokens = lane_str.split("_")
        token_list = []
        for (token, seg_len) in zip(tokens, lane_seg_lens):
            token_list.extend( [token]*seg_len )

        if(end_idx == lane_arr.shape[0] - 1):
            lane_arr_trunc   = lane_arr[start_idx:]
            token_list_trunc = token_list[start_idx:]
        else:
            lane_arr_trunc = lane_arr[start_idx:(end_idx+1)] # inclusive of the endpoint
            token_list_trunc = token_list[start_idx:(end_idx+1)]

        token_count_dict = Counter(token_list_trunc)
        lane_str_trunc = "_".join([x for x in tokens if x in token_count_dict.keys()])
        lane_seg_lens_trunc = [token_count_dict[x] for x in tokens if x in token_count_dict.keys()]

        # Sanity checks post truncation.
        if lane_str_trunc not in lane_str:
            print("Truncated lane string is not a substring, check this out.")
            import pdb; pdb.set_trace()
        if np.sum(lane_seg_lens_trunc) != len(lane_arr_trunc):
            print("Mismatch in length of lane segment lengths vs. total array length.")
            import pdb; pdb.set_trace()

        return lane_str_trunc, lane_arr_trunc, lane_seg_lens_trunc

    @staticmethod
    def _transform_to_local_frame(x_world, y_world, yaw_world, points_world):
        # Transforms a set of points in global/world frame (points_world)
        # into the local frame described by the pose ({x,y,yaw}_world).

        R_local_to_world = np.array([[ np.cos(yaw_world), -np.sin(yaw_world)],
                                     [ np.sin(yaw_world),  np.cos(yaw_world)]])
        t_local_in_world = np.array([x_world, y_world]).reshape((2,1))

        R_world_to_local =  R_local_to_world.T
        t_world_in_local = -R_local_to_world.T @ t_local_in_world

        T_world_to_local = np.block([
                                      [R_world_to_local, t_world_in_local],
                                      [0., 0., 1.]
                                    ])

        N_pts = points_world.shape[0]
        points_world = np.concatenate((
                                        points_world[:,:2].T,
                                        np.ones((1, N_pts))
                                      ), axis=0)
        points_local = T_world_to_local @ points_world
        points_local = points_local[:2, :].T
        return points_local

    @staticmethod
    def _transform_poses_to_local_frame(x_world, y_world, yaw_world, poses_world):
        # Transforms poses in global/world frame (poses_world)
        # into the local frame described by the pose ({x,y,yaw}_world).
        xy_local  = ContextProviderBase._transform_to_local_frame(x_world, y_world, yaw_world, poses_world[:, :2])
        yaw_local = ContextProviderBase._bound_angle_within_pi(poses_world[:, 2] - yaw_world)

        poses_local = np.copy(poses_world)
        poses_local[:, :2] = xy_local
        poses_local[:,  2] = yaw_local

        return poses_local

    @staticmethod
    def _bound_angle_within_pi(ang):
       return (ang + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _in_view(pts, x_min, y_min, x_max, y_max):
        # This function determines if the points described by pts (N x 2) falls inside the
        # bounding box defined by (x_min, y_min, x_max, y_max).  For N pts, the result
        # is a boolean N-vector.
        xs = pts[:, 0]
        ys = pts[:, 1]
        in_x_bounds = np.logical_and( xs >= x_min, xs <= x_max )
        in_y_bounds = np.logical_and( ys >= y_min, ys <= y_max )
        return np.logical_and(in_x_bounds, in_y_bounds)

    def plot_view(self, x, y, yaw):
        # Helper function to view the oriented viewing region around the predicted agent
        # centered at (x, y, yaw) in the world frame.
        ax = plt.gca()
        l1, v1, l2, v2 = self.in_view_bounds
        corner_1 = np.array([[ x + l1 * np.cos(yaw) + v1 * np.cos(yaw+np.pi/2.),
                               y + l1 * np.sin(yaw) + v1 * np.sin(yaw+np.pi/2.) ]])
        corner_2 = np.array([[ x + l1 * np.cos(yaw) + v2 * np.cos(yaw+np.pi/2.),
                               y + l1 * np.sin(yaw) + v2 * np.sin(yaw+np.pi/2.) ]])
        corner_3 = np.array([[ x + l2 * np.cos(yaw) + v1 * np.cos(yaw+np.pi/2.),
                               y + l2 * np.sin(yaw) + v1 * np.sin(yaw+np.pi/2.) ]])
        corner_4 = np.array([[ x + l2 * np.cos(yaw) + v2 * np.cos(yaw+np.pi/2.),
                               y + l2 * np.sin(yaw) + v2 * np.sin(yaw+np.pi/2.) ]])
        poly_xy = np.concatenate((corner_1, corner_2, corner_4, corner_3), axis=0)
        ax.add_patch( plt.Polygon(poly_xy, color='c', fill=False) )

    def _get_average_speed_overpass(self, latlon_queries, around_radius_m=10):
        # Gets the local speed limit for a set of latlon_queries with search radius around_radius_m.

        # query_str seeks to find ways around the query latlon points with a speed limit defined (see OSM/Overpass API).
        query_str = "".join(["way(around:%d,%s,%s)[maxspeed];" % (around_radius_m, lat, lon) for (lat, lon) in latlon_queries])
        query_str = "[out:json][timeout:25];(" + query_str + ");out;"

        try:
            result = self.overpass_api.query(query_str)
        except:
            print(f"Could not query speed limit ways")
            raise

        if len(result.ways) == 0:
            # Try a larger radius search if default of 10 m is too small.
            # Going too much larger than this could be problematic (e.g. a small road next to a highway).
            for query_radius in np.arange(50, 101, 50):
                result = self.overpass_api.query( query_str.replace("around:%d" % around_radius_m, "around:%d" % query_radius) )
                if len(result.ways) > 0:
                    break

        speed_limits_mps = [] # speed limits for each valid way below

        for way in result.ways:
            if "highway" not in way.tags.keys():
                continue # Don't get the speed limit of railways, waterways, etc.  Only roads!

            # Reference: https://wiki.openstreetmap.org/wiki/Key:maxspeed
            sl = way.tags["maxspeed"]

            # Convert to meters per second if required.
            if 'mph' in sl:
                sl_mps = int(sl.split('mph')[0]) * MPH_TO_MPS
            elif sl.isnumeric():
                # Default kph
                sl_mps = int(sl) * KPH_TO_MPS
            else:
                raise ValueError(f"Unexpected speed limit type: {sl}")

            speed_limits_mps.append(sl_mps)

        if speed_limits_mps:
            return np.mean(speed_limits_mps) # Use the mean speed limit of nearby ways.
        else:
            return np.nan                    # No reliable speed limit information found.
