from pathlib import Path
import numpy as np

import time
import overpy
import pyproj
import pickle

from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import load_all_maps

# Map coordinates in WGS84 (EPSG:3857) were taken from the link:
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/map_expansion/map_api.py#L46
MAP_LATLON_DICT = {}
MAP_LATLON_DICT['boston-seaport']           = [42.336849169438615, -71.05785369873047]
MAP_LATLON_DICT['singapore-onenorth']       = [1.2882100868743724, 103.78475189208984]
MAP_LATLON_DICT['singapore-hollandvillage'] = [1.2993652317780957, 103.78217697143555]
MAP_LATLON_DICT['singapore-queenstown']     = [1.2782562240223188, 103.76741409301758]

MPH_TO_MPS = 0.44704 # mps per mph
KPH_TO_MPS = 0.27778 # mps per kph

class NuScenesContextProvider:
    def __init__(self, dataroot="/media/data/nuscenes-data/"):
        nusc = NuScenes('v1.0-trainval', dataroot=dataroot)
        self.helper = PredictHelper(nusc)
        self.maps = load_all_maps(self.helper)

        # WGS84 Pseudo-Mercator Easting(X)/Northing(Y) <-> WGS84 (lat/lon) projections.
        self.trans_wgs_to_mer = pyproj.Transformer.from_crs(4326, 3857)
        self.trans_mer_to_wgs = pyproj.Transformer.from_crs(3857, 4326)

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

    def _get_average_speed_overpass(self, latlon_queries, around_radius_m=10):
        # Gets the local speed limit for a set of latlon_queries.

        # query_str concatenates the resultant speed limit constrained ways for every latlon query.
        query_str = "".join(["way(around:%d,%s,%s)[maxspeed];" % (around_radius_m, lat, lon) for (lat, lon) in latlon_queries])
        query_str = "[out:json][timeout:25];(" + query_str + ");out;"

        if ~hasattr(self, "overpass_api"):
            self.overpass_api = overpy.Overpass()
            # self.overpass_api = overpy.Overpass(url="http://localhost/api/interpreter")

            # Note: The default Overpass server kept timing out, so I followed this suggestion to download a local
            # OSM map copy and use docker.  Need to do this twice for Singapore / Boston.
            # Leaving the default code for now, but if this is used again, may need to use the docker approach.
            # Docker git repo: https://github.com/mediasuitenz/docker-overpass-api

        try:
            result = self.overpass_api.query(query_str)
        except Exception as e:
            print(f"\t\t{e}")

        if len(result.ways) == 0:
            # Try a larger radius search if default of 10 m is too small.
            result_found = False
            for query_radius in np.arange(50, 501, 50):
                result = self.overpass_api.query( query_str.replace("around:%d" % around_radius_m, "around:%d" % query_radius) )
                if len(result.ways) > 0:
                    result_found = True
                    break
        else:
            result_found = True

        if not result_found:
            print("Could not find the local speed limit!")
            import pdb; pdb.set_trace()

        speed_limits_mps = []

        for way in result.ways:
            # Reference: https://wiki.openstreetmap.org/wiki/Key:maxspeed
            sl = way.tags["maxspeed"]

            if 'mph' in sl:
                sl_mps = int(sl.split('mph')[0]) * MPH_TO_MPS
            elif sl.isnumeric():
                # Default kph
                sl_mps = int(sl) * KPH_TO_MPS
            else:
                raise ValueError(f"Unexpected speed limit type: {sl}")

            speed_limits_mps.append(sl_mps)

        return np.mean(speed_limits_mps)

    def _load_map_cache(self, centerline_cache_path):
        self.centerline_dict = pickle.load(open(str(centerline_cache_path), "rb"))

    def _generate_map_cache(self, centerline_cache_path):
        map_keys = self.maps.keys() # Use all maps (if using the remote server).
        # map_singapore_keys = [k for k in map_keys if 'singapore' in k]
        # map_boston_keys = [k for k in map_keys if 'boston' in k]

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

    def get_context(self):
        raise NotImplementedError

class L5KitContextProvider:
    def __init__(self):
        raise NotImplementedError

    def get_context(self):
        raise NotImplementedError


if __name__ == "__main__":
    ncp = NuScenesContextProvider()
