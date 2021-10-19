import cv2
import numpy as np
import matplotlib.pyplot as plt
from .raster_common import convert_world_coords_to_pixels, cv2_subpixel, CV2_SHIFT, CV2_LINE_TYPE, angle_to_color

def extract_waypoints_from_topology(carla_topology, precision=0.5):
    # Taken from no_rendering_mode.py in carla/examples with small loop modification.
    # precision (m) is for interpolation resolution, similar to polyline interpolation approach taken by l5kit.
    topology = [x[0] for x in carla_topology]
    topology = sorted(topology, key=lambda w: w.transform.location.z)
    set_waypoints = []
    for waypoint in topology:
        waypoints = [waypoint]

        # Generate waypoints of a road id. Stop when road id differs
        nxt = waypoint.next(precision)
        if len(nxt) > 0:
            nxt = nxt[0]
            reached_end = False
            while not reached_end:
                waypoints.append(nxt)
                nxt = nxt.next(precision)
                if len(nxt) > 0:
                    nxt = nxt[0]
                    if nxt.road_id != waypoint.road_id:
                        # Add only the first waypoint on a new road then stop.
                        # This is to ensure good connectivity of segments.
                        waypoints.append(nxt)
                        reached_end = True
                else:
                    reached_end = True

        set_waypoints.append(waypoints)

    return set_waypoints

class RoadEntry:
    """ Contains centerline and boundary information for a single road segment,
        extracted from the carla map topology.  Used mostly as a data container for rendering.
    """
    def __init__(self, waypoints):
        self.parse_waypoints(waypoints)
        self.get_lane_boundaries()

    def parse_waypoints(self, waypoints):
        self.centerline = np.array([[waypoint.transform.location.x, -waypoint.transform.location.y] for waypoint in waypoints])
        self.widths = np.array([waypoint.lane_width for waypoint in waypoints])
        self.yaws = -np.radians( np.array([waypoint.transform.rotation.yaw for waypoint in waypoints]) )

    def get_lane_boundaries(self):
        hw = self.widths / 2.
        self.left_side  = self.centerline + np.column_stack([hw*np.cos(self.yaws+np.pi/2), hw*np.sin(self.yaws+np.pi/2)])
        self.right_side = self.centerline + np.column_stack([hw*np.cos(self.yaws-np.pi/2), hw*np.sin(self.yaws-np.pi/2)])

        self.x_min = min(np.amin(self.left_side[:,0]), np.amin(self.right_side[:,0]))
        self.y_min = min(np.amin(self.left_side[:,1]), np.amin(self.right_side[:,1]))
        self.x_max = max(np.amax(self.left_side[:,0]), np.amax(self.right_side[:,0]))
        self.y_max = max(np.amax(self.left_side[:,1]), np.amax(self.right_side[:,1]))

class SemanticRasterizer:
    """ Given historical information about the scene (traffic lights), renders the map information (roads, centerlines, traffic lights).
        This is heavily inspired by the nuscenes-devkit and l5kit repos.
    """
    def __init__(self,
                 carla_topology,
                 raster_size=(500, 500), # pixel height and width
                 raster_resolution=0.1,  # meters / pixel resolution
                 target_center_x = 100,     # pixels along x-axis where target center should be located
                 target_center_y = 250,     # pixels along y-axis where target center should be located
                 ):

        # Road Information.
        set_waypoints  = extract_waypoints_from_topology(carla_topology)                               # extract discretized topology from current map
        self.road_entries   = [RoadEntry(waypoints) for waypoints in set_waypoints]                    # road entries from discretized topology
        self.bounds = np.array([[re.x_min, re.y_min, re.x_max, re.y_max] for re in self.road_entries]) # bounding boxes per road entry

        # Raster Related Information.
        self.raster_height, self.raster_width = raster_size
        self.target_to_pixel = np.array([[1./raster_resolution,  0., target_center_x],\
                                         [0, -1./raster_resolution, target_center_y],\
                                         [0., 0., 1.]])

        # Raster Color Selection.
        # NOTE: crosswalks omitted since the Carla API doesn't
        #       seem to have a clean way to query for them.
        self.road_color = (255, 255, 255)
        self.tl_colors  = {'green' : (0, 128, 0),
                           'red'   : (128, 0, 0),
                           'yellow': (128, 128, 0)}

        # Setting a radius for plotting traffic lights.  Using 1.2 m as in carla_birdeye_view:
        # https://github.com/deepsense-ai/carla-birdeye-view/blob/f666b0f5c11e9a3eb29ea2f9465d6b5526ab1ae0/carla_birdeye_view/mask.py#L329
        self.tl_radius = int(1.2 / raster_resolution)

    def in_bounds(self, render_bounds):
        # Essentially this code just looks for a nonzero IoU between two bounding boxes for it to be "in_bounds".
        x_interval = np.minimum(self.bounds[:,2], render_bounds[2]) - np.maximum(self.bounds[:,0], render_bounds[0])
        y_interval = np.minimum(self.bounds[:,3], render_bounds[3]) - np.maximum(self.bounds[:,1], render_bounds[1])
        in_render_bounds = np.logical_and( x_interval > 0., y_interval > 0.)
        return np.nonzero(in_render_bounds)[0]

    def get_render_bounds(self, pixel_to_world):
        # Given the pixel_to_world transformation and fixed rasterization parameters,
        # this computes the enclosing pixel-axis aligned bounding boxes for the render area.
        render_x1, render_y1, _ = pixel_to_world @ np.array([0., 0., 1.])
        render_x2, render_y2, _ = pixel_to_world @ np.array([self.raster_width, 0., 1.])
        render_x3, render_y3, _ = pixel_to_world @ np.array([0., self.raster_height, 1.])
        render_x4, render_y4, _ = pixel_to_world @ np.array([self.raster_width, self.raster_height, 1.])

        xs = [render_x1, render_x2, render_x3, render_x4]
        ys = [render_y1, render_y2, render_y3, render_y4]

        render_xmin, render_xmax = np.amin(xs), np.amax(xs)
        render_ymin, render_ymax = np.amin(ys), np.amax(ys)

        return np.array([render_xmin, render_ymin, render_xmax, render_ymax])

    def rasterize(self, agent_history, target_agent_id, render_traffic_lights=True):
        img = np.zeros((self.raster_height, self.raster_width, 3), dtype=np.uint8)

        target_pose_current = agent_history.vehicles[target_agent_id].pose_history[-1]
        target_centroid = np.array(target_pose_current[:2])
        target_yaw      = target_pose_current[2]

        R_target_to_world = np.array([[np.cos(target_yaw), -np.sin(target_yaw)],\
                                      [np.sin(target_yaw),  np.cos(target_yaw)]])
        t_target_to_world = target_centroid.reshape(2, 1)

        R_world_to_target = R_target_to_world.T
        t_world_to_target = -R_target_to_world.T @ t_target_to_world

        world_to_pixel = self.target_to_pixel @ np.block([[R_world_to_target, t_world_to_target], [0., 0., 1.]])

        pixel_to_world = np.linalg.inv(world_to_pixel)

        render_bounds = self.get_render_bounds(pixel_to_world)
        indices_in_bounds = self.in_bounds(render_bounds)

        # Plot lane areas.
        for index in indices_in_bounds:
            road_entry = self.road_entries[index]
            lane_polygon = np.concatenate((road_entry.left_side, road_entry.right_side[::-1]), axis=0)
            lane_polygon_px = convert_world_coords_to_pixels(lane_polygon, world_to_pixel)
            lane_polygon_px = cv2_subpixel(lane_polygon_px)
            lane_polygon_px = np.expand_dims(lane_polygon_px, axis=0)
            cv2.fillPoly(img, [lane_polygon_px], self.road_color, shift=CV2_SHIFT, lineType=CV2_LINE_TYPE)

        # Plot crosswalks -> not annotated cleanly (unlike sidewalks) so skipped for now.

        if render_traffic_lights:
            # Plot traffic lights.
            for (tl_id, tl_info) in agent_history.traffic_lights.items():
                tl_xy = convert_world_coords_to_pixels(np.array(tl_info[:2]).reshape(1,2), world_to_pixel)
                tl_xy = np.round(tl_xy, 0).astype(np.int).flatten()
                cv2.circle(img, tuple(tl_xy), self.tl_radius, self.tl_colors[tl_info[2]], -1) # TL visualized.

        # Plot lane centerlines.
        for index in indices_in_bounds:
            road_entry = self.road_entries[index]
            centerline_px = convert_world_coords_to_pixels(road_entry.centerline, world_to_pixel)

            for start_px, end_px in zip(centerline_px[:-1], centerline_px[1:]):
                d_px = end_px - start_px
                angle_px = np.arctan2(-d_px[1], d_px[0]) # minus sign since image coord y-axis is downward facing
                color = angle_to_color(angle_px)

                start_px = tuple( np.round(start_px, 0).astype(np.int) )
                end_px   = tuple( np.round(end_px, 0).astype(np.int) )
                cv2.line(img, start_px, end_px, color, thickness = 5)

        return img
