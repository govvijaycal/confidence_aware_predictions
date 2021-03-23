import cv2
import colorsys
import numpy as np

from .raster_common import convert_world_coords_to_pixels, cv2_subpixel, CV2_SHIFT

class BoxRasterizer:
	def __init__(self,
				 raster_size=(500, 500), # pixel height and width
				 raster_resolution=0.1,  # meters / pixel resolution
				 ego_center_x = 100,     # pixels along x-axis where ego center should be located
				 ego_center_y = 250,     # pixels along y-axis where ego center should be located
				 history_secs=[1.0, 0.6, 0.2, 0.0],
				 closeness_eps=0.1):
		
		# History seconds used to determine which previous frames to plot (in a faded color).
		self.history_secs  = history_secs
		self.history_secs.sort(reverse=True)
		self.closeness_eps = closeness_eps

		# HSV Interpolation for fading and Agent Colors.
		self.min_hsv_value = 0.4
		self.hsec_to_val = lambda hsec : np.clip(1.0 - 0.6 * hsec, self.min_hsv_value, 1.)

		self.npc_vehicle_rgb = (255, 255, 0)  # yellow
		self.ego_vehicle_rgb = (255, 0, 0)    # red
		self.pedestrian_rgb  = (255, 153, 51) # orange

		self.npc_vehicle_hsv = colorsys.rgb_to_hsv(*self.npc_vehicle_rgb)
		self.ego_vehicle_hsv = colorsys.rgb_to_hsv(*self.ego_vehicle_rgb)
		self.pedestrian_hsv  = colorsys.rgb_to_hsv(*self.pedestrian_rgb)

		# Raster Related Information.
		self.raster_height, self.raster_width = raster_size        
		self.ego_to_pixel = np.array([[1./raster_resolution,  0., ego_center_x],\
									  [0, -1./raster_resolution, ego_center_y],\
									  [0., 0., 1.]])

	def _get_mask(self, snapshots, ego_centroid, ego_yaw):
		mask = np.zeros((self.raster_height, self.raster_width), dtype=np.uint8)
		if len(snapshots) == 0:
			pass
		else:
			extents   = np.array([snap['extent'] for snap in snapshots])   # N by 2, half extent along each axis according to carla.BoundingBox API.
			centroids = np.array([snap['centroid'] for snap in snapshots]) # N by 2, XY coordinate in RHS Carla system (i.e. "world").
			yaws      = np.array([snap['yaw'] for snap in snapshots])      # N, yaw (radians) wrt RHS Carla system (i.e. "world").
			
			# Adapted from draw_boxes in BoxRasterizer, L5Kit repo:
			corners_base_coords   = (np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]]))[None, :, :] # base bounding box, shape = (1, 4, 2)
			corners_scaled_coords = corners_base_coords * extents[:, None, :]                      # scaled bounding boxes, shape = (N, 4, 2)

			c = np.cos(yaws)
			s = np.sin(yaws)
			R_box2world = np.moveaxis(np.array(((c, s), (-s, c))), 2, 0)

			corners_world_coords  = corners_scaled_coords @ R_box2world + centroids[:, None, :]   # transformed bounding boxes, shape = (N, 4, 2)

			n_corners = corners_world_coords.shape[0] * 4

			
			R_ego_to_world    = np.array([[np.cos(ego_yaw), -np.sin(ego_yaw)],\
				                         [np.sin(ego_yaw),  np.cos(ego_yaw)]])
			t_ego_to_world    = ego_centroid.reshape(2, 1)

			R_world_to_ego = R_ego_to_world.T
			t_world_to_ego = -R_ego_to_world.T @ t_ego_to_world

			world_to_pixel = self.ego_to_pixel @ np.block([[R_world_to_ego, t_world_to_ego], [0., 0., 1.]])

			corners_pix_coords = convert_world_coords_to_pixels(corners_world_coords.reshape(n_corners, 2), world_to_pixel)		
			corners_pix_coords = cv2_subpixel(corners_pix_coords.reshape(-1, 4, 2)) # N x 4 x 2
			cv2.fillPoly(mask, corners_pix_coords, color=255, shift=CV2_SHIFT)

		return mask

	def rasterize(self, agent_history):		
		img = np.zeros((self.raster_height, self.raster_width, 3), dtype=np.uint8)

		snapshots = agent_history.query(history_secs=self.history_secs, closeness_eps=self.closeness_eps)

		ego_snap_current     = snapshots[np.round(0., 2)]['ego_vehicle'][0]
		ego_centroid_current = np.array(ego_snap_current['centroid'])
		ego_yaw_current      = ego_snap_current['yaw']

		for hsec in self.history_secs:
			scene_dict =  snapshots[np.round(hsec, 2)] 

			if len(scene_dict) == 0:
				continue

			# Figure out the colors to use (HSV).
			val = int(255* self.hsec_to_val(hsec))
			npc_faded_rgb = colorsys.hsv_to_rgb(self.npc_vehicle_hsv[0], self.npc_vehicle_hsv[1], val)
			ego_faded_rgb = colorsys.hsv_to_rgb(self.ego_vehicle_hsv[0], self.ego_vehicle_hsv[1], val)
			ped_faded_rgb = colorsys.hsv_to_rgb( self.pedestrian_hsv[0],  self.pedestrian_hsv[1], val)


			# Plot the NPC vehicles, pedestrians, and ego vehicle for this time step.
			npc_mask = self._get_mask(scene_dict['npc_vehicles'], ego_centroid_current, ego_yaw_current)
			ped_mask = self._get_mask(scene_dict['pedestrians'], ego_centroid_current, ego_yaw_current)
			ego_mask = self._get_mask(scene_dict['ego_vehicle'], ego_centroid_current, ego_yaw_current)

			img[npc_mask==255] = npc_faded_rgb
			img[ped_mask==255] = ped_faded_rgb
			img[ego_mask==255] = ego_faded_rgb			

		return img