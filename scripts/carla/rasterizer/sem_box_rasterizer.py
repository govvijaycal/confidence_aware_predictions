import cv2
import numpy as np
from .semantic_rasterizer import SemanticRasterizer
from .box_rasterizer import BoxRasterizer

class SemBoxRasterizer:
    def __init__(self,
                 carla_topology,
                 render_traffic_lights=True,        # whether to overlay traffic lights on the image
                 raster_size=(500, 500),            # pixel height and width
                 raster_resolution=0.1,             # meters / pixel resolution
                 target_center_x = 100,             # pixels along x-axis where target center should be located
                 target_center_y = 250,             # pixels along y-axis where target center should be located
                 history_secs=[1.0, 0.6, 0.2, 0.0], # which historical timestamps (s) to visualize for boxes
                 closeness_eps=0.1):                # how close (s) for a timestamp to be to a query to be considered the same

        self.sem_rast = SemanticRasterizer(carla_topology,
                                           raster_size=raster_size,
                                           raster_resolution=raster_resolution,
                                           target_center_x = target_center_x,
                                           target_center_y = target_center_y)

        self.box_rast = BoxRasterizer(raster_size=raster_size,
                                      raster_resolution=raster_resolution,
                                      target_center_x = target_center_x,
                                      target_center_y = target_center_y,
                                      history_secs=history_secs,
                                      closeness_eps=closeness_eps)
        self.render_traffic_lights = render_traffic_lights

    def rasterize(self, agent_history, target_agent_id):

        im_sem = self.sem_rast.rasterize(agent_history, target_agent_id, self.render_traffic_lights)
        im_box = self.box_rast.rasterize(agent_history, target_agent_id)

        mask_box = np.any(im_box > 0, -1)     # which pixel locations have nonzero values in im_box
        im_sem[mask_box] = im_box[mask_box] # overlay non-zero box pixel values on top of im_sem

        return im_sem
