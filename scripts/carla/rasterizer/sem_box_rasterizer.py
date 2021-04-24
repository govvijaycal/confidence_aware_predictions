import cv2
import numpy as np
from .semantic_rasterizer import SemanticRasterizer
from .box_rasterizer import BoxRasterizer

class SemBoxRasterizer:
    def __init__(self,
                 carla_topology,
                 raster_size=(500, 500), # pixel height and width
                 raster_resolution=0.1,  # meters / pixel resolution
                 ego_center_x = 100,     # pixels along x-axis where ego center should be located
                 ego_center_y = 250,     # pixels along y-axis where ego center should be located
                 history_secs=[1.0, 0.6, 0.2, 0.0], # which historical timestamps (s) to visualize for boxes
                 closeness_eps=0.1):                # how close (s) for a timestamp to be to a query to be considered the same

        self.sem_rast = SemanticRasterizer(carla_topology,
                                           raster_size=raster_size,
                                           raster_resolution=raster_resolution,
                                           ego_center_x = ego_center_x,
                                           ego_center_y = ego_center_y)

        self.box_rast = BoxRasterizer(raster_size=raster_size,
                                      raster_resolution=raster_resolution,
                                      ego_center_x = ego_center_x,
                                      ego_center_y = ego_center_y,
                                      history_secs=history_secs,
                                      closeness_eps=closeness_eps)

    def rasterize(self, agent_history):

        im_sem = self.sem_rast.rasterize(agent_history)
        im_box = self.box_rast.rasterize(agent_history)

        mask_box = np.any(im_box > 0, -1)     # which pixel locations have nonzero values in im_box
        im_sem[mask_box] = im_box[mask_box] # overlay non-zero box pixel values on top of im_sem

        return im_sem
