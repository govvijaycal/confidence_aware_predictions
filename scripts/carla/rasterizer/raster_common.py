import numpy as np
import colorsys
import cv2

# CV2 pixel shift functions/values taken from L5Kit semantic_rasterizer.py.
CV2_SHIFT       = 9
CV2_LINE_TYPE   = cv2.LINE_AA
CV2_SHIFT_VALUE = 2 ** CV2_SHIFT

def cv2_subpixel(coords):
	coords = coords * CV2_SHIFT_VALUE
	return coords.astype(np.int)

def convert_world_coords_to_pixels(world_coords, world_to_pixel):
	# world_coords: N_pts by 2 matrix of XY coordinates
	# world_to_pixel: 3 x 3 matrix taking in a (homogenous) world XY point and returning a (homogenous) pix_x, pix_y pixel
	N_pts = world_coords.shape[0]
	pix_coords = world_to_pixel @ np.concatenate((world_coords.T, np.ones((1, N_pts)))) # 3 by N_pts homogenous pixels
	return pix_coords[:2, :].T # N_pts by 2 pixels

def angle_to_color(angle):
    angle = angle + np.pi
    angle = np.degrees(angle)

    color = colorsys.hsv_to_rgb( angle/360, 1., 1.)    
    color = [int(255 * c) for c in color]
    
    return tuple(color)