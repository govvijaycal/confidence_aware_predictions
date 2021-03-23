import carla
import os
import sys
import argparse
import cv2
import numpy as np
import random
import time

CARLA_ROOT = os.getenv("CARLA_ROOT")
if CARLA_ROOT is None:
	raise ValueError("CARLA_ROOT must be defined.")

scriptdir = CARLA_ROOT + "PythonAPI/"
sys.path.append(scriptdir)
from examples.synchronous_mode import CarlaSyncMode

sys.path.append(CARLA_ROOT + "PythonAPI/carla/agents/")
from navigation.global_route_planner import GlobalRoutePlanner
from navigation.global_route_planner_dao import GlobalRoutePlannerDAO

import frenet_trajectory_handler as fth

from rasterizer.box_rasterizer import BoxRasterizer
from rasterizer.agent_history import AgentHistory
from rasterizer.semantic_rasterizer import extract_waypoints_from_topology, RoadEntry, SemanticRasterizer

class VehicleAgent(object):
	def __init__(self, vehicle, carla_map, is_rational=True):
		self.vehicle = vehicle
		self.map     = carla_map
		self.planner = GlobalRoutePlanner( GlobalRoutePlannerDAO(self.map, sampling_resolution=0.5) )
		self.planner.setup()

		init_waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))

		goals = init_waypoint.next(75.) # TODO: make goal destination programmable rather than hard-coded.

		route = self.planner.trace_route( init_waypoint.transform.location, goals[1].transform.location)

		way_s, way_xy, way_yaw = fth.extract_path_from_waypoints( route )
		self._frenet_traj = fth.FrenetTrajectoryHandler(way_s, way_xy, way_yaw, s_resolution=0.5)		

		self.control_prev = carla.VehicleControl()
		self.max_steer_angle = np.radians( self.vehicle.get_physics_control().wheels[0].max_steer_angle )

		# Controller params:	
		# TODO: more in-depth rational vs. irrational model.	
		if is_rational:
			self.alpha = 0.8
			self.k_v = 0.1
			self.k_ey = 0.2
			self.x_la = 5.0
		else:		
			self.alpha = 0.8
			self.k_v = 1.0
			self.k_ey = 1.0
			self.x_la = 0.1

	def run_step(self, dt):
		vehicle_loc   = self.vehicle.get_location()
		vehicle_wp    = self.map.get_waypoint(vehicle_loc)
		vehicle_tf    = self.vehicle.get_transform()
		vehicle_vel   = self.vehicle.get_velocity()
		vehicle_accel = self.vehicle.get_acceleration()

		x, y = vehicle_loc.x, -vehicle_loc.y
		psi = -fth.fix_angle(np.radians(vehicle_tf.rotation.yaw))

		s, ey, epsi = \
			self._frenet_traj.convert_global_to_frenet_frame(x, y, psi)
		curv = self._frenet_traj.get_curvature_at_s(s)

		speed = np.sqrt(vehicle_vel.x**2 + vehicle_vel.y**2)
		accel = np.cos(psi) * vehicle_accel.x - np.sin(psi)*vehicle_accel.y

		control = carla.VehicleControl()
		control.hand_brake = False
		control.manual_gear_shift = False    

		if self._frenet_traj.reached_trajectory_end(s):
			control.throttle = 0.
			control.brake    = -1.			
		else:
			# Step 1: Generate reference by identifying a max speed based on curvature + stoplights.
			# TODO: update logic, maybe use speed limits from Carla.
			lat_accel_max = 2.0 # m/s^2
			speed_limit   = 13.  # m/s -> 29 mph ~ 30 mph
			
			if np.abs(curv) > 0.01:
				max_speed = 3.6 * np.sqrt(lat_accel_max / np.abs(curv))            
				max_speed = min(max_speed, speed_limit)
			else:
				max_speed = speed_limit
			
			if speed > max_speed + 2.0:
				control.throttle = 0.0
				control.brake = self.k_v * (speed - max_speed)
			elif speed < max_speed - 2.0:
				control.throttle = self.k_v * (max_speed - speed)
				control.brake    = 0.0
			else:
				control.throttle = 0.1
				control.brake    = 0.0

			if control.throttle > 0.0:
				control.throttle = self.alpha * control.throttle + (1. - self.alpha) * self.control_prev.throttle
			
			elif control.brake > 0.0:
				control.brake    = self.alpha * control.brake    + (1. - self.alpha) * self.control_prev.brake
			
		control.steer = self.k_ey * (ey + self.x_la * epsi) / self.max_steer_angle
		control.steer    = self.alpha * control.steer    + (1. - self.alpha) * self.control_prev.steer

		control.throttle = np.clip(control.throttle, 0.0, 1.0)
		control.brake    = np.clip(control.brake, 0.0, 1.0)
		control.steer    = np.clip(control.steer, -1.0, 1.0)

		self.control_prev = control        
		return control
		
def run_simulation(args):
	actor_list  = []
	camera_list = []

	try:
		client = carla.Client(args.host, args.port)
		client.set_timeout(2.0)
		world = client.get_world()
		world = client.load_world("Town07")
		world.set_weather(getattr(carla.WeatherParameters, "ClearNoon"))

		# 3-intersection starts:
		# Approx. center: -3., -158.
		# -1., -125., 0 deg
		# -4, -185., 180 deg
		# -33., -158., 90 deg

		bp_library = world.get_blueprint_library()

		""" EGO vehicle """
		ego_location = carla.Location(x=-1, y=-120., z=1.)
		ego_rotation = carla.Rotation(yaw=270.)
		ego_transform = carla.Transform(ego_location, ego_rotation)
		ego_vehicle = random.choice(bp_library.filter(args.ego_filter))
		ego_vehicle.set_attribute('role_name', 'hero')
		ego_vehicle = world.spawn_actor(ego_vehicle, ego_transform)
		actor_list.append(ego_vehicle)

		""" NPC vehicle """
		npc_location = carla.Location(x=-4., y=-180., z=1.)
		npc_rotation = carla.Rotation(yaw=90.)
		npc_transform = carla.Transform(npc_location, npc_rotation)

		npc_vehicle = random.choice(bp_library.filter(args.npc_filter))
		npc_vehicle.set_attribute('role_name', 'npc')
		npc_vehicle = world.spawn_actor(npc_vehicle, npc_transform)
		actor_list.append(npc_vehicle)

		""" Drone camera. """
		bp_drone  = bp_library.find('sensor.camera.rgb')	

		# This over the top of the intersection.
		# cam_loc = carla.Location(x=-3., y=-158., z=40.)
		# cam_ori = carla.Rotation(pitch=270.)

		# This is like a surveillance camera view from the side of the intersection.
		cam_loc = carla.Location(x=30., y=-160., z=40.)
		cam_ori = carla.Rotation(pitch=-45, yaw=180., roll=0.)
		cam_transform = carla.Transform(cam_loc, cam_ori)

		bp_drone.set_attribute('image_size_x', str(1920))
		bp_drone.set_attribute('image_size_x', str(1080))
		bp_drone.set_attribute('fov', str(90))
		bp_drone.set_attribute('role_name', 'drone')
		drone = world.spawn_actor(bp_drone, cam_transform)
		camera_list.append(drone)

		ego_agent = VehicleAgent(ego_vehicle, world.get_map(), is_rational=True) # TODO: better interface for this.
		npc_agent = VehicleAgent(npc_vehicle, world.get_map(), is_rational=True)

		writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), args.fps, (1460, 500)) # TODO: use argparse to decide whether to log videos or not.  Maybe do this in post-processing from logfile.

		agent_history = AgentHistory(world.get_actors())
		box_rasterizer    = BoxRasterizer()

		set_waypoints = extract_waypoints_from_topology(world.get_map().get_topology())
		road_entries = [RoadEntry(waypoints) for waypoints in set_waypoints]
		sem_rasterizer = SemanticRasterizer(road_entries)

		frames_to_show = 500 #TODO: argparse for this.
		with CarlaSyncMode(world, *camera_list, fps=args.fps) as sync_mode:
			while frames_to_show > 0:
				snap, img = sync_mode.tick(timeout=2.0)

				agent_history.update(snap)				

				ego_control = ego_agent.run_step(1 / args.fps)
				ego_vehicle.apply_control(ego_control)

				npc_control = npc_agent.run_step(1 / args.fps)
				npc_vehicle.apply_control(npc_control)

				img_array = np.frombuffer(img.raw_data, dtype=np.uint8)
				img_array = np.reshape(img_array, (img.height, img.width, 4))
				img_array = img_array[:, :, :3]

				img_array = cv2.resize(img_array, (960, 500), interpolation = cv2.INTER_AREA)

				# TODO: maybe use raster_common to do the image combination.
				box_view = box_rasterizer.rasterize(agent_history)
				sem_view = sem_rasterizer.rasterize(agent_history)
				mask_box = np.any(box_view > 0, -1)
				sem_view[mask_box] = box_view[mask_box]
				img_birdview = sem_view
				
				mosaic_array = np.zeros((500, 1460, 3), dtype=np.uint8) # H x W x 3
				mosaic_array[:, :960, :] = img_array
				mosaic_array[:, 960:, :] = cv2.cvtColor(img_birdview, cv2.COLOR_RGB2BGR)

				writer.write(mosaic_array) # TODO: clean up saving vs. visualizing.  Add logfile recording.
				# cv2.imshow('mosaic', mosaic_array)
				# ret = cv2.waitKey(10)

				# time.sleep(0.001)
				frames_to_show -= 1
				print(frames_to_show)

				# if ret == 27: # Esc
				# 	break
		writer.release()

	finally:
		for actor in actor_list:
			actor.destroy()
		
		for camera in camera_list:
			camera.destroy()

		cv2.destroyAllWindows()
		print('Done.')

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(
		description='CARLA Multi-Vehicle Simulation')
	argparser.add_argument(
		'--host',
		metavar='H',
		default='127.0.0.1',
		help='IP of the host server (default: 127.0.0.1)')
	argparser.add_argument(
		'-p', '--port',
		metavar='P',
		default=2000,
		type=int,
		help='TCP port to listen to (default: 2000)')
	argparser.add_argument(
		'--ego_filter',
		metavar='PATTERN',
		default='vehicle.lincoln.mkz2017')
	argparser.add_argument(
		'--npc_filter',
		metavar='PATTERN',
		default='vehicle.audi.tt')
	argparser.add_argument( 
		'--logdir', # TODO: update this, ideally with logfiles.
		default='data_synced',
		help='Image logging directory for saved rgb,depth,and semantic segmentation images.')
	argparser.add_argument( 
		'--fps',
		default=5,
		type=int)
	args = argparser.parse_args()
		
	run_simulation(args)
	