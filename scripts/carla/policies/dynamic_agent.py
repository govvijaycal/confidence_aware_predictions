from abc import ABC, abstractmethod
import carla
import os
import sys
import numpy as np
import time

CARLA_ROOT = os.getenv("CARLA_ROOT")
if CARLA_ROOT is None:
    raise ValueError("CARLA_ROOT must be defined.")

sys.path.append(CARLA_ROOT + "PythonAPI/carla/agents/")
from navigation.global_route_planner import GlobalRoutePlanner
from navigation.global_route_planner_dao import GlobalRoutePlannerDAO

scriptdir = os.path.abspath(__file__).split('carla')[0] + 'carla/'
sys.path.append(scriptdir)
from utils import frenet_trajectory_handler as fth
from utils.vehicle_geometry_utils import vehicle_name_to_dimensions

class DynamicAgent(ABC):
    """ A path following agent with collision avoidance constraints over a short horizon. """

    def __init__(self, vehicle, goal_location, dt):
        self.vehicle = vehicle

        world = vehicle.get_world()
        carla_map = world.get_map()
        planner = GlobalRoutePlanner( GlobalRoutePlannerDAO(carla_map, sampling_resolution=0.5) )
        planner.setup()

        # Get the high-level route using Carla's API (basically A* search over road segments).
        init_waypoint = carla_map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
        goal          = carla_map.get_waypoint(goal_location, project_to_road=True, lane_type=(carla.LaneType.Driving))
        route = planner.trace_route(init_waypoint.transform.location, goal.transform.location)

        # Convert the high-level route into a path parametrized by arclength distance s (i.e. Frenet frame).
        way_s, way_xy, way_yaw = fth.extract_path_from_waypoints(route)
        self._frenet_traj = fth.FrenetTrajectoryHandler(way_s, way_xy, way_yaw, s_resolution=0.5)

        vehicle_dims = vehicle_name_to_dimensions(self.vehicle.type_id)
        self.lf = vehicle_dims["lf"]
        self.lr = vehicle_dims["lr"]

        self.DT      =  dt # control timestep, s
        self.A_MIN   = -3.0  # min accel, m/s^2
        self.A_MAX   =  2.0  # max accel, m/s^2
        self.V_MIN   =  0.0  # min speed, m/s
        self.V_MAX   = 20.0  # max speed, m/s
        self.DF_MIN  = -0.5  # min steer angle, rad
        self.DF_MAX  =  0.5  # max steer angle, rad

        self.goal_reached = False # flags when the end of the path is reached and agent should stop

    def get_current_state(self):
        vehicle_tf    = self.vehicle.get_transform()
        vehicle_vel   = self.vehicle.get_velocity()

        # Get the vehicle's current pose + speed in a RH coordinate system.
        x, y  = vehicle_tf.location.x, -vehicle_tf.location.y
        psi   = -fth.fix_angle(np.radians(vehicle_tf.rotation.yaw))
        speed = np.sqrt(vehicle_vel.x**2 + vehicle_vel.y**2)

        # Look up the projection of the current pose to Frenet frame.
        s, ey, epsi, idx = self._frenet_traj.convert_global_to_frenet_frame(x, y, psi)
        curv             = self._frenet_traj.get_curvature_at_s(s)

        state_dict = {"x"      : x,
                      "y"      : y,
                      "psi"    : psi,
                      "speed"  : speed,
                      "s"      : s,
                      "ey"     : ey,
                      "epsi"   : epsi,
                      "curv"   : curv,
                      "ft_idx" : idx}

        return state_dict

    def update_completion(self, s):
        if self.goal_reached or self._frenet_traj.reached_trajectory_end(s, resolution=20.):
            # Stop if the end of the path is reached and signal completion.
            self.goal_reached = True

    def done(self):
        return self.goal_reached

    @abstractmethod
    def run_step(self, pred_dict):
        # This should return a tuple with the following:
        # (1) control, carla.VehicleControl
        # (2) z0, current (x,y,theta,v) state as np.ndarray
        # (3) u0, current (acceleration, steer_angle) input
        # (4) is_opt, flag if the returned solution is optimal (if doing MPC)
        # (5) solve_time, how long it took to compute u0/control in secs.
        raise NotImplementedError