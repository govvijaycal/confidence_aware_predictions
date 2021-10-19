import carla
import os
import sys
import argparse
import cv2
import numpy as np
import random
import time

import matplotlib.pyplot as plt

CARLA_ROOT = os.getenv("CARLA_ROOT")
if CARLA_ROOT is None:
    raise ValueError("CARLA_ROOT must be defined.")

sys.path.append(CARLA_ROOT + "PythonAPI/carla/agents/")
from navigation.basic_agent import BasicAgent

scriptdir = CARLA_ROOT + "PythonAPI/"
sys.path.append(scriptdir)
from examples.synchronous_mode import CarlaSyncMode

scriptdir = os.path.abspath(__file__).split('carla')[0] + 'carla/'
sys.path.append(scriptdir)

SPAWN_LOCATION  = [-10.0, -250, 90] # on the highway in Town04
TARGET_LOCATION = [-10.0, -100, 90]
#########################################################
def make_transform_from_pose(pose, spawn_height=0.5):
    location = carla.Location( x=pose[0], y = pose[1], z=spawn_height)
    rotation = carla.Rotation(yaw=pose[2])
    return carla.Transform(location, rotation)

def spawn_ego(world, vehicle_name):
    bp_library = world.get_blueprint_library()
    dyn_bp = bp_library.filter(vehicle_name)[0]
    town_map = world.get_map()
    ego_start_transform = make_transform_from_pose(SPAWN_LOCATION)
    return world.spawn_actor(dyn_bp, ego_start_transform)

def get_throttle_feedforward(world, v0, fps, vehicle_name):
    """ Given an initial speed, this function uses a binary search approach
        to identify the steady-state throttle input needed maintain that speed.
    """
    speed_eps = 0.3 # m/s, buffer for "converging" to a target speed

    control = carla.VehicleControl()
    control.hand_brake = False
    control.manual_gear_shift = False

    town_map = world.get_map()

    # Binary Search variables.
    completed = False
    max_iters = 10
    curr_iter = 0
    thr_lower = 0.1; thr_upper = 0.9
    thr_curr = 0.5 * (thr_lower + thr_upper)

    result_dict = {}

    for itr in range(max_iters):
        if completed:
            break

        ego_vehicle = None
        with CarlaSyncMode(world, fps=fps) as sync_mode:
            ego_vehicle = spawn_ego(world, vehicle_name)
            for _ in range(10):
                # Give some time for the ego vehicle to settle after spawning.
                sync_mode.tick(timeout=2.0)

            # BasicAgent used to get the vehicle to speed and handle steering.
            agent       = BasicAgent(ego_vehicle, target_speed=v0 * 3.6) # BasicAgent wants speed in kph
            ego_waypoint = town_map.get_waypoint(ego_vehicle.get_location())
            next_waypoint = ego_waypoint.next(300.)[0]
            dest = [next_waypoint.transform.location.x,
                    next_waypoint.transform.location.y,
                    next_waypoint.transform.location.z]
            agent.set_destination(dest)

            print(f"V0: {v0} Iter: {itr}")
            print("\t Getting Up To Speed")
            reached_speed = False
            for _ in range(1000):
                # Get the vehicle sufficiently close to the target speed.
                sync_mode.tick(timeout=2.0)
                control_ba = agent.run_step(1./fps)
                ego_vehicle.apply_control(control_ba)

                vel = ego_vehicle.get_velocity()
                speed = (vel.x**2 + vel.y**2)**0.5

                if abs(speed - v0) < speed_eps:
                    reached_speed = True
                    break

            if not reached_speed:
                raise RuntimeError("Couldn't get up to speed!")

            speed_log = []
            accel_log = []

            print(f"\t Evaluating throttle: {thr_curr}")
            for _ in range(100):
                # Evaluate throttle capability to get us in "steady-state."
                # Essentially applying throttle should keep the speed roughly constant.
                snap = sync_mode.tick(timeout=2.0)

                vel = ego_vehicle.get_velocity()
                acc = ego_vehicle.get_acceleration()
                speed_log.append( (vel.x**2 + vel.y**2)**0.5 )
                accel_log.append( (acc.x**2 + acc.y**2)**0.5 )

                control_ba = agent.run_step(1./fps)
                control.throttle = thr_curr
                control.brake    = 0.
                control.steer    = control_ba.steer
                ego_vehicle.apply_control(control)


            result_dict[f"{thr_curr}"] = {"speed" : speed_log,
                                          "accel" : accel_log}

            # Binary search update / termination procedure.
            if np.abs(speed_log[-1] - v0) < speed_eps:
                completed = True
            elif speed_log[-1] < v0:
                thr_lower = thr_curr
            else:
                thr_upper = thr_curr

            thr_curr  = 0.5 * (thr_lower + thr_upper)

            ego_vehicle.destroy()

    return thr_curr, result_dict

def test_braking(world, v0, fps, vehicle_name):
    """ Given an initial speed, this function returns the average deceleration
        observed when various throttle inputs (brake_levels) are applied.
    """
    speed_eps = 0.3 # m/s, buffer for "converging" to a target speed

    control = carla.VehicleControl()
    control.hand_brake = False
    control.manual_gear_shift = False

    town_map = world.get_map()

    # Binary Search variables.
    result_dict = {}

    brake_levels = np.arange(0.25, 1.1, 0.25)

    ego_vehicle = None
    for brake_lvl in brake_levels:
        with CarlaSyncMode(world, fps=fps) as sync_mode:
            ego_vehicle = spawn_ego(world, vehicle_name)
            for _ in range(10):
                # Give some time for the ego vehicle to settle after spawning.
                sync_mode.tick(timeout=2.0)

            # BasicAgent used to get the vehicle to speed and handle steering.
            agent       = BasicAgent(ego_vehicle, target_speed=v0 * 3.6)
            ego_waypoint = town_map.get_waypoint(ego_vehicle.get_location())
            next_waypoint = ego_waypoint.next(300.)[0]
            dest = [next_waypoint.transform.location.x,
                    next_waypoint.transform.location.y,
                    next_waypoint.transform.location.z]
            agent.set_destination(dest)

            print(f"\t Getting Up To Speed {v0}")
            reached_speed = False
            for _ in range(1000):
                # Get the vehicle sufficiently close to the target speed.
                sync_mode.tick(timeout=2.0)
                control_ba = agent.run_step(1./fps)
                ego_vehicle.apply_control(control_ba)

                vel = ego_vehicle.get_velocity()
                speed = (vel.x**2 + vel.y**2)**0.5

                if abs(speed - v0) < speed_eps:
                    reached_speed = True
                    break

            if not reached_speed:
                raise RuntimeError("Couldn't get up to speed!")

            speed_log = []
            accel_log = []

            print(f"\t Evaluating brake: {brake_lvl}")
            for _ in range(100):
                # Evaluate brake input to deceleration level.
                snap = sync_mode.tick(timeout=2.0)

                vel = ego_vehicle.get_velocity()
                acc = ego_vehicle.get_acceleration()
                speed_log.append( (vel.x**2 + vel.y**2)**0.5 )
                accel_log.append( (acc.x**2 + acc.y**2)**0.5 )

                control_ba = agent.run_step(1./fps)
                control.throttle = 0.
                control.brake    = brake_lvl
                control.steer    = control_ba.steer
                ego_vehicle.apply_control(control)


            # Compute average deceleration
            if speed_log[-1] > 0.2:
                # Did not fully come to a stop before the end of the test.
                # Then delta-t is the time taken for the test.
                speed_diff = speed_log[0] - speed_log[-1]
                time_diff  = len(speed_log) / fps
            else:
                # Came to a stop before the end of the test, need to identify delta-t
                # corresponding to when the vehicle stopped.
                speed_diff = speed_log[0]
                time_diff  = np.amin( np.argwhere(np.array(speed_log) < 0.2) ) / fps
            avg_decel = speed_diff / time_diff

            result_dict[f"{brake_lvl}"] = {"speed" : speed_log,
                                           "accel" : accel_log,
                                           "avg_decel" : avg_decel}
            ego_vehicle.destroy()

    return result_dict

def main(fps, test_throttle, test_brake, vehicle_name):
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(2.0)

        world = client.get_world()
        if world.get_map().name != "Town04":
            world = client.load_world("Town04")
        world.set_weather(getattr(carla.WeatherParameters, "ClearNoon"))

        if test_throttle:
            # Compute throttle LUT and show the corresponding speed/accel profiles.
            thr_LUT = {}
            for v_init in np.arange(2.5, 21.0, 5.0):
                thr_ff, result_dict = get_throttle_feedforward(world, v_init, fps, vehicle_name)

                timesteps = None
                plt.figure()
                for k in result_dict.keys():
                    if timesteps is None:
                        timesteps = np.arange( len(result_dict[k]["speed"]) ) * 1. / fps

                    plt.subplot(211)
                    plt.plot(timesteps, result_dict[k]["speed"], label=str(k))
                    plt.ylabel('speed (m/s)')
                    plt.subplot(212)
                    plt.plot(timesteps, result_dict[k]["accel"], label=str(k))
                    plt.ylabel('acc (m/s^2)')
                    plt.xlabel('s')
                plt.subplot(211); plt.legend()
                plt.subplot(212); plt.legend()
                plt.suptitle(f"{v_init}")
                thr_LUT[v_init] = thr_ff
            plt.show()

            # Mercedes Benz Coupe:
            # thr_LUT = {2.5: 0.35, 7.5: 0.45, 12.5: 0.5, 17.5: 0.6}
            #
            # Lincoln MKZ 2017
            # thr_LUT = {2.5: 0.3, 7.5: 0.45, 12.5: 0.55, 17.5: 0.65}

        if test_brake:
            # Compute brake deceleration at various initial speeds and show the corresponding speed/accel profiles.
            for v_init in np.arange(2.5, 21.0, 5.):
                result_dict = test_braking(world, v_init, fps, vehicle_name)
                timesteps = None
                plt.figure()
                for k in result_dict.keys():
                    if timesteps is None:
                        timesteps = np.arange( len(result_dict[k]["speed"]) ) * 1. / fps

                    plt.subplot(211)
                    plt.plot(timesteps, result_dict[k]["speed"], label=str(k))
                    plt.ylabel('speed (m/s)')
                    plt.subplot(212)
                    plt.plot(timesteps, result_dict[k]["accel"], label=str(k))
                    avg_decel = result_dict[k]["avg_decel"]
                    plt.plot(timesteps, len(timesteps)*[avg_decel], 'k--', label=f"str{k}_avg")
                    plt.ylabel('acc (m/s^2)')
                    plt.xlabel('s')

                    print(f"v0:{v_init}, brake:{str(k)}, avg_decel:{avg_decel}")
                plt.subplot(211); plt.legend()
                plt.subplot(212); plt.legend()
                plt.suptitle(f"{v_init}")

                """
                Mercedes Benz Coupe:
                v0:2.5, brake:0.0, avg_decel:1.004
                v0:2.5, brake:0.25, avg_decel:2.949
                v0:2.5, brake:0.5, avg_decel:3.535
                v0:2.5, brake:0.75, avg_decel:3.770
                v0:2.5, brake:1.0, avg_decel:4.039

                v0:7.5, brake:0.0, avg_decel:1.913
                v0:7.5, brake:0.25, avg_decel:4.313
                v0:7.5, brake:0.5, avg_decel:5.806
                v0:7.5, brake:0.75, avg_decel:6.862
                v0:7.5, brake:1.0, avg_decel:7.550

                v0:12.5, brake:0.0, avg_decel:1.765
                v0:12.5, brake:0.25, avg_decel:4.659
                v0:12.5, brake:0.5, avg_decel:7.710
                v0:12.5, brake:0.75, avg_decel:7.467
                v0:12.5, brake:1.0, avg_decel:8.812

                v0:17.5, brake:0.0, avg_decel:1.843
                v0:17.5, brake:0.25, avg_decel:3.760
                v0:17.5, brake:0.5, avg_decel:7.058
                v0:17.5, brake:0.75, avg_decel:8.036
                v0:17.5, brake:1.0, avg_decel:9.851

                Lincoln MKZ 2017:
                v0:2.5, brake:0.0, avg_decel:0.588
                v0:2.5, brake:0.25, avg_decel:2.514
                v0:2.5, brake:0.5, avg_decel:3.253
                v0:2.5, brake:0.75, avg_decel:3.687
                v0:2.5, brake:1.0, avg_decel:3.951

                v0:7.5, brake:0.0, avg_decel:1.367
                v0:7.5, brake:0.25, avg_decel:3.056
                v0:7.5, brake:0.5, avg_decel:5.867
                v0:7.5, brake:0.75, avg_decel:6.667
                v0:7.5, brake:1.0, avg_decel:6.984

                v0:12.5, brake:0.0, avg_decel:1.503
                v0:12.5, brake:0.25, avg_decel:2.698
                v0:12.5, brake:0.5, avg_decel:4.091
                v0:12.5, brake:0.75, avg_decel:5.336
                v0:12.5, brake:1.0, avg_decel:6.634

                v0:17.5, brake:0.0, avg_decel:1.397
                v0:17.5, brake:0.25, avg_decel:2.511
                v0:17.5, brake:0.5, avg_decel:4.795
                v0:17.5, brake:0.75, avg_decel:5.952
                v0:17.5, brake:1.0, avg_decel:6.637
                """

            plt.show()


    finally:
        print('Done.')

if __name__ == '__main__':
    fps = 20.
    test_throttle = True
    test_brake    = True
    vehicle_name = "vehicle.lincoln.mkz2017" # "vehicle.mercedes-benz.coupe""
    main(fps=20.,
         test_brake=test_brake,
         test_throttle=test_throttle,
         vehicle_name=vehicle_name)
