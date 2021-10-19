import carla
import os
import sys
import cv2
import json
import numpy as np

CARLA_ROOT = os.getenv("CARLA_ROOT")
if CARLA_ROOT is None:
    raise ValueError("CARLA_ROOT must be defined.")

scriptdir = CARLA_ROOT + "PythonAPI/"
sys.path.append(scriptdir)
from examples.synchronous_mode import CarlaSyncMode

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/carla/'
sys.path.append(scriptdir)
from scenarios.run_intersection_scenario import CarlaParams, DroneVizParams, VehicleParams, PredictionParams, RunIntersectionScenario

def setup_intersection_scenario(scenario_dict, ego_init_dict, savedir):
    # This is simply used to start up the scenarios with vehicles determining the route.
    # The route is simply queried for overlays - the actual policies are never run here.
    carla_params     = CarlaParams(**scenario_dict["carla_params"])
    drone_viz_params = DroneVizParams(**scenario_dict["drone_viz_params"])
    pred_params      = PredictionParams()

    vehicles_params_list = []

    policy_type   = "lk_pi"

    for vp_dict in scenario_dict["vehicle_params"]:
        if vp_dict["role"] == "static":
            # Not generating static vehicles
            # vehicles_params_list.append( VehicleParams(**vp_dict) )
            continue
        elif vp_dict["role"] == "target":
            vp_dict["policy_type"] = policy_type
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        elif vp_dict["role"] == "ego":
            vp_dict.update(ego_init_dict)
            vp_dict["policy_type"] = policy_type
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        else:
            raise ValueError(f"Invalid vehicle role: {vp_dict['role']}")

    runner = RunIntersectionScenario(carla_params,
                                     drone_viz_params,
                                     vehicles_params_list,
                                     pred_params,
                                     savedir)
    return runner

def get_drone_snapshot(runner):
    # Get a single drone image on which to overlay trajectories.
    img_drone = None
    with CarlaSyncMode(runner.world, runner.drone, fps=runner.carla_fps) as sync_mode:
        _, img = sync_mode.tick(timeout=runner.timeout)
        img_drone = np.frombuffer(img.raw_data, dtype=np.uint8)
        img_drone = np.reshape(img_drone, (img.height, img.width, 4))
        img_drone = img_drone[:, :, :3]
        img_drone = cv2.resize(img_drone, (runner.viz_params.img_width, runner.viz_params.img_height), interpolation = cv2.INTER_AREA)
    return img_drone

def overlay_trajectories(img, runner, line_thickness=5, goal_radius=10):
    # Code to overlay the reference trajectories for every agent.
    def xy_to_px_center(xy):
        px = runner.A_world_to_drone @ xy + runner.b_world_to_drone
        center_x = int(px[0])
        center_y = int(px[1])
        return center_x, center_y

    for (veh_policy, veh_color) in zip(runner.vehicle_policies, runner.vehicle_colors):
        veh_color = veh_color[::-1] # RGB to BGR

        xy_traj = veh_policy._frenet_traj.trajectory[:, 1:3]

        pts = [xy_to_px_center(xy) for xy in xy_traj]
        for px_ind in range(len(pts)-1):
            cv2.line(img, pts[px_ind], pts[px_ind+1], veh_color, thickness=line_thickness)

        cv2.circle(img, pts[-1], goal_radius, veh_color, thickness=-1)


if __name__ == '__main__':

    TOWN_NUM = 7 # 5 or 7

    if TOWN_NUM == 5:
        scenario_suffix = ""
        scenarios_to_overlay = [1, 2, 3]
    elif TOWN_NUM == 7:
        scenario_suffix = "_t7"
        scenarios_to_overlay = [1, 2, 3, 4]
    else:
        raise ValueError(TOWN_NUM)

    img = None
    for scenario_num in scenarios_to_overlay:
        # Loading + Setup.
        scenario_path = os.path.join(scriptdir, f"scenarios/scenario_{scenario_num:02d}{scenario_suffix}.json")
        ego_init_path = os.path.join(scriptdir, "scenarios/ego_init_00.json")

        scenario_dict = json.load(open(scenario_path, "r"))
        ego_init_dict = json.load(open(ego_init_path, "r"))
        scenario_name = scenario_path.split("/")[-1].split('.json')[0]
        savedir = os.path.join( os.path.abspath(__file__).split("scripts")[0], "results/route_viz/" )

        runner = None
        try:
            runner = setup_intersection_scenario(scenario_dict, ego_init_dict, savedir)

            img = get_drone_snapshot(runner)

            overlay_trajectories(img, runner)

        except Exception as e:
            print(e)

        finally:
            if runner:
                for actor in runner.vehicle_actors:
                    actor.destroy()
                runner.drone.destroy()
            cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(savedir, f"scenario_route{scenario_suffix}_{scenario_num}.png"), img)
