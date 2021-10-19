import os
import glob
import json
import pdb

from scenarios.run_intersection_scenario import CarlaParams, DroneVizParams, VehicleParams, PredictionParams, RunIntersectionScenario

def run_without_tvs(scenario_dict, ego_init_dict, savedir):
    carla_params     = CarlaParams(**scenario_dict["carla_params"])
    drone_viz_params = DroneVizParams(**scenario_dict["drone_viz_params"])
    pred_params      = PredictionParams()

    vehicles_params_list = []

    for vp_dict in scenario_dict["vehicle_params"]:
        if vp_dict["role"] == "static":
            continue
            # vehicles_params_list.append( VehicleParams(**vp_dict) )
        elif vp_dict["role"] == "target":
            pass
        elif vp_dict["role"] == "ego":
            vp_dict.update(ego_init_dict)
            vp_dict["policy_type"] = "blsmpc"
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        else:
            raise ValueError(f"Invalid vehicle role: {vp_dict['role']}")

    runner = RunIntersectionScenario(carla_params,
                                     drone_viz_params,
                                     vehicles_params_list,
                                     pred_params,
                                     savedir)
    runner.run_scenario()

def run_with_tvs(scenario_dict, ego_init_dict, ego_policy_config, savedir):
    carla_params     = CarlaParams(**scenario_dict["carla_params"])
    drone_viz_params = DroneVizParams(**scenario_dict["drone_viz_params"])
    pred_params      = PredictionParams()

    vehicles_params_list = []

    if ego_policy_config == "blsmpc":
        policy_type   = "blsmpc"
        policy_config = ""
    elif ego_policy_config.startswith("smpc"):
        policy_type = "smpc"
        policy_config = ego_policy_config.split("smpc_")[-1]
    else:
        raise ValueError(f"Invalid ego policy config: {ego_policy_config}")

    for vp_dict in scenario_dict["vehicle_params"]:
        if vp_dict["role"] == "static":
            # Not generating static vehicles
            # vehicles_params_list.append( VehicleParams(**vp_dict) )
            continue
        elif vp_dict["role"] == "target":
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        elif vp_dict["role"] == "ego":
            vp_dict.update(ego_init_dict)
            vp_dict["policy_type"] = policy_type
            vp_dict["smpc_config"] = policy_config
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        else:
            raise ValueError(f"Invalid vehicle role: {vp_dict['role']}")

    runner = RunIntersectionScenario(carla_params,
                                     drone_viz_params,
                                     vehicles_params_list,
                                     pred_params,
                                     savedir)
    runner.run_scenario()

if __name__ == '__main__':
    scenario_folder = os.path.join( os.path.dirname( os.path.abspath(__file__)  ), "scenarios/" )
    scenarios_list = sorted(glob.glob(scenario_folder + "scenario_*.json"))
    # scenarios_list = glob.glob(scenario_folder + "scenario_03.json")
    results_folder = os.path.join( os.path.abspath(__file__).split("scripts")[0], "results" )

    for scenario in scenarios_list:
        # Load the scenario and generate parameters.
        scenario_dict = json.load(open(scenario, "r"))
        scenario_name = scenario.split("/")[-1].split('.json')[0]

        ego_init_list = sorted(scenario_dict["ego_init_jsons"])
        for ego_init in ego_init_list:
            # Load the ego vehicle parameters.
            ego_init_dict = json.load(open(os.path.join(scenario_folder, ego_init), "r"))
            ego_init_name = ego_init.split(".json")[0]


            # Run first without any target vehicles.
            savedir = os.path.join( results_folder, f"{scenario_name}_{ego_init_name}_notv")
            # run_without_tvs(scenario_dict, ego_init_dict, savedir)
            # break
            # Run all ego policy options with target vehicles.
            # for ego_policy_config in ["blsmpc", "smpc_full", "smpc_open_loop", "smpc_no_switch"]:
            for ego_policy_config in ["blsmpc", "smpc_open_loop", "smpc_no_switch"]:
            # # for ego_policy_config in ["smpc_no_switch"]:
              savedir = os.path.join( results_folder,
                                      f"{scenario_name}_{ego_init_name}_{ego_policy_config}")
              run_with_tvs(scenario_dict, ego_init_dict, ego_policy_config, savedir)
            # break
