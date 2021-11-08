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
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        elif vp_dict["role"] == "target":
            continue
        elif vp_dict["role"] == "ego":
            vp_dict.update(ego_init_dict)
            # Note: here, confidence params don't matter practically but included for policy construction.
            vp_dict["policy_config"] = {"is_adaptive" : False, "conf_thresh_init" : 3.22}
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        else:
            raise ValueError(f"Invalid vehicle role: {vp_dict['role']}")

    runner = RunIntersectionScenario(carla_params,
                                     drone_viz_params,
                                     vehicles_params_list,
                                     pred_params,
                                     savedir)
    runner.run_scenario()

def run_with_tvs(scenario_dict, ego_init_dict, ego_policy_config, tv_policy_config, savedir):
    carla_params     = CarlaParams(**scenario_dict["carla_params"])
    drone_viz_params = DroneVizParams(**scenario_dict["drone_viz_params"])
    pred_params      = PredictionParams()

    vehicles_params_list = []

    for vp_dict in scenario_dict["vehicle_params"]:
        if vp_dict["role"] == "static":
            vp_dict["policy_config"] = {}
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        elif vp_dict["role"] == "target":
            vp_dict["policy_config"] = tv_policy_config
            vehicles_params_list.append( VehicleParams(**vp_dict) )
        elif vp_dict["role"] == "ego":
            vp_dict.update(ego_init_dict)
            vp_dict["policy_config"] = ego_policy_config
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
    TOWN_NUM = 7 # 5 or 7

    if TOWN_NUM == 7:
        scenes_to_evaluate = [1, 2, 3]
        town_suffix = "_t7"
    else:
        raise NotImplementedError("Only Town7 currently implemented.")

    scenario_folder = os.path.join( os.path.dirname( os.path.abspath(__file__)  ), "scenarios/" )
    scenarios_list = [f"{scenario_folder}scenario_{scene_num:02}{town_suffix}.json" for scene_num in scenes_to_evaluate]
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
            run_without_tvs(scenario_dict, ego_init_dict, savedir)

            # Run all ego policy options with target vehicles.
            for ego_conf_policy in ["low", "adaptive", "high"]:
                for tv_rat_policy in ["irrational", "rational"]:
                    if ego_conf_policy == "low":
                        ego_policy_config = {"is_adaptive" : False, "conf_thresh_init" : 0.211} # 10% confidence level
                    elif ego_conf_policy == "high":
                        ego_policy_config = {"is_adaptive" : False, "conf_thresh_init" : 9.210} # 99% confidence level
                    elif ego_conf_policy == "adaptive":
                        ego_policy_config = {"is_adaptive" : True,  "conf_thresh_init" : 3.220} # start at 80% confidence level
                    else:
                        raise ValueError(f"Invalid EV policy choice: {ego_conf_policy}")

                    if tv_rat_policy == "irrational":
                        tv_policy_config = {"is_rational" : False}
                    elif tv_rat_policy == "rational":
                        tv_policy_config = {"is_rational" : True}
                    else:
                        raise ValueError(f"Invalid TV policy choice: {tv_rat_policy}")

                    savedir = os.path.join( results_folder, f"{scenario_name}_{ego_init_name}_{ego_conf_policy}_{tv_rat_policy}")
                    run_with_tvs(scenario_dict, ego_init_dict, ego_policy_config, tv_policy_config, savedir)