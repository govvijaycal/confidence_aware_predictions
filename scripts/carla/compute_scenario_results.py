import os
import re
import glob
import numpy as np
import pandas as pd

import matplotlib
font = {'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)
from evaluation.closed_loop_metrics import ScenarioResult, ClosedLoopTrajectory, load_scenario_result

def get_metric_dataframe(results_dir):
    scenario_dirs = sorted(glob.glob(results_dir + "*scenario*"))

    if len(scenario_dirs) == 0:
        raise ValueError(f"Could not detect scenario results in directory: {results_dir}")

    # Assumption: format is *scenario_<scene_num>_ego_init_<init_num>_policy
    dataframe = []
    for scenario_dir in scenario_dirs:
        scene_num = int( scenario_dir.split("scenario_")[-1].split("_")[0] )
        init_num  = int( scenario_dir.split("ego_init_")[-1].split("_")[0])
        policy    = re.split("ego_init_[0-9]*_", scenario_dir)[-1]

        pkl_path = os.path.join(scenario_dir, "scenario_result.pkl")
        if not os.path.exists(pkl_path):
            raise RuntimeError(f"Unable to find a scenario_result.pkl in directory: {scenario_dir}")

        notv_pkl_path = os.path.join(re.split(f"{policy}", scenario_dir)[0] + "notv", "scenario_result.pkl")
        if not os.path.exists(notv_pkl_path):
            raise RuntimeError(f"Unable to find a notv scenario_result.pkl in location: {notv_pkl_path}")

        # Load scenario dict for this policy and the notv case (for Hausdorff distance).
        sr      = load_scenario_result(pkl_path)
        notv_sr = load_scenario_result(notv_pkl_path)

        metrics_dict = sr.compute_metrics()
        metrics_dict["hausdorff_dist_notv"] = sr.compute_ego_hausdorff_dist(notv_sr)
        dmins = metrics_dict.pop("dmins_per_TV")
        if dmins:
            metrics_dict["dmin_TV"] = np.amin(dmins) # take the closest distance to any TV in the scene
        else:
            metrics_dict["dmin_TV"] = np.nan # no moving TVs in the scene
        metrics_dict["scenario"] = scene_num
        metrics_dict["initial"]  = init_num
        metrics_dict["policy"]   = policy
        dataframe.append(metrics_dict)

    return pd.DataFrame(dataframe)

def make_trajectory_viz_plot(results_dir, color1="r", color2="b"):
    scenario_dirs = sorted(glob.glob(results_dir + "*scenario*"))

    if len(scenario_dirs) == 0:
        raise ValueError(f"Could not detect scenario results in directory: {results_dir}")

    # Assumption: format is *scenario_<scene_num>_ego_init_<init_num>_policy
    dataframe = []
    for scenario_dir in scenario_dirs:
        scene_num = int( scenario_dir.split("scenario_")[-1].split("_")[0] )
        init_num  = int( scenario_dir.split("ego_init_")[-1].split("_")[0])
        policy    = re.split("ego_init_[0-9]*_", scenario_dir)[-1]

        pkl_path = os.path.join(scenario_dir, "scenario_result.pkl")
        if not os.path.exists(pkl_path):
            raise RuntimeError(f"Unable to find a scenario_result.pkl in directory: {scenario_dir}")

        notv_pkl_path = os.path.join(re.split(f"{policy}", scenario_dir)[0] + "notv", "scenario_result.pkl")
        if not os.path.exists(notv_pkl_path):
            raise RuntimeError(f"Unable to find a notv scenario_result.pkl in location: {notv_pkl_path}")

        # Load scenario dict for this policy and the notv case (for Hausdorff distance).
        sr      = load_scenario_result(pkl_path)
        notv_sr = load_scenario_result(notv_pkl_path)

        # Get time vs. frenet projection for this policy's ego trajectory vs the notv case.
        ts, s_wrt_notv, ey_wrt_notv, epsi_wrt_notv = sr.compute_ego_frenet_projection(notv_sr)

        # Get the closest distance to a TV across all timesteps identified above.
        d_closest = np.ones(ts.shape) * np.inf
        d_trajs_TV = sr.get_distances_to_TV()

        for tv_ind in range(len(d_trajs_TV)):
            t_traj = d_trajs_TV[tv_ind][:,0]
            d_traj = d_trajs_TV[tv_ind][:,1]

            d_interp = np.interp(ts, t_traj, d_traj, left=np.inf, right=np.inf)

            d_closest = np.minimum(d_interp, d_closest)

        # Make the plots.
        t0 = sr.ego_closed_loop_trajectory.state_trajectory[0, 0]
        trel = ts - t0
        plt.figure()
        ax1 = plt.gca()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Route Progress (m)", color=color1)
        ax1.plot(trel[::2], s_wrt_notv[::2], color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_yticks(np.arange(0., 101., 10.))

        ax2 = ax1.twinx()
        ax2.set_ylabel("Closest TV distance (m)", color=color2)
        ax2.plot(trel[::2], d_closest[::2], color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_yticks(np.arange(0., 51., 5.))

        plt.tight_layout()
        plt.savefig(f'{scenario_dir}/traj_viz.svg', bbox_inches='tight')

def normalize_by_notv(df):
    # Compute metrics that involve normalizing by the notv scenario execution.
    # Right now, these metrics are completion_time and max_lateral_acceleration.

    # Add the new columns with normalized values.
    df = df.assign( max_lateral_acceleration_norm = df.max_lateral_acceleration,
                    completion_time_norm = df.completion_time)

    # Do the normalization per scenario / ego initial condition.
    scene_inits = set( [f"{s}_{i}" for (s,i) in zip(df.scenario, df.initial)])

    for scene_init in scene_inits:
        s, i = [int(x) for x in scene_init.split("_")]
        s_i_inds = np.logical_and(df.scenario == s, df.initial == i)
        notv_inds = np.logical_and(s_i_inds, df.policy=="notv")

        if np.sum(notv_inds) != 1:
            raise RuntimeError(f"Unable to find a unique notv execution for scenario {s}, initialization {i}.")

        notv_ind       = np.where(notv_inds)[0].item()
        notv_lat_accel = df.max_lateral_acceleration[notv_ind]
        notv_time      = df.completion_time[notv_ind]

        lat_accel_normalized = df[s_i_inds].max_lateral_acceleration / notv_lat_accel
        df.loc[s_i_inds, "max_lateral_acceleration_norm"] = lat_accel_normalized

        time_normalized = df[s_i_inds].completion_time / notv_time
        df.loc[s_i_inds, "completion_time_norm"] = time_normalized

    return df

def aggregate(df):
    df_aggregate = []

    for scenario in set(df.scenario):
        for policy in set(df.policy):
            subset_inds = np.logical_and( df.scenario == scenario, df.policy == policy )

            res = df[subset_inds].mean(numeric_only=True)
            res.drop(["initial", "scenario"], inplace=True)

            res_dict = {"scenario": int(scenario), "policy": policy}
            res_dict.update(res.to_dict())
            df_aggregate.append(res_dict)

    return pd.DataFrame(df_aggregate)

if __name__ == '__main__':
    compute_metrics = True
    make_traj_viz   = True
    results_dir = os.path.join(os.path.abspath(__file__).split('scripts')[0], 'results/')

    if compute_metrics:
        dataframe = get_metric_dataframe(results_dir)
        dataframe.to_csv(os.path.join(results_dir, "df_full.csv"), sep=",")

        dataframe = normalize_by_notv(dataframe)
        dataframe.to_csv(os.path.join(results_dir, "df_norm.csv"), sep=",")

        dataframe  = aggregate(dataframe)
        dataframe.to_csv(os.path.join(results_dir, "df_final.csv"), sep=",")

    if make_traj_viz:
        make_trajectory_viz_plot(results_dir)
