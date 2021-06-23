import os
import sys
import numpy as np 
from tqdm import tqdm
import pandas as pd

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)
from evaluation.gmm_prediction import GMMPrediction

def extract_traj_and_pred(predict_instance):
    traj_actual = predict_instance["future_traj"] # ground truth trajectory, {[t, x, y, theta]}

    # Overwrite gmm_pred with the GMMPrediction class for easier analysis.
    gmm_pred    = predict_instance["gmm_pred"]    # corresponding GMM prediction

    n_modes     = len(gmm_pred.keys())
    mode_probabilities = np.array( [gmm_pred[mode]['mode_probability']
                                    for mode in range(n_modes)] )
    mus                = np.array( [gmm_pred[mode]['mus'] for mode in range(n_modes)] )
    sigmas             = np.array( [gmm_pred[mode]['sigmas'] for mode in range(n_modes)] )
    n_timesteps = mus.shape[1]

    gmm_pred = GMMPrediction(n_modes, n_timesteps, mode_probabilities, mus, sigmas)

    return traj_actual, gmm_pred

####################################################################################################
########################################## TRAJ METRICS ############################################
def make_traj_metric_dict(ks_eval):
    # Prepares a template dictionary for storing computed metrics.
    keys   = ["sample", "instance", "length", "curvature", "num_modes"]
    keys.extend([f"traj_LL_{k}" for k in ks_eval])
    keys.extend([f"class_top_{k}" for k in ks_eval])
    keys.extend([f"min_ade_{k}" for k in ks_eval])
    keys.extend([f"min_fde_{k}" for k in ks_eval])

    return {k:None for k in keys}

def compute_length_and_curvature(pose_traj):
    # Computes length and curvature of a trajectory via crude approximation.
    # pose_traj is a N by 3 array with columns x, y, theta.
    xy_traj      = pose_traj[:, :2]
    heading_traj = pose_traj[:, 2]

    length_traj  = np.sum(np.linalg.norm(np.diff(xy_traj, axis=0), axis=1)) + \
                   np.linalg.norm(xy_traj[0, :])
    curv_traj    = (heading_traj[-1]) / length_traj

    return length_traj, curv_traj

def compute_trajectory_metrics(predict_dict, ks_eval=[1,3,5]):
    # Given a collection of GMM predictions, computes trajectory metrics per dataset instance
    # returning a Pandas Dataframe for further analysis.
    data_list = []

    for key in tqdm(predict_dict.keys()):
        traj_actual, gmm_pred = extract_traj_and_pred(predict_dict[key])
        traj_xy = traj_actual[:, 1:3]

        metric_dict = make_traj_metric_dict(ks_eval)

        # Note which dataset element this is.
        if "track" in key:
            # L5kit dataset.
            metric_dict["sample"]   = key.split("_track")[0]        # scene_X_frame_Y
            metric_dict["instance"] = "_".join(key.split("_")[-2:]) # track_Z
        else:
            # Nuscenes dataset.
            metric_dict["sample"], metric_dict["instance"] = key.split("_")

        # Compute length/curvature of the ground truth trajectory.
        metric_dict["length"], metric_dict["curvature"] =
            compute_length_curvature(traj_actual[:, 1:])

        metric_dict["num_modes"] = n_modes
        if metric_dict["num_modes"] == 1:
            traj_ll_um = gmm_pred.compute_trajectory_log_likelihood(traj_xy)
            min_ade_um = gmm_pred.compute_min_ADE(traj_xy)
            min_fde_um = gmm_pred.compute_min_FDE(traj_xy)
            for k in ks_eval:
                metric_dict[f"traj_LL_{k}"]   = traj_ll_um
                metric_dict[f"class_top_{k}"] = 1
                metric_dict[f"min_ade_{k}"]   = min_ade_um
                metric_dict[f"min_fde_{k}"]   = min_fde_um
        else:
            metric_dict[f"class_top_{k}"] = gmm_pred.get_class_top_k_scores(traj_xy, ks_eval)
            for k in ks_eval:
                trunc_gmm_k = gmm_pred.get_top_k_GMM(k)
                metric_dict[f"traj_LL_{k}"] = trunc_gmm_k.compute_trajectory_log_likelihood(traj_xy)
                metric_dict[f"min_ade_{k}"] = trunc_gmm_k.compute_min_ADE(traj_xy)
                metric_dict[f"min_fde_{k}"] = trunc_gmm_k.compute_min_FDE(traj_xy)

        data_list.append(metric_dict)

    metrics_df = pd.DataFrame(data_list)
    assert ~np.any(metrics_df.isnull())
    return metrics_df

####################################################################################################
########################################### SET METRICS ############################################
def make_set_metric_dict(ks_eval, betas_eval):
    # Prepares a template dictionary for storing computed set metrics.
    keys = ["set_acc_k{k}_b{beta}" for k in ks_eval for b in betas_eval]
    keys.extend([f"set_area_k{k}_b{beta}" for k in ks_eval for b in betas_eval])
    return {k:None for k in keys}

def compute_set_metrics(predict_dict, ks_eval=[1,3,5], betas_eval=[2,4,6,8,10]):
    # Given a collection of GMM predictions, computes set metrics per dataset instance
    # returning a Pandas Dataframe for further analysis.
    data_list = []

    for key in tqdm(predict_dict.keys()):
        traj_actual, gmm_pred = extract_traj_and_pred(predict_dict[key])
        traj_xy = traj_actual[:, 1:3]

        set_metric_dict = make_set_metric_dict(ks_eval, betas_eval)

        # Note which dataset element this is.
        if "track" in key:
            # L5kit dataset.
            set_metric_dict["sample"]   = key.split("_track")[0]        # scene_X_frame_Y
            set_metric_dict["instance"] = "_".join(key.split("_")[-2:]) # track_Z
        else:
            # Nuscenes dataset.
            set_metric_dict["sample"], set_metric_dict["instance"] = key.split("_")

        for k in ks_eval:
            trunc_gmm_k = gmm_pred.get_top_k_GMM(k)

            for beta in betas_eval:
                set_metric_dict[f"set_acc_k{k}_b{beta}"] =
                    trunc_gmm_k.compute_set_accuracy(traj_xy, beta)
                set_metric_dict[f"set_area_k{k}_b{beta}"] =
                    trunc_gmm_k.compute_set_area(beta)

        data_list.append(set_metric_dict)

    set_metric_df = pd.DataFrame(data_list)
    assert ~np.any(set_metric_df.isnull())
    return set_metric_df
