import os
import sys
import numpy as np 
from tqdm import tqdm
import pandas as pd

scriptdir = os.path.abspath(__file__).split('scripts')[0] + 'scripts/'
sys.path.append(scriptdir)
from evaluation.gmm_prediction import GMMPrediction

def compute_length_and_curvature(pose_traj):
    # pose_traj is a N by 3 array with columns x, y, theta.
    xy_traj      = pose_traj[:, :2]
    heading_traj = pose_traj[:, 2]    
    
    length_traj  = np.sum(np.linalg.norm(np.diff(xy_traj, axis=0), axis=1)) + \
                   np.linalg.norm(xy_traj[0, :])
    curv_traj    = (heading_traj[-1]) / length_traj # crude approximation of curvature

    return length_traj, curv_traj


def eval_prediction_dict(predict_dict, anchors, model_name, ks_eval = [1,3,5]): 
    data_list = []    

    columns   = ["sample", "instance", "model", "length", "curvature"]
    columns.extend([f"traj_LL_{k}" for k in ks_eval])
    columns.extend([f"class_top_{k}" for k in ks_eval])
    columns.extend([f"min_ade_{k}" for k in ks_eval])
    columns.extend([f"min_fde_{k}" for k in ks_eval])
    columns.extend([f"minmax_dist_{k}" for k in ks_eval])

    for key in tqdm(predict_dict.keys()):
        future_traj_gt = predict_dict[key]['future_traj'] # t, x, y, theta
        future_xy_gt = future_traj_gt[:, 1:3] # x, y
        gmm_pred       = predict_dict[key]['gmm_pred']

        n_modes     = len(gmm_pred.keys())
        n_timesteps = future_traj_gt.shape[0]
        mode_probabilities = np.array( [gmm_pred[mode]['mode_probability'] for mode in range(n_modes)] )
        mus                = np.array( [gmm_pred[mode]['mus'] for mode in range(n_modes)] )
        sigmas             = np.array( [gmm_pred[mode]['sigmas'] for mode in range(n_modes)] )

        gmm_pred     = GMMPrediction(n_modes, n_timesteps, mode_probabilities, mus, sigmas)

        sample_token   = '_'.join( key.split('_')[:-2] ) # FIXME!
        instance_token = '_'.join( key.split('_')[-2:] )

        data_list_entry = [sample_token, instance_token, model_name]

        data_list_entry.extend([*compute_length_and_curvature(future_traj_gt[:, 1:])])

        if n_modes == 1:
            num_ks = len(ks_eval)
            data_list_entry.extend([gmm_pred.compute_trajectory_log_likelihood(future_xy_gt)]*num_ks)
            data_list_entry.extend([1]*num_ks) # unimodal
            data_list_entry.extend([gmm_pred.compute_min_ADE(future_xy_gt)]*num_ks)
            data_list_entry.extend([gmm_pred.compute_min_FDE(future_xy_gt)]*num_ks)
            data_list_entry.extend([gmm_pred.compute_minmax_d(future_xy_gt)]*num_ks)
        else:
            gmm_pred_ks  = [gmm_pred.get_top_k_GMM(k) for k in ks_eval]

            data_list_entry.extend([gmm_pred_k.compute_trajectory_log_likelihood(future_xy_gt) \
                                    for gmm_pred_k in gmm_pred_ks])
            data_list_entry.extend(gmm_pred.get_class_top_k_scores(future_xy_gt, anchors, ks_eval))
            data_list_entry.extend([gmm_pred_k.compute_min_ADE(future_xy_gt) \
                                    for gmm_pred_k in gmm_pred_ks])
            data_list_entry.extend([gmm_pred_k.compute_min_FDE(future_xy_gt) \
                                    for gmm_pred_k in gmm_pred_ks])
            data_list_entry.extend([gmm_pred_k.compute_minmax_d(future_xy_gt) \
                                    for gmm_pred_k in gmm_pred_ks])

        data_list.append(data_list_entry)
    metrics_df = pd.DataFrame(data_list, columns=columns)
    return metrics_df

def average_selected_keys(df, keys_to_average):
    avg_dict = {}
    for key in keys_to_average:
        avg_dict[key] = np.mean(df[key])
    return avg_dict
