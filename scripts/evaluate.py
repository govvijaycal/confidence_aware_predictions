import os
import numpy as np
import argparse
from pathlib import Path
import pickle

from models.regression import Regression
from models.multipath import MultiPath
from models.ekf import EKFKinematicBase, EKFKinematicCATR, EKFKinematicCAH, \
                       EKFKinematicCVTR, EKFKinematicCVH
from models.static_multiple_model import StaticMultipleModel
from evaluation.prediction_metrics import compute_trajectory_metrics, compute_set_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate prediction models on either Nuscenes or L5Kit datasets.")
    parser.add_argument("--dataset", choices=["l5kit", "nuscenes"], type=str, required=True, help="Which dataset to use.")
    args = parser.parse_args()

    repo_path = os.path.abspath(__file__).split('scripts')[0]

    # List of common entries to populate.
    ks_eval           = [1, 3, 5]          # Number of truncated modes to consider.
    betas_eval        = [2, 4, 6, 8, 10] # Confidence thresholds (squared Mahalanobis dist).

    train_set         = None
    val_set           = None
    test_set          = None

    name_model_weight_list   = []
    model_kwargs_dict        = {}

    if args.dataset == "nuscenes":
        from datasets.splits import NUSCENES_TRAIN, NUSCENES_VAL, NUSCENES_TEST

        train_set, val_set, test_set = NUSCENES_TRAIN, NUSCENES_VAL, NUSCENES_TEST

        model_kwargs_dict["num_timesteps"]      = 12
        model_kwargs_dict["num_hist_timesteps"] = 2
        model_kwargs_dict["anchors"]            = np.load(repo_path + 'data/nuscenes_clusters_16.npy')
        model_kwargs_dict["weights"]            = np.load(repo_path + 'data/nuscenes_clusters_16_weights.npy')

        name_model_weight_list.append(['nuscenes_ekf_catr', EKFKinematicCATR,    'params.pkl'])
        name_model_weight_list.append(['nuscenes_ekf_cvtr', EKFKinematicCVTR,    'params.pkl'])
        name_model_weight_list.append(['nuscenes_ekf_cah',  EKFKinematicCAH,     'params.pkl'])
        name_model_weight_list.append(['nuscenes_ekf_cvh',  EKFKinematicCVH,     'params.pkl'])
        name_model_weight_list.append(['nuscenes_ekf_smm',  StaticMultipleModel, 'params.pkl'])

    elif args.dataset == "l5kit":
        from datasets.splits import L5KIT_TRAIN, L5KIT_VAL, L5KIT_TEST

        train_set, val_set, test_set = L5KIT_TRAIN, L5KIT_VAL, L5KIT_TEST

        model_kwargs_dict["num_timesteps"]      = 25
        model_kwargs_dict["num_hist_timesteps"] = 5
        model_kwargs_dict["anchors"]            = np.load(repo_path + 'data/l5kit_clusters_16.npy')
        model_kwargs_dict["weights"]            = np.load(repo_path + 'data/l5kit_clusters_16_weights.npy')

        name_model_weight_list.append(['l5kit_ekf_catr', EKFKinematicCATR,    'params.pkl'])
        name_model_weight_list.append(['l5kit_ekf_cvtr', EKFKinematicCVTR,    'params.pkl'])
        name_model_weight_list.append(['l5kit_ekf_cah',  EKFKinematicCAH,     'params.pkl'])
        name_model_weight_list.append(['l5kit_ekf_cvh',  EKFKinematicCVH,     'params.pkl'])
        name_model_weight_list.append(['l5kit_ekf_smm',  StaticMultipleModel, 'params.pkl'])

    else:
        raise ValueError(f"{args.dataset} is not a valid dataset.")

    for name_model_weight in name_model_weight_list:
        name, model, weight = name_model_weight

        # Construct the model.
        if issubclass(model, EKFKinematicBase) or model == StaticMultipleModel:
            m = model()
        elif model == Regression:
            m = model(num_timesteps      = model_kwargs_dict["num_timesteps"],
                      num_hist_timesteps = model_kwargs_dict["num_hist_timesteps"])

        elif model == MultiPath:
            m = model(num_timesteps      = model_kwargs_dict["num_timesteps"],
                      num_hist_timesteps = model_kwargs_dict["num_hist_timesteps"],
                      anchors            = model_kwargs_dict["anchors"],
                      weights            = model_kwargs_dict["weights"])
        else:
            raise ValueError(f"Invalid model: {model}")

        # Load the weights.
        logdir = f"{repo_path}log/{name}/"
        m.load_weights(f"{logdir}{weight}")

        # Make predictions (only if never done before).
        for split_name in ['train', 'val', 'test']:
            print(f"Making predictions for split: {split_name} and model: {name}")
            preds_path = Path( f"{logdir}{split_name}_preds.pkl" )

            if not preds_path.exists():
                predict_dict = m.predict( eval(f"{split_name}_set") )
                pickle.dump( predict_dict, open(str(preds_path), "wb") )
            else:
                print("\tUsing existing predictions.")

        # Evaluate predictions and save results (only if never done before).
        for split_name in ['train', 'val', 'test']:
            print(f"Computing metrics for {split_name}")
            predict_dict = pickle.load( open(f"{logdir}{split_name}_preds.pkl", "rb") )

            metrics_path = Path( f"{logdir}{split_name}_preds_metrics.pkl" )
            if not metrics_path.exists():
                metrics_df = compute_trajectory_metrics(predict_dict,
                                                        ks_eval=ks_eval)
                metrics_df.to_pickle(str(metrics_path))
            else:
                print("\tUsing existing trajectory metrics.")

            set_metrics_path = Path( f"{logdir}{split_name}_preds_set_metrics.pkl" )
            if not set_metrics_path.exists():
                set_metrics_df = compute_set_metrics(predict_dict,
                                                     ks_eval=ks_eval,
                                                     betas_eval=betas_eval)
                set_metrics_df.to_pickle(str(set_metrics_path))
            else:
                print("\tUsing existing set metrics.")
