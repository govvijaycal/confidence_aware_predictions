import os
import sys
import json
import numpy as np
import argparse

from models.regression import Regression
from models.multipath import MultiPath
from models.ekf import EKFKinematicBase, EKFKinematicCATR, EKFKinematicCAH, \
                       EKFKinematicCVTR, EKFKinematicCVH
from models.static_multiple_model import StaticMultipleModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train prediction models on either Nuscenes or L5Kit datasets.")
    parser.add_argument("--dataset", choices=["l5kit", "nuscenes"], type=str, required=True, "Which dataset to use.")
    args = parser.parse_args()

    repo_path = os.path.abspath(__file__).split('scripts')[0]

    # List of common entries to populate.
    train_set         = None
    val_set           = None
    name_model_list   = []
    model_kwargs_dict = {"batch_size" : 32}

    if args.dataset == "nuscenes":
        from datasets.splits import NUSCENES_TRAIN, NUSCENES_VAL

        train_set, val_set = NUSCENES_TRAIN, NUSCENES_VAL

        model_kwargs_dict["num_timesteps"]      = 12
        model_kwargs_dict["num_hist_timesteps"] = 2
        model_kwargs_dict["anchors"]            = np.load(repo_path + 'data/nuscenes_clusters_16.npy')
        model_kwargs_dict["weights"]            = np.load(repo_path + 'data/nuscenes_clusters_16_weights.npy')

        model_kwargs_dict["num_epochs"]         = 50
        model_kwargs_dict["log_epoch_freq"]     = 2
        model_kwargs_dict["save_epoch_freq"]    = 10

        name_model_list.append(['nuscenes_ekf_catr', EKFKinematicCATR])
        name_model_list.append(['nuscenes_ekf_cvtr', EKFKinematicCVTR])
        name_model_list.append(['nuscenes_ekf_cah', EKFKinematicCAH])
        name_model_list.append(['nuscenes_ekf_cvh', EKFKinematicCVH])
        name_model_list.append(['nuscenes_ekf_smm', StaticMultipleModel])

    elif args.dataset == "l5kit":
        from datasets.splits import L5KIT_TRAIN, L5KIT_VAL

        train_set, val_set = L5KIT_TRAIN, L5KIT_VAL

        model_kwargs_dict["num_timesteps"]      = 25
        model_kwargs_dict["num_hist_timesteps"] = 5
        model_kwargs_dict["anchors"]            = np.load(repo_path + 'data/l5kit_clusters_16.npy')
        model_kwargs_dict["weights"]            = np.load(repo_path + 'data/l5kit_clusters_16_weights.npy')

        model_kwargs_dict["num_epochs"]         = 20
        model_kwargs_dict["log_epoch_freq"]     = 1
        model_kwargs_dict["save_epoch_freq"]    = 5

        name_model_list.append(['l5kit_ekf_catr', EKFKinematicCATR])
        name_model_list.append(['l5kit_ekf_cvtr', EKFKinematicCVTR])
        name_model_list.append(['l5kit_ekf_cah', EKFKinematicCAH])
        name_model_list.append(['l5kit_ekf_cvh', EKFKinematicCVH])
        name_model_list.append(['l5kit_ekf_smm', StaticMultipleModel])

    else:
        raise ValueError(f"{args.dataset} is not a valid dataset.")


    for name_model in name_model_list:
        name, model = name_model

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

        logdir = f"{repo_path}log/{name}/"

        # Train the model.
        m.fit(train_set,
              val_set,
              logdir          = logdir,
              log_epoch_freq  = model_kwargs_dict["log_epoch_freq"],
              save_epoch_freq = model_kwargs_dict["save_epoch_freq"],
              num_epochs      = model_kwargs_dict["num_epochs"],
              batch_size      = model_kwargs_dict["batch_size"])
