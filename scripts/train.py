import os
import numpy as np
import argparse

from models.regression import Regression
from models.multipath import MultiPath
from models.ekf import EKFKinematicBase, EKFKinematicCATR, EKFKinematicCAH, \
                       EKFKinematicCVTR, EKFKinematicCVH
from models.static_multiple_model import StaticMultipleModel
from models.lane_follower import LaneFollower

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train prediction models on either Nuscenes or L5Kit datasets.")
    parser.add_argument("--dataset", choices=["l5kit", "nuscenes"], type=str, required=True, help="Which dataset to use.")
    args = parser.parse_args()

    repo_path = os.path.abspath(__file__).split('scripts')[0]

    # List of common entries to populate.
    train_set          = None
    val_set            = None
    name_model_list    = []
    model_kwargs_dict  = {"batch_size" : 32, "lr_min" : 1e-4, "lr_max" : 1e-3}
    lane_follower_dict = {}

    if args.dataset == "nuscenes":
        from datasets.splits import NUSCENES_TRAIN, NUSCENES_VAL

        train_set, val_set = NUSCENES_TRAIN, NUSCENES_VAL

        model_kwargs_dict["num_timesteps"]      = 12
        model_kwargs_dict["num_hist_timesteps"] = 2
        model_kwargs_dict["anchors"]            = np.load(repo_path + 'data/nuscenes_clusters_16.npy')
        model_kwargs_dict["weights"]            = None

        model_kwargs_dict["num_epochs"]         = 50
        model_kwargs_dict["log_epoch_freq"]     = 2
        model_kwargs_dict["save_epoch_freq"]    = 10

        lane_follower_dict["dataset_name"]     = "nuscenes"
        lane_follower_dict["n_max_modes"] = 16
        lane_follower_dict["ekf_cvtr_weights_path"] = repo_path + 'log/nuscenes_ekf_cvtr/params.pkl'

        name_model_list.append(['nuscenes_ekf_catr', EKFKinematicCATR])
        name_model_list.append(['nuscenes_ekf_cvtr', EKFKinematicCVTR])
        name_model_list.append(['nuscenes_ekf_cah', EKFKinematicCAH])
        name_model_list.append(['nuscenes_ekf_cvh', EKFKinematicCVH])
        name_model_list.append(['nuscenes_ekf_smm', StaticMultipleModel])

        name_model_list.append(['nuscenes_lane_follower_um', LaneFollower]) # Unimodal
        name_model_list.append(['nuscenes_lane_follower_mm', LaneFollower]) # Multimodal

        name_model_list.append(['nuscenes_regression_no_context', Regression])
        name_model_list.append(['nuscenes_multipath_no_context', MultiPath])
        name_model_list.append(['nuscenes_regression', Regression])
        name_model_list.append(['nuscenes_multipath', MultiPath])

    elif args.dataset == "l5kit":
        from datasets.splits import L5KIT_TRAIN, L5KIT_VAL

        train_set, val_set = L5KIT_TRAIN, L5KIT_VAL

        model_kwargs_dict["num_timesteps"]      = 25
        model_kwargs_dict["num_hist_timesteps"] = 5
        model_kwargs_dict["anchors"]            = np.load(repo_path + 'data/l5kit_clusters_16.npy')
        model_kwargs_dict["weights"]            = None

        model_kwargs_dict["num_epochs"]         = 20
        model_kwargs_dict["log_epoch_freq"]     = 1
        model_kwargs_dict["save_epoch_freq"]    = 5

        lane_follower_dict["dataset_name"]     = "l5kit"
        lane_follower_dict["n_max_modes"] = 16
        lane_follower_dict["ekf_cvtr_weights_path"] = repo_path + 'log/l5kit_ekf_cvtr/params.pkl'

        name_model_list.append(['l5kit_ekf_catr', EKFKinematicCATR])
        name_model_list.append(['l5kit_ekf_cvtr', EKFKinematicCVTR])
        name_model_list.append(['l5kit_ekf_cah', EKFKinematicCAH])
        name_model_list.append(['l5kit_ekf_cvh', EKFKinematicCVH])
        name_model_list.append(['l5kit_ekf_smm', StaticMultipleModel])

        name_model_list.append(['l5kit_lane_follower_um', LaneFollower]) # Unimodal
        name_model_list.append(['l5kit_lane_follower_mm', LaneFollower]) # Multimodal

        name_model_list.append(['l5kit_regression_no_context', Regression])
        name_model_list.append(['l5kit_multipath_no_context', MultiPath])
        name_model_list.append(['l5kit_regression', Regression])
        name_model_list.append(['l5kit_multipath', MultiPath])

    else:
        raise ValueError(f"{args.dataset} is not a valid dataset.")


    for name_model in name_model_list:
        name, model = name_model

        use_context = True
        if "no_context" in name:
            use_context = False

        # Construct the model.
        if issubclass(model, EKFKinematicBase) or model == StaticMultipleModel:
            m = model()
        elif model == LaneFollower:
            if "_um" in name:
                num_modes = 1
            else:
                num_modes = lane_follower_dict["n_max_modes"]

            m = model(lane_follower_dict["dataset_name"],
                      n_max_modes = num_modes,
                      ekf_cvtr_weights_path=lane_follower_dict["ekf_cvtr_weights_path"])

        elif model == Regression:
            m = model(num_timesteps      = model_kwargs_dict["num_timesteps"],
                      num_hist_timesteps = model_kwargs_dict["num_hist_timesteps"],
                      use_context = use_context,
                      lr_min = model_kwargs_dict["lr_min"],
                      lr_max = model_kwargs_dict["lr_max"])

        elif model == MultiPath:
            m = model(num_timesteps      = model_kwargs_dict["num_timesteps"],
                      num_hist_timesteps = model_kwargs_dict["num_hist_timesteps"],
                      anchors            = model_kwargs_dict["anchors"],
                      weights            = model_kwargs_dict["weights"],
                      use_context = use_context,
                      lr_min = model_kwargs_dict["lr_min"],
                      lr_max = model_kwargs_dict["lr_max"])
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
