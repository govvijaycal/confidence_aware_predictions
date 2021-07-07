import os
import numpy as np

from models.regression import Regression
from models.multipath import MultiPath

if __name__ == '__main__':
    repo_path = os.path.abspath(__file__).split('scripts')[0]

    dataset  = 'l5kit'                 # 'nuscenes' or 'l5kit'
    name     = 'l5kit_multipath'       # log folder name
    model    = MultiPath               # Regression or MultiPath
    weight   = '00010_epochs.h5'       # name of h5 weights
    savedir  = 'l5kit_multipath_10/'   # name for the saved model in SavedModel format

    if dataset == 'nuscenes':
        num_timesteps = 12
        num_hist_timesteps = 2
        anchors = np.load(repo_path + 'data/nuscenes_clusters_16.npy')
    elif dataset == 'l5kit':
        num_timesteps = 25
        num_hist_timesteps = 5
        anchors = np.load(repo_path + 'data/l5kit_clusters_16.npy')
    else:
        raise ValueError(f"Invalid dataset choice: {dataset}")

    # Construct the model.

    if model == Regression:
        m = model(num_timesteps      = num_timesteps,
                  num_hist_timesteps = num_hist_timesteps)

    if model == MultiPath:
        m = model(num_timesteps      = num_timesteps,
                  num_hist_timesteps = num_hist_timesteps,
                  anchors            = anchors)
    else:
        raise ValueError(f"Invalid model: {model}")

    # Load the weights.
    logdir = f"{repo_path}log/{name}/"
    m.load_weights(f"{logdir}{weight}")

    # Export the model.
    m.save_model(f"{logdir}{savedir}")

