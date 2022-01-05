# confidence\_aware\_predictions

Implementation of data-driven and model-based predictions that are confidence-aware, using current error to inform future conservatism.
 * Includes an implementation of MultiPath and corresponding Regression baseline (link: https://arxiv.org/abs/1910.05449)
 * Includes physics-based, context-agnostic Extended Kalman Filters based on simple motion models (constant velocity / acceleration / turning rate ).
 * Includes a physics-based model that considers road context and short-term predictions of nearby agents.
 * Incorporates predictions with a lanekeeping MPC formulation.
 * Adds an adjustable confidence threshold based, roughly, on the residual likelihood (i.e. Mahalanobis distance to the nearest active mode).

This repo also has a script to run closed-loop evaluation in Carla with a rational and irrational NPC agent, used to understand how above models work in conjunction with planning.

---

## Installation

### Overall Setup

First follow instractions in envs/setup.txt to setup the conda environment.  Then follow the dataset-specific instructions following.

### Setup for Nuscenes Devkit

To set the devkit up for dataset preparation / prediction visualization, you can do the following:
 1. git clone https://github.com/govvijaycal/nuscenes-devkit
 2. Follow install directions in envs/setup.txt.
 4. Download minimal data needed for nuscenes prediction and set up following structure.  The folder containing the dataset, **nuscenes_datadir**, should have the following structure:
      - maps
        * 36092f0b03a857c6a3403e25b4b7aab3.png
        * 37819e65e09e5547b8a3ceaefba56bb2.png
        * 53992ee3023e5494b90c316c183be829.png
        * 93406b464a165eaba6d9de76ca09f5da.png
        * basemap
          - boston-seaport.png
          - singapore-hollandvillage.png
          - singapore-onenorth.png
          - singapore-queenstown.png
        * expansion
          - boston-seaport.json
          - singapore-hollandvillage.json
          - singapore-onenorth.json
          - singapore-queenstown.json
        * prediction
          - prediction.json
      - v1.0-trainval
        * attribute.json
        * calibrated\_sensor.json
        * category.json
        * ego\_pose.json
        * instance.json
        * log.json
        * map.json
        * sample\_annotation.json
        * sample\_data.json
        * sample.json
        * scene.json
        * sensor.json
        * visibility.json

### Setup for L5Kit
1. Download the dataset as detailed here: https://github.com/govvijaycal/l5kit#1-datasets
2. Clone the repo: https://github.com/govvijaycal/l5kit
3. Follow install directions as detailed in envs/setup.txt.

### Setup for Carla/Simulation

1. Install CARLA (I used 0.9.10).
2. See envs/setup.txt to set up Carla and pytope.

---

## Data Pipeline

After finishing setup, the next step is to convert datasets into a unified format for training (tfrecords).
Refer to scripts/datasets/prep_l5kit.py and scripts/datasets/prep_nuscenes.py for details on how this is done.

For MultiPath, the trajectory anchors need to be identified: that is done in scripts/datasets/identify_anchors.py.

The preparation and parsing of tfrecord entries can be found in scripts/datasets/tfrecord_utils.py.

For training models, refer to scripts/train.py and the corresponding models in scripts/models/.

For evaluation, refer to scripts/evaluate.py with metrics defined in scripts/evaluation/.

---

## Models

In general, the code should be straighforward to follow for things like the EKF and Multipath variants.  The tricky one is the lane_follower implementation, which uses the scene context determined using the utils in models/context_providers/.  The context providers cache lane centerlines and corresponding speed limits, if available.  There is a very simple Bayesian model used here for prediction: lane priors based on proximity to nearest lane pose and cost likelihood based on inputs determined using a lane tracking model.

---

## Carla Simulation

The main script here is run_all_scenarios.py, which looks up the scenarios defined in carla/scenarios/ and executes all specified initial conditions.  Scenarios are defined by providing an intersection layout (csv) and a json specificying all agents and configuration parameters.  The closed loop trajectories and a video can be saved per execution of a scenario.  See scenarios/run_interesection_scenario.py for further details.

The specific control policies are defined in carla/policies.  These include a PI lanekeeping agent (used for the target vehicle) and a collision-avoiding MPC lanekeeping agent (used for the ego vehicle and considering the multimodal predictions).  confidence_thresh_manager.py implements the online confidence adjustment based on consistency of the target vehicle's behaviors with the corresponding predictions.

With respect to the predictions, it is important to note that the input format / features in Carla must be consistent with the format used during training.  I wrote carla/rasterizer/ to mimic the structure I used for the l5kit rasterizer (see l5kit repo link above in setup section).