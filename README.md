# confidence\_aware\_predictions
Implementation of data-driven and model-based predictions that are confidence-aware, using current error to inform future conservatism.
 * Includes an implementation of MultiPath (link: https://arxiv.org/abs/1910.05449)
 * Includes physics-based, context-agnostic Extended Kalman Filters based on simple motion models (constant velocity / acceleration / turning rate ).
 * (WIP): Includes a physics-based model that considers road context and short-term predictions of nearby agents.
 * (WIP): Adds an adjustable confidence threshold based on the residual likelihood (i.e. Mahalanobis distance to the nearest active mode).

This repo also has a script to run closed-loop evaluation in Carla with a rational and irrational NPC agent, used to understand how above models work in conjunction with planning.

## Setup for Nuscenes Devkit
To set the devkit up for dataset preparation / prediction visualization, you can do the following:
 1. git clone https://github.com/nutonomy/nuscenes-devkit
 2. Adjust the PYTHONPATH: https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/installation.md#setup-pythonpath
 3. Install requirements with conda (see envs/environment.yml which unifies conda/pip install for Carla, Nuscenes, and L5Kit).
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

## Setup for L5Kit
**TODO**

## Setup for Carla
**TODO**