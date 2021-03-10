# confidence_aware_predictions
Implementation of data-driven and model-based predictions that are confidence-aware, using current error to inform future conservatism.
 * Includes an implementation of MultiPath (link: https://arxiv.org/abs/1910.05449)
 * Includes physics-based, context-agnostic Extended Kalman Filters based on simple motion models (constant velocity / acceleration / turning rate ).
 * (WIP): Includes a physics-based model that considers road context and short-term predictions of nearby agents.
 * (WIP): Adds an adjustable confidence threshold based on the residual likelihood (i.e. Mahalanobis distance to the nearest active mode).

This repo also has a script to run closed-loop evaluation in Carla with a rational and irrational NPC agent, used to understand how above models work in conjunction with planning.
