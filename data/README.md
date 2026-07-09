# Data Directory

This folder stores non-source experiment artifacts.

## Layout

- `raw_bags/`: recorded ROS/Gazebo bags
- `processed/raw/`: `.mat` files converted from bag data
- `processed/`: normalized train/val/test datasets and normalization stats
- `trained_models/`: MATLAB-exported model checkpoints

## Current State

As of `2026-07-09`, this directory already contains:

- all 6 scene bags for the `seed42` run
- converted raw `.mat` files for all 6 scenes
- prepared `9D` and `15D` datasets
- trained checkpoints for:
  - `c1_lstm9d.mat`
  - `c2_lstm15d.mat`
  - `c3a_bilstm.mat`

The proposed final model `c3_bidir_attn.mat` is still missing.

## Important Caveat

Several dynamic-scene bags contain severe trajectory explosions or partial drone failures. The prepared datasets are therefore based on sample filtering, not on fully clean four-drone runs. Do not treat the current artifacts as final paper-grade evaluation data.

## Sharing

If these artifacts need to be shared across machines or collaborators, prefer Git LFS or DVC instead of normal Git history.
