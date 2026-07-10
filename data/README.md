# Data Directory

This folder stores non-source experiment artifacts.

## Layout

- `raw_bags/`: recorded ROS/Gazebo bags
- `processed/raw/`: `.mat` files converted from bag data
- `processed/`: normalized train/val/test datasets and normalization stats
- `trained_models/`: MATLAB-exported model checkpoints

## Current State

As of `2026-07-10`, this directory already contains:

- the older `seed42` bags kept for diagnosis/history
- a full six-scene audited `v2` bag set in `raw_bags/v2/`:
  - `scene01_hover_4drones_v2.bag`
  - `scene02_circle_4drones_v2_worldfix_entryfix.bag`
  - `scene02p_lemni_4drones_v2.bag`
  - `scene03_reconfig_4drones_v2.bag`
  - `scene04_wind_4drones_v2.bag`
  - `scene05_longtime_4drones_v2.bag`
- rebuilt raw `.mat` files in `processed/raw/`
- rebuilt `9D` and `15D` train/val/test datasets in `processed/`
- `dataset_build_manifest.json` describing the current dataset build
- trained checkpoints for:
  - `c1_lstm9d.mat`
  - `c2_lstm15d.mat`
  - `c3a_bilstm.mat`
  - `c3_bidir_attn.mat`

The latest canonical-dataset retraining, offline evaluation, and training audit have already been rerun. At the moment, `c3a_bilstm.mat` is the best offline model on the canonical test split, while `c3_bidir_attn.mat` exists but is clearly underperforming and should not be treated as the paper-ready main result yet.

## Important Caveat

The current `processed/raw/` and `processed/` datasets were rebuilt from the audited `v2` bag set and should now be treated as the canonical dataset inputs. As of `2026-07-10`, the processed datasets also use a stricter boundary-safe build policy: windows do not cross scene/drone/time-gap boundaries, train/val/test splits keep guard rows, and normalization stats are fit from train rows only. The current MATLAB checkpoints were retrained against this stricter canonical dataset, but they are still only single-training artifacts, not paper-facing repeated-run evidence.

## Reporting Rule

For the paper-facing experiments, do not report single-run best cases from this directory as final evidence. The canonical `processed/` datasets are for model building, while paper metrics should come from separately tracked independent evaluation runs and be summarized as multi-run averages with variance information.

## Sharing

If these artifacts need to be shared across machines or collaborators, prefer Git LFS or DVC instead of normal Git history.
