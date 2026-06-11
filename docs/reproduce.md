# Reproduce Guide

## Environment

- ROS: Noetic
- Gazebo: 11
- MATLAB: R2023b
- Python: 3.10

## Suggested workflow

1. Prepare rosbag data under `data/raw_bags/`.
2. Run MATLAB preprocessing scripts to generate `.mat` windows in `data/processed/`.
3. Train the desired controller under `matlab/train/`.
4. Export or copy the trained model to `data/trained_models/`.
5. Launch a ROS scene and point the online controller to the matching config file.
6. Save logs and metrics under a new `results/run_YYYY-MM-DD_NNN/` folder.

## Notes

- Keep large datasets and trained weights out of Git history.
- Snapshot the config used for every run to ensure reproducibility.
- Prefer naming runs with date and monotonically increasing indices.
