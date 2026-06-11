# Data Directory

This directory stores experiment artifacts that should not be versioned in Git:

- `raw_bags/`: rosbag recordings collected from ROS/Gazebo runs
- `processed/`: intermediate `.mat` files and cached datasets
- `trained_models/`: exported checkpoints and deployment weights

For collaboration, prefer Git LFS or DVC if these artifacts need to be shared.
