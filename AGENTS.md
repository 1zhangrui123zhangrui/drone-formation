# AGENTS.md

This file is for agent-facing working rules. It is intentionally shorter and more operational than `README.md`.

## Project Priorities

1. Experimental correctness is more important than speed.
2. Academic honesty is non-negotiable.
3. README files must stay synchronized with the real experiment state after meaningful changes.

## Experimental Ground Truth

- The current canonical raw data source is the audited six-scene `v2` bag set under `data/raw_bags/v2/`.
- The current canonical processed datasets are under `data/processed/`.
- `data/trained_models/` still contains older checkpoints unless they are explicitly retrained against the canonical `v2` dataset.

## Reporting Rules

- Do not treat a single best run as a paper result.
- Paper-facing metrics must come from independent repeated runs and be reported with average-based statistics.
- Keep training-set construction bags separate from repeated evaluation bags.
- Do not mix stabilized reproduction results with paper-target-parameter results.

## Parameter Integrity

- Current stabilized reproduction configuration includes `omega=0.2` and `kp_h=1.5`.
- If a result uses those values, label it as a stabilized reproduction configuration.
- If returning toward paper target parameters, do single-variable stability validation first.

## Current Engineering Reality

- Dynamic-scene stabilization fixes already landed in:
  - `ros_ws/src/drone_sim/scripts/teacher_controller.py`
  - `ros_ws/src/drone_sim/scripts/scene_driver.py`
  - `scripts/record_bags_v2.sh`
- The next high-priority stage is retraining models on the canonical `v2` dataset with the MATLAB training pipeline, then running repeated evaluation.
- If the task is paper-faithful model training, prefer `matlab/train/*.m` and `scripts/train_models_matlab.sh`. Treat Python training scripts as engineering fallback only.
- Verified Windows MATLAB <-> WSL ROS connection method for this machine:
  - Launch ROS in WSL with `ROS_MASTER_URI=http://localhost:11311`, `ROS_IP=127.0.0.1`, `ROS_HOSTNAME=localhost`
  - Connect from Windows MATLAB with `rosinit('http://localhost:11311','NodeHost','localhost')`
  - Do not switch MATLAB to `172.20.10.4` or `172.30.32.1` unless the networking mode is deliberately reconfigured and revalidated

## Documentation Sync

- Update `README.md` when project-wide status, next steps, or experiment conclusions change.
- Update `data/README.md` when canonical datasets, raw mats, or model-artifact status changes.
- Keep statements traceable to actual files and executed experiments.
