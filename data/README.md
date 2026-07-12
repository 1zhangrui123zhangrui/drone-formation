# Data Directory

This folder stores non-source experiment artifacts.

## Layout

- `raw_bags/`: recorded ROS/Gazebo bags
- `processed/raw/`: `.mat` files converted from bag data
- `processed/`: normalized train/val/test datasets and normalization stats
- `trained_models/`: MATLAB-exported model checkpoints

## Current State

As of `2026-07-12`, this directory contains:

- the older `seed42` bags kept for diagnosis/history
- the formal paper-facing bag directory `raw_bags/v2/formal_5x5/`
  - expected names are `scene01_hover_seed01.bag` ... `scene05_longtime_seed05.bag`
  - the completed formal set is 5 scenes x 5 seeds = 25 accepted bags
  - `scripts/record_bags_v2.sh all allseeds` records this set
  - `scripts/verify_formal_5x5_bags.py --audit` verifies completeness and per-bag audit status
  - the latest full verification passed on all 25 bags
  - `scripts/semantic_audit_formal_5x5.py` verifies formation-level semantics such as S3 phase changes and collision-free formation keeping
- formal audit outputs in `results/audits/`:
  - `formal_5x5_semantic_summary.csv`
  - `formal_5x5_semantic_aggregate.json`
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

The current `processed/raw/` and `processed/` datasets were rebuilt from the earlier audited `v2` single-run bag set, not yet from the formal `raw_bags/v2/formal_5x5/` repeated-run set. The old pre-formal `v2` bag files have been removed from this directory to avoid accidental reuse. As of `2026-07-10`, the processed datasets also use a stricter boundary-safe build policy: windows do not cross scene/drone/time-gap boundaries, train/val/test splits keep guard rows, and normalization stats are fit from train rows only. The current MATLAB checkpoints were retrained against that stricter earlier dataset, but they are still only single-training artifacts and must not be reported as formal 5-seed paper results.

The formal `S4` wind bags are stable and pass semantic audit, but the teacher closed-loop tracking RMSE is only slightly higher than `S2` (`S4/S2 ≈ 1.04` in the latest semantic audit). Treat this as a weak wind-effect dataset unless later Student closed-loop evaluation or stronger wind settings show a clearer disturbance response.

## Reporting Rule

For the paper-facing experiments, do not report single-run best cases from this directory as final evidence. The canonical `processed/` datasets are for model building, while paper metrics should come from separately tracked independent evaluation runs and be summarized as multi-run averages with variance information.

## Sharing

If these artifacts need to be shared across machines or collaborators, prefer Git LFS or DVC instead of normal Git history.
