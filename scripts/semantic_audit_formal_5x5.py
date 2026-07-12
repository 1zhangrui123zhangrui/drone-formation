#!/usr/bin/env python3
"""Semantic checks for the formal 5-scene x 5-seed ROS bag set.

This is intentionally stricter than basic bag health auditing:
- basic audit checks duration, finite positions, and gross divergence;
- this script checks formation geometry, S3 phase changes, and S4/S2 relation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_formation_bags import DRONE_IDS, PAIR_IDS, read_bag, scene_metrics, synchronized_scene


SCENES = {
    "scene01_hover": {"label": "S1 Hover", "duration_min": 55.0},
    "scene02_circle": {"label": "S2 Circle", "duration_min": 85.0},
    "scene03_reconfig": {"label": "S3 Reconfiguration", "duration_min": 110.0},
    "scene04_wind": {"label": "S4 Wind", "duration_min": 85.0},
    "scene05_longtime": {"label": "S5 Longtime", "duration_min": 170.0},
}

SEEDS = [f"seed{i:02d}" for i in range(1, 6)]


def expected_bag_name(scene: str, seed: str) -> str:
    short = {
        "scene01_hover": "scene01_hover",
        "scene02_circle": "scene02_circle",
        "scene03_reconfig": "scene03_reconfig",
        "scene04_wind": "scene04_wind",
        "scene05_longtime": "scene05_longtime",
    }[scene]
    return f"{short}_{seed}.bag"


def phase_pair_mean(metrics: dict, t: np.ndarray, lo: float, hi: float, key: str) -> float:
    mask = (t >= lo) & (t < hi)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.mean(np.vstack(metrics[key])[:, mask], axis=0)))


def audit_one(path: Path, scene: str, seed: str) -> dict:
    data = read_bag(path)
    t, pos, pdes = synchronized_scene(data)
    metrics = scene_metrics(t, pos, pdes)

    row = {
        "scene": scene,
        "seed": seed,
        "bag": path.name,
        "duration_s": metrics["duration_s"],
        "track_rmse_m": metrics["track_rmse_m"],
        "track_p95_m": metrics["track_p95_m"],
        "track_max_m": metrics["track_max_m"],
        "formation_rmse_m": metrics["formation_rmse_m"],
        "formation_p95_m": metrics["formation_p95_m"],
        "formation_max_m": metrics["formation_max_m"],
        "min_pair_distance_m": metrics["min_pair_distance_m"],
        "status": "PASS",
        "notes": "",
    }

    failures: list[str] = []
    warnings: list[str] = []

    if metrics["duration_s"] < SCENES[scene]["duration_min"]:
        failures.append(f"duration<{SCENES[scene]['duration_min']}")
    if metrics["min_pair_distance_m"] < 0.50:
        failures.append("min_pair_distance<0.50m")
    if metrics["track_max_m"] > 2.0:
        failures.append("track_max>2.0m")

    if scene == "scene03_reconfig":
        desired_1 = phase_pair_mean(metrics, t, 15, 39, "pair_des")
        desired_2 = phase_pair_mean(metrics, t, 48, 76, "pair_des")
        desired_3 = phase_pair_mean(metrics, t, 88, 115, "pair_des")
        actual_1 = phase_pair_mean(metrics, t, 15, 39, "pair_actual")
        actual_2 = phase_pair_mean(metrics, t, 48, 76, "pair_actual")
        actual_3 = phase_pair_mean(metrics, t, 88, 115, "pair_actual")

        row.update(
            {
                "s3_des_phase1_m": desired_1,
                "s3_des_phase2_m": desired_2,
                "s3_des_phase3_m": desired_3,
                "s3_act_phase1_m": actual_1,
                "s3_act_phase2_m": actual_2,
                "s3_act_phase3_m": actual_3,
                "s3_des_delta12_m": abs(desired_2 - desired_1),
                "s3_des_delta23_m": abs(desired_3 - desired_2),
                "s3_act_err_phase1_m": abs(actual_1 - desired_1),
                "s3_act_err_phase2_m": abs(actual_2 - desired_2),
                "s3_act_err_phase3_m": abs(actual_3 - desired_3),
            }
        )

        if abs(desired_2 - desired_1) < 0.15:
            failures.append("S3 desired phase1->phase2 change too small")
        # Phase 2 and 3 can have similar mean pair distance despite different topology;
        # require actual tracking rather than a large mean-distance delta there.
        if max(abs(actual_1 - desired_1), abs(actual_2 - desired_2), abs(actual_3 - desired_3)) > 0.08:
            failures.append("S3 actual phase mean does not follow desired")

    if failures:
        row["status"] = "FAIL"
        row["notes"] = "; ".join(failures)
    elif warnings:
        row["status"] = "WARN"
        row["notes"] = "; ".join(warnings)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-dir", default="data/raw_bags/v2/formal_5x5")
    parser.add_argument("--out-dir", default="results/audits")
    args = parser.parse_args()

    bag_dir = Path(args.bag_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scene in SCENES:
        for seed in SEEDS:
            path = bag_dir / expected_bag_name(scene, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            print(f"[semantic] {scene} {seed}: {path}")
            rows.append(audit_one(path, scene, seed))

    s2 = [r for r in rows if r["scene"] == "scene02_circle"]
    s4 = [r for r in rows if r["scene"] == "scene04_wind"]
    s2_rmse = float(np.mean([r["track_rmse_m"] for r in s2]))
    s4_rmse = float(np.mean([r["track_rmse_m"] for r in s4]))
    wind_ratio = s4_rmse / s2_rmse if s2_rmse > 0 else float("nan")

    aggregate = {
        "num_bags": len(rows),
        "num_pass": sum(r["status"] == "PASS" for r in rows),
        "num_warn": sum(r["status"] == "WARN" for r in rows),
        "num_fail": sum(r["status"] == "FAIL" for r in rows),
        "s2_mean_track_rmse_m": s2_rmse,
        "s4_mean_track_rmse_m": s4_rmse,
        "s4_vs_s2_rmse_ratio": wind_ratio,
        "s4_note": (
            "S4 bags are stable, but wind effect is weak in teacher closed-loop data "
            "if this ratio is close to 1.0."
        ),
    }

    fieldnames = sorted({key for row in rows for key in row.keys()})
    csv_path = out_dir / "formal_5x5_semantic_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / "formal_5x5_semantic_aggregate.json"
    json_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n")

    print(f"[semantic] wrote {csv_path}")
    print(f"[semantic] wrote {json_path}")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    if aggregate["num_fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
