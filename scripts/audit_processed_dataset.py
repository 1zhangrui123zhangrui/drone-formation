#!/usr/bin/env python3
"""Independent audit of the canonical formal-5x5 processed datasets."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io as sio
from prepare_datasets import load_raw_mat, split_contiguous_segments, split_raw_ranges

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/processed/raw"
OUT = ROOT / "data/processed"
SCENES = [f"scene0{i}_{name}" for i, name in enumerate(
    ["hover", "circle", "reconfig", "wind", "longtime"], start=1)]
SEEDS = [f"seed{i:02d}" for i in range(1, 6)]


def load_split(dim, split, field):
    return sio.loadmat(OUT / f"dataset_{dim}d_{split}.mat", squeeze_me=True,
                       struct_as_record=False)[field]


def strings(value):
    return [str(x) for x in np.atleast_1d(value).tolist()]


def main():
    manifest = json.loads((OUT / "dataset_build_manifest.json").read_text())
    failures, checks = [], {}
    raw_counts = Counter()
    raw_total = 0
    for path in sorted(RAW.glob("*.mat")):
        obj = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
        scene, seed = path.stem.rsplit("_", 1)
        drones = np.atleast_1d(obj.drone_id).astype(int)
        raw_total += len(drones)
        for drone in range(1, 5):
            raw_counts[(scene, seed, drone)] = int(np.sum(drones == drone))
    expected = {(s, seed, d) for s in SCENES for seed in SEEDS for d in range(1, 5)}
    missing = sorted(k for k in expected if raw_counts[k] == 0)
    checks["raw_mat_count"] = len(list(RAW.glob("*.mat")))
    checks["raw_sample_total"] = raw_total
    checks["coverage_cells"] = len(expected) - len(missing)
    checks["coverage_expected"] = len(expected)
    checks["coverage_min_samples"] = min(raw_counts[k] for k in expected)
    if checks["raw_mat_count"] != 25 or missing or raw_total != manifest["num_raw_samples"]:
        failures.append("raw coverage/count mismatch")

    split_keys = {}
    for dim in (9, 15):
        for split, field in (("train", "tr"), ("val", "va"), ("test", "te")):
            obj = load_split(dim, split, field)
            p = obj.provenance
            scenes, seeds, segs = strings(p.scene), strings(p.seed), strings(p.segment_id)
            drones = np.atleast_1d(p.drone_id).astype(int)
            times = np.atleast_1d(p.label_time).astype(float)
            n = obj.Y.shape[0]
            if obj.X.shape[-1] != n or not all(len(v) == n for v in (scenes, seeds, segs, drones, times)):
                failures.append(f"{dim}D {split} provenance length mismatch")
            keys = set(zip(segs, np.round(times, 6)))
            split_keys[(dim, split)] = keys
            checks[f"{dim}d_{split}_windows"] = n
            checks[f"{dim}d_{split}_coverage_cells"] = len(set(zip(scenes, seeds, drones)))
            if len(set(zip(scenes, seeds, drones))) != 100:
                failures.append(f"{dim}D {split} incomplete scene/seed/drone coverage")
        a, b, c = (split_keys[(dim, x)] for x in ("train", "val", "test"))
        overlap = {"train_val": len(a & b), "train_test": len(a & c), "val_test": len(b & c)}
        checks[f"{dim}d_split_label_overlap"] = overlap
        if any(overlap.values()): failures.append(f"{dim}D split label leakage")

    # Recompute normalization strictly from the raw ranges declared in the manifest.
    train15, train9 = [], []
    for path in sorted(RAW.glob("*.mat")):
        scene, seed = path.stem.rsplit("_", 1)
        for seg in split_contiguous_segments(load_raw_mat(path), scene, seed, path.name,
                                             manifest["gap_threshold"]):
            start, end = split_raw_ranges(len(seg["X15"]), manifest["window"],
                                          (0.7, 0.2, 0.1))["train"]
            train15.append(seg["X15"][start:end])
            train9.append(seg["X15"][start:end, 6:15])
    for dim, rows, var in ((15, train15, "stats15"), (9, train9, "stats9")):
        x = np.vstack(rows)
        expected_mean, expected_std = x.mean(0), x.std(0)
        expected_std[expected_std < 1e-8] = 1.0
        stats = sio.loadmat(OUT / f"norm_stats_{dim}d.mat", squeeze_me=True,
                            struct_as_record=False)[var]
        err = max(float(np.max(np.abs(stats.mean - expected_mean))),
                  float(np.max(np.abs(stats.std - expected_std))))
        checks[f"{dim}d_train_only_norm_max_abs_error"] = err
        if err > 1e-10: failures.append(f"{dim}D normalization mismatch")

    # S3 transition neighborhoods (around the two commanded changes) must be represented.
    s3 = load_split(15, "train", "tr").provenance
    all_s3_times = []
    for split, field in (("train", "tr"), ("val", "va"), ("test", "te")):
        p = load_split(15, split, field).provenance
        all_s3_times.extend(t for scene, t in zip(strings(p.scene), np.atleast_1d(p.label_time))
                            if scene == "scene03_reconfig")
    checks["s3_transition_40s_windows"] = int(sum(38 <= t <= 50 for t in all_s3_times))
    checks["s3_transition_80s_windows"] = int(sum(76 <= t <= 90 for t in all_s3_times))
    if not checks["s3_transition_40s_windows"] or not checks["s3_transition_80s_windows"]:
        failures.append("S3 transition segment absent")

    checks["manifest_dataset_id"] = manifest.get("dataset_id")
    checks["boundary_policy"] = manifest.get("split_policy")
    if manifest.get("dataset_id") != "formal_5x5_v2": failures.append("wrong manifest source")
    report = {"status": "PASS" if not failures else "FAIL", "failures": failures, "checks": checks}
    path = OUT / "processed_dataset_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
