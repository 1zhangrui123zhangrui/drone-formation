#!/usr/bin/env python3
"""
prepare_datasets.py - Build normalized train/val/test datasets from raw MAT files.

This version enforces leakage-safe dataset construction:
1. group by scene and drone
2. split discontinuous time-series on timestamp gaps
3. split raw rows into train/val/test before normalization
4. compute normalization stats from train rows only
5. build windows inside each split chunk only, so windows never cross
   scene/drone/time-gap boundaries and do not overlap across splits
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io


def load_raw_mat(mat_path: Path):
    raw = scipy.io.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    data = raw["data"]
    if isinstance(data, np.ndarray):
        data = data[0, 0]

    X = np.asarray(data.X, dtype=np.float64)
    Y = np.asarray(data.Y, dtype=np.float64)
    drone_id = np.asarray(data.drone_id, dtype=np.float64)
    t_vec = np.asarray(data.t_vec, dtype=np.float64)

    if X.ndim != 2 or X.shape[1] != 15:
        raise ValueError(f"{mat_path}: expected X to be Nx15, got {X.shape}")
    if Y.ndim != 2 or Y.shape[1] != 4:
        raise ValueError(f"{mat_path}: expected Y to be Nx4, got {Y.shape}")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError(f"{mat_path}: found NaN/Inf in raw arrays")

    return {
        "X": X,
        "Y": Y,
        "drone_id": drone_id,
        "t_vec": t_vec,
    }


def fit_normalize_stats(data: np.ndarray):
    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=0)
    std[std < 1e-8] = 1.0
    return {"mean": mean, "std": std}


def apply_normalize(data: np.ndarray, stats):
    return (data - stats["mean"]) / stats["std"]


def sliding_window(sequence: np.ndarray, window_size: int, stride: int):
    num_rows, num_dims = sequence.shape
    if num_rows < window_size:
        return np.zeros((window_size, num_dims, 0), dtype=np.float32)

    starts = range(0, num_rows - window_size + 1, stride)
    windows = np.empty((window_size, num_dims, len(list(starts))), dtype=np.float32)
    for idx, start in enumerate(range(0, num_rows - window_size + 1, stride)):
        windows[:, :, idx] = sequence[start:start + window_size, :]
    return windows


def split_contiguous_segments(sample, scene_name: str, gap_threshold: float):
    drone_ids = np.asarray(sample["drone_id"]).reshape(-1)
    t_vec = np.asarray(sample["t_vec"]).reshape(-1)
    x15 = np.asarray(sample["X"])
    y4 = np.asarray(sample["Y"])

    segments = []
    for drone_id in sorted(int(v) for v in np.unique(drone_ids)):
        mask = drone_ids == drone_id
        if not np.any(mask):
            continue

        t_d = t_vec[mask]
        x_d = x15[mask, :]
        y_d = y4[mask, :]

        order = np.argsort(t_d, kind="stable")
        t_d = t_d[order]
        x_d = x_d[order, :]
        y_d = y_d[order, :]

        boundaries = [0]
        dt = np.diff(t_d)
        for idx, delta in enumerate(dt, start=1):
            if (not np.isfinite(delta)) or delta <= 0.0 or delta > gap_threshold:
                boundaries.append(idx)
        boundaries.append(len(t_d))

        for seg_idx in range(len(boundaries) - 1):
            start = boundaries[seg_idx]
            end = boundaries[seg_idx + 1]
            if end <= start:
                continue
            segments.append(
                {
                    "scene": scene_name,
                    "drone_id": drone_id,
                    "segment_id": f"{scene_name}_d{drone_id}_seg{seg_idx:02d}",
                    "t_vec": t_d[start:end],
                    "X15": x_d[start:end, :],
                    "Y4": y_d[start:end, :],
                }
            )
    return segments


def split_raw_ranges(num_rows: int, window: int, split_ratio):
    guard = window - 1
    effective = num_rows - 2 * guard
    if effective <= 0:
        return {
            "train": (0, num_rows),
            "val": (0, 0),
            "test": (0, 0),
            "guard": guard,
            "effective_rows": effective,
        }

    n_train = matlab_round_positive(split_ratio[0] * effective)
    n_val = matlab_round_positive(split_ratio[1] * effective)
    if n_train + n_val > effective:
        n_val = max(0, effective - n_train)
    n_test = effective - n_train - n_val

    tr0, tr1 = 0, n_train
    va0, va1 = tr1 + guard, tr1 + guard + n_val
    te0, te1 = va1 + guard, num_rows

    return {
        "train": (tr0, tr1),
        "val": (va0, va1),
        "test": (te0, te1),
        "guard": guard,
        "effective_rows": effective,
        "allocated_rows": {"train": n_train, "val": n_val, "test": n_test},
    }


def chunk_to_windows(chunk_x: np.ndarray, chunk_y: np.ndarray, window: int, stride: int):
    windows = sliding_window(chunk_x, window, stride)
    num_windows = windows.shape[2]
    if num_windows == 0:
        return windows, np.zeros((0, chunk_y.shape[1]), dtype=np.float32)

    label_idx = (window - 1) + np.arange(num_windows) * stride
    labels = chunk_y[label_idx, :].astype(np.float32)
    return windows.astype(np.float32), labels


def concat_windows(windows_list, input_dim: int, window: int):
    if not windows_list:
        return np.zeros((window, input_dim, 0), dtype=np.float32)
    return np.concatenate(windows_list, axis=2)


def concat_labels(labels_list):
    if not labels_list:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(labels_list, axis=0)


def matlab_round_positive(value: float) -> int:
    return int(np.floor(value + 0.5))


def save_split(out_dir: Path, prefix: str, windows: np.ndarray, labels: np.ndarray, i_tr, i_va, i_te):
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = {"X": windows[:, :, i_tr], "Y": labels[i_tr, :].astype(np.float32)}
    va = {"X": windows[:, :, i_va], "Y": labels[i_va, :].astype(np.float32)}
    te = {"X": windows[:, :, i_te], "Y": labels[i_te, :].astype(np.float32)}

    scipy.io.savemat(str(out_dir / f"{prefix}_train.mat"), {"tr": tr}, do_compression=True)
    scipy.io.savemat(str(out_dir / f"{prefix}_val.mat"), {"va": va}, do_compression=True)
    scipy.io.savemat(str(out_dir / f"{prefix}_test.mat"), {"te": te}, do_compression=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare normalized datasets from raw MAT files")
    parser.add_argument("--raw-dir", default="data/processed/raw", help="Directory containing scene MAT files")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory for datasets")
    parser.add_argument("--window", type=int, default=20, help="Sliding window length")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split permutation")
    parser.add_argument("--gap-threshold", type=float, default=0.15,
                        help="Split raw sequences when adjacent timestamps differ by more than this many seconds")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    files = sorted(raw_dir.glob("*.mat"))
    if not files:
        raise SystemExit(f"No .mat files found in {raw_dir}")

    segments = []
    sample_counts = {}

    for mat_path in files:
        print(f"[prepare] loading {mat_path.name}")
        sample = load_raw_mat(mat_path)
        sample_counts[mat_path.name] = int(sample["X"].shape[0])
        segments.extend(split_contiguous_segments(sample, mat_path.stem, args.gap_threshold))

    total_raw_samples = sum(seg["X15"].shape[0] for seg in segments)
    print(f"[prepare] total raw samples across contiguous segments: {total_raw_samples}")

    split_ratio = (0.7, 0.2, 0.1)
    train_rows_15 = []
    train_rows_9 = []
    segment_report = []

    for seg in segments:
        split_info = split_raw_ranges(seg["X15"].shape[0], args.window, split_ratio)
        seg["split_info"] = split_info
        report = {
            "segment_id": seg["segment_id"],
            "scene": seg["scene"],
            "drone_id": int(seg["drone_id"]),
            "rows": int(seg["X15"].shape[0]),
            "guard_rows": int(split_info["guard"]),
            "effective_rows": int(split_info["effective_rows"]),
        }
        for split_name in ("train", "val", "test"):
            start, end = split_info[split_name]
            chunk_len = max(0, end - start)
            report[f"{split_name}_rows"] = int(chunk_len)
            if split_name == "train" and chunk_len > 0:
                train_rows_15.append(seg["X15"][start:end, :])
                train_rows_9.append(seg["X15"][start:end, 6:15])
        segment_report.append(report)

    if not train_rows_15:
        raise SystemExit("No training rows available after boundary-safe splitting")

    x15_train_raw = np.vstack(train_rows_15)
    x9_train_raw = np.vstack(train_rows_9)
    stats15 = fit_normalize_stats(x15_train_raw)
    stats9 = fit_normalize_stats(x9_train_raw)

    split_windows_15 = {"train": [], "val": [], "test": []}
    split_windows_9 = {"train": [], "val": [], "test": []}
    split_labels = {"train": [], "val": [], "test": []}

    for seg, report in zip(segments, segment_report):
        for split_name in ("train", "val", "test"):
            start, end = seg["split_info"][split_name]
            chunk_x15 = seg["X15"][start:end, :]
            chunk_y4 = seg["Y4"][start:end, :]

            chunk_x15_norm = apply_normalize(chunk_x15, stats15)
            chunk_x9_norm = apply_normalize(chunk_x15[:, 6:15], stats9)

            w15, y15 = chunk_to_windows(chunk_x15_norm, chunk_y4, args.window, args.stride)
            w9, y9 = chunk_to_windows(chunk_x9_norm, chunk_y4, args.window, args.stride)
            if y15.shape[0] != y9.shape[0]:
                raise ValueError(f"Window label count mismatch in {seg['segment_id']} split={split_name}")

            split_windows_15[split_name].append(w15)
            split_windows_9[split_name].append(w9)
            split_labels[split_name].append(y15)
            report[f"{split_name}_windows"] = int(w15.shape[2])

    dataset_15d = {
        "tr": {"X": concat_windows(split_windows_15["train"], 15, args.window), "Y": concat_labels(split_labels["train"])},
        "va": {"X": concat_windows(split_windows_15["val"], 15, args.window), "Y": concat_labels(split_labels["val"])},
        "te": {"X": concat_windows(split_windows_15["test"], 15, args.window), "Y": concat_labels(split_labels["test"])},
    }
    dataset_9d = {
        "tr": {"X": concat_windows(split_windows_9["train"], 9, args.window), "Y": concat_labels(split_labels["train"])},
        "va": {"X": concat_windows(split_windows_9["val"], 9, args.window), "Y": concat_labels(split_labels["val"])},
        "te": {"X": concat_windows(split_windows_9["test"], 9, args.window), "Y": concat_labels(split_labels["test"])},
    }

    scipy.io.savemat(str(out_dir / "dataset_15d_train.mat"), {"tr": dataset_15d["tr"]}, do_compression=True)
    scipy.io.savemat(str(out_dir / "dataset_15d_val.mat"), {"va": dataset_15d["va"]}, do_compression=True)
    scipy.io.savemat(str(out_dir / "dataset_15d_test.mat"), {"te": dataset_15d["te"]}, do_compression=True)
    scipy.io.savemat(str(out_dir / "dataset_9d_train.mat"), {"tr": dataset_9d["tr"]}, do_compression=True)
    scipy.io.savemat(str(out_dir / "dataset_9d_val.mat"), {"va": dataset_9d["va"]}, do_compression=True)
    scipy.io.savemat(str(out_dir / "dataset_9d_test.mat"), {"te": dataset_9d["te"]}, do_compression=True)

    scipy.io.savemat(
        str(out_dir / "norm_stats_15d.mat"),
        {"stats15": {"mean": stats15["mean"], "std": stats15["std"]}},
        do_compression=True,
    )
    scipy.io.savemat(
        str(out_dir / "norm_stats_9d.mat"),
        {"stats9": {"mean": stats9["mean"], "std": stats9["std"]}},
        do_compression=True,
    )

    manifest = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "window": args.window,
        "stride": args.stride,
        "seed": args.seed,
        "num_raw_samples": int(total_raw_samples),
        "gap_threshold": args.gap_threshold,
        "split_policy": "scene+drone contiguous segments, chronological raw-row split with guard rows, train-only normalization",
        "splits": {
            "train": int(dataset_15d["tr"]["X"].shape[2]),
            "val": int(dataset_15d["va"]["X"].shape[2]),
            "test": int(dataset_15d["te"]["X"].shape[2]),
        },
        "source_mats": [f.name for f in files],
        "source_sample_counts": sample_counts,
        "train_stats_source_rows": int(x15_train_raw.shape[0]),
        "segments": segment_report,
    }
    (out_dir / "dataset_build_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[prepare] done. Files in: {out_dir}")


if __name__ == "__main__":
    main()
