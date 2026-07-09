#!/usr/bin/env python3
"""
prepare_datasets.py - Build normalized train/val/test datasets from raw MAT files.

Python equivalent of matlab/data_pipeline/prepare_all_datasets.m so the audited
dataset can be rebuilt even when MATLAB/Octave is unavailable in the shell.
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


def normalize(data: np.ndarray):
    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=0)
    std[std < 1e-8] = 1.0
    normalized = (data - mean) / std
    return normalized, {"mean": mean, "std": std}


def sliding_window(sequence: np.ndarray, window_size: int, stride: int):
    num_rows, num_dims = sequence.shape
    if num_rows < window_size:
        return np.zeros((window_size, num_dims, 0), dtype=np.float32)

    starts = range(0, num_rows - window_size + 1, stride)
    windows = np.empty((window_size, num_dims, len(list(starts))), dtype=np.float32)
    for idx, start in enumerate(range(0, num_rows - window_size + 1, stride)):
        windows[:, :, idx] = sequence[start:start + window_size, :]
    return windows


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
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    files = sorted(raw_dir.glob("*.mat"))
    if not files:
        raise SystemExit(f"No .mat files found in {raw_dir}")

    all_x15 = []
    all_y4 = []
    sample_counts = {}

    for mat_path in files:
        print(f"[prepare] loading {mat_path.name}")
        sample = load_raw_mat(mat_path)
        all_x15.append(sample["X"])
        all_y4.append(sample["Y"])
        sample_counts[mat_path.name] = int(sample["X"].shape[0])

    x15 = np.vstack(all_x15)
    y4 = np.vstack(all_y4)
    print(f"[prepare] total raw samples: {x15.shape[0]}")

    x15_norm, stats15 = normalize(x15)
    x9_raw = x15[:, 6:15]
    x9_norm, stats9 = normalize(x9_raw)

    w15 = sliding_window(x15_norm, args.window, args.stride)
    w9 = sliding_window(x9_norm, args.window, args.stride)
    num_windows = w15.shape[2]
    y_w = y4[args.window - 1:args.window - 1 + num_windows, :]
    print(f"[prepare] windows: {num_windows} (W={args.window}, stride={args.stride})")

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(num_windows)
    n_train = matlab_round_positive(0.7 * num_windows)
    n_val = matlab_round_positive(0.2 * num_windows)

    i_tr = idx[:n_train]
    i_va = idx[n_train:n_train + n_val]
    i_te = idx[n_train + n_val:]

    save_split(out_dir, "dataset_15d", w15, y_w, i_tr, i_va, i_te)
    save_split(out_dir, "dataset_9d", w9, y_w, i_tr, i_va, i_te)

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
        "num_raw_samples": int(x15.shape[0]),
        "num_windows": int(num_windows),
        "splits": {
            "train": int(len(i_tr)),
            "val": int(len(i_va)),
            "test": int(len(i_te)),
        },
        "source_mats": [f.name for f in files],
        "source_sample_counts": sample_counts,
    }
    (out_dir / "dataset_build_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[prepare] done. Files in: {out_dir}")


if __name__ == "__main__":
    main()
