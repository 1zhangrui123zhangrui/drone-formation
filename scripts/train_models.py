#!/usr/bin/env python3
"""Train the paper model family on canonical MATLAB datasets using PyTorch."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class WindowDataset(Dataset):
    def __init__(self, mat_path: Path, split_name: str):
        mat = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
        split = mat[split_name][0, 0]
        x = np.asarray(split.X, dtype=np.float32)  # [W, D, N]
        y = np.asarray(split.Y, dtype=np.float32)  # [N, 4]

        if x.ndim != 3 or y.ndim != 2:
            raise ValueError(f"Unexpected dataset shapes in {mat_path}: X={x.shape}, Y={y.shape}")

        self.x = torch.from_numpy(np.transpose(x, (2, 0, 1)))  # [N, W, D]
        self.y = torch.from_numpy(y)
        self.window = int(x.shape[0])
        self.input_dim = int(x.shape[1])

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class LastStepLstmRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class StackedBiLstmRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.bilstm1 = nn.LSTM(input_size=input_dim, hidden_size=64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.bilstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.bilstm1(x)
        out = self.dropout(out)
        _, (hidden, _) = self.bilstm2(out)
        last = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.head(last)


class FeatureAttention(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="tanh")
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.tanh(self.linear(x))
        alpha = torch.softmax(scores, dim=-1)
        return alpha * x


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="tanh")
        nn.init.zeros_(self.proj.bias)
        nn.init.kaiming_normal_(self.score.weight, nonlinearity="linear")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = torch.tanh(self.proj(h))
        e = self.score(u).squeeze(-1)
        alpha = torch.softmax(e, dim=1)
        return torch.sum(h * alpha.unsqueeze(-1), dim=1)


class BiLstmDaRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.feature_attn = FeatureAttention(input_dim)
        self.bilstm1 = nn.LSTM(input_size=input_dim, hidden_size=64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.bilstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.temporal_attn = TemporalAttention(hidden_dim=128, attn_dim=64)
        self.head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_attn(x)
        out, _ = self.bilstm1(x)
        out = self.dropout(out)
        out, _ = self.bilstm2(out)
        context = self.temporal_attn(out)
        return self.head(context)


@dataclass
class ModelSpec:
    key: str
    train_file: str
    val_file: str
    test_file: str
    split_names: Tuple[str, str, str]
    checkpoint_name: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    "c1": ModelSpec("c1", "dataset_9d_train.mat", "dataset_9d_val.mat", "dataset_9d_test.mat", ("tr", "va", "te"), "c1_lstm9d.pt"),
    "c2": ModelSpec("c2", "dataset_15d_train.mat", "dataset_15d_val.mat", "dataset_15d_test.mat", ("tr", "va", "te"), "c2_lstm15d.pt"),
    "c3a": ModelSpec("c3a", "dataset_15d_train.mat", "dataset_15d_val.mat", "dataset_15d_test.mat", ("tr", "va", "te"), "c3a_bilstm.pt"),
    "c3": ModelSpec("c3", "dataset_15d_train.mat", "dataset_15d_val.mat", "dataset_15d_test.mat", ("tr", "va", "te"), "c3_bidir_attn.pt"),
}


def build_model(model_key: str, input_dim: int) -> nn.Module:
    if model_key == "c1":
        return LastStepLstmRegressor(input_dim=input_dim, hidden_dim=128)
    if model_key == "c2":
        return LastStepLstmRegressor(input_dim=input_dim, hidden_dim=192)
    if model_key == "c3a":
        return StackedBiLstmRegressor(input_dim=input_dim)
    if model_key == "c3":
        return BiLstmDaRegressor(input_dim=input_dim)
    raise KeyError(f"Unknown model key: {model_key}")


def build_dataloaders(data_dir: Path, spec: ModelSpec, batch_size: int, num_workers: int) -> Tuple[WindowDataset, WindowDataset, WindowDataset, DataLoader, DataLoader, DataLoader]:
    tr_name, va_name, te_name = spec.split_names
    train_ds = WindowDataset(data_dir / spec.train_file, tr_name)
    val_ds = WindowDataset(data_dir / spec.val_file, va_name)
    test_ds = WindowDataset(data_dir / spec.test_file, te_name)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    loss_sum = 0.0
    mae_sum = 0.0
    count = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        iterator: Iterable[Tuple[torch.Tensor, torch.Tensor]] = loader
        for xb, yb in iterator:
            xb = xb.to(device)
            yb = yb.to(device)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            pred = model(xb)
            loss = criterion(pred, yb)

            if train_mode:
                loss.backward()
                optimizer.step()

            batch_size = int(xb.shape[0])
            loss_sum += float(loss.item()) * batch_size
            mae_sum += float(torch.mean(torch.abs(pred - yb)).item()) * batch_size
            count += batch_size

    return {
        "loss": loss_sum / max(count, 1),
        "mae": mae_sum / max(count, 1),
    }


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu())
            targets.append(yb)

    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = math.sqrt(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse}


def train_one_model(args: argparse.Namespace, model_key: str) -> Dict[str, object]:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = MODEL_SPECS[model_key]
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=data_dir,
        spec=spec,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = build_model(model_key, input_dim=train_ds.input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    epoch_iter = range(1, args.epochs + 1)
    if not args.no_tqdm:
        epoch_iter = tqdm(epoch_iter, desc=f"train:{model_key}")

    start_time = time.time()
    for epoch in epoch_iter:
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
        }
        history.append(record)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.max_train_samples and epoch == 1 and len(train_ds) > args.max_train_samples:
            pass

        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"No best state captured for {model_key}")

    model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, test_loader, device)
    elapsed = time.time() - start_time

    checkpoint_path = out_dir / spec.checkpoint_name
    history_path = checkpoint_path.with_suffix(".history.json")
    summary_path = checkpoint_path.with_suffix(".summary.json")

    manifest_path = data_dir / "dataset_build_manifest.json"
    summary = {
        "model_key": model_key,
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "elapsed_sec": elapsed,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "window": train_ds.window,
        "input_dim": train_ds.input_dim,
        "seed": args.seed,
        "optimizer": {
            "name": "adam",
            "lr": args.lr,
            "scheduler": {"type": "StepLR", "step_size": 10, "gamma": 0.9},
            "batch_size": args.batch_size,
            "patience": args.patience,
        },
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
    }

    torch.save(
        {
            "model_key": model_key,
            "state_dict": model.state_dict(),
            "summary": summary,
            "history": history,
        },
        checkpoint_path,
    )
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/trained_models/python")
    parser.add_argument("--models", nargs="+", default=["c1", "c2", "c3a", "c3"], choices=sorted(MODEL_SPECS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-tqdm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    all_summaries = []
    for model_key in args.models:
        summary = train_one_model(args, model_key)
        all_summaries.append(summary)
        print(json.dumps(summary, indent=2))

    out_dir = Path(args.out_dir)
    manifest = {
        "generated_at_unix": time.time(),
        "models": all_summaries,
    }
    (out_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
