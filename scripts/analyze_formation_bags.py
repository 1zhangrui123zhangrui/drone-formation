#!/usr/bin/env python3
"""Paper-grade offline trajectory and formation analysis for 4-drone bags."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import rosbag  # type: ignore


DRONE_IDS = (1, 2, 3, 4)
DRONE_COLORS = ["#D62728", "#1F77B4", "#2CA02C", "#FF7F0E"]
DRONE_STYLES = ["-", "--", "-.", ":"]
DRONE_NAMES = ["D1", "D2", "D3", "D4"]
REF_COLOR = "#303030"
PAIR_IDS = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.1,
        "grid.linewidth": 0.35,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


@dataclass(frozen=True)
class SceneSpec:
    key: str
    label: str
    bag_name: str
    drift_line: bool = False


SCENES_5 = [
    SceneSpec("s1_hover", "S1 Hover", "scene01_hover_4drones_v2.bag"),
    SceneSpec("s2_circle", "S2 Circle", "scene02_circle_4drones_v2_worldfix_entryfix.bag"),
    SceneSpec("s3_reconfig", "S3 Reconfiguration", "scene03_reconfig_4drones_v2.bag", True),
    SceneSpec("s4_wind", "S4 Wind Disturbance", "scene04_wind_4drones_v2.bag", True),
    SceneSpec("s5_longtime", "S5 Long-Duration", "scene05_longtime_4drones_v2.bag", True),
]


def read_bag(path: Path) -> dict:
    data = {
        i: {"t_odom": [], "pos": [], "t_pdes": [], "pdes": []}
        for i in DRONE_IDS
    }
    topics = []
    for i in DRONE_IDS:
        topics.extend([f"/drone{i}/ground_truth/state", f"/drone{i}/p_des"])

    with rosbag.Bag(str(path), "r") as bag:
        for topic, msg, stamp in bag.read_messages(topics=topics):
            drone_id = int(topic.split("/")[1].replace("drone", ""))
            if topic.endswith("/ground_truth/state"):
                data[drone_id]["t_odom"].append(stamp.to_sec())
                data[drone_id]["pos"].append(
                    [
                        msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z,
                    ]
                )
            elif topic.endswith("/p_des"):
                data[drone_id]["t_pdes"].append(stamp.to_sec())
                data[drone_id]["pdes"].append([msg.point.x, msg.point.y, msg.point.z])

    all_times = [
        t
        for i in DRONE_IDS
        for t in (data[i]["t_odom"] + data[i]["t_pdes"])
    ]
    if not all_times:
        raise RuntimeError(f"No trajectory messages in {path}")
    t0 = min(all_times)

    for i in DRONE_IDS:
        for t_key, x_key in [("t_odom", "pos"), ("t_pdes", "pdes")]:
            t = np.asarray(data[i][t_key], dtype=float) - t0
            x = np.asarray(data[i][x_key], dtype=float)
            order = np.argsort(t)
            data[i][t_key] = t[order]
            data[i][x_key] = x[order] if x.size else x.reshape(0, 3)
    return data


def interp_vec(t_query: np.ndarray, t_src: np.ndarray, x_src: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(t_query, t_src, x_src[:, k]) for k in range(3)])


def synchronized_scene(data: dict) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    starts = []
    ends = []
    for i in DRONE_IDS:
        starts.extend([data[i]["t_odom"][0], data[i]["t_pdes"][0]])
        ends.extend([data[i]["t_odom"][-1], data[i]["t_pdes"][-1]])
    t0 = max(starts)
    t1 = min(ends)
    t = np.arange(t0, t1, 0.1)
    pos = {i: interp_vec(t, data[i]["t_odom"], data[i]["pos"]) for i in DRONE_IDS}
    pdes = {i: interp_vec(t, data[i]["t_pdes"], data[i]["pdes"]) for i in DRONE_IDS}
    return t, pos, pdes


def scene_metrics(t: np.ndarray, pos: dict, pdes: dict) -> dict:
    drone_err = {}
    all_err = []
    for i in DRONE_IDS:
        err = np.linalg.norm(pos[i] - pdes[i], axis=1)
        drone_err[i] = err
        all_err.append(err)

    pair_actual = []
    pair_des = []
    pair_err = []
    min_actual = np.inf
    for a, b in PAIR_IDS:
        da = np.linalg.norm(pos[a] - pos[b], axis=1)
        dd = np.linalg.norm(pdes[a] - pdes[b], axis=1)
        pair_actual.append(da)
        pair_des.append(dd)
        pair_err.append(np.abs(da - dd))
        min_actual = min(min_actual, float(np.min(da)))

    err_all = np.concatenate(all_err)
    pair_err_all = np.concatenate(pair_err)
    return {
        "duration_s": float(t[-1] - t[0]),
        "track_rmse_m": float(np.sqrt(np.mean(err_all**2))),
        "track_mae_m": float(np.mean(err_all)),
        "track_p95_m": float(np.percentile(err_all, 95)),
        "track_max_m": float(np.max(err_all)),
        "formation_rmse_m": float(np.sqrt(np.mean(pair_err_all**2))),
        "formation_mae_m": float(np.mean(pair_err_all)),
        "formation_p95_m": float(np.percentile(pair_err_all, 95)),
        "formation_max_m": float(np.max(pair_err_all)),
        "min_pair_distance_m": float(min_actual),
        "drone_err": drone_err,
        "pair_actual": pair_actual,
        "pair_des": pair_des,
        "pair_err": pair_err,
    }


def setup_axes(ax):
    ax.grid(True, linestyle=":", alpha=0.45)
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def savefig(fig, path: Path):
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_xy(scene: SceneSpec, t: np.ndarray, pos: dict, pdes: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(3.35, 3.1))
    for i in DRONE_IDS:
        ax.plot(
            pdes[i][:, 0],
            pdes[i][:, 1],
            color=DRONE_COLORS[i - 1],
            linestyle=":",
            linewidth=0.9,
            alpha=0.55,
        )
        ax.plot(
            pos[i][:, 0],
            pos[i][:, 1],
            color=DRONE_COLORS[i - 1],
            linestyle=DRONE_STYLES[i - 1],
            label=DRONE_NAMES[i - 1],
        )
        ax.scatter(pos[i][0, 0], pos[i][0, 1], marker="o", s=22, color=DRONE_COLORS[i - 1], edgecolor="black", linewidth=0.45, zorder=5)
        ax.scatter(pos[i][-1, 0], pos[i][-1, 1], marker="s", s=22, color=DRONE_COLORS[i - 1], edgecolor="black", linewidth=0.45, zorder=5)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_title(scene.label)
    ax.set_aspect("equal", adjustable="box")
    setup_axes(ax)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", loc="best", ncol=2)
    savefig(fig, out_dir / f"{scene.key}_xy")


def plot_error(scene: SceneSpec, t: np.ndarray, metrics: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    for i in DRONE_IDS:
        ax.plot(t, metrics["drone_err"][i], color=DRONE_COLORS[i - 1], linestyle=DRONE_STYLES[i - 1], label=DRONE_NAMES[i - 1])
    if scene.drift_line and t[0] <= 60 <= t[-1]:
        ax.axvline(60, color="black", linestyle="--", linewidth=0.8)
        ax.text(60.7, ax.get_ylim()[1] * 0.92, "60 s", fontsize=7, va="top")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\|p-p_{\rm des}\|_2$ (m)")
    ax.set_title(f"{scene.label}: tracking error")
    setup_axes(ax)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", ncol=2)
    savefig(fig, out_dir / f"{scene.key}_tracking_error")


def plot_formation_error(scene: SceneSpec, t: np.ndarray, metrics: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    mean_pair_err = np.mean(np.vstack(metrics["pair_err"]), axis=0)
    max_pair_err = np.max(np.vstack(metrics["pair_err"]), axis=0)
    ax.plot(t, mean_pair_err, color="#1F77B4", label="Mean pair error")
    ax.plot(t, max_pair_err, color="#D62728", linestyle="--", label="Max pair error")
    if scene.drift_line and t[0] <= 60 <= t[-1]:
        ax.axvline(60, color="black", linestyle="--", linewidth=0.8)
        ax.text(60.7, ax.get_ylim()[1] * 0.92, "60 s", fontsize=7, va="top")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance error (m)")
    ax.set_title(f"{scene.label}: formation keeping")
    setup_axes(ax)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, fancybox=False, edgecolor="black")
    savefig(fig, out_dir / f"{scene.key}_formation_error")


def plot_s3_switches(t: np.ndarray, metrics: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    mean_des = np.mean(np.vstack(metrics["pair_des"]), axis=0)
    mean_actual = np.mean(np.vstack(metrics["pair_actual"]), axis=0)
    ax.plot(t, mean_des, color="black", linestyle=":", label="Desired mean pair distance")
    ax.plot(t, mean_actual, color="#1F77B4", label="Actual mean pair distance")
    for ts in (40, 80):
        if t[0] <= ts <= t[-1]:
            ax.axvline(ts, color="#D62728", linestyle="--", linewidth=0.8)
            ax.text(ts + 0.7, ax.get_ylim()[1] * 0.92, f"{ts} s", fontsize=7, va="top")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean pair distance (m)")
    ax.set_title("S3 formation switching evidence")
    setup_axes(ax)
    ax.set_xlim(t[0], t[-1])
    ax.legend(frameon=True, fancybox=False, edgecolor="black")
    savefig(fig, out_dir / "s3_switch_pair_distance")


def plot_s4_vs_s2(s2_t: np.ndarray, s2_m: dict, s4_t: np.ndarray, s4_m: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    s2_mean = np.mean(np.vstack([s2_m["drone_err"][i] for i in DRONE_IDS]), axis=0)
    s4_mean = np.mean(np.vstack([s4_m["drone_err"][i] for i in DRONE_IDS]), axis=0)
    ax.plot(s2_t, s2_mean, color="#1F77B4", label="S2 no wind")
    ax.plot(s4_t, s4_mean, color="#D62728", linestyle="--", label="S4 wind")
    if s4_t[0] <= 60 <= s4_t[-1]:
        ax.axvline(60, color="black", linestyle="--", linewidth=0.8)
        ax.text(60.7, ax.get_ylim()[1] * 0.92, "60 s", fontsize=7, va="top")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean tracking error (m)")
    ax.set_title("Wind disturbance effect")
    setup_axes(ax)
    ax.set_xlim(0, min(s2_t[-1], s4_t[-1]))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, fancybox=False, edgecolor="black")
    savefig(fig, out_dir / "s4_wind_vs_s2_tracking_error")


def write_summary(rows: list[dict], path: Path):
    fields = [
        "scene",
        "duration_s",
        "track_rmse_m",
        "track_mae_m",
        "track_p95_m",
        "track_max_m",
        "formation_rmse_m",
        "formation_mae_m",
        "formation_p95_m",
        "formation_max_m",
        "min_pair_distance_m",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def phase_mean(metrics: dict, t: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    mask = (t >= lo) & (t < hi)
    desired = float(np.mean(np.mean(np.vstack(metrics["pair_des"])[:, mask], axis=0)))
    actual = float(np.mean(np.mean(np.vstack(metrics["pair_actual"])[:, mask], axis=0)))
    return desired, actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-dir", default="data/raw_bags/v2")
    parser.add_argument("--out-dir", default="results/formation_analysis")
    args = parser.parse_args()

    bag_dir = Path(args.bag_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summary_rows = []
    for scene in SCENES_5:
        path = bag_dir / scene.bag_name
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[analyze] {scene.label}: {path}")
        data = read_bag(path)
        t, pos, pdes = synchronized_scene(data)
        metrics = scene_metrics(t, pos, pdes)
        all_results[scene.key] = (scene, t, pos, pdes, metrics)
        row = {"scene": scene.key, **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}}
        summary_rows.append(row)
        plot_xy(scene, t, pos, pdes, fig_dir)
        plot_error(scene, t, metrics, fig_dir)
        plot_formation_error(scene, t, metrics, fig_dir)

    plot_s3_switches(all_results["s3_reconfig"][1], all_results["s3_reconfig"][4], fig_dir)
    plot_s4_vs_s2(
        all_results["s2_circle"][1],
        all_results["s2_circle"][4],
        all_results["s4_wind"][1],
        all_results["s4_wind"][4],
        fig_dir,
    )
    write_summary(summary_rows, out_dir / "formation_summary.csv")

    s3_t, s3_m = all_results["s3_reconfig"][1], all_results["s3_reconfig"][4]
    print("\nS3 mean pair distance by phase, desired -> actual:")
    for lo, hi, name in [(15, 39, "rectangle"), (45, 79, "diamond"), (85, 115, "triangle+center")]:
        des, act = phase_mean(s3_m, s3_t, lo, hi)
        print(f"  {name:15s}: {des:.3f} -> {act:.3f} m")

    s2_rmse = all_results["s2_circle"][4]["track_rmse_m"]
    s4_rmse = all_results["s4_wind"][4]["track_rmse_m"]
    print(f"\nS4 wind effect: S2 RMSE={s2_rmse:.3f} m, S4 RMSE={s4_rmse:.3f} m, ratio={s4_rmse / s2_rmse:.2f}x")

    print(f"\nSaved summary: {out_dir / 'formation_summary.csv'}")
    print(f"Saved figures: {fig_dir}")


if __name__ == "__main__":
    main()
