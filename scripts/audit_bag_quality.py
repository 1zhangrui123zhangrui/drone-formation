#!/usr/bin/env python3
"""
Audit rosbag quality against strict paper-grade acceptance criteria.

The goal is academic integrity and high data quality:
- no fabricated recovery
- no silently accepting badly exploded trajectories
- no treating heavily filtered partial-drone data as final paper data
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/opt/ros/noetic/lib/python3/dist-packages")
import rosbag  # type: ignore


SCENE_RULES = {
    "scene01_hover": {"min_duration_s": 55, "max_abs_pos_m": 20.0},
    "scene02_circle": {"min_duration_s": 85, "max_abs_pos_m": 20.0},
    "scene02p_lemni": {"min_duration_s": 85, "max_abs_pos_m": 20.0},
    "scene03_reconfig": {"min_duration_s": 110, "max_abs_pos_m": 25.0},
    "scene04_wind": {"min_duration_s": 85, "max_abs_pos_m": 30.0},
    "scene05_longtime": {"min_duration_s": 170, "max_abs_pos_m": 35.0},
}


def infer_scene_key(path: Path) -> str:
    name = path.name
    if "scene01_hover" in name:
        return "scene01_hover"
    if "s2_circle" in name or "scene02_circle" in name:
        return "scene02_circle"
    if "scene02p_lemni" in name:
        return "scene02p_lemni"
    if "scene03_reconfig" in name:
        return "scene03_reconfig"
    if "scene04_wind" in name:
        return "scene04_wind"
    if "scene05_longtime" in name:
        return "scene05_longtime"
    return "unknown"


def load_bag_stats(path: Path, drone_ids=(1, 2, 3, 4)):
    raw = {
        i: {
            "t_odom": [],
            "pos": [],
            "vel": [],
            "t_ctrl": [],
            "p_des": [],
            "v_des": [],
            "u_teacher": [],
        }
        for i in drone_ids
    }

    topics = []
    for i in drone_ids:
        topics += [
            f"/drone{i}/ground_truth/state",
            f"/drone{i}/p_des",
            f"/drone{i}/v_des",
            f"/drone{i}/cmd_vel_teacher",
        ]

    with rosbag.Bag(str(path), "r") as bag:
        for topic, msg, t in bag.read_messages(topics=topics):
            did = int(topic.split("/")[1].replace("drone", ""))
            ts = t.to_sec()

            if "ground_truth" in topic:
                raw[did]["t_odom"].append(ts)
                raw[did]["pos"].append(
                    [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
                )
                raw[did]["vel"].append(
                    [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
                )
            elif topic.endswith("p_des"):
                raw[did]["t_ctrl"].append(ts)
                raw[did]["p_des"].append([msg.point.x, msg.point.y, msg.point.z])
            elif topic.endswith("v_des"):
                if len(raw[did]["v_des"]) < len(raw[did]["t_ctrl"]):
                    raw[did]["v_des"].append([msg.vector.x, msg.vector.y, msg.vector.z])
            elif "cmd_vel_teacher" in topic:
                raw[did]["u_teacher"].append(
                    [msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z]
                )

    return raw


def per_drone_audit(raw_stats):
    report = {}
    for did, r in raw_stats.items():
        t_odom = np.array(r["t_odom"])
        pos = np.array(r["pos"]) if r["pos"] else np.empty((0, 3))
        vel = np.array(r["vel"]) if r["vel"] else np.empty((0, 3))

        n_sync = min(len(r["t_ctrl"]), len(r["p_des"]), len(r["v_des"]), len(r["u_teacher"]))
        clean = 0
        clean_ratio = 0.0

        if n_sync >= 5 and len(t_odom) >= 2:
            t_ctrl = np.array(r["t_ctrl"][:n_sync])
            p_des = np.array(r["p_des"][:n_sync])
            v_des = np.array(r["v_des"][:n_sync])
            u_t = np.array(r["u_teacher"][:n_sync])
            p_actual = np.column_stack([np.interp(t_ctrl, t_odom, pos[:, k]) for k in range(3)])
            v_actual = np.column_stack([np.interp(t_ctrl, t_odom, vel[:, k]) for k in range(3)])
            X = np.hstack([p_des, v_des, p_actual, v_actual, u_t[:, :3]])
            Y = u_t
            mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
            clean = int(mask.sum())
            clean_ratio = clean / float(n_sync)

        finite_pos = bool(pos.size and np.isfinite(pos).all())
        max_abs_pos = float(np.max(np.abs(pos))) if pos.size else float("inf")
        duration_s = float(t_odom[-1] - t_odom[0]) if len(t_odom) >= 2 else 0.0

        report[did] = {
            "odom_msgs": int(len(t_odom)),
            "sync_samples": int(n_sync),
            "clean_samples": clean,
            "clean_ratio": clean_ratio,
            "duration_s": duration_s,
            "finite_pos": finite_pos,
            "max_abs_pos_m": max_abs_pos,
        }

    return report


def evaluate(path: Path):
    scene_key = infer_scene_key(path)
    rules = SCENE_RULES.get(scene_key, {"min_duration_s": 0, "max_abs_pos_m": 20.0})

    raw = load_bag_stats(path)
    drones = per_drone_audit(raw)

    failures = []
    for did, d in drones.items():
        if d["odom_msgs"] == 0:
            failures.append(f"drone{did}: no odom data")
            continue
        if not d["finite_pos"]:
            failures.append(f"drone{did}: non-finite position values")
        if d["duration_s"] < rules["min_duration_s"]:
            failures.append(
                f"drone{did}: duration {d['duration_s']:.1f}s < required {rules['min_duration_s']:.1f}s"
            )
        if d["max_abs_pos_m"] > rules["max_abs_pos_m"]:
            failures.append(
                f"drone{did}: max |pos| {d['max_abs_pos_m']:.2f}m > allowed {rules['max_abs_pos_m']:.2f}m"
            )
        if d["clean_samples"] == 0:
            failures.append(f"drone{did}: zero clean synchronized samples")
        elif d["clean_ratio"] < 0.95:
            failures.append(
                f"drone{did}: clean ratio {100*d['clean_ratio']:.1f}% < required 95.0%"
            )

    status = "PASS" if not failures else "FAIL"
    return {
        "bag": str(path),
        "scene": scene_key,
        "status": status,
        "rules": rules,
        "drones": drones,
        "failures": failures,
    }


def main():
    ap = argparse.ArgumentParser(description="Audit a rosbag for paper-grade quality")
    ap.add_argument("bag_path", help="Path to rosbag")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = ap.parse_args()

    path = Path(args.bag_path)
    result = evaluate(path)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(f"[audit] bag:    {result['bag']}")
    print(f"[audit] scene:  {result['scene']}")
    print(f"[audit] status: {result['status']}")
    print(
        f"[audit] rules:  min_duration={result['rules']['min_duration_s']}s  "
        f"max_abs_pos={result['rules']['max_abs_pos_m']}m  clean_ratio>=95%"
    )
    for did in sorted(result["drones"]):
        d = result["drones"][did]
        print(
            f"  drone{did}: duration={d['duration_s']:.1f}s  odom={d['odom_msgs']}  "
            f"sync={d['sync_samples']}  clean={d['clean_samples']}  "
            f"clean_ratio={100*d['clean_ratio']:.1f}%  max|pos|={d['max_abs_pos_m']:.2f}m"
        )
    if result["failures"]:
        print("[audit] failures:")
        for item in result["failures"]:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
