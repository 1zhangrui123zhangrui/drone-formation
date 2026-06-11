#!/usr/bin/env python3
"""
bag_to_mat.py — 从 rosbag 提取15维特征向量并保存为 .mat

用法:
    python3 scripts/bag_to_mat.py <bag_path> <out_mat_path> [--drones 1 2 3 4]

输出 .mat 结构 (MATLAB 加载后):
    data.X          [N_total, 15]  float32, 所有无人机所有时刻的特征
    data.Y          [N_total, 4]   float32, 对应的 Teacher cmd_vel 标签
    data.drone_id   [N_total, 1]   uint8,   来源无人机编号
    data.t_vec      [N_total, 1]   float64, 时间戳（从bag起始归零, s）

15维向量定义（论文§4.2）:
    x = [p_des(3), v_des(3), p_actual(3), v_actual(3), u_teacher_xyz(3)]
      = [col0-2:期望位置, col3-5:期望速度, col6-8:实际位置, col9-11:实际速度, col12-14:Teacher速度指令xyz]

标签 Y = [vx, vy, vz, wz]（Teacher cmd_vel 完整4维）

同步策略: 以 cmd_vel_teacher 时间戳（10Hz）为主时钟，对 odom（50Hz）做线性插值。
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')
import rosbag


def parse_bag(bag_path: str, drone_ids=(1, 2, 3, 4)):
    bag = rosbag.Bag(bag_path, 'r')

    raw = {i: {'t_odom': [], 'pos': [], 'vel': [],
               't_ctrl': [], 'p_des': [], 'v_des': [], 'u_teacher': []}
           for i in drone_ids}

    topics = []
    for i in drone_ids:
        topics += [f'/drone{i}/ground_truth/state',
                   f'/drone{i}/p_des',
                   f'/drone{i}/v_des',
                   f'/drone{i}/cmd_vel_teacher']

    # 单次遍历读取所有话题
    for topic, msg, t in bag.read_messages(topics=topics):
        parts = topic.split('/')
        did = int(parts[1].replace('drone', ''))
        if did not in drone_ids:
            continue
        ts = t.to_sec()

        if 'ground_truth' in topic:
            raw[did]['t_odom'].append(ts)
            raw[did]['pos'].append([msg.pose.pose.position.x,
                                    msg.pose.pose.position.y,
                                    msg.pose.pose.position.z])
            raw[did]['vel'].append([msg.twist.twist.linear.x,
                                    msg.twist.twist.linear.y,
                                    msg.twist.twist.linear.z])
        elif topic.endswith('p_des'):
            raw[did]['t_ctrl'].append(ts)
            raw[did]['p_des'].append([msg.point.x, msg.point.y, msg.point.z])
        elif topic.endswith('v_des'):
            # v_des 和 p_des 同频，只在 p_des 已到位时才追加（防止因话题乱序导致错位）
            if len(raw[did]['v_des']) < len(raw[did]['t_ctrl']):
                raw[did]['v_des'].append([msg.vector.x, msg.vector.y, msg.vector.z])
        elif 'cmd_vel_teacher' in topic:
            raw[did]['u_teacher'].append([msg.linear.x, msg.linear.y, msg.linear.z,
                                          msg.angular.z])
    bag.close()

    all_X, all_Y, all_drone, all_t = [], [], [], []
    t0_global = None

    for did in drone_ids:
        r = raw[did]
        # 取四路数据最短公共长度
        n = min(len(r['t_ctrl']), len(r['p_des']), len(r['v_des']), len(r['u_teacher']))
        if n < 5:
            print(f'  [WARN] drone{did}: only {n} ctrl samples, skipping')
            continue

        t_ctrl  = np.array(r['t_ctrl'][:n])
        p_des   = np.array(r['p_des'][:n])     # (n, 3)
        v_des   = np.array(r['v_des'][:n])     # (n, 3)
        u_t     = np.array(r['u_teacher'][:n]) # (n, 4)

        t_odom  = np.array(r['t_odom'])
        pos     = np.array(r['pos'])            # (M, 3)
        vel     = np.array(r['vel'])            # (M, 3)

        if len(t_odom) < 2:
            print(f'  [WARN] drone{did}: no odom data, skipping')
            continue

        # 将 odom（高频）插值到 ctrl 时钟（10Hz）
        p_actual = np.column_stack([
            np.interp(t_ctrl, t_odom, pos[:, k]) for k in range(3)])
        v_actual = np.column_stack([
            np.interp(t_ctrl, t_odom, vel[:, k]) for k in range(3)])

        # 构造 15 维特征向量: [p_des, v_des, p_actual, v_actual, u_teacher_xyz]
        X = np.hstack([p_des, v_des, p_actual, v_actual, u_t[:, :3]])  # (n, 15)
        Y = u_t  # (n, 4)  完整 Teacher 指令作为标签

        # 过滤 NaN/Inf（通常由 NaN bug 引入）
        mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        X, Y = X[mask], Y[mask]
        t_ctrl_clean = t_ctrl[mask]

        if len(X) < 5:
            print(f'  [WARN] drone{did}: too few clean samples ({len(X)}), skipping')
            continue

        if t0_global is None:
            t0_global = t_ctrl_clean[0]

        all_X.append(X.astype(np.float32))
        all_Y.append(Y.astype(np.float32))
        all_drone.append(np.full(len(X), did, dtype=np.uint8))
        all_t.append((t_ctrl_clean - t0_global).astype(np.float64))
        print(f'  drone{did}: {len(X)} clean samples')

    if not all_X:
        raise RuntimeError(f'No valid data extracted from {bag_path}')

    return {
        'X':        np.vstack(all_X),
        'Y':        np.vstack(all_Y),
        'drone_id': np.concatenate(all_drone).reshape(-1, 1),
        't_vec':    np.concatenate(all_t).reshape(-1, 1),
    }


def main():
    ap = argparse.ArgumentParser(description='Convert rosbag to .mat for MATLAB training')
    ap.add_argument('bag_path',    help='Path to .bag file')
    ap.add_argument('out_mat_path', help='Output .mat path')
    ap.add_argument('--drones', nargs='+', type=int, default=[1, 2, 3, 4],
                    help='Drone IDs to include (default: 1 2 3 4)')
    args = ap.parse_args()

    print(f'[bag_to_mat] reading: {args.bag_path}')
    print(f'[bag_to_mat] drones: {args.drones}')

    data = parse_bag(args.bag_path, tuple(args.drones))
    print(f'[bag_to_mat] total X: {data["X"].shape}  Y: {data["Y"].shape}')
    print(f'[bag_to_mat] NaN in X: {~np.isfinite(data["X"]).all()}  Y: {~np.isfinite(data["Y"]).all()}')

    out = Path(args.out_mat_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    scipy.io.savemat(
        str(out),
        {'data': data, 'bag_name': str(Path(args.bag_path).name)},
        do_compression=True
    )
    print(f'[bag_to_mat] saved → {out}  ({out.stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
