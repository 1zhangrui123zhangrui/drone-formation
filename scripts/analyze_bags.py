#!/usr/bin/env python3
"""
analyze_bags.py - IEEE 期刊风格的 4 机轨迹可视化
读取 5 个场景的 rosbag,生成 XY 轨迹图 + 跟踪误差时序图 + 多场景对比拼图

输出: ~/drone-formation-e2e/results/figures/
  - {scene}_xy.{png,pdf}     # 4 机 XY 轨迹
  - {scene}_error.{png,pdf}  # 4 机跟踪误差时序
  - overview_xy.{png,pdf}    # 6 场景拼图
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import rosbag

# ============ IEEE 期刊图风格 ============
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.4,
    'patch.linewidth': 0.5,
    'grid.linewidth': 0.4,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# 高对比度 4 色板 (色盲友好,印刷友好)
DRONE_COLORS = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E']  # 红 蓝 绿 橙
DRONE_NAMES = ['Drone 1', 'Drone 2', 'Drone 3', 'Drone 4']
REF_COLOR = '#7F7F7F'

BAG_DIR = Path.home() / 'drone-formation-e2e' / 'data' / 'raw_bags'
OUT_DIR = Path.home() / 'drone-formation-e2e' / 'results' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============ bag 解析 ============

def parse_bag(bag_path):
    """读取 4 机的 ground_truth 和 p_des,返回时间归零的字典"""
    print(f'  parsing {bag_path.name}...')
    bag = rosbag.Bag(str(bag_path), 'r')

    data = {i: {'t_odom': [], 'pos': [], 't_pdes': [], 'pdes': []}
            for i in range(1, 5)}

    topics_wanted = []
    for i in range(1, 5):
        topics_wanted.append(f'/drone{i}/ground_truth/state')
        topics_wanted.append(f'/drone{i}/p_des')

    for topic, msg, t in bag.read_messages(topics=topics_wanted):
        parts = topic.split('/')
        drone_id = int(parts[1].replace('drone', ''))
        if 'ground_truth' in topic:
            data[drone_id]['t_odom'].append(t.to_sec())
            data[drone_id]['pos'].append([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z])
        elif 'p_des' in topic:
            data[drone_id]['t_pdes'].append(t.to_sec())
            data[drone_id]['pdes'].append([
                msg.point.x, msg.point.y, msg.point.z])
    bag.close()

    # 归一化时间
    all_times = []
    for i in range(1, 5):
        all_times.extend(data[i]['t_odom'])
    t0 = min(all_times) if all_times else 0.0

    for i in range(1, 5):
        data[i]['t_odom'] = np.array(data[i]['t_odom']) - t0
        data[i]['pos'] = np.array(data[i]['pos'])
        data[i]['t_pdes'] = np.array(data[i]['t_pdes']) - t0
        data[i]['pdes'] = np.array(data[i]['pdes'])

    return data


def compute_tracking_error(data):
    """计算 ||p_actual - p_des|| 时序"""
    errors = {}
    for i in range(1, 5):
        if len(data[i]['t_pdes']) < 2 or len(data[i]['t_odom']) == 0:
            errors[i] = (np.array([]), np.array([]))
            continue
        t_o = data[i]['t_odom']
        t_p = data[i]['t_pdes']
        pdes = data[i]['pdes']
        pos = data[i]['pos']
        p_des_interp = np.column_stack([
            np.interp(t_o, t_p, pdes[:, 0]),
            np.interp(t_o, t_p, pdes[:, 1]),
            np.interp(t_o, t_p, pdes[:, 2])])
        err = np.linalg.norm(pos - p_des_interp, axis=1)
        errors[i] = (t_o, err)
    return errors


# ============ 参考轨迹 ============

def generate_reference(scene_key, T_max):
    """生成场景中心轨迹(虚线参考)"""
    R, omega = 3.0, 0.4
    if scene_key == 'scene01_hover':
        return None
    elif scene_key in ('s2_circle', 'scene05_longtime',
                       'scene03_reconfig', 'scene04_wind'):
        t = np.linspace(0, min(T_max, 60), 600)
        return np.column_stack([R * np.cos(omega * t), R * np.sin(omega * t)])
    elif scene_key == 'scene02p_lemni':
        t = np.linspace(0, min(T_max, 90), 600)
        return np.column_stack([R * np.sin(omega * t),
                                R / 2 * np.sin(2 * omega * t)])
    return None


# ============ 图 1: XY 轨迹 ============

def plot_xy(data, scene_label, scene_key, out_path):
    fig, ax = plt.subplots(figsize=(3.6, 3.4))

    T_max = max([data[i]['t_odom'][-1] if len(data[i]['t_odom']) > 0 else 0
                 for i in range(1, 5)])
    ref = generate_reference(scene_key, T_max)

    # 参考轨迹(灰虚线)
    if ref is not None:
        ax.plot(ref[:, 0], ref[:, 1], color=REF_COLOR, linestyle='--',
                linewidth=1.0, alpha=0.7, label='Reference', zorder=1)

    # 4 机
    for i in range(1, 5):
        pos = data[i]['pos']
        if len(pos) == 0:
            continue
        ax.plot(pos[:, 0], pos[:, 1], color=DRONE_COLORS[i-1],
                linewidth=1.4, label=DRONE_NAMES[i-1], alpha=0.88, zorder=3)
        ax.scatter(pos[0, 0], pos[0, 1], color=DRONE_COLORS[i-1],
                   marker='o', s=30, edgecolor='black', linewidth=0.6,
                   zorder=5, alpha=0.95)
        ax.scatter(pos[-1, 0], pos[-1, 1], color=DRONE_COLORS[i-1],
                   marker='s', s=30, edgecolor='black', linewidth=0.6,
                   zorder=5, alpha=0.95)

    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')
    ax.set_title(scene_label, pad=6)
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle=':', alpha=0.45)
    ax.legend(loc='best', frameon=True, framealpha=0.92,
              edgecolor='black', fancybox=False, borderpad=0.4,
              handlelength=1.6, handletextpad=0.5)

    for ext in ['png', 'pdf']:
        plt.savefig(out_path.with_suffix(f'.{ext}'),
                    dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f'    -> {out_path.name}.{{png,pdf}}')


# ============ 图 2: 跟踪误差时序 ============

def plot_error(errors, scene_label, out_path):
    fig, ax = plt.subplots(figsize=(4.2, 2.6))

    valid_t_max = 0
    valid_e_max = 0
    for i in range(1, 5):
        t, err = errors[i]
        if len(t) == 0:
            continue
        ax.plot(t, err, color=DRONE_COLORS[i-1], linewidth=1.3,
                label=DRONE_NAMES[i-1], alpha=0.88)
        valid_t_max = max(valid_t_max, t[-1])
        valid_e_max = max(valid_e_max, err.max())

    # 60s 漂移检查点
    if valid_t_max > 60:
        ax.axvline(60, color='black', linestyle='--', linewidth=0.7, alpha=0.5)
        y_top = valid_e_max * 1.05 if valid_e_max > 0 else 1
        ax.text(60, y_top, ' 60s drift check', fontsize=6.5,
                ha='left', va='top', alpha=0.65)

    ax.set_xlabel(r'Time (s)')
    ax.set_ylabel(r'$\|\mathbf{p}-\mathbf{p}_{\mathrm{des}}\|_2$ (m)')
    ax.set_title(f'{scene_label}: 3D Tracking Error', pad=6)
    ax.grid(True, linestyle=':', alpha=0.45)
    ax.legend(loc='best', frameon=True, framealpha=0.92, ncol=2,
              edgecolor='black', fancybox=False, borderpad=0.4,
              handlelength=1.6, handletextpad=0.5)
    if valid_t_max > 0:
        ax.set_xlim(0, valid_t_max * 1.02)
    ax.set_ylim(0, None)

    for ext in ['png', 'pdf']:
        plt.savefig(out_path.with_suffix(f'.{ext}'),
                    dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f'    -> {out_path.name}.{{png,pdf}}')


# ============ 图 3: 6 场景拼图 ============

def plot_overview(all_data, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.6))

    for ax, (scene_key, scene_label, data) in zip(axes.flat, all_data):
        T_max = max([data[i]['t_odom'][-1] if len(data[i]['t_odom']) > 0 else 0
                     for i in range(1, 5)])
        ref = generate_reference(scene_key, T_max)
        if ref is not None:
            ax.plot(ref[:, 0], ref[:, 1], color=REF_COLOR, linestyle='--',
                    linewidth=0.8, alpha=0.6)
        for i in range(1, 5):
            pos = data[i]['pos']
            if len(pos) > 0:
                ax.plot(pos[:, 0], pos[:, 1], color=DRONE_COLORS[i-1],
                        linewidth=1.0, alpha=0.85)
                ax.scatter(pos[0, 0], pos[0, 1], color=DRONE_COLORS[i-1],
                           marker='o', s=15, edgecolor='black',
                           linewidth=0.4, zorder=5)

        ax.set_title(scene_label, fontsize=9, pad=4)
        ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle=':', alpha=0.45)
        ax.set_xlabel(r'$x$ (m)', fontsize=8)
        ax.set_ylabel(r'$y$ (m)', fontsize=8)
        ax.tick_params(labelsize=7)

    # 共享 legend 在顶部
    handles = [Line2D([0], [0], color=DRONE_COLORS[i], linewidth=1.6,
                      label=DRONE_NAMES[i]) for i in range(4)]
    handles.append(Line2D([0], [0], color=REF_COLOR, linewidth=1.0,
                          linestyle='--', label='Reference'))
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False,
               fontsize=8.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.35, wspace=0.30)

    for ext in ['png', 'pdf']:
        plt.savefig(out_path.with_suffix(f'.{ext}'),
                    dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f'    -> {out_path.name}.{{png,pdf}}')


# ============ 主程序 ============

SCENES = [
    ('scene01_hover',     'S1: Static Hover'),
    ('s2_circle',         'S2: Circular Cruise'),
    ('scene02p_lemni',    'S2$^\\prime$: Lemniscate'),
    ('scene03_reconfig',  'S3: Formation Reconfig.'),
    ('scene04_wind',      'S4: Wind Disturbance'),
    ('scene05_longtime',  'S5: Long-Duration Flight'),
]

if __name__ == '__main__':
    print(f'Output directory: {OUT_DIR}\n')

    all_data_for_overview = []
    for scene_key, scene_label in SCENES:
        if scene_key == 's2_circle':
            bag_path = BAG_DIR / 's2_circle_4drones_seed42.bag'
        else:
            bag_path = BAG_DIR / f'{scene_key}_4drones_seed42.bag'

        if not bag_path.exists():
            print(f'[WARN] missing: {bag_path.name}, skip\n')
            continue

        print(f'== {scene_label} ==')
        data = parse_bag(bag_path)
        errors = compute_tracking_error(data)
        plot_xy(data, scene_label, scene_key, OUT_DIR / f'{scene_key}_xy')
        plot_error(errors, scene_label, OUT_DIR / f'{scene_key}_error')
        all_data_for_overview.append((scene_key, scene_label, data))
        print()

    if len(all_data_for_overview) >= 2:
        print('== 6-scene overview ==')
        plot_overview(all_data_for_overview, OUT_DIR / 'overview_xy')

    print(f'\n[DONE] all figures in {OUT_DIR}')
    print('\nFile listing:')
    os.system(f'ls -lh {OUT_DIR}/*.png {OUT_DIR}/*.pdf 2>/dev/null')