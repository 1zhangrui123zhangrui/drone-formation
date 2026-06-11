#!/usr/bin/env python3
"""诊断 bag 文件实际数据范围"""
import rosbag
import numpy as np
from pathlib import Path

BAG_DIR = Path.home() / 'drone-formation-e2e' / 'data' / 'raw_bags'
BAGS = [
    's2_circle_4drones_seed42.bag',       # 之前同步采样完美的那份
    'scene01_hover_4drones_seed42.bag',   # 图正常的
    'scene04_wind_4drones_seed42.bag',    # 图爆炸的
    'scene05_longtime_4drones_seed42.bag',# 图爆炸的
]

for bag_file in BAGS:
    bag_path = BAG_DIR / bag_file
    if not bag_path.exists():
        print(f'MISSING: {bag_file}\n')
        continue

    print(f'\n{"="*70}')
    print(f'  {bag_file}')
    print(f'{"="*70}')

    bag = rosbag.Bag(str(bag_path))
    for i in range(1, 5):
        positions = []
        for _, msg, t in bag.read_messages(topics=[f'/drone{i}/ground_truth/state']):
            positions.append([msg.pose.pose.position.x,
                              msg.pose.pose.position.y,
                              msg.pose.pose.position.z])
        if not positions:
            print(f'  drone{i}: NO DATA')
            continue
        p = np.array(positions)
        print(f'  drone{i}: {len(p)} msgs')
        print(f'    first 3: ', end='')
        for k in range(min(3, len(p))):
            print(f'({p[k,0]:+7.2f},{p[k,1]:+7.2f},{p[k,2]:+6.2f})', end=' ')
        print()
        print(f'    last 3:  ', end='')
        for k in range(max(0,len(p)-3), len(p)):
            print(f'({p[k,0]:+7.2f},{p[k,1]:+7.2f},{p[k,2]:+6.2f})', end=' ')
        print()
        print(f'    range x:[{p[:,0].min():+8.1f}, {p[:,0].max():+8.1f}]  '
              f'y:[{p[:,1].min():+8.1f}, {p[:,1].max():+8.1f}]  '
              f'z:[{p[:,2].min():+6.2f}, {p[:,2].max():+6.2f}]')
    bag.close()