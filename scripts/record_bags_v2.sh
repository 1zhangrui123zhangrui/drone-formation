#!/usr/bin/env bash
# record_bags_v2.sh — 重录所有场景 bag（NaN修复版，omega=0.2）
#
# 使用方法：
#   bash scripts/record_bags_v2.sh [SCENE]
#   SCENE: all | s1 | s2 | s2p | s3 | s4 | s5（默认 all）
#
# 每个场景步骤：
#   1. roslaunch 启动仿真
#   2. 等待 12s（让 spawn + 电机使能完成）
#   3. rosbag record 开始（后台）
#   4. 等待场景时长后停止录包和仿真
#
# 输出目录：~/drone-formation-e2e/data/raw_bags/v2/

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAG_DIR="$ROOT_DIR/data/raw_bags/v2"
mkdir -p "$BAG_DIR"

# 所有 Topics
TOPICS=(
  /drone1/ground_truth/state /drone2/ground_truth/state
  /drone3/ground_truth/state /drone4/ground_truth/state
  /drone1/p_des /drone2/p_des /drone3/p_des /drone4/p_des
  /drone1/v_des /drone2/v_des /drone3/v_des /drone4/v_des
  /drone1/cmd_vel_teacher /drone2/cmd_vel_teacher
  /drone3/cmd_vel_teacher /drone4/cmd_vel_teacher
  /drone1/cmd_vel /drone2/cmd_vel /drone3/cmd_vel /drone4/cmd_vel
)

SCENE="${1:-all}"

run_scene() {
  local NAME="$1"
  local LAUNCH_PKG="drone_sim"
  local LAUNCH_FILE="$2"
  local DURATION="$3"       # 录包时长(s)
  local BAG_NAME="$4"
  local WAIT_BEFORE="${5:-12}"  # launch 后等待秒数

  echo ""
  echo "========================================"
  echo "  SCENE: $NAME  (duration=${DURATION}s)"
  echo "========================================"

  local BAG_PATH="$BAG_DIR/${BAG_NAME}.bag"

  # 启动仿真（后台）
  source "$ROOT_DIR/ros_ws/devel/setup.bash"
  roslaunch "$LAUNCH_PKG" "$LAUNCH_FILE" gui:=false &
  LAUNCH_PID=$!
  echo "[record] launch PID=$LAUNCH_PID, waiting ${WAIT_BEFORE}s..."
  sleep "$WAIT_BEFORE"

  # 开始录包（后台）
  echo "[record] starting rosbag record -> $BAG_PATH"
  rosbag record -O "$BAG_PATH" "${TOPICS[@]}" &
  BAG_PID=$!

  # 等待录包时长
  echo "[record] recording for ${DURATION}s..."
  sleep "$DURATION"

  # 停止录包
  kill "$BAG_PID" 2>/dev/null || true
  sleep 1

  # 停止仿真
  kill "$LAUNCH_PID" 2>/dev/null || true
  rosnode kill -a 2>/dev/null || true
  sleep 3

  # 快速诊断
  echo "[record] diagnosing $BAG_NAME..."
  python3 - <<EOF
import sys
sys.path.insert(0,'/opt/ros/noetic/lib/python3/dist-packages')
import rosbag, numpy as np
from pathlib import Path
bp = Path('$BAG_PATH')
if not bp.exists():
    print('  ERROR: bag not found at $BAG_PATH'); sys.exit(1)
bag = rosbag.Bag(str(bp))
positions = []
times = []
for _, msg, t in bag.read_messages(topics=['/drone1/ground_truth/state']):
    positions.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
    times.append(t.to_sec())
bag.close()
if not positions:
    print('  ERROR: no drone1 data'); sys.exit(1)
p = np.array(positions)
t_arr = np.array(times) - times[0]
dist = np.linalg.norm(p - p[:3].mean(axis=0), axis=1)
explosion = np.where(dist > 20)[0]
nan_mask = ~np.isfinite(p).all(axis=1)
print(f'  drone1: {len(p)} msgs, dur={t_arr[-1]:.0f}s')
print(f'  xyz range: x[{p[:,0].min():.1f},{p[:,0].max():.1f}] y[{p[:,1].min():.1f},{p[:,1].max():.1f}] z[{p[:,2].min():.2f},{p[:,2].max():.2f}]')
if explosion.size:
    print(f'  EXPLOSION >20m at t={t_arr[explosion[0]]:.1f}s  <-- BAG MAY BE BAD')
elif nan_mask.any():
    print(f'  NaN detected!  <-- BAG BAD')
else:
    print(f'  CLEAN - bag looks good')
EOF
  echo "========================================"
}

case "$SCENE" in
  all|s1) run_scene "S1 Hover"        "scene01_hover_4drones.launch"    55  "scene01_hover_4drones_v2"     12 ;;
esac
case "$SCENE" in
  all|s2) run_scene "S2 Circle"       "scene02_circle_4drones.launch"   55  "scene02_circle_4drones_v2"    12 ;;
esac
case "$SCENE" in
  all|s2p) run_scene "S2' Lemniscate" "scene02p_lemni_4drones.launch"   55  "scene02p_lemni_4drones_v2"    12 ;;
esac
case "$SCENE" in
  all|s3) run_scene "S3 Reconfig"     "scene03_reconfig_4drones.launch" 85  "scene03_reconfig_4drones_v2"  12 ;;
esac
case "$SCENE" in
  all|s4) run_scene "S4 Wind"         "scene04_wind_4drones.launch"     55  "scene04_wind_4drones_v2"      15 ;;
esac
case "$SCENE" in
  all|s5) run_scene "S5 Longtime"     "scene02_circle_4drones.launch"   115 "scene05_longtime_4drones_v2"  12 ;;
esac

echo ""
echo "[record_bags_v2] DONE. Bags in: $BAG_DIR"
ls -lh "$BAG_DIR"/*.bag 2>/dev/null || echo "(no bags found)"
