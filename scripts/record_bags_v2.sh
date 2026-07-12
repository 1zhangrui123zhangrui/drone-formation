#!/usr/bin/env bash
# record_bags_v2.sh — 重录所有场景 bag（正式审计版）
#
# 使用方法：
#   bash scripts/record_bags_v2.sh [SCENE] [SEED]
#   SCENE: all | s1 | s2 | s3 | s4 | s5（默认 all）
#   SEED:  seed01 | seed02 | ... | seed05 | allseeds（默认 seed01）
# 可选环境变量：
#   RECORD_LAUNCH_ARGS='omega:=0.15'
#
# 每个场景步骤：
#   1. roslaunch 启动仿真
#   2. 等待 4 机真正 ready（服务、话题、起飞高度、仿真时间）
#   3. rosbag record 开始（后台）
#   4. 按仿真时间而不是墙钟时间等待正式时长
#
# 正式输出目录：~/drone-formation-e2e/data/raw_bags/v2/formal_5x5/
# 录完后会自动执行高标准质量审计。只有 PASS 的 bag 才会进入正式目录。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_BAG_DIR="$ROOT_DIR/data/raw_bags/v2"
BAG_DIR="$RAW_BAG_DIR/formal_5x5"
TMP_DIR="$RAW_BAG_DIR/tmp_recording"
mkdir -p "$BAG_DIR"
mkdir -p "$TMP_DIR"
read -r -a EXTRA_LAUNCH_ARGS <<< "${RECORD_LAUNCH_ARGS:-}"

# 所有 Topics
TOPICS=(
  /clock
  /drone1/ground_truth/state /drone2/ground_truth/state
  /drone3/ground_truth/state /drone4/ground_truth/state
  /drone1/p_des /drone2/p_des /drone3/p_des /drone4/p_des
  /drone1/v_des /drone2/v_des /drone3/v_des /drone4/v_des
  /drone1/command/twist /drone2/command/twist
  /drone3/command/twist /drone4/command/twist
  /drone1/cmd_vel_teacher /drone2/cmd_vel_teacher
  /drone3/cmd_vel_teacher /drone4/cmd_vel_teacher
)

SCENE="${1:-all}"
SEED_ARG="${2:-seed01}"
SEEDS=()

case "$SCENE" in
  all|s1|s2|s3|s4|s5) ;;
  *)
    echo "[record_bags_v2] invalid scene: $SCENE"
    echo "Expected: all | s1 | s2 | s3 | s4 | s5"
    exit 2
    ;;
esac

case "$SEED_ARG" in
  allseeds) SEEDS=(seed01 seed02 seed03 seed04 seed05) ;;
  seed0[1-5]) SEEDS=("$SEED_ARG") ;;
  *)
    echo "[record_bags_v2] invalid seed: $SEED_ARG"
    echo "Expected: seed01..seed05 or allseeds"
    exit 2
    ;;
esac

seed_number() {
  local seed="$1"
  echo "${seed#seed}" | sed 's/^0*//'
}

run_scene() {
  local NAME="$1"
  local LAUNCH_PKG="drone_sim"
  local LAUNCH_FILE="$2"
  local DURATION="$3"       # 正式录包仿真时长(s)
  local SCENE_KEY="$4"
  local READY_SIM_TIME="${5:-8}"     # 不早于该仿真时间开始录
  local MIN_ALTITUDE="${6:-1.0}"     # 4 机都需高于该高度
  local READY_TIMEOUT="${7:-120}"    # ready 判定墙钟超时(s)
  local SEED="$8"

  echo ""
  echo "========================================"
  echo "  SCENE: $NAME  (duration=${DURATION}s)"
  echo "========================================"

  local EFFECTIVE_BAG_NAME="${SCENE_KEY}_${SEED}"
  local FINAL_BAG_PATH="$BAG_DIR/${EFFECTIVE_BAG_NAME}.bag"
  local TMP_BAG_PATH="$TMP_DIR/${EFFECTIVE_BAG_NAME}.bag"
  local BAG_PID=""
  local LAUNCH_PID=""

  cleanup_scene() {
    if [[ -n "${BAG_PID}" ]]; then
      kill -- "-${BAG_PID}" 2>/dev/null || true
      wait "${BAG_PID}" 2>/dev/null || true
      BAG_PID=""
    fi
    if [[ -n "${LAUNCH_PID}" ]]; then
      kill -- "-${LAUNCH_PID}" 2>/dev/null || true
      wait "${LAUNCH_PID}" 2>/dev/null || true
      LAUNCH_PID=""
    fi
    sleep 2
  }

  # 启动仿真（后台）
  source "$ROOT_DIR/ros_ws/devel/setup.bash"
  local LAUNCH_ARGS=(gui:=false "seed:=$(seed_number "$SEED")" "${EXTRA_LAUNCH_ARGS[@]}")
  setsid roslaunch "$LAUNCH_PKG" "$LAUNCH_FILE" "${LAUNCH_ARGS[@]}" &
  LAUNCH_PID=$!
  echo "[record] launch PID=$LAUNCH_PID"
  echo "[record] launch args: ${LAUNCH_ARGS[*]}"

  echo "[record] waiting for scene ready..."
  python3 "$ROOT_DIR/scripts/wait_scene_ready.py" \
    --num-drones 4 \
    --min-sim-time "$READY_SIM_TIME" \
    --min-altitude "$MIN_ALTITUDE" \
    --timeout "$READY_TIMEOUT"

  # 开始录包（后台）
  rm -f "$TMP_BAG_PATH" "$TMP_BAG_PATH.active" "$FINAL_BAG_PATH"
  echo "[record] starting rosbag record -> $TMP_BAG_PATH"
  setsid rosbag record -O "$TMP_BAG_PATH" "${TOPICS[@]}" &
  BAG_PID=$!

  # 按仿真时间等待正式录包时长
  local SIM_TIMEOUT
  SIM_TIMEOUT="$(python3 - <<PY
duration = float(${DURATION})
print(int(max(180.0, duration * 4.0)))
PY
)"
  echo "[record] recording for ${DURATION}s of simulated time (wall timeout=${SIM_TIMEOUT}s)..."
  set +e
  python3 "$ROOT_DIR/scripts/wait_sim_time.py" --duration "$DURATION" --timeout "$SIM_TIMEOUT"
  local WAIT_STATUS=$?
  set -e

  # 停止录包
  cleanup_scene

  # 正式质量审计
  if [[ "$WAIT_STATUS" -ne 0 ]]; then
    echo "[record] simulated-time wait failed for $EFFECTIVE_BAG_NAME (status=$WAIT_STATUS)"
    echo "[record] FAIL -> rejected incomplete bag: $TMP_BAG_PATH"
    return 1
  fi
  echo "[record] auditing $EFFECTIVE_BAG_NAME..."
  if python3 "$ROOT_DIR/scripts/audit_bag_quality.py" "$TMP_BAG_PATH"; then
    mv "$TMP_BAG_PATH" "$FINAL_BAG_PATH"
    echo "[record] PASS -> accepted formal bag: $FINAL_BAG_PATH"
  else
    echo "[record] FAIL -> rejected bag kept out of formal set: $TMP_BAG_PATH"
    return 1
  fi
  echo "========================================"
}

for SEED in "${SEEDS[@]}"; do
  case "$SCENE" in
    all|s1) run_scene "S1 Hover"    "scene01_hover_4drones.launch"    60  "scene01_hover"     8   1.0 120 "$SEED" ;;
  esac
  case "$SCENE" in
    all|s2) run_scene "S2 Circle"   "scene02_circle_4drones.launch"   90  "scene02_circle"    12  1.0 150 "$SEED" ;;
  esac
  case "$SCENE" in
    all|s3) run_scene "S3 Reconfig" "scene03_reconfig_4drones.launch" 120 "scene03_reconfig"  14  1.0 180 "$SEED" ;;
  esac
  case "$SCENE" in
    all|s4) run_scene "S4 Wind"     "scene04_wind_4drones.launch"     90  "scene04_wind"      12  1.0 180 "$SEED" ;;
  esac
  case "$SCENE" in
    all|s5) run_scene "S5 Longtime" "scene05_longtime_4drones.launch" 180 "scene05_longtime"  15  1.0 240 "$SEED" ;;
  esac
done

echo ""
echo "[record_bags_v2] DONE. Bags in: $BAG_DIR"
ls -lh "$BAG_DIR"/*.bag 2>/dev/null || echo "(no bags found)"
