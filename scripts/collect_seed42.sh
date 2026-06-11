#!/bin/bash
# 一键采集 5 个场景 seed=42 的基线数据
# 总耗时约 12-15 分钟,可挂在后台跑

set +e   # 单次失败不退出整个脚本

BAG_DIR=~/drone-formation-e2e/data/raw_bags
mkdir -p $BAG_DIR

# 场景列表(scene02_circle 已录,跳过)
SCENES=(
    "scene01_hover 60"
    "scene02p_lemni 90"
    "scene03_reconfig 120"
    "scene04_wind 90"
    "scene05_longtime 180"
)

clean_ros() {
    killall -9 gzserver gzclient rosmaster roscore rosout 2>/dev/null
    pkill -9 -f "scene_driver|teacher_controller|enable_motors|wind_driver|rosbag" 2>/dev/null
    sleep 4
}

trap clean_ros EXIT

for ENTRY in "${SCENES[@]}"; do
    read SCENE DURATION <<< "$ENTRY"
    BAG_NAME="${SCENE}_4drones_seed42.bag"
    
    echo ""
    echo "============================================="
    echo "  Running: $SCENE  (duration=${DURATION}s)"
    echo "  $(date)"
    echo "============================================="
    
    clean_ros
    
    # 启动 launch (后台,日志重定向避免刷屏)
    roslaunch drone_sim ${SCENE}_4drones.launch gui:=false \
        > /tmp/launch_${SCENE}.log 2>&1 &
    LAUNCH_PID=$!
    echo "[batch] launch PID=$LAUNCH_PID, waiting 25s..."
    sleep 25
    
    # 录制 bag
    echo "[batch] recording $BAG_NAME for ${DURATION}s..."
    rosbag record \
        -O ${BAG_DIR}/${BAG_NAME} \
        --duration=${DURATION} \
        /drone1/p_des /drone1/v_des /drone1/ground_truth/state /drone1/cmd_vel /drone1/cmd_vel_teacher \
        /drone2/p_des /drone2/v_des /drone2/ground_truth/state /drone2/cmd_vel /drone2/cmd_vel_teacher \
        /drone3/p_des /drone3/v_des /drone3/ground_truth/state /drone3/cmd_vel /drone3/cmd_vel_teacher \
        /drone4/p_des /drone4/v_des /drone4/ground_truth/state /drone4/cmd_vel /drone4/cmd_vel_teacher
    
    if [ -f ${BAG_DIR}/${BAG_NAME} ]; then
        echo "[batch] $BAG_NAME done, size: $(du -h ${BAG_DIR}/${BAG_NAME} | cut -f1)"
    else
        echo "[batch] WARNING: $BAG_NAME not created!"
    fi
done

clean_ros

echo ""
echo "============================================="
echo "  All done. Bag files:"
echo "============================================="
ls -lh $BAG_DIR/*.bag