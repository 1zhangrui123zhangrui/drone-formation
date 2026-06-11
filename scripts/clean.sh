#!/bin/bash
# 杀掉所有 ROS 和 Gazebo 残留进程
killall -9 gzserver gzclient 2>/dev/null
killall -9 rosmaster roscore rosout 2>/dev/null
killall -9 python3 2>/dev/null
sleep 2
echo "[clean] 残留进程清理完毕"