#!/usr/bin/env bash
# Source this script in WSL2 mirrored-network mode so ROS1 advertises a
# concrete IPv4 address instead of localhost/127.0.0.1.
#
# Usage:
#   source scripts/use_ros_mirrored_ip.sh
#
# After sourcing, start roslaunch/roscore in the same shell.

_ros_pick_ipv4() {
  hostname -I | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | head -n 1
}

ROS_MIRRORED_IP="$(_ros_pick_ipv4)"
if [[ -z "${ROS_MIRRORED_IP}" ]]; then
  echo "[use_ros_mirrored_ip] failed to detect IPv4 from hostname -I" >&2
  return 1 2>/dev/null || exit 1
fi

export ROS_MASTER_URI="http://${ROS_MIRRORED_IP}:11311"
export ROS_IP="${ROS_MIRRORED_IP}"
export ROS_HOSTNAME="${ROS_MIRRORED_IP}"

echo "[use_ros_mirrored_ip] ROS_MASTER_URI=${ROS_MASTER_URI}"
echo "[use_ros_mirrored_ip] ROS_IP=${ROS_IP}"
echo "[use_ros_mirrored_ip] ROS_HOSTNAME=${ROS_HOSTNAME}"
