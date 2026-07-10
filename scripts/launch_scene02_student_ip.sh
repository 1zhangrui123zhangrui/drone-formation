#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/ros_ws/devel/setup.bash"
source "$ROOT_DIR/scripts/use_ros_mirrored_ip.sh"

exec roslaunch drone_sim scene02_circle_4drones_student.launch "${@}"
