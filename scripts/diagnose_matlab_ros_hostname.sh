#!/usr/bin/env bash
# Diagnose Windows-hostname reachability for MATLAB<->WSL ROS1 communication.
#
# Purpose:
#   ROS1 peer transport can fail when Windows MATLAB registers a hostname
#   that WSL/Gazebo cannot resolve. This script prints the concrete values
#   we need and the exact /etc/hosts line to add if hostname resolution is
#   still broken.
#
# Usage:
#   bash scripts/diagnose_matlab_ros_hostname.sh

set -euo pipefail

if [[ ! -x /mnt/c/Windows/System32/hostname.exe ]]; then
  echo "[diagnose] Windows hostname.exe not found under /mnt/c/Windows/System32" >&2
  exit 1
fi

WIN_HOST_RAW="$(/mnt/c/Windows/System32/hostname.exe | tr -d '\r' | iconv -f GBK -t UTF-8 2>/dev/null || /mnt/c/Windows/System32/hostname.exe | tr -d '\r')"
WIN_HOST_PUNY="$(python3 - <<'PY'
import sys
name = sys.stdin.read().strip()
print(name.encode('idna').decode('ascii'))
PY
<<<"$WIN_HOST_RAW")"

WIN_IPV4S="$(/mnt/c/Windows/System32/ipconfig.exe | iconv -f GBK -t UTF-8 | grep 'IPv4 地址' | sed 's/.*: //' | tr -d '\r')"
WSL_IPS="$(hostname -I | xargs)"

echo "[diagnose] Windows hostname (Unicode): $WIN_HOST_RAW"
echo "[diagnose] Windows hostname (punycode): $WIN_HOST_PUNY"
echo "[diagnose] WSL visible IPv4(s): $WSL_IPS"
echo "[diagnose] Windows IPv4 candidates:"
echo "$WIN_IPV4S"
echo
echo "[diagnose] Current /etc/hosts lookup:"
getent hosts "$WIN_HOST_PUNY" || true
echo
echo "[diagnose] If WSL cannot resolve the punycode hostname, add a line like:"
PRIMARY_IP="$(echo "$WIN_IPV4S" | head -n 1)"
echo "  $PRIMARY_IP $WIN_HOST_PUNY $WIN_HOST_RAW"
echo
echo "[diagnose] Then reconnect MATLAB with an explicit IPv4 node host:"
echo "  run matlab/run_connect_ros_wsl_windows.m"
