#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
source /opt/ros_ws/install/setup.bash
# Keep the previous image's `run.py ...` / `-m obstacle_detector ...` interface.
case "${1:-}" in
  *.py|-m) exec python3 "$@" ;;
  *) exec "$@" ;;
esac
