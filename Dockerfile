FROM ros:humble-ros-base-jammy AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip python3-colcon-common-extensions \
      ros-humble-rclpy ros-humble-sensor-msgs ros-humble-std-msgs \
      ros-humble-launch-ros fonts-dejavu-core libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-humble.txt /tmp/requirements-humble.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-humble.txt

WORKDIR /opt/ros_ws
COPY . src/obstacle_detector_ros/
RUN source /opt/ros/humble/setup.bash \
    && colcon build --packages-select obstacle_detector_ros \
    && source install/setup.bash \
    && python3 -c "import rclpy, obstacle_detector; from obstacle_detector_ros import loader_node, processor_node" \
    && ros2 pkg executables obstacle_detector_ros

COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh
WORKDIR /app
COPY run.py /app/run.py
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
STOPSIGNAL SIGINT
CMD ["ros2", "launch", "obstacle_detector_ros", "pipeline.launch.py", "--show-args"]

# Explicit test target exercises native DDS in Humble.
FROM base AS test
RUN python3 -m pip install --no-cache-dir pytest==8.3.5
RUN source /opt/ros/humble/setup.bash && source /opt/ros_ws/install/setup.bash \
    && python3 -m pytest /opt/ros_ws/src/obstacle_detector_ros/tests -v

FROM base AS runtime
