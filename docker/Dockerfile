FROM osrf/ros:noetic-desktop-full

ARG DEBIAN_FRONTEND=noninteractive
ARG GAZEBO_VERSION=11
ARG MATLAB_RELEASE=R2023b

ENV ROS_DISTRO=noetic
ENV GAZEBO_VERSION=${GAZEBO_VERSION}
ENV MATLAB_RELEASE=${MATLAB_RELEASE}
ENV WORKSPACE_ROOT=/workspace/drone-formation-e2e

RUN apt-get update && apt-get install -y \
    gazebo${GAZEBO_VERSION} \
    python3-pip \
    python3-rosdep \
    python3-catkin-tools \
    python3-vcstool \
    ros-noetic-geographic-msgs \
    ros-noetic-mavros \
    ros-noetic-mavros-extras \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true

WORKDIR ${WORKSPACE_ROOT}

# MATLAB 由于许可限制，通常由宿主机挂载安装目录或通过网络许可接入。
CMD ["/bin/bash"]
