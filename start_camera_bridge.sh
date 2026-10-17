#!/usr/bin/env bash
# Script khởi động cầu nối camera từ Gazebo Harmonic sang ROS 2
# Topic ROS 2 đích: /drone/camera/image_raw

ROOT="${DRONE_ROOT:-$HOME/Workspace/260727-drone-setup}"
source /opt/ros/jazzy/setup.bash

GZ_CAM_TOPIC="/world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/imager/image"
ROS_CAM_TOPIC="/drone/camera/image_raw"

echo "=========================================================="
echo "  KHỞI ĐỘNG CẦU NỐI CAMERA GAZEBO -> ROS 2"
echo "  Gazebo Topic : $GZ_CAM_TOPIC"
echo "  ROS 2 Topic  : $ROS_CAM_TOPIC"
echo "=========================================================="

ros2 run ros_gz_bridge parameter_bridge \
  "$GZ_CAM_TOPIC@sensor_msgs/msg/Image@gz.msgs.Image" \
  --ros-args -r "$GZ_CAM_TOPIC:=$ROS_CAM_TOPIC"
