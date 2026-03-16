cmake_minimum_required(VERSION 3.0.2)
project(image_bridge)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  message_filters
  nav_msgs
  tf2_ros
  geometry_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/image_bridge_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)
