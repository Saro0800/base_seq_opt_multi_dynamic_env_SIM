cmake_minimum_required(VERSION 3.0.2)
project(human_control)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  roscpp
  std_msgs
  geometry_msgs
)

find_package(gazebo REQUIRED)

catkin_package(
  LIBRARIES actor_control_plugin
  CATKIN_DEPENDS gazebo_ros roscpp std_msgs geometry_msgs
  DEPENDS GAZEBO
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(${GAZEBO_LIBRARY_DIRS})

# Gazebo plugin
add_library(actor_control_plugin SHARED
  src/actor_control_plugin.cpp
)

target_link_libraries(actor_control_plugin
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
)

add_library(robot_control_plugin SHARED
  src/robot_control_plugin.cpp
)

target_link_libraries(robot_control_plugin
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
)

add_library(human_static_control_plugin SHARED
  src/human_static_control_plugin.cpp
)

target_link_libraries(human_static_control_plugin
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
)

# Install plugin
install(TARGETS actor_control_plugin robot_control_plugin human_static_control_plugin
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/wasd_controller.py
  scripts/fpv_viewer.py
  scripts/map_viewer.py
  scripts/robot_controller.py
  scripts/rviz_markers.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files and world snippets
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY worlds/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds
)

install(DIRECTORY meshes/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/meshes
)
