cmake_minimum_required(VERSION 3.22)
project(moveit_ur10)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

add_executable(move_group_interface_tutorial src/move_group_interface_tutorial.cpp)

ament_target_dependencies(move_group_interface_tutorial
  rclcpp
  moveit_ros_planning_interface
)

install(TARGETS
  move_group_interface_tutorial
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  src/simple_move.py
  DESTINATION lib/${PROJECT_NAME}
)
# Install Python scripts as executables
install(PROGRAMS
  scripts/rebar_detector.py
  scripts/rebar_supervisor.py
  scripts/seam_tracker_moveit.py
  scripts/seam_detector.py
  scripts/seam_weld_detector.py
  scripts/seam_weld_supervisor.py
  scripts/cube_edge_detector.py
  scripts/cube_edge_supervisor.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files (skip if you already have this)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
    PATTERN "setup_assistant.launch" EXCLUDE)
endif()

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})
