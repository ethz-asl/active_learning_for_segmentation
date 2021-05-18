cmake_minimum_required(VERSION 2.8.3)
project(active_3d_planning_app_curiosity)

find_package(catkin_simple REQUIRED)

find_package(airsim_ros_pkgs REQUIRED)

catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-std=c++14)

catkin_package()



cs_add_executable(trajectory_caller_node
        src/experiments/trajectory_caller_node.cpp
        )

target_link_libraries(trajectory_caller_node ${catkin_LIBRARIES})
