cmake_minimum_required(VERSION 2.8.3)
project(embodied_active_learning)

find_package(catkin_simple REQUIRED)

catkin_python_setup()

catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-std=c++11)

catkin_package()
