cmake_minimum_required(VERSION 2.8.3)
project(volumetric_labeling)

find_package(catkin_simple REQUIRED)

catkin_python_setup()
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-std=c++17 -Wall -Wextra)

# catkin_package()

#############
# Libraries #
#############
cs_add_library(${PROJECT_NAME}
        src/labler/volumetric_labler.cpp
        )

###############
# Executables #
###############
cs_add_executable(volumetric_labler_node
        app/volumetric_labler_node.cpp
        )
target_link_libraries(volumetric_labler_node ${PROJECT_NAME})

###########
# Export #
###########

cs_install()
cs_export()
