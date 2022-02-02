cmake_minimum_required(VERSION 2.8.3)
project(embodied_active_learning)
add_definitions(-std=c++17)

# To create the above file, navigate to the directory containing this CMakeLists.txt file and then run:
# echo "set(AIRSIM_ROOT $HOME/[path to your AirSim install directory])" > ./AirsimPath.txt
# Be sure to replace [path to AirSim install dir] with the actual path where you installed your copy of AirSim.

# We add all these sources so they can be compiled with e.g. gcc, since the Airsim(Plugin) binaries need to be compiled
# with clang for compatibility with unreal engine


find_package(catkin_simple REQUIRED)

catkin_python_setup()
catkin_simple(ALL_DEPS_REQUIRED)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_CXX_STANDARD 11)
set(CXX_EXP_LIB "-nostdinc++ -I/usr/include/c++/8 -I/usr/include/x86_64-linux-gnu/c++/8 -nodefaultlibs
-l/usr/lib/x86_64-linux-gnu/libc++.so -l/usr/lib/x86_64-linux-gnu/libc++abi.so
-lm -lc -lgcc_s -lgcc
-lstdc++fs -fmax-errors=10 -Wnoexcept -Wstrict-null-sentinel")

set(CMAKE_CXX_STANDARD 17)

catkin_package()
cs_add_executable(curiosity_planner_node
        src/planner/curiosity_planner_node.cpp
        )


target_link_libraries(curiosity_planner_node ${catkin_LIBRARIES})
