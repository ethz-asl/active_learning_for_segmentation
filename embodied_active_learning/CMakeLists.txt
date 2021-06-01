cmake_minimum_required(VERSION 2.8.3)
project(embodied_active_learning)
add_definitions(-std=c++17)

# Setup Airsim
include(AirsimPath.txt)

# To create the above file, navigate to the directory containing this CMakeLists.txt file and then run:
# echo "set(AIRSIM_ROOT $HOME/[path to your AirSim install directory])" > ./AirsimPath.txt
# Be sure to replace [path to AirSim install dir] with the actual path where you installed your copy of AirSim.

# We add all these sources so they can be compiled with e.g. gcc, since the Airsim(Plugin) binaries need to be compiled
# with clang for compatibility with unreal engine
add_subdirectory("${AIRSIM_ROOT}/cmake/rpclib_wrapper" rpclib_wrapper)
add_subdirectory("${AIRSIM_ROOT}/cmake/AirLib" AirLib)
add_subdirectory("${AIRSIM_ROOT}/cmake/MavLinkCom" MavLinkCom)
set(RPC_LIB_INCLUDES "${AIRSIM_ROOT}/external/rpclib/rpclib-2.2.1/include")
set(RPC_LIB rpc) # name of .a file with lib prefix



find_package(catkin_simple REQUIRED)

catkin_python_setup()
catkin_simple(ALL_DEPS_REQUIRED)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${AIRSIM_ROOT}/AirLib/deps/eigen3
        ${AIRSIM_ROOT}/AirLib/include
        ${RPC_LIB_INCLUDES}
        ${AIRSIM_ROOT}/MavLinkCom/include
        ${AIRSIM_ROOT}/MavLinkCom/common_utils
        ${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_CXX_STANDARD 11)
set(CXX_EXP_LIB "-nostdinc++ -I/usr/include/c++/8 -I/usr/include/x86_64-linux-gnu/c++/8 -nodefaultlibs
-l/usr/lib/x86_64-linux-gnu/libc++.so -l/usr/lib/x86_64-linux-gnu/libc++abi.so
-lm -lc -lgcc_s -lgcc
-lstdc++fs -fmax-errors=10 -Wnoexcept -Wstrict-null-sentinel")

set(RPC_LIB_INCLUDES " ${AIRSIM_ROOT}/external/rpclib/rpclib-2.2.1/include")
set(RPC_LIB rpc) # name of .a file with lib prefix
message(STATUS "found RPC_LIB_INCLUDES=${RPC_LIB_INCLUDES}")
find_package(airsim_ros_pkgs REQUIRED)


catkin_package()


cs_add_executable(trajectory_caller_node
        src/planner/trajectory_caller_node.cpp
        )

target_link_libraries(trajectory_caller_node ${catkin_LIBRARIES} AirLib ${RPC_LIB})
