cmake_minimum_required(VERSION 3.0.2)
project(habitat_ros_utils)

find_package(catkin REQUIRED COMPONENTS rospy nav_msgs
  message_generation)


add_service_files(
DIRECTORY srv
FILES
PublishTransform.srv
)

generate_messages( DEPENDENCIES std_msgs)

catkin_package(CATKIN_DEPENDS rospy)

# do not wildcard install files since the root folder of the package will contain a debian folder for releasing
catkin_install_python(PROGRAMS
  src/transform_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
