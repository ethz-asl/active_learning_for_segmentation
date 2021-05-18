#include <cstdlib>
#include <queue>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <active_3d_planning_core/tools/defaults.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <tf/transform_datatypes.h>

#include "airsim_ros_pkgs/SetLocalPosition.h"
#include "ros/ros.h"
#include "trajectory_msgs/MultiDOFJointTrajectory.h"

namespace active_3d_planning {

// Ros-wrapper for c++ voxblox code to evaluates voxblox maps upon request from
// the eval_plotting_node. Largely based on the voxblox_ros/voxblox_eval.cc
// code. Pretty ugly and non-general code but just needs to work in this
// specific case atm...
class TrajectoryCallerNode {
 public:
  TrajectoryCallerNode(const ros::NodeHandle& nh,
                       const ros::NodeHandle& nh_private);

  void callback(const trajectory_msgs::MultiDOFJointTrajectory trajectory);

  void odomCallback(const nav_msgs::Odometry& msg);

  void collisionCallback(const std_msgs::Bool& msg);

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  ros::ServiceClient client;
  ros::Subscriber sub;
  ros::Subscriber odomSub;
  ros::Subscriber collisionSub;
  std::queue<geometry_msgs::Transform_<std::allocator<void>>> points;
  bool followingGoal;
  bool rotatingMode;
  bool only_move_in_yaw_direction;
  bool verbose;
  geometry_msgs::Transform_<std::allocator<void>>* currentGoal;
};

TrajectoryCallerNode::TrajectoryCallerNode(const ros::NodeHandle& nh,
                                           const ros::NodeHandle& nh_private)
    : nh_(nh),
      nh_private_(nh_private),
      followingGoal(false),
      only_move_in_yaw_direction(false),
      verbose(false) {
  client = nh_.serviceClient<airsim_ros_pkgs::SetLocalPosition>(
      "/local_position_goal");

  sub = nh_private_.subscribe("/command/trajectory", 10,
                              &TrajectoryCallerNode::callback, this);
  odomSub = nh_private_.subscribe("/odom", 10,
                                  &TrajectoryCallerNode::odomCallback, this);
  collisionSub = nh_private_.subscribe(
      "/collision", 10, &TrajectoryCallerNode::collisionCallback, this);

  nh.param("verbose", verbose, true);
  nh.param("move_in_yaw", only_move_in_yaw_direction, true);
  currentGoal = nullptr;
}

void TrajectoryCallerNode::collisionCallback(const std_msgs::Bool& msg) {
  if (msg.data) {
    std::wcout
        << "Collision detected in trajectory caller node. Going to clear "
           "stored waypoints"
        << std::endl;
    followingGoal = false;
    // Clear stored points
    std::queue<geometry_msgs::Transform_<std::allocator<void>>> empty;
    std::swap(points, empty);
  }
}

void TrajectoryCallerNode::odomCallback(const nav_msgs::Odometry& msg) {
  if (!followingGoal) {
    // Not following a reference point. Check if there is a new one
    // There is a point we can follow

    if (points.empty()) return;

    currentGoal = &points.front();
    if (verbose)
      std::cout << "Got new goal point: " << *currentGoal
                << " will publish it to service" << std::endl;

    airsim_ros_pkgs::SetLocalPosition srv;
    srv.request.x = (*currentGoal).translation.x;
    srv.request.y = (*currentGoal).translation.y;
    srv.request.z = (*currentGoal).translation.z;
    auto current_position = msg.pose.pose.position;

    if (only_move_in_yaw_direction && !rotatingMode) {
      // If only move in yaw direction, set yaw of waypoints (except goal point)
      // to moving direction
      auto delta_x = srv.request.x - current_position.x;
      auto delta_y = srv.request.y - current_position.y;
      srv.request.yaw = atan2(delta_y, delta_x);
    } else {
      // Rotate to goal yaw
      tf::Quaternion q((*currentGoal).rotation.x, (*currentGoal).rotation.y,
                       (*currentGoal).rotation.z, (*currentGoal).rotation.w);
      tf::Matrix3x3 m(q);
      double roll, pitch, yaw;
      m.getRPY(roll, pitch, yaw);
      srv.request.yaw = yaw;
    }

    if (client.call(srv)) {
      followingGoal = true;
    } else {
      if (verbose) std::cout << "client not ready yet " << std::endl;
    }
  } else {
    // Currently following a goal. Lets check if we reached it

    // Track the current pose
    auto current_position_ =
        Eigen::Vector3d(msg.pose.pose.position.x, msg.pose.pose.position.y,
                        msg.pose.pose.position.z);

    auto goal_position_ = Eigen::Vector3d((*currentGoal).translation.x,
                                          (*currentGoal).translation.y,
                                          (*currentGoal).translation.z);

    if ((goal_position_ - current_position_).norm() < 0.1) {
      if (verbose) std::cout << "reached goal (translation)" << std::endl;
      // We are at goal point, need to check yaw
      if (only_move_in_yaw_direction) {
        if (points.size() == 1 && !rotatingMode) {
          // If this is the last waypoint, go into rotating mode to rotate to
          // goal yaw
          rotatingMode = true;
          followingGoal = false;
          if (verbose)
            std::cout << "reached goal. Going to rotate to correct yaw angle"
                      << std::endl;
        } else {
            // Don't have to correct yaw. Move on
            rotatingMode = false;
            points.pop();
            followingGoal = false;
        }
      } else {
        // We always track the yaw angle OR we are not in rotating mode
        double yaw = tf::getYaw(msg.pose.pose.orientation);
        double goal_yaw = tf::getYaw((*currentGoal).rotation);
        if (defaults::angleDifference(goal_yaw, yaw) < 0.05) {
          // We reached target yaw, disable rotating mode
          rotatingMode = false;
          points.pop();
          followingGoal = false;
        }
      }
    }
  }
}

void TrajectoryCallerNode::callback(
    const trajectory_msgs::MultiDOFJointTrajectory trajectory) {
  for (auto point : trajectory.points) {
    for (auto pose : point.transforms) {
      points.push(pose);
      if (verbose)
        std::cout << "adding pose to goal" << pose.rotation.y << " "
                  << pose.rotation.w << std::endl;
    }
  }
}

}  // namespace active_3d_planning

int main(int argc, char** argv) {
  ros::init(argc, argv, "evaluation_node");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                 ros::console::levels::Debug);
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  active_3d_planning::TrajectoryCallerNode node(nh, nh_private);
  ros::spin();
  return 0;
}