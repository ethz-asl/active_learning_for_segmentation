#include <cstdlib>
#include <queue>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <active_3d_planning_core/tools/defaults.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <tf/transform_datatypes.h>
#include <vehicles/multirotor/api/MultirotorRpcLibClient.hpp>
#include "airsim_ros_pkgs/SetLocalPosition.h"
#include "ros/ros.h"
#include "trajectory_msgs/MultiDOFJointTrajectory.h"

namespace active_3d_planning {

typedef struct Waypoint {
  geometry_msgs::Transform_<std::allocator<void>> pose;
  bool isGoal{};
} Waypoint;

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
  ros::Publisher waypointReachedPub;
  ros::Publisher goalPointReachedPub;

  std::queue<Waypoint*> points;
  bool followingGoal;
  bool rotatingMode;
  bool only_move_in_yaw_direction;
  bool verbose;
  bool trackTrajectoryYaw;
  double goalYaw;
  bool real_time;
  double move_time;
  bool collision_detected;
  bool clear_wp_on_collision;


    msr::airlib::MultirotorRpcLibClient* airsimClient;
  Waypoint* currentGoal;
};

TrajectoryCallerNode::TrajectoryCallerNode(const ros::NodeHandle& nh,
                                           const ros::NodeHandle& nh_private)
    : nh_(nh),
      nh_private_(nh_private),
      followingGoal(false),
      only_move_in_yaw_direction(true),
      verbose(false),
      trackTrajectoryYaw(false),
      real_time(false),
      move_time(0.1),
      goalYaw(0),
      clear_wp_on_collision(true),
      collision_detected(false){
  client = nh_.serviceClient<airsim_ros_pkgs::SetLocalPosition>(
      "/local_position_goal");

  sub = nh_private_.subscribe("/command/trajectory", 10,
                              &TrajectoryCallerNode::callback, this);
  odomSub = nh_private_.subscribe("/odom", 10,
                                  &TrajectoryCallerNode::odomCallback, this);
  collisionSub = nh_private_.subscribe(
      "/collision", 10, &TrajectoryCallerNode::collisionCallback, this);

  waypointReachedPub = nh_.advertise<std_msgs::Bool>("waypoint_reached", 100);

  nh_private_.param<bool>("verbose", verbose, true);
  //  nh_.param<bool>("move_in_yaw", only_move_in_yaw_direction, true);
  nh_private_.param<bool>("move_in_yaw", only_move_in_yaw_direction, false);

  nh_private_.param<bool>("real_time", real_time, true);
  nh_private_.param<bool>("clear_wp_on_collision", clear_wp_on_collision, true);
  nh_private_.param<double>("move_time", move_time, 0.1);

  if(!real_time) {
      airsimClient = new msr::airlib::MultirotorRpcLibClient;
      airsimClient->enableApiControl(true);
      airsimClient->armDisarm(true);
//      airsimClient->takeoffAsync(1)->waitOnLastTask();
      std::cout << "Airsim Client start" << std::endl;
  }
  std::cout << "MOVE IN YAW " << only_move_in_yaw_direction << std::endl;
  currentGoal = nullptr;
}

void TrajectoryCallerNode::collisionCallback(const std_msgs::Bool& msg) {
  if (msg.data) {
    collision_detected = true;
    std::wcout
        << "Collision detected in trajectory caller node. Going to clear "
           "stored waypoints"
        << std::endl;
    followingGoal = false;
    // Clear stored points
    std::queue<Waypoint*> empty;
    std::swap(points, empty);
//
//    auto* pt = new Waypoint;
//    auto lastPose = currentGoal;
//
//    tf::Quaternion q(currentGoal->pose.rotation.x, currentGoal->pose.rotation.y,
//                     currentGoal->pose.rotation.z,
//                     currentGoal->pose.rotation.w);
//    tf::Matrix3x3 m(q);
//    double roll, pitch, yaw;
//    m.getRPY(roll, pitch, yaw);
//
//    pt->pose.translation.x = lastPose->pose.translation.x - cos(yaw) * 0.5;
//    pt->pose.translation.y = lastPose->pose.translation.y - sin(yaw) * 0.5;
//    pt->pose.translation.z = lastPose->pose.translation.z;
//
//    pt->pose.rotation.y = lastPose->pose.rotation.y;
//    pt->pose.rotation.x = lastPose->pose.rotation.x;
//    pt->pose.rotation.z = lastPose->pose.rotation.z;
//    pt->pose.rotation.w = lastPose->pose.rotation.w;
//    pt->isGoal = true;
//    points.push(pt);
  }
}

void TrajectoryCallerNode::odomCallback(const nav_msgs::Odometry& msg) {
  if (!followingGoal) {
    // Not following a reference point. Check if there is a new one
    // There is a point we can follow

    if (points.empty()) return;
    currentGoal = points.front();
    if (verbose)
      std::cout << "Got new goal point: " << currentGoal->pose
                << " will publish it to service" << std::endl;

    airsim_ros_pkgs::SetLocalPosition srv;
    srv.request.x = currentGoal->pose.translation.x;
    srv.request.y = currentGoal->pose.translation.y;
    srv.request.z = currentGoal->pose.translation.z;
    auto current_position = msg.pose.pose.position;

    if (only_move_in_yaw_direction && !currentGoal->isGoal) {
      // If only move in yaw direction, set yaw of waypoints (except goal point)
      // to moving direction
      auto delta_x = srv.request.x - current_position.x;
      auto delta_y = srv.request.y - current_position.y;
      std::cout << "DELTAS: " << delta_x << " " << delta_y << std::endl;
      if (abs(delta_x) + abs(delta_y) <= 0.05) {
        srv.request.yaw = goalYaw;
      } else {
        srv.request.yaw = atan2(delta_y, delta_x);
      }

      if (verbose) std::cout << "forcing yaw in view direction" << std::endl;
    } else {
      std::cout << "forcing yaw to goal orientation" << std::endl;
      // Rotate to goal yaw
      tf::Quaternion q(
          currentGoal->pose.rotation.x, currentGoal->pose.rotation.y,
          currentGoal->pose.rotation.z, currentGoal->pose.rotation.w);
      tf::Matrix3x3 m(q);
      double roll, pitch, yaw;
      m.getRPY(roll, pitch, yaw);
      srv.request.yaw = yaw;
    }
    goalYaw = srv.request.yaw;
    std::cout << "---> Goal Yaw: " << goalYaw << std::endl;
    if (real_time) {
      if (client.call(srv)) {
        followingGoal = true;
      } else {
        if (verbose) std::cout << "client not ready yet " << std::endl;
      }
    } else {
        followingGoal = true;
        ros::Duration(move_time).sleep();
        Pose pose;
        pose.position.x() = srv.request.x;
        pose.position.y() = srv.request.y;
        pose.position.z() = srv.request.z;
        tf::Quaternion q;
        q.setRPY(0,0,srv.request.yaw);
        pose.orientation.x() = q.x();
        pose.orientation.y() = q.y();
        pose.orientation.z() = q.z();
        pose.orientation.w() = q.w();

        airsimClient->simSetVehiclePose(pose, false, "drone_1");

    }
  } else {
    // Currently following a goal. Lets check if we reached it

    // Track the current pose
    auto current_position_ =
        Eigen::Vector3d(msg.pose.pose.position.x, msg.pose.pose.position.y,
                        msg.pose.pose.position.z);

    auto goal_position_ = Eigen::Vector3d(currentGoal->pose.translation.x,
                                          currentGoal->pose.translation.y,
                                          currentGoal->pose.translation.z);

    if ((goal_position_ - current_position_).norm() < 0.1) {
      collision_detected = false;
      double yaw = tf::getYaw(msg.pose.pose.orientation);
      if (defaults::angleDifference(goalYaw, yaw) < 0.1) {
        if (verbose) std::cout << "reached goal point (rotation)" << std::endl;
        // Reached target yaw.
        std_msgs::Bool wp_msg;
        wp_msg.data = true;
        if (currentGoal->isGoal) {
          // Reached Goal
          std::cout << "reached a goal point" << std::endl;
          goalPointReachedPub.publish(wp_msg);
        } else {
          std::cout << "reached a way point" << std::endl;
          waypointReachedPub.publish(wp_msg);
        }
        auto* off = points.front();
        points.pop();
        delete off;
        followingGoal = false;
      }
    }
  }
}

void TrajectoryCallerNode::callback(
    const trajectory_msgs::MultiDOFJointTrajectory trajectory) {
  for (const auto& point : trajectory.points) {
    for (auto pose : point.transforms) {
      if (verbose)
        std::cout << "adding pose to goal" << pose.rotation.y << " "
                  << pose.rotation.w << std::endl;

      auto* pt = new Waypoint;
      pt->pose = pose;
      pt->isGoal = false;
      points.push(pt);
    }
  }
  // Add goal point for yaw tracking
  auto* lastPose = points.back();
  auto* pt = new Waypoint;
  pt->pose.translation.x = lastPose->pose.translation.x;
  pt->pose.translation.y = lastPose->pose.translation.y;
  pt->pose.translation.z = lastPose->pose.translation.z;

  pt->pose.rotation.y = lastPose->pose.rotation.y;
  pt->pose.rotation.x = lastPose->pose.rotation.x;
  pt->pose.rotation.z = lastPose->pose.rotation.z;
  pt->pose.rotation.w = lastPose->pose.rotation.w;
  pt->isGoal = true;
  points.push(pt);

  if (verbose)
    std::cout << "adding goal point" << pt->pose.rotation.y << " "
              << pt->pose.rotation.w << std::endl;
}

}  // namespace active_3d_planning

int main(int argc, char** argv) {
  ros::init(argc, argv, "trajectory_caller_node");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                 ros::console::levels::Debug);
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  active_3d_planning::TrajectoryCallerNode node(nh, nh_private);
  ros::spin();
  return 0;
}