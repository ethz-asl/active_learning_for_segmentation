#include "airsim_ros_pkgs/SetLocalPosition.h"
#include "embodied_active_learning/waypoint_reached.h"
#include "ros/ros.h"
#include "std_srvs/SetBool.h"
#include "trajectory_msgs/MultiDOFJointTrajectory.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <active_3d_planning_core/tools/defaults.h>
#include <cstdlib>
#include <nav_msgs/Odometry.h>
#include <queue>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <tf/transform_datatypes.h>
#include <vehicles/multirotor/api/MultirotorRpcLibClient.hpp>

namespace active_3d_planning {
// Stores waypoint information (gain, pose and if it is goalpoint)
typedef struct Waypoint {
  geometry_msgs::Transform_<std::allocator<void>> pose;
  bool isGoal{};
  double gain{};
} Waypoint;

// Ros node to track spefic waypoints provided by the active mapper.
// Calls the Airsim PID controller to track the points.
class TrajectoryCallerNode {
public:
  TrajectoryCallerNode(const ros::NodeHandle &nh,
                       const ros::NodeHandle &nh_private);

  // Callback for trajectories published by active plner
  void callback(const trajectory_msgs::MultiDOFJointTrajectory trajectory);

  // Callback for odometry
  void odomCallback(const nav_msgs::Odometry &msg);

  void collisionCallback(const std_msgs::Bool &msg);

  void gainCallback(const std_msgs::Float32 msg);

  bool setRunning(std_srvs::SetBool::Request &req,
                  std_srvs::SetBool::Response &res);

private:
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  ros::ServiceClient client;
  ros::Subscriber sub;
  ros::Subscriber odom_sub;
  ros::Subscriber collision_sub;
  ros::Subscriber gain_sub;
  ros::Publisher waypoint_reached_pub;
  ros::Publisher goalPointReachedPub;

  std::queue<Waypoint *> points;
  // Whether robot is idle or currently tracking a goal
  bool followingGoal;
  // If robot reached goal position but not orientation
  bool rotatingMode;
  // If true, robot only moves in yaw direction
  bool only_move_in_yaw_direction;
  bool verbose;
  // Current yaw of the goal position
  double goal_yaw;
  // Real time uses PID controller. If false, will jump to next waypoint
  bool real_time;
  // How long to wait at each waypoint if realtime is set to false
  double move_time;
  // If true, clear all stored waypoints on collision
  bool clear_wp_on_collision;
  bool running;
  ::ros::ServiceServer running_service;
  msr::airlib::MultirotorRpcLibClient *airsimClient;
  Waypoint *current_goal;
};

TrajectoryCallerNode::TrajectoryCallerNode(const ros::NodeHandle &nh,
                                           const ros::NodeHandle &nh_private)
    : nh_(nh), nh_private_(nh_private), followingGoal(false),
      only_move_in_yaw_direction(true), verbose(false), real_time(false),
      move_time(0.1), goal_yaw(0), clear_wp_on_collision(true), running(true) {

  client = nh_.serviceClient<airsim_ros_pkgs::SetLocalPosition>(
      "/local_position_goal");
  sub = nh_private_.subscribe("/command/trajectory", 10,
                              &TrajectoryCallerNode::callback, this);
  odom_sub = nh_private_.subscribe("/odom", 10,
                                   &TrajectoryCallerNode::odomCallback, this);
  collision_sub = nh_private_.subscribe(
      "/collision", 10, &TrajectoryCallerNode::collisionCallback, this);

  gain_sub = nh_private_.subscribe("/gain", 10,
                                   &TrajectoryCallerNode::gainCallback, this);

  waypoint_reached_pub =
      nh_.advertise<embodied_active_learning::waypoint_reached>(
          "waypoint_reached", 100);

  running_service = nh_private_.advertiseService(
      "set_running", &TrajectoryCallerNode::setRunning, this);

  nh_private_.param<bool>("verbose", verbose, true);
  verbose = true;
  nh_private_.param<bool>("move_in_yaw", only_move_in_yaw_direction, false);
  nh_private_.param<bool>("real_time", real_time, true);
  nh_private_.param<bool>("clear_wp_on_collision", clear_wp_on_collision, true);
  nh_private_.param<double>("move_time", move_time, 0.1);

  if (!real_time) {
    // Not real time -> use airsim client to jump to waypoints
    airsimClient = new msr::airlib::MultirotorRpcLibClient;
    airsimClient->enableApiControl(true);
    airsimClient->armDisarm(true);
  }
  current_goal = nullptr;
  std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>> TRAJECTORY CALLER NODE RUNNING " << std::endl;
}

void TrajectoryCallerNode::collisionCallback(const std_msgs::Bool &msg) {
  if (msg.data) {
    if (clear_wp_on_collision) {
      std::wcout
          << "Collision detected in trajectory caller node. Going to clear "
             "stored waypoints"
          << std::endl;
      followingGoal = false;
      // Clear stored points
      std::queue<Waypoint *> empty;
      std::swap(points, empty);
    }
  }
}

void TrajectoryCallerNode::odomCallback(const nav_msgs::Odometry &msg) {
  if (!running)
    return;

  if (!followingGoal) {
    // Not following a reference point. Check if there is a new one
    if (points.empty())
      return;

    // Get next point to track
    current_goal = points.front();

    if (verbose)
      std::cout << "Got new goal point: " << current_goal->pose
                << " will publish it to service" << std::endl;

    // Message for PID controller
    airsim_ros_pkgs::SetLocalPosition srv;
    srv.request.x = current_goal->pose.translation.x;
    srv.request.y = current_goal->pose.translation.y;
    srv.request.z = current_goal->pose.translation.z;
    auto current_position = msg.pose.pose.position;

    if (only_move_in_yaw_direction && !current_goal->isGoal) {
      // We are currently tracking waypoints and not final goalpoints
      // If only move in yaw direction, set yaw of waypoints (except goal point)
      // to moving direction
      auto delta_x = srv.request.x - current_position.x;
      auto delta_y = srv.request.y - current_position.y;
      if (abs(delta_x) + abs(delta_y) <= 0.1) {
        // If we overshoot over the goalpoint, the robot will rotate 180 deg and
        // drive back for a really small amount. Check if overshoot occured
        // (delta really small) and don't turn around in this case
        srv.request.yaw = goal_yaw;
      } else {
        srv.request.yaw = atan2(delta_y, delta_x);
      }

      if (verbose)
        std::cout << "forcing yaw in view direction" << std::endl;
    } else {
      // Rotate to goal yaw
      tf::Quaternion q(
          current_goal->pose.rotation.x, current_goal->pose.rotation.y,
          current_goal->pose.rotation.z, current_goal->pose.rotation.w);
      tf::Matrix3x3 m(q);
      double roll, pitch, yaw;
      m.getRPY(roll, pitch, yaw);
      srv.request.yaw = yaw;
    }
    goal_yaw = srv.request.yaw;
    if (real_time) {
      // Call PID node
      if (client.call(srv)) {
        followingGoal = true;
      } else {
        if (verbose)
          std::cout << "client not ready yet " << std::endl;
      }
    } else {
      // Call airsim client to jump to position
      followingGoal = true;
      ros::Duration(move_time).sleep();
      Pose pose;
      pose.position.x() = srv.request.x;
      pose.position.y() = srv.request.y;
      pose.position.z() = srv.request.z;
      tf::Quaternion q;
      q.setRPY(0, 0, srv.request.yaw);
      pose.orientation.x() = q.x();
      pose.orientation.y() = q.y();
      pose.orientation.z() = q.z();
      pose.orientation.w() = q.w();

      airsimClient->simSetVehiclePose(pose, false, "drone_1");
    }
  } else {
    // Currently following a goalpoint (final point of trajectory, need to
    // orientate yaw). Lets check if we reached it Track the current pose
    auto current_position_ =
        Eigen::Vector3d(msg.pose.pose.position.x, msg.pose.pose.position.y,
                        msg.pose.pose.position.z);

    auto goal_position_ = Eigen::Vector3d(current_goal->pose.translation.x,
                                          current_goal->pose.translation.y,
                                          current_goal->pose.translation.z);

    if ((goal_position_ - current_position_).norm() < 0.1) {
      // Goal position reached, what about orientation?
      double yaw = tf::getYaw(msg.pose.pose.orientation);
      if (defaults::angleDifference(goal_yaw, yaw) < 0.2) {
        // Orientation reached
        if (verbose)
          std::cout << "reached goal point (rotation)" << std::endl;
        // Reached target yaw. Publish a message containing gain information
        embodied_active_learning::waypoint_reached wp_msg;
        wp_msg.reached = true;
        wp_msg.gain = current_goal->gain;
        if (current_goal->isGoal) {
          // Reached Goal
          if (verbose)
            std::cout << "reached a goal point"
                      << "gain value" << current_goal->gain << std::endl;
          waypoint_reached_pub.publish(wp_msg);
        } else {
          if (verbose)
            std::cout << "reached a way point" << std::endl;
        }
        auto *off = points.front();
        points.pop();
        delete off;
        followingGoal = false;
      }
    }
  }
}

bool TrajectoryCallerNode::setRunning(std_srvs::SetBool::Request &req,
                                      std_srvs::SetBool::Response &res) {
  res.success = true;
  if (req.data) {
    running = true;
    std::cout << "Started trajectory caller" << std::endl;
  } else {
    running = false;
    std::cout << "Stopped trajectory caller" << std::endl;
  }
  return true;
}

void TrajectoryCallerNode::gainCallback(const std_msgs::Float32 msg) {
  points.back()->gain = msg.data;
}

void TrajectoryCallerNode::callback(
    const trajectory_msgs::MultiDOFJointTrajectory trajectory) {
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> GOT TRAJECTORY MESSAGE" << std::endl;
  // Add all points of trajectory to point queue
  for (const auto &point : trajectory.points) {
    for (auto pose : point.transforms) {
      if (verbose)
        std::cout << "adding pose to goal" << pose.rotation.y << " "
                  << pose.rotation.w << std::endl;

      auto *pt = new Waypoint;
      pt->pose = pose;
      pt->isGoal = false;
      points.push(pt);
    }
  }
  // Add goal point for yaw tracking as last point
  auto *lastPose = points.back();
  auto *pt = new Waypoint;
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

} // namespace active_3d_planning

int main(int argc, char **argv) {
  ros::init(argc, argv, "trajectory_caller_node");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                 ros::console::levels::Debug);
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  active_3d_planning::TrajectoryCallerNode node(nh, nh_private);
  ros::spin();
  return 0;
}