#!/usr/bin/env python3
import random
import math

# ros
import tf
import rospy
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from geometry_msgs.msg import Transform, Quaternion, Point, Twist


class BumpAndRotatePlanner:
  """
  Simple implementation of a bump and rotate mapper.
  Driver in yaw direction until it hits something, then rotates for a random amount
  """

  def __init__(self):
    '''  Initialize ros node and read params '''
    self.current_goal_x = 0
    self.current_goal_y = 0
    self.current_goal_yaw = random.random() * 2 * math.pi - math.pi
    self.running = False
    self.collision_count = 0
    self.cnt = 0
    self.force_replan = True
    self.collided = False
    self.current_pose = None

    self._start_service = rospy.Service(
      "planner_node/toggle_running", SetBool, self.toggle_running)
    self._odom_cb = rospy.Subscriber("odometry", Odometry, self.callback)
    self._collision_sub = rospy.Subscriber(
      "/airsim/drone_1/collision", Bool, self.collision_callback)

    self._trajectory_pub = rospy.Publisher(
      "command/trajectory", MultiDOFJointTrajectory)
    rospy.loginfo("_bump_and_rotate_planner running")

  def toggle_running(self, req):
    """ Starts mapper """
    self.running = req.data
    return True, 'running'

  def publish_goal(self):
    """ Publishes a goal to the trajectory follower node"""
    msg = MultiDOFJointTrajectory()
    msg.header.frame_id = "world"
    q = tf.transformations.quaternion_from_euler(
      0, 0, self.current_goal_yaw)
    transforms = Transform(
      translation=Point(
        self.current_goal_x,
        self.current_goal_y,
        0),
      rotation=Quaternion(
        q[0],
        q[1],
        q[2],
        q[3]))
    msg.points.append(
      MultiDOFJointTrajectoryPoint(
        [transforms], [
          Twist()], [
          Twist()], rospy.Time(0)))
    self._trajectory_pub.publish(msg)

  def sample_and_publish_next_trajectory(self, step=0.2):
    """ Samples next waypoint in yaw direction """
    self.current_goal_x += step * math.cos(self.current_goal_yaw)
    self.current_goal_y += step * math.sin(self.current_goal_yaw)
    self.publish_goal()

  def collision_callback(self, collision_message, step=0.6):
    """ On collision, drive back and then rotate for a random angle"""
    if not self.running:
      return

    if not self.collided:
      # First collision -> Drive backwards a little bit
      q = self.current_pose.orientation
      yaw = tf.transformations.euler_from_quaternion(
        (q.x, q.y, q.z, q.w))[2]
      self.current_goal_x -= step * math.cos(yaw)
      self.current_goal_y -= step * math.sin(yaw)
      self.collided = True

    rospy.loginfo("[COLLIDED]. Going to publish new goal {:.4f},{:.4f},{:.4f}".format(
      self.current_goal_x, self.current_goal_y, self.current_goal_yaw))
    self.publish_goal()
    self.collision_count += 1

    if self.collision_count >= 20:
      # Severe collision error (Stuck somewhere). Start driving backwards again to try to fix it
      rospy.logwarn(
        "[COLLIDED COUNT] Collision count is bigger than 20. Going to rerun backtracker again")

      self.collided = False
      self.force_replan = True
    if self.collision_count >= 100:
      # We are doomed. Publish origin as goal point and hope for the best, that we get there somehow
      rospy.logwarn("Severe collision error. Going to move back to origin")
      self.collision_count = 0
      self.current_goal_x = 0
      self.current_goal_y = 0
      self.current_goal_yaw = 0
      self.force_replan = True

  def callback(self, odom_msg):
    if not self.running:
      return

    self.current_pose = odom_msg.pose.pose
    x = self.current_pose.position.x
    y = self.current_pose.position.y

    if self.force_replan or abs(
        y - self.current_goal_y) + abs(x - self.current_goal_x) < 0.1:
      # reached waypoint
      if self.collided:
        # Sample random yaw if collided
        self.current_goal_yaw = random.random() * 2 * math.pi - math.pi
      else:
        self.sample_and_publish_next_trajectory()
        self.collision_count = 0

      self.force_replan = False
      self.collided = False
      self.collision_count = 0


if __name__ == '__main__':
  rospy.init_node('bump_and_rotate_planner_node', anonymous=True)
  planner = BumpAndRotatePlanner()
  rospy.spin()
