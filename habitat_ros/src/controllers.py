#!/usr/bin/env python3
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose, Twist
import rospy
import numpy as np


class SimplePidController:
  def __init__(self):
    self.error_x = 0
    self.error_y = 0
    self.error_z = 0
    self.position_x = 0
    self.position_y = 0
    self.position_z = 0

    self.goal_x = 5
    self.goal_y = -8
    self.goal_z = 8
    self.goal_yaw = 0

    self._odom_cb = rospy.Subscriber("odom", Odometry, self.odomCallback)
    self._goal_cb = rospy.Subscriber("goalpoint", Twist, self.goal)

    self._gp_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
    self.cb_t = 0.1
    self.call_time = 0

    self.have_goal = False
    self.prev_x_error = None
    self.prev_y_error = None
    self.prev_ang_error = None

    self.stuck_count = 0
    from airsim_ros_pkgs.srv import SetLocalPosition
    self.prxy = rospy.Service('/local_position_goal', SetLocalPosition, self.srv_cb)
    self.prev_raw_x_err = 0
    self.prev_raw_y_err = 0
    self.prev_raw_ang_err = 0


  def srv_cb(self, req):
    # print("Got request!", req)
    self.goalCB(req.x, req.y, req.z, req.yaw)
    return True, 'ok'

  def goal(self, goal: Twist):
    # print("Got goal:", goal)
    self.goalCB(goal.linear.x, goal.linear.y, goal.linear.z, goal.angular.x)

    # print("errors")
    print(self.position_x - self.goal_x)
    print(self.position_y - self.goal_y)
    print(self.position_z - self.goal_z)


  def goalCB(self, x,y,z, yaw):
    self.have_goal = True
    self.goal_x = x
    self.goal_y = y
    self.goal_z = z
    self.goal_yaw = yaw

  def odomCallback(self, odom: Odometry):
    if rospy.Time.now().secs - self.call_time < self.cb_t or not self.have_goal:
      return

    self.call_time = rospy.Time.now().secs
    self.position_x = odom.pose.pose.position.x
    self.position_y = odom.pose.pose.position.y
    self.position_z = odom.pose.pose.position.z
    self.position_yaw = Rotation.from_quat(
      [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z,
       odom.pose.pose.orientation.w]).as_euler('xyz')[-1]

    def f(x,y):
      import math
      return min(y-x, y-x+2*math.pi, y-x-2*math.pi, key=abs)

    yaw_error = f(self.position_yaw, self.goal_yaw)

    self.ang_err = np.max([np.min([yaw_error, 0.6]), -0.6])
    input = Twist()

    self.raw_x_err = self.position_x - self.goal_x
    self.raw_y_err = self.position_y - self.goal_y
    self.raw_ang_err = self.position_yaw - self.goal_yaw



    self.x_error = np.min([0.8, np.max([(self.position_x - self.goal_x)*1, -0.8])])*1
    self.y_error = np.min([0.8, np.max([(self.position_y - self.goal_y)*1, -0.8])])*1



    # if (yaw_error < 0.001):


    if self.prev_ang_error is None:
      self.prev_x_error = self.x_error
      self.prev_y_error = self.y_error
      self.prev_ang_err = self.ang_err

    # rospy.infora("errors:")
    # print(self.position_x - self.goal_x)
    # print(self.position_y - self.goal_y)
    # print(self.position_z - self.goal_z)

    input.angular.z =  self.ang_err * 1.4 + 0.1* (self.ang_err - self.prev_ang_err)
    input.linear.x = - self.x_error * 1  + 0.1* (self.x_error - self.prev_x_error)
    input.linear.y = - self.x_error  * 1  + 0.1* (self.x_error - self.prev_x_error)
    input.linear.z = - self.y_error * 1 + 0.1 * (self.y_error - self.prev_y_error)
    import math
    if (abs(self.raw_ang_err - self.prev_raw_ang_err) <= 0.01 and abs(self.raw_x_err - self.prev_raw_x_err) <= 0.01 and abs(self.raw_y_err - self.prev_raw_y_err) <= 0.01):
      self.stuck_count += 1
      if self.stuck_count >= 50:
        print("Errors same. Most likely no movemenet cause stuck. Going to add noise")
        import random
        input.angular.z += (random.random() - 0.5)* 0.2
        input.linear.x += (random.random() - 0.5)* 0.2
        input.linear.y += (random.random() - 0.5)* 0.2
        input.linear.z += (random.random() - 0.5)* 0.2
    else:
      self.stuck_count = 0
    # print("Inputs:", input.linear)
    # print("yawrate:", input.angular.z)
    # print("Input:", input.linear)
    self.prev_x_error = self.x_error
    self.prev_y_error = self.y_error
    self.prev_ang_err = self.ang_err

    self.prev_raw_x_err = self.prev_raw_x_err
    self.prev_raw_y_err =  self.prev_raw_y_err
    self.prev_raw_ang_err =  self.prev_raw_ang_err

    self._gp_pub.publish(input)


if __name__ == '__main__':
  rospy.init_node('pid_node', anonymous=True)
  planner = SimplePidController()
  rospy.spin()
