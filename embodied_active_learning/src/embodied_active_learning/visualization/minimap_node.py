#!/usr/bin/env python
"""
Publishes a minimap with the robot pose as Image
"""
import time
import pickle

# ros
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf
from PIL import Image as PILImage

import numpy as np
import cv2
from std_msgs.msg import Bool


class MinimapManager:

  def __init__(self):
    '''  Initialize ros node and read params '''
    self.current_pose = None

    self._odom_cb = rospy.Subscriber("odometry", Odometry, self.odom_cb)
    self._minimap_pub = rospy.Publisher("minimap", Image, queue_size=10)

    self.save_trajectory = False
    self.output_path = "/home/rene/thesis/trajectories/"
    self.startup_time = str(time.time())
    minimap_path = rospy.get_param("/minimap/path", None)

    if minimap_path is None:
      rospy.logerr("[MinimapManager] Minimap path was not specified!")
      return

    try:
      map_struct = pickle.load(open(minimap_path, "rb"))
    except Exception as e:
      rospy.logerr("[MinimapManager] Could not open minimap {}".format(
        str(e)))

    self.map_rgb = map_struct['map'].astype(np.uint8)

    if len(self.map_rgb.shape) == 2 or self.map_rgb.shape[-1] == 1:
      rospy.loginfo("Provided map was only one channel")
      self.map_rgb = (map_struct['map'] > 0).astype(np.uint8)
      # Convert to 3 channel image
      self.map_rgb = np.stack(
        [(map_struct['map'] == 0) * 255 for _ in range(3)],
        axis=-1).astype(np.uint8)

    self.length_per_pixel = map_struct['dimensions'][
      'lengthPerPixel']  # Conversion from pixel to meters in unreal
    self.top_start, self.left_start = map_struct['start'][
                                        'top'], map_struct['start']['left']  # Start position of the drone
    self.top_lim, self.left_lim = self.map_rgb.shape[0:2]
    self.bridge = CvBridge()
    self.top_start = self.top_start
    self.left_start = self.left_start
    self.name = rospy.get_param("/experiment_name", "experiment") + "_" + str(time.time())
    if self.save_trajectory:
      self.trajectory_plot_map = self.map_rgb.copy()
      self.last_pose = None
      self._image_captured_sub = rospy.Subscriber("/image_captured", Bool, self.add_pose_to_map)

    rospy.Timer(rospy.Duration(1), self.publish_mini_map)

  def odom_cb(self, msg):
    """ Saves current pose from odometry"""
    self.current_pose = msg.pose.pose

  def add_pose_to_map(self, msg):
    if msg.data and self.last_pose is not None:
      if self.current_pose is not None:
        # Draw arrow on map
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        q = self.current_pose.orientation
        yaw = tf.transformations.euler_from_quaternion(
          (q.x, q.y, q.z, q.w))[2]
        direction = np.asarray([np.sin(yaw), np.cos(yaw)])
        length = 30

        top_coord = int(y / self.length_per_pixel) + self.top_start
        left_coord = int(x / self.length_per_pixel) + self.left_start

        if top_coord < 0 or left_coord < 0 or top_coord >= self.top_lim or left_coord >= self.left_lim:
          rospy.logwarn("Got invalid coordinate for minimap")
          return
        start = (top_coord, left_coord)
        endpoint = start + (direction * length).astype(int)
        self.trajectory_plot_map = cv2.arrowedLine(self.trajectory_plot_map, (start[1], start[0]),
                                                   (endpoint[1], endpoint[0]), (0, 0, 255),
                                                   2,
                                                   tipLength=0.7)

        # Draw connection between points
        x_prev = self.last_pose.position.x
        y_prev = self.last_pose.position.y

        top_coord_prev = int(y_prev / self.length_per_pixel) + self.top_start
        left_coord_prev = int(x_prev / self.length_per_pixel) + self.left_start

        self.trajectory_plot_map = cv2.line(self.trajectory_plot_map, (left_coord_prev, top_coord_prev),
                                            (left_coord, top_coord), (255, 0, 0), thickness=1)
        PILImage.fromarray(self.trajectory_plot_map).save("{}/{}_trajectoryMap.png".format(self.output_path, self.name))

    self.last_pose = self.current_pose

  def publish_mini_map(self, event):
    """ Publishes the minimap with the robot pose as Image"""
    if self.current_pose is not None:
      map = self.map_rgb.copy()
      # Draw arrow on map
      x = self.current_pose.position.x
      y = self.current_pose.position.y
      q = self.current_pose.orientation
      yaw = tf.transformations.euler_from_quaternion(
        (q.x, q.y, q.z, q.w))[2]

      with open("{}/{}_trajectory.csv".format(self.output_path, self.name), "a") as f:
        f.write("{},{},{}\n".format(x, y, yaw))

      direction = np.asarray([np.sin(yaw), np.cos(yaw)])
      length = 30
      top_coord = int(y / self.length_per_pixel) + self.top_start
      left_coord = int(x / self.length_per_pixel) + self.left_start
      if top_coord < 0 or left_coord < 0 or top_coord >= self.top_lim or left_coord >= self.left_lim:
        rospy.logwarn("Got invalid coordinate for minimap")
        return
      start = (top_coord, left_coord)
      endpoint = start + (direction * length).astype(int)
      map = cv2.arrowedLine(map, (start[1], start[0]),
                            (endpoint[1], endpoint[0]), (0, 0, 255),
                            5,
                            tipLength=1)
      # Publish Map Image
      self._minimap_pub.publish(
        self.bridge.cv2_to_imgmsg(map, encoding='rgb8'))


if __name__ == '__main__':
  rospy.init_node('minimap_node', anonymous=True)
  ed = MinimapManager()
  rospy.spin()
