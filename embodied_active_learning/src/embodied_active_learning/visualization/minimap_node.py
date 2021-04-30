#!/usr/bin/env python2
"""
Publishes a minimap with the robot pose as Image
"""

# ros
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
import tf
import numpy as np
import pickle
from cv_bridge import CvBridge
import cv2


class MinimapManager:

    def __init__(self):
        '''  Initialize ros node and read params '''
        self.currentPose = None

        self._odom_cb = rospy.Subscriber("odometry", Odometry, self.odomCb)
        self._minimap_pub = rospy.Publisher("minimap", Image, queue_size=10)
        minimapPath = rospy.get_param("/minimap/path", None)
        if minimapPath is None:
            rospy.logerr("[MinimapManager] Minimap path was not specified!")
            return

        try:
            map_struct = pickle.load(open(minimapPath, "rb"))
        except Exception as e:
            rospy.logerr("[MinimapManager] Could not open minimap {}".format(
                str(e)))

        self.map_rgb = map_struct['map'].astype(np.uint8)  # Binary map

        if len(self.map_rgb.shape) == 2 or self.map_rgb.shape[-1] == 1:
            rospy.loginfo("Provided map only was one channel")
            # Convert to 3 channel image
            self.map_rgb = np.stack(
                [(map_struct['map'] == 0) * 255 for _ in range(3)],
                axis=-1).astype(np.uint8)

        self.lengthPerPixel = map_struct['dimensions'][
            'lengthPerPixel']  # Conversion from pixel to meters in unreal
        self.top_start, self.left_start = map_struct['start'][
            'top'], map_struct['start']['left']  # Start position of the drone
        self.top_lim, self.left_lim = self.map_rgb.shape[0:2]
        self.bridge = CvBridge()

        rospy.Timer(rospy.Duration(1), self.publishMiniMap)

    def odomCb(self, msg):
        """ Saves current pose from odometry"""
        self.currentPose = msg.pose.pose

    def publishMiniMap(self, event):
        """ Publishes the minimap with the robot pose as Image"""
        if self.currentPose is not None:
            map = self.map_rgb.copy()
            # Draw arrow on map
            x = self.currentPose.position.x
            y = self.currentPose.position.y
            q = self.currentPose.orientation
            yaw = tf.transformations.euler_from_quaternion(
                (q.x, q.y, q.z, q.w))[2]
            direction = np.asarray([np.sin(yaw), np.cos(yaw)])
            length = 30
            top_coord = int(y / self.lengthPerPixel) + self.top_start
            left_coord = int(x / self.lengthPerPixel) + self.left_start
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
