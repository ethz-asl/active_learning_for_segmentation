"""
Simple Data Acquisitor. Extracts visible camera images + semseg classes at constant frequency
"""

import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image

import time

import os

from cv_bridge import CvBridge, CvBridgeError
import cv2


class ConstantRateDataAcquisitor:

    def __init__(self):
        self.bridge = CvBridge()

        self.rate = rospy.get_param("~data_mode/rate", 0.5)
        self.path = rospy.get_param("~data_mode/path", "/home/rene/thesis/imgs")
        self.period = 1/self.rate

        # TODO change to dynamic topics based on vehicle names etc.
        self._rgb_sub = Subscriber("/airsim/airsim_node/drone_1/front/Scene", Image)
        self._depth_sub = Subscriber("/airsim/airsim_node/drone_1/front/DepthPlanner", Image)
        self._semseg_sub = Subscriber("/airsim/airsim_node/drone_1/front/Segmentation", Image)
        self.last_request = rospy.get_rostime()

        self.path = self.path + "/experiment_" +  str(time.time())
        os.mkdir(self.path)

        ts = ApproximateTimeSynchronizer([self._rgb_sub,self._depth_sub,self._semseg_sub], 10,0.1, allow_headerless = True)
        ts.registerCallback(self.callback)


    def callback(self, rgb_msg, depth_msg, semseg_msg):
        """ Saves te supplied rgb and semseg image as PNGs """

        if  (rospy.get_rostime() - self.last_request).to_sec() < self.period:
            # Too early, go back to sleep :)
            return

        self.last_request = rospy.get_rostime()

        try:
            ts = str(time.time())
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv2.imwrite("{}/{}_rgb.png".format(self.path, ts), cv_image)

            cv_image = self.bridge.imgmsg_to_cv2(semseg_msg, "mono8")
            cv2.imwrite("{}/{}_semseg.png".format(self.path, ts), cv_image)
        except CvBridgeError as e:
            print(e)
