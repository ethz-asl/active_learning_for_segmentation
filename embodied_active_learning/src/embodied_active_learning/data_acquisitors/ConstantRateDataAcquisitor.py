"""
Simple Data Acquisitor. Extracts visible camera images + semseg classes at constant frequency
"""

import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import time
import os

import sys
# Imports that get messed up by ros python 2.7 imports
python27_imports = [p for p in sys.path if "2.7" in p]
sys.path = [p for p in sys.path if "2.7" not in p]
import cv2
sys.path.extend(python27_imports)


class ConstantRateDataAcquisitor:

    def __init__(self):
        self.bridge = CvBridge()
        params = rospy.get_param("/data_generation", {})

        self.rate = params.get("rate", 1)
        self.path = params.get("output_folder", "/tmp")
        self.period = 1 / self.rate

        self._rgb_sub = Subscriber("rgbImage", Image)
        self._depth_sub = Subscriber("depthImage", Image)
        self._semseg_sub = Subscriber("semsegImage", Image)

        self.last_request = rospy.get_rostime()

        self.path = self.path + "/experiment_" + str(time.time())
        os.mkdir(self.path)

        ts = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub, self._semseg_sub],
            10,
            0.1,
            allow_headerless=True)
        ts.registerCallback(self.callback)
        rospy.loginfo("Started ConstantRateDataAcquisitor")

    def callback(self, rgb_msg, depth_msg, semseg_msg):
        """ Saves te supplied rgb and semseg image as PNGs """

        if (rospy.get_rostime() - self.last_request).to_sec() < self.period:
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
