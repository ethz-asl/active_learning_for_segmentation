"""
Simple Data Acquisitor. Extracts visible camera images + semseg classes at constant frequency
"""

import time
import os

import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image
import numpy as np
from PIL import Image as PilImage
from embodied_active_learning.utils.config import DataAcquistorConfig
from embodied_active_learning.utils.airsim.airsim_semantics import AirSimSemanticsConverter


class ConstantRateDataAcquisitor:
  """ Class that Samples Semantic+Depth+RGB images in a constant rate"""

  def __init__(self, config: DataAcquistorConfig, semantic_converter: AirSimSemanticsConverter):
    self.running = False
    self.rate = config.rate
    self.path = config.output_folder
    self.period = 1 / self.rate

    self._rgb_sub = Subscriber("rgbImage", Image)
    self._depth_sub = Subscriber("depthImage", Image)
    self._semseg_sub = Subscriber("semsegImage", Image)

    self.last_request = rospy.get_rostime()

    self.path = self.path + "/" + rospy.get_param("/experiment_name", "experiment") + "_" + str(time.time())
    self.semantic_converter = semantic_converter

    os.mkdir(self.path)

    ts = ApproximateTimeSynchronizer([self._rgb_sub, self._depth_sub, self._semseg_sub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(self.callback)
    self.running = True
    rospy.loginfo("Started ConstantRateDataAcquisitor")

  def callback(self, rgb_msg, depth_msg, semseg_msg):
    """ Saves te supplied rgb and semseg image as PNGs """
    if not self.running or (rospy.get_rostime() - self.last_request).to_sec() < self.period:
      # Too early, go back to sleep :)
      return

    self.last_request = rospy.get_rostime()

    ts = str(time.time())

    img = np.frombuffer(rgb_msg.data, dtype=np.uint8)
    img = img.reshape(rgb_msg.height, rgb_msg.width, 3)[:, :, [2, 1, 0]]
    PilImage.fromarray(img).save("{}/{}_rgb.png".format(self.path, ts))

    # Only take one channel, infrared has 3 channels with same information
    mask = np.frombuffer(semseg_msg.data, dtype=np.uint8).copy()
    mask = mask.reshape(semseg_msg.height, semseg_msg.width, 3)[:, :, 0]
    PilImage.fromarray(self.semantic_converter.map_infrared_to_nyu(mask)).save(
      "{}/{}_mask.png".format(self.path, ts))
