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
from std_msgs.msg import Bool
from embodied_active_learning.msg import waypoint_reached


class GoalpointsDataAcquisitor:
  """ Class that Samples Semantic+Depth+RGB images in a constant rate"""

  def __init__(self, params, semantic_converter):
    self.path = params.get("output_folder", "/tmp")

    self._rgb_sub = Subscriber("rgbImage", Image)
    self._depth_sub = Subscriber("depthImage", Image)
    self._semseg_sub = Subscriber("semsegImage", Image)

    self._point_reached = rospy.Subscriber("/planner/waypoint_reached", waypoint_reached, self.set_image_request)

    self.image_requested = False

    self.path = self.path + "/" + rospy.get_param("/experiment_name", "experiment") + "_" + str(time.time())
    self.semantic_converter = semantic_converter

    os.mkdir(self.path)

    ts = ApproximateTimeSynchronizer(
      [self._rgb_sub, self._depth_sub, self._semseg_sub],
      10,
      0.1,
      allow_headerless=True)
    ts.registerCallback(self.callback)
    self.capture_pub = rospy.Publisher("/image_captured_goal", Bool, queue_size=10)
    rospy.loginfo("Started GoalpointsDataAcquisitor")

  def set_image_request(self, msg):
    if msg.reached:
      print("Requesting image for goalpoint.", msg.gain)
      self.image_gain = msg.gain
      self.image_requested = True

  def callback(self, rgb_msg, depth_msg, semseg_msg):
    """ Saves te supplied rgb and semseg image as PNGs """
    if not self.image_requested:
      return

    self.image_requested = False

    b = Bool()
    b.data = True
    self.capture_pub.publish(b)
    print("saving image")
    ts = str(time.time())
    img = np.frombuffer(rgb_msg.data, dtype=np.uint8)
    img = img.reshape(rgb_msg.height, rgb_msg.width, 3)[:, :, [2, 1, 0]]
    PilImage.fromarray(img).save("{}/{}_rgb.png".format(self.path, ts))

    # Only take one channel, infrared has 3 channels with same information
    mask = np.frombuffer(semseg_msg.data, dtype=np.uint8).copy()
    mask = mask.reshape(semseg_msg.height, semseg_msg.width, 3)[:, :, 0]
    PilImage.fromarray(self.semantic_converter.map_infrared_to_nyu(mask)).save(
      "{}/{}_mask.png".format(self.path, ts))

    # Append gain info to order images by gain later
    with open("{}/gain_info.txt".format(self.path), "a") as f:
      f.write("{}/{}_mask.png,{};\n".format(self.path, ts, self.image_gain))
