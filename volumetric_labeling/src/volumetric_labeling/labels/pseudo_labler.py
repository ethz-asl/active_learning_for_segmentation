""" Requests pseudo labels from panoptic map """
from typing import List
import time
import numpy as np

import rospy
from rospy.service import ServiceException
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from embodied_active_learning_core.online_learning.sample import TrainSample

from panoptic_mapping_msgs.srv import RenderCameraImage


class PseudoLabler:
  """ Class that requests pseudo label by projecting the labels of the panoptic map """

  def __init__(self, weights_method: str = "uncertainty", topic='/planner/planner_node/render_camera_view'):
    self.service_proxy = rospy.ServiceProxy(topic, RenderCameraImage)
    self.bridge = CvBridge()
    self.weights_method = weights_method

  def label_many(self, samples: List[TrainSample], cache_time=0.5):
    """ Requests labels for all images provided as samples """

    t1 = time.time()
    for sample in samples:
      self.update_labels_for_training_entry(sample, cache_time=cache_time)
    print(f"Requesting Pseudo Labels for {len(samples)} images took {time.time() - t1}")

  def update_labels_for_training_entry(self, sample: TrainSample, cache_time: float = 0.5, verbose: bool = False):
    """
    Args:
      sample: Train Sample to annotate
      cache_time: If entry was annotated during the last cache_time, don't reproject pseudo labels.
      verbose: verbosity
    Returns: Nothing, updates mask of sample
    """

    if sample.is_gt_sample:  # e.g. if training sample is from replay buffer
      if verbose:
        rospy.loginfo(f"Skipping sample #{sample.number} since it already has a GT mask.")
      return

    depth_message = sample.depth
    transform = sample.transform
    try:
      if sample.mask is None or rospy.Time.now().to_sec() - sample.last_label_update > cache_time:
        t1 = time.time()
        response = self.service_proxy(True, True, depth_message, transform, depth_message.header.frame_id,
                                      self.weights_method,
                                      depth_message.header.stamp)
        img_msg = response.class_image
        uncertainty_msg = response.uncertainty_image
        mask = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 1)
        sample.update_mask(mask)
        weight = np.frombuffer(uncertainty_msg.data, dtype=np.float32).reshape(uncertainty_msg.height,
                                                                               uncertainty_msg.width, 1)
        sample.update_weights(weight)

        sample.last_label_update = rospy.Time.now().to_sec()
        if verbose:
          print(f"Requesting Pseudo Labels for image took {time.time() - t1}")

    except ServiceException as e:
      rospy.logerr("ServiceException while getting pseudo labels: " + e.__str__())
      return False

    return True

  def get_labels_for_depth_image(self, depth_message: Image):
    """ Extracts frame id and timestamp from image and requests label for this camera position + depth image """
    resp = self.service_proxy(True, False, depth_message, None, depth_message.header.frame_id,
                              depth_message.header.stamp)
    img_msg = resp.class_image
    img = np.frombuffer(img_msg.data, dtype=np.uint8)
    return img.reshape(img_msg.height, img_msg.width, 1)
