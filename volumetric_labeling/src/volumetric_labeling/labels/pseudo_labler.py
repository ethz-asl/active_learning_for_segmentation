from typing import List
import numpy as np
from panoptic_mapping_msgs.srv import RenderCameraImage
from sensor_msgs.msg import Image
from geometry_msgs.msg import Transform
from cv_bridge import CvBridge
import rospy
import time
from embodied_active_learning.online_learning.sample import TrainSample
from embodied_active_learning.utils.config import PseudoLablerConfig
from rospy.service import ServiceException

class PseudoLabler:
  # TODO hardcoded
  def __init__(self, pseudo_labler_config: PseudoLablerConfig, topic = '/mapper/render_camera_view'):
    self.service_proxy = rospy.ServiceProxy(topic, RenderCameraImage)
    self.bridge = CvBridge()
    self.pseudo_labler_config = pseudo_labler_config


  def label_many(self, samples: List[TrainSample], cache_time = 0.5):
    t1 = time.time()
    for sample in samples:
      self.update_labels_for_training_entry(sample, cache_time = cache_time)
    print(f"Requesting Pseudo Labels for {len(samples)} images took {time.time() - t1}")

  def update_labels_for_training_entry(self, sample: TrainSample, cache_time: float = 0.5, verbose: bool = False):
    """

    Args:
      entry:
      cache_time: Time since last requested pseudo label

    Returns:

    """
    if (sample.is_gt_sample):
      rospy.loginfo(f"Skipping sample #{sample.number} since it already has a GT mask.")
      return

    depth_message = sample.depth
    transform = sample.transform
    try:
      if sample.mask is None or rospy.Time.now().to_sec() - sample.last_label_update > cache_time:
        t1 = time.time()
        response = self.service_proxy(True, True, depth_message, transform, depth_message.header.frame_id, self.pseudo_labler_config.weights_method,
                                  depth_message.header.stamp)
        img_msg = response.class_image
        uncertainty_msg = response.uncertainty_image
        mask = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 1)
        # weight = bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')
        weight = np.frombuffer(uncertainty_msg.data, dtype=np.float32).reshape(uncertainty_msg.height, uncertainty_msg.width, 1)
        sample.update_mask(mask)

        # Normalize weight
        weight = 1/np.max(weight) * weight
        # Invert weight
        weight = 1 - weight
        # Normalize summed weight
        weight = 1/np.sum(weight) * weight
        # print("Unique weight values: ", np.unique(weight))
        sample.update_weights(weight)

        sample.last_label_update = rospy.Time.now().to_sec()
        if verbose:
          print(f"Requesting Pseudo Labels for image took {time.time() - t1}")
    except ServiceException as e:
      rospy.logerr("ServiceException while getting pseudo labels: " + e.__str__())
      return False

    return True

  def get_labels_for_pose_and_depth(self, pose, depth_message):
    """

    Args:
      entry:
      cache_time: Time since last requested pseudo label

    Returns:

    """
    # Use Depth, Use Transform, Depth Msg, Transform Msg, Sensor Frame, Stamp
    t1 = time.time()
    transform = Transform()
    transform.translation.x = pose[0][0]
    transform.translation.y = pose[0][1]
    transform.translation.z = pose[0][2]
    transform.rotation.x = pose[1][0]
    transform.rotation.y = pose[1][1]
    transform.rotation.z = pose[1][2]
    transform.rotation.w = pose[1][3]
    resp = self.service_proxy(True, True, depth_message, transform, depth_message.header.frame_id,
                              depth_message.header.stamp)
    img_msg = resp.class_image
    img = np.frombuffer(img_msg.data, dtype=np.uint8)
    img = img.reshape(img_msg.height, img_msg.width, 1)
    pseudo_labels = img
    print(f"Requesting Pseudo Labels for image took {time.time() - t1}")
    return pseudo_labels

  def get_labels_for_depth_image(self, depth_message: Image):
    # Use Depth, Use Transform, Depth Msg, Transform Msg, Sensor Frame, Stamp
    print("ECNODING", depth_message.encoding)
    print("width, height", depth_message.width, depth_message.height)
    resp = self.service_proxy(True, False, depth_message, None, depth_message.header.frame_id,
                              depth_message.header.stamp)
    img_msg = resp.class_image
    img = np.frombuffer(img_msg.data, dtype=np.uint8)
    img = img.reshape(img_msg.height, img_msg.width, 1)
    pseudo_labels = img

    return pseudo_labels
