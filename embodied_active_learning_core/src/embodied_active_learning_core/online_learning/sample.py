""" Class for one training sample"""
import numpy as np
import cv2

from geometry_msgs.msg import Transform
from sensor_msgs.msg import Image, CameraInfo


class TrainSample:
  image: np.ndarray
  depth: Image
  mask: np.ndarray
  number: int
  uncertainty: float
  transform: Transform
  camera: CameraInfo
  last_label_update: float
  is_gt_sample: bool
  weights: np.ndarray
  gt_mask: np.ndarray
  waypoint_gain: float
  is_waypoint: bool

  def __init__(self, image: np.ndarray, depth: Image = None, number: int = -1, mask: np.ndarray = None,
               uncertainty: float = 0, pose: tuple = None, camera: CameraInfo = None, downscaling_factor=1,
               is_gt_sample=False, weights=None, gt_mask=None, waypoint_gain=0.0, is_waypoint=False):
    """ Datastructure used to store training data with additional information (e.g. pseudo labels, camera position uncertainty etc.)
      :args
        image: Input image
        depth: Depth image
        number: Unique image number
        mask: Annotation Mask for this sample
        uncertainty: Average uncertainty score of this image
        pose: (Translation, Rotation Quaternion) pose information of camera for this image
        downscaling_factor: How much to downscale images (to not use too much memory)
        is_gt_sample: Flag to show if data is groundtruth data (e.g. image was humanly annotated or cames from source domain buffer with gt labels)
        weights: Custom weights for each pixel (e.g. pseudo labels certainty)
        waypoint_gain: Associated gain of waypoint when capturing this image
        is_waypoint: Flag to set if image was taken at a requested waypoint from the planner or just during the mission
    """
    # Resize image and set to tensor
    self.image = cv2.resize(image.copy(),
                            dsize=(image.shape[1] // downscaling_factor, image.shape[0] // downscaling_factor),
                            interpolation=cv2.INTER_CUBIC)
    self.depth = depth
    self.number = number
    self.mask = mask
    self.uncertainty = uncertainty
    self.pose = pose
    self.downscaling_factor = downscaling_factor
    self.last_label_update = 0
    self.is_gt_sample = is_gt_sample  # flag to tell the pseudo labler to not relabel this image

    if pose is not None:
      transform = Transform()
      transform.translation.x = pose[0][0]
      transform.translation.y = pose[0][1]
      transform.translation.z = pose[0][2]
      transform.rotation.x = pose[1][0]
      transform.rotation.y = pose[1][1]
      transform.rotation.z = pose[1][2]
      transform.rotation.w = pose[1][3]
      self.transform = transform

    self.camera = camera
    self.weights = weights
    self.gt_mask = gt_mask
    self.waypoint_gain = waypoint_gain
    self.is_waypoint = is_waypoint

  def update_mask(self, mask: np.ndarray):
    """ updates pseudo labels  """
    # Resize image and set to tensor
    self.mask: np.ndarray = cv2.resize(mask.copy(),
                                       dsize=(mask.shape[1] // self.downscaling_factor,
                                              mask.shape[0] // self.downscaling_factor),
                                       interpolation=cv2.INTER_NEAREST)

    if self.mask.shape != self.image.shape[:-1]:
      print(f"[WARN] mask and image shapes do not match! Mask shape: {self.mask.shape} Image shape: {self.image.shape}")

  def update_gt_mask(self, mask: np.ndarray):
    """ Updates gt mask """
    # Resize image and set to tensor
    self.gt_mask: np.ndarray = cv2.resize(mask.copy(),
                                          dsize=(mask.shape[1] // self.downscaling_factor,
                                                 mask.shape[0] // self.downscaling_factor),
                                          interpolation=cv2.INTER_NEAREST)

    if self.gt_mask.shape != self.image.shape[:-1]:
      print(
        f"[WARN] gt_mask and image shapes do not match! Mask shape: {self.mask.shape} Image shape: {self.image.shape}")

  def update_weights(self, weights: np.ndarray):
    """ Updates weights """
    # Resize image and set to tensor
    self.weights: np.ndarray = cv2.resize(weights.copy(),
                                          dsize=(weights.shape[1] // self.downscaling_factor,
                                                 weights.shape[0] // self.downscaling_factor),
                                          interpolation=cv2.INTER_NEAREST)

    if self.weights.shape != self.image.shape[:-1]:
      print(
        f"[WARN] weights and image shapes do not match! Mask shape: {self.weights.shape} Image shape: {self.image.shape}")
