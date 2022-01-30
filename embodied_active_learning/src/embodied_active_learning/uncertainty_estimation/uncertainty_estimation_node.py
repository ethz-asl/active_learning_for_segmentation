#!/usr/bin/env python
"""
Node that takes an RGB input image and predicts semantic classes + uncertainties
"""
from typing import List, Optional, Tuple
from typing import Union

import cv2
import numpy as np
import os
import pickle
import tf
import time
import torch

# Ros
from cv_bridge import CvBridge
import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from std_srvs.srv import SetBool

# Core
from embodied_active_learning_core.online_learning.sample import TrainSample
from embodied_active_learning_core.semseg.models import model_manager
from embodied_active_learning_core.semseg.models.uncertainty_wrapper import UncertaintyFitter
from embodied_active_learning_core.semseg.uncertainty.uncertainty_estimator import GroundTruthErrorEstimator, \
  SimpleSoftMaxEstimator, \
  ClusteredUncertaintyEstimator, DynamicThresholdWrapper, UncertaintyEstimator
from embodied_active_learning_core.utils.pytorch.image_transforms import prepare_img

# This package
from embodied_active_learning.msg import waypoint_reached
import embodied_active_learning.utils.airsim.airsim_semantics as semantics
from embodied_active_learning.online_learning.online_learner import OnlineLearner
from embodied_active_learning.utils.config import Configs, UNCERTAINTY_TYPE_SOFTMAX
from embodied_active_learning_core.utils.utils import batch


def get_uncertainty_estimator_for_config(config: Configs) -> Tuple[any, UncertaintyEstimator]:
  """
  Returns an uncertainty estimator consisting of a segmentation network + ucnertainty estimation
  :param params: Params as they are stored in rosparams
  :return: Uncertainty Estimator
  """
  model = None

  # Get Network
  network_config = config.uncertainty_estimation_config.network_config
  network = model_manager.get_model_for_config(network_config)
  if network_config.name in model_manager.ONLINE_LEARNING_NETWORKS:
    network = OnlineLearner(network, model_manager.get_optimizer_params_for_model(network_config, network), config)

  def predict_image(numpy_img: np.ndarray, net: torch.nn.Module = network,
                    has_cuda: bool = torch.cuda.is_available(), network_config=network_config) \
      -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """ Predicts semseg (+uncertainty) for numpy input. Also normalizes input if needed and resizes output"""
    orig_size = numpy_img.shape[:2][::-1]
    img_torch = torch.tensor(
      prepare_img(numpy_img, normalize=network_config.normalize_imgs).transpose(2, 0, 1)[None]).float()
    if has_cuda:
      img_torch = img_torch.cuda()

    pred = net(img_torch)
    if type(pred) == tuple:  # Model also predicts uncertainty
      return cv2.resize(pred[0][0].data.cpu().numpy().transpose(1, 2, 0), orig_size,
                        interpolation=cv2.INTER_NEAREST), cv2.resize(pred[1][0].data.cpu().numpy().squeeze(), orig_size,
                                                                     interpolation=cv2.INTER_NEAREST)
    else:
      return cv2.resize(pred[0].data.cpu().numpy().transpose(1, 2, 0), orig_size, interpolation=cv2.INTER_NEAREST)

  prediction_fn = predict_image

  ####
  # Load Uncertainty Estimator
  ####
  estimator: Optional[Union[UncertaintyEstimator, DynamicThresholdWrapper]] = None
  uncertainty_config = config.uncertainty_estimation_config.uncertainty_config

  if uncertainty_config.type == UNCERTAINTY_TYPE_SOFTMAX:
    rospy.loginfo("Creating SimpleSoftMaxEstimator for uncertainty estimation")
    estimator = SimpleSoftMaxEstimator(prediction_fn, from_logits=uncertainty_config.from_logits)
  elif uncertainty_config.type == "model_built_in":
    estimator = ClusteredUncertaintyEstimator(prediction_fn)
  elif uncertainty_config.type == "gt_error":
    estimator = GroundTruthErrorEstimator(prediction_fn)
  else:
    raise ValueError(f"Unknown Uncertainty Estimator Type: {uncertainty_config.type}")

  if uncertainty_config.threshold_type == uncertainty_config.THRESHOLD_TYPE_ABSOLUTE:
    estimator = DynamicThresholdWrapper(
      uncertanity_estimator=estimator,
      initial_threshold=uncertainty_config.threshold, quantile=1, update=False)

  elif uncertainty_config.threshold_type == uncertainty_config.THRESHOLD_TYPE_DYNAMIC:
    estimator = DynamicThresholdWrapper(
      uncertanity_estimator=estimator,
      initial_threshold=uncertainty_config.threshold, quantile=uncertainty_config.max, update=True)

  if estimator is None:
    raise ValueError("Could not find estimator for specified parameters")

  if uncertainty_config.type == "model_built_in" and network_config.name in model_manager.ONLINE_LEARNING_NETWORKS:
    # If we use model_built_in this means we need to refit GMM after training
    def gmm_refitting_callback(online_learner: OnlineLearner):
      """ Function that is executed after every learning epoch. Refits GMM + Dynamic Thresholder"""
      if online_learner.train_iter % uncertainty_config.refitting_rate != 0:
        return
      online_learner.start_stop_experiment_proxy(False) # Stop experiment during refitting

      rospy.loginfo("Refitting Uncertainties")

      def train_data_generator():
        """ Iterates over all images in training buffer of online learner and passes them to the function"""
        to_be_used: List[TrainSample] = online_learner.training_buffer.get_all()
        online_learner.model.eval()
        device = next(online_learner.model.parameters()).device
        for b in batch(to_be_used, 4):
          input = []
          for item in b:
            if item.mask is None:
              continue
            entry = online_learner.train_transforms(
              image=item.image,
              mask=item.mask,
              weight=item.mask  # not used
            )
            input.append(entry['image'])

          input = torch.stack(input).to(device)
          yield input.cuda()

      with UncertaintyFitter(online_learner.model, total_features=10000, features_per_batch=500) as fitter:
        for b in train_data_generator():
          fitter(b)

      online_learner.model.eval()
      all_data = np.asarray([])
      for b in train_data_generator():
        uncertainty = online_learner.model(b)[1].cpu().detach().numpy()
        all_data = np.append(all_data, uncertainty.ravel())

      if type(estimator) == DynamicThresholdWrapper:
        estimator.fit(all_data)

    network.post_training_hooks.append(gmm_refitting_callback)

  return network, estimator


class UncertaintyManager:
  """
  Class that publishes uncertainties + semantc segmentation to different topics

  Publishes to:
      ~/semseg/image
      ~/semseg/uncertainty
      ~/semseg/depth
      ~/semseg/points

  Subscribes to:
      rgbImage
      depthImage  <- Currently not needed but might be useful later
      cameraInfo
      odometry
  """

  def __init__(self):
    '''  Initialize ros node and read params '''

    # Parse Params
    self.publish_semseg_pc = rospy.get_param("~publishPc", False)
    self.gt_is_available = rospy.get_param("~gtIsAvailable", True)
    self.config = Configs(rospy.get_param("/experiment_name", "unknown_experiment"))
    self.is_resumed = rospy.get_param("/uncertainty/uncertainty_node/resume_experiment", False)

    # --- ROS Publishers and Subscribers
    self._semseg_pub = rospy.Publisher("~/semseg/image", Image, queue_size=5)
    self._uncertainty_pub = rospy.Publisher("~/semseg/uncertainty", Image, queue_size=5)
    self._depth_pub = rospy.Publisher("~/semseg/depth", Image, queue_size=5)
    self._rgb_pub = rospy.Publisher("~/semseg/rgb", Image, queue_size=5)
    self._semseg_camera_pub = rospy.Publisher("~/semseg/cam", CameraInfo, queue_size=5)
    if self.publish_semseg_pc:
      self._semseg_pc_pub = rospy.Publisher("~/semseg/points", PointCloud2, queue_size=5)

    self._rgb_sub = Subscriber("rgbImage", Image)
    self._depth_sub = Subscriber("depthImage", Image)
    self._semseg_gt_sub = Subscriber("semsegGtImage", Image)
    self._camera_sub = Subscriber("cameraInfo", CameraInfo)
    self._odom_sub = Subscriber("odometry", Odometry)
    self._point_reached = rospy.Subscriber("/mapper/waypoint_reached", waypoint_reached, self.wp_reached)
    # --- ROS Publishers and Subscribers

    self._start_service = rospy.Service("toggle_running", SetBool, self.toggle_running)

    self.air_sim_semantics_converter: semantics.AirSimSemanticsConverter = semantics.AirSimSemanticsConverter(
      self.config.experiment_config.semantic_mapping_path)

    if self.is_resumed:
      step = int(rospy.get_param("/step_to_resume", "step_000").replace("step_", ""))
      self.folder_to_resume = rospy.get_param("/folder_to_resume")
      checkpoint = os.path.join(self.folder_to_resume, "checkpoints", f"best_iteration_{step}.pth")
      if not os.path.exists(checkpoint):
        print(f"Did not find checkpoint to resume experiment. File:{checkpoint}")
        rospy.signal_shutdown("Checkpoint not found")
        exit()

      self.config.uncertainty_estimation_config.network_config.checkpoint = checkpoint

    self.net, self.uncertainty_estimator = get_uncertainty_estimator_for_config(self.config)
    self.running = False
    self.last_request = rospy.get_rostime()
    self.period = 1 / self.config.uncertainty_estimation_config.rate
    self.tf_listener = tf.TransformListener()

    self.imgCount = 0
    self.reached_gp = False
    self.gp_gain = 0.0
    self.last_training_sample: Optional[TrainSample] = None

    self.uncertainty_buffer = []
    self.dyn_th_refitting_interval = 10
    self.uncertainty_buffer_values_per_img = 500

    self.cv_bridge = CvBridge()

    subs_to_sync = [self._rgb_sub, self._depth_sub, self._camera_sub]
    if self.gt_is_available:
      subs_to_sync.append(self._semseg_gt_sub)

    ts = ApproximateTimeSynchronizer(subs_to_sync,
                                     queue_size=20,
                                     slop=0.5,
                                     allow_headerless=True)

    ts.registerCallback(self.callback if self.gt_is_available else self.no_gt_callback)

    rospy.loginfo("Uncertainty estimator running")

    if self.config.uncertainty_estimation_config.network_config.name in model_manager.ONLINE_LEARNING_NETWORKS:
      # Online Learning
      rospy.Timer(rospy.Duration(1 / self.config.online_learning_config.rate), self.train_callback)  # Training Callback
      rospy.loginfo("Training timer running. Using Rate: {}".format(self.config.online_learning_config.rate))
      self.started = True
      if self.is_resumed:
        self.resume_from_folder()

  def wp_reached(self, data):
    """ Once waypoint is reached, update reached_gp flag to make sure to capture this image for online training"""
    self.reached_gp = True
    self.gp_gain = data.gain

  def toggle_running(self, req):
    """ start / stops the uncertainty estimator """
    self.running = req.data
    return True, 'running'

  def train_callback(self, event):
    if self.config.uncertainty_estimation_config.network_config.name in model_manager.ONLINE_LEARNING_NETWORKS:
      self.net.train(batch_size=self.config.online_learning_config.batch_size)

      if self.last_training_sample is not None:
        self.net.addSample(self.last_training_sample)
        rospy.loginfo(
          f"Added sample #{self.last_training_sample.number}. Num Samples in buffer: {len(self.net.training_buffer)}")
        self.last_training_sample = None

  def resume_from_folder(self):
    self.started = False
    rospy.loginfo("Resuming from folder:", self.folder_to_resume)
    step_param = rospy.get_param("/step_to_resume")
    step = int(step_param.replace("step_", ""))
    rospy.loginfo("Step to resume:", step)
    online_path = os.path.join(self.folder_to_resume, "online_training", step_param)
    if not os.path.exists(online_path):
      rospy.logerr("Did not find folder:", online_path)
      rospy.signal_shutdown("Folder not found")
      exit()

    for entry in os.listdir(online_path):
      train_item = pickle.load(open(os.path.join(online_path, entry), "rb"))
      self.net.addSample(train_item)
    self.net.train_iter = step + 1
    self.net.samples_seen += 1  # So training is not triggered

    print("Loaded train samples.")

    for cb in self.net.post_training_hooks:
      # rospy.wait_for_service("/start_stop_experiment",30)
      self.start_stop_follower = rospy.ServiceProxy("/airsim/trajectory_caller_node/set_running", SetBool)
      # self.start_stop_experiment_proxy(False)
      cb(self.net)
      self.start_stop_follower(True)
      self.started = True

  def no_gt_callback(self, rgb_msg, depth_msg, camera):
    """ Pseudo callback if no gt is available. Simply sends zeros as segementation mask """
    gt_img = Image()
    gt_img.header = depth_msg.header
    gt_img.height = depth_msg.height
    gt_img.width = depth_msg.width
    gt_img.encoding = "mono8"
    gt_img.step = gt_img.width
    gt_img.data = np.zeros(depth_msg.height, depth_msg.width).astype(np.uint8).flatten().tolist()
    self.callback(rgb_msg, depth_msg, camera, gt_img)

  def callback(self, rgb_msg: Image, depth_msg: Image, camera: CameraInfo, gt_img_msg: Image, publish_images=True):
    """
    Publishes semantic segmentation + Pointcloud to ropstopics

    NOTE: If publishing frequency of messages is not a multiple of the period, uncertainty images won't be published at exactly the specified frequency
    TODO Maybe change implementation to store messages with each callback and process them using rospy.Timer(), but currently not needed

    :param rgb_msg: Image message with RGB image
    :param depth_msg:  Image message with depth image (MUST BE CV_32FC1)
    :param camera: Camera info
    :param publish_images: if image messages should be published. If false only PC is published
    :return: Nothing
    """

    is_waypoint = False
    if not self.started or not self.running or not self.reached_gp:
      if not self.started or not self.running or (rospy.get_rostime() - self.last_request).to_sec() < self.period:
        # Too early, go back to sleep :)
        return
    else:
      rospy.loginfo("Reached waypoint. Going to force network training to capture right image")
      self.reached_gp = False
      is_waypoint = True

    self.last_request = rospy.get_rostime()
    # Monitor executing time
    start_time = time.time()

    # Get Image from message data
    img = np.frombuffer(rgb_msg.data, dtype=np.uint8)
    # Depth image
    img_depth = np.frombuffer(depth_msg.data, dtype=np.float32)
    img_gt = np.frombuffer(gt_img_msg.data, dtype=np.uint8)
    img_gt = img_gt.reshape(rgb_msg.height, rgb_msg.width, -1)[:, :, 0]
    if (img_gt.shape[-1] == 3):  # Got airsim image, lets map it
      rospy.loginfo_once(
        "Got groundtruth image with 3 Channels. Assuming this is from airsim, performin arisim mapping")
      img_gt = self.air_sim_semantics_converter.map_infrared_to_nyu(img_gt.copy())
    # Convert BGR to RGB
    img = img.reshape(rgb_msg.height, rgb_msg.width, 3)[:, :, [2, 1, 0]]
    img_shape = img.shape

    semseg, uncertainty = self.uncertainty_estimator.predict(img, img_gt.copy())

    time_diff = time.time() - start_time
    rospy.loginfo_throttle(10, " ==> segmented images in {:.4f}s, {:.4f} FPs. Image Number: {}".format(
      time_diff, 1 / time_diff, self.imgCount))

    if publish_images:
      # Create and publish image message
      semseg_msg = Image()
      semseg_msg.header = rgb_msg.header
      semseg_msg.height = img_shape[0]
      semseg_msg.width = img_shape[1]
      semseg_msg.step = rgb_msg.width
      semseg_msg.data = semseg.flatten().tolist()
      semseg_msg.encoding = "mono8"

      uncertainty_msg = self.cv_bridge.cv2_to_imgmsg(uncertainty.astype(np.float32), "32FC1")
      uncertainty_msg.header = rgb_msg.header

      # publish
      self._semseg_pub.publish(semseg_msg)
      self._uncertainty_pub.publish(uncertainty_msg)
      self._depth_pub.publish(depth_msg)
      self._rgb_pub.publish(rgb_msg)

      if self.config.uncertainty_estimation_config.network_config.name in model_manager.ONLINE_LEARNING_NETWORKS:
        # If network is online learner, we need to add training images
        self.imgCount += 1

        if self.imgCount < self.config.online_learning_config.late_start_n_imgs:
          print(f"Waiting with online training for burn in period of {self.config.online_learning_config.late_start_n_imgs} imgs")
          # Burn in period of 650 images
          return

        pose = None
        try:
          (trans, rot) = self.tf_listener.lookupTransform('/world', semseg_msg.header.frame_id, semseg_msg.header.stamp)
          pose = (trans, rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
          rospy.logerr("[ERROR] Lookup error for pose of current image!")

        sample = TrainSample(image=img, depth=depth_msg, number=self.imgCount, mask=None, uncertainty=0,
                             pose=pose, camera=camera, is_waypoint=is_waypoint,
                             waypoint_gain=self.gp_gain if is_waypoint else 0)

        # Set GT Mask for statistics
        if img_gt is not None:
          sample.update_gt_mask(img_gt)

        if is_waypoint:  # Always add waypoints directly to training
          self.net.addSample(sample)
        else:  # Otherwise save it as last seen sample that will be added by training callback
          self.last_training_sample = sample

        if self.config.log_config.log_poses:
          out_folder = self.config.log_config.get_pose_log_folder()
          with open(os.path.join(out_folder, "poses.csv"), "a+") as f:
            f.write(
              f"{sample.transform.translation.x},{sample.transform.translation.y},{sample.transform.translation.z}," +
              f"{sample.transform.rotation.x},{sample.transform.rotation.y},{sample.transform.rotation.z},{sample.transform.rotation.w}\n")

        print("checking", self.net.samples_seen, " with ", self.config.online_learning_config.bundle_size)
        # Check if epoch has been reached
        if self.net.samples_seen > 0 and self.net.samples_seen % self.config.online_learning_config.bundle_size == 0:
          self.net.epoch_size_reached = True
          # self.config.online_learning_config.bundle_size = 100 #resetting to 100
        #
        # if self.net.samples_seen > 0 and self.net.samples_seen % 80 == 0: # refitting step
        #   self.net.start_stop_experiment_proxy(False)
        #   for c in self.net.post_training_hooks:
        #     c(self.net)
        #   self.net.start_stop_experiment_proxy(True)

if __name__ == '__main__':
  rospy.init_node('uncertainty_estimation_node', anonymous=True)
  um = UncertaintyManager()
  rospy.spin()
