#!/usr/bin/env python
"""
Node that takes an RGB input image and predicts semantic classes + uncertainties
"""
# ros
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_srvs.srv import SetBool
import tf
from cv_bridge import CvBridge
from typing import List, Optional, Tuple
import time
import torch
import numpy as np
import yaml
import cv2
import os

from embodied_active_learning.utils.utils import depth_to_3d
import embodied_active_learning.online_learning.online_learning
from embodied_active_learning.uncertainty_estimation.uncertainty_estimator import SimpleSoftMaxEstimator, \
  ClusteredUncertaintyEstimator, DynamicThresholdWrapper, UncertaintyEstimator
import embodied_active_learning.utils.airsim_semantics as semantics
from embodied_active_learning.msg import waypoint_reached

from embodied_active_learning.pseudo_labels.pseudo_labler import PseudoLabler

from embodied_active_learning.online_learning.sample import TrainSample

from embodied_active_learning.utils.config import Configs, UNCERTAINTY_TYPE_SOFTMAX

from embodied_active_learning.utils.pytorch_utils import prepare_img as normalized_prepare_img
from embodied_active_learning.semseg.models.model_manager import get_model_for_config
from embodied_active_learning.semseg.models.uncertainty_wrapper import UncertaintyFitter

# from embodied_active_learning.online_learning.online_learning import create_optimizer, get_encoder_and_decoder_params
from embodied_active_learning.utils.pytorch_utils import create_optimizer, get_encoder_and_decoder_params

def get_uncertainty_estimator_for_config(config: Configs) -> Tuple[any, UncertaintyEstimator]:
  """
  Returns an uncertainty estimator consisting of a segmentation network + ucnertainty estimation
  :param params: Params as they are stored in rosparams
  :return: Uncertainty Estimator
  """
  model = None

  # uncertainty_estimation_config

  def prepare_img(img):
    if config.online_learning_config.normalize_imgs:
      return normalized_prepare_img(img)
    else:
      return img / 255

  network_config = config.uncertainty_estimation_config.network_config
  network = get_model_for_config(network_config)
  # post_training_hooks
  if (config.uncertainty_estimation_config.network_config.name in ["online-lightweight-refinenet",
                                                                   "online-clustered-lightweight-refinenet"]):

    if config.uncertainty_estimation_config.network_config.name == "online-clustered-lightweight-refinenet":  # TODO if using uncertainty
      enc_params, dec_params = get_encoder_and_decoder_params(network.base_model)
    else:
      enc_params, dec_params = get_encoder_and_decoder_params(network)

    optimisers = [
      create_optimizer(
        optim_type=network_config.encoder.optim_type,
        parameters=enc_params,
        lr=network_config.encoder.lr
      ),
      create_optimizer(
        optim_type=network_config.decoder.optim_type,
        parameters=dec_params,
        lr=network_config.decoder.lr
      )
    ]

    network = embodied_active_learning.online_learning.online_learning.OnlineLearner(network, optimisers, config,
                                                                                     refitting_callback=None)

  def predict_image(numpy_img: np.ndarray, net: any = network,
                    has_cuda: bool = torch.cuda.is_available()):  # -> Optional[np.ndarray, Tuple[np.ndarray,np.ndarray]]:
    orig_size = numpy_img.shape[:2][::-1]
    img_torch = torch.tensor(
      prepare_img(numpy_img).transpose(2, 0, 1)[None]).float()
    if has_cuda:
      img_torch = img_torch.cuda()

    pred = net(img_torch)
    if type(pred) == tuple:  # Model also predicts uncertainty
      return cv2.resize(pred[0][0].data.cpu().numpy().transpose(1, 2, 0), orig_size,
                        interpolation=cv2.INTER_NEAREST), cv2.resize(pred[1][0].data.cpu().numpy().squeeze(), orig_size,
                                                                     interpolation=cv2.INTER_NEAREST)
    else:
      return cv2.resize(pred[0].data.cpu().numpy().transpose(1, 2, 0), orig_size, interpolation=cv2.INTER_NEAREST)

    # # Resize image to target prediction
    # return cv2.resize(pred[0].data.cpu().numpy().transpose(1, 2, 0), orig_size, interpolation=cv2.INTER_NEAREST)

  prediction_fn = predict_image

  ####
  # Load Uncertainty Estimator
  ####
  estimator: Optional[UncertaintyEstimator] = None
  uncertainty_config = config.uncertainty_estimation_config.uncertainty_config
  if uncertainty_config.type == UNCERTAINTY_TYPE_SOFTMAX:
    rospy.loginfo(
      "Creating SimpleSoftMaxEstimator for uncertainty estimation")
    estimator = SimpleSoftMaxEstimator(prediction_fn, from_logits=uncertainty_config.from_logits)
    # TODO new config type
  elif uncertainty_config.type == "softmax_static_threshold":  # TODO hardcoded

    # def __init__(self, uncertanity_estimator: UncertaintyEstimator, initial_threshold=0.2, quantile=0.90, update=True,
    #              max_value=1):
    estimator = DynamicThresholdWrapper(
      uncertanity_estimator=SimpleSoftMaxEstimator(prediction_fn, from_logits=uncertainty_config.from_logits),
      initial_threshold=uncertainty_config.threshold, quantile=1, update=False)
  elif uncertainty_config.type == "model_built_in":
    estimator = DynamicThresholdWrapper(ClusteredUncertaintyEstimator(prediction_fn),
                                        initial_threshold=uncertainty_config.threshold,
                                        quantile=uncertainty_config.quantile, max_value=uncertainty_config.max,
                                        update=True)

    from embodied_active_learning.utils.pytorch_utils import batch

    def refitting_callback(online_learner: embodied_active_learning.online_learning.online_learning.OnlineLearner):
      print("in refitting callback function")
      t1 = time.time()
      with UncertaintyFitter(online_learner.model, total_features=10000, features_per_batch=1000) as fitter:
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
          fitter(input.cuda())



      # Model fitted. going to calculate max values dynamically.
      to_be_used: List[TrainSample] = online_learner.training_buffer.get_all()
      online_learner.model.eval()
      device = next(online_learner.model.parameters()).device
      all_data = np.asarray([])
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
        uncertainty = online_learner.model(input)[1].cpu().detach().numpy()
        print("all data", all_data)
        all_data = np.append(all_data, uncertainty.ravel())
      print(f"Fitting step took: {time.time() - t1}s")
      print("Fitting estimator")
      print("all data", all_data)
      estimator.fit(all_data)
    network.post_training_hooks.append(refitting_callback)

    # estimator = GroundTruthErrorEstimator(prediction_fn)
  # elif uncertainty_config.type == "thresholder_softmax":
  #   estimator = DynamicThresholdWrapper(SimpleSoftMaxEstimator(model, from_logits=estimator_params.get(
  #     'from_logits', True)), initial_threshold=estimator_params.get('threshold', 0.8),
  #                                       quantile=estimator_params.get('quantile', 0.9),
  #                                       update=estimator_params.get('update', True))
  # elif uncertainty_config.type == "gt_error":
  #   rospy.loginfo(
  #     "Creating GroundTruthError for uncertainty estimation")
  #   estimator = GroundTruthErrorEstimator(model, params['air_sim_semantics_converter']);
  # elif uncertainty_config.type == "model_uncertainty":
  #   rospy.loginfo(
  #     "Creating Model Uncertainty for uncertainty estimation")
  #   estimator = DynamicThresholdWrapper(ClusteredUncertaintyEstimator(model),
  #                                       initial_threshold=estimator_params.get('threshold', 0.8),
  #                                       quantile=estimator_params.get('quantile', 0.9), max_value=70,
  #                                       update=estimator_params.get('update', True));
  if estimator is None:
    raise ValueError("Could not find estimator for specified parameters")

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

    self.config = Configs(rospy.get_param("/experiment_name", "unknown_experiment"))

    # params = rospy.get_param("/uncertainty")
    self.air_sim_semantics_converter: semantics.AirSimSemanticsConverter = semantics.AirSimSemanticsConverter(
      self.config.experiment_config.semantic_mapping_path)
    self.config.semantics_converter = self.air_sim_semantics_converter


    self.is_resumed = rospy.get_param("/uncertainty/uncertainty_node/resume_experiment", False)
    self.folder_to_resume = None
    step_param = rospy.get_param("/step_to_resume", "step_000")
    step = int(step_param.replace("step_", ""))

    if self.is_resumed:
        self.folder_to_resume = rospy.get_param("/folder_to_resume")
        checkpoint = os.path.join(self.folder_to_resume, "checkpoints", f"best_iteration_{step}.pth")
        if not os.path.exists(checkpoint):
          print("Did not find checkpoint: ", checkpoint)
          rospy.signal_shutdown("Checkpoint not found")
          exit()

        self.config.uncertainty_estimation_config.network_config.checkpoint = checkpoint

    self.net, self.uncertainty_estimator = get_uncertainty_estimator_for_config(self.config)

    self.running = False
    # --- ROS Publishers and Subscribers
    self._semseg_pub = rospy.Publisher("~/semseg/image",
                                       Image,
                                       queue_size=5)
    self._uncertainty_pub = rospy.Publisher("~/semseg/uncertainty",
                                            Image,
                                            queue_size=5)
    self._depth_pub = rospy.Publisher("~/semseg/depth",
                                      Image,
                                      queue_size=5)
    self._rgb_pub = rospy.Publisher("~/semseg/rgb",
                                    Image,
                                    queue_size=5)
    self._semseg_depth_pub = rospy.Publisher("~/semseg/depth",
                                             Image,
                                             queue_size=5)
    self._semseg_camera_pub = rospy.Publisher("~/semseg/cam",
                                              CameraInfo,
                                              queue_size=5)
    self._semseg_pc_pub = rospy.Publisher("~/semseg/points",
                                          PointCloud2,
                                          queue_size=5)
    self._rgb_sub = Subscriber("rgbImage", Image)
    self._depth_sub = Subscriber("depthImage", Image)
    self._semseg_gt_sub = Subscriber("semsegGtImage", Image)
    self._camera_sub = Subscriber("cameraInfo", CameraInfo)
    self._odom_sub = Subscriber("odometry", Odometry)
    # --- ROS Publishers and Subscribers

    self.pseudo_labler = PseudoLabler(self.config.pseudo_labler_config)
    self.last_request = rospy.get_rostime()
    self.period = 1 / self.config.uncertainty_estimation_config.rate

    self.tf_listener = tf.TransformListener()
    self._start_service = rospy.Service("toggle_running", SetBool, self.toggle_running)

    self._point_reached = rospy.Subscriber("/mapper/waypoint_reached", waypoint_reached, self.wp_reached)

    self.imgCount = 0
    self.reached_gp = False
    self.gp_gain = 0.0
    self.last_training_sample: Optional[TrainSample] = None

    self.cv_bridge = CvBridge()

    # If uncertainty estimator is dynamic threshold wrapper, set callbacks to refit it
    # TODO this is kind of ugly, find a better, cleaner way
    if type(self.uncertainty_estimator) == DynamicThresholdWrapper:
      self.net.refitting_callback = lambda u_list, self=self: self.uncertainty_estimator.fit(u_list)
      self.net.threshold_image = self.uncertainty_estimator.threshold_image
      self.net.uncertainty_callback = lambda x: self.uncertainty_estimator.predict(x, None)

    ts = ApproximateTimeSynchronizer(
      [self._rgb_sub, self._depth_sub, self._camera_sub, self._semseg_gt_sub],
      queue_size=20,
      slop=0.5,
      allow_headerless=True)
    ts.registerCallback(self.callback)

    rospy.loginfo("Uncertainty estimator running")

    if type(self.net) == embodied_active_learning.online_learning.online_learning.OnlineLearner:
      rospy.Timer(rospy.Duration(1 / self.config.online_learning_config.rate), self.train_callback)
      rospy.loginfo("Training timer running. Using Rate: {}".format(self.config.online_learning_config.rate))

      print("*" * 20)
      print("*" * 20)
      print("*" * 20)
      #TODO hardcoded
      self.started = False
      if self.is_resumed:
        print("Resuming from folder:", self.folder_to_resume)
        step_param = rospy.get_param("/step_to_resume")
        step = int(step_param.replace("step_",""))
        print("Step to resume:", step)
        online_path = os.path.join(self.folder_to_resume, "online_training", step_param)
        if not os.path.exists(online_path):
          print("Did not find folder:", online_path)
          print("Shutting down!")
          rospy.signal_shutdown("Folder not found")
          exit()

        print("Num Images:", len(os.listdir(online_path)))

        for entry in os.listdir(online_path):
          import pickle
          train_item = pickle.load(open(os.path.join(online_path, entry), "rb"))
          self.net.addSample(train_item)
        self.net.train_iter = step + 1
        self.net.samples_seen+= 1 # So training is not triggered
        print("Loaded train samples.")

        print("*"*20)
        print("*"*20)

        for cb in self.net.post_training_hooks:
          # rospy.wait_for_service("/start_stop_experiment",30)
          self.start_stop_follower = rospy.ServiceProxy("/airsim/trajectory_caller_node/set_running", SetBool)
          # self.start_stop_experiment_proxy(False)
          cb(self.net)
          self.start_stop_follower(True)
          self.started = True
      else:
        self.started = True


    # Log Config
    log_folder = self.config.log_config.get_log_folder()
    with open(os.path.join(log_folder, "config.txt"), "w") as f:
      f.write(str(self.config) + "\n")
    with open(os.path.join(log_folder, "params.yaml"), "w") as f:
      # Fix since reospy.get_params("/") does not work
      all_params = np.unique(np.asarray([name.split("/")[1] for name in rospy.get_param_names()]))
      p_dict = {}
      for p in all_params:
        p_dict[p] = rospy.get_param("/" + p)
      f.write(yaml.dump(p_dict, default_flow_style=False))

  def wp_reached(self, data):
    """ Once waypoint is reached, update reached_gp flag to make sure to capture this image for online training"""
    self.reached_gp = True
    self.gp_gain = data.gain

  def toggle_running(self, req):
    """ start / stops the uncertainty estimator """
    self.running = req.data
    return True, 'running'

  def train_callback(self, event):
    if type(self.net) == embodied_active_learning.online_learning.online_learning.OnlineLearner:
      self.net.train(batch_size=self.config.online_learning_config.batch_size)

      if self.last_training_sample is not None:
        self.net.addSample(self.last_training_sample)
        print(
          f"Added sample #{self.last_training_sample.number}. Num Samples in buffer: {len(self.net.training_buffer)}")
        self.last_training_sample = None

  def callback(self, rgb_msg, depth_msg, camera, gt_img_msg, publish_images=True):
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
      print("GP VALUE:", self.gp_gain)
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
    img_gt = img_gt.reshape(rgb_msg.height, rgb_msg.width, 3)[:, :, 0]
    img_gt = self.air_sim_semantics_converter.map_infrared_to_nyu(img_gt.copy())
    # Convert BGR to RGB
    img = img.reshape(rgb_msg.height, rgb_msg.width, 3)[:, :, [2, 1, 0]]
    img_shape = img.shape

    semseg, uncertainty = self.uncertainty_estimator.predict(img, img_gt.copy())

    time_diff = time.time() - start_time
    rospy.loginfo(" ==> segmented images in {:.4f}s, {:.4f} FPs. Image Number: {}".format(
      time_diff, 1 / time_diff, self.imgCount))

    # Publish uncertainty pointcloud with uncertainty as b value
    (x, y, z) = depth_to_3d(img_depth, camera)
    color = (uncertainty * 254).astype(np.uint8).reshape(-1)

    # img_color = self.air_sim_semantics_converter.semantic_prediction_to_nyu_color(semseg).astype(np.uint8)#.reshape(-1)
    # ''' Stack uint8 rgb image into a single float array (efficiently) for ros compatibility '''
    # r = np.ravel(img_color[:, :, 0]).astype(int)
    # g = np.ravel(img_color[:, :, 1]).astype(int)
    # b = np.ravel(img_color[:, :, 2]).astype(int)
    # color = np.left_shift(r, 16) + np.left_shift(g, 8) + b
    # packed = pack('%di' % len(color), *color)
    # unpacked = unpack('%df' % len(color), packed)
    # data = (np.vstack([x, y, z, np.array(unpacked)])).T
    #
    # pc_msg = PointCloud2()
    # pc_msg.header.frame_id = rgb_msg.header.frame_id
    # pc_msg.header.stamp = rgb_msg.header.stamp
    # pc_msg.width = data.shape[0]
    # pc_msg.height = 1
    # pc_msg.fields = [
    #   PointField('x', 0, PointField.FLOAT32, 1),
    #   PointField('y', 4, PointField.FLOAT32, 1),
    #   PointField('z', 8, PointField.FLOAT32, 1),
    #   PointField('rgb', 12, PointField.FLOAT32, 1)
    # ]
    # pc_msg.is_bigendian = False
    # pc_msg.point_step = 16
    # pc_msg.row_step = pc_msg.point_step * pc_msg.width
    # pc_msg.is_dense = True
    # pc_msg.data = np.float32(data).tostring()
    # self._semseg_pc_pub.publish(pc_msg)

    if publish_images:
      # make RGB, use some nice colormaps:
      # uncertainty_uint8 = np.uint8(cm.seismic(uncertainty) *
      #                              255)[:, :, 0:3]  # Remove alpha channel

      # semseg = (cm.hsv(semseg / self.num_classes) * 255).astype(
      #   np.uint8)[:, :, 0:3]  # Remove alpha channel
      # semseg = self.air_sim_semantics_converter.semantic_prediction_to_nyu_color(semseg)
      # Create and publish image message
      semseg_msg = Image()
      semseg_msg.header = rgb_msg.header
      # print("SEmseg image shape", semseg.shape)
      semseg_msg.height = img_shape[0]
      semseg_msg.width = img_shape[1]
      semseg_msg.step = rgb_msg.width
      semseg_msg.data = semseg.flatten().tolist()
      semseg_msg.encoding = "mono8"
      # Publish for panoptic mapper
      self._semseg_pub.publish(semseg_msg)
      self._depth_pub.publish(depth_msg)
      self._rgb_pub.publish(rgb_msg)

      uncertainty_msg = self.cv_bridge.cv2_to_imgmsg(uncertainty.astype(np.float32), "32FC1")
      uncertainty_msg.header = rgb_msg.header
      self._uncertainty_pub.publish(uncertainty_msg)

      # uncertainty_msg = Image()
      # uncertainty_msg.header = rgb_msg.header
      # uncertainty_msg.height = img_shape[0]
      # uncertainty_msg.width = img_shape[1]
      # uncertainty_msg.step = rgb_msg.width
      #
      # uncertainty_msg.data = uncertainty_uint8.flatten().tolist()
      # uncertainty_msg.encoding = "rgb8"
      # self._uncertainty_pub.publish(uncertainty_msg)
      if type(self.net) == embodied_active_learning.online_learning.online_learning.OnlineLearner:
        # If network is online learner, we need to add training images
        # First downsample image, as otherwise the cuda memory is too much
        #
        # factor = 2
        # img = cv2.resize(img, dsize=(img.shape[1] // factor, img.shape[0] // factor), interpolation=cv2.INTER_CUBIC)
        # img_gt = cv2.resize(img_gt.copy(), dsize=(img_gt.shape[1] // factor, img_gt.shape[0] // factor),
        #                     interpolation=cv2.INTER_NEAREST)
        # # Convert to torch tensor
        # img_torch = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float()
        # gt_torch = torch.tensor(self.air_sim_semantics_converter.map_infrared_to_nyu(img_gt)).long()
        self.imgCount += 1

        # if self.imgCount < 200:
        #   print("Waiting with online training for burn in period of 200 imgs")
        #   # Burn in period of 650 images
        #   return
        # if self.imgCount == 650:
        #   print("reached 650 images. Going to reset mapper")
        #   start_stop_experiment_proxy = rospy.ServiceProxy("/start_stop_experiment", SetBool)
        #   start_stop_experiment_proxy(True)
        #

        # In case of map replay we also need to store the current pose of th epc
        pose = None
        try:
          # TODO not hardcode
          (trans, rot) = self.tf_listener.lookupTransform('/world', semseg_msg.header.frame_id,
                                                          semseg_msg.header.stamp)
          pose = (trans, rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
          rospy.logerr("[ERROR] Lookup error for pose of current image!")

        normalize = True

        # if normalize:
        #   img = prepare_img(img)
        # else:
        #   img = img / 255

        sample = TrainSample(image=img, depth=depth_msg, number=self.imgCount, mask=None, uncertainty=0,
                             pose=pose, camera=camera, is_waypoint=is_waypoint, waypoint_gain=self.gp_gain if is_waypoint else 0)

        # Set GT Mask for statistics
        if img_gt is not None:
          sample.update_gt_mask(img_gt)

        # elif self.imgCount % 3 == 1: # Only train every 3rd image
        #   self.net.train(batch_size=self.batch_size)
        # if self.imgCount % 3 == 2: # Only train every 2 image
        if is_waypoint:
          print("FORCE ADDING WAYPOINT")
          self.net.addSample(sample)
        else:
          self.last_training_sample = sample
        # if self.last_training_sample:
        #   self.net.addSample(sample)

        if self.config.log_config.log_poses:
          out_folder = self.config.log_config.get_pose_log_folder()
          with open(os.path.join(out_folder, "poses.csv"), "a+") as f:
            f.write(
              f"{sample.transform.translation.x},{sample.transform.translation.y},{sample.transform.translation.z}," +
              f"{sample.transform.rotation.x},{sample.transform.rotation.y},{sample.transform.rotation.z},{sample.transform.rotation.w}\n")

        print("samples", self.net.samples_seen, self.config.online_learning_config.epoch_size)
        if self.net.samples_seen > 0 and self.net.samples_seen % self.config.online_learning_config.epoch_size == 0:

          self.net.epoch_size_reached = True

          if self.config.log_config.log_maps:
            dump_path = self.config.log_config.get_map_dump_folder()
            dataset_number = 0
            while os.path.exists(os.path.join(dump_path, f"step_{dataset_number:03}")):
              dataset_number += 1

            os.mkdir(os.path.join(dump_path, f"step_{dataset_number:03}"))
            # save_map_proxy = rospy.ServiceProxy("/mapper/planner_node/save_map", SaveLoadMap)
            # save_map_proxy(os.path.join(dump_path, f"step_{dataset_number:03}", "full_map.panmap"))
            print("Going to extract full map as pc to:",
                  os.path.join(dump_path, f"step_{dataset_number:03}", "current_map.pcd"))
            from panoptic_mapping_msgs.srv import SaveLoadMap
            #TODO hardcoded
            save_map_as_pc_proxy = rospy.ServiceProxy("/planner/planner_node/save_pointcloud", SaveLoadMap)
            save_map_as_pc_proxy(os.path.join(dump_path, f"step_{dataset_number:03}", "current_map.pcd"))
            save_map = rospy.ServiceProxy("/planner/planner_node/save_map", SaveLoadMap)
            save_map(os.path.join(dump_path, f"step_{dataset_number:03}", "current_map.panmap"))

        # def add_sample(event):
        #   print("Adding sample and trainging")
        #   pseudo_labels = self.pseudo_labler.get_labels_for_pose_and_depth(pose, depth_msg)
        #   print("GT contained:", np.unique(gt_torch.detach().numpy()))
        #   img_pseudo = cv2.resize(pseudo_labels, dsize=(img_gt.shape[1] // factor, img_gt.shape[0] // factor),
        #                       interpolation=cv2.INTER_NEAREST)
        #   print("Pseudo contained:", np.unique(img_pseudo))
        #   pseudo_torch = torch.tensor(img_pseudo).long()
        #
        #   # Add training sample to online net
        #   self.net.addSample(img_torch, pseudo_torch, uncertainty_score=np.mean(uncertainty), pose=pose,
        #                      camera=camera if self.replay_old_pc else None,
        #                      depth=img_depth.reshape(rgb_msg.height, rgb_msg.width) if self.replay_old_pc else None)
        #   # Train for one step
        #   self.net.train(batch_size=self.batch_size)
        #
        # rospy.Timer(rospy.Duration(0.2), add_sample, oneshot=True)


if __name__ == '__main__':
  rospy.init_node('uncertainty_estimation_node', anonymous=True)
  um = UncertaintyManager()
  rospy.spin()
