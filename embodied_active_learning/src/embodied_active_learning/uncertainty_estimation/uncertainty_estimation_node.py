#!/usr/bin/env python
"""
Node that takes an RGB input image and predicts semantic classes + uncertainties
"""
# ros
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
import tf

import time
from struct import pack, unpack
import datetime
import torch
import numpy as np
from matplotlib import cm
import cv2

from refinenet.models.resnet import rf_lw50, rf_lw101, rf_lw152
from refinenet.utils.helpers import prepare_img

from embodied_active_learning.utils.utils import get_pc_for_image, depth_to_3d
import embodied_active_learning.utils.online_learning
from embodied_active_learning.uncertainty_estimation.uncertainty_estimator import SimpleSoftMaxEstimator, \
  GroundTruthErrorEstimator, ClusteredUncertaintyEstimator, DynamicThresholdWrapper
import embodied_active_learning.airsim_utils.semantics as semantics
from embodied_active_learning.utils.online_learning import get_online_learning_refinenet
from embodied_active_learning.msg import waypoint_reached

from kimera_interfacer.msg import SyncSemantic

def get_uncertainty_estimator_for_params(params: dict):
  """
  Returns an uncertainty estimator consisting of a segmentation network + ucnertainty estimation
  :param params: Params as they are stored in rosparams
  :return: Uncertainty Estimator
  """
  model = None

  # ------
  # Offline training
  # ------
  if params['network']['name'] == "lightweight-refinenet":
    size = params['network'].get('size', 101)
    classes = params['network'].get('classes', 40)
    pretrained = params['network'].get('pretrained', True)
    rospy.loginfo("Using RefineNet as Semantic Segmentation Network")
    rospy.loginfo(
      "Parameters\n- Size: {}\n- Classes: {}\n- pretrained: {}".format(
        size, classes, pretrained))

    has_cuda = torch.cuda.is_available()
    if size == 50:
      net = rf_lw50(classes, pretrained=pretrained).eval()
    elif size == 101:
      net = rf_lw101(classes, pretrained=pretrained).eval()
    elif size == 152:
      net = rf_lw152(classes, pretrained=pretrained).eval()
    else:
      rospy.logerr("Unkown encoder size {}".format(size))

    if has_cuda:
      net = net.cuda()

    def predict_image(numpy_img, net=net, has_cuda=has_cuda):
      if type(numpy_img) == np.ndarray:
        orig_size = numpy_img.shape[:2][::-1]
        img_torch = torch.tensor(
          prepare_img(numpy_img).transpose(2, 0, 1)[None]).float()
      else:
        img_torch = numpy_img

      if has_cuda:
        img_torch = img_torch.cuda()
      pred = net(img_torch)[0].data.cpu().numpy().transpose(1, 2, 0)
      # Resize image to target prediction
      return cv2.resize(pred, orig_size, interpolation=cv2.INTER_NEAREST)

    model = predict_image

  # ------
  # Online training
  # ------
  elif params['network']['name'] == "online-lightweight-refinenet":
    size = params['network'].get('size', 101)
    classes = params['network'].get('classes', 40)
    save_path = params['network'].get('save_path', None)
    pretrained = params['network'].get('pretrained', True)
    checkpoint = params['network'].get('checkpoint', None)

    rospy.loginfo("Using ONLINE-RefineNet as Semantic Segmentation Network")
    rospy.loginfo(
      "Parameters\n- Size: {}\n- Classes: {}\n- pretrained: {}".format(
        size, classes, pretrained))

    has_cuda = torch.cuda.is_available()
    model_slug = rospy.get_param("/experiment_name", "experiment") + "_" + str(
      datetime.datetime.fromtimestamp(time.time()).strftime("%d_%m__%H_%M_%S"))
    net = get_online_learning_refinenet(size, classes, pretrained, save_path=save_path, model_slug=model_slug,
                                        replay_map=params.get("replay_old_pc", False))

    if checkpoint is not None:
      net.model.load_state_dict(torch.load(checkpoint))
      rospy.loginfo("Loaded network from checkpoint: {}".format(checkpoint))
    if has_cuda:
      net = net.cuda()

    def predict_image(numpy_img, net=net, has_cuda=has_cuda):
      if type(numpy_img) == np.ndarray:
        orig_size = numpy_img.shape[:2][::-1]
        img_torch = torch.tensor(
          prepare_img(numpy_img).transpose(2, 0, 1)[None]).float()
      else:
        orig_size = numpy_img.shape[-2:][::-1]
        img_torch = numpy_img

      if has_cuda:
        img_torch = img_torch.cuda()

      pred = net(img_torch)[0].data.cpu().numpy().transpose(1, 2, 0)

      return cv2.resize(pred, orig_size, interpolation=cv2.INTER_NEAREST)

    model = predict_image

  # ------
  # GMM model
  # ------
  elif params['network']['name'] == "online-lightweight-refinenet-with-uncertainty":
    size = params['network'].get('size', 101)
    classes = params['network'].get('classes', 40)
    save_path = params['network'].get('save_path', None)
    pretrained = params['network'].get('pretrained', True)
    checkpoint = params['network'].get('checkpoint', None)

    rospy.loginfo("Using ONLINE-RefineNet with UNCERTAINTY as Semantic Segmentation Network")
    rospy.loginfo(
      "Parameters\n- Size: {}\n- Classes: {}\n- pretrained: {}".format(
        size, classes, pretrained))

    has_cuda = torch.cuda.is_available()
    model_slug = rospy.get_param("/experiment_name", "experiment") + "_" + str(
      datetime.datetime.fromtimestamp(time.time()).strftime("%d_%m__%H_%M_%S"))
    net = get_online_learning_refinenet(size, classes, pretrained, save_path=save_path, model_slug=model_slug,
                                        with_uncertainty=True)

    if checkpoint is not None:
      net.model.load_state_dict(torch.load(checkpoint))
      rospy.loginfo("Loaded network from checkpoint: {}".format(checkpoint))
    else:
      rospy.logwarn("GMM model has no checkpoint. Clusters will be randomly initialized!")
    if has_cuda:
      net = net.cuda()

    def predict_image(numpy_img, net=net, has_cuda=has_cuda):
      orig_size = numpy_img.shape[:2][::-1]
      img_torch = torch.tensor(
        prepare_img(numpy_img).transpose(2, 0, 1)[None]).float()
      if has_cuda:
        img_torch = img_torch.cuda()

      pred, uncertainty = net(img_torch)
      pred = pred[0].cpu().numpy().transpose(1, 2, 0)
      # Resize image to target prediction
      uncertainty = uncertainty[0].cpu().numpy().transpose(1, 2, 0)

      uncertainty_resizted = np.squeeze(cv2.resize(uncertainty, orig_size, interpolation=cv2.INTER_CUBIC))
      return cv2.resize(pred, orig_size, interpolation=cv2.INTER_NEAREST), uncertainty_resizted

    model = predict_image

  if model is None:
    raise ValueError("Could not find model for specified parameters")

  ####
  # Load Uncertainty Estimator
  ####
  estimator = None
  estimator_params = params.get('method', {})
  estimator_type = estimator_params.get('type', 'softmax')

  if estimator_type == "softmax":
    rospy.loginfo(
      "Creating SimpleSoftMaxEstimator for uncertainty estimation")
    estimator = SimpleSoftMaxEstimator(model,
                                       from_logits=estimator_params.get(
                                         'from_logits', True))
  elif estimator_type == "thresholder_softmax":
    estimator = DynamicThresholdWrapper(SimpleSoftMaxEstimator(model, from_logits=estimator_params.get(
      'from_logits', True)), initial_threshold=estimator_params.get('threshold', 0.8),
                                        quantile=estimator_params.get('quantile', 0.9),
                                        update=estimator_params.get('update', True))
  elif estimator_type == "gt_error":
    rospy.loginfo(
      "Creating GroundTruthError for uncertainty estimation")
    estimator = GroundTruthErrorEstimator(model, params['air_sim_semantics_converter']);
  elif estimator_type == "model_uncertainty":
    rospy.loginfo(
      "Creating Model Uncertainty for uncertainty estimation")
    estimator = DynamicThresholdWrapper(ClusteredUncertaintyEstimator(model),
                                        initial_threshold=estimator_params.get('threshold', 0.8),
                                        quantile=estimator_params.get('quantile', 0.9), max_value=70,
                                        update=estimator_params.get('update', True));
  if estimator is None:
    raise ValueError("Could not find estimator for specified parameters")

  return net, estimator


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
    params = rospy.get_param("/uncertainty")
    self.air_sim_semantics_converter = semantics.AirSimSemanticsConverter(
      rospy.get_param("/semantic_mapping_path",
                      "../../../cfg/airsim/semanticClasses.yaml"))
    params['air_sim_semantics_converter'] = self.air_sim_semantics_converter;
    try:
      self.net, self.uncertainty_estimator = get_uncertainty_estimator_for_params(
        params)
    except ValueError as e:
      rospy.logerr(
        "Could not load uncertainty estimator. Uncertainty estimator NOT running! \n Reason: {}".format(
          str(e)))
      rospy.logerr("Specified parameters: {}".format(str(params)))
      return

    self.running = False
    self._semseg_pub = rospy.Publisher("~/semseg/image",
                                       Image,
                                       queue_size=5)
    self._uncertainty_pub = rospy.Publisher("~/semseg/uncertainty",
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

    self._sync_sem_seg_pub = rospy.Publisher("/sync_semantic",
                                          SyncSemantic,
                                          queue_size=5)


    self.batch_size = params['network'].get('batch_size', 4)
    self.replay_old_pc = params.get('replay_old_pc', False)

    self._rgb_sub = Subscriber("rgbImage", Image)
    self._depth_sub = Subscriber("depthImage", Image)
    self._semseg_gt_sub = Subscriber("semsegGtImage", Image)
    self._camera_sub = Subscriber("cameraInfo", CameraInfo)
    self._odom_sub = Subscriber("odometry", Odometry)

    self.last_request = rospy.get_rostime()
    self.period = 1 / params.get('rate', 2)
    self.num_classes = params['network'].get('classes', 40)

    self.tf_listener = tf.TransformListener()
    self._start_service = rospy.Service("toggle_running", SetBool, self.toggle_running)

    self._point_reached = rospy.Subscriber("/planner/waypoint_reached", waypoint_reached, self.wp_reached)

    self.imgCount = 0
    self.reached_gp = False

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

  def wp_reached(self, data):
    """ Once waypoint is reached, update reached_gp flag to make sure to capture this image for online training"""
    self.reached_gp = True

  def toggle_running(self, req):
    """ start / stops the uncertainty estimator """
    self.running = req.data
    return True, 'running'

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
    if not self.running or not self.reached_gp:
      if not self.running or (rospy.get_rostime() - self.last_request).to_sec() < self.period:
        # Too early, go back to sleep :)
        return
    else:
      rospy.loginfo("Reached waypoint. Going to force network training to capture right image")
      self.reached_gp = False

    self.last_request = rospy.get_rostime()
    # Monitor executing time
    start_time = time.time()

    # Get Image from message data
    img = np.frombuffer(rgb_msg.data, dtype=np.uint8)
    # Depth image
    img_depth = np.frombuffer(depth_msg.data, dtype=np.float32)
    img_gt = np.frombuffer(gt_img_msg.data, dtype=np.uint8)
    img_gt = img_gt.reshape(rgb_msg.height, rgb_msg.width, 3)[:, :, 0]
    # Convert BGR to RGB
    img = img.reshape(rgb_msg.height, rgb_msg.width, 3)[:, :, [2, 1, 0]]
    img_shape = img.shape

    semseg, uncertainty = self.uncertainty_estimator.predict(img, img_gt.copy())
    time_diff = time.time() - start_time
    rospy.loginfo(" ==> segmented images in {:.4f}s, {:.4f} FPs".format(
      time_diff, 1 / time_diff))

    # Publish uncertainty pointcloud with uncertainty as b value
    (x, y, z) = depth_to_3d(img_depth, camera)
    color = (uncertainty * 254).astype(np.uint8).reshape(-1)

    # img_color = self.air_sim_semantics_converter.semantic_prediction_to_nyu_color(semseg).astype(np.uint8)#.reshape(-1)
    # ''' Stack uint8 rgb image into a single float array (efficiently) for ros compatibility '''
    # r = np.ravel(img_color[:, :, 0]).astype(int)
    # g = np.ravel(img_color[:, :, 1]).astype(int)
    # b = np.ravel(img_color[:, :, 2]).astype(int)
    # color = np.left_shift(r, 16) + np.left_shift(g, 8) + b
    packed = pack('%di' % len(color), *color)
    unpacked = unpack('%df' % len(color), packed)
    data = (np.vstack([x, y, z, np.array(unpacked)])).T

    pc_msg = PointCloud2()
    pc_msg.header.frame_id = rgb_msg.header.frame_id
    pc_msg.header.stamp = rgb_msg.header.stamp
    pc_msg.width = data.shape[0]
    pc_msg.height = 1
    pc_msg.fields = [
      PointField('x', 0, PointField.FLOAT32, 1),
      PointField('y', 4, PointField.FLOAT32, 1),
      PointField('z', 8, PointField.FLOAT32, 1),
      PointField('rgb', 12, PointField.FLOAT32, 1)
    ]
    pc_msg.is_bigendian = False
    pc_msg.point_step = 16
    pc_msg.row_step = pc_msg.point_step * pc_msg.width
    pc_msg.is_dense = True
    pc_msg.data = np.float32(data).tostring()
    self._semseg_pc_pub.publish(pc_msg)

    if publish_images:
      # make RGB, use some nice colormaps:
      uncertainty_uint8 = np.uint8(cm.seismic(uncertainty) *
                                   255)[:, :, 0:3]  # Remove alpha channel

      # semseg = (cm.hsv(semseg / self.num_classes) * 255).astype(
      #   np.uint8)[:, :, 0:3]  # Remove alpha channel
      semseg = self.air_sim_semantics_converter.semantic_prediction_to_nyu_color(semseg)
      # Create and publish image message
      semseg_msg = Image()
      semseg_msg.header = rgb_msg.header
      print("SEmseg image shape", semseg.shape)
      semseg_msg.height = img_shape[0]
      semseg_msg.width = img_shape[1]
      semseg_msg.step = rgb_msg.width
      semseg_msg.data = semseg.flatten().tolist()
      semseg_msg.encoding = "rgb8"
      self._semseg_pub.publish(semseg_msg)

      uncertainty_msg = Image()
      uncertainty_msg.header = rgb_msg.header
      uncertainty_msg.height = img_shape[0]
      uncertainty_msg.width = img_shape[1]
      uncertainty_msg.step = rgb_msg.width

      uncertainty_msg.data = uncertainty_uint8.flatten().tolist()
      uncertainty_msg.encoding = "rgb8"
      self._uncertainty_pub.publish(uncertainty_msg)



      sync_sem_msg = SyncSemantic()
      # sync_sem_msg.header = rgb_msg.header
      sync_sem_msg.image = rgb_msg
      sync_sem_msg.sem = semseg_msg
      sync_sem_msg.depth = depth_msg
      self._sync_sem_seg_pub.publish(sync_sem_msg)

    if type(self.net) == embodied_active_learning.utils.online_learning.OnlineLearner:
      # If network is online learner, we need to add training images
      # First downsample image, as otherwise the cuda memory is too much
      factor = 2
      img = cv2.resize(img, dsize=(img.shape[1] // factor, img.shape[0] // factor), interpolation=cv2.INTER_CUBIC)
      img_gt = cv2.resize(img_gt.copy(), dsize=(img_gt.shape[1] // factor, img_gt.shape[0] // factor),
                          interpolation=cv2.INTER_NEAREST)
      # Convert to torch tensor
      img_torch = torch.tensor(prepare_img(img).transpose(2, 0, 1)[None]).float()
      gt_torch = torch.tensor(self.air_sim_semantics_converter.map_infrared_to_nyu(img_gt)).long()
      self.imgCount += 1

      if self.imgCount < 650:
        print("Waiting with online training for burn in period of 650 imgs")
        # Burn in period of 650 images
        return
      if self.imgCount == 650:
        print("reached 650 images. Going to reset planner")
        start_stop_experiment_proxy = rospy.ServiceProxy("/start_stop_experiment", SetBool)
        start_stop_experiment_proxy(True)



      # In case of map replay we also need to store the current pose of th epc
      pose = None
      if self.replay_old_pc:
        try:
          # TODO not hardcode
          (trans, rot) = self.tf_listener.lookupTransform('/drone_1', semseg_msg.header.frame_id,
                                                          semseg_msg.header.stamp)
          pose = (trans, rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
          rospy.logerr("[ERROR] Lookup error for pose of current image!")


      # Add training sample to online net
      self.net.addSample(img_torch, gt_torch, uncertainty_score=np.mean(uncertainty), pose=pose,
                         camera=camera if self.replay_old_pc else None,
                         depth=img_depth.reshape(rgb_msg.height, rgb_msg.width) if self.replay_old_pc else None)
      # Train for one step
      self.net.train(batch_size=self.batch_size)


if __name__ == '__main__':
  rospy.init_node('uncertainty_estimation_node', anonymous=True)
  um = UncertaintyManager()
  rospy.spin()
