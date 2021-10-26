import os.path
import random

import refinenet.models.uncertainty_utils
import rospy
from std_srvs.srv import SetBool
from std_msgs.msg import Int16

from struct import pack, unpack

import numpy as np
import rospy
import torch
import torch.nn as nn
from embodied_active_learning.utils.pytorch_utils import batch
import re

from matplotlib import cm

import torch.nn.functional as F
import time
from sensor_msgs.msg import PointCloud2, PointField
from embodied_active_learning.uncertainty_estimation.uncertainty_estimator import SimpleSoftMaxEstimator

from refinenet.models.uncertainty_utils import UncertaintyFitter
from refinenet.models.refinenet_with_uncertainty import get_uncertainty_net
from sensor_msgs.msg import Image
from embodied_active_learning.utils.utils import get_pc_for_image, importance_sampling_by_uncertainty
import tf
import cv2


def get_encoder_and_decoder_params(model):
  """Filter model parameters into two groups: encoder and decoder."""
  enc_params = []
  dec_params = []
  for k, v in model.named_parameters():
    if bool(re.match(".*conv1.*|.*bn1.*|.*layer.*", k)):
      enc_params.append(v)
    else:
      dec_params.append(v)
  return enc_params, dec_params


def create_optimizer(optim_type, parameters, **kwargs):
  if optim_type.lower() == "sgd":
    optim = torch.optim.SGD
  elif optim_type.lower() == "adam":
    optim = torch.optim.Adam
  else:
    raise ValueError(
      "Optim {} is not supported. "
      "Only supports 'SGD' and 'Adam' for now.".format(optim_type)
    )
  return optim(parameters, **kwargs)


def get_online_learning_refinenet(size, classes=40, pretrained=True, params={}, save_path="", model_slug="",
                                  with_uncertainty=False, refitting_callback=lambda b: None,
                                  replay_map=False) -> OnlineLearner:
  """ Wraps on online learning wrapper around a light weight refininet
  Args:
    size: encoder size of the refinenet
    classes: how many classes (nyu = 40)
    pretrained: use nyu pretrained model
    params: additional paramse (learning rate, weight decay,...)
    save_path: Where to save the checkpoints
    model_slug: Short identifiere to append to the checkpoints
    with_uncertainty: Use GMM-uncertainty net or only refinenet
    refitting_callback: Callback that gets called when the model is refitting its uncertanties or a better threshold was found
    replay_map: Wether or not to replay map after a certain number of iterations

  Returns:
  """
  if with_uncertainty:
    net = get_uncertainty_net(size=size, num_classes=40, n_feature_for_uncertainty=64, n_components=20,
                              covariance_type="tied", reg_covar=1e-6)
  else:
    from refinenet.models.resnet import rf_lw50, rf_lw101, rf_lw152

    net = None
    if size == 50:
      net = rf_lw50(classes, pretrained=pretrained)
    elif size == 101:
      net = rf_lw101(classes, pretrained=pretrained)
    elif size == 152:
      net = rf_lw152(classes, pretrained=pretrained)
    else:
      print("Unkown encoder size {}".format(size))
      return None

  enc_optim_type = params.get("enc_optim_type", "sgd")
  enc_lr = params.get("enc_lr", 0.00001)
  enc_weight_decay = params.get("enc_weight_decay", 0)
  enc_momentum = params.get("enc_momentum", 0.9)

  dec_optim_type = params.get("dec_optim_type", "sgd")
  dec_lr = params.get("dec_lr", 0.008 * 0.5)
  dec_weight_decay = params.get("dec_weight_decay", 0)
  dec_momentum = params.get("dec_momentum", 0.9)

  if with_uncertainty:
    enc_params, dec_params = get_encoder_and_decoder_params(net.base_model)
  else:
    enc_params, dec_params = get_encoder_and_decoder_params(net)

  optimisers = [
    create_optimizer(
      optim_type=enc_optim_type,
      parameters=enc_params,
      lr=enc_lr,
      weight_decay=enc_weight_decay,
      momentum=enc_momentum,
    ),
    create_optimizer(
      optim_type=dec_optim_type,
      parameters=dec_params,
      lr=dec_lr,
      weight_decay=dec_weight_decay,
      momentum=dec_momentum,
    )
  ]

  return OnlineLearner(net, optimisers, save_path=save_path, model_slug=model_slug,
                       refitting_callback=refitting_callback, replay_map=replay_map)


class OnlineLearner:
  """
   Class that wraps around a torch model providin replay buffer etc.
  """

  def __init__(self, model, optimisers, training_loss=nn.CrossEntropyLoss(), save_frequency=100, save_path="",
               model_slug="", replacement_strategy="RANDOM", sample_strategy="RANDOM",
               refitting_callback=lambda b: None,
               replay_map=False, max_buffer_length=500, replay_batch_length=11):
    """
    Constructor of the OnlineLearner model.
    Should be called using the get_online_learning_refinenet() function

    Args:
      model: Pytorch model that should be trained
      optimisers:  Optimisers
      training_loss:  Training loss to use
      save_frequency:  In which frequency to save checkpoints
      save_path: Folder to save checkpoints
      model_slug: Short model identifier that will be part of the checkpoint name
      replacement_strategy: How to replace old buffer elements (RANDOM / UNCERTAINTY)
      sample_strategy: How to sample the training batch from the replay buffer (RANDOM / UNCERTAINTY)
      refitting_callback: Callback that gets called when the model is refitting its uncertanties or a better threshold was found
      replay_map: bool, if images from the replay buffer should be projected as pointcloud
    """
    # TODO maybe move to config?
    self.max_buffer_length = 500

    self.model = model
    self.training_loss = training_loss
    self.replay_map = replay_map
    self.optimisers = optimisers
    self.LABELED_BUFFER = []
    self.train_iter = 0
    self.crit = training_loss
    self.save_path = save_path
    self.save_frequency = save_frequency
    self.model_slug = model_slug
    self.imgCount = 0
    self.replacement_strategy = replacement_strategy
    self.sample_strategy = sample_strategy
    self.replay_batch_length = replay_batch_length

    if torch.has_cuda:
      self.training_loss = self.training_loss.cuda()

    self.uncertainty_estimator = SimpleSoftMaxEstimator(lambda x: x, from_logits=True)
    self.threshold_image = lambda x: x

    self.FEATURE_BUFFER = []
    self.UNCERTAINTY_BUFFER = []
    self.refitting_callback = refitting_callback
    # TODO not hardcode
    self.start_stop_experiment_proxy = rospy.ServiceProxy("/start_stop_experiment", SetBool)
    # TODO not hardcode
    self.image_count_pub = rospy.Publisher("/train_count", Int16)

    if self.replay_map:
      # Replay old uncertanty images from buffer as pointcloud
      # TODO change topic not hardcoded
      self.replay_pc_publishers = rospy.Publisher("/semseg/points", PointCloud2)
      self.br = tf.TransformBroadcaster()

      self._uncertainty_pub = rospy.Publisher("~/semseg/uncertainty",
                                              Image,
                                              queue_size=5)

  def __call__(self, *args):
    with torch.no_grad():
      self.model.eval()
      return self.model(*args)

  def cuda(self):
    self.model.cuda()
    return self

  def addSample(self, image, mask, uncertainty_score=None, pose=None, camera=None, depth=None):
    """
    Adds a new sample to the training replay buffer

    Args:
      image: Numpy or torch image
      mask: Numpy or torch mask
      uncertainty_score:  The uncertanty score of this image
      pose:  if replay is set, the pose of this image
      camera: if replay is set, the camera_msg of this image
      depth: if replay is set, the depth image message
    """

    # Convert numpy to torch
    if type(image) is np.ndarray:
      image = image.transpose((2, 0, 1))
      if np.any(image > 1):
        image = image / 255
      image = torch.from_numpy(image).float()
      mask = torch.from_numpy(label).long()

    self.imgCount += 1

    # Add image to training buffer
    self.LABELED_BUFFER.append(
      {'image': image, 'mask': mask, 'number': self.imgCount, 'uncertainty': uncertainty_score, 'pose': pose,
       'camera': camera, 'depth': depth})

    if len(self.LABELED_BUFFER) >= self.max_buffer_length:
      # Resample buffer
      if self.replacement_strategy == "UNCERTAINTY":
        # Resample using uncertanty
        new_buffer = importance_sampling_by_uncertainty(self.LABELED_BUFFER, self.max_buffer_length // 2)
      else:
        # resasmple random
        new_buffer = random.sample(self.LABELED_BUFFER, self.max_buffer_length // 2)

      self.LABELED_BUFFER.clear()
      self.LABELED_BUFFER.extend(new_buffer)

  def train(self, batch_size=-1, freeze_bn=False, grad_norm=0, verbose=True):
    t1 = time.time()
    if self.sample_strategy == "UNCERTAINTY":
      to_be_used = importance_sampling_by_uncertainty(self.LABELED_BUFFER[:-1], N=min(self.replay_batch_length - 1,
                                                                                      len(self.LABELED_BUFFER[:-1])))
    else:
      to_be_used = random.sample(self.LABELED_BUFFER[:-1],
                                 min(self.replay_batch_length - 1, len(self.LABELED_BUFFER[:-1])))

    # Make sure to add newest image
    to_be_used.append(self.LABELED_BUFFER[-1])
    # Shuffle images to not always have most uncertain images in first batch
    random.shuffle(to_be_used)
    self.model.train()

    if freeze_bn:
      for m in self.model.modules():
        if isinstance(m, nn.BatchNorm2d):
          m.eval()

    device = next(self.model.parameters()).device

    loss = torch.tensor(0.0).cuda()
    predicted_images_cnt = 0
    for b in batch(to_be_used, batch_size):
      target = torch.stack([item['mask'] for item in b]).to(device)
      input = torch.stack([item['image'] for item in b]).to(device).squeeze(dim=1)
      predictions = self.model(input)

      if type(predictions) is tuple:  # This means model also provides uncertainties
        # remove uncertainty from prediction
        predictions, uncertainties = predictions

        # Save features for refitting the model later
        features = self.model.features[self.model.feature_layer_name].cpu().detach().numpy()
        # reshaping
        feature_size = features.shape[1]
        features = features.transpose([0, 2, 3, 1]).reshape([-1, feature_size])
        # subsampling (because storing all these embeddings would be too much)
        features = features[np.random.choice(features.shape[0],
                                             size=[300],
                                             replace=False)]
        self.FEATURE_BUFFER.append(features)

        # Add uncertanty to buffer and update values
        for idx, u in enumerate(uncertainties):
          u_np = u.cpu().detach().numpy().ravel()
          self.UNCERTAINTY_BUFFER.append(u_np)
          to_be_used[idx]['uncertainty'] = np.mean(u_np)

      else:
        with torch.no_grad():
          for idx, pred in enumerate(predictions):
            self.model.eval()
            pred_without_argmax = self.model(input[0].unsqueeze(dim=0))
            pred_without_argmax = pred_without_argmax.cpu().detach().numpy().squeeze()
            uncertainty = self.uncertainty_estimator.predict(pred_without_argmax.transpose(1, 2, 0), None)[1]
            u = self.threshold_image(uncertainty).squeeze()
            # Add uncertainty to buffer to fit distribution and determine threshold dynamically
            self.UNCERTAINTY_BUFFER.append(uncertainty)
            to_be_used[idx]['uncertainty'] = np.mean(uncertainty)

      predicted_images_cnt += 1

      loss += self.crit(
        F.interpolate(
          predictions, size=target.size()[1:], mode="bilinear", align_corners=False
        ).squeeze(dim=1),
        target,
      )

    ##
    # Train Model using optimizerss
    ##
    for opt in self.optimisers:
      opt.zero_grad()
    loss.backward()

    if grad_norm > 0.0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)

    for opt in self.optimisers:
      opt.step()

    if (len(self.UNCERTAINTY_BUFFER) > 25):  # TODO move to params
      self.refitting_callback(self.UNCERTAINTY_BUFFER)
      self.UNCERTAINTY_BUFFER.clear()

    self.train_iter += 1
    if verbose:
      rospy.loginfo("Training Iteration #{}, loss: {}, buffer size: {}, time: {}".format(self.train_iter, loss.item(),
                                                                                         len(self.LABELED_BUFFER),
                                                                                         time.time() - t1))
    ##
    # Save checkpointsl
    ##
    if self.train_iter % self.save_frequency == 5:
      t1 = time.time()
      rospy.loginfo("Saving model")
      torch.save(self.model.state_dict(),
                 os.path.join(self.save_path, "{}_{:04d}.pth".format(self.model_slug, self.train_iter)))
      rospy.loginfo("Saving model took: {}".format(time.time() - t1))

    ##
    # Feature extraction for refitting of GMM model
    ##
    if type(self.model) == refinenet.models.uncertainty_utils.UncertaintyModel and len(self.FEATURE_BUFFER) > 300:
      t1 = time.time()
      rospy.loginfo("Going to refit GMM model...")
      # Stop moving
      self.start_stop_experiment_proxy(False)
      features = np.asarray(self.FEATURE_BUFFER).reshape(-1, self.FEATURE_BUFFER[0].shape[-1])
      features = features[np.random.choice(features.shape[0], size=30000, replace=False), :]
      self.model.clustering.fit(features)
      features = torch.from_numpy(features).unsqueeze(dim=0)
      uncertainty_values = self.model.clustering.gmm.gmm.log_prob(
        self.model.clustering.pca(features.cuda())).cpu().detach().numpy()
      # Send uncertainty values to callback to refit threshold
      self.refitting_callback([uncertainty_values])
      self.FEATURE_BUFFER.clear()
      self.start_stop_experiment_proxy(True)
      rospy.loginfo("Refitting model took: {}".format(time.time() - t1))

    ##
    # REPLAY
    # If replay map is set, replay it all 300 images. TODO, move to params
    ##
    if self.replay_map and self.train_iter % 300 == 0:
      t1 = time.time()
      rospy.logdebug("Going to replay full map")
      self.start_stop_experiment_proxy(False)
      t1 = time.time()
      # Make sure to add newest image
      self.model.eval()
      with torch.no_grad():
        device = next(self.model.parameters()).device
        for img in self.LABELED_BUFFER:
          rospy.logdebug("Publishing uncertainty image: {}".format(img['number']))
          _, uncertainty = self.uncertainty_callback(img['image'])

          time_ros = rospy.Time.now()
          self.br.sendTransform(img['pose'][0], img['pose'][1],
                                time_ros,
                                "replay_camera",
                                "drone_1") # TODO: not hardcode
          img_depth = img['depth']
          img_shape = img_depth.shape
          camera = img['camera']
          uncertainty = cv2.resize(uncertainty, (img_depth.shape[1], img_depth.shape[0]), interpolation=cv2.INTER_CUBIC)
          color = (uncertainty * 254).astype(np.uint8).reshape(-1)
          pc_msg = get_pc_for_image(color, img_depth, camera)
          pc_msg.header.frame_id = "replay_camera"
          pc_msg.header.stamp = time_ros
          self.replay_pc_publishers.publish(pc_msg)

          # make RGB, use some nice colormaps:
          uncertainty_uint8 = np.uint8(cm.seismic(uncertainty) *
                                       255)[:, :, 0:3]  # Remove alpha channel

          uncertainty_msg = Image()
          uncertainty_msg.header.frame_id = "replay_camera"
          uncertainty_msg.header.stamp = time_ros
          uncertainty_msg.height = img_shape[0]
          uncertainty_msg.width = img_shape[1]
          uncertainty_msg.step = img_shape[0]

          uncertainty_msg.data = uncertainty_uint8.flatten().tolist()
          uncertainty_msg.encoding = "rgb8"
          self._uncertainty_pub.publish(uncertainty_msg)

          rospy.sleep(0.1)  # Max publish at 10Hz for voxblox to keep up
      # resume experiment. This also restarts the active planner
      self.start_stop_experiment_proxy(True)
      ropsy.loginfo("Replaying map took {}".format(time.time() - t1))

    self.image_count_pub.publish(self.train_iter)
