import math
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
import tf
import rospy
import random
import pickle
import os.path
import numpy as np
import cv2
from typing import List
from std_srvs.srv import SetBool, Empty
from std_msgs.msg import Int16
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image
from matplotlib import cm
from embodied_active_learning.utils.utils import get_pc_for_image
from embodied_active_learning.utils.pytorch_utils import batch
from embodied_active_learning.utils.config import Configs
from embodied_active_learning.uncertainty_estimation.uncertainty_estimator import SimpleSoftMaxEstimator
from embodied_active_learning.pseudo_labels.pseudo_labler import PseudoLabler
from embodied_active_learning.online_learning.sample import TrainSample
from embodied_active_learning.online_learning.replay_buffer import ReplayBuffer, BufferUpdatingMethod, BufferSamplingMethod, DatasetSplitReplayBuffer
from embodied_active_learning.utils.pytorch_utils import get_train_transforms
import math
from embodied_active_learning.semseg.datasets.dataset_manager import get_test_dataset_for_folder
from embodied_active_learning.evaluation.evaluation import EarlyStoppingWrapper

class OnlineLearner:
  """
   Class that wraps around a torch model providin replay buffer etc.
  """

#
  def __init__(self, model : torch.nn.Module, optimisers : any, config: Configs, training_loss =nn.CrossEntropyLoss(ignore_index = 255, reduction = 'none'), save_frequency=20,
               refitting_callback=lambda b: None,
               replay_map=False, max_buffer_length=500, replay_batch_length=8, nyu_replay_split = 0):
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
    self.online_learning_config = config.online_learning_config
    self.config = config
    self.post_training_hooks = [] # List containing post training hooks

    self.training_buffer : ReplayBuffer = DatasetSplitReplayBuffer(get_test_dataset_for_folder(self.config.online_learning_config.replay_dataset_path, normalize = not self.config.uncertainty_estimation_config.network_config.groupnorm),
                                                                   max_buffer_length = self.online_learning_config.replay_buffer_size,
                                                                   nyu_split_ratio= self.online_learning_config.old_domain_ratio,
                                                                   replacement_strategy = BufferUpdatingMethod.from_string(self.online_learning_config.replacement_strategy),
                                                                   sampling_strategy = BufferSamplingMethod.from_string(self.online_learning_config.sampling_strategy))

    self.model: torch.nn.Module = model
    self.best_miou = 0
    self.training_loss = training_loss
    self.optimisers = optimisers
    self.train_iter = 0
    self.crit = training_loss
    self.pseudo_labler: PseudoLabler = PseudoLabler(self.config.pseudo_labler_config)
    self.epoch_size_reached = False

    if torch.has_cuda:
      self.training_loss = self.training_loss.cuda()

    self.uncertainty_estimator = SimpleSoftMaxEstimator(lambda x: x, from_logits=True)
    self.threshold_image = lambda x: x

    self.FEATURE_BUFFER = []
    self.UNCERTAINTY_BUFFER = []
    self.refitting_callback = refitting_callback
    self.samples_seen = 0
    # TODO not hardcode
    self.start_stop_experiment_proxy = rospy.ServiceProxy("/start_stop_experiment", SetBool)
    # TODO not hardcode
    self.image_count_pub = rospy.Publisher("/train_count", Int16, queue_size=10)
    # TODO not hardcode
    self.train_transforms = get_train_transforms(img_height= 480, img_width = 640, normalize=self.online_learning_config.normalize_imgs)

    if self.config.uncertainty_estimation_config.replay_old_pc:
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

  def addSample(self, online_sample: TrainSample):
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
    # Add image to training buffer
    self.training_buffer.add_sample(online_sample)
    self.samples_seen += 1

  def train(self, batch_size=-1, grad_norm=0, verbose=True):
    epochs = self.online_learning_config.num_epochs

    if len(self.training_buffer) < self.online_learning_config.min_buffer_length:
      print(f"replay buffer length:{len(self.training_buffer)}/{self.online_learning_config.min_buffer_length}. Waiting for more samples before training")
      return

    if self.config.online_learning_config.use_epoch_based_training:
      if not self.epoch_size_reached:
        return
      else:
        print("Dumping whole training set at current time")
        self.start_stop_experiment_proxy(False)
        self.dump_buffer()
        to_be_used: List[TrainSample] = self.training_buffer.get_all()
        random.shuffle(to_be_used)
        # 15 percent valid setup
        valid_ratio = 0.1 # TODO not hardcode
        valid_end = math.ceil(valid_ratio * len(to_be_used) + 1)
        valid: List[TrainSample] = to_be_used[:valid_end]
        # TODO normalize
        valid_ds = [{'image': torch.from_numpy(sample.image.transpose(2,0,1)).unsqueeze(0) / 255, 'mask': torch.from_numpy(sample.mask).unsqueeze(0)} for sample in valid]
        to_be_used = to_be_used[valid_end:]
        print("Size valid:", len(valid_ds))
        for sample in valid:
          print(np.unique(sample.mask))
    else:
      to_be_used : List[TrainSample] = self.training_buffer.draw_samples(min(self.replay_batch_length, len(self.training_buffer)))
      # Request pseudo labels for images
      self.pseudo_labler.label_many(to_be_used)

    t1 = time.time()
    # print("NET:")
    # print(self.model)
    print("Optimizer")
    print(self.optimisers)
    ##
    # Save checkpoints
    ##
    if self.train_iter % self.online_learning_config.save_frequency == 0:
      t1 = time.time()
      rospy.loginfo("Saving model")
      torch.save(self.model.state_dict(),
                 os.path.join(self.config.log_config.get_checkpoint_folder(), "{}_{:04d}.pth".format(self.config.uncertainty_estimation_config.network_config.name, self.train_iter)))
      rospy.loginfo("Saving model took: {}".format(time.time() - t1))
    #
    # def __init__(self, model: torch.nn.Module, validation_dataloader: torch.utils.data.DataLoader,
    #              additional_dataloaders: List[torch.utils.data.DataLoader] = [], dataset_names=['Validation'],
    #              burn_in_steps=5):

    import torchvision
    from embodied_active_learning.utils import pytorch_utils
    # TODO change to albumnation
    transform = torchvision.transforms.Compose(
      [pytorch_utils.Transforms.Normalize(torch.tensor(np.array([0.72299159, 0.67166396, 0.63768772]).reshape((3, 1, 1))) if self.online_learning_config.normalize_imgs
                                          else torch.tensor(np.array([0, 0, 0]).reshape((3, 1, 1))),
                                          torch.tensor( np.array([0.2327359, 0.24695725, 0.25931836]).reshape( (3, 1, 1))) if self.online_learning_config.normalize_imgs
                                          else torch.tensor(np.array([1, 1, 1]).reshape((3, 1, 1)))),
                                          pytorch_utils.Transforms.AsFloat()
       ])

    testLoader = torch.utils.data.DataLoader(pytorch_utils.DataLoader.DataLoaderSegmentation(self.config.online_learning_config.test_dataset_path, transform=transform, num_imgs=120,verbose=False), batch_size=8)

    score_file = os.path.join(self.config.log_config.get_log_folder(), "scores.csv")
    weights_file = os.path.join(self.config.log_config.get_checkpoint_folder(), f"best_iteration_{self.train_iter}.pth")
    early_stopping = EarlyStoppingWrapper(self.model, valid_ds, [testLoader], dataset_names= ['Validation', f'Test ({os.path.basename(self.config.online_learning_config.test_dataset_path)})'], training_step=self.train_iter)


    # Print First Score Values
    early_stopping.score_and_maybe_save(log_file=score_file, weights_file=weights_file)

    for epoch in range(epochs):
      self.image_count_pub.publish(self.train_iter)
      self.model.train()

      if self.online_learning_config.freeze_bn:
        for m in self.model.modules():
          if isinstance(m, nn.BatchNorm2d):
            m.eval()

      device = next(self.model.parameters()).device

      for b in batch(to_be_used, batch_size):
        loss = torch.tensor(0.0).cuda()
        predicted_images_cnt = 0

        target = []
        input = []
        weights = []

        for item in b:
          if item.mask is None:
            continue

          if item.weights is None:
            item.weights = item.mask * 0 + 1/(item.mask.shape[-2]*item.mask.shape[-1])

          entry = self.train_transforms(
            image = item.image,
            mask = item.mask,
            weight = item.weights
          )
          target.append(entry['mask'])
          weights.append(entry['weight'])
          input.append(entry['image'])

        target = torch.stack(target).to(device).long()
        input = torch.stack(input).to(device)
        weights = torch.stack(weights).to(device)

        if self.online_learning_config.use_weights:
          thresholded_weights = 0 * weights
          max_sample = torch.max(weights, axis=1)[0]
          max_sample = torch.max(max_sample, axis=1)[0]
          for idx, weight in enumerate(weights):
            thresholded_weights[idx, :, :] = weight * (weight / max_sample[idx] > 0.8)
          del weights
          del weight
          thresholded_weights = (thresholded_weights > 0).float()
          thresholded_weights = thresholded_weights / torch.sum(thresholded_weights)

        predictions = self.model(input)
        if type(predictions) == tuple:
          predictions = predictions[0]

        predicted_images_cnt += 1
        loss_crit = self.crit(
          F.interpolate(
            predictions, size=target.size()[1:], mode="bilinear", align_corners=False
          ).squeeze(dim=1),
          target,
        )

        if self.online_learning_config.use_weights:
          lass_value = thresholded_weights * loss_crit
          loss_reduced = torch.sum(lass_value)
        else:
          loss_reduced = torch.mean(loss_crit)

        loss += loss_reduced

        ##
        # Train Model using optimizerss
        ##
        for opt in self.optimisers:
          opt.zero_grad()
        loss.backward()

        for opt in self.optimisers:
          opt.step()

        if self.online_learning_config.use_weights:
          thresholded_weights = thresholded_weights.cpu()

      if verbose:
        rospy.loginfo("Training Iteration #{} Epoch: {}/{}, loss: {}, buffer size: {}, time: {}".format(self.train_iter, epoch, epochs, loss.item(),
                                                                                           len(self.training_buffer),
                                                                                           time.time() - t1))
      early_stopping.score_and_maybe_save(log_file=score_file, weights_file=weights_file)

    self.train_iter += 1
    rospy.loginfo("Loading best model so far")
    self.model.load_state_dict(torch.load(weights_file))

    for callback in self.post_training_hooks:
      print("Calling Post Training Hook")
      callback(self)

    if self.config.online_learning_config.use_epoch_based_training and self.epoch_size_reached:
        self.start_stop_experiment_proxy(True)
        self.epoch_size_reached = False

        if self.config.online_learning_config.reset_map:
        # TODO hardcoded
          srv = rospy.ServiceProxy("/mapper/planner_node/reset_semantic_map", Empty)
          srv()
          print("="*100, "\n Resetted map ")

  def dump_buffer(self):
    dump_path = self.config.log_config.get_dataset_dump_folder()
    print(f"Dumping dataset to {dump_path}")
    self.pseudo_labler.label_many(self.training_buffer.entries)

    dataset_number = 0
    while os.path.exists(os.path.join(dump_path, f"step_{dataset_number:03}")):
      dataset_number += 1

    os.mkdir(os.path.join(dump_path, f"step_{dataset_number:03}"))
    for entry in self.training_buffer.entries:
      print("dumping ", entry.number, " to training folder")
      pickle.dump(entry, open(os.path.join(dump_path, f"step_{dataset_number:03}", f"training_entry_{entry.number:03}.pkl"),'wb'))

  def addManySamples(self, samples: List[TrainSample]):
    for s in samples:
      self.training_buffer.add_sample(s)
    self.samples_seen += len(samples)


  def updateUncertainties(self, predictions, input: List[torch.tensor], to_be_used: List[TrainSample]):
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
        to_be_used[idx].uncertainty = np.mean(u_np)

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
          to_be_used[idx].uncertainty = np.mean(uncertainty)


  def replay_old_images(self):
    ##
    # REPLAY
    # If replay map is set, replay it all 300 images. TODO, move to params
    ##
    t1 = time.time()
    rospy.logdebug("Going to replay full map")
    self.start_stop_experiment_proxy(False)
    t1 = time.time()
    # Make sure to add newest image
    self.model.eval()
    with torch.no_grad():
      device = next(self.model.parameters()).device
      for img in self.training_buffer:
        rospy.logdebug("Publishing uncertainty image: {}".format(img['number']))
        _, uncertainty = self.uncertainty_callback(img.image)

        time_ros = rospy.Time.now()
        self.br.sendTransform(img.pose[0], img.pose[1],
                              time_ros,
                              "replay_camera",
                              "drone_1") # TODO: not hardcode
        img_depth = img.depth
        img_shape = img_depth.shape
        camera = img.camera
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

        uncertainty_msg.data = uncertainty_uint8.flatten().toList()
        uncertainty_msg.encoding = "rgb8"
        self._uncertainty_pub.publish(uncertainty_msg)

        rospy.sleep(0.1)  # Max publish at 10Hz for voxblox to keep up
    # resume experiment. This also restarts the active mapper
    self.start_stop_experiment_proxy(True)
    rospy.loginfo("Replaying map took {}".format(time.time() - t1))