from typing import Callable, List, Optional

import time
import rospy
import random
import pickle
import os.path
import math

import torch.nn.functional as F
import torch.nn as nn
import torch

from std_srvs.srv import SetBool
from std_msgs.msg import Int16

from embodied_active_learning_core.online_learning.sample import TrainSample
from embodied_active_learning_core.online_learning.replay_buffer import ReplayBuffer, BufferUpdatingMethod, \
  BufferSamplingMethod, DatasetSplitReplayBuffer
from embodied_active_learning_core.utils.utils import batch
from embodied_active_learning_core.utils.pytorch.dataloaders import DataLoaderSegmentation
from volumetric_labeling.labels.pseudo_labler import PseudoLabler
from embodied_active_learning_core.utils.pytorch.image_transforms import get_train_transforms, get_validation_transforms

from embodied_active_learning.utils.config import Configs
from embodied_active_learning_core.semseg.datasets.dataset_manager import get_test_dataset_for_folder
from embodied_active_learning_core.utils.pytorch.evaluation import EarlyStoppingWrapper
from embodied_active_learning_core.utils.pytorch.image_transforms import prepare_img
from panoptic_mapping_msgs.srv import SaveLoadMap

class OnlineLearner:
  """
   Class that wraps around a torch model. Can be used to train network
  """

  def __init__(self, model: torch.nn.Module, optimisers: List[any], config: Configs,
               training_loss=nn.CrossEntropyLoss(ignore_index=255, reduction='none')):
    """
      Constructor of the OnlineLearner model.
      Should be called using the get_online_learning_refinenet() function
    """
    self.online_learning_config = config.online_learning_config
    self.config = config

    # Function that will be called once the leraning process is done (e.g. refitting of GMM for uncertainty)
    self.post_training_hooks: List[Optional[Callable]] = []  # Functions
    print("Create replay DS for DS located at:", self.config.online_learning_config.replay_dataset_path)


    self.replay_set = get_test_dataset_for_folder(self.config.online_learning_config.replay_dataset_path,
                                  normalize=not self.config.uncertainty_estimation_config.network_config.groupnorm)
    # Get trainings buffer
    self.training_buffer: ReplayBuffer = DatasetSplitReplayBuffer(
      self.replay_set,
      max_buffer_length=self.online_learning_config.replay_buffer_size,
      source_buffer_len=200,
      replacement_strategy=BufferUpdatingMethod.from_string(self.online_learning_config.replacement_strategy),
      sampling_strategy=BufferSamplingMethod.from_string(self.online_learning_config.sampling_strategy),
      split_ratio=self.online_learning_config.old_domain_ratio)

    self.model: torch.nn.Module = model
    self.best_miou = 0
    self.training_loss = training_loss
    self.optimisers = optimisers
    self.train_iter = 0
    self.crit = training_loss
    self.pseudo_labler: PseudoLabler = PseudoLabler(self.config.pseudo_labler_config.weights_method)
    self.epoch_size_reached = False

    if torch.has_cuda:
      self.training_loss = self.training_loss.cuda()

    self.samples_seen = 0
    self.start_stop_experiment_proxy = rospy.ServiceProxy("/start_stop_experiment", SetBool)
    self.train_count_pub = rospy.Publisher("/train_count", Int16, queue_size=10)
    self.train_transforms = get_train_transforms(img_height=480, img_width=640,
                                                 normalize=self.online_learning_config.normalize_imgs)


  def __call__(self, *args):
    with torch.no_grad():
      self.model.eval()
      return self.model(*args)

  def cuda(self):
    self.model.cuda()
    return self

  def addSample(self, online_sample: TrainSample):
    """
    Adds a new sample to the training buffer
    """
    # Add image to training buffer
    self.training_buffer.add_sample(online_sample)
    self.samples_seen += 1

  def train(self, batch_size=-1, verbose=True):
    epochs = self.online_learning_config.num_epochs
    if torch.cuda.is_available():
      print("swapping to cuda")
      self.model = self.model.cuda()

    if len(self.training_buffer) < self.online_learning_config.min_buffer_length:
      print(
        f"replay buffer length:{len(self.training_buffer)}/{self.online_learning_config.min_buffer_length}. Waiting for more samples before training")
      return

    if self.config.online_learning_config.use_bundle_based_training:
      if not self.epoch_size_reached:
        return
      else:
        self.start_stop_experiment_proxy(False)

        if self.config.online_learning_config.use_pseudo_labels:
          self.pseudo_labler.label_many(self.training_buffer.entries)
        self.dump_buffer(dump_path=self.config.log_config.get_dataset_dump_folder())

        to_be_used: List[TrainSample] = self.training_buffer.get_all()
        random.shuffle(to_be_used)
        # 15 percent valid setup
        valid_ratio = 0.15
        valid_end = math.ceil(valid_ratio * len(to_be_used) + 1)
        valid: List[TrainSample] = to_be_used[:valid_end]

        valid_ds = [{'image': torch.from_numpy(
          prepare_img(sample.image, normalize=self.online_learning_config.normalize_imgs).transpose(2, 0, 1)).unsqueeze(
          0).float(),
                     'mask': torch.from_numpy(sample.mask).unsqueeze(0)} for sample in valid]
        to_be_used = to_be_used[valid_end:]

    else:
      to_be_used: List[TrainSample] = self.training_buffer.draw_samples(
        min(self.online_learning_config.replay_batch_length, len(self.training_buffer)))
      valid_ds = None  # No validation set in full online learning
      if self.config.online_learning_config.use_pseudo_labels:
        # Request pseudo labels for images
        self.pseudo_labler.label_many(to_be_used)

    t1 = time.time()
    test_loaders = []
    test_names = []
    for name, path in self.online_learning_config.test_datasets:
      test_names.append(name)
      test_loaders.append(torch.utils.data.DataLoader(
        DataLoaderSegmentation(path, transform=get_validation_transforms(
          normalize=self.online_learning_config.normalize_imgs,
          additional_targets={'mask': 'mask'}), num_imgs=120,
                               verbose=False), batch_size=8))

    score_file = os.path.join(self.config.log_config.get_log_folder(), "scores.csv")
    weights_file = os.path.join(self.config.log_config.get_checkpoint_folder(), f"best_iteration_{self.train_iter}.pth")
    early_stopping = EarlyStoppingWrapper(self.model, valid_ds, test_loaders, dataset_names=['Validation', *test_names],
                                          training_step=self.train_iter)

    if self.train_iter == 0:
      # Print First Score Values
      early_stopping.score_and_maybe_save(log_file=score_file, weights_file=weights_file)

    for epoch in range(epochs):
      self.train_count_pub.publish(self.train_iter)  # Sends heartbeat that learning is still running
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
          # Pseudo Label mask is missing (maybe collected in parallel to training start?)
          if item.mask is None:
            continue

          # Set dummy weight if not available
          if item.weights is None:
            if self.online_learning_config.use_weights:
              print("[WARNING] found training item without weights, but training mode uses weights.")
            item.weights = item.mask * 0 + 1 / (item.mask.shape[-2] * item.mask.shape[-1])

          # Apply training transforms
          entry = self.train_transforms(
            image=item.image,
            mask=item.mask,
            weight=item.weights
          )
          target.append(entry['mask'])
          weights.append(entry['weight'])
          input.append(entry['image'])

        target = torch.stack(target).to(device).long()
        input = torch.stack(input).to(device)
        weights = torch.stack(weights).to(device)

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
          lass_value = weights * loss_crit
          loss_reduced = torch.sum(lass_value)
        else:
          loss_reduced = torch.mean(loss_crit)

        loss += loss_reduced

        ##
        # Train Model using optimizers
        ##
        for opt in self.optimisers:
          opt.zero_grad()
        loss.backward()

        for opt in self.optimisers:
          opt.step()

      if verbose:
        rospy.loginfo(
          "Training Iteration #{} Epoch: {}/{}, loss: {}, buffer size: {}, time: {}".format(self.train_iter, epoch,
                                                                                            epochs, loss.item(),
                                                                                            len(self.training_buffer),
                                                                                            time.time() - t1))
      if self.train_iter % self.config.online_learning_config.save_frequency == 0:
        early_stopping.score_and_maybe_save(log_file=score_file, weights_file=weights_file)

    self.train_iter += 1

    if self.config.online_learning_config.use_bundle_based_training:
      rospy.loginfo("Loading best model so far")
      self.model.load_state_dict(torch.load(weights_file))

    self.train_count_pub.publish(self.train_iter)

    for callback in self.post_training_hooks:
      rospy.loginfo("Calling Post Training Hook")
      callback(self)

    if self.config.online_learning_config.use_bundle_based_training:
      self.start_stop_experiment_proxy(True)
      self.epoch_size_reached = False

  def dump_buffer(self, dump_path):
    """ Extract the stored training data to the dump_path folder"""
    print(f"Dumping dataset to {dump_path}")

    dataset_number = 0
    while os.path.exists(os.path.join(dump_path, f"step_{dataset_number:03}")):
      dataset_number += 1

    os.mkdir(os.path.join(dump_path, f"step_{dataset_number:03}"))
    for entry in self.training_buffer.entries:
      print("dumping ", entry.number, " to training folder")
      pickle.dump(entry,
                  open(os.path.join(dump_path, f"step_{dataset_number:03}", f"training_entry_{entry.number:03}.pkl"),
                       'wb'))

  def addManySamples(self, samples: List[TrainSample]):
    """
      Adds a list of Training samples
    """
    for s in samples:
      self.training_buffer.add_sample(s)
    self.samples_seen += len(samples)
