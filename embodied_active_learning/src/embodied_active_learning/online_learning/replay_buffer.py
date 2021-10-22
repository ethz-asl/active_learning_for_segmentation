""" Provides replay buffer with source and target dataset """

from typing import List, Optional
from enum import Enum
from embodied_active_learning.online_learning.sample import TrainSample
from dataclasses import dataclass, field
import random
import math
from embodied_active_learning.utils.pytorch_utils import DataLoader
import numpy as np

class BufferUpdatingMethod(Enum):
  RANDOM = 1
  UNCERTAINTY = 2

  @staticmethod
  def from_string(name:str):
    if name == "random":
      return BufferUpdatingMethod.RANDOM
    if name == "uncertainty":
      return BufferUpdatingMethod.UNCERTAINTY
    raise ValueError("Unknown Buffer updating method " + name)

class BufferSamplingMethod(Enum):
  RANDOM = 1
  UNCERTAINTY = 2

  @staticmethod
  def from_string(name:str):
    if name == "random":
      return BufferSamplingMethod.RANDOM
    if name == "uncertainty":
      return BufferSamplingMethod.UNCERTAINTY
    raise ValueError("Unknown Buffer sampling method " + name)


@dataclass
class ReplayBuffer:
  max_buffer_length: int
  replacement_strategy: BufferUpdatingMethod = BufferUpdatingMethod.RANDOM
  sampling_strategy: BufferSamplingMethod = BufferSamplingMethod.RANDOM
  entries: List[TrainSample] = field(default_factory=list)

  def add_sample(self, sample: TrainSample):
    if self.__len__() >= self.max_buffer_length:
      print(f"Labeled buffer getting too large. Max Length: {self.max_buffer_length}. Going to downsample it")
      # Resample buffer
      if self.replacement_strategy == BufferUpdatingMethod.RANDOM:
        self.entries = random.sample(self.entries, self.max_buffer_length // 2)
      else:
        raise ValueError("NOT IMPLEMENTED")
    self.entries.append(sample)

  def draw_samples(self, sample_size: int) -> List[TrainSample]:
    if self.sampling_strategy == BufferSamplingMethod.RANDOM:
      return random.sample(self.entries, sample_size)
    elif self.sampling_strategy == BufferSamplingMethod.UNCERTAINTY:
      raise ValueError("NOT IMPLEMENTED")
    else:
      raise ValueError("Unknown sampling strategy provied", self.sampling_strategy)

  def get_all(self):
    random.shuffle(self.entries)
    return self.entries

  def __len__(self):
    return len(self.entries)

  def __iter__(self):
    return iter(self.entries)



class DatasetSplitReplayBuffer(ReplayBuffer):

  def __init__(self, dataset, max_buffer_length, replacement_strategy = BufferUpdatingMethod.RANDOM, sampling_strategy = BufferSamplingMethod.RANDOM, nyu_split_ratio = 0.2, nyu_buffer_len = 200):
    super(DatasetSplitReplayBuffer, self).__init__(max_buffer_length =  max_buffer_length, replacement_strategy = replacement_strategy, sampling_strategy = sampling_strategy)
    self.nyu_split_ratio = nyu_split_ratio
    self.nyu_imgs: List[TrainSample] = []
    for entry in dataset:
      if(len(self.nyu_imgs) >= nyu_buffer_len):
        break

      sample = TrainSample(image=(entry['image'].cpu().detach().numpy().transpose(1,2,0)*255).astype(np.uint8), number=len(self.nyu_imgs), mask=None, is_gt_sample=True)
      sample.update_mask(entry['mask'].long().cpu().unsqueeze(dim = 0).detach().numpy().transpose(1,2,0))
      self.nyu_imgs.append(sample)

  def draw_samples(self, sample_size: int) -> List[TrainSample]:
    replay_size = math.ceil(sample_size * (1 - self.nyu_split_ratio))
    samples = super(DatasetSplitReplayBuffer, self).draw_samples(replay_size)
    if self.nyu_split_ratio != 0:
      samples.extend(random.sample(self.nyu_imgs, sample_size - replay_size))
    return samples

  def get_all(self):
    full_train_set = []
    full_train_set.extend(super().get_all())
    random.shuffle(self.nyu_imgs)

    if self.nyu_split_ratio != 0:
      full_train_set.extend(self.nyu_imgs[:math.ceil(len(full_train_set)* self.nyu_split_ratio)])

    return full_train_set

  def __len__(self):
    return len(self.entries)

  def __iter__(self):
    return iter(self.entries)


class NyuSplitReplayBuffer(ReplayBuffer):

  def __init__(self, max_buffer_length, replacement_strategy = BufferUpdatingMethod.RANDOM, sampling_strategy = BufferSamplingMethod.RANDOM, nyu_split_ratio = 0.2, nyu_buffer_len = 200):
    super(NyuSplitReplayBuffer, self).__init__(max_buffer_length =  max_buffer_length, replacement_strategy = replacement_strategy, sampling_strategy = sampling_strategy)
    self.nyu_split_ratio = nyu_split_ratio
    nyu_loader = DataLoader.NYUDepth(root_dir = "/home/rene/Downloads/nyu_depth", image_set='train', transforms=None, length = None)
    self.nyu_imgs: List[TrainSample] = []
    for entry in nyu_loader:
      if(len(self.nyu_imgs) >= nyu_buffer_len):
        break

      sample = TrainSample(image=(entry['image'].cpu().detach().numpy().transpose(1,2,0)*255).astype(np.uint8), number=len(self.nyu_imgs), mask=None, is_gt_sample=True)
      sample.update_mask(entry['mask'].long().cpu().unsqueeze(dim = 0).detach().numpy().transpose(1,2,0))
      self.nyu_imgs.append(sample)

  def draw_samples(self, sample_size: int) -> List[TrainSample]:
    replay_size = math.ceil(sample_size * (1 - self.nyu_split_ratio))
    samples = super(NyuSplitReplayBuffer, self).draw_samples(replay_size)
    if self.nyu_split_ratio != 0:
      samples.extend(random.sample(self.nyu_imgs, sample_size - replay_size))
    return samples

  def get_all(self):
    full_train_set = []
    full_train_set.extend(super().get_all())
    random.shuffle(self.nyu_imgs)

    if self.nyu_split_ratio != 0:
      full_train_set.extend(self.nyu_imgs[:math.ceil(len(full_train_set)* self.nyu_split_ratio)])

    return full_train_set

  def __len__(self):
    return len(self.entries)

  def __iter__(self):
    return iter(self.entries)