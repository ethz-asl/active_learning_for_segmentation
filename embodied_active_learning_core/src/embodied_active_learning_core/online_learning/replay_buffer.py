""" Provides replay buffer with source and target dataset """

from typing import List
from enum import Enum
from dataclasses import dataclass, field
import random
import math
import numpy as np

from embodied_active_learning_core.online_learning.sample import TrainSample
from embodied_active_learning_core.utils.utils import importance_sampling_by_uncertainty


class BufferUpdatingMethod(Enum):
  RANDOM = 1
  UNCERTAINTY = 2

  @staticmethod
  def from_string(name: str):
    if name == "random":
      return BufferUpdatingMethod.RANDOM
    if name == "uncertainty":
      return BufferUpdatingMethod.UNCERTAINTY
    raise ValueError("Unknown Buffer updating method " + name)


class BufferSamplingMethod(Enum):
  RANDOM = 1
  UNCERTAINTY = 2

  @staticmethod
  def from_string(name: str):
    if name == "random":
      return BufferSamplingMethod.RANDOM
    if name == "uncertainty":
      return BufferSamplingMethod.UNCERTAINTY
    raise ValueError("Unknown Buffer sampling method " + name)


@dataclass
class ReplayBuffer:
  """ Simple replay buffer only storing target domain data """
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
        self.entries = importance_sampling_by_uncertainty(self.entries, self.max_buffer_length // 2)
    self.entries.append(sample)

  def draw_samples(self, sample_size: int) -> List[TrainSample]:
    if self.sampling_strategy == BufferSamplingMethod.RANDOM:
      return random.sample(self.entries, sample_size)
    elif self.sampling_strategy == BufferSamplingMethod.UNCERTAINTY:
      return importance_sampling_by_uncertainty(self.entries, sample_size)
    else:
      raise ValueError("Unknown sampling strategy provided", self.sampling_strategy)

  def get_all(self):
    random.shuffle(self.entries)
    return self.entries

  def __len__(self):
    return len(self.entries)

  def __iter__(self):
    return iter(self.entries)


class DatasetSplitReplayBuffer(ReplayBuffer):
  def __init__(self, dataset, max_buffer_length, replacement_strategy=BufferUpdatingMethod.RANDOM,
               sampling_strategy=BufferSamplingMethod.RANDOM, split_ratio=0.2, source_buffer_len=200):
    """
      Replay Buffer that also has images from a source dataset
      :args
        dataset: Source dataset which should be replayed
        max_buffer_length: Max size of replay buffer (without source dataset)
        replacement_strategy: Which images to keep when buffer size reaches limit
        sampling_strategy: Method used to sample images when calling get_sample
        split_ratio: How many percent of a sampled batch should be from the source domain
        source_buffer_len: How many images to load into the source domain buffer
    """
    super(DatasetSplitReplayBuffer, self).__init__(max_buffer_length=max_buffer_length,
                                                   replacement_strategy=replacement_strategy,
                                                   sampling_strategy=sampling_strategy)
    self.split_ratio = split_ratio
    self.source_imgs: List[TrainSample] = []
    for entry in dataset:
      if len(self.source_imgs) >= source_buffer_len:
        break

      # Convert sample to numpy images and add to buffer
      sample = TrainSample(image=(entry['image'].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8),
                           number=len(self.source_imgs), mask=None, is_gt_sample=True)
      sample.update_mask(entry['mask'].long().cpu().unsqueeze(dim=0).detach().numpy().transpose(1, 2, 0))
      self.source_imgs.append(sample)
    print("Created replay DS buffer for ds located with size:", len(dataset))
  def draw_samples(self, sample_size: int) -> List[TrainSample]:
    """ Returns samples consisting of target domain images and additional replayed images from the source domain """
    replay_size = math.ceil(sample_size * (1 - self.split_ratio))
    samples = super(DatasetSplitReplayBuffer, self).draw_samples(replay_size)
    random.shuffle(self.source_imgs)
    print("drawing samples from replay 2 buffer. Split ratio: ", self.split_ratio, "len now:", len(samples))
    if self.split_ratio != 0:
      samples.extend(random.sample(self.source_imgs, sample_size - replay_size))
    print("len later:", len(samples))
    return samples

  def get_all(self):
    """ Returns all source images in the buffer together with the replay buffer """
    full_train_set = []
    full_train_set.extend(super().get_all())

    random.shuffle(self.source_imgs)
    print("drawing all from replay 2 buffer. Split ratio: ", self.split_ratio, "len now:", len(full_train_set))
    print("index:", math.ceil(len(full_train_set) * self.split_ratio))
    print("len source", len(self.source_imgs))
    if self.split_ratio != 0:
      full_train_set.extend(self.source_imgs[:math.ceil(len(full_train_set) * self.split_ratio)])

    print("len later:", len(full_train_set))

    return full_train_set

  def __len__(self):
    return len(self.entries) + len(self.source_imgs)

  def __iter__(self):
    return iter(self.get_all())
