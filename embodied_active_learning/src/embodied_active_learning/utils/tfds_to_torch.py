import torch
import tensorflow_datasets as tfds
import tensorflow as tf

class TFDSIterableDataset(torch.utils.data.IterableDataset):
  def __init__(self, *args, **kwargs):
    super().__init__()
    self.tfds_data = tfds.load(*args, **kwargs)

  def __iter__(self):
    return iter(tfds.as_numpy(self.tfds_data))

  def __len__(self):
    return len(self.tfds_data)

class TFDataIterableDataset(torch.utils.data.IterableDataset):
  def __init__(self, ds):
    super().__init__()
    self.tf_dataset = tfds.as_numpy(ds)
    self.tf_iter = iter(self.tf_dataset)
    self.cache = []

  def __iter__(self):
      for batch in self.tf_dataset:
          yield torch.from_numpy(batch[0]), torch.from_numpy(batch[1])

  def __getitem__(self, item):
    # REALLY REALLY UGLY. DO NOT DO THIS
    if (len(self.cache) <= item):
      self.cache.append(next(self.tf_iter))

    sample = {}
    sample['image'] =  torch.from_numpy(self.cache[item][0])
    sample['mask'] = torch.from_numpy(self.cache[item][1])
    return sample

  def __len__(self):
    return len(self.tf_dataset)


def data_converter(image, label):
  image = tf.image.convert_image_dtype(image, tf.float32)
  label = tf.cast(label, tf.int64)
  # move channel from last to 2nd
  image = tf.transpose(image, perm=[2, 0, 1])
  return image, label