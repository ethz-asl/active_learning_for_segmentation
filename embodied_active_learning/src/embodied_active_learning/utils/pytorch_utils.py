"""
    Helper class with functionalities that are often used with pytorch
"""
import torch
from PIL import Image
import torch.utils.data as torchData
import numpy as np
import os
from typing import List

def batch(iterable, n=1):
  """ Helper function that creates baches from any iterable """
  if n <= 0:
    yield iterable[:]
  else:
    l = len(iterable)
    for ndx in range(0, l, n):
      yield iterable[ndx:min(ndx + n, l)]


def semseg_compute_confusion(y_hat_lbl, y_lbl, num_classes, ignore_label):
  """ Helper function that computes the confusion matrix. (If possible use e.g. densetorch, as cython is faster """
  assert torch.is_tensor(y_hat_lbl) and torch.is_tensor(y_lbl), 'Inputs must be torch tensors'
  assert y_lbl.device == y_hat_lbl.device, 'Input tensors have different device placement'

  assert y_hat_lbl.dim() == 3 or y_hat_lbl.dim() == 4 and y_hat_lbl.shape[1] == 1
  assert y_lbl.dim() == 3 or y_lbl.dim() == 4 and y_lbl.shape[1] == 1
  if y_hat_lbl.dim() == 4:
    y_hat_lbl = y_hat_lbl.squeeze(1)
  if y_lbl.dim() == 4:
    y_lbl = y_lbl.squeeze(1)

  mask = y_lbl != ignore_label
  y_hat_lbl = y_hat_lbl[mask]
  y_lbl = y_lbl[mask]

  # hack for bincounting 2 arrays together
  x = y_hat_lbl + num_classes * y_lbl
  bincount_2d = torch.bincount(x.long().reshape(-1), minlength=num_classes ** 2)
  assert bincount_2d.numel() == num_classes ** 2, 'Internal error'
  conf = bincount_2d.view((num_classes, num_classes)).long()
  return conf


def semseg_accum_confusion_to_iou(confusion_accum, ignore_zero=False):
  """
  Converts confsuion matrix to iou_mean and iou per class
  """
  conf = confusion_accum.double()
  diag = conf.diag()
  union = (conf.sum(dim=1) + conf.sum(dim=0) - diag)
  iou_per_class = 100 * diag / (union.clamp(min=1e-12))
  if ignore_zero:
    unseen_classes = torch.where(iou_per_class == 0)[0]
    seen_classes = torch.where(iou_per_class != 0)[0]
  else:
    unseen_classes = torch.where(union == 0)[0]
    seen_classes = torch.where(union != 0)[0]

  iou_mean = iou_per_class[seen_classes].mean()
  return iou_mean, iou_per_class, unseen_classes


class DataLoader:

  class CombinedDataset(torchData.Dataset):
    def __init__(self, datasets, transform=None):
      super().__init__()
      self.datasets = datasets
      self.transform = transform

      lengths = [len(d) for d in datasets]
      self._length = 0
      for l in lengths:
        self._length += l

      idxs = np.asarray([idx for _ in range(l) for idx, l in enumerate(lengths)])
      permuted_idxs = np.random.permutation(idxs)
      self.idx_to_ds = permuted_idxs

    def __getitem__(self, index):
      ds_internal_idx = np.sum(self.idx_to_ds[0: index] == self.idx_to_ds[index])

      data = self.datasets[self.idx_to_ds[index]][ds_internal_idx]

      if self.transform != None:
        data = self.transform(data)

      return data



    def __len__(self):
      return self._length


  class DataLoaderSegmentation(torchData.Dataset):
    """ Datloader to load images produced by the data acquisitors"""

    def __init__(self, folder_path, num_imgs=None, transform=None, limit_imgs=None, cpu_mode=False):
      super().__init__()
      print("Creating dataloader with params: {},{},{},{},{}".format(folder_path, num_imgs, transform, limit_imgs,
                                                                     cpu_mode))

      if (os.path.exists(os.path.join(folder_path, "gain_info.txt"))):
        import pandas as pd
        print("found gain info file. Going to order images by gains.")

        def set_file_name(entry):
          entry['file'] = os.path.basename(entry['file'])
          entry['gain'] = float(entry['gain'].replace(";", ""))
          return entry

        gp_gains = pd.read_csv(os.path.join(folder_path, "gain_info.txt"), names=['file', 'gain'])
        gp_gains.apply(set_file_name, axis=1)
        gp_gains_sorted = gp_gains.sort_values(by="gain", ascending=False)
        files_sorted = gp_gains_sorted['file']
        self.img_files = []
        self.mask_files = []
        for entry in files_sorted:
          self.img_files.append(os.path.join(folder_path, entry.replace("mask", "rgb")))
          self.mask_files.append(os.path.join(folder_path, entry))
        print("found {} images".format(len(self.img_files)))

      else:
        self.img_files = sorted(
          [os.path.join(folder_path, f) for f in os.listdir(folder_path) if "img" in f or "rgb" in f])
        self.mask_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if "mask" in f])

        if limit_imgs is not None and limit_imgs != 0:
          self.img_files = self.img_files[::(len(self.img_files) // limit_imgs + 1)]
          self.mask_files = self.mask_files[::(len(self.mask_files) // limit_imgs + 1)]
          print("[DATALOADER] limited images to {}".format(len(self.mask_files)))

      if (num_imgs is not None):
        print("[DATALOADER] going to limit images to {}".format(num_imgs))
        self.img_files = self.img_files[0:num_imgs]
        self.mask_files = self.mask_files[0:num_imgs]

      self.transform = transform
      self.masks_names = ["mask"]
      self.cpu_mode = cpu_mode

      if len(self.img_files) != len(self.mask_files):
        print("[ERROR] - Labels and Mask count did not match")
        self.img_files = None
        self.mask_files = None

    def __getitem__(self, index):
      img_path = self.img_files[index]
      mask_path = self.mask_files[index]
      data = (np.asarray(Image.open(img_path))[:, :, 0:3] / 255).transpose((2, 0, 1)).copy()
      label = np.asarray(Image.open(mask_path))[:, :].copy()
      data = {'image': torch.from_numpy(data).float(), 'mask': torch.from_numpy(label).float()}
      if torch.cuda.is_available() and not self.cpu_mode:
        data['image'] = data['image'].cuda()
        data['mask'] = data['mask'].cuda()

      if self.transform != None:
        data = self.transform(data)
      return data

    def __len__(self):
      return len(self.img_files)


def get_nyu_custom_combined_ds(folder_path, num_imgs=None, transform=None, limit_imgs=None, cpu_mode=False, nyu_ratio = 1):

  arisim_ds = DataLoader.DataLoaderSegmentation(folder_path, num_imgs=num_imgs,limit_imgs=limit_imgs, cpu_mode=False)
  nyu_length = nyu_ratio * len(arisim_ds)
  print("Returning dataset with nyu length: ", nyu_length)
  import tensorflow as tf
  import tensorflow_datasets as tfds
  from embodied_active_learning.utils.tfds_to_torch import TFDataIterableDataset, data_converter
  tf.config.set_visible_devices([], 'GPU')


  data = tfds.load('Nyu_depth_full_v2_labeled',
                   split='full',
                   as_supervised=True)
  traindata = TFDataIterableDataset(data.shuffle(buffer_size=len(data)).take(nyu_length).map(data_converter))

  return DataLoader.CombinedDataset([arisim_ds, traindata], transform=transform)

class Transforms:

  class Normalize:
    def __init__(self, mean, std, cpu_mode=False):
      self.mean = mean
      self.std = std
      if torch.cuda.is_available() and not cpu_mode:
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def __call__(self, sample):
      image, mask = sample['image'], sample['mask']
      image = (image - self.mean) / (self.std)
      return {'image': image, 'mask': mask}


  class ToCuda:
    def __init__(self):
      pass

    def __call__(self, sample):
      sample['image'] = sample['image'].cuda()
      sample['mask'] = sample['mask'].cuda()
      return sample

  class ToCpu:
    def __init__(self):
      pass

    def __call__(self, sample):
      sample['image'] = sample['image'].cpu()
      sample['mask'] = sample['mask'].cpu()
      return sample
  class AsFloat:
    def __init__(self):
      pass

    def __call__(self, sample):
      image, mask = sample['image'], sample['mask']
      image = image.float()
      return {'image': image, 'mask': mask}

  class TargetAsLong:
    def __init__(self, target_name):
      self.target_name = target_name

    def __call__(self, sample):
      sample[self.target_name] = sample[self.target_name].long()
      return sample

  class AsDensetorchSample:
    def __init__(self, names):
      self.names = names

    def __call__(self, sample):
      sample['names'] = self.names
      sample['image'] = (sample['image'].detach().numpy() * 255).astype(np.uint8).transpose((1, 2, 0))
      sample['mask'] = (sample['mask'].detach().numpy()).astype(np.uint8)
      return sample
