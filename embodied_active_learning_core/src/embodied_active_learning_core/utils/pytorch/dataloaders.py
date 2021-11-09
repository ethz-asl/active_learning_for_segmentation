import torch.utils.data as torchData
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch

from embodied_active_learning_core.utils.pytorch.image_transforms import get_validation_transforms


class CombinedDataset(torchData.Dataset):
  """ Dataset that combines multiple torch datasets (e.g. 1 replay set + 1 Training set) """

  def __init__(self, datasets, transform=None):
    super().__init__()
    self.datasets = datasets
    self.transform = transform

    lengths = [len(d) for d in datasets]

    # Calculate length of this dataset as sum of lengths of all dasets
    self._length = 0
    for l in lengths:
      self._length += l

    # Randomly assign each requested idx to a dataset, making sure to not have diversity of images in batches
    idxs = np.asarray([idx for _ in range(l) for idx, l in enumerate(lengths)])
    permuted_idxs = np.random.permutation(idxs)
    self.idx_to_ds = permuted_idxs

  def __getitem__(self, index):
    # Dataset internal index (e.g. if first two images are from ds1 and next two are from ds2, index 2 would be mapped
    # to internal idx 0 and dataset index 1
    ds_internal_idx = np.sum(self.idx_to_ds[0: index] == self.idx_to_ds[index])
    data = self.datasets[self.idx_to_ds[index]][ds_internal_idx]

    # Apply transformations
    if self.transform is not None:
      data = self.transform(data)

    return data

  def __len__(self):
    return self._length


class DataLoaderSegmentation(torchData.Dataset):
  """ Datloader to load images produced by the data acquisitors"""

  def __init__(self, folder_path, num_imgs=None, verbose=True,
               transform=get_validation_transforms(normalize=False, additional_targets={'mask': 'mask'}),
               limit_imgs=None, cpu_mode=False):
    super().__init__()
    if verbose:
      print("Creating dataloader with params: {},{},{},{},{}".format(folder_path, num_imgs, transform, limit_imgs,
                                                                     cpu_mode))

    # If gain file exists, sort images based on waypoint gain
    if os.path.exists(os.path.join(folder_path, "gain_info.txt")):
      if verbose:
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
      if verbose:
        print("found {} images".format(len(self.img_files)))

    else:
      self.img_files = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if "img" in f or "rgb" in f])
      self.mask_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if "mask" in f])

      for i in range(len(self.mask_files)):
        if self.img_files[i].replace("img", "").replace("rgb", "") != self.mask_files[i].replace("mask", ""):
          print(f"[Error] Missmatch in training image and mask files. {self.img_files[i]} != {self.mask_files[i]}")
          exit()

      if limit_imgs is not None and limit_imgs != 0:
        self.img_files = self.img_files[::(len(self.img_files) // limit_imgs + 1)]
        self.mask_files = self.mask_files[::(len(self.mask_files) // limit_imgs + 1)]
        if verbose:
          print("[DATALOADER] limited images to {}".format(len(self.mask_files)))

    if num_imgs is not None:
      if verbose:
        print("[DATALOADER] going to limit images to {}".format(num_imgs))
      self.img_files = self.img_files[0:num_imgs]
      self.mask_files = self.mask_files[0:num_imgs]

    self.transform = transform
    self.cpu_mode = cpu_mode

    if len(self.img_files) != len(self.mask_files):
      print("[ERROR] - Labels and Mask count did not match")
      exit()

  def __getitem__(self, index):
    img_path = self.img_files[index]
    mask_path = self.mask_files[index]

    data = np.asarray(Image.open(img_path))[:, :, 0:3].copy()
    label = np.asarray(Image.open(mask_path))[:, :].copy()

    entry = self.transform(image=data, mask=label)
    if torch.cuda.is_available() and not self.cpu_mode:
      entry['image'] = entry['image'].cuda()
      entry['mask'] = entry['mask'].cuda()

    return entry

  def __len__(self):
    return len(self.img_files)
