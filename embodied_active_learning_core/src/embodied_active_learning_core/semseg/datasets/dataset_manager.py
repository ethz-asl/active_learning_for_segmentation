import os
import torch
import glob
import numpy as np
from PIL import Image as PImage

from embodied_active_learning_core.utils.pytorch.image_transforms import get_validation_transforms, get_train_transforms


class GenericFromFolderDataset(torch.utils.data.Dataset):
  """ Datloader to load images produced by the data acquisitors"""

  def __init__(self, folder_path, num_imgs=None, transform=None, image_ending="image.png", mask_ending="mask.png",
               weights_ending="weights.npy", mask_transform_fn = None):
    super().__init__()
    self.entries = []
    for f in sorted(glob.glob(f"{folder_path}/*" + image_ending)):
      mask_file = f.replace(image_ending, mask_ending)
      weights_file = f.replace(image_ending, weights_ending)
      mask_content =  np.asarray(PImage.open(mask_file)).copy()

      # Allows to transform masks (e.g. nyu <-> eigen13)
      if mask_transform_fn is not None:
        mask_content = mask_transform_fn(mask_content)

      data = {
        'image': np.asarray(PImage.open(f)).copy(),
        'mask': mask_content,
        'weight': np.load(weights_file) if os.path.exists(weights_file) else mask_content
      }
      self.entries.append(data)
    self.transform = transform

  def __getitem__(self, index):
    data = self.entries[index]

    if self.transform is not None:
      data = self.transform(image=data['image'],
                            mask=data['mask'], weight=data['weight'])
    return data

  def __len__(self):
    return len(self.entries)


def get_train_dataset_for_folder(dataset_folder, normalize=False):
  return GenericFromFolderDataset(dataset_folder, transform=get_train_transforms(normalize=normalize))


def get_test_dataset_for_folder(dataset_folder, normalize=False, mapping_fn = None):
  return GenericFromFolderDataset(dataset_folder, transform=get_validation_transforms(normalize=normalize), mask_transform_fn = mapping_fn)
