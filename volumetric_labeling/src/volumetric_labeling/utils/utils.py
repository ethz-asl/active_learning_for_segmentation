from typing import List

import cv2
from scipy.ndimage import median_filter
import os
import random
import numpy as np

from densetorch.engine.miou import fast_cm
import torch

from embodied_active_learning_core.online_learning.sample import TrainSample
from embodied_active_learning_core.utils.pytorch.evaluation import semseg_accum_confusion_to_iou


def dump_images_to_folder(samples: List[TrainSample], idx_to_label: List[int], out_folder: str = "",
                          valid_size: int = 15):
  """
  Creates a Train and Validation dataset containing image and masks.

  Parameters:
  samples: List with TrainSamples containing projected labels already
  idx_to_label: List with image idxs which should use the gt mask
  out_folder: where to create the folder
  valid_size: Percent of validation set size

  Returns: Nothing
  """

  imgs_idx = [*range(len(samples))]
  valid_split = random.choices(imgs_idx, k=valid_size)
  os.mkdir(os.path.join(out_folder, "train"))
  os.mkdir(os.path.join(out_folder, "valid"))
  base_folder = out_folder

  for idx, sample in enumerate(samples):
    out_folder = os.path.join(base_folder, "train" if idx not in valid_split else "valid")
    cv2.imwrite(os.path.join(out_folder, f"{idx:04d}_image.png"), cv2.cvtColor(sample.image, cv2.COLOR_BGR2RGB))

    if idx not in idx_to_label:  # This images were labeled, directly store the gt mask not the projection
      cv2.imwrite(os.path.join(out_folder, f"{idx:04d}_mask.png"), sample.mask)
    else:
      # corrupt gt image and save it
      cv2.imwrite(os.path.join(out_folder, f"{idx:04d}_mask.png"), median_filter(sample.gt_mask, size=10))

    sample.weights = sample.weights / (np.max(sample.weights))
    np.save(open(os.path.join(out_folder, f"{idx:04d}_weights.npy"), "wb"), sample.weights)


def get_miou(prediction, gt, ignore_label=255, num_classes=40, only_gt_classes=True):
  gt_classes = np.asarray([])

  conf_mat = 0 * np.eye(num_classes)
  for idx in range(len(prediction)):
    pseudo_mask = prediction[idx]
    gt_mask = gt[idx]
    valid = pseudo_mask != ignore_label
    conf_mat += fast_cm(
      pseudo_mask[valid], gt_mask[valid], num_classes
    )

    gt_classes = np.unique(np.append(gt_classes, gt_mask[valid].ravel()))

  iou_per_class = semseg_accum_confusion_to_iou(torch.from_numpy(conf_mat)).detach().numpy()
  miou = np.mean(iou_per_class[np.unique(gt_classes.astype(int))] if only_gt_classes else iou_per_class)

  return miou, iou_per_class
