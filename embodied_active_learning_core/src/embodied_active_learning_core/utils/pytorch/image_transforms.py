import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

IMG_SCALE = 1.0 / 255
IMG_MEAN = np.array([0.72299159, 0.67166396, 0.63768772]).reshape((1, 1, 3))
IMG_STD = np.array([0.2327359, 0.24695725, 0.25931836]).reshape((1, 1, 3))


def prepare_img(img, normalize=False):
  """ Divides to input image by 255. If normalized selected, also normalizes the image by subtracting mean and dividng by STD """
  return (img * IMG_SCALE - IMG_MEAN) / IMG_STD if normalize else img * IMG_SCALE


def get_validation_transforms(img_height=480, img_width=640, normalize=True,
                              additional_targets={'mask': 'mask', 'weight': 'mask'}):
  transforms = []
  if normalize:
    transforms.append(A.Normalize(mean=IMG_MEAN, std=IMG_STD))
  else:
    transforms.append(A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)))

  transforms.append(ToTensorV2())

  return A.Compose(transforms, additional_targets=additional_targets)


def get_train_transforms(img_height=480, img_width=640, normalize=True,
                         additional_targets={'mask': 'mask', 'weight': 'mask'}):
  transforms = [
    A.Resize(img_height, img_width),
    A.HorizontalFlip(p=0.5),
    A.RandomSizedCrop(min_max_height=(img_height // 2, img_height), height=img_height, width=img_width, p=0.5),
    A.RandomBrightnessContrast(p=0.8),
    A.RandomGamma(p=0.8)
  ]

  if normalize:
    transforms.append(A.Normalize(mean=IMG_MEAN, std=IMG_STD))
  else:
    transforms.append(A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)))

  transforms.append(ToTensorV2())

  return A.Compose(transforms, additional_targets=additional_targets)


class Transforms:
  """ Frequently used Image transforms """

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
