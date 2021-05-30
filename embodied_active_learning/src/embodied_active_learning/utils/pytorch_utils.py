"""
    Helper class with functionalities that are often used with pytorch
"""
import torch
from PIL import Image
import torch.utils.data as torchData
import numpy as np
import os


def semseg_compute_confusion(y_hat_lbl, y_lbl, num_classes, ignore_label):
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


def semseg_accum_confusion_to_iou(confusion_accum, ignore_zero = False):
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
    return iou_mean, iou_per_class,unseen_classes


class DataLoader:
    class DataLoaderSegmentation(torchData.Dataset):
        def __init__(self, folder_path, num_imgs = None, transform = None, limit_imgs = None, cpu_mode = False):
            super().__init__()
            self.img_files = sorted([os.path.join(folder_path,f) for f in os.listdir(folder_path) if "img" in f or "rgb" in f])
            self.mask_files = sorted([os.path.join(folder_path,f) for f in os.listdir(folder_path) if "mask" in f])

            if limit_imgs is not None:
                self.img_files = self.img_files[::len(self.img_files)//self.limit_imgs]
                self.mask_files = self.mask_files[::len(self.img_files)//self.limit_imgs]
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
            data = (np.asarray(Image.open(img_path))[:,:,0:3] / 255).transpose((2, 0, 1)).copy()
            label = np.asarray(Image.open(mask_path))[:,:].copy()
            data = {'image': torch.from_numpy(data).float(), 'mask':torch.from_numpy(label).float()}
            if torch.cuda.is_available() and not self.cpu_mode:
                data['image'] = data['image'].cuda()
                data['mask'] = data['mask'].cuda()

            if self.transform != None:
                data = self.transform(data)
            return data

        def __len__(self):
            return len(self.img_files)

class Transforms:
    class Normalize:
        def __init__(self, mean, std, cpu_mode = False):
            self.mean = mean
            self.std = std
            if torch.cuda.is_available() and not cpu_mode:
                self.mean = self.mean.cuda()
                self.std = self.std.cuda()

        def __call__(self, sample):
            image, mask = sample['image'], sample['mask']
            image = (image - self.mean) / (self.std)
            return {'image': image, 'mask': mask}

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