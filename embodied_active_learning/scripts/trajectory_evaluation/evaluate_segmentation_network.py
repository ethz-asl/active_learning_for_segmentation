"""
 Evaluates a given semantic network on the specified testset
"""
import argparse
import os.path

import numpy as np
import torch.nn.functional as F
import torch
import torch.utils.data as data

from torchvision import transforms
from refinenet.models.resnet import rf_lw50, rf_lw101, rf_lw152

import os
import time
import pandas as pd
from embodied_active_learning.utils import pytorch_utils
from embodied_active_learning.utils.airsim import airsim_semantics

from embodied_active_learning.online_learning.online_learner import get_online_learning_refinenet

from densetorch.engine.miou import fast_cm


def getListOfCheckpoints(dirName):
  if dirName is None:
    return None
  # create a list of file and sub directories
  # names in the given directory
  listOfFile = os.listdir(dirName)
  allFiles = list()
  # Iterate over all the entries
  for entry in listOfFile:
    # Create full path
    fullPath = os.path.join(dirName, entry)
    # If entry is a directory then get the list of files in this directory
    if os.path.isdir(fullPath):
      allFiles = allFiles + getListOfCheckpoints(fullPath)
    elif ".pth" in fullPath:
      allFiles.append(fullPath)

  return allFiles


parser = argparse.ArgumentParser(
  description='Evaluates a given model on the testset')

parser.add_argument('--testset_folder',
                    help='Folder where testset images are stored',
                    default="/home/rene/thesis/test_data/",
                    type=str)

parser.add_argument('--network',
                    help='Folder where testset images are stored',
                    default="refinenet_50",
                    type=str)

parser.add_argument('--checkpoint',
                    help='Folder where testset images are stored',
                    default=None,
                    # "/home/rene/catkin_ws/src/active_learning_for_segmentation/embodied_active_learning/refinenet/checkpoints/checkpoint.pth.tar",
                    type=str)

parser.add_argument('--semantics_mapping',
                    default="/home/rene/catkin_ws/src/active_learning_for_segmentation/embodied_active_learning/cfg/airsim/semanticClasses.yaml",
                    # "/home/rene/catkin_ws/src/active_learning_for_segmentation/embodied_active_learning/refinenet/checkpoints/checkpoint.pth.tar",
                    type=str)

args = parser.parse_args()
testsetFolder = args.testset_folder
network = None
checkpoint = args.checkpoint
if args.network == "refinenet_50":
  network = rf_lw50(40, pretrained=True)
elif args.network == "refinenet_101":
  network = rf_lw101(40, pretrained=True)
elif args.network == "refinenet_152":
  network = rf_lw152(40, pretrained=True)
elif args.network == "GMM_refinenet_50":
  network = get_online_learning_refinenet(50, 40, True, save_path="", model_slug="",
                                          with_uncertainty=True).model
elif args.network == "GMM_refinenet_101":
  network = get_online_learning_refinenet(101, 40, True, save_path="", model_slug="",
                                          with_uncertainty=True).model
elif args.network == "GMM_refinenet_152":
  network = get_online_learning_refinenet(152, 40, True, save_path="", model_slug="",
                                          with_uncertainty=True).model
else:
  print("network {} not found!".format(args.network))
  exit()

# network = torch.nn.DataParallel(network)
nyuMappingsYaml = args.semantics_mapping
airSimSemanticsConverter = airsim_semantics.AirSimSemanticsConverter(nyuMappingsYaml)

# Taken from Refinenet Repo
# IMG_MEAN = torch.tensor(np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)))
# IMG_STD = torch.tensor(np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1)))

IMG_SCALE = 1.0 / 255
IMG_MEAN =  torch.tensor(np.array([0.72299159, 0.67166396, 0.63768772]).reshape((3, 1, 1)))
IMG_STD = torch.tensor(np.array([0.2327359 , 0.24695725, 0.25931836]).reshape((3, 1, 1)))

transform = transforms.Compose(
  [pytorch_utils.Transforms.Normalize(IMG_MEAN, IMG_STD), pytorch_utils.Transforms.AsFloat()])
# transform = transforms.Compose(
#   [pytorch_utils.Transforms.AsFloat()])
testLoader = data.DataLoader(
  pytorch_utils.DataLoader.DataLoaderSegmentation(testsetFolder, transform=transform, num_imgs=120),
  batch_size=8)

checkpoint_files = [checkpoint] if not os.path.isdir(checkpoint) else getListOfCheckpoints(checkpoint)

all_results = []
t = time.time()

for checkpoint in sorted(checkpoint_files, reverse=True):

  entry_as_dict = {'planner': "unknown", 'n_imgs': -1}
  if checkpoint is not None:
    print("Restoring from checkpoint:\n===> ", checkpoint)
    network.load_state_dict(torch.load(checkpoint))
    print("Successfully loaded network")

    imgs = int(checkpoint.split("_")[-1].replace(".pth", ""))
    entry_as_dict = {'cp': checkpoint, 'planner': "_", 'n_imgs': imgs}

  network.eval()

  confusion = np.zeros((40, 40))  # torch.zeros(40, 40)
  network = network.cuda()
  if torch.cuda.is_available():
    network = network.cuda()

  batches = len(testLoader)
  cnt = 0

  for batch in testLoader:
    cnt += 1
    imgs = batch['image']
    masks = batch['mask']
    h, w = masks.shape[-2:]
    with torch.no_grad():
      print("Scoring batch {}/{}".format(cnt, batches))
      predictions = network(imgs)  # [0]
      if type(predictions) == tuple:
        predictions = predictions[0]

      predictions = F.interpolate(predictions, (h, w), mode="bilinear", align_corners=False)
      predictions_categorical = torch.argmax(predictions, dim=-3)
      idx = masks <= 40

      confusion_matrix = fast_cm(
        predictions_categorical[idx].cpu().detach().numpy().astype(np.uint8),
        masks[idx].cpu().detach().numpy().astype(np.uint8), 40
      )

      confusion += confusion_matrix

  mIoU, classIoU, unseenClasses = pytorch_utils.semseg_accum_confusion_to_iou(torch.from_numpy(confusion),
                                                                              ignore_zero=False)

  print("IoU per Classes:")
  for idx, IoU in enumerate(classIoU.cpu().detach().numpy()):
    print("  {:<15}:{:.1f}%".format(airSimSemanticsConverter.get_nyu_name_for_nyu_id(idx), IoU))
    entry_as_dict[airSimSemanticsConverter.get_nyu_name_for_nyu_id(idx)] = IoU
  print("mIoU:")
  print("  {:.1f}%".format(mIoU.cpu().detach().numpy().item()))
  print("Acc:")
  print("  {:.2f}%".format(np.sum(np.diag(confusion)) / (np.sum(confusion)) * 100))

  entry_as_dict['mIoU'] = mIoU.cpu().detach().numpy().item()
  entry_as_dict['acc'] = np.sum(np.diag(confusion)) / (np.sum(confusion))
  all_results.append(entry_as_dict)

  df = pd.DataFrame(all_results)
  name = args.checkpoint
  if name[-1] == "/":
    name = name[:-1]
  df.to_csv('results_{}_{}.csv'.format(os.path.basename(name), t))
