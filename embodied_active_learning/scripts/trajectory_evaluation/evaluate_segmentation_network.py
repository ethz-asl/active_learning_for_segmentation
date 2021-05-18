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
import json
import os
import time
import pandas as pd
from embodied_active_learning.airsim_utils import semantics
from embodied_active_learning.utils import pytorch_utils


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
    elif "checkpoint.pth.tar" in fullPath:
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
                    default="refinenet_101",
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
if args.network == "refinenet_101":
  network = rf_lw50(40, pretrained=True)
elif args.network == "refinenet_50":
  network = rf_lw101(40, pretrained=True)
elif args.network == "refinenet_152":
  network = rf_lw152(40, pretrained=True)
else:
  print("network {} not found!".format(args.network))
  exit()

network = torch.nn.DataParallel(network)
nyuMappingsYaml = args.semantics_mapping
airSimSemanticsConverter = semantics.AirSimSemanticsConverter(nyuMappingsYaml)

# Taken from Refinenet Repo
IMG_MEAN = torch.tensor(np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)))
IMG_STD = torch.tensor(np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1)))
transform = transforms.Compose(
  [pytorch_utils.Transforms.Normalize(IMG_MEAN, IMG_STD), pytorch_utils.Transforms.AsFloat()])
testLoader = data.DataLoader(pytorch_utils.DataLoader.DataLoaderSegmentation(testsetFolder, transform=transform),
                             batch_size=8)

checkpoint_files = [checkpoint] if not os.path.isdir(checkpoint) else getListOfCheckpoints(checkpoint)

all_results = []

for checkpoint in checkpoint_files:

  entry_as_dict = {'planner': "unknown", 'n_imgs': -1}
  if checkpoint is not None:
    print("Restoring from checkpoint:\n===> ", checkpoint)

    network.load_state_dict(torch.load(checkpoint)['model'])
    print("Successfully loaded network")

    data = json.load(open(os.path.split(checkpoint)[-2] + "/args.json", 'r'))
    name = os.path.basename(data.get('train_path', "unknown")).replace("experiment_", "")
    n_imgs = data.get('num_imgs', -1)
    entry_as_dict = {'planner': name, 'n_imgs': n_imgs}

  network.eval()

  confusion = torch.zeros(40, 40)

  if torch.cuda.is_available():
    confusion = confusion.cuda()

  batches = len(testLoader)
  cnt = 0

  for batch in testLoader:
    cnt += 1
    imgs = batch['image']
    masks = batch['mask']
    h, w = masks.shape[-2:]
    with torch.no_grad():
      print("Scoring batch {}/{}".format(cnt, batches))
      predictions = network(imgs)
      predictions = F.interpolate(predictions, (h, w), mode="bilinear", align_corners=False)
      predictions_categorical = torch.argmax(predictions, dim=-3)
      confusion_matrix = pytorch_utils.semseg_compute_confusion(predictions_categorical, masks, num_classes=40,
                                                                ignore_label=None)
      confusion += confusion_matrix

  mIoU, classIoU, unseenClasses = pytorch_utils.semseg_accum_confusion_to_iou(confusion, ignore_zero=True)

  print("IoU per Classes:")
  for idx, IoU in enumerate(classIoU.cpu().detach().numpy()):
    print("  {:<15}:{:.1f}%".format(airSimSemanticsConverter.get_nyu_name_for_nyu_id(idx), IoU))
    entry_as_dict[airSimSemanticsConverter.get_nyu_name_for_nyu_id(idx)] = IoU
  print("mIoU:")
  print("  {:.1f}%".format(mIoU.cpu().detach().numpy().item()))

  entry_as_dict['mIoU'] = mIoU.cpu().detach().numpy().item()
  all_results.append(entry_as_dict)

df = pd.DataFrame(all_results)
df.to_csv('results_{}.csv'.format(time.time()))
