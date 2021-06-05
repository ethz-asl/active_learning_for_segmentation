"""
Samples a certain amount of random viewpoints and stores the RGBD Images + GT Semantic Classes
"""
import pickle
import random
import torch
from embodied_active_learning.utils.pytorch_utils import batch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from refinenet.models.resnet import rf_lw50, rf_lw101, rf_lw152

from refinenet.utils.helpers import prepare_img
import os
import re
import argparse

from embodied_active_learning.airsim_utils import semantics


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_encoder_and_decoder_params(model):
  """Filter model parameters into two groups: encoder and decoder."""
  enc_params = []
  dec_params = []
  for k, v in model.named_parameters():
    if bool(re.match(".*conv1.*|.*bn1.*|.*layer.*", k)):
      enc_params.append(v)
    else:
      dec_params.append(v)
  return enc_params, dec_params


def create_optimizer(optim_type, parameters, **kwargs):
  if optim_type.lower() == "sgd":
    optim = torch.optim.SGD
  elif optim_type.lower() == "adam":
    optim = torch.optim.Adam
  else:
    raise ValueError(
      "Optim {} is not supported. "
      "Only supports 'SGD' and 'Adam' for now.".format(optim_type)
    )
  # args = get_args(optim)
  # kwargs = {key: kwargs[key] for key in args if key in kwargs}
  return optim(parameters, **kwargs)

parser = argparse.ArgumentParser(
    description='Creates test set with random images from environment')


parser.add_argument('--imgs_folder',
                    help='Output folder where images should be saved',
                    default="/home/rene/thesis/online_learning_imgs_1/",
                    type=str)

parser.add_argument('--info_file',
                    help='info file',
                    default="/home/rene/thesis/online_learning_imgs_1/info.txt",
                    type=str)

parser.add_argument('--checkpoint_path',
                    help='info file',
                    default="/home/rene/thesis/online_learning_imgs_1/info.txt",
                    type=str)

parser.add_argument('--batch_size',
                    help='batch size',
                    default= 6,
                    type=int)

parser.add_argument('--save_path',
                    default= "/home/rene/thesis/online_learning_imgs_1/",
                    type=str)

parser.add_argument('--model_slug',
                    help='',
                    default= "online_lr_e_4_d_4",
                    type=str)

parser.add_argument('--enc_optim_type',
                    default= "sgd",
                    type=str)

parser.add_argument('--dec_optim_type',
                    default= "sgd",
                    type=str)

parser.add_argument('--dec_lr',
                    default= 4e-4,
                    type=float)
parser.add_argument('--enc_lr',
                    default= 4e-4,
                    type=float)


parser.add_argument('--enc_momentum',
                    default= 0.9,
                    type=float)
parser.add_argument('--dec_momentum',
                    default= 0.9,
                    type=float)

parser.add_argument(
    '--nyu_mapping',
    help='Path to nyu mapping .yaml',
    default=
    "/home/rene/catkin_ws/src/active_learning_for_segmentation/embodied_active_learning/cfg/airsim/semanticClasses.yaml",
    type=str)


args = parser.parse_args()
air_sim_semantics_converter = semantics.AirSimSemanticsConverter(args.nyu_mapping)

with open(args.info_file, "r") as f:
    lines = [line.replace("[","").replace("]","").replace(";","").replace("\n","") for line in f]

l = []
for line in lines:
    l2 = []
    for i in line.split(","):
        l2.append(int(i))
    l.append(l2)






#####################################################################3



model = rf_lw101(40, pretrained=True).cuda()



enc_optim_type = args.enc_optim_type
enc_lr = args.enc_lr
enc_weight_decay = 0
enc_momentum = args.enc_momentum
dec_optim_type = args.dec_optim_type
dec_lr = args.dec_lr
dec_weight_decay = 0
dec_momentum = args.dec_momentum


enc_params, dec_params = get_encoder_and_decoder_params(model)
crit = nn.CrossEntropyLoss()

optimisers = [
    create_optimizer(
        optim_type=enc_optim_type,
        parameters=enc_params,
        lr=enc_lr,
        weight_decay=enc_weight_decay,
        momentum=enc_momentum,
    ),
    create_optimizer(
        optim_type=dec_optim_type,
        parameters=dec_params,
        lr=dec_lr,
        weight_decay=dec_weight_decay,
        momentum=dec_momentum,
    )
]

cnt = 0
for buffer in l:
    imgs = [ os.path.join(args.imgs_folder, "img_{:04d}.png".format(cnt)) for cnt in buffer]
    masks = [ os.path.join(args.imgs_folder, "mask_{:04d}.png".format(cnt)) for cnt in buffer]

    imgs = [torch.tensor(prepare_img(np.asarray(Image.open(img))[:,:,0:3].copy()).transpose(2, 0, 1)[None]).float() for img in imgs]
    masks = [torch.tensor(air_sim_semantics_converter.map_infrared_to_nyu(np.asarray(Image.open(img_gt)).copy())).long() for img_gt in masks]
    to_be_used = []

    for img,mask in zip(imgs,masks):
        to_be_used.append({'image': img, 'mask': mask})

    loss = torch.tensor(0.0).cuda()

    for b in batch(to_be_used, args.batch_size):
      target = torch.stack([item['mask'] for item in b]).cuda()
      predictions = model(torch.stack([item['image'] for item in b]).cuda().squeeze(dim = 1))

      loss += crit(
          F.interpolate(
            predictions, size=target.size()[1:], mode="bilinear", align_corners=False
          ).squeeze(dim=1),
          target,
        )


    for opt in optimisers:
      opt.zero_grad()
    loss.backward()

    for opt in optimisers:
      opt.step()
    cnt+=1

    print("Step:{}, Loss: {}".format(cnt, loss.cpu().item()))

torch.save(model.state_dict(), os.path.join(args.save_path, "{}_checkpoint.pth.tar".format(args.model_slug)))
print("Saved model to ", os.path.join(args.save_path, "{}_checkpoint.pth.tar".format(args.model_slug)))