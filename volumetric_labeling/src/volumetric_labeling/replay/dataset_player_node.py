#!/usr/bin/env python
"""
Node that replays a dataset
"""
from typing import List

import rospy

from embodied_active_learning.utils.config import Configs
from embodied_active_learning.online_learning.sample import TrainSample
from embodied_active_learning.uncertainty_estimation.uncertainty_estimation_node import get_uncertainty_estimator_for_config

import os
import pickle
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import tf
from scipy.ndimage import median_filter
import datetime
import cv2

LABELING_STRATEGY_UNIFORM = "UNIFORM"
LABELING_STRATEGY_RANDOM = "RANDOM"
LABELING_STRATEGY_GT = "GT"
LABELING_STRATEGY_UNSEEN_VOXELS = "UNSEEN_VOXELS"
LABELING_STRATEGY_NONE = "NONE"
LABELING_STRATEGY_WEIGHTS = "WEIGHTS"
LABELING_STRATEGY_MOST_SEEN_VOXELS = "MOST_SEEN_VOXELS"
LABELING_STRATEGY_UNCERTAINTY = "UNCERTAINTY"
LABELING_STRATEGY_UNCERTAINTY_RAW = "UNCERTAINTY_RAW"
LABELING_STRATEGY_MODEL_MAP_MISSMATCH = "MODEL_MAP_MISSMATCH"

NYU_ID_TO_NAME = { 0: 'wall',
                   1: 'floor',
                   2: 'cabinet',
                   3: 'bed',
                   4: 'chair',
                   5: 'sofa',
                   6: 'table',
                   7: 'door',
                   8: 'window',
                   9: 'bookshelf',
                   10: 'picture',
                   11: 'counter',
                   12: 'blinds',
                   13: 'desk',
                   14: 'shelves',
                   15: 'curtain',
                   16: 'dresser',
                   17: 'pillow',
                   18: 'mirror',
                   19: 'floor mat',
                   20: 'clothes',
                   21: 'ceiling',
                   22: 'books',
                   23: 'refridgerator',
                   24: 'television',
                   25: 'paper',
                   26: 'towel',
                   27: 'shower curtain',
                   28: 'box',
                   29: 'whiteboard',
                   30: 'person',
                   31: 'night stand',
                   32: 'toilet',
                   33: 'sink',
                   34: 'lamp',
                   35: 'bathtub',
                   36: 'bag',
                   37: 'otherstructure',
                   38: 'otherfurniture',
                   39: 'otherprop'}



class DatasetPlayer:

  def __init__(self):
    '''  Initialize ros node and read params '''

    print("ALL", rospy.get_param("~"))
    depth_topic = rospy.get_param("~depth_topic", "depth")
    img_topic = rospy.get_param("~img_topic", "rgb")
    semseg_topic = rospy.get_param("~semseg_topic", "semseg")
    uncertainty_topic = rospy.get_param("~uncertainty_topic", "uncertainty")

    self.depth_pub = rospy.Publisher(depth_topic, Image)
    self.rgb_pub = rospy.Publisher(img_topic, Image)
    self.semseg_pub = rospy.Publisher(semseg_topic, Image)
    self.uncertainty_pub = rospy.Publisher(uncertainty_topic, Image)
    self.cv_bridge = CvBridge()
    self.br = tf.TransformBroadcaster()

    self.output_folder = rospy.get_param("~out_folder", None)


    if self.output_folder is None or not os.path.exists(self.output_folder):
      rospy.logerr(f"Missing output folder: {self.output_folder}")
      rospy.signal_shutdown("")
      return

    replay_path = rospy.get_param("~experiment_path", None)

    if replay_path is None or not os.path.exists(replay_path):
      rospy.logerr("Did not find experiment. Is experiment_path set correctly?")
      rospy.signal_shutdown("")
      return

    step_to_replay = rospy.get_param("~step", "step_000")
    full_path = os.path.join(replay_path, "online_training", step_to_replay)
    num_images_to_label = rospy.get_param("~images_to_label", 10)

    if not os.path.exists(full_path):
      rospy.logerr(f"Did not find experiment. Is experiment_path and step set correctly? Full Path: {full_path}")
      rospy.signal_shutdown("")
      return


    self.all_samples: List[TrainSample] = []
    self.gt_labeled_indexes = []  # indexes of sample that should be labeled
    for entry in sorted(os.listdir(full_path)):
      try:
        self.all_samples.append(pickle.load(open(os.path.join(full_path, entry), "rb")))
      except Exception as e:
        rospy.logwarn(
          f"Could not load trainings data stored at {os.path.join(full_path, entry)}. Corrupted file?. Error: ", e)

    rospy.loginfo(f"Done Loading dataest. Found {len(self.all_samples)} samples")

    labeling_strategy = rospy.get_param("~labeling_strategy", LABELING_STRATEGY_UNIFORM)
    print("labeling strategy:", labeling_strategy)


    if not os.path.exists(os.path.join(self.output_folder, os.path.basename(replay_path))):
      os.mkdir(os.path.join(self.output_folder, os.path.basename(replay_path)))
    self.output_folder = os.path.join(self.output_folder, os.path.basename(replay_path))

    if not os.path.exists(os.path.join(self.output_folder, "original")):
      os.mkdir(os.path.join(self.output_folder, "original"))
      self.dump_all("original")

    if not os.path.exists(os.path.join(self.output_folder, f"dataset_{labeling_strategy}")):
      os.mkdir(os.path.join(self.output_folder, f"dataset_{labeling_strategy}"))

    self.output_folder = os.path.join(self.output_folder, f"dataset_{labeling_strategy}")

    self.output_name = f"dataset_{labeling_strategy}_labeled_imgs_{num_images_to_label}_median_filter_10_pc_distance_5"

    if os.path.exists(os.path.join(self.output_folder, self.output_name)):
      print("Folder exists!")
      exit()


    if labeling_strategy == LABELING_STRATEGY_UNIFORM:
      self.gt_labeled_indexes = [*np.linspace(0, len(self.all_samples) - 1, num_images_to_label).astype(int)]

    elif labeling_strategy == LABELING_STRATEGY_NONE:
      pass

    elif labeling_strategy == LABELING_STRATEGY_RANDOM:
      import random
      self.gt_labeled_indexes = random.sample(range(len(self.all_samples) - 1), num_images_to_label)

    elif labeling_strategy == LABELING_STRATEGY_GT:
      self.gt_labeled_indexes = [*range(len(self.all_samples))]

    elif labeling_strategy == LABELING_STRATEGY_UNCERTAINTY:
      c = Configs(os.path.basename(self.output_folder))
      c.print_config()
      net, est = get_uncertainty_estimator_for_config(c)
      for s in self.all_samples:
        s.uncertainty = est.predict(s.image, None)[1] # [1] for uncertainty

      from embodied_active_learning.replay.pointcloud_helper import ImagePointcloudCalculatorUncertainty
      calc = ImagePointcloudCalculatorUncertainty(self.all_samples)
      self.gt_labeled_indexes = []
      for _ in range(num_images_to_label):  # request 10 images
        next_img_idx = np.argmax(calc.pc_idx_to_intersection)
        print(f"Most unlabeled points in image {next_img_idx}. #{calc.pc_idx_to_intersection[next_img_idx]} Points")
        self.gt_labeled_indexes.append(next_img_idx)
        calc.mark_as_labeled(next_img_idx)

    elif labeling_strategy == LABELING_STRATEGY_UNCERTAINTY_RAW:
      c = Configs(os.path.basename(self.output_folder))
      c.print_config()
      net, est = get_uncertainty_estimator_for_config(c)
      for s in self.all_samples:
        s.uncertainty = est.predict(s.image, None)[1] # [1] for uncertainty
      self.gt_labeled_indexes = [*np.asarray([np.mean(s.uncertainty) for s in self.all_samples]).argsort()[-num_images_to_label:]]
      print("indexes: ", self.gt_labeled_indexes)


    elif labeling_strategy == LABELING_STRATEGY_MODEL_MAP_MISSMATCH:
      c = Configs(os.path.basename(self.output_folder))
      c.print_config()
      net, est = get_uncertainty_estimator_for_config(c)
      for s in self.all_samples:
        s.uncertainty = (s.mask != (est.predict(s.image, None)[0])).astype(np.float32)

      from embodied_active_learning.replay.pointcloud_helper import ImagePointcloudCalculatorUncertainty
      calc = ImagePointcloudCalculatorUncertainty(self.all_samples)
      self.gt_labeled_indexes = []
      for _ in range(num_images_to_label):  # request 10 images
        next_img_idx = np.argmax(calc.pc_idx_to_intersection)
        print(f"Most unlabeled points in image {next_img_idx}. #{calc.pc_idx_to_intersection[next_img_idx]} Points")
        self.gt_labeled_indexes.append(next_img_idx)
        calc.mark_as_labeled(next_img_idx)

    elif labeling_strategy == LABELING_STRATEGY_WEIGHTS:
      self.gt_labeled_indexes = [*np.asarray([np.mean(s.weights) for s in self.all_samples]).argsort()[:num_images_to_label]]

    elif labeling_strategy == LABELING_STRATEGY_MOST_SEEN_VOXELS:
      from embodied_active_learning.replay.pointcloud_helper import ImagePointcloudCalculatorMostSeen
      calc = ImagePointcloudCalculatorMostSeen(self.all_samples)
      self.gt_labeled_indexes = []
      for _ in range(num_images_to_label):  # request 10 images
        next_img_idx = np.argmax(calc.pc_idx_to_intersection)
        print(f"Most unlabeled points in image {next_img_idx}. #{calc.pc_idx_to_intersection[next_img_idx]} Points")
        self.gt_labeled_indexes.append(next_img_idx)
        calc.mark_as_labeled(next_img_idx)

    elif labeling_strategy == LABELING_STRATEGY_UNSEEN_VOXELS:
      from embodied_active_learning.replay.pointcloud_helper import ImagePointcloudCalculator
      calc = ImagePointcloudCalculator(self.all_samples)
      self.gt_labeled_indexes  = []
      for _ in range(num_images_to_label):  # request 10 images
        next_img_idx = np.argmax(calc.pc_idx_to_intersection)
        print(f"Most unlabeled points in image {next_img_idx}. #{calc.pc_idx_to_intersection[next_img_idx]} Points")
        self.gt_labeled_indexes .append(next_img_idx)
        calc.mark_as_labeled(next_img_idx)

    else:
      rospy.logerr(f"Unknown labeling strategy {labeling_strategy}. Going to stop data player")
      rospy.signal_shutdown("")

    rospy.loginfo("Labeled images:" + str(self.gt_labeled_indexes))
    self.current_sample_idx = 0

    image_idx = []
    image_idx.extend(self.gt_labeled_indexes[::-1])
    image_idx.extend([i for i in range(len(self.all_samples)) if i not in self.gt_labeled_indexes])
    image_idx.append(len(self.all_samples))
    image_idx.append(len(self.all_samples) + 1)
    print("Image idx:", image_idx)
    self.sample_iter = iter(image_idx)

  @staticmethod
  def dump_images_to_folder(samples, idx_to_label, out_folder="", valid_size=15):
    import random
    import numpy as np
    # Available idxs:
    all_idxs = [i for i in range(len(samples)) if i not in idx_to_label]
    if len(all_idxs) < 15:
      all_idxs = [*range(len(samples))]

    valid_split = random.choices(all_idxs, k=valid_size)
    os.mkdir(os.path.join(out_folder, "train"))
    os.mkdir(os.path.join(out_folder, "valid"))

    base_folder = out_folder
    for idx, sample in enumerate(samples):
      out_folder = os.path.join(base_folder, "train" if idx not in valid_split else "valid")
      cv2.imwrite(os.path.join(out_folder, f"{idx:04d}_image.png"), cv2.cvtColor(sample.image, cv2.COLOR_BGR2RGB))
      if idx not in idx_to_label:
        cv2.imwrite(os.path.join(out_folder, f"{idx:04d}_mask.png"), sample.mask)
      else:
        # corrupt gt image and save it
        cv2.imwrite(os.path.join(out_folder, f"{idx:04d}_mask.png"), median_filter(sample.gt_mask, size=10))

      sample.weights = sample.weights / (np.max(sample.weights))
      np.save(open(os.path.join(out_folder, f"{idx:04d}_weights.npy"), "wb"), sample.weights)


  def dump_all(self, folder):
   out = os.path.join(self.output_folder, folder)
   if not os.path.exists(out):
     os.mkdir(out)

   print("Scoring dataset before dumping. Dataset: {folder}")
   import pandas as pd
   mIoU, classIoU, unseenClasses = self.get_miou([m.mask for m in self.all_samples],
                                                 [m.gt_mask for m in self.all_samples])
   data = {
     'mIoU': mIoU
   }

   for idx, value in enumerate(classIoU.detach().numpy()):
     data[NYU_ID_TO_NAME[idx]] = value

   print(f"mIoU: {mIoU}")
   pd.DataFrame.from_dict([data]).to_csv(os.path.join(out, f"scores.csv"), index=False)
   pd.DataFrame(data = np.asarray(self.gt_labeled_indexes)).to_csv(os.path.join(out, f"selected_imgs.csv"), index=False)

   self.dump_images_to_folder(self.all_samples, self.gt_labeled_indexes, out_folder=out)
   # do not skip. way too many experiments right now
   # TODO uncomment
   # print(f"Dumping dataset to :{out}")
   # for entry in self.all_samples:
   #   pickle.dump(entry, open(os.path.join(out, f"training_entry_{entry.number:03}.pkl"), "wb"))
   # rospy.signal_shutdown("done")


  def dump_all(self, folder, dump_images = True, score_name = "scores.csv"):
   out = os.path.join(self.output_folder, folder)
   if not os.path.exists(out):
     os.mkdir(out)

   print("Scoring dataset before dumping. Dataset: {folder}")
   import pandas as pd
   mIoU, classIoU, unseenClasses = self.get_miou([m.mask for m in self.all_samples],
                                                 [m.gt_mask for m in self.all_samples])
   data = {
     'mIoU': mIoU,
     'fullmIoU': np.mean(classIoU.detach().cpu().numpy())
   }

   for idx, value in enumerate(classIoU.detach().numpy()):
     data[NYU_ID_TO_NAME[idx]] = value

   print(f"mIoU: {mIoU}")
   pd.DataFrame.from_dict([data]).to_csv(os.path.join(out, score_name), index=False)
   pd.DataFrame(data = np.asarray(self.gt_labeled_indexes)).to_csv(os.path.join(out, f"selected_imgs.csv"), index=False)
   if dump_images:
    self.dump_images_to_folder(self.all_samples, self.gt_labeled_indexes, out_folder=out)

   # do not skip. way too many experiments right now
   # TODO uncomment
   # print(f"Dumping dataset to :{out}")
   # for entry in self.all_samples:
   #   pickle.dump(entry, open(os.path.join(out, f"training_entry_{entry.number:03}.pkl"), "wb"))
   # rospy.signal_shutdown("done")


  @staticmethod
  def get_miou(prediction, gt, ignore_label = 255, num_classes = 40):
    from embodied_active_learning.utils import pytorch_utils
    from densetorch.engine.miou import fast_cm
    import torch

    conf_mat = 0*np.eye(40)
    for idx in range(len(prediction)):
      pseudo_mask = prediction[idx]
      gt_mask = gt[idx]
      valid = pseudo_mask != ignore_label
      conf_mat += fast_cm(
        pseudo_mask[valid], gt_mask[valid], num_classes
      )
    mIoU, classIoU, unseenClasses = pytorch_utils.semseg_accum_confusion_to_iou(torch.from_numpy(conf_mat),
                                                                                ignore_zero=True)
    return  mIoU.detach().numpy().item(), classIoU, unseenClasses


  def experiment_end_cb(self, score_name = "scores.csv", create_ds = True):
    from embodied_active_learning.pseudo_labels.pseudo_labler import PseudoLabler
    from embodied_active_learning.utils.config import PseudoLablerConfig
    config = PseudoLablerConfig()
    config.weights_method = "uncertainty"

    PseudoLabler(config, topic='/volumetric_labler_node/panoptic/render_camera_view').label_many(self.all_samples, cache_time=0)
    self.dump_all(self.output_name, dump_images=create_ds, score_name = score_name)

    rospy.signal_shutdown("End of Experiment reached")



  def publish_images(self, timer_event):
    if self.current_sample_idx == len(self.all_samples):
      rospy.loginfo("End of experiment reached. Replayed all images")
      next(self.sample_iter)
      self.experiment_end_cb(create_ds=False)

    # if self.current_sample_idx == 66 or self.current_sample_idx == 112 or self.current_sample_idx == 132:
    #   self.current_sample_idx += 1 # skip this one
    #   return

    if self.current_sample_idx >= len(self.all_samples):
      return

    rospy.loginfo(
      f"Publishing sample: {self.current_sample_idx + 1}/{len(self.all_samples)} ({100 * (self.current_sample_idx + 1) / len(self.all_samples):.2f}%)")


    current_train_sample: TrainSample = self.all_samples[self.current_sample_idx]
    depth_msg = current_train_sample.depth

    try:
      ts = rospy.Time.now()
      self.br.sendTransform(current_train_sample.pose[0], current_train_sample.pose[1],
                            ts, depth_msg.header.frame_id, "world")


      if self.current_sample_idx in self.gt_labeled_indexes:
        rospy.loginfo(f"Going to publish GT mask for this image")
        rgb_msg = self.cv_bridge.cv2_to_imgmsg(current_train_sample.image, "rgb8")
        # Corrupt GT mask
        semseg_msg = self.cv_bridge.cv2_to_imgmsg(median_filter(current_train_sample.gt_mask.astype(np.uint8), size = 10), "mono8")
        uncertainty_msg = self.cv_bridge.cv2_to_imgmsg(current_train_sample.weights.astype(np.float32) * 0 - 1, "32FC1")
      else:
        rgb_msg = self.cv_bridge.cv2_to_imgmsg(current_train_sample.image, "rgb8")
        semseg_msg = self.cv_bridge.cv2_to_imgmsg(current_train_sample.mask.astype(np.uint8), "mono8")
        uncertainty_msg = self.cv_bridge.cv2_to_imgmsg(current_train_sample.weights.astype(np.float32), "32FC1")

      depth_msg.header.stamp = ts
      rgb_msg.header = depth_msg.header
      semseg_msg.header = depth_msg.header
      uncertainty_msg.header = depth_msg.header

      self.depth_pub.publish(depth_msg)
      self.rgb_pub.publish(rgb_msg)
      self.semseg_pub.publish(semseg_msg)
      self.uncertainty_pub.publish(uncertainty_msg)


    except CvBridgeError as e:
      rospy.logerr(str(e))

    self.current_sample_idx = next(self.sample_iter)


if __name__ == '__main__':
  rospy.init_node('dataset_player_node', anonymous=True)
  player = DatasetPlayer()
  # Set up timer
  replay_rate = rospy.get_param("~replay_rate", 1)
  rospy.Timer(rospy.Duration(1 / replay_rate), player.publish_images)
  # Spin
  rospy.spin()
