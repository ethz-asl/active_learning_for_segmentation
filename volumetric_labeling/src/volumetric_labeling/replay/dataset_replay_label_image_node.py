#!/usr/bin/env python
"""
Node that replays a dataset and annotates on image level
"""
from typing import List
import pandas as pd
import random
import os
import pickle
import numpy as np
import tf
from scipy.ndimage import median_filter

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from embodied_active_learning_core.online_learning.sample import TrainSample
from embodied_active_learning_core.semseg.uncertainty.uncertainty_estimator import get_uncertainty_estimator_for_network
from embodied_active_learning_core.config.config import UncertaintyEstimatorConfig, NetworkConfig
from embodied_active_learning_core.semseg.models import model_manager

from volumetric_labeling.labels.pseudo_labler import PseudoLabler
from volumetric_labeling.utils.image_level_metrics import ImagePointcloudCalculatorMostUnlabeled, \
  ImagePointcloudCalculatorUnlabeledUncertainty, ImagePointcloudCalculatorMostSeen
from volumetric_labeling.utils.utils import dump_images_to_folder, get_miou

LABELING_STRATEGY_UNIFORM = "UNIFORM"
LABELING_STRATEGY_RANDOM = "RANDOM"
LABELING_STRATEGY_GT = "GT"
LABELING_STRATEGY_UNLABELED_VOXELS = "UNLABELED_VOXELS"
LABELING_STRATEGY_NONE = "NONE"
LABELING_STRATEGY_WEIGHTS = "WEIGHTS"
LABELING_STRATEGY_MOST_SEEN_VOXELS = "MOST_SEEN_VOXELS"
LABELING_STRATEGY_UNCERTAINTY = "UNCERTAINTY"
LABELING_STRATEGY_UNCERTAINTY_RAW = "UNCERTAINTY_RAW"
LABELING_STRATEGY_MODEL_MAP_MISSMATCH = "MODEL_MAP_MISSMATCH"


class DatasetPlayer:

  def __init__(self):
    '''  Initialize ros node and read params '''
    depth_topic = rospy.get_param("~depth_topic", "depth")
    img_topic = rospy.get_param("~img_topic", "rgb")
    semseg_topic = rospy.get_param("~semseg_topic", "semseg")
    uncertainty_topic = rospy.get_param("~uncertainty_topic", "uncertainty")

    self.depth_pub = rospy.Publisher(depth_topic, Image, queue_size=1)
    self.rgb_pub = rospy.Publisher(img_topic, Image, queue_size=1)
    self.semseg_pub = rospy.Publisher(semseg_topic, Image, queue_size=1)
    self.uncertainty_pub = rospy.Publisher(uncertainty_topic, Image, queue_size=1)
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

    rospy.loginfo(f"Done Loading dataset. Found {len(self.all_samples)} samples")

    labeling_strategy = rospy.get_param("~labeling_strategy", LABELING_STRATEGY_UNIFORM)

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
      rospy.logerr(f"Output Folder: {os.path.join(self.output_folder, self.output_name)} exists!")
      exit()

    # Convert labeling strategy to image idxs which should be annotated

    if labeling_strategy == LABELING_STRATEGY_UNIFORM:
      self.gt_labeled_indexes = [*np.linspace(0, len(self.all_samples) - 1, num_images_to_label).astype(int)]

    elif labeling_strategy == LABELING_STRATEGY_NONE:
      pass

    elif labeling_strategy == LABELING_STRATEGY_RANDOM:
      self.gt_labeled_indexes = random.sample(range(len(self.all_samples)), num_images_to_label)

    elif labeling_strategy == LABELING_STRATEGY_GT:
      self.gt_labeled_indexes = [*range(len(self.all_samples))]

    elif labeling_strategy == LABELING_STRATEGY_WEIGHTS:
      self.gt_labeled_indexes = [
        *np.asarray([np.mean(s.weights) for s in self.all_samples]).argsort()[:num_images_to_label]]

    elif labeling_strategy == LABELING_STRATEGY_MOST_SEEN_VOXELS:
      calc = ImagePointcloudCalculatorMostSeen(self.all_samples)
      self.gt_labeled_indexes = []
      for _ in range(num_images_to_label):  # request 10 images
        next_img_idx = np.argmax(calc.pc_idx_to_intersection)
        self.gt_labeled_indexes.append(next_img_idx)
        calc.mark_as_labeled(next_img_idx)

    elif labeling_strategy == LABELING_STRATEGY_UNLABELED_VOXELS:
      calc = ImagePointcloudCalculatorMostUnlabeled(self.all_samples)
      self.gt_labeled_indexes = []
      for _ in range(num_images_to_label):
        next_img_idx = np.argmax(calc.pc_idx_to_intersection)
        self.gt_labeled_indexes.append(next_img_idx)
        calc.mark_as_labeled(next_img_idx)
    else:
      # All of these methods need a model and maybe uncertainty
      network_config = NetworkConfig.from_ros_config()
      uncertainty_config = UncertaintyEstimatorConfig.from_ros_config()
      uncertainty_estimator = get_uncertainty_estimator_for_network(model_manager.get_model_for_config(network_config),
                                                                    uncertainty_config)

      if labeling_strategy == LABELING_STRATEGY_UNCERTAINTY:
        for s in self.all_samples:
          s.uncertainty = uncertainty_estimator.predict(s.image, None)[1]  # [1] for uncertainty

        calc = ImagePointcloudCalculatorUnlabeledUncertainty(self.all_samples)
        self.gt_labeled_indexes = []
        for _ in range(num_images_to_label):
          next_img_idx = np.argmax(calc.pc_idx_to_intersection)
          self.gt_labeled_indexes.append(next_img_idx)
          calc.mark_as_labeled(next_img_idx)

      elif labeling_strategy == LABELING_STRATEGY_UNCERTAINTY_RAW:
        for s in self.all_samples:
          s.uncertainty = uncertainty_estimator.predict(s.image, None)[1]  # [1] for uncertainty
        self.gt_labeled_indexes = [
          *np.asarray([np.mean(s.uncertainty) for s in self.all_samples]).argsort()[-num_images_to_label:]]

      elif labeling_strategy == LABELING_STRATEGY_MODEL_MAP_MISSMATCH:
        for s in self.all_samples:
          s.uncertainty = (s.mask != (uncertainty_estimator.predict(s.image, None)[0])).astype(np.float32)

        calc = ImagePointcloudCalculatorUnlabeledUncertainty(self.all_samples)
        self.gt_labeled_indexes = []
        for _ in range(num_images_to_label):  # request 10 images
          next_img_idx = np.argmax(calc.pc_idx_to_intersection)
          self.gt_labeled_indexes.append(next_img_idx)
          calc.mark_as_labeled(next_img_idx)

      else:
        rospy.logerr(f"Unknown labeling strategy {labeling_strategy}. Going to stop data player")
        rospy.signal_shutdown("")

    rospy.loginfo("Labeled images:" + str(self.gt_labeled_indexes))
    self.current_sample_idx = 0

    image_idx = []
    # First project all annotated images
    image_idx.extend(self.gt_labeled_indexes[::-1])
    # Project all others
    image_idx.extend([i for i in range(len(self.all_samples)) if i not in self.gt_labeled_indexes])
    image_idx.append(len(self.all_samples))
    self.sample_iter = iter(image_idx)

  def dump_all(self, folder: str, dump_images: bool = True, score_name: str = "scores.csv"):
    """ Saves current miou score and images
    args:
      folder: output folder where score file should be written
      dump_images: flag if images should also be extracted (stored as train + validation set with current pseudo labels)
      score_name: Name of score file that should be stored  containing miou score and additional infos
    """
    out = os.path.join(self.output_folder, folder)
    if not os.path.exists(out):
      os.mkdir(out)

    rospy.loginfo(f"Scoring dataset before dumping. Dataset: {folder}")
    mIoU, classIoU = get_miou([m.mask for m in self.all_samples],
                              [m.gt_mask for m in self.all_samples])
    data = {
      'mIoU': mIoU,
    }

    for idx, value in enumerate(classIoU):
      data[f"class_{idx}"] = value
    pd.DataFrame.from_dict([data]).to_csv(os.path.join(out, score_name), index=False)
    pd.DataFrame(data=np.asarray(self.gt_labeled_indexes)).to_csv(os.path.join(out, f"selected_imgs.csv"), index=False)
    if dump_images:
      dump_images_to_folder(self.all_samples, self.gt_labeled_indexes, out_folder=out)

  def experiment_end_cb(self, score_name="scores.csv", create_ds=True):
    """
      End of experiment callback. Writes score of current pseudo labels to the score file and creates a
      dataset that can be used to train a network
    """
    PseudoLabler(topic='/volumetric_labler_node/panoptic/render_camera_view').label_many(self.all_samples,
                                                                                         cache_time=0)
    self.dump_all(self.output_name, dump_images=create_ds, score_name=score_name)
    rospy.signal_shutdown("End of Experiment reached")

  def publish_images(self, timer_event):
    if self.current_sample_idx == 10:  # len(self.all_samples):
      self.current_sample_idx = -1
      rospy.loginfo("End of experiment reached. Replayed all images")
      self.experiment_end_cb(create_ds=True)

    if self.current_sample_idx == -1:
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
        semseg_msg = self.cv_bridge.cv2_to_imgmsg(median_filter(current_train_sample.gt_mask.astype(np.uint8), size=10),
                                                  "mono8")
        # Use -1 as uncertainty score to symbolize that this is groundtruth
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
