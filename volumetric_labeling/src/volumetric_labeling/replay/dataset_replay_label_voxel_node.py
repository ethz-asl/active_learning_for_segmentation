#!/usr/bin/env python
"""
  Node that replays a dataset and then annotates directly on voxel level
"""
from typing import List
import rospy
import os
import pickle
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import tf
import datetime
import pandas as pd
import time

from panoptic_mapping_msgs.srv import SaveLoadMap

from embodied_active_learning_core.online_learning.sample import TrainSample

from volumetric_labeling.labels.pseudo_labler import PseudoLabler
from volumetric_labeling.utils.utils import dump_images_to_folder, get_miou
from volumetric_labeling.srv import label_request

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
    self.gt_pub = rospy.Publisher("gt", Image, queue_size=1)
    self.uncertainty_pub = rospy.Publisher(uncertainty_topic, Image, queue_size=1)
    self.cv_bridge = CvBridge()
    self.br = tf.TransformBroadcaster()

    self.output_folder = rospy.get_param("~out_folder", None)
    self.num_instances_to_label = rospy.get_param("~instances_to_label", 40)
    self.save_map_every_nth_instance = rospy.get_param("~save_map_every_nth_instance", 5)
    self.min_instance_size = rospy.get_param("~min_instance_size", 5)  # Voxels

    if self.output_folder is None or not os.path.exists(self.output_folder):
      rospy.logerr(f"Missing output folder: {self.output_folder}")
      rospy.signal_shutdown("")
      return

    replay_path = rospy.get_param("~experiment_path", None)

    if replay_path is None or not os.path.exists(replay_path):
      rospy.logerr("Did not find experiment. Is experiment_path set correctly?")
      rospy.signal_shutdown("")
      return

    step_to_replay = rospy.get_param("~step", "step_002")
    full_path = os.path.join(replay_path, "online_training", step_to_replay)

    if not os.path.exists(full_path):
      rospy.logerr(f"Did not find experiment. Is experiment_path and step set correctly? Full Path: {full_path}")
      rospy.signal_shutdown("")
      return

    self.all_samples: List[TrainSample] = []

    for entry in sorted(os.listdir(full_path)):
      try:
        self.all_samples.append(pickle.load(open(os.path.join(full_path, entry), "rb")))
      except Exception as e:
        rospy.logwarn(
          f"Could not load trainings data stored at {os.path.join(full_path, entry)}. Corrupted file?. Error: ", e)

    rospy.loginfo(f"Done Loading dataset. Found {len(self.all_samples)} samples")
    self.scoring_method = rospy.get_param("~scoring_method")

    self.current_sample_idx = 0

    if not os.path.exists(os.path.join(self.output_folder, os.path.basename(replay_path))):
      os.mkdir(os.path.join(self.output_folder, os.path.basename(replay_path)))
    self.output_folder = os.path.join(self.output_folder, os.path.basename(replay_path))

    if not os.path.exists(os.path.join(self.output_folder, "original")):
      os.mkdir(os.path.join(self.output_folder, "original"))
      self.dump_all("original")

    self.output_name = f"dataset_{self.scoring_method}_{datetime.datetime.now().strftime('%m_%d_%Y__%H_%M_%S')}"

    image_idx = []
    image_idx.extend([*range(len(self.all_samples) + 1)])
    self.sample_iter = iter(image_idx)

    self.pseudo_labler = PseudoLabler(topic='/volumetric_labler_node/panoptic/render_camera_view')

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
      'fullmIoU': np.mean(classIoU)
    }

    for idx, value in enumerate(classIoU):
      data[f"class_{idx}"] = value
    pd.DataFrame.from_dict([data]).to_csv(os.path.join(out, score_name), index=False)
    if dump_images:
      dump_images_to_folder(self.all_samples, [], out_folder=out)

  def labeling_process(self):
    """
      Requests to annotate different instances and saves the corresponding miou score of the pseudo label
    """
    # Create folder storing all labels and information for this run
    self.output_folder = os.path.join(self.output_folder, self.output_name)
    if not os.path.exists(self.output_folder):
      os.mkdir(self.output_folder)

    srv = rospy.ServiceProxy('/label_instance', label_request)

    self.pseudo_labler.label_many(self.all_samples, cache_time=0)
    self.dump_all(f"scores_{0}.csv", dump_images=True, score_name=f"scores_{0}.csv")
    pd.DataFrame.from_dict([{
      'changed': 0,
      'GTClasses': 0,
      'InstanceClass': 0,
      'Size': 0
    }]).to_csv(
      os.path.join(self.output_folder, f"label_information_{0}.csv"), index=False)

    for i in range(1, self.num_instances_to_label):
      try:
        result = srv(self.min_instance_size, self.scoring_method)

        self.pseudo_labler.label_many(self.all_samples, cache_time=0)
        self.dump_all(f"scores_{i}.csv", dump_images=False, score_name=f"scores_{i}.csv")
        pd.DataFrame.from_dict([{
          'changed': result.Changed,
          'GTClasses': result.GTClasses,
          'InstanceClass': result.InstanceClass,
          'Size': result.Size,
          'Type': self.scoring_method
        }]).to_csv(
          os.path.join(self.output_folder, f"label_information_{i}.csv"), index=False)

        if i % self.save_map_every_nth_instance == 0:
          map_srv = rospy.ServiceProxy('/volumetric_labler_node/panoptic/save_map', SaveLoadMap)
          map_srv(os.path.join(self.output_folder, f"map_labeled_{i}.panmap"))
      except rospy.service.ServiceException:
        rospy.logerr(f"Service error. Could not annotate instance #{i}")

  def publish_images(self, timer_event):
    """ Publishes an image which will then be fused into the semantic map """
    if self.current_sample_idx == -1:
      # Make sure timer does not disturb process once labeling starts.
      return

    if self.current_sample_idx == 6:  # len(self.all_samples):  # End of experiment reached
      self.current_sample_idx = -1
      self.labeling_process()
      rospy.signal_shutdown("End of Experiment reached")
      exit()

    rospy.loginfo(
      f"Publishing sample: {self.current_sample_idx + 1}/{len(self.all_samples)} ({100 * (self.current_sample_idx + 1) / len(self.all_samples):.2f}%)")

    current_train_sample: TrainSample = self.all_samples[self.current_sample_idx]
    depth_msg = current_train_sample.depth

    try:
      ts = rospy.Time.now()
      self.br.sendTransform(current_train_sample.pose[0], current_train_sample.pose[1],
                            ts, depth_msg.header.frame_id, "world")

      rgb_msg = self.cv_bridge.cv2_to_imgmsg(current_train_sample.image, "rgb8")
      semseg_msg = self.cv_bridge.cv2_to_imgmsg(current_train_sample.mask.astype(np.uint8), "mono8")
      uncertainty_msg = self.cv_bridge.cv2_to_imgmsg(current_train_sample.weights.astype(np.float32), "32FC1")
      gt_msg = self.cv_bridge.cv2_to_imgmsg(current_train_sample.gt_mask.astype(np.uint8), 'mono8')

      depth_msg.header.stamp = ts
      rgb_msg.header = depth_msg.header
      semseg_msg.header = depth_msg.header
      uncertainty_msg.header = depth_msg.header
      gt_msg.header = depth_msg.header

      self.depth_pub.publish(depth_msg)
      self.rgb_pub.publish(rgb_msg)
      self.semseg_pub.publish(semseg_msg)
      self.uncertainty_pub.publish(uncertainty_msg)
      self.gt_pub.publish(gt_msg)


    except CvBridgeError as e:
      rospy.logerr(str(e))

    self.current_sample_idx = next(self.sample_iter)


if __name__ == '__main__':
  rospy.init_node('dataset_player_node', anonymous=True)
  player = DatasetPlayer()
  # Set up timer
  replay_rate = rospy.get_param("~replay_rate", 1)

  time.sleep(3)
  rospy.Timer(rospy.Duration(1 / replay_rate), player.publish_images)
  # Spin
  rospy.spin()
