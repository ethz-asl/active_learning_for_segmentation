from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime
import os
from embodied_active_learning.utils.airsim_semantics import AirSimSemanticsConverter
import re

# Data Acq. Type
DATA_ACQ_TYPE_CONSTANT_SAMPLER = "constantSampler"
DATA_ACQ_TYPE_GOALPOINTS_SAMPLER = "goalpointSampler"

UNCERTAINTY_TYPE_SOFTMAX = "softmax"

OPTIMIZER_TYPE_SGD = "sgd"
OPTIMIZER_TYPE_ADAM = "adam"


@dataclass
class ParsableConfig:
  def from_dict(self, data: Dict, exclude_keys=[]):
    for key in data.keys():
      if key in exclude_keys:
        continue

      if not hasattr(self, key):
        print(f"[OnlineLearningConfig] invalid key: {key}")
        continue

      setattr(self, key, data[key])

@dataclass
class PseudoLablerConfig(ParsableConfig):
  weights_method: str = "uncertainty"

  @staticmethod
  def from_ros_config(namespace=""):
    import rospy
    config = PseudoLablerConfig()
    config.from_dict(rospy.get_param(namespace + "/pseudo_labler", {}))
    return config

@dataclass
class OnlineLearningConfig(ParsableConfig):
  batch_size: int = 4
  replay_buffer_size: int = 500
  use_weights: bool = False
  old_domain_ratio: float = 0.3
  sampling_strategy: str = "uncertainty"
  replacement_strategy: str = "uncertainty"
  save_frequency: int = 20
  replay_batch_length: int = 4
  min_buffer_length: int = 10
  rate: float = 0.5
  use_epoch_based_training: bool = False
  epoch_size = 200
  num_epochs = 20
  freeze_bn: bool = True
  reset_map: bool = False
  normalize_imgs: bool = True
  test_dataset_path: str = "/media/rene/Empty/TestSets/Flat/LivingRoomWithDresser"
  replay_dataset_path: str = "/home/rene/thesis/log/offline_training/PlannerWithImpactFactorCuriosity_run108_12__23_08_2021/scannet"

  @staticmethod
  def from_ros_config(namespace=""):
    import rospy
    config = OnlineLearningConfig()
    config.from_dict(rospy.get_param(namespace + "/online_learning", {}))
    return config


def print_config_object(obj, depth=0):
  indent = depth
  for idx, e in enumerate(re.split("([^=]+\()", str(obj))):
    if idx == 0 or idx % 2 == 1:
      print(" " * indent, e.split("=")[-1].replace("(", "").strip())
    else:
      for m in re.finditer(r"(\w+)=([^=, ]+)", str(e)):
        key = m.groups()[0]
        value = m.groups()[1].replace(")", "")
        print(" " * indent, key.strip(), ":", value)
    if "(" in e:
      indent += 2
    if ")" in e:
      indent -= 2

  # re.compile(r"([^=]+\()")
  # print("->")
  # obj = re.sub(r"((, [^=(),]+)|(\([^=(),]+))=", r"\n" +" "*depth + r"\1", str(obj))
  # data = str(obj).replace("(", "\n").replace(")","\n")
  # print(data)


class Configs:
  def __init__(self, experiment_name, namespace="", load_from_ros = True):
    if load_from_ros:
      self.log_config: LogConfig = LogConfig.from_ros_config(namespace)
      self.acq_config: DataAcquistorsConfig = DataAcquistorsConfig.from_ros_config(namespace)
      self.experiment_config: ExperimentConfig = ExperimentConfig.from_ros_config(namespace)
      self.uncertainty_estimation_config: UncertaintyEstimationConfig = UncertaintyEstimationConfig.from_ros_config(
        namespace)
      self.online_learning_config: OnlineLearningConfig = OnlineLearningConfig.from_ros_config(namespace)
      self.pseudo_labler_config: PseudoLablerConfig =  PseudoLablerConfig.from_ros_config(namespace)
    else:
      self.log_config: LogConfig = LogConfig()
      self.acq_config: DataAcquistorsConfig = DataAcquistorsConfig()
      self.experiment_config: ExperimentConfig = ExperimentConfig()
      self.uncertainty_estimation_config: UncertaintyEstimationConfig = UncertaintyEstimationConfig()
      self.online_learning_config: OnlineLearningConfig = OnlineLearningConfig()
      self.pseudo_labler_config: PseudoLablerConfig =  PseudoLablerConfig()
    self.semantics_converter: AirSimSemanticsConverter = None

    # Set experiment names
    self.log_config.experiment_name = experiment_name
    for c in self.acq_config.configs:
      c.experiment_name = experiment_name

  def __repr__(self):
    return "\n".join([str(o) for o in
                      [self.log_config, self.acq_config, self.experiment_config, self.uncertainty_estimation_config,
                       self.online_learning_config]])

  def print_config(self):
    print("=" * 20)
    print_config_object(self.log_config)
    print("DataAcquistors")
    for c in self.acq_config.configs:
      print_config_object(c, depth=2)
    print_config_object(self.experiment_config)
    print_config_object(self.uncertainty_estimation_config)
    print_config_object(self.pseudo_labler_config)

    print("=" * 20)


@dataclass()
class LogConfig(ParsableConfig):
  log_path: str = "/home/rene/thesis/log/"
  experiment_name: str = "unknown_experiment"
  available_folders: List[str] = field(default_factory=list)
  log_poses: bool = False
  log_maps: bool = True
  startup_time: str = field(default_factory=lambda: datetime.now().strftime("%M_%H__%d_%m_%Y"))

  def get_or_create_folder(self, parent_folder, sub_folder):
    full_path = os.path.join(parent_folder, sub_folder)
    if not full_path in self.available_folders and not os.path.exists(full_path):
      os.mkdir(full_path)
      self.available_folders.append(full_path)

    return full_path

  def get_log_folder(self):
    return self.get_or_create_folder(self.log_path,
                                     self.experiment_name + self.startup_time)

  def get_dataset_dump_folder(self):
    return self.get_or_create_folder(self.get_log_folder(), "online_training")

  def get_map_dump_folder(self):
    return self.get_or_create_folder(self.get_log_folder(), "map")

  def get_pose_log_folder(self):
    return self.get_or_create_folder(self.get_log_folder(), "pose")

  def get_checkpoint_folder(self):
    return self.get_or_create_folder(self.get_log_folder(), "checkpoints")

  @staticmethod
  def from_ros_config(namespace=""):
    import rospy
    config = LogConfig()
    config.from_dict(rospy.get_param(namespace + "/log", {}))
    return config

  def load_ros_config(self, namespace=""):
    import rospy
    self.from_dict(rospy.get_param(namespace + "/log", {}))


@dataclass
class DataAcquistorConfig(ParsableConfig):
  type: str = "unknown_sampler"
  rate: int = 1
  output_folder: str = None
  experiment_name: str = "unknown_experiment"

  def get_log_folder(self):
    if self.log_folder is None:
      self.log_folder = os.path.join(self.output_folder,
                                     self.experiment_name + datetime.now().strftime("%H_%M_%S__%d_%m_%Y"))
      os.mkdir(self.log_folder)
    return self.log_folder


@dataclass
class ExperimentConfig(ParsableConfig):
  semantic_mapping_path: str = None
  start_position_x: float = 0.0
  start_position_y: float = 0.0
  start_position_z: float = 0.0
  start_position_yaw: float = 0.0
  max_images: int = 2000

  @staticmethod
  def from_ros_config(namespace=""):
    import rospy
    c = ExperimentConfig()
    c.from_dict(rospy.get_param(namespace + "/experiment"))
    return c


@dataclass
class DataAcquistorsConfig(ParsableConfig):
  configs: List[DataAcquistorConfig] = field(default_factory=list)

  @staticmethod
  def from_ros_config(namespace=""):
    import rospy
    config = DataAcquistorsConfig()
    for params in rospy.get_param(namespace + "/data_generation", []):
      data_acq_config = DataAcquistorConfig()
      data_acq_config.from_dict(params)
      config.configs.append(data_acq_config)
    return config


@dataclass
class OptimizerConfig(ParsableConfig):
  optim_type = OPTIMIZER_TYPE_SGD
  lr: float = 1e-4
  momentum: float = 0.9


@dataclass
class UncertaintyEstimatorConfig(ParsableConfig):
  type: str = UNCERTAINTY_TYPE_SOFTMAX
  from_logits: str = True

  # If using thresholding
  threshold: int = 0
  quantile:int  = 0.9
  max:int = 0

@dataclass
class NetworkConfig(ParsableConfig):
  name: str = "online-lightweight-refinenet"
  size: int = 50
  save_path: str = None
  checkpoint: str = None
  classes: int = 40
  pretrained: bool = True
  groupnorm: bool = True
  encoder: OptimizerConfig = field(default_factory=OptimizerConfig)
  decoder: OptimizerConfig = field(default_factory=OptimizerConfig)

  def from_dict(self, data: Dict, exclude_keys=[]):
    super().from_dict(data, exclude_keys=["encoder", "decoder"])
    self.encoder.from_dict(data.get("encoder", {}))
    self.decoder.from_dict(data.get("decoder", {}))


@dataclass
class UncertaintyEstimationConfig:
  network_config: NetworkConfig = field(default_factory=NetworkConfig)
  uncertainty_config: UncertaintyEstimatorConfig = field(default_factory=UncertaintyEstimatorConfig)
  rate: float = 1
  replay_old_pc: bool = False

  @staticmethod
  def from_ros_config(namespace=""):
    import rospy
    config: UncertaintyEstimationConfig = UncertaintyEstimationConfig()
    config.network_config.from_dict(rospy.get_param(namespace + "/uncertainty/network", {}))
    config.uncertainty_config.from_dict(rospy.get_param(namespace + "/uncertainty/method", {}))

    config.rate = rospy.get_param(namespace + "/uncertainty/rate", 1)
    config.replay_old_pc = rospy.get_param(namespace + "/uncertainty/replay_old_pc", False)

    return config

  def load_ros_config(self, namespace=""):
    import rospy
    self.from_dict(rospy.get_param(namespace + "/log", {}))
