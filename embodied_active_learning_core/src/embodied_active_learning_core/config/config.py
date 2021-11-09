""" Contains dataclasses storing configs for more advanced objects in order to use typing hints in python """

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ParsableConfig:
  """ Main class for a config that can be loaded from a dict."""

  def from_dict(self, data: Dict, exclude_keys=[]):
    for key in data.keys():
      if key in exclude_keys:
        continue

      if not hasattr(self, key):
        print(f"[OnlineLearningConfig] key unknown: {key}")
        # Load key anyway in order to be more flexible when loading configs
      setattr(self, key, data[key])


UNCERTAINTY_TYPE_SOFTMAX = "softmax"
UNCERTAINTY_MODEL_BUILT_IN = "model_built_in"
UNCERTAINTY_SOFTMAX_STATIC_THRESHOLD = "softmax_static_threshold"


@dataclass
class UncertaintyEstimatorConfig(ParsableConfig):
  """ Config used for uncetainty estimation """
  type: str = UNCERTAINTY_TYPE_SOFTMAX
  from_logits: bool = True

  # If using thresholding
  threshold: int = 0
  quantile: int = 0.9
  max: int = 0

  @staticmethod
  def from_ros_config(namespace=""):
    import rospy
    c = UncertaintyEstimatorConfig()
    c.from_dict(rospy.get_param(namespace + "/method", {}))
    return c


OPTIMIZER_TYPE_SGD = "sgd"
OPTIMIZER_TYPE_ADAM = "adam"


@dataclass
class OptimizerConfig(ParsableConfig):
  optim_type = OPTIMIZER_TYPE_ADAM
  lr: float = 3e-3
  momentum: float = 0.9

NETWORK_CONFIG_LIGHTWEIGHT_REFINENET = "lightweight-refinenet"
NETWORK_CONFIG_ONLINE_LIGHTWEIGHT_REFINENET = "online-lightweight-refinenet"
NETWORK_CONFIG_ONLINE_CLUSTERED_LIGHTWEIGHT_REFINENET = "online-clustered-lightweight-refinenet"

@dataclass
class NetworkConfig(ParsableConfig):
  """ Generic Config for a semantic segmentation network"""
  name: str = "online-lightweight-refinenet"
  size: int = 50
  save_path: str = None
  checkpoint: str = None
  classes: int = 40
  pretrained: bool = True
  groupnorm: bool = True
  encoder: OptimizerConfig = field(default_factory=OptimizerConfig)
  decoder: OptimizerConfig = field(default_factory=OptimizerConfig)

  @staticmethod
  def from_ros_config(namespace=""):
    import rospy
    c = NetworkConfig()
    c.from_dict(rospy.get_param(namespace + "/network", {}))
    return c

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
    config.network_config = NetworkConfig.from_ros_config(namespace + "/uncertainty")
    config.uncertainty_config = UncertaintyEstimatorConfig.from_ros_config(namespace + "/uncertainty")
    config.rate = rospy.get_param(namespace + "/uncertainty/rate", 1)
    config.replay_old_pc = rospy.get_param(namespace + "/uncertainty/replay_old_pc", False)

    return config

  def load_ros_config(self, namespace=""):
    import rospy
    self.from_dict(rospy.get_param(namespace + "/log", {}))
