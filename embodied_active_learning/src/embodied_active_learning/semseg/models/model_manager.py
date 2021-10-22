from embodied_active_learning.utils.config import NetworkConfig
from .refinenet import rf_lw50, rf_lw101, rf_lw152
import torch

LIGHTWEIGHT_REFINENET = "lightweight-refinenet"
ONLINE_LIGHTWEIGHT_REFINENET = "online-lightweight-refinenet"
ONLINE_CLUSTERED_LIGHTWEIGHT_REFINENET = "online-clustered-lightweight-refinenet"


from embodied_active_learning.semseg.models.uncertainty_wrapper import UncertaintyModel


def get_uncertainty_net(network_config: NetworkConfig, n_components=None, n_feature_for_uncertainty=128,
                        feature_layer="mflow_conv_g3_b3_joint_varout_dimred", imagenet=False, pretrained=True,
                        covariance_type="tied",
                        reg_covar=1e-6, **kwargs):
  model = get_refinenet(network_config, load_cp = False)
  model_with_uncertainty = UncertaintyModel(model, feature_layer, n_feature_for_uncertainty=n_feature_for_uncertainty,
                                            n_components=network_config.classes if n_components is None else n_components,
                                            covariance_type=covariance_type, reg_covar=reg_covar)

  if network_config.checkpoint is not None:
    model_with_uncertainty.load_state_dict(torch.load(network_config.checkpoint))
  if torch.cuda.is_available():
    model_with_uncertainty = model_with_uncertainty.cuda()
  return model_with_uncertainty


def get_refinenet(network_config: NetworkConfig, load_cp = False):
  """
    Returns a lightweight-refinenet model for the given configuration
  """
  networks_builders = {
    50: rf_lw50,
    101: rf_lw101,
    152: rf_lw152
  }

  if network_config.size not in networks_builders.keys():
    raise ValueError(f"Encoder Size: {network_config.size} not supported for model type {network_config.name}")

  net = networks_builders[network_config.size](network_config.classes, pretrained=network_config.pretrained,
                                               groupnorm=network_config.groupnorm).eval()

  if load_cp and network_config.checkpoint is not None:
    try:
      net.load_state_dict(torch.load(network_config.checkpoint))
    except Exception as e:
      print("Could not load model checkpoint. Going to try data parallel checkpoint")
      net_parallel = torch.nn.DataParallel(net)
      net_parallel.load_state_dict(torch.load(network_config.checkpoint))
      net = net_parallel.module

  if torch.cuda.is_available():
    net = net.cuda()
  return net


def get_model_for_config(network_config: NetworkConfig):
  if network_config.name == LIGHTWEIGHT_REFINENET:
    net = get_refinenet(network_config)
  elif network_config.name == ONLINE_LIGHTWEIGHT_REFINENET:
    net = get_refinenet(network_config)
  elif network_config.name == ONLINE_CLUSTERED_LIGHTWEIGHT_REFINENET:
    net = get_uncertainty_net(network_config, n_feature_for_uncertainty=64, n_components=40,
                            covariance_type='full', reg_covar=0.000001, groupnorm = True) # TOOD move to config
  else:
    raise ValueError(f"No model found for name -{network_config.name}-  ")
  return net
