import torch
import re

from embodied_active_learning_core.config.config import NETWORK_CONFIG_LIGHTWEIGHT_REFINENET, \
  NETWORK_CONFIG_ONLINE_CLUSTERED_LIGHTWEIGHT_REFINENET, NETWORK_CONFIG_ONLINE_LIGHTWEIGHT_REFINENET, NetworkConfig

from embodied_active_learning_core.semseg.models.uncertainty_wrapper import UncertaintyModel
from embodied_active_learning_core.semseg.models.refinenet import rf_lw50, rf_lw101, rf_lw152

REFINENET_NAMES = [NETWORK_CONFIG_LIGHTWEIGHT_REFINENET, NETWORK_CONFIG_ONLINE_LIGHTWEIGHT_REFINENET,
                   NETWORK_CONFIG_ONLINE_CLUSTERED_LIGHTWEIGHT_REFINENET]

ONLINE_LEARNING_NETWORKS = [NETWORK_CONFIG_ONLINE_LIGHTWEIGHT_REFINENET,
                            NETWORK_CONFIG_ONLINE_CLUSTERED_LIGHTWEIGHT_REFINENET]


def get_uncertainty_net(network_config: NetworkConfig, n_components=None, n_feature_for_uncertainty=128,
                        feature_layer="mflow_conv_g3_b3_joint_varout_dimred", imagenet=False, pretrained=True,
                        covariance_type="tied",
                        reg_covar=1e-6, **kwargs):
  if network_config.name in REFINENET_NAMES:
    model = get_refinenet(network_config, load_cp=False)
    model_with_uncertainty = UncertaintyModel(model, feature_layer, n_feature_for_uncertainty=n_feature_for_uncertainty,
                                              n_components=network_config.classes if n_components is None else n_components,
                                              covariance_type=covariance_type, reg_covar=reg_covar)

    if network_config.checkpoint is not None:
      model_with_uncertainty.load_state_dict(torch.load(network_config.checkpoint))
    if torch.cuda.is_available():
      model_with_uncertainty = model_with_uncertainty.cuda()
    return model_with_uncertainty

  raise ValueError(f"Unknown model for uncertainty: {network_config.name}")


def get_refinenet(network_config: NetworkConfig, load_cp=False):
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


def get_refinenet_optimizer(network_config: NetworkConfig, network: torch.nn.Module):
  enc_params = []
  dec_params = []
  for k, v in network.named_parameters():
    if bool(re.match(".*conv1.*|.*bn1.*|.*layer.*", k)):
      enc_params.append(v)
    else:
      dec_params.append(v)

  return [
    create_optimizer(
      optim_type=network_config.encoder.optim_type,
      parameters=enc_params,
      lr=network_config.encoder.lr
    ),
    create_optimizer(
      optim_type=network_config.decoder.optim_type,
      parameters=dec_params,
      lr=network_config.decoder.lr
    )
  ]


def get_optimizer_params_for_model(network_config: NetworkConfig, model: torch.nn.Module):
  if network_config.name in REFINENET_NAMES:
    return get_refinenet_optimizer(network_config, model)

  raise ValueError(f"No optimizer provided for model: {network_config.name}")


def get_model_for_config(network_config: NetworkConfig):
  if network_config.name == NETWORK_CONFIG_LIGHTWEIGHT_REFINENET:
    net = get_refinenet(network_config)
  elif network_config.name == NETWORK_CONFIG_ONLINE_LIGHTWEIGHT_REFINENET:
    net = get_refinenet(network_config)
  elif network_config.name == NETWORK_CONFIG_ONLINE_CLUSTERED_LIGHTWEIGHT_REFINENET:
    net = get_uncertainty_net(network_config, n_feature_for_uncertainty=64, n_components=40,
                              covariance_type='full', reg_covar=0.000001, groupnorm=network_config.groupnorm)
  else:
    raise ValueError(f"No model found for name: {network_config.name}")

  return net


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
  return optim(parameters, **kwargs)
