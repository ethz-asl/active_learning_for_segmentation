import rospy
import torch
import numpy as np
import cv2

#from refinenet.models.resnet import rf_lw50, rf_lw101, rf_lw152
from semseg_density.model.refinenet import rf_lw50,rf_lw101,rf_lw152

from embodied_active_learning.uncertainty_estimation.uncertainty_estimator import SimpleSoftMaxEstimator, \
  GroundTruthErrorEstimator, ClusteredUncertaintyEstimator, DynamicThresholdWrapper
# from embodied_active_learning.online_learning.online_learning import get_online_learning_refinenet
from embodied_active_learning.utils.config import Configs, UNCERTAINTY_TYPE_SOFTMAX
from embodied_active_learning.utils.pytorch_utils import prepare_img as normalized_prepare_img

def get_uncertainty_estimator_for_config(config: Configs):
  """
  Returns an uncertainty estimator consisting of a segmentation network + ucnertainty estimation
  :param params: Params as they are stored in rosparams
  :return: Uncertainty Estimator
  """
  model = None
  # uncertainty_estimation_config


  def prepare_img(img):
    if config.online_learning_config.normalize_imgs:
      return normalized_prepare_img(img)
    else:
      return img / 255

  # ------
  # Offline training
  # ------
  network_config =  config.uncertainty_estimation_config.network_config
  if network_config.name == "lightweight-refinenet":
    rospy.loginfo("Using RefineNet as Semantic Segmentation Network")
    rospy.loginfo(str(config.uncertainty_estimation_config.network_config))
    has_cuda = torch.cuda.is_available()

    if network_config.size == 50:
      net = rf_lw50(network_config.classes, pretrained=network_config.pretrained).eval()
    elif network_config.size == 101:
      net = rf_lw101(network_config.classes, pretrained=network_config.pretrainedtrained).eval()
    elif network_config.size == 152:
      net = rf_lw152(network_config.classes, pretrained=network_config.pretrained).eval()
    else:
      rospy.logerr("Unkown encoder size {}".format(network_config.size))

    if network_config.checkpoint is not None:
      net.load_state_dict(torch.load(network_config.checkpoint))

    if has_cuda:
      net = net.cuda()

    def predict_image(numpy_img, net=net, has_cuda=has_cuda):
      if type(numpy_img) == np.ndarray:
        orig_size = numpy_img.shape[:2][::-1]
        img_torch = torch.tensor(
          prepare_img(numpy_img).transpose(2, 0, 1)[None]).float()
      else:
        img_torch = numpy_img

      if has_cuda:
        img_torch = img_torch.cuda()
      pred = net(img_torch)[0].data.cpu().numpy().transpose(1, 2, 0)
      # Resize image to target prediction
      return cv2.resize(pred, orig_size, interpolation=cv2.INTER_NEAREST)

    model = predict_image

  # ------
  # Online training
  # ------
  elif network_config.name == "online-lightweight-refinenet":
    rospy.loginfo("Using ONLINE-RefineNet as Semantic Segmentation Network")
    rospy.loginfo(str(network_config))
    has_cuda = torch.cuda.is_available()

    online_net = get_online_learning_refinenet(config)
    net = online_net.model

    if network_config.checkpoint is not None:
      try:
        net.load_state_dict(torch.load(network_config.checkpoint))
      except Exception as e:
        print("Could not load model checkpoint. Going to try data parallel checkpoint")

        try:
          net = torch.nn.DataParallel(online_net.model)
          net.load_state_dict(torch.load(network_config.checkpoint))
          online_net.model = net.module
        except Exception as e:
          rospy.logerr("Could not load parallel model checkpoint. Shutting down")
          rospy.signal_shutdown("Could not load checkpoint")

      rospy.loginfo("Loaded network from checkpoint: {}".format(network_config.checkpoint))
    net = online_net
    if has_cuda:
      online_net = online_net.cuda()

    def predict_image(numpy_img, net=net, has_cuda=has_cuda):
      if type(numpy_img) == np.ndarray:
        orig_size = numpy_img.shape[:2][::-1]
        img_torch = torch.tensor(
          prepare_img(numpy_img).transpose(2, 0, 1)[None]).float()
      else:
        orig_size = numpy_img.shape[-2:][::-1]
        img_torch = numpy_img

      if has_cuda:
        img_torch = img_torch.cuda()

      pred = net(img_torch)[0].data.cpu().numpy().transpose(1, 2, 0)

      return cv2.resize(pred, orig_size, interpolation=cv2.INTER_NEAREST)

    model = predict_image

  # ------
  # GMM model
  # ------
  elif network_config.name == "online-lightweight-refinenet-with-uncertainty":
    rospy.loginfo("Using ONLINE-RefineNet with UNCERTAINTY as Semantic Segmentation Network")
    rospy.loginfo(
      "Parameters\n- Size: {}\n- Classes: {}\n- pretrained: {}".format(
        size, classes, pretrained))

    has_cuda = torch.cuda.is_available()
    net = get_online_learning_refinenet(config, with_uncertainty=True)

    if network_config.checkpoint is not None:
      net.model.load_state_dict(torch.load(network_config.checkpoint))
      rospy.loginfo("Loaded network from checkpoint: {}".format(network_config.checkpoint))
    else:
      rospy.logwarn("GMM model has no checkpoint. Clusters will be randomly initialized!")
    if has_cuda:
      net = net.cuda()

    def predict_image(numpy_img, net=net, has_cuda=has_cuda):
      orig_size = numpy_img.shape[:2][::-1]
      img_torch = torch.tensor(
        prepare_img(numpy_img).transpose(2, 0, 1)[None]).float()
      if has_cuda:
        img_torch = img_torch.cuda()

      pred, uncertainty = net(img_torch)
      pred = pred[0].cpu().numpy().transpose(1, 2, 0)
      # Resize image to target prediction
      uncertainty = uncertainty[0].cpu().numpy().transpose(1, 2, 0)

      uncertainty_resizted = np.squeeze(cv2.resize(uncertainty, orig_size, interpolation=cv2.INTER_CUBIC))
      return cv2.resize(pred, orig_size, interpolation=cv2.INTER_NEAREST), uncertainty_resizted

    model = predict_image

  if model is None:
    raise ValueError("Could not find model for specified parameters")

  ####
  # Load Uncertainty Estimator
  ####
  estimator = None
  uncertainty_config = config.uncertainty_estimation_config.uncertainty_config
  if uncertainty_config.type == UNCERTAINTY_TYPE_SOFTMAX:
    rospy.loginfo(
      "Creating SimpleSoftMaxEstimator for uncertainty estimation")
    estimator = SimpleSoftMaxEstimator(model,from_logits=uncertainty_config.from_logits)
    # TODO new config type
  elif uncertainty_config.type == "thresholder_softmax":
    estimator = DynamicThresholdWrapper(SimpleSoftMaxEstimator(model, from_logits=estimator_params.get(
      'from_logits', True)), initial_threshold=estimator_params.get('threshold', 0.8),
                                        quantile=estimator_params.get('quantile', 0.9),
                                        update=estimator_params.get('update', True))
  elif uncertainty_config.type == "gt_error":
    rospy.loginfo(
      "Creating GroundTruthError for uncertainty estimation")
    estimator = GroundTruthErrorEstimator(model, params['air_sim_semantics_converter']);
  elif uncertainty_config.type == "model_uncertainty":
    rospy.loginfo(
      "Creating Model Uncertainty for uncertainty estimation")
    estimator = DynamicThresholdWrapper(ClusteredUncertaintyEstimator(model),
                                        initial_threshold=estimator_params.get('threshold', 0.8),
                                        quantile=estimator_params.get('quantile', 0.9), max_value=70,
                                        update=estimator_params.get('update', True));
  if estimator is None:
    raise ValueError("Could not find estimator for specified parameters")

  return net, estimator