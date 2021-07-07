import numpy as np
import embodied_active_learning.airsim_utils.semantics as semantics
import scipy
import rospy

class UncertaintyEstimator:
  def predict(self, image: np.ndarray, gt_image: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Class needs to be overwritten")


class DynamicThresholdWrapper(UncertaintyEstimator):
  """ Wraps around any Uncertanty estimator and thresholds the output.
      If requested, this threshold wrapper dynamically updates the threshold based on previous uncertainty estimates
  """

  def __init__(self, uncertanity_estimator: UncertaintyEstimator, initial_threshold=0.2, quantile=0.90, update=True,
               max_value=1):
    """
    Args:
      uncertanity_estimator:  Uncertainty estimator that should be used as basis
      initial_threshold: Initial threshold that should be used. If update is false, this will always stay the same
      quantile: In which qunatile the threshold should be set
      update: If true, automatically calculates the threshold based on a normal distribution and the provided quantile
      max_value: Maxmum value that occurs. Needed to rescale to [0,1]
    """
    self.uncertanity_estimator = uncertanity_estimator
    self.threshold = initial_threshold
    self.quantile = quantile
    self.update = update
    self.max_value = max_value

  def threshold_image(self, uncertainty: np.ndarray):
    uncertainty[uncertainty < self.threshold] = 0
    uncertainty = (uncertainty - self.threshold) / (self.max_value - self.threshold)
    uncertainty[uncertainty > 1] = 1
    uncertainty[uncertainty < 0] = 0
    return uncertainty

  def predict(self, image: np.ndarray, gt_image: np.ndarray):
    ss, uncertanty = self.uncertanity_estimator.predict(image, gt_image)
    return ss, self.threshold_image(uncertanty)

  def fit(self, all_uncertanties: list):
    """
    Retfits the threshold wrapper -> Recalculate threshold
    Args:
      all_uncertanties: List containing np.array
    """
    if self.update:
      # Calculate mean and variance of previous seen values
      mean, var = scipy.stats.distributions.norm.fit(np.asarray(all_uncertanties).ravel())
      # Calculate threshold based on normal distribution and provided quantile
      self.threshold = scipy.stats.distributions.norm.ppf(self.quantile, mean, var)
      self.uncertanity_estimator.scale_max = np.max(np.asarray(all_uncertanties).ravel())
      rospy.loginfo("Updated threshold to {}. New mean {}. Using Quantile {}".format(self.threshold, mean, self.quantile))


class ClusteredUncertaintyEstimator(UncertaintyEstimator):
  def __init__(self, model):
    self.model = model

  def predict(self, image, gt_image):
    """
    :arg image: numpy array of dimensions [height, width, batch]
    :return: Tuple:
        First: Semantic Image [height,width, batch] np.uint8
        Second: Error Image [height, width, batch] float [0,1]
    """
    prediction, uncertainty = self.model(image)
    sem_seg = np.argmax(prediction, axis=-1).astype(np.uint8)
    return sem_seg, uncertainty


class GroundTruthErrorEstimator(UncertaintyEstimator):
  def __init__(self, model, semantics_converter: semantics.AirSimSemanticsConverter):
    """
    :param model: Function that maps an input numpy array to on output numpy array
    """
    self.model = model
    self.semantics_converter = semantics_converter;

  def predict(self, image, gt_image):
    """
    :arg image: numpy array of dimensions [height, width, batch]
    :return: Tuple:
        First: Semantic Image [height,width, batch] np.uint8
        Second: Error Image [height, width, batch] float [0,1]
    """
    prediction = self.model(image)
    gt_image = self.semantics_converter.map_infrared_to_nyu(gt_image)
    sem_seg = np.argmax(prediction, axis=-1).astype(np.uint8)
    error = (sem_seg != gt_image).astype(np.float)

    return sem_seg, error


class SimpleSoftMaxEstimator(UncertaintyEstimator):

  def __init__(self, model, from_logits=False):
    """
    :param model: Function that maps an input numpy array to on output numpy array
    :param from_logits: whether the output of the model are logits or softmax predictions
    """
    self.model = model
    self.from_logits = from_logits

  def predict(self, image, gt_image):
    """
    :arg image: numpy array of dimensions [height, width, batch]
    :return: Tuple:
        First: Semantic Image [height,width, batch] np.uint8
        Second: Uncertainty Image [height, width, batch] float [0,1]
    """
    prediction = self.model(image)

    if self.from_logits:
      prediction = softmax(prediction, axis=-1)

    sem_seg = np.argmax(prediction, axis=-1).astype(np.uint8)
    # uncertainty defined as entropy
    uncertainty = -np.sum(prediction * np.log(prediction), axis=-1)
    # entropy is upper bounded by 1/log(num_classes). Use this fact to ensure uncertainty is in [0,1]
    uncertainty = uncertainty / np.log(sem_seg.shape[-1])

    return sem_seg, uncertainty


def softmax(X, theta=1.0, axis=None):
  """
  Compute the softmax of each element along an axis of X.

  Parameters
  ----------
  X: ND-Array. Probably should be floats.
  theta (optional): float parameter, used as a multiplier
      prior to exponentiation. Default = 1.0
  axis (optional): axis to compute values along. Default is the
      first non-singleton axis.

  Returns an array the same size as X. The result will sum to 1
  along the specified axis.
  """

  # make X at least 2d
  y = np.atleast_2d(X)

  # find axis
  if axis is None:
    axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

  # multiply y against the theta parameter,
  y = y * float(theta)

  # subtract the max for numerical stability
  y = y - np.expand_dims(np.max(y, axis=axis), axis)

  # exponentiate y
  y = np.exp(y)

  # take the sum along the specified axis
  ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

  # finally: divide elementwise
  p = y / ax_sum

  # flatten if X was 1D
  if len(X.shape) == 1:
    p = p.flatten()

  return p
