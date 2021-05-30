import numpy as np
import embodied_active_learning.airsim_utils.semantics as semantics

class GroundTruthErrorEstimator:
    def __init__(self, model, semantics_converter : semantics.AirSimSemanticsConverter):
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


class SimpleSoftMaxEstimator:

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
        uncertainty = -np.sum(prediction*np.log(prediction), axis=-1)
        # entropy is upper bounded by 1/log(num_classes). Use this fact to ensure uncertainty is in [0,1]
        uncertainty = uncertainty/np.log(sem_seg.shape[-1])

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
