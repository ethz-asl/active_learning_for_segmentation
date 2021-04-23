import numpy as np


class SimpleSoftMaxEstimator:
    def __init__(self, model, from_logits = False):
        """
        :param model: Function that maps an input NUMPY array to on output NUMPY array
        :param from_logits: whether the output of the model are logits or softmax predictions
        """
        self.model = model
        self.from_logits = from_logits



    def predict(self, image):
        """
        :arg image: numpy array of dimensions [height, width, batch]
        :return: Tuple:
            First: Semantic Image [height,width, batch] np.uint8
            Second: Uncertainty Image [height, width, batch] float [0,1]
        """
        prediction = self.model(image)

        if self.from_logits:
            prediction = softmax(prediction, axis = -1)

        # TODO maybe improve using pytorch that returns both in one call
        semSeg = np.argmax(prediction, axis = -1).astype(np.uint8)
        uncertainty = np.max(prediction, axis = -1)

        return semSeg, uncertainty

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
    if len(X.shape) == 1: p = p.flatten()

    return p
