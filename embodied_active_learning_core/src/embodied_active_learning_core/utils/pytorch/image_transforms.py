import numpy as np

IMG_SCALE = 1.0 / 255
IMG_MEAN = np.array([0.72299159, 0.67166396, 0.63768772]).reshape((1, 1, 3))
IMG_STD = np.array([0.2327359, 0.24695725, 0.25931836]).reshape((1, 1, 3))


def prepare_img(img):
  return (img * IMG_SCALE - IMG_MEAN) / IMG_STD
