#!/usr/bin/env python
"""
Node that replays a dataset
"""
import rospy

from embodied_active_learning.uncertainty_estimation.uncertainty_estimation_node import get_uncertainty_estimator_for_config
from embodied_active_learning.utils.config import Configs

class DatasetPlayer:
  def __init__(self):
    '''  Initialize ros node and read params '''
    pass



if __name__ == '__main__':
  rospy.init_node('dataset_player_node', anonymous=True)
  um = DatasetPlayer()
  rospy.spin()

