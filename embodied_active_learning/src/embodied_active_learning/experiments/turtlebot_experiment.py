#!/usr/bin/env python3
"""
Main Class that manages an embodied active learning experiments.
First moves the drone to an initial position and then starts the mapper
and starts the data acquisitor
"""

# ros
import time

import rospy
from std_srvs.srv import SetBool
from std_msgs.msg import Int16
from panoptic_mapping_msgs.srv import SaveLoadMap

from embodied_active_learning.data_acquisitors import constant_rate_data_acquisitor, goalpoints_data_acquisitor
from embodied_active_learning.utils import config

import os
import numpy as np
import yaml


def get_data_acquisitior(data_aq_config: config.DataAcquistorConfig, semantic_converter):
  rospy.loginfo("Create Data Acquisitior for params: {}".format(str(data_aq_config)))

  if data_aq_config.type == config.DATA_ACQ_TYPE_CONSTANT_SAMPLER:
    return constant_rate_data_acquisitor.ConstantRateDataAcquisitor(data_aq_config,
                                                                    semantic_converter)
  elif data_aq_config.type == config.DATA_ACQ_TYPE_GOALPOINTS_SAMPLER:
    return goalpoints_data_acquisitor.GoalpointsDataAcquisitor(data_aq_config,
                                                               semantic_converter)

  raise ValueError("Invalid Data Sampler supplied:  {}".format(data_aq_config.type))


class ExperimentManager:

  def __init__(self):
    '''  Initialize ros node and read params '''
    self.last_collision = 0
    self.running = True
    self.data_aquisitors = []
    # Get Configs
    self.configs = config.Configs(rospy.get_param("/experiment_name", "unknown_experiment"))
    self.configs.print_config()

    # Log Configc
    self.configs.log_config.log_path = "/home/zrene/out"
    log_folder = self.configs.log_config.get_log_folder()
    with open(os.path.join(log_folder, "config.txt"), "w") as f:
      f.write(str(self.configs) + "\n")
    with open(os.path.join(log_folder, "params.yaml"), "w") as f:
      # Fix since reospy.get_params("/") does not work
      all_params = np.unique(np.asarray([name.split("/")[1] for name in rospy.get_param_names()]))
      p_dict = {}
      for p in all_params:
        p_dict[p] = rospy.get_param("/" + p)
      f.write(yaml.dump(p_dict, default_flow_style=False))

    # Parse parameters
    self.ns_planner = rospy.get_param('~ns_planner',
                                      "/panoptic_mapper/mapper/toggle_running")
    self.ns_airsim = rospy.get_param('~ns_airsim', "/airsim/airsim_node")
    self.vehicle_name = rospy.get_param('~vehicle', "drone_1")
    self.planner_delay = rospy.get_param('~delay', 0.0)  # Waiting time until the mapper is launched
    self.startup_timeout = rospy.get_param('~startup_timeout', 0.0)  # Max allowed time for startup, 0 for inf

    self.data_aquisition_mode = rospy.get_param("~data_mode/mode", "constant")
    self.previous_training_event = time.time()
    self.run_planner_srv = None

    self.max_train_duration = self.configs.experiment_config.max_iterations  # number of images

    x, y, z, yaw = self.configs.experiment_config.start_position_x, self.configs.experiment_config.start_position_y \
      , self.configs.experiment_config.start_position_z, self.configs.experiment_config.start_position_yaw
    self.initial_pose = [x, y, z, yaw]

    self.train_count_sub = rospy.Subscriber("/train_count", Int16, self.train_count_callback)
    #self.trajectory_fallowing_proxy = rospy.ServiceProxy("/airsim/trajectory_caller_node/set_running", SetBool)
    self.event_path = os.path.join(self.configs.log_config.get_log_folder(), 'events.csv')

  def run(self):
    with open(self.event_path, "w") as f:
      f.write("running,timestamp,description\n")
      f.write(f"{True},{rospy.Time.now().to_sec()},started experiment....\n")

    if self.launch():
      try:
        # Start data qauisitors after simulation launched
        self.data_aquisitors = [get_data_acquisitior(c, self.air_sim_semantics_converter) for c in
                                self.configs.acq_config.configs]
        for acq in self.data_aquisitors:
          acq.running = True

      except ValueError as e:
        rospy.logerr("Could not create data acquisitor.\n {}".format(
          str(e)))

  def launch(self):
    # Log Events happening (Training, Moving, Collision)
    rospy.loginfo("Experiment setup")

    # Launch mapper (by service, every mapper needs to advertise this service when ready)
    rospy.loginfo("Waiting for mapper to be ready...")
    if self.startup_timeout > 0.0:
      try:
        rospy.wait_for_service(self.ns_planner + "/toggle_running",
                               self.startup_timeout)
      except rospy.ROSException:
        rospy.logerr(
          "Planner startup failed (timeout after {}s).".format(
            self.startup_timeout))
        return False
    else:
      rospy.wait_for_service(self.ns_planner + "/toggle_running")

    if rospy.get_param("/resume", False):
      map_path = rospy.get_param("/map_to_load")
      rospy.loginfo(f"Resuming from exsting experiment. Map Path: {map_path}")
      if not os.path.exists(map_path):
        rospy.logerr("Map path does not exist!")
        rospy.signal_shutdown("Invalid Map Path")
        exit()
        return

      rospy.wait_for_service("/planner/planner_node/load_map", 10)
      load_map_srv = rospy.ServiceProxy("/planner/planner_node/load_map", SaveLoadMap)
      load_map_srv(map_path)

    if self.planner_delay > 0:
      rospy.loginfo(
        "Waiting for mapper to be ready... done. Launch in %d seconds.",
        self.planner_delay)
      rospy.sleep(self.planner_delay)
    else:
      rospy.loginfo("Waiting for mapper to be ready... done.")

    self.run_planner_srv = rospy.ServiceProxy(
      self.ns_planner + "/toggle_running", SetBool)
    self.run_planner_srv(True)

    # If experiment is resumed, we first need to refit GMM for uncertainty.
    # Don't start moving and capturing images right away
    if rospy.get_param("/resume", False):
      #self.trajectory_fallowing_proxy(False)
      for acq in self.data_aquisitors:
        acq.running = False
    else:
      #self.trajectory_fallowing_proxy(True)
      for acq in self.data_aquisitors:
        acq.running = True

    rospy.wait_for_service("/uncertainty/toggle_running", 5)
    try:
      self.uncertainty_srv = rospy.ServiceProxy("/uncertainty/toggle_running", SetBool)
      self.uncertainty_srv(True)
    except rospy.ROSException:
      print("Could not start uncertainty estimator")

    rospy.loginfo("\n" + "*" * 39 +
                  "\n* Succesfully started the simulation! *\n" + "*" * 39)
    self.start_stop_service = rospy.Service("/start_stop_experiment", SetBool, self.run_service_callback)

    return True

  def run_service_callback(self, req):
    """ Services that stops the whole experiments and starts it again.
        Gets called while maps is replaying or networking is refitting GMM uncertainties
    """

    with open(self.event_path, "a") as f:
      f.write(f"{req.data},{rospy.Time.now().to_sec()},ServiceCallback\n")

    if req.data:
      rospy.loginfo("Going to resume all services")
      rospy.loginfo("Resuming data acquisitors")
      for acq in self.data_aquisitors:
        acq.running = True
      rospy.loginfo("Resuming trajectory follower")

      #self.trajectory_fallowing_proxy(True)
      self.uncertainty_srv(True)
      self.running = True

    else:
      rospy.loginfo("Going to stop all services")
      rospy.loginfo("Stopping data acquisitors")
      self.running = False
      for acq in self.data_aquisitors:
        acq.running = False
      rospy.loginfo("Stopping trajectory follower")
      # self.trajectory_fallowing_proxy(False)
      self.uncertainty_srv(False)
    return True, 'running'

  def train_count_callback(self, train_count: Int16):
    """
    Callback that gets executed with the current train iteration of the online network
    """
    self.previous_training_event = time.time()
    if train_count.data >= self.max_train_duration:
      rospy.signal_shutdown("Reached train count. Shutting down")
      exit()


if __name__ == '__main__':
  rospy.init_node('airsim_experiment_node', anonymous=True)
  ed = ExperimentManager()
  ed.run()
  rospy.spin()
