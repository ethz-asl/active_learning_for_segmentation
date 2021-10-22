#!/usr/bin/env python
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
from std_msgs.msg import Bool
from airsim_ros_pkgs.srv import SetLocalPosition
from airsim_ros_pkgs.srv import Takeoff
from panoptic_mapping_msgs.srv import SaveLoadMap
from embodied_active_learning.data_acquisitors import constant_rate_data_acquisitor, goalpoints_data_acquisitor
from embodied_active_learning.utils import airsim_semantics, config
import airsim
import os


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
    configs = config.Configs(rospy.get_param("/experiment_name", "unknown_experiment"))
    configs.print_config()
    self.configs = configs
    # Parse parameters
    self.ns_planner = rospy.get_param('~ns_planner',
                                      "/panoptic_mapper/mapper/toggle_running")
    self.ns_airsim = rospy.get_param('~ns_airsim', "/airsim/airsim_node")
    self.vehicle_name = rospy.get_param('~vehicle', "drone_1")
    self.planner_delay = rospy.get_param(
      '~delay', 0.0)  # Waiting time until the mapper is launched
    self.startup_timeout = rospy.get_param(
      '~startup_timeout', 0.0)  # Max allowed time for startup, 0 for inf

    self.data_aquisition_mode = rospy.get_param("~data_mode/mode",
                                                "constant")
    self.last_count = time.time()
    self.air_sim_semantics_converter = airsim_semantics.AirSimSemanticsConverter(configs.experiment_config.semantic_mapping_path)

    self._takeoff_proxy = rospy.ServiceProxy(self.ns_airsim + "/" +
                                             self.vehicle_name + "/takeoff",
                                             Takeoff)  # Service to take off
    self._move_to_position_proxy = rospy.ServiceProxy(
      self.ns_airsim + '/local_position_goal/override',
      SetLocalPosition)  # service to navigate to given position

    self.run_planner_srv = None

    x,y,z,yaw = configs.experiment_config.start_position_x,configs.experiment_config.start_position_y\
                ,configs.experiment_config.start_position_z,configs.experiment_config.start_position_yaw


    self.max_train_duration = configs.experiment_config.max_images # number of images
    self.initial_pose = [x, y, z, yaw]

    self.train_count_sub = rospy.Subscriber("/train_count", Int16, self.train_count_callback)
    self.trajectory_fallowing_proxy = rospy.ServiceProxy("/airsim/trajectory_caller_node/set_running", SetBool)


    self._collision_sub = rospy.Subscriber("/airsim/drone_1/collision", Bool, self.collision_callback)

    self.event_path = os.path.join(self.configs.log_config.get_log_folder(), 'events.csv')
    print("LOGGING TO", self.event_path)

    with open(self.event_path, "w") as f:
      f.write("running,timestamp,description\n")
      f.write(f"{True},{rospy.Time.now().to_sec()},started experiment....\n")

    if self.launch_simulation():
      print("running")
      try:
        # Start data qauisitors after simulation launched
        self.data_aquisitors = [get_data_acquisitior(c, self.air_sim_semantics_converter) for c in
                                configs.acq_config.configs]
      except ValueError as e:
        rospy.logerr("Could not create data acquisitor.\n {}".format(
          str(e)))
      #
      # print("Stopping acc.")
      # for acq in self.data_aquisitors:
      #   acq.running = False

  def collision_callback(self, evnet):
    if (rospy.Time.now().to_sec() - self.last_collision) > 10:
      with open(self.event_path, "a") as f:
        f.write(f"{True},{rospy.Time.now().to_sec()},Collision\n")

      print("COLLISION IN EXPERIMENT. Going to restart active mapper")
      self.last_collision = rospy.Time.now().to_sec()
      try:
        pass
        # client = airsim.MultirotorClient()
        # client.confirmConnection()
        # client.enableApiControl(True)
        # currPose = client.simGetVehiclePose()
        # currPose.position.x_val = 0
        # currPose.position.y_val = 0
        # currPose.position.z_val = 0
        # currPose.orientation.x_val = 0
        # currPose.orientation.y_val = 0
        # currPose.orientation.z_val = 0
        # currPose.orientation.w_val = 1
        # client.simSetVehiclePose(currPose, True)
        # self.run_planner_srv(True)
      except:
        print("Could not restart mapper")

  def launch_simulation(self):
    rospy.loginfo("Experiment setup: waiting for airsim to launch")
    # Wait for airsim simulation to setup
    try:
      rospy.wait_for_service(
        self.ns_airsim + "/{}/takeoff".format(self.vehicle_name),
        self.startup_timeout)

    except rospy.ROSException:
      rospy.logerr(
        "Simulation startup failed (timeout after {} s).".format(
          self.startup_timeout))
      return False

    rospy.loginfo("Setting semantic classes to NYU mode")
    self.air_sim_semantics_converter.set_airsim_classes()

    self.last_collision = rospy.Time.now().to_sec() # Reset collision timer
    rospy.loginfo("Taking off")
    for cnt in range(3):
      try:
        self._takeoff_proxy(True)
      except:
        rospy.logwarn("Could not take off. Error connecting to takeoff proxy. Retry count: {}/{}".format(cnt + 1, 3))
        rospy.sleep(1)
        if cnt == 2:
          rospy.logerr("Take off service failed 3 times. Going to stop simulation!")
          exit()
        continue
      break
    # Jump to origin
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    currPose = client.simGetVehiclePose()
    currPose.position.x_val = 0
    currPose.position.y_val = 0
    currPose.position.z_val = 0
    currPose.orientation.x_val = 0
    currPose.orientation.y_val = 0
    currPose.orientation.z_val = 0
    currPose.orientation.w_val = 1
    client.simSetVehiclePose(currPose, True)
    rospy.sleep(1)

    rospy.loginfo("Moving to initial position...")
    rospy.wait_for_service(
      '/airsim/airsim_node/local_position_goal/override',
      self.startup_timeout)
    self._move_to_position_proxy(self.initial_pose[0], self.initial_pose[1],
                                 -self.initial_pose[2],
                                 self.initial_pose[3], self.vehicle_name)


    # Launch mapper (by service, every mapper needs to advertise this service when ready)


    rospy.loginfo("Waiting for mapper to be ready...")
    if self.startup_timeout > 0.0:
      try:
        print(self.ns_planner + "/toggle_running")
        rospy.wait_for_service(self.ns_planner + "/toggle_running",
                               self.startup_timeout)
      except rospy.ROSException:
        rospy.logerr(
          "Planner startup failed (timeout after {}s).".format(
            self.startup_timeout))
        return False
    else:
      rospy.wait_for_service(self.ns_planner + "/toggle_running")


    def start_planner():
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

    if rospy.get_param("/resume", False):
      map_path = rospy.get_param("/map_to_load")
      print("LOADING MAP NOW. Path: ", map_path)
      rospy.wait_for_service("/planner/planner_node/load_map", 25)
      load_map_srv = rospy.ServiceProxy("/planner/planner_node/load_map", SaveLoadMap)
      load_map_srv(map_path)
      # def cb(arg):
      #   if arg: # Start planner
      #     start_planner()
      # self.start_stop_planner = rospy.Service("/start_planner", SetBool, cb)

    # else:
    start_planner()

    if rospy.get_param("/resume", False):
      self.trajectory_fallowing_proxy(False) # = rospy.ServiceProxy("/airsim/trajectory_caller_node/set_running", SetBool)
    else:
      self.trajectory_fallowing_proxy(True)

    rospy.wait_for_service("/uncertainty/toggle_running", 5)
    # Uncertainty estimation
    try:
      self.uncertainty_srv = rospy.ServiceProxy("/uncertainty/toggle_running", SetBool)
      self.uncertainty_srv(True)
    except rospy.ROSException:
      print("Could not start uncertainty estimator")


    rospy.loginfo("\n" + "*" * 39 +
                  "\n* Succesfully started the simulation! *\n" + "*" * 39)
    self.old_pose = client.simGetVehiclePose()
    self.movement_cnt = 0
    self.start_stop_service = rospy.Service("/start_stop_experiment", SetBool, self.run_service_callback)
    # Safety timer to stop experiment when online network was not trained for a certain time (probably died)
    def timer_cb(event):
      print("Timer cb checking.....")

      # check poses
      currPose = client.simGetVehiclePose()
      if self.running and (currPose.position.x_val == self.old_pose.position.x_val
        and currPose.position.y_val == self.old_pose.position.y_val
        and currPose.position.z_val == self.old_pose.position.z_val
        and currPose.orientation.x_val == self.old_pose.orientation.x_val
        and currPose.orientation.y_val == self.old_pose.orientation.y_val
        and currPose.orientation.z_val == self.old_pose.orientation.z_val
        and currPose.orientation.w_val == self.old_pose.orientation.w_val
      ):
        print("no movement detected!")
        self.movement_cnt += 1
        if self.movement_cnt >= 2:
          print("No MOVEMENT DETECTED")
          rospy.signal_shutdown("MOVEMENT missing")
          exit()
      else:
        self.movement_cnt = 0
      self.old_pose = currPose

      if time.time() - self.last_count >= 60 * 15:
        rospy.logerr("Did not get training event in last 15 minutes. Going to shut down node!")
        rospy.signal_shutdown("Heartbeat missing")
        exit()
    self.timer = rospy.Timer(rospy.Duration(40), timer_cb)


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

      self.trajectory_fallowing_proxy(True)
      self.uncertainty_srv(True)
      self.running = True

      # rospy.loginfo("Resetting active mapper")
      # if self.run_planner_srv is not None:
      #   self.run_planner_srv(True)
      # else:
      #   rospy.logwarn("Planning service not available")

    else:
      rospy.loginfo("Going to stop all services")
      rospy.loginfo("Stopping data acquisitors")
      self.running = False
      for acq in self.data_aquisitors:
        acq.running = False
      rospy.loginfo("Stopping trajectory follower")
      self.trajectory_fallowing_proxy(False)
      self.uncertainty_srv(False)
    return True, 'running'

  def train_count_callback(self, train_count: Int16):
    """
    Callback that gets executed with the current train iteration of the online network
    """
    self.last_count = time.time()
    #
    if train_count.data >= self.max_train_duration:
      rospy.signal_shutdown("Reached train count. Shutting down")
      exit()


if __name__ == '__main__':
  rospy.init_node('airsim_experiment_node', anonymous=True)
  ed = ExperimentManager()
  rospy.spin()
