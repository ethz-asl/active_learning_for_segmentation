#!/usr/bin/env python
"""
Main Class that manages an embodied active learning experiments.
First moves the drone to an initial position and then starts the planner
and starts the data acquisitor
"""

# ros
import time

import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool
from std_msgs.msg import Int16

from airsim_ros_pkgs.srv import SetLocalPosition
from airsim_ros_pkgs.srv import Takeoff

from embodied_active_learning.data_acquisitors import constant_rate_data_acquisitor, goalpoints_data_acquisitor
from embodied_active_learning.airsim_utils import semantics

import airsim


def get_data_acquisitior(params, semantic_converter):
  rospy.loginfo("Create Data Acquisitior for params: {}".format(str(params)))
  type = params.get("type", "constantSampler")
  if type == "constantSampler":
    return constant_rate_data_acquisitor.ConstantRateDataAcquisitor(params,
                                                                    semantic_converter)
  elif type == "goalpointSampler":
    return goalpoints_data_acquisitor.GoalpointsDataAcquisitor(params,
                                                               semantic_converter)

  raise ValueError("Invalid Data Sampler supplied:  {}".format(type))


class ExperimentManager:

  def __init__(self):
    '''  Initialize ros node and read params '''
    # Parse parameters
    self.ns_planner = rospy.get_param('~ns_planner',
                                      "/drone_1/planner")
    self.ns_airsim = rospy.get_param('~ns_airsim', "/airsim/airsim_node")
    self.vehicle_name = rospy.get_param('~vehicle', "drone_1")
    self.planner_delay = rospy.get_param(
      '~delay', 0.0)  # Waiting time until the planner is launched
    self.startup_timeout = rospy.get_param(
      '~startup_timeout', 0.0)  # Max allowed time for startup, 0 for inf

    self.data_aquisition_mode = rospy.get_param("~data_mode/mode",
                                                "constant")
    self.last_count = time.time()
    self.air_sim_semantics_converter = semantics.AirSimSemanticsConverter(
      rospy.get_param("/semantic_mapping_path",
                      "../../../cfg/airsim/semanticClasses.yaml"))

    self._takeoff_proxy = rospy.ServiceProxy(self.ns_airsim + "/" +
                                             self.vehicle_name + "/takeoff",
                                             Takeoff)  # Service to take off
    self._move_to_position_proxy = rospy.ServiceProxy(
      self.ns_airsim + '/local_position_goal/override',
      SetLocalPosition)  # service to navigate to given position

    self.run_planner_srv = None

    x = rospy.get_param("/start_position/x", 0)
    y = rospy.get_param("/start_position/y", 0)
    z = rospy.get_param("/start_position/z", 0)
    yaw = rospy.get_param("/start_position/yaw", 0)

    self.max_train_duration = rospy.get_param("train_duration", 1610)  # number of images
    self.initial_pose = [x, y, z, yaw]

    self.train_count_sub = rospy.Subscriber("/train_count", Int16, self.train_count_callback)
    self.start_stop_service = rospy.Service("/start_stop_experiment", SetBool, self.run_service_callback)
    self.trajectory_fallowing_proxy = rospy.ServiceProxy("/airsim/trajectory_caller_node/set_running", SetBool)



    if self.launch_simulation():
      try:
        # Start data qauisitors after simulation launched
        self.data_aquisitors = [get_data_acquisitior(p, self.air_sim_semantics_converter) for p in
                                rospy.get_param("/data_generation")]
      except ValueError as e:
        rospy.logerr("Could not create data acquisitor.\n {}".format(
          str(e)))

      print("Stopping acc.")
      for acq in self.data_aquisitors:
        acq.running = False

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
    rospy.sleep(10)
    # Launch planner (by service, every planner needs to advertise this service when ready)
    rospy.loginfo("Waiting for planner to be ready...")
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

    if self.planner_delay > 0:
      rospy.loginfo(
        "Waiting for planner to be ready... done. Launch in %d seconds.",
        self.planner_delay)
      rospy.sleep(self.planner_delay)
    else:
      rospy.loginfo("Waiting for planner to be ready... done.")

    self.run_planner_srv = rospy.ServiceProxy(
      self.ns_planner + "/toggle_running", SetBool)

    self.run_planner_srv(True)

    # Uncertainty estimation
    try:
      uncertainty_srv = rospy.ServiceProxy("/uncertainty/toggle_running", SetBool)
      uncertainty_srv(True)
    except rospy.ROSException:
      print("Could not start uncertainty estimator")

    rospy.loginfo("\n" + "*" * 39 +
                  "\n* Succesfully started the simulation! *\n" + "*" * 39)

    # Safety timer to stop experiment when online network was not trained for a certain time (probably died)
    def timer_cb(event):
      if time.time() - self.last_count >= 60 * 5:
        rospy.logerr("Did not get training event in last 5 minutes. Going to shut down node!")
        # rospy.signal_shutdown("Heartbeat missing")
        # exit()

    rospy.Timer(rospy.Duration(10), timer_cb)

    return True

  def run_service_callback(self, req):
    """ Services that stops the whole experiments and starts it again.
        Gets called while maps is replaying or networking is refitting GMM uncertainties
    """

    if req.data:
      rospy.loginfo("Going to resume all services")
      rospy.loginfo("Resuming data acquisitors")
      for acq in self.data_aquisitors:
        acq.running = True
      rospy.loginfo("Resuming trajectory follower")

      self.trajectory_fallowing_proxy(True)

      rospy.loginfo("Resetting active planner")
      if self.run_planner_srv is not None:
        self.run_planner_srv(True)
      else:
        rospy.logwarn("Planning service not available")

    else:
      rospy.loginfo("Going to stop all services")
      rospy.loginfo("Stopping data acquisitors")
      for acq in self.data_aquisitors:
        acq.running = False
      rospy.loginfo("Stopping trajectory follower")
      self.trajectory_fallowing_proxy(False)

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
