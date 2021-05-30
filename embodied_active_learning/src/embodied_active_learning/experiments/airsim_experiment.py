#!/usr/bin/env python
"""
Main Class that manages an embodied active learning experiments.
First moves the drone to an initial position and then starts the planner
and starts the data acquisitor
"""

# ros
import rospy
from std_srvs.srv import SetBool
from airsim_ros_pkgs.srv import SetLocalPosition
from airsim_ros_pkgs.srv import Takeoff

from embodied_active_learning.data_acquisitors import constant_rate_data_acquisitor, goalpoints_data_acquisitor
from embodied_active_learning.airsim_utils import semantics


def getDataAcquisitior(params, semantic_converter):
    rospy.loginfo("Create Data Acquisitior for params: {}".format(str(params)))
    type = params.get("type", "constantSampler")
    if type == "constantSampler":
        return constant_rate_data_acquisitor.ConstantRateDataAcquisitor(params,
            semantic_converter)
    elif type =="goalpointSampler":
        return goalpoints_data_acquisitor.GoalpointsDataAcquisitor(params,
            semantic_converter)

    raise ValueError("Invalid Data Sampler supplied:  {}".format(type))


class ExperimentManager:

    def __init__(self):
        '''  Initialize ros node and read params '''
        # Parse parameters
        self.ns_planner = rospy.get_param('~ns_planner',
                                          "/firefly/planner_node")
        self.ns_airsim = rospy.get_param('~ns_airsim', "/airsim/airsim_node")
        self.vehicle_name = rospy.get_param('~vehicle', "drone_1")
        self.planner_delay = rospy.get_param(
            '~delay', 0.0)  # Waiting time until the planner is launched
        self.startup_timeout = rospy.get_param(
            '~startup_timeout', 0.0)  # Max allowed time for startup, 0 for inf

        self.data_aquisition_mode = rospy.get_param("~data_mode/mode",
                                                    "constant")

        x = rospy.get_param("/start_position/x", 0)
        y = rospy.get_param("/start_position/y", 0)
        z = rospy.get_param("/start_position/z", 0)
        yaw = rospy.get_param("/start_position/yaw", 0)
        self.initial_pose = [x, y, z, yaw]
        self.air_sim_semantics_converter = semantics.AirSimSemanticsConverter(
            rospy.get_param("semantic_mapping_path",
                            "../../../cfg/airsim/semanticClasses.yaml")) #semanticClassesCustomFlat

        self._takeoff_proxy = rospy.ServiceProxy(self.ns_airsim + "/" +
                                                 self.vehicle_name + "/takeoff",
                                                 Takeoff)  # Service to take off
        self._move_to_position_proxy = rospy.ServiceProxy(
            self.ns_airsim + '/local_position_goal/override',
            SetLocalPosition)  # service to navigate to given position

        if self.launch_simulation():
            try:
                self.data_aquisitors = [getDataAcquisitior(p, self.air_sim_semantics_converter) for p in rospy.get_param("/data_generation")]
            except ValueError as e:
                rospy.logerr("Could not create data acquisitor.\n {}".format(
                    str(e)))

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
        self._takeoff_proxy(True)
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

        run_planner_srv = rospy.ServiceProxy(
            self.ns_planner + "/toggle_running", SetBool)

        run_planner_srv(True)
        rospy.loginfo("\n" + "*" * 39 +
                      "\n* Succesfully started the simulation! *\n" + "*" * 39)
        return True


if __name__ == '__main__':
    rospy.init_node('airsim_experiment_node', anonymous=True)
    ed = ExperimentManager()
    rospy.spin()
