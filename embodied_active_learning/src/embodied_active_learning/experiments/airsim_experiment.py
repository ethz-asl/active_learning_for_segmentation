#!/usr/bin/env python
"""
Main Class that manages an embodied active learning experiment.
First moves the drone to an initial position and then starts the planner
"""

# ros
import rospy
from std_srvs.srv import SetBool
from airsim_ros_pkgs.srv import SetLocalPosition
from airsim_ros_pkgs.srv import Takeoff

from embodied_active_learning.data_acquisitors import ConstantRateDataAcquisitor

class ExperimentManager:

    def __init__(self):
        '''  Initialize ros node and read params '''
        # Parse parameters
        self.ns_planner = rospy.get_param('~ns_planner', "/firefly/planner_node")
        self.ns_airsim = rospy.get_param('~ns_airsim', "/airsim/airsim_node")
        self.vehicle_name = rospy.get_param('~vehicle', "drone_1")
        self.planner_delay = rospy.get_param('~delay', 0.0)  # Waiting time until the planner is launched
        self.startup_timeout = rospy.get_param('~startup_timeout', 0.0)  # Max allowed time for startup, 0 for inf

        self.data_aquisition_mode = rospy.get_param("~data_mode/mode", "constant")

        self.initial_pose = [0,0,1,0] # x y z yaw
        self._takeoff_proxy = rospy.ServiceProxy(self.ns_airsim + "/" + self.vehicle_name + "/takeoff", Takeoff) # Service to take off
        self._move_to_position_proxy = rospy.ServiceProxy(self.ns_airsim + '/local_position_goal/override', SetLocalPosition) # service to navigate to given position
        self.launch_simulation()


        if self.data_aquisition_mode == "constant":
            self.data_aquisitor = ConstantRateDataAcquisitor.ConstantRateDataAcquisitor()
        else:
            rospy.logerr("Currently only [constant] data acquisition is supported")


    def launch_simulation(self):
        rospy.loginfo("Experiment setup: waiting for airsim to launch")
        # Wait for airsim simulation to setup
        if self.startup_timeout > 0.0:
            try:
                rospy.wait_for_service(self.ns_airsim  + "/{}/takeoff".format(self.vehicle_name), self.startup_timeout)

            except rospy.ROSException:
                rospy.logerr("Simulation startup failed (timeout after {} s).".format(self.startup_timeout))
                return
        else:
            rospy.wait_for_service(self.ns_airsim  + "/drone_1/takeoff", self.startup_timeout)

        rospy.loginfo("Taking off")
        self._takeoff_proxy(True)
        rospy.loginfo("Moving to initial position...")
        rospy.wait_for_service('/airsim/airsim_node/local_position_goal/override', self.startup_timeout)
        self._move_to_position_proxy(self.initial_pose[0],self.initial_pose[1],-self.initial_pose[2],self.initial_pose[3], 'drone_1')
        rospy.sleep(10)
        # Launch planner (by service, every planner needs to advertise this service when ready)
        rospy.loginfo("Waiting for planner to be ready...")
        if self.startup_timeout > 0.0:
            try:
                rospy.wait_for_service(self.ns_planner + "/toggle_running", self.startup_timeout)
            except rospy.ROSException:
                rospy.logerr("Planner startup failed (timeout after {}s).".format(self.startup_timeout))
                return
        else:
            rospy.wait_for_service(self.ns_planner + "/toggle_running")

        if self.planner_delay > 0:
            rospy.loginfo("Waiting for planner to be ready... done. Launch in %d seconds.", self.planner_delay)
            rospy.sleep(self.planner_delay)
        else:
            rospy.loginfo("Waiting for planner to be ready... done.")
        run_planner_srv = rospy.ServiceProxy(self.ns_planner + "/toggle_running", SetBool)
        run_planner_srv(True)
        rospy.loginfo("\n" + "*" * 39 + "\n* Succesfully started the simulation! *\n" + "*" * 39)


if __name__ == '__main__':
    rospy.init_node('airsim_experiment_node', anonymous=True)
    ed = ExperimentManager()
    rospy.spin()