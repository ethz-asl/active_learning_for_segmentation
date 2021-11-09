#!/usr/bin/env python
# ROS IMPORTS
import sys
sys.path.append("/home/rene/catkin_ws/src/active_learning_for_segmentation/habitat_ros/src")

from sensor_callbacks import *
from geometry_msgs.msg import Twist

import sys
sys.path = [
    b for b in sys.path if "2.7" not in b
]  # remove path's related to ROS from environment or else certain packages like cv2 can't be imported

import os

os.chdir('/home/rene/catkin_ws/src/habitat-lab')
from simulators import ContinuousEnvironmentSimulator
# Done Imports

rospy.init_node("habitat_ros", anonymous=False, log_level=rospy.DEBUG)

_sensor_rate = rospy # hz
_r = rospy.Rate(rospy.get_param("~sensors/frequency"))


def startVelocitySubscription(simulator: ContinuousEnvironmentSimulator):
  """ Starts a subscription to the /cmd_vel topic and uses z velocities to move robot"""
  def velocityCallback(vel):
    simulator.lock.acquire()
    simulator.control["forward_velocity"] = vel.linear.z
    simulator.control["y"] = vel.linear.y
    simulator.control["z"] = vel.linear.z
    simulator.control["rotation_velocity"] =  vel.angular.z
    # print("control:", simulator.control)
    simulator.lock.release()

  rospy.Subscriber("/cmd_vel", Twist, velocityCallback, queue_size=1)

def addSensorsCallbacksToSimulator(simulator: ContinuousEnvironmentSimulator, params = "~sensors/availableSensors"):
  """
  Adds callbacks for sensors defined in the default_params.yaml configuration to publish them as rostopic.
  Args:
    simulator: ContinuousEnvironmentSimulator
    params: Where params are stored on the parameter server
  """

  availableSensors = rospy.get_param(params)

  for sensor in availableSensors:
    if sensor['callback'] not in sensorCallback.all.keys():
      rospy.logwarn("Did not find registered callback for sensor %s: ", sensor)
      continue

    if sensor['publisher'] not in rosPublisherCreator.all.keys():
      rospy.logwarn("Did not find registered publisher %s",  sensor['publisher'])
      continue

    publisher = rosPublisherCreator.all[sensor['publisher']](sensor['topic'])

    rospy.logdebug("Created publisher for %s", sensor['topic'])

    cb = sensorCallback.all[sensor['callback']]
    simulator.addSensorCallback(sensor['name'], lambda data, cb=cb, publisher=publisher: publisher.publish(cb(data)))

    rospy.logdebug("Registered Callback %s for %s with topic %s", sensor['callback'], sensor['name'], sensor['topic'])



def main():
  with ContinuousEnvironmentSimulator() as simulator:
    addSensorsCallbacksToSimulator(simulator)
    startVelocitySubscription(simulator)

    # Publish sensor observations
    while not rospy.is_shutdown():
      simulator.publishObservations()
      _r.sleep()

if __name__ == "__main__":
  main()
