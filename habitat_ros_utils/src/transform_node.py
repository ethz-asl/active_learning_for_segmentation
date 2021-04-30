#!/usr/bin/env python2
# ROS IMPORTS
from habitat_ros_utils.srv import PublishTransform, PublishTransformResponse
import rospy
import np
import tf


br = tf.TransformBroadcaster()

def handleTransform(transform):
  br.sendTransform((transform.tx,transform.ty,transform.tz),
                   np.asarray([transform.rx, transform.ry,transform.rz,transform.rw]),
                   rospy.Time.now(),transform.child_id,  transform.parent_id)
  return PublishTransformResponse(True)
if __name__ == "__main__":
  rospy.init_node("transform_service_node", anonymous=False, log_level=rospy.DEBUG)
  rospy.Service("/habitat_ros/publishTransform", PublishTransform, handleTransform)
  rospy.spin()