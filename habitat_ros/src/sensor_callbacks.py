from decorators import sensorCallback, rosPublisherCreator
import rospy
import std_msgs.msg
import magnum
import quaternion
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge

@sensorCallback
def RGBACallback(img):
  return getImageMessage(img[:,:,0:3], img.shape[0], img.shape[1], "rgb8")

@sensorCallback
def mono8Callback(img):
  return getImageMessage(img, img.shape[0], img.shape[1], "mono8")


@sensorCallback
def depthCallback(img):
  return getDepthImageMessage(img*10, img.shape[0], img.shape[1])

@rosPublisherCreator
def ImagePublisher(topic):
  return rospy.Publisher(
    topic, Image, queue_size=1
  )


def getPoseMessage(position: magnum.Vector3, orientation: quaternion.quaternion, frame = "map"):
  p = PoseStamped()
  p.header.frame_id = frame
  p.pose.position.x = position.x
  p.pose.position.x = position.y
  p.pose.position.z = position.z
  # Make sure the quaternion is valid and normalized
  p.pose.orientation.x = orientation.x
  p.pose.orientation.y = orientation.y
  p.pose.orientation.z = orientation.z
  p.pose.orientation.w = orientation.w
  return p


def getImageMessage(imageArray, height: int, width: int, encoding: str):
  rgb_msg = Image()
  rgb_msg.height = height
  rgb_msg.width = width
  rgb_msg.data = imageArray.astype(np.uint8).flatten().tolist()
  rgb_msg.encoding = encoding
  return rgb_msg


def getDepthImageMessage(imageArray, height: int, width: int):
  h = std_msgs.msg.Header()
  h.stamp = rospy.Time.now()
  image_message = CvBridge().cv2_to_imgmsg(imageArray, encoding="passthrough")
  image_message.header = h
  return image_message



