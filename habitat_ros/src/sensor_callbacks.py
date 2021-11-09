from decorators import sensorCallback, rosPublisherCreator
import rospy
import std_msgs.msg
import magnum
import quaternion
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo


def get_camera_info_msg(height = 480, width = 640):
  camera_info_msg = CameraInfo()
  fx, fy = width / 2, height / 2
  cx, cy = width / 2, height / 2
  camera_info_msg.width = width
  camera_info_msg.height = height
  camera_info_msg.K = np.float32([fx, 0, cx, 0, fy, cy, 0, 0, 1])
  camera_info_msg.D = np.float32([0, 0, 0, 0, 0])
  camera_info_msg.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]

  return camera_info_msg


@sensorCallback
def depth_cam_cb(*args):
  msg= get_camera_info_msg()
  msg.header.frame_id = "agent_cam"
  msg.header.stamp = rospy.Time.now()
  return msg

@sensorCallback
def rgb_cam_cb(*args):
  msg= get_camera_info_msg()
  msg.header.frame_id = "agent_cam"
  msg.header.stamp = rospy.Time.now()
  return msg


@sensorCallback
def RGBACallback(img):
  return getImageMessage(img[:,:,0:3], img.shape[0], img.shape[1], "rgb8")

@sensorCallback
def mono8Callback(img):
  return getImageMessage(img, img.shape[0], img.shape[1], "mono8")


@sensorCallback
def depthCallback(img):
  return getDepthImageMessage(img*1, img.shape[0], img.shape[1])

@rosPublisherCreator
def ImagePublisher(topic):
  return rospy.Publisher(
    topic, Image, queue_size=1
  )


@rosPublisherCreator
def CameraPublisher(topic):
  return rospy.Publisher(
    topic, CameraInfo, queue_size=1
  )




def getPoseMessage(position: magnum.Vector3, orientation: quaternion.quaternion, frame = "map"):
  p = TransformStamped()
  p.header.stamp = rospy.Time.now()
  p.header.frame_id = frame
  p.child_frame_id = "cam"

  # # Make sure the quaternion is valid and normalized
  # from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector
  # from habitat_sim import geo
  #
  # rotation_mp3d_habitat = quat_from_two_vectors(geo.GRAVITY, np.array([0, 0, -1]))
  #
  # # Make sure the quaternion is valid and normalized
  # p.transform.rotation.x = rotation_mp3d_habitat.x
  # p.transform.rotation.y = rotation_mp3d_habitat.y
  # p.transform.rotation.z = rotation_mp3d_habitat.z
  # p.transform.rotation.w = rotation_mp3d_habitat.w
  #
  # position = quat_rotate_vector(rotation_mp3d_habitat, position)
  p.transform.translation.x = position[0]
  p.transform.translation.y = position[1]
  p.transform.translation.z = position[2]

  p.transform.rotation.x = orientation.x
  p.transform.rotation.y = orientation.y
  p.transform.rotation.z = orientation.z
  p.transform.rotation.w = orientation.w
  return p


def getImageMessage(imageArray, height: int, width: int, encoding: str):
  rgb_msg = Image()
  rgb_msg.header.stamp = rospy.Time.now()
  rgb_msg.header.frame_id = "agent_cam"
  rgb_msg.height = height
  rgb_msg.width = width
  if len(imageArray.shape) != 2:
    imageArray = imageArray[:,:,[2,1,0]]
  rgb_msg.data = imageArray.astype(np.uint8).flatten().tolist()
  rgb_msg.encoding = "bgr8" if len(imageArray.shape) == 2 else "mono8"
  # rgb_msg.step = rgb_msg.width
  uncertainty_msg = CvBridge().cv2_to_imgmsg(imageArray.astype(np.uint8),  "bgr8" if len(imageArray.shape) == 3 else "mono8")
  uncertainty_msg.header = rgb_msg.header
  return uncertainty_msg


def getDepthImageMessage(imageArray, height: int, width: int):
  if (np.any(imageArray != 0)):
    imageArray[imageArray == 0] = 100 # fix 0 depth values

  imageArray = imageArray
  rgb_msg = Image()
  rgb_msg.header.stamp = rospy.Time.now()
  rgb_msg.header.frame_id = "agent_cam"
  rgb_msg.height = height
  rgb_msg.width = width
  rgb_msg.step = width
  rgb_msg.data = imageArray.astype(np.float32).flatten().tolist()

  uncertainty_msg = CvBridge().cv2_to_imgmsg(imageArray.astype(np.float32), "32FC1")
  uncertainty_msg.header = rgb_msg.header
  return uncertainty_msg
  # h = std_msgs.msg.Header()
  # h.stamp = rospy.Time.now()
  # h.frame_id = "agent_cam"
  # image_message = CvBridge().cv2_to_imgmsg(imageArray, encoding="passthrough")
  # image_message.header = h
  # print(imageArray.shape)
  # return image_message



