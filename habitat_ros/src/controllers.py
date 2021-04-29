from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation

class SimplePidController:
  def __init__(self):
    self.error_x = 0
    self.error_y = 0
    self.error_z = 0
    self.yaw = 0

  def odomCallback(self, odom:Odometry):
    self.position_x = odom.pose.pose.position.x
    self.position_y = odom.pose.pose.position.y
    self.position_z = odom.pose.pose.position.z
    self.position_yaw = Rotation.from_quat([odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w]).as_euler('xyz')[-1]
