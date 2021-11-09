from struct import pack, unpack
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo


def depth_to_3d(img_depth, camera_info, distorted=False):
  """ Create point cloud from depth image and camera infos. Returns a single array for x, y and z coords """
  f, center_x, center_y = camera_info.K[0], camera_info.K[2], camera_info.K[
    5]
  width = camera_info.width
  height = camera_info.height
  img_depth = img_depth.reshape((height, width))
  cols, rows = np.meshgrid(np.linspace(0, width - 1, num=width),
                           np.linspace(0, height - 1, num=height))

  # Process depth image from ray length to camera axis depth
  if distorted:
    distance = ((rows - center_y) ** 2 + (cols - center_x) ** 2) ** 0.5
    points_z = img_depth / (1 + (distance / f) ** 2) ** 0.5
  else:
    points_z = img_depth

  # Create x and y position
  points_x = points_z * (cols - center_x) / f
  points_y = points_z * (rows - center_y) / f

  return points_x.reshape(-1), points_y.reshape(-1), points_z.reshape(-1)


def get_pc_for_image(color: np.ndarray, img_depth: Image, camera: CameraInfo) -> Image:
  """
  returns an image as pointcloud message
  Args:
    img_to_pub: Numpy array (H,W,1) to publish
    depth_msg:  Depth message containing depth image
    camera_msg: Camera message containing camera params

    Returns: PointCloud2
  """

  (x, y, z) = depth_to_3d(img_depth, camera)
  packed = pack('%di' % len(color), *color)
  unpacked = unpack('%df' % len(color), packed)
  data = (np.vstack([x, y, z, np.array(unpacked)])).T

  pc_msg = PointCloud2()
  pc_msg.header.frame_id = camera.header.frame_id
  pc_msg.header.stamp = camera.header.stamp
  pc_msg.width = data.shape[0]
  pc_msg.height = 1
  pc_msg.fields = [
    PointField('x', 0, PointField.FLOAT32, 1),
    PointField('y', 4, PointField.FLOAT32, 1),
    PointField('z', 8, PointField.FLOAT32, 1),
    PointField('rgb', 12, PointField.FLOAT32, 1)
  ]
  pc_msg.is_bigendian = False
  pc_msg.point_step = 16
  pc_msg.row_step = pc_msg.point_step * pc_msg.width
  pc_msg.is_dense = True
  pc_msg.data = np.float32(data).tostring()
  return pc_msg
