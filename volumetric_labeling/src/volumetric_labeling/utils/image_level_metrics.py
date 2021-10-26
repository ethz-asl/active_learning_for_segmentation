import time
from typing import List
import open3d as o3d
import numpy as np
from cv_bridge import CvBridge
import cv2
from scipy.spatial.transform import Rotation as R

from embodied_active_learning_core.online_learning.sample import TrainSample


def calc_pc_for_images(files: List[TrainSample], return_all_values=False):
  """ Unprojects all given images into 3d pointclouds
      Contains some ugly rotations to convert images in the right frames, which is kind of ugly but works :)
  """
  bridge = CvBridge()
  max_x = -np.inf
  max_y = -np.inf
  max_z = -np.inf
  min_x = np.inf
  min_y = np.inf
  min_z = np.inf

  pointclouds = []
  poses = []
  yaws = []
  all_values = []

  first_pose = None

  for idx_img in range(len(files)):
    try:
      pose = files[idx_img].pose
      img = cv2.resize(files[idx_img].image, (640, 480), interpolation=cv2.INTER_LINEAR)
      depth = bridge.imgmsg_to_cv2(files[idx_img].depth).copy()
      # Remove points that are too far away (e.g. trees outside windows)
      depth[depth >= 3.5] = 0

      translation = np.asarray([pose[0][0], pose[0][1], pose[0][2]]) / 1000
      if idx_img == 0:
        first_pose = translation

      poses.append(translation)

      # Rotation magic begins
      R_world_cam = R.from_quat(pose[1])
      R_cam_cam_frame = R.from_quat([0.5, 0.5, 0.5, -0.5])
      R_world_cam_frame = R_world_cam * R_cam_cam_frame
      yaw = R_world_cam_frame.as_euler('zyx', degrees=True)[0]
      yaws.append(yaw)
      rot_mat = np.eye(4)
      rot_mat[:-1, :-1] = (
          R.from_euler('y', -yaw, degrees=True) * R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])).as_matrix()

      # Rotation magic ends

      rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(img),
                                                                      o3d.geometry.Image(depth.astype(np.float32)))
      pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsic(640, 480, 320, 240, 320, 240)), extrinsic=rot_mat)

      # Align everything with respect to first image pose
      xyz_load = np.asarray(pcd.points)
      xyz_load[:, -1] += -(translation[0] - first_pose[0])
      xyz_load[:, 0] += (translation[1] - first_pose[1])

      m_x, m_y, m_z = np.max(xyz_load, axis=0)
      max_x = max(max_x, m_x)
      max_y = max(max_y, m_y)
      max_z = max(max_z, m_z)

      m_x, m_y, m_z = np.min(xyz_load, axis=0)
      min_x = min(min_x, m_x)
      min_y = min(min_y, m_y)
      min_z = min(min_z, m_z)
      pointclouds.append(xyz_load)
      all_values.append(np.asarray(pcd.colors)[:, 0])

    except ValueError as e:
      print("Error processing img:", idx_img)
      pointclouds.append(np.zeros((1, 3)))
      all_values.append(np.zeros((1, 1)))

  # Make sure x,y,z are always positive
  for p in pointclouds:
    p -= np.asarray([min_x, min_y, min_z])
  for p in poses:
    p -= np.asarray([min_x, min_z, min_y])

  max_x -= min_x
  max_y -= min_y
  max_z -= min_z
  print(f"processed: {len(files)} samples to create pointclouds of trajectory")
  if return_all_values:
    return pointclouds, all_values

  return pointclouds


class ImagePointcloudCalculatorMostSeen:
  """
    Returns images based on MostSeen metric.
    MostSeen metric means, that images that share voxels with the most other images will be prioritized.

    e.g.: Image #1 Contains Voxels from which some are seen from #2 and #3
          Image #2 Contains voxels from which some are seen from #1
          Image #3 Contains voxels from which some are seen from #1

          --> Image #1 Will be requested
  """

  def __init__(self, all_samples, resolution=0.05):
    now = time.time()
    self.all_pcs = calc_pc_for_images(all_samples)
    print(f"Processing PCs took {time.time() - now}s")
    # contains the combination of all points from the pointclouds
    all_pcs_reduced = []
    # contains the combination of all points from the pointcloud but stored as unique number e.g. (pos_x*250 + pos_y*250**2 + pos_z * 250**3)
    self.all_pcs_as_index = []

    now = time.time()
    for pc in self.all_pcs:
      # Voxelize pointcloud
      all_pcs_reduced.append(np.unique((pc * 1000) // resolution, axis=0))

    # Reduced combination of all points from pointclouds downsampled to voxel grid
    accounting = np.unique(np.vstack(all_pcs_reduced), axis=0)
    max_elem = np.max(accounting)
    self.accounting_as_idx = accounting[:, 0] * max_elem + accounting[:, 1] * max_elem ** 2 + accounting[:,
                                                                                              2] * max_elem ** 3

    for pc in all_pcs_reduced:
      self.all_pcs_as_index.append(pc[:, 0] * max_elem + pc[:, 1] * max_elem ** 2 + pc[:, 2] * max_elem ** 3)
    print(f"Done creating index. Took {time.time() - now}s")
    self.intersection_score = 0 * self.accounting_as_idx
    now = time.time()
    self.calc_intersection()
    print(f"Done calculating intersection. Took {time.time() - now}s")

  def mark_as_labeled(self, pc_idx):
    """ Marks a given image as labeled """
    # Find pointcloud which should be removed since it is labeled
    pc = self.all_pcs_as_index[pc_idx]
    # Find unique indices for unprojected voxels
    _, _, all_pts_idx = np.intersect1d(pc, self.accounting_as_idx, return_indices=True)
    # delete all voxels since they are now labeled
    self.accounting_as_idx = np.delete(self.accounting_as_idx, all_pts_idx)
    self.intersection_score = np.delete(self.intersection_score, all_pts_idx)
    # Fill up with zeros so indices are not messed up (img #4 is still at position 3)
    self.all_pcs_as_index[pc_idx] = np.zeros((1, 3))
    # Newly calculate intersection for all other images
    self.calc_intersection()

  def calc_intersection(self):
    """ Calculates overlap between unprojected voxels for each image """
    self.pc_idx_to_intersection = []
    # For each voxel store how many images have seen this voxel
    self.intersection_score = 0 * self.intersection_score

    for idx, pc in enumerate(self.all_pcs_as_index):
      # For each pointcloud
      _, _, all_pts_idx = np.intersect1d(pc, self.accounting_as_idx, return_indices=True)
      # Add +1 for each voxel of this pointcloud
      self.intersection_score[all_pts_idx] += 1

    for idx, pc in enumerate(self.all_pcs_as_index):
      # Now for each image sum over intersection_score of each voxel
      intersection = np.intersect1d(pc, self.accounting_as_idx)
      self.pc_idx_to_intersection.append(np.sum(intersection))


class ImagePointcloudCalculatorMostUnlabeled:
  """
    Returns images based on MostUnlabeled metric.
    MostUnlabeled metric means, that images with the most unlabeled voxels are prioritized
  """

  def __init__(self, all_samples, resolution=0.05):
    now = time.time()
    self.all_pcs = calc_pc_for_images(all_samples)
    print(f"Processing PCs took {time.time() - now}s")
    # contains the combination of all points from the pointclouds
    all_pcs_reduced = []
    # contains the combination of all points from the pointcloud but stored as unique number e.g. (pos_x*250 + pos_y*250**2 + pos_z * 250**3)
    self.all_pcs_as_index = []

    now = time.time()
    for pc in self.all_pcs:
      # Voxelize pointcloud
      all_pcs_reduced.append(np.unique((pc * 1000) // resolution, axis=0))

    # Reduced combination of all points from pointclouds downsampled to voxel grid
    accounting = np.unique(np.vstack(all_pcs_reduced), axis=0)
    max_elem = np.max(accounting)
    self.accounting_as_idx = accounting[:, 0] * max_elem + accounting[:, 1] * max_elem ** 2 + accounting[:,
                                                                                              2] * max_elem ** 3

    for pc in all_pcs_reduced:
      self.all_pcs_as_index.append(pc[:, 0] * max_elem + pc[:, 1] * max_elem ** 2 + pc[:, 2] * max_elem ** 3)

    print(f"Done creating index. Took {time.time() - now}s")
    self.intersection_score = 0 * self.accounting_as_idx
    now = time.time()
    self.calc_intersection()
    print(f"Done calculating intersection. Took {time.time() - now}s")

  def mark_as_labeled(self, pc_idx):
    """ Marks a given image as labeled """
    # Find pointcloud which should be removed since it is labeled
    pc = self.all_pcs_as_index[pc_idx]
    # Find unique indices for unprojected voxels
    _, _, all_pts_idx = np.intersect1d(pc, self.accounting_as_idx, return_indices=True)
    # delete all voxels since they are now labeled
    self.accounting_as_idx = np.delete(self.accounting_as_idx, all_pts_idx)
    # Fill up with zeros so indices are not messed up (img #4 is still at position 3)
    self.all_pcs_as_index[pc_idx] = np.zeros((1, 3))
    # Newly calculate intersection for all other images
    self.calc_intersection()

  def calc_intersection(self):
    """ Calculate image intersections """
    self.pc_idx_to_intersection = []
    for pc in self.all_pcs_as_index:
      # Calculate how many unlabeled voxels belong to this image
      unlabeled_voxels_in_frustrum = np.intersect1d(pc, self.accounting_as_idx).shape[0]
      self.pc_idx_to_intersection.append(unlabeled_voxels_in_frustrum)


class ImagePointcloudCalculatorUnlabeledUncertainty:
  """
    Returns images based on UnlabeledUncertainty metric.
    UnlabeledUncertainty metric means, that images with the highest uncertainty score when ignoring annotated voxels are annotated
  """

  def __init__(self, all_samples, resolution=0.05):
    now = time.time()
    self.all_pcs, self.all_values = calc_pc_for_images(all_samples, return_all_values=True)
    print(f"Processing PCs took {time.time() - now}s")
    # contains the combination of all points from the pointclouds
    all_pcs_reduced = []
    # contains the combination of all points from the pointcloud but stored as unique number e.g. (pos_x*250 + pos_y*250**2 + pos_z * 250**3)
    self.all_pcs_as_index = []

    now = time.time()
    for pc in self.all_pcs:
      all_pcs_reduced.append(np.unique((pc * 1000) // resolution, axis=0))

    # Reduced combination of all points from pointclouds downsampled to voxel grid
    accounting = np.unique(np.vstack(all_pcs_reduced), axis=0)
    max_elem = np.max(accounting)
    self.accounting_as_idx = accounting[:, 0] * max_elem + accounting[:, 1] * max_elem ** 2 + accounting[:,
                                                                                              2] * max_elem ** 3

    for pc in all_pcs_reduced:
      self.all_pcs_as_index.append(pc[:, 0] * max_elem + pc[:, 1] * max_elem ** 2 + pc[:, 2] * max_elem ** 3)
    print(f"Done creating index. Took {time.time() - now}s")
    self.intersection_score = 0 * self.accounting_as_idx
    now = time.time()
    self.calc_intersection()
    print(f"Done calculating intersection. Took {time.time() - now}s")

  def mark_as_labeled(self, pc_idx):
    """ Marks a given image as labeled """
    # Find pointcloud which should be removed since it is labeled
    pc = self.all_pcs_as_index[pc_idx]
    # Find unique indices for unprojected voxels
    _, _, all_pts_idx = np.intersect1d(pc, self.accounting_as_idx, return_indices=True)
    # delete all voxels since they are now labeled
    self.accounting_as_idx = np.delete(self.accounting_as_idx, all_pts_idx)
    # Fill up with zeros so indices are not messed up (img #4 is still at position 3)
    self.all_pcs_as_index[pc_idx] = np.zeros((1, 3))
    # Newly calculate intersection for all other images
    self.calc_intersection()

  def calc_intersection(self):
    """ Calculates uncertainty score for each pointcloud which is the sum of uncertainty values of unlabeled voxels """
    self.pc_idx_to_intersection = []
    self.intersection_score = 0 * self.intersection_score

    for idx, pc in enumerate(self.all_pcs_as_index):
      _, pc_pts_idx, all_pts_idx = np.intersect1d(pc, self.accounting_as_idx, return_indices=True)
      # Sum up uncertainty value for each unlabeled voxel of this image
      masked_uncertainty = self.all_values[idx][pc_pts_idx]
      self.pc_idx_to_intersection.append(np.sum(masked_uncertainty))
