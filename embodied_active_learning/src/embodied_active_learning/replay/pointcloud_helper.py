import matplotlib.pyplot as plt
import math
import time
import open3d as o3d
import numpy as np
from cv_bridge import CvBridge
import cv2
from scipy.spatial.transform import Rotation as R

def calc_pc_for_images(files):
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
    first_pose = None

    for idx_img in range(len(files)):
      try:
        pose = files[idx_img].pose
        img = cv2.resize(files[idx_img].image, (640, 480), interpolation=cv2.INTER_LINEAR)
        depth = bridge.imgmsg_to_cv2(files[idx_img].depth).copy()
        # Remove points that are too far away (e.g. images)
        depth[depth >= 3.5] = 0
        rot_mat = np.eye(4)
        quat = pose[1]

        translation = np.asarray([pose[0][0], pose[0][1], pose[0][2]]) / 1000
        if idx_img == 0:
          first_pose = translation

        poses.append(translation)

        R_world_cam = R.from_quat(quat)
        R_cam_cam_frame = R.from_quat([0.5, 0.5, 0.5, -0.5])
        R_world_cam_frame = R_world_cam * R_cam_cam_frame
        yaw = R_world_cam_frame.as_euler('zyx', degrees=True)[0]
        yaws.append(yaw)
        rot_mat[:-1, :-1] = (
              R.from_euler('y', -yaw, degrees=True) * R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])).as_matrix()

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(img),
                                                                        o3d.geometry.Image(depth.astype(np.float32)))
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(
          o3d.camera.PinholeCameraIntrinsic(640, 480, 320, 240, 320, 240)), extrinsic=rot_mat)

        xyz_load = np.asarray(pcd.points)
        # move z
        xyz_load[:, -1] += -(translation[0] - first_pose[0])
        xyz_load[:, 0] += (translation[1] - first_pose[1])

        min_heigth = np.min(xyz_load, axis=0)[1]
        # mask = np.logical_and(xyz_load[:,1] > 0.0001 +min_heigth, xyz_load[:,1] < 0.0013 + min_heigth)
        xyz_no_ceil_floor = xyz_load  # [mask,:]

        m_x, m_y, m_z = np.max(xyz_no_ceil_floor, axis=0)
        max_x = max(max_x, m_x)
        max_y = max(max_y, m_y)
        max_z = max(max_z, m_z)

        m_x, m_y, m_z = np.min(xyz_no_ceil_floor, axis=0)
        min_x = min(min_x, m_x)
        min_y = min(min_y, m_y)
        min_z = min(min_z, m_z)
        pointclouds.append(xyz_no_ceil_floor)

      #         print(f"#{idx_img}: min: ({min_x*100:3f},{min_y*100:3f},{min_z*100:3f}), ({max_x*100:3f},{max_y*100:3f},{max_z})")
      except ValueError as e:
        print("Error processing img:", idx_img)
        pointclouds.append(np.zeros((1, 3)))

    for p in pointclouds:
      p -= np.asarray([min_x, min_y, min_z])
    for p in poses:
      p -= np.asarray([min_x, min_z, min_y])

    max_x -= min_x
    max_y -= min_y
    max_z -= min_z
    print(f"processed: {len(files)} samples to create pointclouds of trajectory")

    return pointclouds

class ImagePointcloudCalculatorMostSeen:

  def __init__(self, all_samples, resolution=0.05):
    now = time.time()
    self.all_pcs = calc_pc_for_images(all_samples)
    print(f"Processing PCs took {time.time() - now}s")
    print(len(self.all_pcs))
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

    print("accounting index size: ", self.accounting_as_idx.shape)

    for pc in all_pcs_reduced:
      self.all_pcs_as_index.append(pc[:, 0] * max_elem + pc[:, 1] * max_elem ** 2 + pc[:, 2] * max_elem ** 3)
    print(f"Done creating index. Took {time.time() - now}s")
    self.intersection_score = 0 * self.accounting_as_idx
    now = time.time()
    self.calc_intersection()
    print(f"Done calculating intersection. Took {time.time() - now}s")

  def mark_as_labeled(self, pc_idx):
    pc = self.all_pcs_as_index[pc_idx]
    _, _, all_pts_idx = np.intersect1d(pc, self.accounting_as_idx, return_indices=True)
    self.accounting_as_idx = np.delete(self.accounting_as_idx, all_pts_idx)
    self.intersection_score = np.delete(self.intersection_score, all_pts_idx)
    self.all_pcs_as_index[pc_idx] = np.zeros((1, 3))
    self.calc_intersection()

  def calc_intersection(self):
    self.pc_idx_to_intersection = []
    self.intersection_score = 0 * self.intersection_score

    for idx, pc in enumerate(self.all_pcs_as_index):
      _, _, all_pts_idx = np.intersect1d(pc, self.accounting_as_idx, return_indices=True)
      self.intersection_score[all_pts_idx] += 1

    for idx, pc in enumerate(self.all_pcs_as_index):
      intersection = np.intersect1d(pc, self.accounting_as_idx)
      self.pc_idx_to_intersection.append(np.sum(intersection))


class ImagePointcloudCalculator:

  def __init__(self, all_samples, resolution=0.05):
    now = time.time()
    self.all_pcs = calc_pc_for_images(all_samples)
    print(f"Processing PCs took {time.time() - now}s")
    print(len(self.all_pcs))
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

    print("accounting index size: ", self.accounting_as_idx.shape)

    for pc in all_pcs_reduced:
      self.all_pcs_as_index.append(pc[:, 0] * max_elem + pc[:, 1] * max_elem ** 2 + pc[:, 2] * max_elem ** 3)
    print(f"Done creating index. Took {time.time() - now}s")
    now = time.time()
    self.calc_intersection()
    print(f"Done calculating intersection. Took {time.time() - now}s")

  def mark_as_labeled(self, pc_idx):
    pc = self.all_pcs_as_index[pc_idx]
    _, _, all_pts_idx = np.intersect1d(pc, self.accounting_as_idx, return_indices=True)
    self.accounting_as_idx = np.delete(self.accounting_as_idx, all_pts_idx)
    self.all_pcs_as_index[pc_idx] = np.zeros((1, 3))
    self.calc_intersection()

  def calc_intersection(self):
    self.pc_idx_to_intersection = []
    for pc in self.all_pcs_as_index:
      self.pc_idx_to_intersection.append(np.intersect1d(pc, self.accounting_as_idx).shape[0])


class ImagePointcloudCalculatorUncertainty:

  @staticmethod
  def calc_pc_for_images(files):
    bridge = CvBridge()
    max_x = -np.inf
    max_y = -np.inf
    max_z = -np.inf
    min_x = np.inf
    min_y = np.inf
    min_z = np.inf

    pointclouds = []
    all_vals = []
    poses = []
    yaws = []
    first_pose = None

    for idx_img in range(len(files)):
      try:
        pose = files[idx_img].pose
        img = cv2.resize(files[idx_img].uncertainty, (640 // 6, 480 // 6), interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        depth = bridge.imgmsg_to_cv2(files[idx_img].depth).copy()
        # Remove points that are too far away (e.g. images)
        depth[depth >= 3.5] = 0
        rot_mat = np.eye(4)
        quat = pose[1]

        translation = np.asarray([pose[0][0], pose[0][1], pose[0][2]]) / 1000
        if idx_img == 0:
          first_pose = translation

        poses.append(translation)

        R_world_cam = R.from_quat(quat)
        R_cam_cam_frame = R.from_quat([0.5, 0.5, 0.5, -0.5])
        R_world_cam_frame = R_world_cam * R_cam_cam_frame
        yaw = R_world_cam_frame.as_euler('zyx', degrees=True)[0]
        yaws.append(yaw)
        rot_mat[:-1, :-1] = (
            R.from_euler('y', -yaw, degrees=True) * R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])).as_matrix()

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(img),
                                                                        o3d.geometry.Image(depth.astype(np.float32)))
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(
          o3d.camera.PinholeCameraIntrinsic(640, 480, 320, 240, 320, 240)), extrinsic=rot_mat)

        xyz_load = np.asarray(pcd.points)
        # move z
        xyz_load[:, -1] += -(translation[0] - first_pose[0])
        xyz_load[:, 0] += (translation[1] - first_pose[1])

        values = np.asarray(pcd.colors)[:, 0]
        xyz_no_ceil_floor = xyz_load

        m_x, m_y, m_z = np.max(xyz_no_ceil_floor, axis=0)
        max_x = max(max_x, m_x)
        max_y = max(max_y, m_y)
        max_z = max(max_z, m_z)

        m_x, m_y, m_z = np.min(xyz_no_ceil_floor, axis=0)
        min_x = min(min_x, m_x)
        min_y = min(min_y, m_y)
        min_z = min(min_z, m_z)
        pointclouds.append((xyz_no_ceil_floor))
        all_vals.append(values)

      #         print(f"#{idx_img}: min: ({min_x*100:3f},{min_y*100:3f},{min_z*100:3f}), ({max_x*100:3f},{max_y*100:3f},{max_z})")
      except ValueError as e:
        print("Error processing img:", idx_img)
        pointclouds.append(np.zeros((1, 3)))
        all_vals.append(np.zeros((1, 1)))

    for p in pointclouds:
      p -= np.asarray([min_x, min_y, min_z])
    for p in poses:
      p -= np.asarray([min_x, min_z, min_y])

    max_x -= min_x
    max_y -= min_y
    max_z -= min_z
    print(f"processed: {len(files)} samples to create pointclouds of trajectory")
    return pointclouds, all_vals

  def __init__(self, all_samples, resolution=0.05):
    now = time.time()
    self.all_pcs, self.all_values = self.calc_pc_for_images(all_samples)
    print(f"Processing PCs took {time.time() - now}s")
    print(len(self.all_pcs))
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

    print("accounting index size: ", self.accounting_as_idx.shape)

    for pc in all_pcs_reduced:
      self.all_pcs_as_index.append(pc[:, 0] * max_elem + pc[:, 1] * max_elem ** 2 + pc[:, 2] * max_elem ** 3)
    print(f"Done creating index. Took {time.time() - now}s")
    self.intersection_score = 0 * self.accounting_as_idx
    now = time.time()
    self.calc_intersection()
    print(f"Done calculating intersection. Took {time.time() - now}s")

  def mark_as_labeled(self, pc_idx):
    pc = self.all_pcs_as_index[pc_idx]
    _, _, all_pts_idx = np.intersect1d(pc, self.accounting_as_idx, return_indices=True)
    self.accounting_as_idx = np.delete(self.accounting_as_idx, all_pts_idx)
    self.all_pcs_as_index[pc_idx] = np.zeros((1, 3))
    self.calc_intersection()

  def calc_intersection(self):
    self.pc_idx_to_intersection = []
    self.intersection_score = 0 * self.intersection_score

    for idx, pc in enumerate(self.all_pcs_as_index):
      _, pc_pts_idx, all_pts_idx = np.intersect1d(pc, self.accounting_as_idx, return_indices=True)
      # pc_pts_idx intersection between image and unseen voxels
      masked_uncertainty = self.all_values[idx][pc_pts_idx]
      self.pc_idx_to_intersection.append(np.sum(masked_uncertainty))
