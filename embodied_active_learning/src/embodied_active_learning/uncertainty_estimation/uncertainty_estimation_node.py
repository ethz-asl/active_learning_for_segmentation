#!/usr/bin/env python
"""
Node that takes an RGB input image and predicts semantic classes + uncertainties
"""

# ros
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2, PointField, Image
from message_filters import ApproximateTimeSynchronizer, Subscriber

import torch
import numpy as np

import os
from matplotlib import cm
import time
import sys

from struct import pack, unpack
# Ugly way to use refinenet TODO find a better way, install as pip package...
sys.path.append(os.getcwd() + "/../../../")
sys.path.append(os.getcwd() + "/../../../refinenet")

from refinenet.models.resnet import rf_lw50, rf_lw101, rf_lw152
from refinenet.utils.helpers import prepare_img
import cv2

from uncertainty_estimator import SimpleSoftMaxEstimator


def getUncertaintyEstimatorForParams(params):
    """
    Returns an uncertainty estimator consisting of a segmentation network + ucnertainty estimation
    :param params: Params as they are stored in rosparams
    :return: Uncertainty Estimator
    """
    model = None

    if params['network']['name'] == "lightweight-refinenet":
        size = params['network'].get('size', 101)
        classes = params['network'].get('classes', 40)
        pretrained = params['network'].get('pretrained', True)
        rospy.loginfo("Using RefineNet as Semantic Segmentation Network")
        rospy.loginfo(
            "Parameters\n- Size: {}\n- Classes: {}\n- pretrained: {}".format(
                size, classes, pretrained))

        has_cuda = False  # torch.cuda.is_available() <- Disable cuda locally to not get memory issues
        if size == 50:
            net = rf_lw50(classes, pretrained=pretrained).eval()
        elif size == 101:
            net = rf_lw101(classes, pretrained=pretrained).eval()
        elif size == 152:
            net = rf_lw152(classes, pretrained=pretrained).eval()
        else:
            rospy.logerr("Unkown encoder size {}".format(size))

        if has_cuda:
            net = net.cuda()

        def predictImage(numpy_img, net=net, has_cuda=has_cuda):
            orig_size = numpy_img.shape[:2][::-1]
            img_torch = torch.tensor(
                prepare_img(numpy_img).transpose(2, 0, 1)[None]).float()

            if has_cuda:
                img_torch = img_torch.cuda()
            pred = net(img_torch)[0].data.cpu().numpy().transpose(1, 2, 0)
            # Resize image to target prediction
            return cv2.resize(pred, orig_size, interpolation=cv2.INTER_NEAREST)

        model = predictImage

    if model is None:
        raise ValueError("Could not find model for specified parameters")

    estimator = None
    estimatorParams = params.get('method', {})

    if estimatorParams.get('type', 'softmax'):
        rospy.loginfo(
            "Creating SimpleSoftMaxEstimator for uncertainty estimation")
        estimator = SimpleSoftMaxEstimator(model,
                                           from_logits=estimatorParams.get(
                                               'from_logits', True))

    if estimator is None:
        raise ValueError("Could not find estimator for specified parameters")

    return estimator


class UncertaintyManager:
    """
    Class that publishes uncertainties + semantc segmentation to different topics

    Publishes to:
        ~/semseg/image
        ~/semseg/uncertainty
        ~/semseg/depth
        ~/semseg/points

    Subscribes to:
        rgbImage
        depthImage  <- Currently not needed but might be useful later
        cameraInfo
        odometry
    """

    def __init__(self):
        '''  Initialize ros node and read params '''
        params = rospy.get_param("/uncertainty")
        try:
            self.uncertainty_estimator = getUncertaintyEstimatorForParams(
                params)
        except ValueError as e:
            rospy.logerr(
                "Could not load uncertainty estimator. Uncertainty estimator NOT running! \n Reason: {}"
                .format(str(e)))
            rospy.logerr("Specified parameters: {}".format(str(params)))
            return

        self._semseg_pub = rospy.Publisher("~/semseg/image",
                                           Image,
                                           queue_size=5)
        self._uncertainty_pub = rospy.Publisher("~/semseg/uncertainty",
                                                Image,
                                                queue_size=5)
        self._semseg_depth_pub = rospy.Publisher("~/semseg/depth",
                                                 Image,
                                                 queue_size=5)
        self._semseg_camera_pub = rospy.Publisher("~/semseg/cam",
                                                  CameraInfo,
                                                  queue_size=5)
        self._semseg_pc_pub = rospy.Publisher("~/semseg/points",
                                              PointCloud2,
                                              queue_size=5)

        self._rgb_sub = Subscriber("rgbImage", Image)
        self._depth_sub = Subscriber("depthImage", Image)
        self._camera_sub = Subscriber("cameraInfo", CameraInfo)
        self._odom_sub = Subscriber("odometry", Odometry)

        self.last_request = rospy.get_rostime()
        self.period = params.get('rate', 1)
        self.num_classes = params['network'].get('classes', 40)

        ts = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub, self._camera_sub, self._odom_sub],
            queue_size=20,
            slop=0.2,
            allow_headerless=True)
        ts.registerCallback(self.callback)
        rospy.loginfo("Uncertainty estimator running")

    def depth_to_3d(self, img_depth, cameraInfo, distorted=False):
        """ Create point cloud from depth image and camera infos. Returns a single array for x, y and z coords """
        f, center_x, center_y = cameraInfo.K[0], cameraInfo.K[2], cameraInfo.K[
            5]
        width = cameraInfo.width
        height = cameraInfo.height
        img_depth = img_depth.reshape((height, width))
        cols, rows = np.meshgrid(np.linspace(0, width - 1, num=width),
                                 np.linspace(0, height - 1, num=height))

        # Process depth image from ray length to camera axis depth
        if distorted:
            distance = ((rows - center_y)**2 + (cols - center_x)**2)**0.5
            points_z = img_depth / (1 + (distance / f)**2)**0.5
        else:
            points_z = img_depth

        # Create x and y position
        points_x = points_z * (cols - center_x) / f
        points_y = points_z * (rows - center_y) / f

        return points_x.reshape(-1), points_y.reshape(-1), points_z.reshape(-1)

    def callback(self, rgb_msg, depth_msg, camera, publishImages=True):
        """
        Publishes semantic segmentation + Pointcloud to ropstopics

        NOTE: If publishing frequency of messages is not a multiple of the period, uncertainty images won't be published at exactly the specified frequency
        TODO Maybe change implementation to store messages with each callback and process them using rospy.Timer(), but currently not needed

        :param rgb_msg: Image message with RGB image
        :param depth_msg:  Image message with depth image (MUST BE CV_32FC1)
        :param camera: Camera info
        :param publishImages: if image messages should be published. If false only PC is published
        :return: Nothing
        """
        if (rospy.get_rostime() - self.last_request).to_sec() < self.period:
            # Too early, go back to sleep :)
            return

        self.last_request = rospy.get_rostime()
        # Monitor executing time
        startTime = time.time()
        # Get Image from message data
        img = np.frombuffer(rgb_msg.data, dtype=np.uint8)
        # Depth image
        img_depth = np.frombuffer(depth_msg.data, dtype=np.float32)
        # Convert BGR to RGB
        img = img.reshape(rgb_msg.height, rgb_msg.width, 3)[:, :, [2, 1, 0]]
        img_shape = img.shape

        semseg, uncertainty = self.uncertainty_estimator.predict(img)
        timeDiff = time.time() - startTime
        print("segmented images in {:.4f}s, {:.4f} FPs".format(
            timeDiff, 1 / timeDiff))

        (x, y, z) = self.depth_to_3d(img_depth, camera)
        color = (uncertainty*254).astype(int).reshape(-1)
        packed = pack('%di' % len(color), *color)
        unpacked = unpack('%df' % len(color), packed)
        data = (np.vstack([x, y, z,  np.array(unpacked)])).T

        pc_msg = PointCloud2()
        pc_msg.header.frame_id = rgb_msg.header.frame_id
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
        self._semseg_pc_pub.publish(pc_msg)

        if publishImages:
            # make RGB, use some nice colormaps:
            uncertainty = np.uint8(cm.seismic(uncertainty) *
                                   255)[:, :, 0:3]  # Remove alpha channel
            semseg = (cm.hsv(semseg / self.num_classes) * 255).astype(
                np.uint8)[:, :, 0:3]  # Remove alpha channel

            # Create and publish image message
            semseg_msg = Image()
            semseg_msg.header = rgb_msg.header

            semseg_msg.height = img_shape[0]
            semseg_msg.width = img_shape[1]
            semseg_msg.step = rgb_msg.width
            semseg_msg.data = semseg.flatten().tolist()
            semseg_msg.encoding = "rgb8"
            self._semseg_pub.publish(semseg_msg)

            uncertainty_msg = Image()
            uncertainty_msg.header = rgb_msg.header
            uncertainty_msg.height = img_shape[0]
            uncertainty_msg.width = img_shape[1]
            uncertainty_msg.step = rgb_msg.width
            uncertainty_msg.data = uncertainty.flatten().tolist()
            uncertainty_msg.encoding = "rgb8"
            self._uncertainty_pub.publish(uncertainty_msg)


if __name__ == '__main__':
    rospy.init_node('uncertainty_estimation_node', anonymous=True)
    um = UncertaintyManager()
    rospy.spin()
