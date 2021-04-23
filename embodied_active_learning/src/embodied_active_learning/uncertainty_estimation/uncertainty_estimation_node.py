#!/usr/bin/env python3
"""
Node that takes an RGB input image and predicts semantic classes + uncertainties
"""

# ros
import rospy
from sensor_msgs.msg import Image

import numpy as np
import torch

from refinenet.models.resnet import rf_lw50, rf_lw101, rf_lw152
from refinenet.utils.helpers import prepare_img

import sys
import time

# Imports that get messed up by ros python 2.7 imports
python27_imports = [p for p in sys.path if "2.7" in p]
sys.path = [p for p in sys.path if "2.7" not in p]
import cv2
sys.path.extend(python27_imports)
# done removing ros paths

from uncertainty_estimator import SimpleSoftMaxEstimator

class UncertaintyManager:

    def __init__(self):
        '''  Initialize ros node and read params '''
        model_type = rospy.get_param("~model_type", "refinenet")

        if model_type == "refinenet":
            rospy.loginfo("Using RefineNet as Semantic Segmentation Network")

            has_cuda = torch.cuda.is_available()
            net = rf_lw101(40, pretrained=True).eval()
            
            if has_cuda:
                net = net.cuda()

            def predictImage(numpy_img, net = net, has_cuda = has_cuda):
                orig_size = numpy_img.shape[:2][::-1]
                img_torch = torch.tensor(prepare_img(numpy_img).transpose(2, 0, 1)[None]).float()

                if has_cuda:
                    img_torch = img_torch.cuda()
                pred =  net(img_torch)[0].data.cpu().numpy().transpose(1,2,0)
                # Resize image to target prediction
                return cv2.resize(pred, orig_size, interpolation=cv2.INTER_NEAREST)

            rospy.loginfo("Creating SimpleSoftMaxEstimator for uncertainty estimation")

            self.uncertainty_estimator = SimpleSoftMaxEstimator(predictImage, from_logits= True)
        else:
            rospy.logerr("Model Type {} is not supported!", model_type)


        self._semseg_pub = rospy.Publisher("~/semseg", Image)
        self._uncertainty_pub = rospy.Publisher("~/uncertainty", Image)
        self._rgb_sub = rospy.Subscriber("/airsim/airsim_node/drone_1/front/Scene", Image, self.rgb_callback)

        rospy.loginfo("Uncertainty Manager Running...")

    def rgb_callback(self, msg):
        """ Gets executed every time we get an image"""
        startTime = time.time()
        # Get Image from message data
        img = np.frombuffer(msg.data, dtype=np.uint8)
        # Convert BGR to RGB
        img = img.reshape(msg.height, msg.width, 3)[:, :, [2, 1, 0]]
        img_shape = img.shape

        semseg, uncertainty = self.uncertainty_estimator.predict(img)

        # convert uncertainty to uint8
        uncertainty = (uncertainty * 255).astype(np.uint8)

        # Create and publish image message
        semseg_msg = Image()
        semseg_msg.height = img_shape[0]
        semseg_msg.width = img_shape[1]
        semseg_msg.step = msg.width
        semseg_msg.data = semseg.flatten().tolist()
        semseg_msg.encoding = "mono8"
        self._semseg_pub.publish(semseg_msg)

        uncertainty_msg = Image()
        uncertainty_msg.height = img_shape[0]
        uncertainty_msg.width = img_shape[1]
        uncertainty_msg.step = msg.width
        uncertainty_msg.data = uncertainty.flatten().tolist()
        uncertainty_msg.encoding = "mono8"
        self._uncertainty_pub.publish(uncertainty_msg)

        timeDiff = time.time() - startTime
        print("published segmented image in {:.4f}s, {:.4f} FPs".format(
            timeDiff, 1 / timeDiff))


if __name__ == '__main__':
    rospy.init_node('uncertainty_estimation_node', anonymous=True)
    um = UncertaintyManager()
    rospy.spin()
