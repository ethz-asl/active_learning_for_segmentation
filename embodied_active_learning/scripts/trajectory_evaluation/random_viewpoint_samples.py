"""
Samples a certain amount of random viewpoints and stores the RGBD Images + GT Semantic Classes
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from PIL import Image
import airsim
import os
import time
import argparse

from embodied_active_learning.airsim_utils import semantics

parser = argparse.ArgumentParser(
    description='Creates test set with random images from environment')
parser.add_argument('--n_imgs',
                    help='Number of images that should be created',
                    default=100,
                    type=int)
parser.add_argument(
    '--min_sem_classes',
    help='Minimal amount of semantic classes in an image in order to be saved',
    default=4,
    type=int)
parser.add_argument('--out_folder',
                    help='Output folder where images should be saved',
                    default="./test_data",
                    type=str)
parser.add_argument('--map',
                    help='Path to map.pickle file',
                    default="scripts/map_generation/map.pickle",
                    type=str)

parser.add_argument('--nyu_mapping',
                    help='Path to nyu mapping .yaml',
                    default="/home/rene/catkin_ws/src/active_learning_for_segmentation/embodied_active_learning/cfg/airsim/semanticClasses.yaml",
                    type=str)
args = parser.parse_args()

outputFolder = args.out_folder
pointsToSample = args.n_imgs
minSemanticClasses = args.min_sem_classes
nyuMappingsYaml = args.nyu_mapping

airSimSemanticsConverter = semantics.AirSimSemanticsConverter(nyuMappingsYaml)
airSimSemanticsConverter.setAirsimClasses()


# Mapping from airsim type to string
typeToName = {'0': "img", '1': "depth", '5': "semantic", '7': "mask"} # 7 is semantics as infrared

# Font to write numbers on preview image
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 0, 255)
lineType = 2

map_struct = pickle.load(open(args.map, "rb"))
map_occupied = map_struct['map']  # Binary map
map_rgb = np.stack([(map_occupied == 0) * 255 for i in range(3)],
                   axis=-1).astype(np.uint8)  # Black White 3 channel image
lengthPerPixel = map_struct['dimensions'][
    'lengthPerPixel']  # Conversion from pixel to meters in unreal
top_start, left_start = map_struct['start']['top'], map_struct['start'][
    'left']  # Start position of the dron
top_lim, left_lim = map_occupied.shape

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
currPose = client.simGetVehiclePose()

cnt = 0
global previewImg  # images to draw on

while cnt < pointsToSample:
    top, left, yaw = 0, 0, 0

    while True:
        # Sample random data points that is inside map
        top, left = np.random.randint(0,
                                      top_lim), np.random.randint(0, left_lim)
        if map_occupied[top, left] != 0:
            continue
        yaw = np.random.rand() * 2 * math.pi
        break

    y = -(top_start - top) * lengthPerPixel
    x = -(left_start - left) * lengthPerPixel
    currPose.position.x_val = x
    currPose.position.y_val = y
    currPose.position.z_val = 0
    currPose.orientation.x_val = 0
    currPose.orientation.y_val = 0
    currPose.orientation.z_val = np.sin(yaw / 2)
    currPose.orientation.w_val = np.cos(yaw / 2)
    client.simSetVehiclePose(currPose, True)
    time.sleep(0.1)  # Just to be sure :)

    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene),
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, True),
        airsim.ImageRequest("0", airsim.ImageType.Infrared, False, False)
    ])

    # Check if there are enough semantic classes
    for response in responses:
        if response.image_type == 7: # infrared
            # get numpy array
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            # 3 channels with same value, just use one
            img_rgb_flatten = img1d.reshape(response.height, response.width, 3)[:,:,0]

            if len(np.unique(img_rgb_flatten)) < minSemanticClasses:
                print(
                    "Image #{} did not have enough semantic classes ({})\n Going to request another pose"
                    .format(cnt, len(np.unique(img_rgb_flatten))))
                cnt = cnt - 1
                break
            img_rgb_flatten = airSimSemanticsConverter.mapInfraredToNyu(img_rgb_flatten)

            print(img_rgb_flatten.shape, img_rgb_flatten)
            print("Classes in img:", ",".join([airSimSemanticsConverter.getNyuNameForNyuId(idx) for idx in np.unique(img_rgb_flatten)]))
            file_name =  '{}_{:04d}.png'.format(typeToName[str(response.image_type)], cnt)

            Image.fromarray(img_rgb_flatten.astype(np.uint8)).save(os.path.join(outputFolder, file_name))
            print("Saved image ({}/{})".format(cnt, pointsToSample - 1))
            # Draw poses on image
            direction = np.asarray([np.sin(yaw), np.cos(yaw)])
            length = 20
            start = (top, left)
            endpoint = start + (direction * length).astype(int)
            previewImg = cv2.arrowedLine(map_rgb, (start[1], start[0]),
                                         (endpoint[1], endpoint[0]),
                                         (255, 0, 0), 10)
            previewImg = cv2.putText(previewImg, str(cnt), (start[1], start[0]),
                                     font, fontScale, fontColor, lineType)

    # Now we know, that there are enough semantic classes, save rgb + depth too
    for response in responses:
        if response.pixels_as_float:
            airsim.write_pfm(
                os.path.join(
                    outputFolder,
                    '{}_{:04d}.pfm'.format(typeToName[str(response.image_type)],
                                           cnt)),
                airsim.get_pfm_array(response))
        elif response.image_type != 7:
            airsim.write_file(
                os.path.join(
                    outputFolder, './{}_{:04d}.png'.format(
                        typeToName[str(response.image_type)], cnt)),
                response.image_data_uint8)

    cnt += 1

Image.fromarray(previewImg).save(
    os.path.join(outputFolder, 'selected_poses.png'))
plt.imshow(previewImg)
plt.show()
