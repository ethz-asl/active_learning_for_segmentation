"""
Samples a certain amount of random viewpoints and stores the RGBD Images + GT Semantic Classes
"""
import pickle

import numpy as np
import math
import cv2
from PIL import Image
import airsim
import os
import time
import argparse
import tf

from embodied_active_learning.utils.airsim import airsim_semantics


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(
    description='Creates test set with random images from environment')
parser.add_argument('--n_imgs',
                    help='Number of images that should be created',
                    default=100,
                    type=int)
parser.add_argument(
    '--min_sem_classes',
    help='Minimal amount of semantic classes in an image in order to be saved',
    default=5,
    type=int)

parser.add_argument("--sample_z", help = "Sample z", type=str2bool, nargs='?',
                        const=True, default=False)

parser.add_argument("--sample_pitch", help = "Sample pitch", type=str2bool, nargs='?',
                        const=True, default=False)
parser.add_argument("--sample_roll", help = "Sample roll", type=str2bool, nargs='?',
                        const=True, default=False)

parser.add_argument('--out_folder',
                    help='Output folder where images should be saved',
                    default="/home/rene/thesis/test_data_new",
                    type=str)
parser.add_argument('--map',
                    help='Path to map.pickle file',
                    default="scripts/map_generation/map.pickle",
                    type=str)

parser.add_argument('--minimap',
                    help='Path to minimap.pickle file',
                    default="scripts/map_generation/map.pickle",
                    type=str)

parser.add_argument(
    '--nyu_mapping',
    help='Path to nyu mapping .yaml',
    default=
    "/home/rene/catkin_ws/src/active_learning_for_segmentation/embodied_active_learning/cfg/airsim/semanticClassesFlat.yaml",
    type=str)
args = parser.parse_args()

outputFolder = args.out_folder
pointsToSample = args.n_imgs
minSemanticClasses = args.min_sem_classes
nyuMappingsYaml = args.nyu_mapping

airSimSemanticsConverter = airsim_semantics.AirSimSemanticsConverter(nyuMappingsYaml)
airSimSemanticsConverter.set_airsim_classes()

# Mapping from airsim type to string
typeToName = {
    '0': "img",
    '1': "depth",
    '5': "semantic",
    '7': "mask"
}  # 7 is semantics as infrared

# Font to write numbers on preview image
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 0, 255)
lineType = 2

map_struct = pickle.load(open(args.map, "rb"))
map_occupied = map_struct['map']  # Binary map
map_rgb = pickle.load(open(args.minimap, "rb"))['map']
if map_rgb.shape[-1] != 3:
    map_rgb = np.stack([(map_rgb == 0) * 255 for i in range(3)],
                       axis=-1).astype(np.uint8)  # Black White 3 channel image
lengthPerPixel = map_struct['dimensions'][
    'lengthPerPixel']  # Conversion from pixel to meters in unreal
top_start, left_start = map_struct['start']['top'], map_struct['start'][
    'left']  # Start position of the dron
top_start = top_start - 50
left_start = left_start# - 400
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
        if map_occupied[top, left] != 1:
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

    pitch = 0
    roll  = 0
    if args.sample_z:
        currPose.position.z_val = np.random.rand() * 1.5 - 1
    if args.sample_pitch:
        pitch = -np.random.rand() * 0.5 * math.pi + 0.25*math.pi
    if args.sample_roll:
        roll = -np.random.rand() * 0.5 * math.pi + 0.25*math.pi

    (x,y,z,w) = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    currPose.orientation.x_val = x
    currPose.orientation.y_val = y
    currPose.orientation.z_val = z
    currPose.orientation.w_val = w

    client.simSetVehiclePose(currPose, True)

    time.sleep(0.2)

    responses = client.simGetImages([
        airsim.ImageRequest("front", airsim.ImageType.Scene),
        airsim.ImageRequest("front", airsim.ImageType.DepthPlanner, True),
        airsim.ImageRequest("front", airsim.ImageType.Infrared, False, False)
    ])

    cnt += 1
    validImage = True
    # Check if there are enough semantic classes
    for response in responses:
        if response.image_type == 7:  # infrared (Semantics are encoded as infrared value)
            # get numpy array
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            # 3 channels with same value, just use one
            img_semantics = img1d.reshape(response.height, response.width,
                                            3)[:, :, 0]

            if len(np.unique(img_semantics)) < minSemanticClasses:
                print(
                    "Image #{} did not have enough semantic classes ({})\n Going to request another pose"
                    .format(cnt, len(np.unique(img_semantics))))
                cnt = cnt - 1
                validImage = False
                break

            img_semantics = airSimSemanticsConverter.map_infrared_to_nyu(
                img_semantics)

            print(
                "Classes in img:", ",".join([
                    airSimSemanticsConverter.get_nyu_name_for_nyu_id(idx)
                    for idx in np.unique(img_semantics)
                ]))
            file_name = '{}_{:04d}.png'.format(
                typeToName[str(response.image_type)], cnt)

            Image.fromarray(img_semantics.astype(np.uint8)).save(
                os.path.join(outputFolder, file_name))
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
            Image.fromarray(previewImg).save(
                os.path.join(outputFolder, 'selected_poses.png'))
    if not validImage:
        continue

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
            print("Saved airsim image {}_{:04d}.png".format(
                        typeToName[str(response.image_type)], cnt))


Image.fromarray(previewImg).save(
    os.path.join(outputFolder, 'selected_poses.png'))
# plt.imshow(previewImg)
# plt.show()
