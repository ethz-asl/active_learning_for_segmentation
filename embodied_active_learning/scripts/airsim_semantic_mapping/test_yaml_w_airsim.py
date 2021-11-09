"""
Sets the airsim IDs as specified in the mapping yaml file and captures a semantic image.
Show image and different classes in this image

can be used for debuggin purposes
"""
import yaml
import airsim
import numpy as np
import matplotlib.pyplot as plt

pathToAirsimMapping = '/home/rene/catkin_ws/src/active_learning_for_segmentation/embodied_active_learning/cfg/airsim/semanticClassesFlat.yaml'
with open(pathToAirsimMapping) as file:
    classMappings = yaml.load(file, Loader=yaml.FullLoader)
    airsim2Nyu = classMappings['airsimInfraredToNyu']
    client = airsim.MultirotorClient()

    print(
        "Going to overwrite semantic mapping of airsim using config stored at",
        pathToAirsimMapping)
    client.simSetSegmentationObjectID(".*",
        40)  # Default assign otherpro to everything that is not matched

    for _class in classMappings['classMappings']:
        if _class['regex'] != ['']:
            classAndId = "{:<20}".format("{}({})".format(
                _class['className'], _class['classId']))
            regexPattern = "{}".format("|".join(_class['regex']))
            print("{} : Regex Patterns: {}".format(classAndId, regexPattern))
            # m[_class['classId']] = _class['className']
            for pattern in _class['regex']:
                pattern = pattern
                res = client.simSetSegmentationObjectID(pattern,
                                                        _class['classId'], True)
                if not res:
                    print("===> {} ({})".format(res, pattern))

    responses = client.simGetImages(
        [airsim.ImageRequest("0", airsim.ImageType.Infrared, False, False)])

    for idx, response in enumerate(responses):
        filename = 'c:/temp/py_seg_' + str(idx)
        # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        img1d = np.fromstring(response.image_data_uint8,
                              dtype=np.uint8)  #get numpy array
        for i in np.unique(img1d):
            print("airsim id", i)
            try:
                print("{}:{}".format(i, airsim2Nyu[i]))
                # print(m[airsim2Nyu[i]])
            except:
                print("otherpro")
        img_rgb = img1d.reshape(response.height, response.width, 3)
        plt.imshow(img_rgb)
        plt.show()
