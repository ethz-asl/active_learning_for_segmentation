import yaml
import re

def testSemantics():
    """ Tests semantic mapping for current unreal environment.
    To use this, simply copy paste the outline of the unreal editor into the following regex form https://regexr.com/5s2nn
    Than copy the output to the desired yaml file
    """

    availableMeshes = "/home/rene/catkin_ws/src/active_learning_for_segmentation/embodied_active_learning/scripts/airsim_semantic_mapping/available_classes.yaml"
    pathToAirsimMapping = '/home/rene/catkin_ws/src/active_learning_for_segmentation/embodied_active_learning/cfg/airsim/semanticClasses.yaml'

    matched_print = []
    error_print = []

    with open(pathToAirsimMapping) as file:
        with open(availableMeshes) as meshFile:
            classMappings = yaml.load(file, Loader=yaml.FullLoader)

            meshes = yaml.load(meshFile, Loader=yaml.FullLoader)['classes']
            for meshName in meshes:
                found = False
                name = None
                regex = None

                for _class in classMappings['classMappings']:
                    if found:
                        break

                    if _class['regex'] != ['']:
                        for pattern in _class['regex']:
                            if re.compile("^{}$".format(pattern)).match(meshName):
                                found = True
                                name = _class['className']
                                regex = pattern
                                break

                    classAndId = "{:<20}".format("{}({})".format(_class['className'],_class['classId']))
                if not found:
                    error_print.append("==> Did not find a matching regex expression for\n {}.".format(meshName))
                else:
                    fixedLengthStart = "{:<30}".format("Assigned [{}] to ".format(name))
                    matched_print.append("{} [{}] due to regex rule: {}".format(fixedLengthStart, "{:<40}".format(meshName), regex))

        print("---- MATCHED ------")
        print("\n".join(matched_print))
        print("\n---- UNMATCHED (will be assigned otherpro) ------")
        print("\n".join(error_print))