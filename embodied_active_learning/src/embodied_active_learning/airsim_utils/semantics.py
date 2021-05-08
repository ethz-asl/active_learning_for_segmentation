import yaml
import numpy as np


class AirSimSemanticsConverter:

    def __init__(self, pathToAirsimMapping):
        self.pathToAirsimMapping = pathToAirsimMapping
        self.yamlConfig = None
        with open(pathToAirsimMapping) as file:
            self.yamlConfig = yaml.load(file, Loader=yaml.FullLoader)

        self.nyuIdToName = {}
        for _class in self.yamlConfig['classMappings']:
            self.nyuIdToName[_class['classId']] = _class['className']

    def setAirsimClasses(self, debug=False):
        """ Sets all class IDs in the Airsim environment to NYU classes """

        import airsim
        client = airsim.MultirotorClient()

        print(
            "Going to overwrite semantic mapping of airsim using config stored at",
            self.pathToAirsimMapping)
        client.simSetSegmentationObjectID(
            ".*", 39, True)  # Set otherpro as default class for everything

        for _class in self.yamlConfig['classMappings']:
            if _class['regex'] != ['']:
                if debug:
                    classAndId = "{:<20}".format("{}({})".format(
                        _class['className'], _class['classId']))
                    regexPattern = "{}".format("|".join(_class['regex']))
                    print("{} : Regex Patterns: {}".format(
                        classAndId, regexPattern))
                for pattern in _class['regex']:
                    pattern = pattern
                    res = client.simSetSegmentationObjectID(
                        pattern, _class['classId'], True)
                    if not res:
                        print(
                            "Did not find matching Airsim mesh for pattern ({})"
                            .format(pattern))
        print("Airsim IDs Set")

    def getNyuNameForNyuId(self, id):
        return self.nyuIdToName.get(id, "unknown id {}".format(id))

    def mapInfraredToNyu(self, infraredImg):
        """
        Maps an infrared value to the original nyu class. For some reason setting airsim ID to 1 will not resutls in
        an infrared value of 1 but 16.
        Args:
            infraredImg: Numpy array (h,w)
        """
        mapping = self.yamlConfig['airsimInfraredToNyu']
        for infraredId in mapping.keys():
            infraredImg[infraredImg == infraredId] = mapping[infraredId]

        invalidIds = infraredImg >= 40
        if np.any(invalidIds):
            print(
                "[WARNING] found infrared IDs that were not assigned an NYU class. Will map them to otherpro ({} items)"
                .format(np.sum(invalidIds)))
            infraredImg[invalidIds] = 39

        return infraredImg
