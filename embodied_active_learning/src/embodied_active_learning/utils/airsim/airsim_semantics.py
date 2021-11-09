"""
Helper Class that converts AirSim Classes to NYU Classes
"""
import yaml
import numpy as np
import pandas as pd


class AirSimSemanticsConverter:
  """
  Helper Class that converts AirSim Classes to NYU Classes
  """

  def __init__(self, path_to_airsim_mapping, verbosity=1):
    self.path_to_airsim_mapping = path_to_airsim_mapping
    # TODO change, not hardcoded
    path_to_csv = "/home/rene/catkin_ws/src/active_learning_for_segmentation/embodied_active_learning/cfg/nyu40_segmentation_mapping.csv"
    self.yaml_config = None
    with open(path_to_airsim_mapping) as file:
      self.yaml_config = yaml.load(file, yaml.FullLoader)

    csv = pd.read_csv(path_to_csv)
    self.idx_to_color = csv[['red', 'green', 'blue']].to_numpy()

    self.nyu_id_to_name = {}
    for _class in self.yaml_config['classMappings']:
      self.nyu_id_to_name[_class['classId']] = _class['className']
    self.verbosity = verbosity

  def set_airsim_classes(self, debug=False):
    """ Sets all class IDs in the Airsim environment to NYU classes """

    import airsim
    client = airsim.MultirotorClient()

    print(
      "Going to overwrite semantic mapping of airsim using config stored at",
      self.path_to_airsim_mapping)
    client.simSetSegmentationObjectID(
      ".*", 39, True)  # Set otherpro as default class for everything

    for _class in self.yaml_config['classMappings']:
      if _class.get('regex', [None]) != [None]:
        if debug:
          class_and_id = "{:<20}".format("{}({})".format(
            _class['className'], _class['classId']))
          regex_pattern = "{}".format("|".join(_class['regex']))
          print("{} : Regex Patterns: {}".format(
            class_and_id, regex_pattern))
        for pattern in _class['regex']:
          pattern = pattern.replace(".*", "[\w]*")
          res = client.simSetSegmentationObjectID(
            pattern, _class['classId'] + 1, True)
          if not res and self.verbosity > 1:
            print(
              "Did not find matching Airsim mesh for pattern ({})".format(pattern))
    print("Airsim IDs Set")

  def get_nyu_name_for_nyu_id(self, id):
    """ Returns the NYU name for a NYU ID """

    return self.nyu_id_to_name.get(id, "unknown id {}".format(id))

  def map_infrared_to_nyu(self, infrared_img):
    """
    Maps an infrared value to the original nyu class. For some reason setting airsim ID to 1 will not
    result in an infrared value of 1 but 16./home/rene/thesis/debug_ss
    Args:
        infrared_img: Numpy array (h,w)
    """
    mapping = self.yaml_config['airsimInfraredToNyu']
    for infrared_id in mapping.keys():
      infrared_img[infrared_img == infrared_id] = mapping[infrared_id]

    invalid_ids = infrared_img >= 40
    if np.any(invalid_ids):
      print(
        "[WARNING] found infrared IDs that were not assigned an NYU class. Will map them to otherpro ({} - #{} items)".format(
          np.unique(infrared_img[invalid_ids]),
          np.sum(np.sum(invalid_ids))))
      infrared_img[invalid_ids] = 39

    return infrared_img  # Add one to make sure that invalid can be mapped to 0

  def semantic_prediction_to_nyu_color(self, predictions: np.ndarray):
    predictions_col = np.stack([predictions, predictions, predictions], axis=-1)
    for _class in np.unique(predictions.ravel()):
      class_occurence = predictions == _class
      if _class != 255:
        predictions_col[class_occurence, :] = self.idx_to_color[_class + 1, :]
      else:

        predictions_col[class_occurence, :] = self.idx_to_color[0, :]
    return predictions_col
