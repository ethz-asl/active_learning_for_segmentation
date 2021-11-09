# Active_learning_for_segmentation

## Embodied Active Learning Package
Main package built on top of all other packages. Conduct embodied experiments with either airsim or gibson

See [here](embodied_active_learning)


## Embodied Active Learning Core
Contains main functionality needed for embodied active learning package. 
- Models
- Uncertanties
- Replay / Trainingsbuffer

See [here](embodied_active_learning_core)


## Volumetric Labeling
Contains code for volumetric labeling and image selection.
- Pseudo Label selection
- Code to find subset of images or voxels to annotate

See [here](volumetric_labeling)

## Habitat ROS 
A package that connects the habitat simulator with the ros interface is located [here](habitat_ros/README.md)