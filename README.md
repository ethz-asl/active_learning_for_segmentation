# Volumetric Labeling
Package to select and annotate a subset of images or voxels.

## Installation
Installation instructions for Linux.

**Prerequisites**

```shell script
sudo apt-get install python-catkin-tools
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
catkin config --extend /opt/ros/melodic  # exchange melodic for your ros distro if necessary
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin config --merge-devel
```

**Installation**
 
Source and compile:
```shell script
source ../../devel/setup.bash
catkin build volumetric_labeling # Builds this package only
```

### Examples
These example assume that a folder 'embodied_experiment_output' exists and contains folder 'step000' with pickled trainingsdata

**Voxel Based**
```
roslaunch volumetric_labeling voxel_based_labler.launch output_folder:=<output> experiment_path:=embodied_experiment_output step:=step_000
```

**Image Based**
```
roslaunch volumetric_labeling image_labler.launch output_folder:=<output> experiment_path:=embodied_experiment_output step:=step_000 images_to_label:=<n_imgs_to_label>
```

**Available Labeling Strategies**

The following annotation strategies are available

**Image Based** \
Argument labeling_strategy
- UNIFORM
- RANDOM
- UNLABELED_VOXELS
- WEIGHTS
- MOST_SEEN_VOXELS
- UNCERTAINTY
- UNCERTAINTY_RAW
- MODEL_MAP_MISSMATCH

**Voxel Based** \
Argument scoring_method
- SIZE
- ENTROPY
- BELONGS_PROBABILITY
- UNCERTAINTY
- RANDOM
