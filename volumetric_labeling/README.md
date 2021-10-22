# Embodied Active Learning
**embodied_active_learning** is a framework autonomously gather trainings by actively roaming around an unknown environment

# Table of Contents
**Setup**
* [Dependencies](#Dependencies)
* [Structure](#Structure)
* [Installation](#Installation)
* [Data Repository](#Data-Repository)

**Examples**

**Documentation**

# Setup
## Dependencies
  * `catkin_simple` ([https://github.com/catkin/catkin_simple](https://github.com/catkin/catkin_simple))
  * `mav_active_3d_path_planning` ([https://github.com/ethz-asl/mav_active_3d_planning](https://github.com/ethz-asl/mav_active_3d_planning))
  
# Structure
The embodied active learning package consists of four main parts:
- **data_acquisitors** Contains Code to sample RGBD + Semantic Labels during the robots execution.
- **uncertainty_estimation** Contains Code to predict uncertainty and semantic classes for RGB images
- **experiments** Contains code to control the experiments (Start Simulation, Fly to start point, ...)
- **visualization** Contains code that is only used for visualization purposes (e.g. Minimap)

**Experiment Configs**
In order to have different configurations for your experiment (e.g. Different Segmentation Network,...) please create a .yaml file in the cfg/experiments folder.
The yaml file must consist of three main parts as follows:
```yaml 
uncertainty:
  network:
    name: "<NetworkName>"
  # Additional Parameters depending on network name
data_generation:
  type: "<DataAcquisitorName>"
  # Additional parameters depending on selected sampler

start_position:
  x: 0
  y: 0
  z: 0
  yaw: 3.14
```
Currently the following configurations are possible:

**Uncertainty Estimation**<br>
- **Network**

    | Network Name  | Parameter      |Description  |
    | ------------- |:-------------:| -----|
    | lightweight-refinenet | - | Uses lightweight-refinenet as semantic segemenation Network |
    |    | size      |   Size of the encoder [50,101,152]  |
    | |  classes     |    How many output classes (default 40) |
    | |  pretrained     |   Use pretrained network (default True) |
    | | | |

- **Method** <br>

    | Type | Parameter      |Description  |
    | ------------- |:-------------:| -----|
    | Softmax | - | Directly uses softmax values as uncertainty|
    | |from_logits | Set to False if the networks last layer is a softmax layer |
    ||||

 **Data Generation** <br>

-  
    | Type | Parameter      |Description  |
    | ------------- |:-------------:| -----|
    | constantSampler | - | Sample Trainings images with a constant frequency|
    | | rate  | Frequency that is used to sample images (default 0.5) |
    | | output_folder | Where captured images should be stored (default /tmp)|
    ||||


## Installation
Installation instructions for Linux.

**Prerequisites**

1. If not already done so, install [ROS](http://wiki.ros.org/ROS/Installation) (Desktop-Full is recommended).

2. If not already done so, create a catkin workspace with [catkin tools](https://catkin-tools.readthedocs.io/en/latest/):

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

1. Move to your catkin workspace:
```shell script
cd ~/catkin_ws/src
```

2. Install system dependencies:
```shell script
sudo apt-get install python-wstool python-catkin-tools
```

3. Download repo using a SSH key or via HTTPS:
```shell script
git clone https://github.com/ethz-asl/active_learning_for_segmentation # HTTPS
```
4. Initialize git submodules
```shell script
cd active_learning_for_segmentation
git submodule init
```
   
4. Download and install the dependencies of the packages you intend to use.
    TODO
5. Source and compile:
```shell script
source ../../devel/setup.bash
catkin build embodied_active_learning # Builds this package only
catkin build # Builds entire workspace, recommended for full install.
```

# Examples

# Documentation