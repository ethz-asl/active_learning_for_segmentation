# Embodied Active Learning
**embodied_active_learning** is a framework autonomously gather trainings by actively roaming around an unknown environment
# Setup
## Dependencies
  * `catkin_simple` ([https://github.com/catkin/catkin_simple](https://github.com/catkin/catkin_simple))
  * `mav_active_3d_path_planning` ([https://github.com/ethz-asl/mav_active_3d_planning](https://github.com/ethz-asl/mav_active_3d_planning))
    * Branch: zrene/release_embodied
  * `panoptic_mapping` ([https://github.com/ethz-asl/panoptic_mapping](https://github.com/ethz-asl/panoptic_mapping))
    * Branch: zrene/embodied_active_learning
  * `light_weight_refinenet` ([https://github.com/renezurbruegg/light-weight-refinenet](https://github.com/renezurbruegg/light-weight-refinenet))
  
# Structure
The embodied active learning package consists of five main parts:
- **data_acquisitors** Contains Code to sample RGBD + Semantic Labels during the robots execution.
- **uncertainty_estimation** Contains uncertainty node used to predict uncertanties for images
- **experiments** Contains code to control the experiments (Start Simulation, Fly to start point, ...)
- **visualization** Contains code that is only used for visualization purposes (e.g. Minimap)
- **online_learning** Contains code to train a network online

**Experiment Configs**

The file ```utils/config.py``` contains a generic config object which is used for all configurations. 
The values of the object are loaded from .yaml files and can overwrite any property of the object.

Each experiment consists of 5different configuration files:
1. ```cfg/airsim/<classes>.yaml``` contains definition on how to map mesh names to semantic classes in airsim
2. ```cfg/mapper/<panoptic_config>.yaml``` contains panoptic configuration for the underlying map
3. ```cfg/airsim/<planner_config>.yaml``` contains mav_active_planning configuration for the IPP planner
4. ```cfg/experiments/<fully-/self-supervised>/<experiment>.yaml``` contains configuration for data collection, training and uncertainty estimation. This is the main configuration that you will want to update.
5. ```cfg/experiments/environments/<environment>/boundary.yaml``` contains boundaries of the environment with respect to the robots initial position
## Installation
Installation instructions for Linux.

**Prerequisites**

1. If not already done so, install [ROS](http://wiki.ros.org/ROS/Installation) (Desktop-Full is recommended).

2. If not already done so, create a catkin workspace with [catkin tools](https://catkin-tools.readthedocs.io/en/latest/):

```shell script
sudo apt-get install python3-catkin-tools
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
catkin config --extend /opt/ros/noetic  # exchange melodic for your ros distro if necessary
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
  * `mav_active_3d_path_planning` ([https://github.com/ethz-asl/mav_active_3d_planning](https://github.com/ethz-asl/mav_active_3d_planning))
  * `panoptic_mapping` ([https://github.com/ethz-asl/panoptic_mapping](https://github.com/ethz-asl/panoptic_mapping))
  * `light_weight_refinenet` ([https://github.com/renezurbruegg/light-weight-refinenet](https://github.com/renezurbruegg/light-weight-refinenet))
5. Source and compile:
    ```shell script
    source ../../devel/setup.bash
    catkin build embodied_active_learning # Builds this package only
    catkin build # Builds entire workspace, recommended for full install.
    ```
5. Install the pip package
    ```shell script
    pip install -e .
    ```

# Examples
Have a look at the different configuration files in  ```cfg/experiments/<fully-/self-supervised>``` to get an overview of the configuration possibilites.

Starting an experiment.
1. Make sure Airsim is up and running
2. Run ros experiment 
   ```bash
   roslaunch embodied_active_learning panoptic_run.launch
    ```
## Simulators
Currently, the following Simulators are supported:
1. AirSim (use airsim_experiment.py)
2. Habitat using habitat_ros as an ros_interface (use habitat_experiment.py)

## FAQ
# Adding a new model 
1. Add your model to the  ```embodied_active_learning_core``` package as discribed in the readme located in that package.
2. In your config.yaml file change the model name to the newly created model name
3. Thats it!

# Adding a new uncertainty estimator 
1. Add your uncertainy estimator to the  ```embodied_active_learning_core``` package as discribed in the readme located in that package.
2. In your config.yaml file change the uncertainty type to the newly created one.
3. In ```uncertainty_estimation/uncertainty_estimation_node.py```update the method ```get_uncertainty_estimator_for_config```
   1. Below the  # Load Uncertainty Estimator secetion, load your uncertainty estimator
   2. If a custom callback after training is needed (Refitting GMM models, ...) a custom callback can be added to the network using ```network.post_training_hooks.append(<my callback>>)```
