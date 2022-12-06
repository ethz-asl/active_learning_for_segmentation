# Active Learning For Segmentation
![](documentation/movie.gif)

## Table of Contents
* [Paper and Video](#Paper-and-Video)
* [Installation](#Installation)
* [Examples](#Example)
* [Code Structure](#Embodied-Active-Learning-Package)


## Paper and Video
If you find this package useful for your research, please consider citing our paper:
* René Zurbrügg, Hermann Blum, Cesar Cadena, Roland Siegwart and Lukas Schmid. "**Embodied Active Domain Adaptation for Semantic Segmentation via Informative Path Planning**" in *IEEE Robotics and Automation Letters (RA-L)*, 2022.
  \[[ArXiv](https://arxiv.org/abs/2203.00549) | [Video](https://www.youtube.com/watch?v=FeFPEdZzT3w)]

  ```bibtex
  @article{Zurbrgg2022EmbodiedAD,
    title={Embodied Active Domain Adaptation for Semantic Segmentation via Informative Path Planning},
    author={R. {Zurbr{\"u}gg} and H. {Blum} and C. {Cadena} and R. {Siegwart} and L. {Schmid}},
    journal={ArXiv},
    year={2022},
    volume={abs/2203.00549}
  }
  ```
  
### Reproducing the Experiments
#### Installation 
Installation instructions for Linux. It is expected that Unreal Engine is already installed!

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
sudo apt-get install python3-wstool python3-catkin-tools
```

3. Download repo using a SSH key or via HTTPS:
```shell script
git clone https://github.com/ethz-asl/active_learning_for_segmentation
```
4. Download dependencies
New Workspace
```shell script
wstool init . ./embodied_active_learning/dependencies.rosinstall
wstool update
```
Existing Workspace
```shell script
wstool merge -t . ./embodied_active_learning/dependencies.rosinstall
wstool update
```

4. Downloading Additional Dependencies
```shell script
wstool merge -t . ./mav_active_3d_path_planning/mav_active_3d_planning_https.rosinstal
wstool update
```
```shell script
wstool merge -t . ./panoptic_mapping/panoptic_mapping.rosinstall
wstool update
```
5. Install Airsim
  ```shell script
  cd ../AirSim
  ./setup.sh
  ```
6. Install Pip Dependencies
   ``` shell script
   cd ../active_learning_for_segmentation/embodied_active_learning
   pip install -e . 
   cd ../embodied_active_learning_core
   pip install -e . 
   cd ../volumetric_labeling
   pip install -e .
   cd  ../../light-weight-refinenet
   pip install -e .
   cd ../densetorch
   pip install -e .
   ```
8. Source and compile:
    ```shell script
    source ../../devel/setup.bash
    catkin build embodied_active_learning
    ```

### Example
In order to run an experiment, first download the Flat Environment [Here](https://drive.google.com/file/d/17TVKpT9kzytpazMqiCQee4QbiRqwl0Qz/view?usp=sharing).
Secondly, download the [Data Package](https://drive.google.com/file/d/1pBuiNNASQqvGZU6xewf7gnOOwXU3DD5b/view?usp=sharing) and extract it at a location of your choice.  
The Data Package contains the following Files:
- `replayset`: Subset of the Scannet Dataset, which is used as replay buffer.
- `testsets`: Selected Images with groundtruth annotations used for evaluation
- `scannet_50_classes_40_clusters.pth`: Checkpoint of the pretrained Network and uncertainty Estimator.

#### Launching the Experiment
1. Modify the files `embodied_active_learning/cfg/experiments/self_supervised/RunScannetDensity.yaml` as well as `embodied_active_learning/cfg/experiments/mapper/single_tsdf.yaml`. Update all paths (`/home/rene/...`) with your local information.
2. Start Unreal Engine
3. Execute `roslaunch embodied_active_learning panoptic_run.launch` to start an experiment.

#### Outputs
This will create the following files during the experiment:
1. `~/out/<RunName>/checkpoints/best_iteration_X.pth` checkpoint of the model after each training cycle.
2. `~/out/<RunName>/online_training/step_XXX` folder contatining all training samples (images with their respective pseudo labels) that were used for training. 
3. `~/out/<RunName>/poses.csv` CSV file containing all requested robot poses. (X,Y,Z,Quaternion)
4. `~/out/<RunName>/scroes.csv` CSV file containing all mIoU scores on the training and test set.
5. `~/out/<RunName>/params.yaml` yaml file containing all parameters that were set for this experiment

Additional information can be found in the [Emboded Active Learning Package](embodied_active_learning).


### Others
## Repository Content

### Embodied Active Learning Package
Main package built on top of all other packages. Conduct embodied experiments with either airsim or gibson

See [here](embodied_active_learning)


### Embodied Active Learning Core
Contains main functionality needed for embodied active learning package. 
- Models
- Uncertanties
- Replay / Trainingsbuffer

See [here](embodied_active_learning_core)


### Volumetric Labeling
Contains code for volumetric labeling and image selection.
- Pseudo Label selection
- Code to find subset of images or voxels to annotate

See [here](volumetric_labeling)

### Habitat ROS 
A package that connects the habitat simulator with the ros interface is located [here](habitat_ros/README.md)
