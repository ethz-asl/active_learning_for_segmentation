#include "volumetric_labler.h"
#include <glog/logging.h>
#include <ros/ros.h>

int main(int argc, char **argv) {
  // Start Ros.
  ros::init(argc, argv, "volumetric_labler",
            ros::init_options::NoSigintHandler);

  // Always add these arguments for proper logging.
  config_utilities::RequiredArguments ra(
      &argc, &argv, {"--logtostderr", "--colorlogtostderr"});

  // Setup logging.
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::ParseCommandLineFlags(&argc, &argv, false);

  // Setup node.
  ros::NodeHandle nh("");
  ros::NodeHandle nh_private("~");
  volumetric_labeling::VolumetricLabler labler(nh, nh_private);

  // Setup spinning.
  ros::AsyncSpinner spinner(labler.getConfig().ros_spinner_threads);
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
