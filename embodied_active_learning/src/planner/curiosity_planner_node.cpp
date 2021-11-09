#include "active_3d_planning_ros/module/module_factory_ros.h"
#include "active_3d_planning_ros/planner/ros_planner.h"

#include "active_3d_planning_panoptic/initialization/panoptic_package.h"
#include "active_3d_planning_mav/initialization/mav_package.h"

#include "panoptic_mapping_ros/panoptic_mapper.h"
#include "active_3d_planning_panoptic/map/single_tsdf_panoptic_map.h"

#include <minkindr_conversions/kindr_tf.h>
#include <glog/logging.h>
#include <chrono>
#include <thread>

int main(int argc, char **argv) {
    // leave some time for the rest to settle
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // init ros
    ros::init(argc, argv, "curiosity_planner_node");

    // prevent the linker from optimizing these packages away...
    active_3d_planning::initialize::panoptic_package();
    active_3d_planning::initialize::mav_package();

    // Set logging to debug for testing
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    google::ParseCommandLineFlags(&argc, &argv, false);

    // node handles
    ros::NodeHandle nh("");
    ros::NodeHandle nh_private("~");

    ros::Publisher handler;

    // Creating and start to mapper
    panoptic_mapping::PanopticMapper mapper(nh, nh_private);

    ros::NodeHandle planner_nh("planner");
    ros::NodeHandle planner_nh_private("~planner");

    // Setup
    active_3d_planning::ros::ModuleFactoryROS factory;
    active_3d_planning::Module::ParamMap param_map;
    active_3d_planning::ros::RosPlanner::setupFactoryAndParams(&factory, &param_map, planner_nh_private);

    // Create and launch the mapper
    active_3d_planning::ros::RosPlanner plnner_node(planner_nh, planner_nh_private, &factory, &param_map);

    // REALLY REALLY UGLY. Manually copies map pointer to planner
    active_3d_planning::Map& map = plnner_node.getMap();
    if (typeid(map) == typeid(active_3d_planning::map::PanopticMap)) {
        active_3d_planning::map::PanopticMap& internal_map_ = dynamic_cast<active_3d_planning::map::PanopticMap&>(map);
        internal_map_.setPanopticMapper(std::unique_ptr<panoptic_mapping::PanopticMapper>(&mapper));
    }

    ros::AsyncSpinner spinner(mapper.getConfig().ros_spinner_threads);
    spinner.start();
    plnner_node.planningLoop();
    ros::waitForShutdown();

    return 0;
}