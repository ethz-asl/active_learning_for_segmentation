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


class DelayedPlannerStarter
{
    public:
    active_3d_planning::ros::RosPlanner* planner;

    void callback(const ros::TimerEvent& event) {
        planner->planningLoop();
        std::cout << "Planning loop started" << std::endl;
    }
};

int main(int argc, char **argv) {
    // leave some time for the rest to settle

    std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::cout << "=========> @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;

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
    // Setup spinning.


//    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::cout << "=========> 22222222" << std::endl;

    ros::NodeHandle planner_nh("planner");
    ros::NodeHandle planner_nh_private("~planner");
    float planner_delay = 10.0;  // TODO parameter

    // Setup
    active_3d_planning::ros::ModuleFactoryROS factory;
    std::cout << "=========> @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
    active_3d_planning::Module::ParamMap param_map;
    std::cout << "=========> @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
    active_3d_planning::ros::RosPlanner::setupFactoryAndParams(&factory, &param_map, planner_nh_private);
    std::cout << "=========> @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;

    // Create and launch the mapper
    active_3d_planning::ros::RosPlanner node(planner_nh, planner_nh_private, &factory, &param_map);
    std::cout << "=========> @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
    // REALLY REALLY UGLY. DO NOT DO THIS TODO @zrene
    active_3d_planning::Map& map = node.getMap();

    std::cout << "=========> @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
    if (typeid(map) == typeid(active_3d_planning::map::PanopticMap)) {
        active_3d_planning::map::PanopticMap& internal_map_ = dynamic_cast<active_3d_planning::map::PanopticMap&>(map);
    std::cout << "=========> @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
//        internal_map_.setPanopticMapper()
        internal_map_.setPanopticMapper(std::unique_ptr<panoptic_mapping::PanopticMapper>(&mapper));//.reset(&mapper);
    std::cout << "=========> @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
    }

    DelayedPlannerStarter delayed_starter;
    std::cout << "=========> @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
    delayed_starter.planner = &node;
//    std::cout << "Curisity Map loaded. Going to wait for " << planner_delay << " seconds before starting mapper..." << std::endl;
//    ros::Timer timer = nh.createTimer(ros::Duration(planner_delay), &DelayedPlannerStarter::callback, &delayed_starter, true);

    ros::AsyncSpinner spinner(mapper.getConfig().ros_spinner_threads);
    spinner.start();
    std::cout << "starting planning loop" << std::endl;
    node.planningLoop();
    std::cout << " after loop" << std::endl;
    ros::waitForShutdown();
    return 0;
}