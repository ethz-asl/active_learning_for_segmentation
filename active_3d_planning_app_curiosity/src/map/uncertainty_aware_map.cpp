#include "../../include/active_3d_planning_app_curiosity/map/uncertainty_aware_map.h"

#include <voxblox_ros/ros_params.h>
#include "active_3d_planning_core/data/system_constraints.h"

namespace active_3d_planning {
    namespace map {

        ModuleFactoryRegistry::Registration<UncertaintyAwareMap> UncertaintyAwareMap::registration(
                "VoxbloxMapWithUncertainty");

        void UncertaintyAwareMap::setupFromParamMap(
                active_3d_planning::ModuleBase::ParamMap *param_map) {
            VoxbloxMap::setupFromParamMap(param_map);
        }

        UncertaintyAwareMap::UncertaintyAwareMap(active_3d_planning::PlannerI &planner)
                : VoxbloxMap(
                planner) {


            ros::NodeHandle nh("uncertainty_map");
            ros::NodeHandle nh_private("~uncertainty_map");

            ros::NodeHandle nh_uncertainty("");
            ros::NodeHandle nh_uncertainty_private("~");

            esdf_server_.reset(new voxblox::EsdfServer(nh, nh_private));
            uncertainty_esdf_server.reset(new voxblox::EsdfServer(nh_uncertainty, nh_uncertainty_private));

            esdf_server_->setTraversabilityRadius(
                    planner_.getSystemConstraints().collision_radius);

            // cache constants
            c_voxel_size_ = esdf_server_->getEsdfMapPtr()->voxel_size();
            c_block_size_ = esdf_server_->getEsdfMapPtr()->block_size();
            c_maximum_weight_ = voxblox::getTsdfIntegratorConfigFromRosParam(nh_private)
                    .max_weight;  // direct access is not exposed

        }

        bool UncertaintyAwareMap::isObserved(const Eigen::Vector3d &point) {
            std::cout << "GOT IS OBSERVEED CALL " << std::endl;
            return VoxbloxMap::isObserved(point);
        }

        int UncertaintyAwareMap::getUncertaintyAtPosition(const Eigen::Vector3d &point) {
//            uncertainty_esdf_server->getEsdfMapPtr()->getEsdfLayer().getBlockPtrByCoordinates(point)->getVoxelPtrByCoordinates()
//            auto block = uncertainty_esdf_server->getEsdfMapPtr()->getEsdfLayerConstPtr()->getVoxelPtrByCoordinates(point);
//            uncertainty_tsdf_server->getTsdfMapPtr()->getTsdfLayerConstPtr()->getVoxelPtrByCoordinates(point);
            return -1;
//            getVertexColor(uncertainty_esdf_server->getEsdfMapPtr(),)
        }
    }
}