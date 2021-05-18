#ifndef ACTIVE_3D_PLANNING_APP_CURIOSITY_UNCERTAINTY_AWARE_MAP_H
#define ACTIVE_3D_PLANNING_APP_CURIOSITY_UNCERTAINTY_AWARE_MAP_H

#include "active_3d_planning_voxblox/map/voxblox.h"

namespace active_3d_planning {
    namespace map {
        class UncertaintyAwareMap : public VoxbloxMap {
        public:
            explicit UncertaintyAwareMap(PlannerI& planner);  // NOLINT;
            // Overwrite parent method
            void setupFromParamMap(Module::ParamMap* param_map) override;
            // Returns uncertainty value for given voxel
            int getUncertaintyAtPosition(const Eigen::Vector3d& point);

            // check whether point is part of the map
            bool isObserved(const Eigen::Vector3d& point) override;

        protected:
            static ModuleFactoryRegistry::Registration<UncertaintyAwareMap> registration;
            // esdf server that contains the map, subscribe to external ESDF/TSDF updates
            std::unique_ptr<voxblox::EsdfServer> uncertainty_esdf_server;

        };
    }
}


#endif //ACTIVE_3D_PLANNING_APP_CURIOSITY_UNCERTAINTY_AWARE_MAP_H
