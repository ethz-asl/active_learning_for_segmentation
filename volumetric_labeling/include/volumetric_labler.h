#ifndef VOLUMETRIC_LABELING_VOLUMETRIC_LABLER_H_
#define VOLUMETRIC_LABELING_VOLUMETRIC_LABLER_H_
#include "embodied_active_learning/label_request.h"
#include "panoptic_mapping_ros/panoptic_mapper.h"
#include "volumetric_labeling/label_request.h"
#include <panoptic_mapping/3rd_party/config_utilities.hpp>
#include <panoptic_mapping_utils/voxel_clustering/voxel_clustering.h>

namespace volumetric_labeling {
class VolumetricLabler {
public:
  // Config.
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 2;
    int ros_spinner_threads = std::thread::hardware_concurrency();
    bool useSpatialLabeling = true;
    int minInstanceSize = 20; // #Voxels
    bool useGtClustering = true;
    bool ignoreBackground = true;
    std::string scoringMethod = "uncertainty";

    Config() { setConfigName("VolumetricLabler"); }

  protected:
    void setupParamsAndPrinting() override;

    void checkParams() const override;
  };

  /* Construction */
  VolumetricLabler(const ros::NodeHandle &nh,
                   const ros::NodeHandle &nh_private);

  virtual ~VolumetricLabler() = default;
  const Config &getConfig() const { return config_; }

private:
  std::unique_ptr<panoptic_mapping::PanopticMapper> mapper_;
  // Run a second panoptic mapper in order to store gt labels for each voxel.
  // TODO(zrene) migrate to use second map in panoptic_mapper.
  std::unique_ptr<panoptic_mapping::PanopticMapper> gt_mapper_;
  Config config_;
  // Node handles.
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  // Services
  ros::ServiceServer label_service;
  bool spatial_labeling_callback(label_request::Request &request,
                                 label_request::Response &response);

  void getOverlapForInstance(const panoptic_mapping::Submap &gt_submap, const panoptic_mapping::Submap &prediction_submap, std::vector<Instance> *gt_instances, Instance *requested_instance,std::vector<int>* overlaps);
};
} // namespace volumetric_labeling
#endif