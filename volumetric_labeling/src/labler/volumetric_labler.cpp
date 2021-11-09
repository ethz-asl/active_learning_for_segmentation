#include "volumetric_labler.h"
#include <memory>
#include <random>

namespace volumetric_labeling {
void VolumetricLabler::Config::checkParams() const { /* Nothing to do here */
}

void VolumetricLabler::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("useSpatialLabeling", &useSpatialLabeling);
  setupParam("minInstanceSize", &minInstanceSize);
  setupParam("useGtClustering", &useGtClustering);
  setupParam("scoringMethod", &scoringMethod);
  setupParam("ignoreBackground", &ignoreBackground);
}

VolumetricLabler::VolumetricLabler(const ros::NodeHandle &nh,
                                   const ros::NodeHandle &nh_private)
    : nh_(nh), nh_private_(nh_private),
      config_(config_utilities::getConfigFromRos<VolumetricLabler::Config>(
                  nh_private)
                  .checkValid()) {
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();

  mapper_ = std::make_unique<panoptic_mapping::PanopticMapper>(
      ros::NodeHandle(nh_private_, "panoptic"),
      ros::NodeHandle(nh_private_, "panoptic"));

  if (config_.useSpatialLabeling) {
    gt_mapper_ = std::make_unique<panoptic_mapping::PanopticMapper>(
        ros::NodeHandle(ros::NodeHandle(nh_private_, "panoptic"), "gt"),
        ros::NodeHandle(ros::NodeHandle(nh_private_, "panoptic"), "gt"));
    label_service = nh_.advertiseService(
        "label_instance", &VolumetricLabler::spatial_labeling_callback, this);
  }
}

void VolumetricLabler::getOverlapForInstance(
    const panoptic_mapping::Submap &gt_submap,
    const panoptic_mapping::Submap &prediction_submap,
    std::vector<Instance> *gt_instances, Instance *requested_instance,
    std::vector<int> *overlaps) {

  panoptic_mapping::voxel_clustering::getConnectedInstancesForSeeds(
      *requested_instance, &gt_submap, gt_instances);

  for (const Instance &i : *gt_instances) {
    int overlap_cnt = 0;
    int unlabeled_size = 0;
    for (auto voxel_idx : i) {
      if (std::find(requested_instance->begin(), requested_instance->end(),
                    voxel_idx) != requested_instance->end()) {

        // Check if already labeled
        if (!prediction_submap.getClassLayer().getVoxelPtrByGlobalIndex(
                voxel_idx) ||
            prediction_submap.getClassLayer()
                .getVoxelPtrByGlobalIndex(voxel_idx)
                ->is_groundtruth) {
          continue;
        }
        unlabeled_size++;
        if (config_.ignoreBackground) {
          int gt_class = gt_submap.getClassLayer()
                             .getVoxelPtrByGlobalIndex(voxel_idx)
                             ->current_index;
          if (gt_class == 0 || gt_class == 1 || gt_class == 21) {
            continue; // Skipp Wall / Floor / Ceiling
          }
        }
        overlap_cnt++;
      }
    }
    overlaps->push_back(overlap_cnt);
  }
}

bool VolumetricLabler::spatial_labeling_callback(
    label_request::Request &request, label_request::Response &response) {
  const panoptic_mapping::Submap &prediction_submap =
      mapper_->getSubmapCollection().getSubmap(
          mapper_->getSubmapCollection().getActiveFreeSpaceSubmapID());
  const panoptic_mapping::Submap &gt_submap =
      gt_mapper_->getSubmapCollection().getSubmap(
          mapper_->getSubmapCollection().getActiveFreeSpaceSubmapID());

  panoptic_mapping::voxel_clustering::ScoringMethod scoring_method;
  LOG_IF(INFO, config_.verbosity >= 1)
      << " Reuested scoring method: " << config_.scoringMethod;

  if (config_.scoringMethod == "size") {
    scoring_method = panoptic_mapping::voxel_clustering::SIZE;
  } else if (config_.scoringMethod == "entropy") {
    scoring_method = panoptic_mapping::voxel_clustering::ENTROPY;
  } else if (config_.scoringMethod == "probability") {
    scoring_method = panoptic_mapping::voxel_clustering::BELONGS_PROBABILITY;
  } else if (config_.scoringMethod == "uncertainty") {
    scoring_method = panoptic_mapping::voxel_clustering::UNCERTAINTY;
  } else if (config_.scoringMethod == "random") {
    scoring_method =
        panoptic_mapping::voxel_clustering::SIZE; // Random just needs any
                                                  // scoring method
  }

  // Stores all instances in the predicted submap
  std::vector<Instance> predicted_instances;
  if (!panoptic_mapping::voxel_clustering::getAllConnectedInstances(
          &prediction_submap, &predicted_instances)) {
    LOG(ERROR) << "Could not get all connected Instances.";
    return false;
  }
  std::priority_queue<panoptic_mapping::voxel_clustering::InstanceInfo>
      instances_with_score;

  panoptic_mapping::voxel_clustering::scoreConnectedInstances(
      &prediction_submap, &predicted_instances, false, scoring_method,
      &instances_with_score);

  LOG_IF(INFO, config_.verbosity >= 1)
      << "Found #" << predicted_instances.size()
      << " Instances in predicted map." << std::endl;

  // Instance which should be annotated
  Instance *requested_instance = nullptr;
  // Information about this instance
  panoptic_mapping::voxel_clustering::InstanceInfo instance_info(0, 0, -1);

  // Used for random labeling
  std::vector<Instance> non_zero_instances;
  std::random_device rd;  // Will be used to obtain a seed for the random
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine
  std::uniform_int_distribution<> *distrib;

  if (config_.scoringMethod == "random") {
    // Copy all instances that match request from queue to array
    while (!instances_with_score.empty()) {
      instance_info = instances_with_score.top();
      instances_with_score.pop();
      if (instance_info.size < config_.minInstanceSize)
        continue;
      non_zero_instances.push_back(instance_info.instance);
    }
    // Sample random instance
    distrib =
        new std::uniform_int_distribution<>(0, non_zero_instances.size() - 1);
    int idx_sampled = (*distrib)(gen);
    requested_instance = &non_zero_instances.at(idx_sampled);

  } else {
    // We want to use scoring functions. Select instance with highest score that
    // matches requirements (min size,...)
    while (!instances_with_score.empty()) {
      instance_info = instances_with_score.top();
      instances_with_score.pop();

      if (!instances_with_score.empty()) {
        if (config_.ignoreBackground && (instance_info.assigned_class == 0 ||
                                         instance_info.assigned_class == 1 ||
                                         instance_info.assigned_class == 21)) {
          continue; // Skipp Wall / Floor / Ceiling in requests if possible
        }

        if (instance_info.size < config_.minInstanceSize)
          continue;
      }
      requested_instance = &instance_info.instance;
      break;
    }
  }

  if (!requested_instance) {
    LOG(WARNING) << "No Instance found that is bigger than "
                 << config_.minInstanceSize;
    return false;
  }

  // We have now found an instance we want to annotate.
  // Find instance in gt map with most overlap if requested in config

  std::vector<Instance> gt_instances;
  Instance req_instance;

  if (config_.useGtClustering) {
    // For each instance in the gt map store overlap
    std::vector<int> gt_overlap;
    // Idx of instance with most overlap
    int max_overlap_idx = 0;
    // If there is no instance with overlap (e.g. only background requested),
    // this defines how many new instances can be requested before the loop
    // terminates
    const int max_retry_count = 1000;
    int try_count = 0;

    // Keep going aslong as the instances in the predicted map are not empty or
    // max try count is reached
    while (
        (config_.scoringMethod != "random" && !instances_with_score.empty()) &&
        try_count++ < max_retry_count) {
      gt_instances.clear();
      gt_overlap.clear();
      getOverlapForInstance(gt_submap, prediction_submap, &gt_instances,
                            requested_instance, &gt_overlap);

      LOG_IF(INFO, config_.verbosity >= 1)
          << "Found #" << gt_overlap.size()
          << " groundtruth instances with overlap.";

      max_overlap_idx = std::max_element(gt_overlap.begin(), gt_overlap.end()) -
                        gt_overlap.begin();

      if (gt_overlap[max_overlap_idx] == 0) {
        // No overlap found. Should not happen but if it does (e.g. only
        // background matches request), request next instance in queue
        LOG(WARNING) << "No Groundtruth instance with overlap found. Going to "
                        "request another instance.";
        // Request next instance in queue
        if (config_.scoringMethod == "random") {
          // Select another random instance
          requested_instance = &non_zero_instances.at((*distrib)(gen));
        } else {
          instance_info = instances_with_score.top();
          instances_with_score.pop();
          if (!instances_with_score.empty()) {
            if (config_.ignoreBackground && instance_info.assigned_class == 0 ||
                instance_info.assigned_class == 1 ||
                instance_info.assigned_class == 21) {
              continue; // Skipp Wall / Floor / Ceiling in requests if possible
            }

            if (instance_info.size < config_.minInstanceSize)
              continue;
          }
          req_instance = instance_info.instance;
          requested_instance = &req_instance;
        }
      } else {
        // Found an instance with valid overlap.
        break;
      }
    }

    LOG_IF(INFO, config_.verbosity >= 1)
        << "Max Overlap for Instance #" << max_overlap_idx << ". Overlap: #"
        << gt_overlap[max_overlap_idx] << "("
        << gt_instances[max_overlap_idx].size() /
               static_cast<float>(gt_overlap[max_overlap_idx])

        << "%) size: " << gt_instances[max_overlap_idx].size();
    // Set requested instance to gt instance with max overlap
    requested_instance = &gt_instances[max_overlap_idx];
  }

  response.Size = 0;
  response.Changed = 0;
  // Store which labels had been assigned
  std::set<int> assigned_labels;
  // Label the instance
  for (auto voxel_idx : *requested_instance) {
    const panoptic_mapping::ClassVoxelType *predicted_voxel =
        prediction_submap.getClassLayer().getVoxelPtrByGlobalIndex(voxel_idx);
    if (!predicted_voxel) {
      LOG(WARNING) << "A given voxel was not found in the predicted map.";
      continue;
    }

    if (predicted_voxel->is_groundtruth) {
      continue; // Skip gt voxels. These do not count towards size since they
                // are already labeled.
    }
    response.Size++;

    const panoptic_mapping::ClassVoxelType *gt_voxel =
        gt_submap.getClassLayer().getVoxelPtrByGlobalIndex(voxel_idx);

    if (!gt_voxel) {
      LOG(WARNING) << "Grondtruth voxel for given predicted voxel was null.";
      continue;
    }

    if (gt_voxel->current_index != predicted_voxel->current_index) {
      // Really need to change this voxel
      response.Changed++;
    }
    // Mark as GT annotated and change classes
    mapper_->labelClassVoxel(
        voxel_idx, mapper_->getSubmapCollection().getActiveFreeSpaceSubmapID(),
        gt_voxel->current_index);
    // Store new classes
    assigned_labels.insert(gt_voxel->current_index);
  }

  for (auto c : assigned_labels) {
    response.GTClasses.push_back(c);
  }
  return true;
}
} // namespace volumetric_labeling
