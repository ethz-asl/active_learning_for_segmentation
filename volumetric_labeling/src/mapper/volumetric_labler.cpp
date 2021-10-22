#include "volumetric_labler.h"
#include <memory>
//#include <panoptic_mapping_utils/voxel_clustering/voxel_clustering.h>
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
  std::cout << " use spacial labeling " << config_.useSpatialLabeling
            << std::endl;

  if (config_.useSpatialLabeling) {
    gt_mapper_ = std::make_unique<panoptic_mapping::PanopticMapper>(
        ros::NodeHandle(ros::NodeHandle(nh_private_, "panoptic"), "gt"),
        ros::NodeHandle(ros::NodeHandle(nh_private_, "panoptic"), "gt"));
    label_service = nh_.advertiseService(
        "label_instance", &VolumetricLabler::spatial_labeling_callback, this);
  }
}
void printQueue(
    std::priority_queue<panoptic_mapping::voxel_clustering::InstanceInfo> q) {
  int i = 0;
  // printing content of queue
  while (!q.empty()) {
    std::cout << "[" << ++i << "] Size: " << q.top().size
              << " Score: " << q.top().mean_score
              << " class: " << q.top().assigned_class << std::endl;
    q.pop();
  }
}

void VolumetricLabler::getOverlapForInstance(
    const panoptic_mapping::Submap &gt_submap,
    const panoptic_mapping::Submap &prediction_submap,
    std::vector<Instance>* gt_instances, Instance *requested_instance,
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
            //              std::cout << "Skipped BG instance" << std::endl;
            continue; // Skipp Wall / Floor / Ceiling
          }
        }
        overlap_cnt++;
      }
    }
    std::cout << overlap_cnt << " : " << unlabeled_size << std::endl;
//    std::cout << overlap_cnt << std::endl;
    overlaps->push_back(overlap_cnt);
  }
}

bool VolumetricLabler::spatial_labeling_callback(
    label_request::Request &request, label_request::Response &response) {
  std::cout << " STERING SPATIAL LABELING " << std::endl;
  const panoptic_mapping::Submap &prediction_submap =
      mapper_->getSubmapCollection().getSubmap(
          mapper_->getSubmapCollection().getActiveFreeSpaceSubmapID());
  const panoptic_mapping::Submap &gt_submap =
      gt_mapper_->getSubmapCollection().getSubmap(
          mapper_->getSubmapCollection().getActiveFreeSpaceSubmapID());

  std::vector<Instance> predicted_instances;
  if (!panoptic_mapping::voxel_clustering::getAllConnectedInstances(
          &prediction_submap, &predicted_instances)) {
    LOG(ERROR) << "Could not get all connected Instances.";
    //    return false;
  }
  std::priority_queue<panoptic_mapping::voxel_clustering::InstanceInfo>
      instances_with_score;

  panoptic_mapping::voxel_clustering::ScoringMethod scoring_method;
  std::cout << " Reuested scoring method: " << config_.scoringMethod
            << std::endl;
  if (config_.scoringMethod == "size") {
    scoring_method = panoptic_mapping::voxel_clustering::SIZE;
  } else if (config_.scoringMethod == "entropy") {
    scoring_method = panoptic_mapping::voxel_clustering::ENTROPY;
  } else if (config_.scoringMethod == "probability") {
    scoring_method = panoptic_mapping::voxel_clustering::BELONGS_PROBABILITY;
  } else if (config_.scoringMethod == "uncertainty") {
    scoring_method = panoptic_mapping::voxel_clustering::UNCERTAINTY;
  } else if (config_.scoringMethod == "random") {
    scoring_method =  panoptic_mapping::voxel_clustering::SIZE;
  }

  panoptic_mapping::voxel_clustering::scoreConnectedInstances(
      &prediction_submap, &predicted_instances, false, scoring_method,
      &instances_with_score);

  LOG_IF(INFO, config_.verbosity >= 1)
      << "Found #" << predicted_instances.size()
      << " Instances in predicted map" << std::endl;

  Instance *requested_instance = nullptr;
  printQueue(instances_with_score);

  panoptic_mapping::voxel_clustering::InstanceInfo instance_info(0,0,-1);
  std::vector<Instance> non_zero_instances;
  std::random_device rd;  //Will be used to obtain a seed for the random
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine
  std::uniform_int_distribution<> *distrib;
  if (config_.scoringMethod != "random") {
    while (!instances_with_score.empty()) {
      instance_info = instances_with_score.top();

      instances_with_score.pop();
      if (!instances_with_score.empty()) {
        if (instance_info.assigned_class == 0 ||
            instance_info.assigned_class == 1 ||
            instance_info.assigned_class == 21) {
          continue; // Skipp Wall / Floor / Ceiling in requests if possible
        }

        if (instance_info.size < config_.minInstanceSize)
          continue;
      }

      requested_instance = &instance_info.instance;
      break;
    }
  } else {
      // RANDOM
        while (!instances_with_score.empty()) {
        instance_info =
            instances_with_score.top();

        instances_with_score.pop();
        if (instance_info.size < config_.minInstanceSize)
          continue;
        non_zero_instances.push_back(instance_info.instance);
      }// Sample randomly
      std::cout << "got non zero: "<< non_zero_instances.size() << std::endl;

//     Select Randomly. Do NOT use GT Clustering.

       distrib = new std::uniform_int_distribution<>(0,non_zero_instances.size() - 1);
    std::cout << "sampled zero: "<< non_zero_instances.size() << std::endl;
      int idx_sampled = (*distrib)(gen);
      std::cout << "indice: " << idx_sampled <<std::endl;
       requested_instance = &non_zero_instances.at(idx_sampled);
       std::cout << "done" << std::endl;
  }
  //  // RANDOM
  //  std::vector<Instance> non_zero_instances;
  //    while (!instances_with_score.empty()) {
  //    panoptic_mapping::voxel_clustering::InstanceInfo instance_info =
  //        instances_with_score.top();
  //
  //    instances_with_score.pop();
  //    if (instance_info.size < config_.minInstanceSize)
  //      continue;
  //    non_zero_instances.push_back(instance_info.instance);
  //  }// Sample randomly

  // Select Randomly. Do NOT use GT Clustering.
  //  std::random_device rd;  //Will be used to obtain a seed for the random
  //  number engine std::mt19937 gen(rd()); //Standard mersenne_twister_engine
  //  seeded with rd() std::uniform_int_distribution<> distrib(0,
  //  non_zero_instances.size() - 1); requested_instance =
  //  &non_zero_instances.at(distrib(gen));

  if (!requested_instance) {
    LOG(WARNING) << "No Instance found that is bigger than "
                 << config_.minInstanceSize;
    return false;
  }

  std::vector<Instance> gt_instances;
  Instance req_instance;
  int try_count = 0;
  if (config_.useGtClustering) {
    std::cout << "using clustering" << std::endl;
    std::vector<int> gt_overlap;
    int max_overlap_idx = 0;
    while ((config_.scoringMethod != "random" && !instances_with_score.empty() ) || try_count++ < 500) {
      gt_instances.clear();
      gt_overlap.clear();
      getOverlapForInstance(gt_submap, prediction_submap, &gt_instances,
                            requested_instance, &gt_overlap);

      int i = 0;
      for(auto overlap: gt_overlap) {
        std::cout << "[" << i++ << "]: " << overlap << std::endl;
      }

      LOG_IF(INFO, config_.verbosity >= 1)
          << "Found #" << gt_overlap.size()
          << " groundtruth instances with overlap" << std::endl;
      max_overlap_idx = std::max_element(gt_overlap.begin(), gt_overlap.end()) -
                        gt_overlap.begin();

      std::cout << "Max idx:" << max_overlap_idx << std::endl;

      if (gt_overlap[max_overlap_idx] == 0) {
        // No overlap found.
        std::cout << "Overlap was zero..." << std::endl;
        if (config_.scoringMethod != "random") {
          instance_info =
                  instances_with_score.top();

          instances_with_score.pop();
          if (!instances_with_score.empty()) {
            if (instance_info.assigned_class == 0 ||
                instance_info.assigned_class == 1 ||
                instance_info.assigned_class == 21) {
              continue; // Skipp Wall / Floor / Ceiling in requests if possible
            }

            if (instance_info.size < config_.minInstanceSize)
              continue;
          }
          req_instance = instance_info.instance;
          requested_instance = &req_instance;
        } else {
          requested_instance = &non_zero_instances.at((*distrib)(gen));
        }
      } else {
        break;
      }
    }

    LOG_IF(INFO, config_.verbosity >= 1)
        << "Max Overlap for Instance #" << max_overlap_idx << ". Overlap: #"
        << gt_overlap[max_overlap_idx] << "("
        << gt_instances[max_overlap_idx].size() /
               static_cast<float>(gt_overlap[max_overlap_idx])

        << "%) size " << gt_instances[max_overlap_idx].size() << std::endl;
    requested_instance = &gt_instances[max_overlap_idx];
  }
  response.Size = 0;
  response.Changed = 0;
  std::set<int> assigned_labels;
  std::cout << "Labeling #" << requested_instance->size() << " Voxels"
            << std::endl;
  int zero_voxels = 0;
  // Label the instance
  for (auto voxel_idx : *requested_instance) {
    const panoptic_mapping::ClassVoxelType *predicted_voxel =
        prediction_submap.getClassLayer().getVoxelPtrByGlobalIndex(voxel_idx);
    if (!predicted_voxel) {
//      std::cout << " Voxel not found in predicted map " << std::endl;
      // Voxel was zero
      zero_voxels++;
      continue;
    }

    if (predicted_voxel->is_groundtruth) {
      //     std::cout << "skipping gt voxel" << std::endl;
      continue; // Skip gt voxels
    }

    response.Size++;
    const panoptic_mapping::ClassVoxelType *gt_voxel =
        gt_submap.getClassLayer().getVoxelPtrByGlobalIndex(voxel_idx);

    if (!gt_voxel)
      continue;
    if (gt_voxel->current_index != predicted_voxel->current_index) {
      response.Changed++;
    }
    mapper_->labelClassVoxel(
        voxel_idx, mapper_->getSubmapCollection().getActiveFreeSpaceSubmapID(),
        gt_voxel->current_index);

    assigned_labels.insert(gt_voxel->current_index);
  }

  for (auto c : assigned_labels) {
    response.GTClasses.push_back(c);
  }
  return true;
}
} // namespace volumetric_labeling
