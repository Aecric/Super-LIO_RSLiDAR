#include "lio/reloc/kiss_global_matcher.h"

#include <algorithm>
#include <cmath>
#include <exception>

#include <glog/logging.h>
#include <pcl/filters/voxel_grid.h>

#include "kiss_matcher/KISSMatcher.hpp"

namespace LI2Sup {

bool KissGlobalMatcher::SetTarget(const BASIC::CloudPtr& map) {
  target_.clear();
  if (!map || map->empty()) return false;

  BASIC::CloudPtr ds = map;
  if (options_.target_pre_voxel > 0) {
    const float leaf = static_cast<float>(options_.target_pre_voxel);
    pcl::VoxelGrid<BASIC::PointType> grid;
    grid.setLeafSize(leaf, leaf, leaf);
    grid.setInputCloud(map);
    ds.reset(new BASIC::PointCloudType());
    grid.filter(*ds);
  }
  if (options_.max_target_points > 0 && ds->size() > options_.max_target_points) {
    LOG(WARNING) << "[KISS] map has " << ds->size() << " points after "
                 << options_.target_pre_voxel << " m downsampling (max "
                 << options_.max_target_points << "); cold start disabled";
    return false;
  }

  target_.reserve(ds->size());
  for (const auto& p : ds->points)
    if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
      target_.emplace_back(p.x, p.y, p.z);
  if (target_.size() < 1000) {
    LOG(WARNING) << "[KISS] map has only " << target_.size() << " valid points; cold start disabled";
    target_.clear();
    return false;
  }
  LOG(INFO) << "[KISS] target ready: " << map->size() << " -> " << target_.size() << " points";
  return true;
}

std::vector<KissGlobalMatcher::Result> KissGlobalMatcher::Match(const BASIC::CloudPtr& query) const {
  std::vector<Result> results;
  if (target_.empty() || !query) return results;

  std::vector<Eigen::Vector3f> source;
  source.reserve(query->size());
  for (const auto& p : query->points)
    if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
      source.emplace_back(p.x, p.y, p.z);
  if (source.size() < 1000) {
    LOG(WARNING) << "[KISS] query has only " << source.size() << " valid points";
    return results;
  }
  for (const double voxel : options_.voxel_sizes) results.push_back(MatchOnce(source, voxel));
  return results;
}

KissGlobalMatcher::Result KissGlobalMatcher::MatchOnce(
    const std::vector<Eigen::Vector3f>& source, double voxel) const {
  Result result;
  kiss_matcher::KISSMatcherConfig config(static_cast<float>(voxel));
  config.use_quatro_ = options_.use_quatro;
  kiss_matcher::KISSMatcher matcher(config);
  kiss_matcher::RegistrationSolution solution;
  try {
    solution = matcher.estimate(source, target_);
  } catch (const std::exception& error) {
    LOG(ERROR) << "[KISS] voxel " << voxel << ": estimate threw: " << error.what();
    return result;
  }
  result.rotation_inliers = matcher.getNumRotationInliers();
  result.final_inliers = matcher.getNumFinalInliers();

  if (!solution.valid || result.rotation_inliers < options_.min_rotation_inliers ||
      result.final_inliers < options_.min_final_inliers) {
    LOG(INFO) << "[KISS] voxel " << voxel << ": no candidate (valid=" << solution.valid
              << ", rot_inliers=" << result.rotation_inliers
              << ", final_inliers=" << result.final_inliers << ")";
    return result;
  }

  // The solver rotation is not guaranteed to be orthonormal.
  Eigen::Quaterniond q(solution.rotation);
  q.normalize();
  const Eigen::Matrix3d R = q.toRotationMatrix();
  const double tilt_deg = std::acos(std::clamp(R(2, 2), -1.0, 1.0)) * 180.0 / M_PI;
  if (tilt_deg > options_.max_tilt_deg) {
    LOG(INFO) << "[KISS] voxel " << voxel << ": tilt " << tilt_deg << " deg > "
              << options_.max_tilt_deg << " deg for a gravity-levelled query";
    return result;
  }

  result.T_map_query.block<3, 3>(0, 0) = R;
  result.T_map_query.block<3, 1>(0, 3) = solution.translation;
  result.valid = true;
  LOG(INFO) << "[KISS] voxel " << voxel << ": candidate t=[" << solution.translation.transpose()
            << "], yaw=" << std::atan2(R(1, 0), R(0, 0)) * 180.0 / M_PI
            << " deg, rot_inliers=" << result.rotation_inliers
            << ", final_inliers=" << result.final_inliers;
  return result;
}

}  // namespace LI2Sup
