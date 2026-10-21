#include "lio/reloc/fc_gate.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <vector>

#include <Eigen/Geometry>
#include <glog/logging.h>
#include <pcl/filters/voxel_grid.h>

namespace LI2Sup {

std::string FcGate::Report::Describe() const {
  std::ostringstream out;
  out << "inlier=" << inlier_ratio << ", wall_inlier=" << wall_inlier_ratio
      << ", coverage=" << coverage << ", rmse=" << rmse << " m, dt=" << delta_trans
      << " m, dr=" << delta_rot_deg << " deg";
  return out.str();
}

bool FcGate::SetTarget(const BASIC::CloudPtr& map) {
  ready_ = false;
  if (!map || map->empty()) return false;
  BASIC::CloudPtr ds(new BASIC::PointCloudType());
  const float leaf = static_cast<float>(options_.map_voxel);
  pcl::VoxelGrid<BASIC::PointType> grid;
  grid.setLeafSize(leaf, leaf, leaf);
  grid.setInputCloud(map);
  grid.filter(*ds);
  if (ds->empty()) return false;
  tree_.setInputCloud(ds);
  ready_ = true;
  LOG(INFO) << "[FC] target ready: " << ds->size() << " points @ " << options_.map_voxel << " m";
  return true;
}

FcGate::Report FcGate::Check(const BASIC::CloudPtr& source, const Eigen::Matrix4d& guess,
                             const Eigen::Matrix4d& refined, double max_delta_trans,
                             double max_delta_rot_deg) const {
  Report report;
  if (!ready_ || !source || source->empty() || !refined.allFinite()) return report;

  const Eigen::Matrix3d R = refined.block<3, 3>(0, 0);
  const Eigen::Vector3d t = refined.block<3, 1>(0, 3);
  report.delta_trans = (t - guess.block<3, 1>(0, 3)).norm();
  const Eigen::AngleAxisd delta_rot(guess.block<3, 3>(0, 0).transpose() * R);
  report.delta_rot_deg = std::abs(delta_rot.angle()) * 180.0 / M_PI;

  std::vector<int> index(1);
  std::vector<float> dist2(1);
  const double inlier2 = options_.inlier_dist * options_.inlier_dist;
  std::size_t total = 0, inliers = 0, wall_total = 0, wall_inliers = 0;
  double inlier_sq = 0;
  std::array<int, 8> sector_points{}, sector_inliers{};
  for (const auto& p : source->points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
    const Eigen::Vector3d q = R * Eigen::Vector3d(p.x, p.y, p.z) + t;
    BASIC::PointType query;
    query.x = static_cast<float>(q.x());
    query.y = static_cast<float>(q.y());
    query.z = static_cast<float>(q.z());
    ++total;
    const bool hit = tree_.nearestKSearch(query, 1, index, dist2) > 0 && dist2[0] <= inlier2;
    if (hit) {
      ++inliers;
      inlier_sq += dist2[0];
    }
    const double dz = q.z() - t.z();
    if (dz < options_.wall_band_min || dz > options_.wall_band_max) continue;
    ++wall_total;
    const double bearing = std::atan2(q.y() - t.y(), q.x() - t.x()) + M_PI;
    const int sector = std::min(7, static_cast<int>(bearing / (M_PI / 4)));
    ++sector_points[sector];
    if (hit) {
      ++wall_inliers;
      ++sector_inliers[sector];
    }
  }
  if (total == 0) return report;

  report.inlier_ratio = static_cast<double>(inliers) / total;
  report.wall_inlier_ratio = wall_total ? static_cast<double>(wall_inliers) / wall_total : 0.0;
  report.rmse = inliers ? std::sqrt(inlier_sq / inliers) : 0.0;
  int observed = 0, covered = 0;
  for (int i = 0; i < 8; ++i) {
    if (sector_points[i] < options_.min_sector_points) continue;
    ++observed;
    if (sector_inliers[i] >= options_.min_sector_points) ++covered;
  }
  report.coverage = observed ? static_cast<double>(covered) / observed : 0.0;

  report.pass = report.inlier_ratio >= options_.min_inlier_ratio &&
                report.wall_inlier_ratio >= options_.min_wall_inlier_ratio &&
                report.coverage >= options_.min_coverage &&
                inliers > 0 && report.rmse <= options_.max_rmse &&
                report.delta_trans <= max_delta_trans && report.delta_rot_deg <= max_delta_rot_deg;
  return report;
}

}  // namespace LI2Sup
