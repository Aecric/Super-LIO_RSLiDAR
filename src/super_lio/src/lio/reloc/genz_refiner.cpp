#include "lio/reloc/genz_refiner.h"

#include <cmath>
#include <limits>

#include <glog/logging.h>
#include <pcl/filters/voxel_grid.h>

#include "genz_icp/Registration.hpp"
#include "genz_icp/VoxelHashMap.hpp"

namespace LI2Sup {
namespace {

std::vector<Eigen::Vector3d> ToVector(const BASIC::CloudPtr& cloud, double voxel) {
  BASIC::CloudPtr ds = cloud;
  if (voxel > 0) {
    const float leaf = static_cast<float>(voxel);
    pcl::VoxelGrid<BASIC::PointType> grid;
    grid.setLeafSize(leaf, leaf, leaf);
    grid.setInputCloud(cloud);
    ds.reset(new BASIC::PointCloudType());
    grid.filter(*ds);
  }
  std::vector<Eigen::Vector3d> points;
  points.reserve(ds->size());
  for (const auto& p : ds->points)
    if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
      points.emplace_back(p.x, p.y, p.z);
  return points;
}

Sophus::SE3d ToSophus(const Eigen::Matrix4d& T) {
  Eigen::Quaterniond q(T.block<3, 3>(0, 0));
  q.normalize();
  return Sophus::SE3d(q, T.block<3, 1>(0, 3));
}

}  // namespace

GenzRefiner::GenzRefiner(const RelocOptions::Genz& options) : options_(options) {}
GenzRefiner::~GenzRefiner() = default;

bool GenzRefiner::TargetReady() const { return map_ && !map_->Empty(); }

bool GenzRefiner::SetTarget(const BASIC::CloudPtr& map) {
  map_.reset();
  if (!map || map->empty()) return false;
  // The map is static: no range limit and no cleanup.
  map_ = std::make_unique<genz_icp::VoxelHashMap>(
      options_.map_voxel, std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
      options_.planarity_threshold, options_.max_points_per_voxel);
  map_->AddPoints(ToVector(map, 0.0));
  LOG(INFO) << "[GenZ] target ready: " << map_->map_.size() << " voxels @ "
            << options_.map_voxel << " m";
  return TargetReady();
}

bool GenzRefiner::Refine(const BASIC::CloudPtr& source, const Eigen::Matrix4d& guess,
                         Eigen::Matrix4d& refined) const {
  if (!TargetReady() || !source) return false;
  const auto points = ToVector(source, options_.source_voxel);
  if (points.size() < 100) {
    LOG(WARNING) << "[GenZ] source has only " << points.size() << " points";
    return false;
  }

  genz_icp::Registration registration(options_.max_iterations, options_.convergence);
  registration.SetTerminalStatusEnabled(false);
  Sophus::SE3d pose = ToSophus(guess);
  // Kernel = radius / 3, as in the GenZ-ICP odometry pipeline.
  for (const double radius : {options_.coarse_max_corr, options_.fine_max_corr}) {
    const auto [next, planar, non_planar] =
        registration.RegisterFrame(points, *map_, pose, radius, radius / 3.0);
    if (planar.size() + non_planar.size() < 50) {
      LOG(WARNING) << "[GenZ] only " << planar.size() + non_planar.size()
                   << " correspondences within " << radius << " m";
      return false;
    }
    pose = next;
  }
  refined = pose.matrix();
  return refined.allFinite();
}

}  // namespace LI2Sup
