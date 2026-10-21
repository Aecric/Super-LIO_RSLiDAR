#ifndef SUPER_LIO_KISS_GLOBAL_MATCHER_H_
#define SUPER_LIO_KISS_GLOBAL_MATCHER_H_

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "basic/alias.h"
#include "lio/reloc/reloc_options.h"

namespace LI2Sup {

/// KISS-Matcher (MIT-SPARK) prior-free 6-DoF registration against the fixed map.
/// Ported from hikari_loclite reloc/kiss_global_matcher.
class KissGlobalMatcher {
public:
  struct Result {
    bool valid = false;
    Eigen::Matrix4d T_map_query = Eigen::Matrix4d::Identity();
    std::size_t rotation_inliers = 0;
    std::size_t final_inliers = 0;
  };

  explicit KissGlobalMatcher(const RelocOptions::Kiss& options) : options_(options) {}

  /// Downsample and cache the map. Returns false if the map is unusable.
  bool SetTarget(const BASIC::CloudPtr& map);
  bool TargetReady() const { return !target_.empty(); }

  /// query must be gravity levelled; one result per configured voxel size,
  /// valid ones are candidates for T_map_query.
  std::vector<Result> Match(const BASIC::CloudPtr& query) const;

private:
  Result MatchOnce(const std::vector<Eigen::Vector3f>& source, double voxel) const;

  RelocOptions::Kiss options_;
  std::vector<Eigen::Vector3f> target_;
};

}  // namespace LI2Sup

#endif
