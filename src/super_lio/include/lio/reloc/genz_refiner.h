#ifndef SUPER_LIO_GENZ_REFINER_H_
#define SUPER_LIO_GENZ_REFINER_H_

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "basic/alias.h"
#include "lio/reloc/reloc_options.h"

namespace genz_icp {
struct VoxelHashMap;
}

namespace LI2Sup {

/// GenZ-ICP (adaptive point-to-plane / point-to-point) refinement of a
/// candidate pose against a static voxel hash of the fixed map.
class GenzRefiner {
public:
  explicit GenzRefiner(const RelocOptions::Genz& options);
  ~GenzRefiner();

  bool SetTarget(const BASIC::CloudPtr& map);
  bool TargetReady() const;

  /// source is in the query frame; returns the refined T_map_query.
  /// Coarse then fine correspondence radius. false on degenerate input.
  bool Refine(const BASIC::CloudPtr& source, const Eigen::Matrix4d& guess,
              Eigen::Matrix4d& refined) const;

private:
  RelocOptions::Genz options_;
  std::unique_ptr<genz_icp::VoxelHashMap> map_;
};

}  // namespace LI2Sup

#endif
