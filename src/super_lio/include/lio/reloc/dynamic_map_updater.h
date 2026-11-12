#ifndef SUPER_LIO_RELOC_DYNAMIC_MAP_UPDATER_H_
#define SUPER_LIO_RELOC_DYNAMIC_MAP_UPDATER_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pcl/registration/ndt.h>

#include "OctVoxMap/OctVoxMap.hpp"
#include "basic/Manifold.h"
#include "basic/alias.h"
#include "lio/reloc/fc_gate.h"
#include "lio/reloc/genz_refiner.h"
#include "lio/reloc/kiss_global_matcher.h"
#include "lio/reloc/reloc_options.h"

namespace LI2Sup {

class DynamicMapUpdater {
public:
  using OctVoxMapType = OctVoxMap<BASIC::V3, BASIC::scalar>;
  using FineKey = Eigen::Vector3i;
  using ExcludeFn = std::function<bool(const BASIC::V3&)>;

  struct FineKeyHash {
    std::size_t operator()(const FineKey& k) const noexcept {
      std::size_t h = static_cast<std::size_t>(static_cast<uint32_t>(k[0])) * 73856093u;
      h ^= static_cast<std::size_t>(static_cast<uint32_t>(k[1])) * 19349663u;
      h ^= static_cast<std::size_t>(static_cast<uint32_t>(k[2])) * 83492791u;
      return h;
    }
  };

  struct Grid2DHash {
    std::size_t operator()(const Eigen::Vector2i& k) const noexcept {
      std::size_t h = static_cast<std::size_t>(static_cast<uint32_t>(k[0])) * 73856093u;
      h ^= static_cast<std::size_t>(static_cast<uint32_t>(k[1])) * 19349663u;
      return h;
    }
  };

  struct ScanUpdateInput {
    double stamp = 0.0;
    BASIC::SE3 pose;                     // T_map_imu
    BASIC::V3 lidar_lever_arm = BASIC::V3::Zero();  // g_lidar_imu.t_
    BASIC::V3 velocity = BASIC::V3::Zero();
    double angular_speed = 0.0;          // rad/s
    bool frame_still = false;
    bool driverseat_collecting = false;
    std::size_t ds_size = 0;
    std::size_t matched = 0;
    std::size_t base_matched = 0;
    const BASIC::PointCloudType* scan_undistort_imu = nullptr;
    ExcludeFn in_vehicle_zone;
  };

  struct CommitResult {
    bool success = false;
    std::uint64_t generation = 0;
    BASIC::CloudPtr updated_map;
    pcl::NormalDistributionsTransform<BASIC::PointType, BASIC::PointType>::Ptr ndt;
    std::unique_ptr<GenzRefiner> genz;
    std::unique_ptr<FcGate> fc;
    std::unique_ptr<KissGlobalMatcher> kiss;
    bool pgm_updated = false;
    std::string map_yaml_path;
    std::size_t added_voxels = 0;
    std::size_t removed_voxels = 0;
  };

  DynamicMapUpdater(std::string map_dir, std::string map_name, const RelocOptions& options);
  ~DynamicMapUpdater();

  DynamicMapUpdater(const DynamicMapUpdater&) = delete;
  DynamicMapUpdater& operator=(const DynamicMapUpdater&) = delete;

  /// Initializes L0 immutable base map and builds the adaptive 2D ground Z grid.
  void setBaseMap(const BASIC::CloudPtr& base_map);

  /// Adaptive local ground height at (x, y) in map frame (supports negative / sloped ground).
  float estimateGroundZ(float x, float y) const;

  /// Populates `ivox` from L0 \ L1_removed (marked as base), L1_added, and transient L2 `extra`.
  void populateIndex(OctVoxMapType& ivox, const BASIC::VV3& transient_extra) const;

  /// Purges any unconfirmed candidates falling inside an active vehicle exclusion zone.
  void purgePendingInZone(const ExcludeFn& in_vehicle_zone);

  /// Clears unconfirmed candidates when tracking resets (LOST / manual re-init / standby).
  void onTrackingReset();

  /// Online per-scan entry: updates L1 + `ivox` on valid keyframes. Returns true if L1 changed.
  bool processScan(const ScanUpdateInput& input, OctVoxMapType& ivox);

  /// Schedules a background commit if dirty L1 changes exceed the large-update threshold.
  void maybeScheduleCommit(double now_sec, bool frame_still);

  /// Non-blocking poll for a completed background commit ready for O(1) main-thread swap.
  bool pollCommitResult(CommitResult& out);

  /// Synchronously persists uncommitted large changes (if threshold met) on standby / exit.
  bool flushSyncIfNeeded();

  /// Restores disk global.pcd + map.pgm from `<map_dir>/patch/*.origin.*` and resets L0/L1.
  /// Outputs `restored_map` (the clean origin L0 cloud) and human-readable `message`.
  bool resetToOrigin(BASIC::CloudPtr& restored_map, std::string& out_yaml_path,
                     bool& pgm_restored, std::string& message);

  /// Synthesizes visualization cloud of L1 patch (+1.0 intensity = added, -1.0 = carved).
  BASIC::CloudPtr buildPatchDebugCloud() const;

  std::size_t addedCount() const { return added_voxels_.size(); }
  std::size_t removedCount() const { return removed_subvoxels_.size(); }

private:
  struct GroundGrid {
    float resolution = 1.5f;
    float inv_resolution = 1.0f / 1.5f;
    float global_ground_z = 0.0f;
    std::unordered_map<Eigen::Vector2i, float, Grid2DHash> cells;

    float query(float x, float y) const;
  };

  struct AddedVoxel {
    BASIC::V3 centroid = BASIC::V3::Zero();
    BASIC::VV3 fine_points;  // 0.10 m surface samples (capped per 0.25 m sub-voxel)
  };

  struct PendingAdd {
    BASIC::V3 mean_pt = BASIC::V3::Zero();
    BASIC::VV3 fine_points;
    int hits = 0;
    int misses = 0;
    double first_stamp = 0.0;
    double last_stamp = 0.0;
  };

  struct CarveMiss {
    BASIC::V3 first_origin = BASIC::V3::Zero();
    float max_baseline = 0.0f;
    int count = 0;
    double last_stamp = 0.0;
  };

  struct CommitJob {
    std::uint64_t generation = 0;
    std::string map_dir;
    std::string map_name;
    RelocOptions options;
    BASIC::CloudPtr base_map;
    GroundGrid ground_grid;
    float sub_inv_resolution = 4.0f;
    std::unordered_map<FineKey, AddedVoxel, FineKeyHash> added_voxels;
    std::unordered_set<FineKey, FineKeyHash> removed_subvoxels;
    std::unordered_set<FineKey, FineKeyHash> newly_occupied;
    std::unordered_set<FineKey, FineKeyHash> newly_cleared;
    bool rebuild_reloc_targets = true;
  };

  void buildGroundGrid(const BASIC::PointCloudType& cloud);
  void computeDirtySets(std::unordered_set<FineKey, FineKeyHash>& newly_occupied,
                        std::unordered_set<FineKey, FineKeyHash>& newly_cleared) const;
  std::size_t computeMaxClusterSize(const std::unordered_set<FineKey, FineKeyHash>& a,
                                    const std::unordered_set<FineKey, FineKeyHash>& b) const;
  void cleanupStaleCandidates(double now_sec);
  void joinCommitThread();

  static CommitResult executeCommitJob(const CommitJob& job);
  static BASIC::CloudPtr synthesizePersistentMap(const CommitJob& job);
  static bool ensureOriginBackup(const std::string& map_dir, const std::string& map_name);
  static bool writePcdAtomic(const std::string& map_dir, const std::string& target_path,
                             const BASIC::PointCloudType& cloud);
  static bool syncPgmMap(const CommitJob& job, const BASIC::PointCloudType& updated_map,
                         std::string& out_yaml_path);

  std::string map_dir_;
  std::string map_name_;
  RelocOptions options_;
  float sub_resolution_ = 0.25f;
  float sub_inv_resolution_ = 4.0f;

  BASIC::CloudPtr base_map_;
  std::unordered_map<FineKey, BASIC::V3, FineKeyHash> l0_centroids_;
  GroundGrid ground_grid_;

  // L1 persistent patch state (mutated only on main tracking thread).
  std::unordered_map<FineKey, AddedVoxel, FineKeyHash> added_voxels_;
  std::unordered_set<FineKey, FineKeyHash> removed_subvoxels_;

  // Online multi-keyframe evidence accumulators.
  std::unordered_map<FineKey, PendingAdd, FineKeyHash> pending_add_;
  std::unordered_map<FineKey, CarveMiss, FineKeyHash> carve_miss_;
  bool has_last_kf_ = false;
  BASIC::SE3 last_kf_pose_;
  std::size_t kf_counter_ = 0;

  // Commit tracking & async worker state.
  std::unordered_set<FineKey, FineKeyHash> committed_added_keys_;
  std::unordered_set<FineKey, FineKeyHash> committed_removed_keys_;
  std::unordered_map<FineKey, AddedVoxel, FineKeyHash> inflight_added_snapshot_;
  std::unordered_set<FineKey, FineKeyHash> inflight_removed_snapshot_;
  double last_commit_time_ = -1e9;
  double dirty_since_time_ = -1.0;

  std::uint64_t generation_ = 0;
  std::atomic<bool> commit_running_{false};
  std::thread commit_thread_;
  std::mutex result_mutex_;
  std::unique_ptr<CommitResult> pending_result_;
};

}  // namespace LI2Sup

#endif  // SUPER_LIO_RELOC_DYNAMIC_MAP_UPDATER_H_
