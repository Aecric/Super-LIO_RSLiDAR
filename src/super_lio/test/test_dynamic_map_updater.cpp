#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <pcl/io/pcd_io.h>

#include "lio/params.h"
#include "lio/reloc/dynamic_map_updater.h"

namespace {

namespace fs = std::filesystem;
using BASIC::CloudPtr;
using BASIC::PointCloudType;
using BASIC::PointType;
using BASIC::SE3;
using BASIC::SO3;
using BASIC::V3;
using LI2Sup::DynamicMapUpdater;
using LI2Sup::RelocOptions;

class TempMapDir {
public:
  TempMapDir() {
    static int seq = 0;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    dir_ = fs::temp_directory_path() /
           ("super_lio_dmu_test_" + std::to_string(stamp) + "_" + std::to_string(++seq));
    fs::create_directories(dir_);
  }

  ~TempMapDir() {
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }

  const fs::path& path() const { return dir_; }

private:
  fs::path dir_;
};

CloudPtr makeRoomMap(bool with_removable_box = false) {
  CloudPtr cloud(new PointCloudType());
  // Flat floor at z = 0.0 and two orthogonal walls at x = 6.0 and y = 6.0.
  for (int ix = -12; ix <= 24; ++ix) {
    for (int iy = -12; iy <= 24; ++iy) {
      PointType p;
      p.x = ix * 0.25f;
      p.y = iy * 0.25f;
      p.z = 0.0f;
      p.intensity = 10.0f;
      cloud->points.push_back(p);
    }
  }
  for (int iy = -12; iy <= 24; ++iy) {
    for (int iz = 0; iz <= 10; ++iz) {
      PointType wx;
      wx.x = 6.0f;
      wx.y = iy * 0.25f;
      wx.z = iz * 0.25f;
      wx.intensity = 20.0f;
      cloud->points.push_back(wx);

      PointType wy;
      wy.x = iy * 0.25f;
      wy.y = 6.0f;
      wy.z = iz * 0.25f;
      wy.intensity = 20.0f;
      cloud->points.push_back(wy);
    }
  }
  if (with_removable_box) {
    // Sub-voxel (12, 0, 2) -> [3.0, 3.25) x [0.0, 0.25) x [0.5, 0.75), centroid (3.125, 0.125, 0.625).
    PointType b;
    b.x = 3.125f;
    b.y = 0.125f;
    b.z = 0.625f;
    b.intensity = 50.0f;
    cloud->points.push_back(b);
  }
  cloud->width = static_cast<uint32_t>(cloud->points.size());
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

RelocOptions makeFastTestOptions() {
  RelocOptions opts;
  opts.cold_start.enable = false;  // Keep unit tests fast without KISS-Matcher indexing.
  auto& m = opts.map_update;
  m.enable = true;
  m.kf_trans_thresh = 0.0;
  m.kf_rot_thresh_deg = 0.0;
  m.min_match_ratio = 0.5;
  m.min_matched_points = 30;
  m.min_base_ratio = 0.5;
  m.min_base_matched = 20;
  m.confirm_hits = 2;
  m.confirm_min_age_sec = 0.1;
  m.carve_confirm_misses = 2;
  m.carve_added_misses = 2;
  m.carve_min_baseline = 0.30;
  m.carve_ray_radius = 0.08;
  m.pending_ttl_sec = 10.0;
  m.min_changed_voxels = 1;
  m.min_cluster_voxels = 1;
  m.commit_cooldown_sec = 0.0;
  m.max_commit_defer_sec = 0.0;
  m.pgm.enable = true;
  m.pgm.create_if_missing = true;
  m.pgm.resolution = 0.25;
  return opts;
}

DynamicMapUpdater::ScanUpdateInput makeScanInput(double stamp, const V3& pos,
                                                 const PointCloudType* scan) {
  DynamicMapUpdater::ScanUpdateInput in;
  in.stamp = stamp;
  in.pose = SE3(SO3(), pos);
  in.ds_size = 100;
  in.matched = 90;
  in.base_matched = 80;
  in.scan_undistort_imu = scan;
  return in;
}

PointCloudType makeWorldPointScan(const V3& p_world, const V3& sensor_pos) {
  PointCloudType scan;
  PointType p;
  p.x = static_cast<float>(p_world.x() - sensor_pos.x());
  p.y = static_cast<float>(p_world.y() - sensor_pos.y());
  p.z = static_cast<float>(p_world.z() - sensor_pos.z());
  p.intensity = 10.0f;
  scan.points.push_back(p);
  scan.width = 1;
  scan.height = 1;
  scan.is_dense = true;
  return scan;
}

std::vector<uint8_t> readP5Pixels(const fs::path& pgm_path, int& width, int& height) {
  std::ifstream ifs(pgm_path, std::ios::binary);
  std::string magic;
  int maxval = 0;
  ifs >> magic >> width >> height >> maxval;
  char sep = 0;
  ifs.get(sep);
  std::vector<uint8_t> pixels(static_cast<std::size_t>(width * height));
  ifs.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
  return pixels;
}

class DynamicMapUpdaterTest : public ::testing::Test {
protected:
  void SetUp() override {
    LI2Sup::g_ivox_resolution = 0.5f;
  }
};

// P2 #1: Cleanup of temporary files must never mask rename/copy failures.
TEST_F(DynamicMapUpdaterTest, AtomicReplaceFailureIsNotMaskedByTempCleanup) {
  TempMapDir tmp;
  const CloudPtr base = makeRoomMap();
  const fs::path pcd_path = tmp.path() / "global.pcd";
  ASSERT_EQ(pcl::io::savePCDFileBinary(pcd_path.string(), *base), 0);

  RelocOptions opts = makeFastTestOptions();
  DynamicMapUpdater updater(tmp.path().string(), "global.pcd", opts);
  updater.setBaseMap(base);
  DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
  updater.populateIndex(ivox, {});

  const V3 obs_world(2.1, 0.1, 0.5);
  const V3 pos1(0.0, 0.0, 0.0), pos2(0.0, 0.3, 0.0);
  const auto s1 = makeWorldPointScan(obs_world, pos1);
  const auto s2 = makeWorldPointScan(obs_world, pos2);
  updater.processScan(makeScanInput(1.0, pos1, &s1), ivox);
  ASSERT_TRUE(updater.processScan(makeScanInput(2.0, pos2, &s2), ivox));

  // Commit #1 succeeds and creates patch/global.origin.pcd.
  DynamicMapUpdater::CommitResult commit;
  ASSERT_TRUE(updater.flushSyncIfNeeded(false, &commit));

  // Replace destination global.pcd with a directory so both rename and copy_file fail,
  // while removing the temporary file inside patch/ still succeeds.
  fs::remove(pcd_path);
  fs::create_directories(pcd_path);

  CloudPtr restored;
  std::string yaml_path;
  bool pgm_restored = false;
  std::string msg;
  EXPECT_FALSE(updater.resetToOrigin(restored, yaml_path, pgm_restored, msg));
}

// P2 #2: Existing map.yaml with nonzero origin yaw must project and carve in rotated grid coords.
TEST_F(DynamicMapUpdaterTest, OccupancySyncHonorsOriginYawRotation) {
  TempMapDir tmp;
  const CloudPtr base = makeRoomMap(/*with_removable_box=*/true);
  ASSERT_EQ(pcl::io::savePCDFileBinary((tmp.path() / "global.pcd").string(), *base), 0);

  // Create a 20x20 grid (res = 0.25 m) rotated by +90 deg around origin (1.0, -2.0).
  // World point (wx, wy) maps to:
  //   dx = wx - 1.0, dy = wy - (-2.0)
  //   u = +dy, v = -dx  (for yaw = +pi/2)
  const int W = 20, H = 20;
  {
    std::ofstream yofs(tmp.path() / "map.yaml", std::ios::trunc);
    yofs << "image: map.pgm\n"
         << "mode: trinary\n"
         << "resolution: 0.25\n"
         << "origin: [1.0, -2.0, 1.5707963267948966]\n"
         << "negate: 0\n"
         << "occupied_thresh: 0.65\n"
         << "free_thresh: 0.25\n";
  }
  {
    std::vector<uint8_t> init_px(W * H, 254);
    // Pre-mark the removable box at world (3.125, 0.125)? Wait: dx = +2.125 -> v = -2.125 (outside).
    // Let's also pre-mark a cell at rotated coords for an added obstacle at world (-0.375, 0.625):
    //   dx = -1.375, dy = +2.625 -> u = 2.625 (ix = 10), v = 1.375 (iy = 5, row = 14).
    std::ofstream pofs(tmp.path() / "map.pgm", std::ios::binary | std::ios::trunc);
    pofs << "P5\n" << W << " " << H << "\n255\n";
    pofs.write(reinterpret_cast<const char*>(init_px.data()), static_cast<std::streamsize>(init_px.size()));
  }

  RelocOptions opts = makeFastTestOptions();
  DynamicMapUpdater updater(tmp.path().string(), "global.pcd", opts);
  updater.setBaseMap(base);
  DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
  updater.populateIndex(ivox, {});

  // Add a static obstacle at world (-0.375, 0.625, 0.50).
  // Unrotated formula would compute ix = floor((-0.375 - 1.0)/0.25) = -6 (out of bounds!).
  // Rotated (+pi/2) formula computes u = 2.625 -> ix = 10, v = 1.375 -> iy = 5 (row = 19 - 5 = 14).
  const V3 obs_world(-0.375, 0.625, 0.50);
  const V3 pos1(-0.375, -1.5, 0.50), pos2(0.025, -1.5, 0.50);
  const auto add1 = makeWorldPointScan(obs_world, pos1);
  const auto add2 = makeWorldPointScan(obs_world, pos2);
  updater.processScan(makeScanInput(1.0, pos1, &add1), ivox);
  ASSERT_TRUE(updater.processScan(makeScanInput(2.0, pos2, &add2), ivox));

  DynamicMapUpdater::CommitResult commit1;
  ASSERT_TRUE(updater.flushSyncIfNeeded(false, &commit1));
  ASSERT_TRUE(commit1.pgm_updated);

  int out_w = 0, out_h = 0;
  std::vector<uint8_t> px1 = readP5Pixels(tmp.path() / "map.pgm", out_w, out_h);
  ASSERT_EQ(out_w, W);
  ASSERT_EQ(out_h, H);
  EXPECT_EQ(px1[(H - 1 - 5) * W + 10], 0u);

  // Carve that added obstacle with behind-hit rays passing dead-center through obs_world.
  const V3 far1 = pos1 + (obs_world - pos1) * 2.2;
  const V3 far2 = pos2 + (obs_world - pos2) * 2.2;
  const auto carve1 = makeWorldPointScan(far1, pos1);
  const auto carve2 = makeWorldPointScan(far2, pos2);
  updater.processScan(makeScanInput(3.0, pos1, &carve1), ivox);
  ASSERT_TRUE(updater.processScan(makeScanInput(4.0, pos2, &carve2), ivox));

  DynamicMapUpdater::CommitResult commit2;
  ASSERT_TRUE(updater.flushSyncIfNeeded(false, &commit2));
  ASSERT_TRUE(commit2.pgm_updated);

  std::vector<uint8_t> px2 = readP5Pixels(tmp.path() / "map.pgm", out_w, out_h);
  EXPECT_EQ(px2[(H - 1 - 5) * W + 10], 254u);
}

// P2 #3: Stale candidate evidence must expire immediately after pending_ttl_sec even between
// periodic 20-keyframe sweeps.
TEST_F(DynamicMapUpdaterTest, StalePendingEvidenceExpiresBeforeAcceptingNewObservation) {
  TempMapDir tmp;
  const CloudPtr base = makeRoomMap();
  RelocOptions opts = makeFastTestOptions();
  opts.map_update.confirm_hits = 3;
  opts.map_update.confirm_min_age_sec = 0.5;
  opts.map_update.pending_ttl_sec = 10.0;

  DynamicMapUpdater updater(tmp.path().string(), "global.pcd", opts);
  updater.setBaseMap(base);
  DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
  updater.populateIndex(ivox, {});

  const V3 obs_world(2.2, 0.2, 0.5);
  const V3 p1(0.0, 0.0, 0.0), p2(0.0, 0.3, 0.0), p3(0.0, 0.6, 0.0);
  const auto s1 = makeWorldPointScan(obs_world, p1);
  const auto s2 = makeWorldPointScan(obs_world, p2);
  const auto s3 = makeWorldPointScan(obs_world, p3);
  // Two hits before pause (confirm_hits = 3).
  EXPECT_FALSE(updater.processScan(makeScanInput(1.0, p1, &s1), ivox));
  EXPECT_FALSE(updater.processScan(makeScanInput(2.0, p2, &s2), ivox));
  EXPECT_EQ(updater.addedCount(), 0u);

  // Pause for 25 seconds (> pending_ttl_sec = 10s). On the 3rd keyframe (not divisible by 20),
  // stale hits must NOT be revived to reach confirm_hits = 3.
  EXPECT_FALSE(updater.processScan(makeScanInput(27.0, p3, &s3), ivox));
  EXPECT_EQ(updater.addedCount(), 0u);
}

// P2 #4: A corner-clipping ray that fails carve_ray_radius must not suppress a valid ray
// piercing the stored point in the same keyframe.
TEST_F(DynamicMapUpdaterTest, CarvingDeduplicatesOnlyAfterRayProximityCheck) {
  TempMapDir tmp;
  const CloudPtr base = makeRoomMap(/*with_removable_box=*/true);
  RelocOptions opts = makeFastTestOptions();
  opts.map_update.carve_confirm_misses = 2;
  opts.map_update.carve_min_baseline = 0.30;
  opts.map_update.carve_ray_radius = 0.06;  // Tight 6 cm radius around centroid (3.125, 0.125, 0.625).

  DynamicMapUpdater updater(tmp.path().string(), "global.pcd", opts);
  updater.setBaseMap(base);
  DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
  updater.populateIndex(ivox, {});

  // Build a scan from sensor at (0.0, y_s, 0.625) containing 16 corner-clipping rays that pass
  // through sub-voxel (12, 0, 2) near y = 0.24 / z = 0.74 (distance > 0.14 m > 0.06 m from
  // centroid) PLUS 1 valid ray passing straight through (3.125, 0.125, 0.625).
  auto build_mixed_scan = [](double y_sensor) {
    PointCloudType scan;
    for (int k = 0; k < 16; ++k) {
      // Ray from (0, y_sensor, 0.625) through (3.125, 0.245, 0.740) extended to x = 5.5 + 0.05*k.
      const double scale = (5.5 + 0.05 * k) / 3.125;
      PointType clip;
      clip.x = static_cast<float>(3.125 * scale);
      clip.y = static_cast<float>((0.245 - y_sensor) * scale);
      clip.z = static_cast<float>((0.740 - 0.625) * scale);
      scan.points.push_back(clip);
    }
    // Valid center ray through (3.125, 0.125, 0.625) hitting wall at x = 5.8.
    const double valid_scale = 5.8 / 3.125;
    PointType center;
    center.x = static_cast<float>(3.125 * valid_scale);
    center.y = static_cast<float>((0.125 - y_sensor) * valid_scale);
    center.z = 0.0f;
    scan.points.push_back(center);
    scan.width = static_cast<uint32_t>(scan.points.size());
    scan.height = 1;
    scan.is_dense = true;
    return scan;
  };

  const PointCloudType s1 = build_mixed_scan(0.0);
  const PointCloudType s2 = build_mixed_scan(0.40);
  EXPECT_FALSE(updater.processScan(makeScanInput(1.0, V3(0.0, 0.0, 0.625), &s1), ivox));
  EXPECT_TRUE(updater.processScan(makeScanInput(2.0, V3(0.0, 0.40, 0.625), &s2), ivox));
  EXPECT_EQ(updater.removedCount(), 1u);
}

// P2 #5: When factory map directory has no map.pgm/map.yaml, later commits must never back up
// generated occupancy files as factory originals, and resetToOrigin must remove generated files.
TEST_F(DynamicMapUpdaterTest, PreservesOriginalAbsenceOfOccupancyFilesAcrossMultipleCommits) {
  TempMapDir tmp;
  const CloudPtr base = makeRoomMap();
  const fs::path pcd_path = tmp.path() / "global.pcd";
  ASSERT_EQ(pcl::io::savePCDFileBinary(pcd_path.string(), *base), 0);

  RelocOptions opts = makeFastTestOptions();
  DynamicMapUpdater updater(tmp.path().string(), "global.pcd", opts);
  updater.setBaseMap(base);
  DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
  updater.populateIndex(ivox, {});

  // Commit #1 generates map.pgm and map.yaml.
  const V3 obs1_w(2.1, 0.1, 0.5);
  const V3 p1(0.0, 0.0, 0.0), p2(0.0, 0.3, 0.0);
  const auto s1 = makeWorldPointScan(obs1_w, p1);
  const auto s2 = makeWorldPointScan(obs1_w, p2);
  updater.processScan(makeScanInput(1.0, p1, &s1), ivox);
  ASSERT_TRUE(updater.processScan(makeScanInput(2.0, p2, &s2), ivox));
  ASSERT_TRUE(updater.flushSyncIfNeeded(false, nullptr));
  ASSERT_TRUE(fs::exists(tmp.path() / "map.pgm"));
  ASSERT_TRUE(fs::exists(tmp.path() / "map.yaml"));

  // Commit #2 runs when generated map.pgm and map.yaml already exist on disk.
  const V3 obs2_w(2.1, 1.1, 0.5);
  const V3 p3(0.0, 0.6, 0.0), p4(0.0, 0.9, 0.0);
  const auto s3 = makeWorldPointScan(obs2_w, p3);
  const auto s4 = makeWorldPointScan(obs2_w, p4);
  updater.processScan(makeScanInput(3.0, p3, &s3), ivox);
  ASSERT_TRUE(updater.processScan(makeScanInput(4.0, p4, &s4), ivox));
  ASSERT_TRUE(updater.flushSyncIfNeeded(false, nullptr));

  // Factory backup must still record that map.pgm/map.yaml were originally absent.
  EXPECT_FALSE(fs::exists(tmp.path() / "patch" / "map.origin.pgm"));
  EXPECT_FALSE(fs::exists(tmp.path() / "patch" / "map.origin.yaml"));
  EXPECT_TRUE(fs::exists(tmp.path() / "patch" / "map.origin.pgm.absent"));
  EXPECT_TRUE(fs::exists(tmp.path() / "patch" / "map.origin.yaml.absent"));

  // Reset must restore global.pcd and remove the generated map.pgm / map.yaml.
  CloudPtr restored;
  std::string yaml_path;
  bool pgm_restored = true;
  std::string msg;
  ASSERT_TRUE(updater.resetToOrigin(restored, yaml_path, pgm_restored, msg));
  EXPECT_FALSE(pgm_restored);
  EXPECT_TRUE(yaml_path.empty());
  EXPECT_FALSE(fs::exists(tmp.path() / "map.pgm"));
  EXPECT_FALSE(fs::exists(tmp.path() / "map.yaml"));
  ASSERT_NE(restored, nullptr);
  EXPECT_EQ(restored->size(), base->size());
}

// P2 #6: Standby flush must return a consumable CommitResult with rebuilt reloc targets,
// including when consuming a completed background commit.
TEST_F(DynamicMapUpdaterTest, StandbyFlushRebuildsTargetsAndPreservesCompletedBackgroundResult) {
  TempMapDir tmp;
  const CloudPtr base = makeRoomMap();
  ASSERT_EQ(pcl::io::savePCDFileBinary((tmp.path() / "global.pcd").string(), *base), 0);

  RelocOptions opts = makeFastTestOptions();
  DynamicMapUpdater updater(tmp.path().string(), "global.pcd", opts);
  updater.setBaseMap(base);
  DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
  updater.populateIndex(ivox, {});

  // Part A: Synchronous standby flush with uncommitted dirty voxels rebuilds reloc targets.
  const V3 obs1_w(2.1, 0.1, 0.5);
  const V3 p1(0.0, 0.0, 0.0), p2(0.0, 0.3, 0.0);
  const auto s1 = makeWorldPointScan(obs1_w, p1);
  const auto s2 = makeWorldPointScan(obs1_w, p2);
  updater.processScan(makeScanInput(1.0, p1, &s1), ivox);
  ASSERT_TRUE(updater.processScan(makeScanInput(2.0, p2, &s2), ivox));

  DynamicMapUpdater::CommitResult standby_res;
  ASSERT_TRUE(updater.flushSyncIfNeeded(/*rebuild_reloc_targets=*/true, &standby_res));
  EXPECT_TRUE(standby_res.success);
  EXPECT_NE(standby_res.updated_map, nullptr);
  EXPECT_NE(standby_res.ndt, nullptr);
  EXPECT_NE(standby_res.genz, nullptr);
  EXPECT_NE(standby_res.fc, nullptr);
  EXPECT_TRUE(standby_res.pgm_updated);

  // Part B: Launch a background commit and immediately enter standby before pollCommitResult() runs;
  // flushSyncIfNeeded must preserve and return that background CommitResult instead of dropping it.
  const V3 obs2_w(2.1, 1.1, 0.5);
  const V3 p3(0.0, 0.6, 0.0), p4(0.0, 0.9, 0.0);
  const auto s3 = makeWorldPointScan(obs2_w, p3);
  const auto s4 = makeWorldPointScan(obs2_w, p4);
  updater.processScan(makeScanInput(5.0, p3, &s3), ivox);
  ASSERT_TRUE(updater.processScan(makeScanInput(6.0, p4, &s4), ivox));
  updater.maybeScheduleCommit(10.0, /*frame_still=*/true);

  DynamicMapUpdater::CommitResult bg_standby_res;
  ASSERT_TRUE(updater.flushSyncIfNeeded(/*rebuild_reloc_targets=*/true, &bg_standby_res));
  EXPECT_TRUE(bg_standby_res.success);
  EXPECT_NE(bg_standby_res.updated_map, nullptr);
  EXPECT_NE(bg_standby_res.ndt, nullptr);
  EXPECT_NE(bg_standby_res.genz, nullptr);
  EXPECT_NE(bg_standby_res.fc, nullptr);
}

}  // namespace
