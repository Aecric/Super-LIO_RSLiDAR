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

// Round-2 P2 #1: processScan must remain paused when driverseat_collecting is true.
TEST_F(DynamicMapUpdaterTest, DriverseatCollectingGatesDynamicMapUpdates) {
  TempMapDir tmp;
  const CloudPtr base = makeRoomMap();
  RelocOptions opts = makeFastTestOptions();

  DynamicMapUpdater updater(tmp.path().string(), "global.pcd", opts);
  updater.setBaseMap(base);
  DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
  updater.populateIndex(ivox, {});

  const V3 obs_world(2.1, 0.1, 0.5);
  const V3 p1(0.0, 0.0, 0.0), p2(0.0, 0.3, 0.0);
  const auto s1 = makeWorldPointScan(obs_world, p1);
  const auto s2 = makeWorldPointScan(obs_world, p2);

  auto in1 = makeScanInput(1.0, p1, &s1);
  in1.driverseat_collecting = true;
  auto in2 = makeScanInput(2.0, p2, &s2);
  in2.driverseat_collecting = true;
  EXPECT_FALSE(updater.processScan(in1, ivox));
  EXPECT_FALSE(updater.processScan(in2, ivox));
  EXPECT_EQ(updater.addedCount(), 0u);
}

// Round-2 P2 #2: Failed PGM synchronization must retain dirty cells so a subsequent commit
// re-projects them into map.pgm instead of leaving Nav2 permanently out of sync with global.pcd.
TEST_F(DynamicMapUpdaterTest, RetainsDirtyCellsWhenOccupancySyncFailsAndRetriesLater) {
  TempMapDir tmp;
  const CloudPtr base = makeRoomMap();
  ASSERT_EQ(pcl::io::savePCDFileBinary((tmp.path() / "global.pcd").string(), *base), 0);

  const int W = 20, H = 20;
  {
    std::ofstream yofs(tmp.path() / "map.yaml", std::ios::trunc);
    yofs << "image: map.pgm\n"
         << "mode: trinary\n"
         << "resolution: 0.25\n"
         << "origin: [0.0, 0.0, 0.0]\n"
         << "negate: 0\n"
         << "occupied_thresh: 0.65\n"
         << "free_thresh: 0.25\n";
  }
  {
    std::vector<uint8_t> init_px(W * H, 254);
    std::ofstream pofs(tmp.path() / "map.pgm", std::ios::binary | std::ios::trunc);
    pofs << "P5\n" << W << " " << H << "\n255\n";
    pofs.write(reinterpret_cast<const char*>(init_px.data()),
               static_cast<std::streamsize>(init_px.size()));
  }

  // Block writePgmAtomic() by placing a directory at patch/.map.pgm.tmp.
  fs::create_directories(tmp.path() / "patch" / ".map.pgm.tmp");

  RelocOptions opts = makeFastTestOptions();
  DynamicMapUpdater updater(tmp.path().string(), "global.pcd", opts);
  updater.setBaseMap(base);
  DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
  updater.populateIndex(ivox, {});

  // Obstacle at (2.125, 0.625, 0.50) -> ix = 8, iy = 2 (row = 19 - 2 = 17).
  const V3 obs_w(2.125, 0.625, 0.50);
  const V3 p1(0.0, 0.0, 0.0), p2(0.0, 0.3, 0.0);
  const auto s1 = makeWorldPointScan(obs_w, p1);
  const auto s2 = makeWorldPointScan(obs_w, p2);
  updater.processScan(makeScanInput(1.0, p1, &s1), ivox);
  ASSERT_TRUE(updater.processScan(makeScanInput(2.0, p2, &s2), ivox));

  DynamicMapUpdater::CommitResult commit1;
  ASSERT_TRUE(updater.flushSyncIfNeeded(false, &commit1));
  EXPECT_TRUE(commit1.success);
  EXPECT_FALSE(commit1.pgm_updated);
  EXPECT_TRUE(commit1.pgm_sync_pending);

  int out_w = 0, out_h = 0;
  std::vector<uint8_t> px1 = readP5Pixels(tmp.path() / "map.pgm", out_w, out_h);
  EXPECT_EQ(px1[(H - 1 - 2) * W + 8], 254u);

  // Clear the transient I/O obstacle and flush with NO new scans: retained dirty cells must retry.
  fs::remove_all(tmp.path() / "patch" / ".map.pgm.tmp");

  DynamicMapUpdater::CommitResult commit2;
  ASSERT_TRUE(updater.flushSyncIfNeeded(false, &commit2));
  EXPECT_TRUE(commit2.success);
  EXPECT_TRUE(commit2.pgm_updated);
  EXPECT_FALSE(commit2.pgm_sync_pending);

  std::vector<uint8_t> px2 = readP5Pixels(tmp.path() / "map.pgm", out_w, out_h);
  EXPECT_EQ(px2[(H - 1 - 2) * W + 8], 0u);
}

// Round-2 P2 #3: A base sub-voxel holding both a locked floor point and an in-band obstacle
// point must remain protected from carving even if OctVox keeps the obstacle representative.
TEST_F(DynamicMapUpdaterTest, MixedFloorAndObstacleBaseSubVoxelIsProtectedFromCarving) {
  TempMapDir tmp;
  const CloudPtr base = makeRoomMap(/*with_removable_box=*/false);
  // Sub-voxel (12, 0, 0) spans [3.0, 3.25) x [0.0, 0.25) x [0.0, 0.25).
  // Insert an obstacle-height point BEFORE the floor point (> 0.10 m apart) so OctVox stores
  // z = 0.20 as the sub-voxel representative while the base cloud still has floor at z = 0.02.
  PointType obs_low;
  obs_low.x = 3.125f;
  obs_low.y = 0.125f;
  obs_low.z = 0.20f;
  obs_low.intensity = 30.0f;
  PointType floor_pt;
  floor_pt.x = 3.125f;
  floor_pt.y = 0.125f;
  floor_pt.z = 0.02f;
  floor_pt.intensity = 10.0f;
  // Move the mixed obstacle point to the front of the base cloud before floor points in (12, 0, 0).
  base->points.insert(base->points.begin(), obs_low);
  base->points.push_back(floor_pt);
  base->width = static_cast<uint32_t>(base->points.size());

  RelocOptions opts = makeFastTestOptions();
  opts.map_update.obs_z_min = 0.10;
  opts.map_update.carve_confirm_misses = 2;
  opts.map_update.carve_min_baseline = 0.30;
  opts.map_update.carve_ray_radius = 0.08;

  DynamicMapUpdater updater(tmp.path().string(), "global.pcd", opts);
  updater.setBaseMap(base);
  DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
  updater.populateIndex(ivox, {});

  // Confirm the stored representative in ivox is indeed the z=0.20 point.
  V3 rep;
  bool is_base = false;
  const auto mixed_fk = ivox.toFineKey(V3(3.125, 0.125, 0.20));
  ASSERT_TRUE(ivox.getSubVoxel(mixed_fk, rep, &is_base));
  ASSERT_TRUE(is_base);
  ASSERT_NEAR(rep.z(), 0.20, 0.05);

  // Shoot behind-hit rays dead-center through rep across 2 keyframes with baseline 0.40 m.
  const V3 s1_pos(0.0, 0.0, 0.20);
  const V3 s2_pos(0.0, 0.40, 0.20);
  const V3 far1 = s1_pos + (rep - s1_pos) * 1.85;
  const V3 far2 = s2_pos + (rep - s2_pos) * 1.85;
  const auto scan1 = makeWorldPointScan(far1, s1_pos);
  const auto scan2 = makeWorldPointScan(far2, s2_pos);

  EXPECT_FALSE(updater.processScan(makeScanInput(1.0, s1_pos, &scan1), ivox));
  EXPECT_FALSE(updater.processScan(makeScanInput(2.0, s2_pos, &scan2), ivox));
  EXPECT_EQ(updater.removedCount(), 0u);
  EXPECT_TRUE(ivox.getSubVoxel(mixed_fk, rep, &is_base));
}

// Round-3 P2 #1: Carving a previously persisted L1 addition after a PGM sync failure must still
// update global.pcd, live reloc targets, and map.pgm on the next commit/flush.
TEST_F(DynamicMapUpdaterTest, CarvingPersistedAdditionAfterPgmFailureUpdatesPcdAndOccupancy) {
  TempMapDir tmp;
  const CloudPtr base = makeRoomMap();
  const fs::path pcd_path = tmp.path() / "global.pcd";
  ASSERT_EQ(pcl::io::savePCDFileBinary(pcd_path.string(), *base), 0);

  const int W = 20, H = 20;
  {
    std::ofstream yofs(tmp.path() / "map.yaml", std::ios::trunc);
    yofs << "image: map.pgm\n"
         << "mode: trinary\n"
         << "resolution: 0.25\n"
         << "origin: [0.0, 0.0, 0.0]\n"
         << "negate: 0\n"
         << "occupied_thresh: 0.65\n"
         << "free_thresh: 0.25\n";
  }
  {
    std::vector<uint8_t> init_px(W * H, 254);
    std::ofstream pofs(tmp.path() / "map.pgm", std::ios::binary | std::ios::trunc);
    pofs << "P5\n" << W << " " << H << "\n255\n";
    pofs.write(reinterpret_cast<const char*>(init_px.data()),
               static_cast<std::streamsize>(init_px.size()));
  }

  fs::create_directories(tmp.path() / "patch" / ".map.pgm.tmp");

  RelocOptions opts = makeFastTestOptions();
  DynamicMapUpdater updater(tmp.path().string(), "global.pcd", opts);
  updater.setBaseMap(base);
  DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
  updater.populateIndex(ivox, {});

  // Add obstacle at (2.125, 0.625, 0.50); commit1 persists global.pcd (+1 pt) while PGM fails.
  const V3 obs_w(2.125, 0.625, 0.50);
  const V3 p1(0.0, 0.0, 0.0), p2(0.0, 0.3, 0.0);
  const auto s1 = makeWorldPointScan(obs_w, p1);
  const auto s2 = makeWorldPointScan(obs_w, p2);
  updater.processScan(makeScanInput(1.0, p1, &s1), ivox);
  ASSERT_TRUE(updater.processScan(makeScanInput(2.0, p2, &s2), ivox));

  DynamicMapUpdater::CommitResult c1;
  ASSERT_TRUE(updater.flushSyncIfNeeded(true, &c1));
  EXPECT_TRUE(c1.success);
  EXPECT_TRUE(c1.pgm_sync_pending);
  EXPECT_EQ(c1.updated_map->size(), base->size() + 1);

  // Carve that L1 obstacle BEFORE retrying PGM.
  const V3 cpos1(0.0, 0.625, 0.50), cpos2(0.0, 1.025, 0.50);
  const V3 far1 = cpos1 + (obs_w - cpos1) * 2.5;
  const V3 far2 = cpos2 + (obs_w - cpos2) * 2.5;
  const auto cs1 = makeWorldPointScan(far1, cpos1);
  const auto cs2 = makeWorldPointScan(far2, cpos2);
  updater.processScan(makeScanInput(3.0, cpos1, &cs1), ivox);
  ASSERT_TRUE(updater.processScan(makeScanInput(4.0, cpos2, &cs2), ivox));
  ASSERT_EQ(updater.addedCount(), 0u);

  // Unblock PGM and flush: departed obstacle must be removed from global.pcd AND map.pgm.
  fs::remove_all(tmp.path() / "patch" / ".map.pgm.tmp");
  DynamicMapUpdater::CommitResult c2;
  ASSERT_TRUE(updater.flushSyncIfNeeded(true, &c2));
  EXPECT_TRUE(c2.success);
  EXPECT_TRUE(c2.pgm_updated);
  EXPECT_FALSE(c2.pgm_sync_pending);
  ASSERT_NE(c2.updated_map, nullptr);
  EXPECT_EQ(c2.updated_map->size(), base->size());

  CloudPtr on_disk(new PointCloudType());
  ASSERT_EQ(pcl::io::loadPCDFile<PointType>(pcd_path.string(), *on_disk), 0);
  EXPECT_EQ(on_disk->size(), base->size());

  int out_w = 0, out_h = 0;
  std::vector<uint8_t> px = readP5Pixels(tmp.path() / "map.pgm", out_w, out_h);
  EXPECT_EQ(px[(H - 1 - 2) * W + 8], 254u);
}

// Round-3 P2 #2: Direct returns beyond max_add_range (inside max_carve_range) must still decay
// previously accumulated carve misses on that obstacle.
TEST_F(DynamicMapUpdaterTest, DirectHitBeyondAdditionRangeDecaysAccumulatedCarveEvidence) {
  TempMapDir tmp;
  const CloudPtr base = makeRoomMap(/*with_removable_box=*/false);
  // Place an L0 obstacle at (5.125, 0.125, 0.625): distance > max_add_range (4.0), < max_carve_range (6.0).
  PointType far_box;
  far_box.x = 5.125f;
  far_box.y = 0.125f;
  far_box.z = 0.625f;
  far_box.intensity = 50.0f;
  base->points.push_back(far_box);
  base->width = static_cast<uint32_t>(base->points.size());

  RelocOptions opts = makeFastTestOptions();
  opts.map_update.max_add_range = 4.0;
  opts.map_update.max_carve_range = 6.0;
  opts.map_update.carve_confirm_misses = 2;
  opts.map_update.carve_min_baseline = 0.30;
  opts.map_update.carve_ray_radius = 0.08;

  DynamicMapUpdater updater(tmp.path().string(), "global.pcd", opts);
  updater.setBaseMap(base);
  DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
  updater.populateIndex(ivox, {});

  const V3 obs_w(5.125, 0.125, 0.625);
  const V3 p1(0.0, 0.0, 0.625), p2(0.0, 0.20, 0.625), p3(0.0, 0.40, 0.625);
  const auto miss1 = makeWorldPointScan(p1 + (obs_w - p1) * 1.15, p1);
  const auto direct2 = makeWorldPointScan(obs_w, p2);
  const auto miss3 = makeWorldPointScan(p3 + (obs_w - p3) * 1.15, p3);

  // KF1 accumulates 1 miss; KF2 directly hits the obstacle at 5.125 m (> max_add_range = 4.0)
  // and must decay the miss count back to 0; KF3 therefore only reaches count=1 (< 2).
  EXPECT_FALSE(updater.processScan(makeScanInput(1.0, p1, &miss1), ivox));
  EXPECT_FALSE(updater.processScan(makeScanInput(2.0, p2, &direct2), ivox));
  EXPECT_FALSE(updater.processScan(makeScanInput(3.0, p3, &miss3), ivox));
  EXPECT_EQ(updater.removedCount(), 0u);
}

// Round-3 P2 #3: On coarse occupancy maps (pixel > sub-voxel), clearing one obstacle must
// reproject surviving obstacles sharing that pixel even outside the +-1 fine-grid neighborhood.
TEST_F(DynamicMapUpdaterTest, CoarseOccupancyMapReprojectsSurvivingObstacleSharingClearedPixel) {
  TempMapDir tmp;
  const CloudPtr base = makeRoomMap();
  ASSERT_EQ(pcl::io::savePCDFileBinary((tmp.path() / "global.pcd").string(), *base), 0);

  const int W = 10, H = 10;
  {
    std::ofstream yofs(tmp.path() / "map.yaml", std::ios::trunc);
    yofs << "image: map.pgm\n"
         << "mode: trinary\n"
         << "resolution: 1.0\n"
         << "origin: [0.0, 0.0, 0.0]\n"
         << "negate: 0\n"
         << "occupied_thresh: 0.65\n"
         << "free_thresh: 0.25\n";
  }
  {
    std::vector<uint8_t> init_px(W * H, 254);
    std::ofstream pofs(tmp.path() / "map.pgm", std::ios::binary | std::ios::trunc);
    pofs << "P5\n" << W << " " << H << "\n255\n";
    pofs.write(reinterpret_cast<const char*>(init_px.data()),
               static_cast<std::streamsize>(init_px.size()));
  }

  RelocOptions opts = makeFastTestOptions();
  DynamicMapUpdater updater(tmp.path().string(), "global.pcd", opts);
  updater.setBaseMap(base);
  DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
  updater.populateIndex(ivox, {});

  // Both obstacles share pixel (ix=2, iy=0) [2.0, 3.0) x [0.0, 1.0), but sit 3 sub-voxels apart:
  // A at (2.125, 0.125, 0.50) -> fine XY (8, 0)
  // B at (2.875, 0.875, 0.50) -> fine XY (11, 3)
  const V3 obs_a(2.125, 0.125, 0.50);
  const V3 obs_b(2.875, 0.875, 0.50);
  const V3 p1(0.0, 0.0, 0.50), p2(0.0, 0.3, 0.50);
  auto make_ab = [&](const V3& s_pos) {
    PointCloudType sc;
    for (const auto& pw : {obs_a, obs_b}) {
      PointType p;
      p.x = static_cast<float>(pw.x() - s_pos.x());
      p.y = static_cast<float>(pw.y() - s_pos.y());
      p.z = static_cast<float>(pw.z() - s_pos.z());
      p.intensity = 10.0f;
      sc.points.push_back(p);
    }
    sc.width = 2;
    sc.height = 1;
    sc.is_dense = true;
    return sc;
  };
  const auto sc1 = make_ab(p1);
  const auto sc2 = make_ab(p2);
  updater.processScan(makeScanInput(1.0, p1, &sc1), ivox);
  ASSERT_TRUE(updater.processScan(makeScanInput(2.0, p2, &sc2), ivox));

  DynamicMapUpdater::CommitResult c1;
  ASSERT_TRUE(updater.flushSyncIfNeeded(false, &c1));
  ASSERT_TRUE(c1.pgm_updated);

  // Carve only Obstacle A while Obstacle B remains in pixel (2, 0).
  const V3 cpos1(0.0, 0.125, 0.50), cpos2(0.0, 0.425, 0.50);
  const auto cs1 = makeWorldPointScan(cpos1 + (obs_a - cpos1) * 2.2, cpos1);
  const auto cs2 = makeWorldPointScan(cpos2 + (obs_a - cpos2) * 2.2, cpos2);
  updater.processScan(makeScanInput(3.0, cpos1, &cs1), ivox);
  ASSERT_TRUE(updater.processScan(makeScanInput(4.0, cpos2, &cs2), ivox));

  DynamicMapUpdater::CommitResult c2;
  ASSERT_TRUE(updater.flushSyncIfNeeded(false, &c2));
  ASSERT_TRUE(c2.pgm_updated);

  int out_w = 0, out_h = 0;
  std::vector<uint8_t> px = readP5Pixels(tmp.path() / "map.pgm", out_w, out_h);
  ASSERT_EQ(out_w, W);
  ASSERT_EQ(out_h, H);
  // Pixel (ix=2, iy=0) must remain occupied (0) because Obstacle B still occupies that pixel.
  EXPECT_EQ(px[(H - 1 - 0) * W + 2], 0u);
}

// Round-4 P2 #1: An occupancy-only retry must report reloc_targets_rebuilt == false so the
// caller preserves existing relocalization targets (including KISS) instead of clearing them.
TEST_F(DynamicMapUpdaterTest, OccupancyOnlyRetryReportsRelocTargetsNotRebuilt) {
  TempMapDir tmp;
  const CloudPtr base = makeRoomMap();
  ASSERT_EQ(pcl::io::savePCDFileBinary((tmp.path() / "global.pcd").string(), *base), 0);

  const int W = 20, H = 20;
  {
    std::ofstream yofs(tmp.path() / "map.yaml", std::ios::trunc);
    yofs << "image: map.pgm\n"
         << "mode: trinary\n"
         << "resolution: 0.25\n"
         << "origin: [0.0, 0.0, 0.0]\n"
         << "negate: 0\n"
         << "occupied_thresh: 0.65\n"
         << "free_thresh: 0.25\n";
  }
  {
    std::vector<uint8_t> init_px(W * H, 254);
    std::ofstream pofs(tmp.path() / "map.pgm", std::ios::binary | std::ios::trunc);
    pofs << "P5\n" << W << " " << H << "\n255\n";
    pofs.write(reinterpret_cast<const char*>(init_px.data()),
               static_cast<std::streamsize>(init_px.size()));
  }

  fs::create_directories(tmp.path() / "patch" / ".map.pgm.tmp");

  RelocOptions opts = makeFastTestOptions();
  DynamicMapUpdater updater(tmp.path().string(), "global.pcd", opts);
  updater.setBaseMap(base);
  DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
  updater.populateIndex(ivox, {});

  const V3 obs_w(2.125, 0.625, 0.50);
  const V3 p1(0.0, 0.0, 0.0), p2(0.0, 0.3, 0.0);
  const auto s1 = makeWorldPointScan(obs_w, p1);
  const auto s2 = makeWorldPointScan(obs_w, p2);
  updater.processScan(makeScanInput(1.0, p1, &s1), ivox);
  ASSERT_TRUE(updater.processScan(makeScanInput(2.0, p2, &s2), ivox));

  DynamicMapUpdater::CommitResult c1;
  ASSERT_TRUE(updater.flushSyncIfNeeded(true, &c1));
  EXPECT_TRUE(c1.success);
  EXPECT_TRUE(c1.reloc_targets_rebuilt);
  EXPECT_TRUE(c1.pgm_sync_pending);

  fs::remove_all(tmp.path() / "patch" / ".map.pgm.tmp");
  DynamicMapUpdater::CommitResult c2;
  ASSERT_TRUE(updater.flushSyncIfNeeded(true, &c2));
  EXPECT_TRUE(c2.success);
  EXPECT_FALSE(c2.reloc_targets_rebuilt);
  EXPECT_TRUE(c2.pgm_updated);
  EXPECT_FALSE(c2.pgm_sync_pending);
}

// Round-4 P2 #2: When generating occupancy files from scratch, a failed map.yaml write must roll
// back the newly created map.pgm so subsequent retries do not hit has_pgm != has_yaml -> kSkipped.
TEST_F(DynamicMapUpdaterTest, RollsBackGeneratedPgmWhenYamlWriteFailsAndRetriesCleanly) {
  TempMapDir tmp;
  const CloudPtr base = makeRoomMap();
  ASSERT_EQ(pcl::io::savePCDFileBinary((tmp.path() / "global.pcd").string(), *base), 0);

  // Block writeMapYamlAtomic() by pre-creating patch/.map.yaml.tmp as a directory.
  fs::create_directories(tmp.path() / "patch" / ".map.yaml.tmp");

  RelocOptions opts = makeFastTestOptions();
  DynamicMapUpdater updater(tmp.path().string(), "global.pcd", opts);
  updater.setBaseMap(base);
  DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
  updater.populateIndex(ivox, {});

  const V3 obs_w(2.125, 0.625, 0.50);
  const V3 p1(0.0, 0.0, 0.0), p2(0.0, 0.3, 0.0);
  const auto s1 = makeWorldPointScan(obs_w, p1);
  const auto s2 = makeWorldPointScan(obs_w, p2);
  updater.processScan(makeScanInput(1.0, p1, &s1), ivox);
  ASSERT_TRUE(updater.processScan(makeScanInput(2.0, p2, &s2), ivox));

  DynamicMapUpdater::CommitResult c1;
  ASSERT_TRUE(updater.flushSyncIfNeeded(false, &c1));
  EXPECT_TRUE(c1.success);
  EXPECT_FALSE(c1.pgm_updated);
  EXPECT_TRUE(c1.pgm_sync_pending);
  EXPECT_FALSE(fs::exists(tmp.path() / "map.pgm"));
  EXPECT_FALSE(fs::exists(tmp.path() / "map.yaml"));

  // Unblock YAML write and flush with no new scans: both files must be generated cleanly.
  fs::remove_all(tmp.path() / "patch" / ".map.yaml.tmp");
  DynamicMapUpdater::CommitResult c2;
  ASSERT_TRUE(updater.flushSyncIfNeeded(false, &c2));
  EXPECT_TRUE(c2.success);
  EXPECT_TRUE(c2.pgm_updated);
  EXPECT_FALSE(c2.pgm_sync_pending);
  EXPECT_TRUE(fs::exists(tmp.path() / "map.pgm"));
  EXPECT_TRUE(fs::exists(tmp.path() / "map.yaml"));
}

TEST_F(DynamicMapUpdaterTest, StandbyOccupancyRetryPreservesBackgroundRebuildFlag) {
  TempMapDir tmp;
  const CloudPtr base = makeRoomMap();
  ASSERT_EQ(pcl::io::savePCDFileBinary((tmp.path() / "global.pcd").string(), *base), 0);
  fs::create_directories(tmp.path() / "patch" / ".map.yaml.tmp");

  DynamicMapUpdater updater(tmp.path().string(), "global.pcd", makeFastTestOptions());
  updater.setBaseMap(base);
  DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
  updater.populateIndex(ivox, {});
  const V3 obstacle(2.125, 0.625, 0.50);
  const V3 p1(0.0, 0.0, 0.0), p2(0.0, 0.3, 0.0);
  const auto s1 = makeWorldPointScan(obstacle, p1);
  const auto s2 = makeWorldPointScan(obstacle, p2);
  updater.processScan(makeScanInput(1.0, p1, &s1), ivox);
  ASSERT_TRUE(updater.processScan(makeScanInput(2.0, p2, &s2), ivox));
  updater.maybeScheduleCommit(3.0, true);

  // Standby consumes the background PCD result and immediately retries occupancy sync.
  DynamicMapUpdater::CommitResult standby;
  ASSERT_TRUE(updater.flushSyncIfNeeded(true, &standby));
  ASSERT_TRUE(standby.success);
  EXPECT_TRUE(standby.pgm_sync_pending);
  EXPECT_TRUE(standby.reloc_targets_rebuilt);
  EXPECT_NE(standby.ndt, nullptr);
  EXPECT_NE(standby.genz, nullptr);
  EXPECT_NE(standby.fc, nullptr);

  fs::remove_all(tmp.path() / "patch" / ".map.yaml.tmp");
  DynamicMapUpdater::CommitResult retry;
  ASSERT_TRUE(updater.flushSyncIfNeeded(true, &retry));
  EXPECT_TRUE(retry.success);
  EXPECT_TRUE(retry.pgm_updated);
  EXPECT_FALSE(retry.pgm_sync_pending);
  EXPECT_FALSE(retry.reloc_targets_rebuilt);
}

TEST_F(DynamicMapUpdaterTest, RegeneratesInterruptedOccupancyPairFromAbsentOrigins) {
  for (const bool leftover_pgm : {true, false}) {
    SCOPED_TRACE(leftover_pgm ? "orphan PGM" : "orphan YAML");
    TempMapDir tmp;
    const CloudPtr base = makeRoomMap();
    ASSERT_EQ(pcl::io::savePCDFileBinary((tmp.path() / "global.pcd").string(), *base), 0);
    fs::create_directories(tmp.path() / "patch" / ".map.yaml.tmp");

    DynamicMapUpdater updater(tmp.path().string(), "global.pcd", makeFastTestOptions());
    updater.setBaseMap(base);
    DynamicMapUpdater::OctVoxMapType ivox({0.5f, 100000});
    updater.populateIndex(ivox, {});
    const V3 obstacle(2.125, 0.625, 0.50);
    const V3 p1(0.0, 0.0, 0.0), p2(0.0, 0.3, 0.0);
    const auto s1 = makeWorldPointScan(obstacle, p1);
    const auto s2 = makeWorldPointScan(obstacle, p2);
    updater.processScan(makeScanInput(1.0, p1, &s1), ivox);
    ASSERT_TRUE(updater.processScan(makeScanInput(2.0, p2, &s2), ivox));
    DynamicMapUpdater::CommitResult first;
    ASSERT_TRUE(updater.flushSyncIfNeeded(false, &first));
    ASSERT_TRUE(first.success);
    ASSERT_TRUE(first.pgm_sync_pending);
    fs::remove_all(tmp.path() / "patch" / ".map.yaml.tmp");

    // Simulate interrupted generation (or failed rollback) leaving exactly one file.
    {
      std::ofstream orphan(tmp.path() / (leftover_pgm ? "map.pgm" : "map.yaml"));
      orphan << "incomplete generation\n";
    }
    const fs::path sentinel = tmp.path() / "patch" / "map.origin.yaml.absent";
    fs::remove(sentinel);
    fs::create_symlink(sentinel.filename(), sentinel);  // exists() returns ELOOP.
    DynamicMapUpdater::CommitResult failed;
    ASSERT_TRUE(updater.flushSyncIfNeeded(false, &failed));
    EXPECT_TRUE(failed.success);
    EXPECT_FALSE(failed.pgm_updated);
    EXPECT_TRUE(failed.pgm_sync_pending);

    fs::remove(sentinel);
    { std::ofstream marker(sentinel); marker << "absent\n"; }
    DynamicMapUpdater::CommitResult retry;
    ASSERT_TRUE(updater.flushSyncIfNeeded(false, &retry));
    EXPECT_TRUE(retry.success);
    EXPECT_TRUE(retry.pgm_updated);
    EXPECT_FALSE(retry.pgm_sync_pending);
    ASSERT_TRUE(fs::exists(tmp.path() / "map.yaml"));
    int width = 0, height = 0;
    const auto pixels = readP5Pixels(tmp.path() / "map.pgm", width, height);
    ASSERT_GT(width, 0);
    ASSERT_GT(height, 0);
    const int ix = static_cast<int>(std::floor((obstacle.x() + 4.0) / 0.25));
    const int iy = static_cast<int>(std::floor((obstacle.y() + 4.0) / 0.25));
    ASSERT_LT(ix, width);
    ASSERT_LT(iy, height);
    EXPECT_EQ(pixels[(height - 1 - iy) * width + ix], 0u);
  }
}

}  // namespace
