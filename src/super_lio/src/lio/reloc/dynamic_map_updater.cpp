#include "lio/reloc/dynamic_map_updater.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <queue>
#include <sstream>

#include <glog/logging.h>
#include <pcl/common/io.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>

#include "lio/params.h"

namespace LI2Sup {

namespace fs = std::filesystem;
using BASIC::CloudPtr;
using BASIC::PointCloudType;
using BASIC::PointType;
using BASIC::SE3;
using BASIC::V3;
using BASIC::VV3;
using BASIC::scalar;

namespace {

constexpr std::size_t kMaxFineSamplesPerVoxel = 8;
constexpr float kFineSampleRes = 0.10f;
constexpr float kFineSampleInvRes = 1.0f / kFineSampleRes;

fs::path patchDirPath(const std::string& map_dir) {
  return fs::path(map_dir) / "patch";
}

fs::path originPcdPath(const std::string& map_dir, const std::string& map_name) {
  const fs::path p(map_name);
  const std::string stem = p.stem().string().empty() ? "global" : p.stem().string();
  const std::string ext = p.extension().string().empty() ? ".pcd" : p.extension().string();
  return patchDirPath(map_dir) / (stem + ".origin" + ext);
}

bool copyFileAtomic(const fs::path& src, const fs::path& dst, const fs::path& tmp_dir) {
  std::error_code ec;
  if (!fs::exists(src, ec)) return false;
  fs::create_directories(tmp_dir, ec);
  if (ec) return false;
  const fs::path tmp = tmp_dir / ("." + dst.filename().string() + ".tmp");
  fs::copy_file(src, tmp, fs::copy_options::overwrite_existing, ec);
  if (ec) {
    fs::remove(tmp, ec);
    return false;
  }
  fs::rename(tmp, dst, ec);
  if (ec) {
    fs::copy_file(tmp, dst, fs::copy_options::overwrite_existing, ec);
    fs::remove(tmp, ec);
    if (ec) return false;
  }
  return true;
}

void mergeFineSample(VV3& samples, const V3& pt) {
  const Eigen::Vector3i qk = (pt * kFineSampleInvRes).array().floor().cast<int>();
  for (auto& existing : samples) {
    const Eigen::Vector3i ek = (existing * kFineSampleInvRes).array().floor().cast<int>();
    if (ek == qk) {
      existing = scalar(0.5) * (existing + pt);
      return;
    }
  }
  if (samples.size() < kMaxFineSamplesPerVoxel) {
    samples.push_back(pt);
  }
}

std::string trimString(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return {};
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

struct MapYamlMeta {
  double resolution = 0.05;
  double origin_x = 0.0;
  double origin_y = 0.0;
  int negate = 0;
  bool valid = false;
};

bool parseMapYaml(const fs::path& yaml_path, MapYamlMeta& meta) {
  std::ifstream ifs(yaml_path);
  if (!ifs.is_open()) return false;

  bool has_res = false;
  bool in_origin_list = false;
  std::vector<double> origin_vals;

  std::string raw_line;
  while (std::getline(ifs, raw_line)) {
    const auto hash_pos = raw_line.find('#');
    const std::string line = trimString(
        hash_pos == std::string::npos ? raw_line : raw_line.substr(0, hash_pos));
    if (line.empty()) continue;

    if (in_origin_list) {
      if (line[0] == '-') {
        const std::string num_str = trimString(line.substr(1));
        try {
          origin_vals.push_back(std::stod(num_str));
        } catch (...) {
          return false;
        }
        if (origin_vals.size() >= 3) in_origin_list = false;
        continue;
      }
      in_origin_list = false;
    }

    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = trimString(line.substr(0, colon));
    const std::string val = trimString(line.substr(colon + 1));

    if (key == "resolution") {
      try {
        meta.resolution = std::stod(val);
        has_res = meta.resolution > 1e-6;
      } catch (...) {
        return false;
      }
    } else if (key == "negate") {
      try {
        meta.negate = std::stoi(val);
      } catch (...) {
      }
    } else if (key == "origin") {
      origin_vals.clear();
      if (val.empty()) {
        in_origin_list = true;
      } else if (val.front() == '[') {
        const auto close = val.find(']');
        const std::string inner = val.substr(1, close == std::string::npos ? std::string::npos : close - 1);
        std::stringstream ss(inner);
        std::string item;
        while (std::getline(ss, item, ',')) {
          const std::string t = trimString(item);
          if (!t.empty()) {
            try {
              origin_vals.push_back(std::stod(t));
            } catch (...) {
              return false;
            }
          }
        }
      }
    }
  }

  if (!has_res || origin_vals.size() < 2) return false;
  meta.origin_x = origin_vals[0];
  meta.origin_y = origin_vals[1];
  meta.valid = true;
  return true;
}

bool readNextPgmToken(std::istream& is, std::string& token) {
  token.clear();
  char ch = 0;
  while (is.get(ch)) {
    if (ch == '#') {
      std::string dummy;
      std::getline(is, dummy);
      continue;
    }
    if (!std::isspace(static_cast<unsigned char>(ch))) {
      token.push_back(ch);
      break;
    }
  }
  if (token.empty()) return false;
  while (is.peek() != EOF) {
    const int next = is.peek();
    if (next == '#' || std::isspace(static_cast<unsigned char>(next))) break;
    token.push_back(static_cast<char>(is.get()));
  }
  return true;
}

struct PgmImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;
};

bool readPgmFile(const fs::path& pgm_path, PgmImage& img) {
  std::ifstream ifs(pgm_path, std::ios::binary);
  if (!ifs.is_open()) return false;

  std::string magic, w_str, h_str, max_str;
  if (!readNextPgmToken(ifs, magic) || !readNextPgmToken(ifs, w_str) ||
      !readNextPgmToken(ifs, h_str) || !readNextPgmToken(ifs, max_str)) {
    return false;
  }
  int w = 0, h = 0, maxval = 0;
  try {
    w = std::stoi(w_str);
    h = std::stoi(h_str);
    maxval = std::stoi(max_str);
  } catch (...) {
    return false;
  }
  if (w <= 0 || h <= 0 || maxval <= 0 || maxval > 255) return false;

  const std::size_t total = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  img.width = w;
  img.height = h;
  img.pixels.resize(total);

  if (magic == "P5") {
    // Exactly one whitespace byte separates maxval from raw binary raster.
    char sep = 0;
    if (!ifs.get(sep)) return false;
    ifs.read(reinterpret_cast<char*>(img.pixels.data()), static_cast<std::streamsize>(total));
    return static_cast<std::size_t>(ifs.gcount()) == total;
  }
  if (magic == "P2") {
    std::string tok;
    for (std::size_t i = 0; i < total; ++i) {
      if (!readNextPgmToken(ifs, tok)) return false;
      img.pixels[i] = static_cast<uint8_t>(std::clamp(std::stoi(tok), 0, 255));
    }
    return true;
  }
  return false;
}

bool writePgmAtomic(const fs::path& tmp_dir, const fs::path& dst_path, const PgmImage& img) {
  std::error_code ec;
  fs::create_directories(tmp_dir, ec);
  if (ec) return false;
  const fs::path tmp = tmp_dir / ".map.pgm.tmp";
  {
    std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) return false;
    ofs << "P5\n" << img.width << " " << img.height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(img.pixels.data()),
              static_cast<std::streamsize>(img.pixels.size()));
    ofs.flush();
    if (!ofs.good()) return false;
  }
  fs::rename(tmp, dst_path, ec);
  if (ec) {
    fs::copy_file(tmp, dst_path, fs::copy_options::overwrite_existing, ec);
    fs::remove(tmp, ec);
    if (ec) return false;
  }
  return true;
}

bool writeMapYamlAtomic(const fs::path& tmp_dir, const fs::path& yaml_path,
                        double resolution, double origin_x, double origin_y) {
  std::error_code ec;
  fs::create_directories(tmp_dir, ec);
  if (ec) return false;
  const fs::path tmp = tmp_dir / ".map.yaml.tmp";
  {
    std::ofstream ofs(tmp, std::ios::trunc);
    if (!ofs.is_open()) return false;
    ofs << std::fixed << std::setprecision(6);
    ofs << "image: map.pgm\n";
    ofs << "mode: trinary\n";
    ofs << "resolution: " << resolution << "\n";
    ofs << "origin: [" << origin_x << ", " << origin_y << ", 0]\n";
    ofs << "negate: 0\n";
    ofs << "occupied_thresh: 0.65\n";
    ofs << "free_thresh: 0.25\n";
    ofs.flush();
    if (!ofs.good()) return false;
  }
  fs::rename(tmp, yaml_path, ec);
  if (ec) {
    fs::copy_file(tmp, yaml_path, fs::copy_options::overwrite_existing, ec);
    fs::remove(tmp, ec);
    if (ec) return false;
  }
  return true;
}

}  // namespace

float DynamicMapUpdater::GroundGrid::query(float x, float y) const {
  const Eigen::Vector2i key(static_cast<int>(std::floor(x * inv_resolution)),
                            static_cast<int>(std::floor(y * inv_resolution)));
  auto it = cells.find(key);
  if (it != cells.end()) return it->second;

  // Fallback: inverse-distance-weighted lookup in a small local neighborhood.
  float w_sum = 0.0f;
  float z_sum = 0.0f;
  for (int dx = -3; dx <= 3; ++dx) {
    for (int dy = -3; dy <= 3; ++dy) {
      if (dx == 0 && dy == 0) continue;
      auto nit = cells.find(Eigen::Vector2i(key.x() + dx, key.y() + dy));
      if (nit == cells.end()) continue;
      const float d2 = static_cast<float>(dx * dx + dy * dy);
      const float w = 1.0f / d2;
      w_sum += w;
      z_sum += w * nit->second;
    }
  }
  return w_sum > 0.0f ? (z_sum / w_sum) : global_ground_z;
}

DynamicMapUpdater::DynamicMapUpdater(std::string map_dir, std::string map_name,
                                     const RelocOptions& options)
    : map_dir_(std::move(map_dir)), map_name_(std::move(map_name)), options_(options) {
  sub_resolution_ = 0.5f * g_ivox_resolution;
  if (sub_resolution_ <= 1e-4f) sub_resolution_ = 0.25f;
  sub_inv_resolution_ = 1.0f / sub_resolution_;
}

DynamicMapUpdater::~DynamicMapUpdater() {
  joinCommitThread();
}

void DynamicMapUpdater::joinCommitThread() {
  if (commit_thread_.joinable()) {
    commit_thread_.join();
  }
  commit_running_.store(false);
}

void DynamicMapUpdater::setBaseMap(const CloudPtr& base_map) {
  base_map_.reset(new PointCloudType());
  if (base_map) {
    *base_map_ = *base_map;
  }

  l0_centroids_.clear();
  if (!base_map_ || base_map_->empty()) return;

  struct Accum {
    V3 sum = V3::Zero();
    int count = 0;
  };
  std::unordered_map<FineKey, Accum, FineKeyHash> acc;
  acc.reserve(base_map_->size() / 2 + 16);
  for (const auto& pt : *base_map_) {
    const V3 v(pt.x, pt.y, pt.z);
    if (!v.allFinite()) continue;
    const FineKey fk = (v * sub_inv_resolution_).array().floor().cast<int>();
    auto& a = acc[fk];
    a.sum += v;
    a.count += 1;
  }
  l0_centroids_.reserve(acc.size());
  for (const auto& [fk, a] : acc) {
    l0_centroids_.emplace(fk, a.sum / static_cast<scalar>(a.count));
  }

  buildGroundGrid(*base_map_);
  LOG(INFO) << " ---> [MapUpdate] L0 base sub-voxels=" << l0_centroids_.size()
            << ", adaptive global floor z=" << ground_grid_.global_ground_z
            << " m (cells=" << ground_grid_.cells.size() << ")";
}

void DynamicMapUpdater::buildGroundGrid(const PointCloudType& cloud) {
  ground_grid_ = GroundGrid{};
  const float res = static_cast<float>(std::max(0.5, options_.map_update.ground_grid_res));
  ground_grid_.resolution = res;
  ground_grid_.inv_resolution = 1.0f / res;

  if (cloud.empty()) return;

  std::unordered_map<Eigen::Vector2i, std::vector<float>, Grid2DHash> cell_zs;
  for (const auto& pt : cloud.points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
    const Eigen::Vector2i key(static_cast<int>(std::floor(pt.x * ground_grid_.inv_resolution)),
                              static_cast<int>(std::floor(pt.y * ground_grid_.inv_resolution)));
    cell_zs[key].push_back(pt.z);
  }
  if (cell_zs.empty()) return;

  // Step 1: Extract the lowest supported horizontal surface band in each XY cell,
  // filtering out isolated underground multipath outliers.
  struct CellCandidate {
    float z = 0.0f;
    int support = 0;
  };
  std::unordered_map<Eigen::Vector2i, CellCandidate, Grid2DHash> candidates;
  candidates.reserve(cell_zs.size());

  constexpr float kBandThickness = 0.12f;
  float min_cand_z = std::numeric_limits<float>::max();
  float max_cand_z = std::numeric_limits<float>::lowest();

  for (auto& [key, zs] : cell_zs) {
    if (zs.size() < 3) continue;
    std::sort(zs.begin(), zs.end());
    const int min_support = zs.size() >= 12 ? 4 : 2;
    std::size_t chosen_start = zs.size();
    std::size_t chosen_end = zs.size();
    for (std::size_t i = 0; i < zs.size(); ++i) {
      std::size_t j = i;
      while (j < zs.size() && (zs[j] - zs[i]) <= kBandThickness) ++j;
      if (static_cast<int>(j - i) >= min_support) {
        chosen_start = i;
        chosen_end = j;
        break;
      }
    }
    if (chosen_start == zs.size()) continue;
    const std::size_t mid = chosen_start + (chosen_end - chosen_start) / 2;
    const float z_med = zs[mid];
    candidates[key] = CellCandidate{z_med, static_cast<int>(chosen_end - chosen_start)};
    min_cand_z = std::min(min_cand_z, z_med);
    max_cand_z = std::max(max_cand_z, z_med);
  }

  if (candidates.empty()) {
    std::vector<float> all_z;
    all_z.reserve(cloud.size());
    for (const auto& pt : cloud.points) {
      if (std::isfinite(pt.z)) all_z.push_back(pt.z);
    }
    if (!all_z.empty()) {
      const std::size_t idx = all_z.size() / 10;
      std::nth_element(all_z.begin(), all_z.begin() + idx, all_z.end());
      ground_grid_.global_ground_z = all_z[idx];
    }
    return;
  }

  // Step 2: Weighted 0.15 m histogram over cell low-Z candidates to locate the
  // dominant true floor level (handles negative z such as -0.3 m or -1.8 m).
  constexpr float kHistBin = 0.15f;
  std::unordered_map<int, double> hist;
  for (const auto& [key, cand] : candidates) {
    const int bin = static_cast<int>(std::floor(cand.z / kHistBin));
    const double w = std::min(cand.support, 30);
    hist[bin] += w;
    // Smooth into immediate neighbors so boundary splits don't bias the mode.
    hist[bin - 1] += 0.35 * w;
    hist[bin + 1] += 0.35 * w;
  }
  int best_bin = 0;
  double best_weight = -1.0;
  for (const auto& [bin, w] : hist) {
    // Tie-break slightly toward lower bins when two flat levels have comparable area.
    if (w > best_weight || (std::abs(w - best_weight) < 1e-6 && bin < best_bin)) {
      best_weight = w;
      best_bin = bin;
    }
  }
  const float mode_center = (static_cast<float>(best_bin) + 0.5f) * kHistBin;
  double mode_sum = 0.0;
  double mode_w = 0.0;
  for (const auto& [key, cand] : candidates) {
    if (std::abs(cand.z - mode_center) <= 0.25f) {
      const double w = std::min(cand.support, 30);
      mode_sum += w * cand.z;
      mode_w += w;
    }
  }
  ground_grid_.global_ground_z = mode_w > 0.0 ? static_cast<float>(mode_sum / mode_w) : mode_center;

  // Step 3: Seed confirmed floor cells near the global floor mode, then grow across
  // sloped ramps via bounded-gradient BFS (rejects tabletops / parked vehicle roofs).
  const float max_var = static_cast<float>(std::max(0.3, options_.map_update.max_floor_variation));
  const float max_step = std::max(0.22f, 0.18f * res);  // ~18% max slope per cell

  std::unordered_map<Eigen::Vector2i, float, Grid2DHash> confirmed;
  std::queue<Eigen::Vector2i> bfs;
  for (const auto& [key, cand] : candidates) {
    if (std::abs(cand.z - ground_grid_.global_ground_z) <= 0.30f) {
      confirmed[key] = cand.z;
      bfs.push(key);
    }
  }

  while (!bfs.empty()) {
    const Eigen::Vector2i cur = bfs.front();
    bfs.pop();
    const float cur_z = confirmed[cur];
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        if (dx == 0 && dy == 0) continue;
        const Eigen::Vector2i nb(cur.x() + dx, cur.y() + dy);
        if (confirmed.count(nb)) continue;
        auto cit = candidates.find(nb);
        if (cit == candidates.end()) continue;
        const float step_limit = (dx != 0 && dy != 0) ? (1.414f * max_step) : max_step;
        if (std::abs(cit->second.z - cur_z) <= step_limit &&
            std::abs(cit->second.z - ground_grid_.global_ground_z) <= max_var) {
          confirmed[nb] = cit->second.z;
          bfs.push(nb);
        }
      }
    }
  }

  // Step 4: Populate final ground grid — confirmed cells directly, and occluded cells
  // (e.g., floor hidden beneath an initial parked car/table) + 1-cell halo via IDW.
  ground_grid_.cells = confirmed;
  std::unordered_set<Eigen::Vector2i, Grid2DHash> to_fill;
  for (const auto& [key, zs] : cell_zs) {
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        const Eigen::Vector2i k(key.x() + dx, key.y() + dy);
        if (!confirmed.count(k)) to_fill.insert(k);
      }
    }
  }

  constexpr int kIdwRadius = 6;
  for (const auto& key : to_fill) {
    float w_sum = 0.0f;
    float z_sum = 0.0f;
    for (int dx = -kIdwRadius; dx <= kIdwRadius; ++dx) {
      for (int dy = -kIdwRadius; dy <= kIdwRadius; ++dy) {
        if (dx == 0 && dy == 0) continue;
        const int d2 = dx * dx + dy * dy;
        if (d2 > kIdwRadius * kIdwRadius) continue;
        auto cit = confirmed.find(Eigen::Vector2i(key.x() + dx, key.y() + dy));
        if (cit == confirmed.end()) continue;
        const float w = 1.0f / static_cast<float>(d2);
        w_sum += w;
        z_sum += w * cit->second;
      }
    }
    ground_grid_.cells[key] = (w_sum > 0.0f) ? (z_sum / w_sum) : ground_grid_.global_ground_z;
  }
}

float DynamicMapUpdater::estimateGroundZ(float x, float y) const {
  return ground_grid_.query(x, y);
}

void DynamicMapUpdater::populateIndex(OctVoxMapType& ivox, const VV3& transient_extra) const {
  if (base_map_ && !base_map_->empty()) {
    VV3 base_pts;
    base_pts.reserve(base_map_->size());
    for (const auto& pt : *base_map_) {
      const V3 v(pt.x, pt.y, pt.z);
      if (!v.allFinite()) continue;
      const FineKey fk = ivox.toFineKey(v);
      if (removed_subvoxels_.count(fk)) continue;
      base_pts.push_back(v);
    }
    ivox.insertBase(base_pts);
  }

  for (const auto& [fk, vox] : added_voxels_) {
    ivox.upsertSubVoxel(fk, vox.centroid, /*is_base=*/false);
  }

  if (!transient_extra.empty()) {
    ivox.insert(transient_extra);
  }
}

void DynamicMapUpdater::purgePendingInZone(const ExcludeFn& in_vehicle_zone) {
  if (!in_vehicle_zone) return;
  for (auto it = pending_add_.begin(); it != pending_add_.end();) {
    if (in_vehicle_zone(it->second.mean_pt)) {
      it = pending_add_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = carve_miss_.begin(); it != carve_miss_.end();) {
    const V3 approx_center = (it->first.cast<scalar>() + V3::Constant(scalar(0.5))) * sub_resolution_;
    if (in_vehicle_zone(approx_center)) {
      it = carve_miss_.erase(it);
    } else {
      ++it;
    }
  }
}

void DynamicMapUpdater::onTrackingReset() {
  pending_add_.clear();
  carve_miss_.clear();
  has_last_kf_ = false;
}

void DynamicMapUpdater::cleanupStaleCandidates(double now_sec) {
  const double ttl = options_.map_update.pending_ttl_sec;
  if (ttl <= 0.0 || now_sec <= 0.0) return;
  for (auto it = pending_add_.begin(); it != pending_add_.end();) {
    if (now_sec - it->second.last_stamp > ttl) {
      it = pending_add_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = carve_miss_.begin(); it != carve_miss_.end();) {
    if (now_sec - it->second.last_stamp > ttl) {
      it = carve_miss_.erase(it);
    } else {
      ++it;
    }
  }
}

bool DynamicMapUpdater::processScan(const ScanUpdateInput& input, OctVoxMapType& ivox) {
  const auto& cfg = options_.map_update;
  if (!cfg.enable || !base_map_ || base_map_->empty()) return false;

  // Standstill or driverseat collection: do not accumulate keyframe evidence,
  // though maybeScheduleCommit() will still check if a background commit is due.
  if (input.frame_still) return false;
  if (cfg.pause_during_driverseat && input.driverseat_collecting) return false;
  if (!input.scan_undistort_imu || input.scan_undistort_imu->empty()) return false;

  // High-confidence tracking & L0 anchor gate.
  if (input.ds_size == 0 || static_cast<int>(input.matched) < cfg.min_matched_points) return false;
  const double match_ratio = static_cast<double>(input.matched) / static_cast<double>(input.ds_size);
  if (match_ratio < cfg.min_match_ratio) return false;

  if (static_cast<int>(input.base_matched) < cfg.min_base_matched) return false;
  const double base_ratio = static_cast<double>(input.base_matched) / static_cast<double>(input.matched);
  if (base_ratio < cfg.min_base_ratio) return false;

  if (!input.velocity.allFinite() || input.velocity.norm() > cfg.max_speed) return false;
  if (!std::isfinite(input.angular_speed) || input.angular_speed > cfg.max_angular_vel) return false;

  // Keyframe displacement gate: ensures multi-view baseline and bounded compute.
  if (has_last_kf_) {
    const double d_trans = (input.pose.t_ - last_kf_pose_.t_).norm();
    const Eigen::Matrix3d dR = (last_kf_pose_.R_.transpose() * input.pose.R_).cast<double>();
    const double cos_ang = std::clamp((dR.trace() - 1.0) * 0.5, -1.0, 1.0);
    const double d_rot_deg = std::acos(cos_ang) * (180.0 / M_PI);
    if (d_trans < cfg.kf_trans_thresh && d_rot_deg < cfg.kf_rot_thresh_deg) {
      return false;
    }
  }
  last_kf_pose_ = input.pose;
  has_last_kf_ = true;

  if (++kf_counter_ % 20 == 0) {
    cleanupStaleCandidates(input.stamp);
  }

  const V3 sensor_origin = input.pose.R_ * input.lidar_lever_arm + input.pose.t_;
  const float min_r = static_cast<float>(cfg.min_range);
  const float max_add_r = static_cast<float>(cfg.max_add_range);
  const float max_carve_r = static_cast<float>(cfg.max_carve_range);
  const float carve_margin = static_cast<float>(cfg.carve_margin);
  const float min_r2 = min_r * min_r;
  const float max_add_r2 = max_add_r * max_add_r;
  const float max_protect_r2 = (max_carve_r + 1.5f) * (max_carve_r + 1.5f);
  const float min_carve_hit_r = min_r + carve_margin + 0.20f;
  const float min_carve_hit_r2 = min_carve_hit_r * min_carve_hit_r;

  struct HitBucket {
    V3 sum = V3::Zero();
    int count = 0;
    VV3 fine_samples;
  };
  std::unordered_map<FineKey, HitBucket, FineKeyHash> add_hits;
  std::unordered_set<FineKey, FineKeyHash> protected_carve_keys;
  std::unordered_map<FineKey, V3, FineKeyHash> carve_ray_endpoints;

  const auto& pts = input.scan_undistort_imu->points;
  add_hits.reserve(pts.size() / 4 + 16);
  protected_carve_keys.reserve(pts.size() / 2 + 16);
  carve_ray_endpoints.reserve(pts.size() / 4 + 16);

  constexpr float kRayEndpointInvRes = 5.0f;  // 0.20 m endpoint deduplication
  constexpr float kEdgeGuard = 0.08f;

  for (const auto& sp : pts) {
    const V3 p_imu(sp.x, sp.y, sp.z);
    if (!p_imu.allFinite()) continue;
    const V3 p_world = input.pose.R_ * p_imu + input.pose.t_;
    const V3 ray_vec = p_world - sensor_origin;
    const float r2 = static_cast<float>(ray_vec.squaredNorm());
    if (r2 < 0.09f) continue;

    if (r2 <= max_protect_r2) {
      const FineKey fk = ivox.toFineKey(p_world);
      protected_carve_keys.insert(fk);
      // Protect immediate sub-voxel face neighbors when a return lands near a boundary.
      for (int axis = 0; axis < 3; ++axis) {
        const float coord = static_cast<float>(p_world[axis]) * sub_inv_resolution_;
        const float frac = coord - std::floor(coord);
        if (frac * sub_resolution_ < kEdgeGuard) {
          FineKey nfk = fk;
          nfk[axis] -= 1;
          protected_carve_keys.insert(nfk);
        } else if ((1.0f - frac) * sub_resolution_ < kEdgeGuard) {
          FineKey nfk = fk;
          nfk[axis] += 1;
          protected_carve_keys.insert(nfk);
        }
      }
    }

    if (r2 >= min_r2 && r2 <= max_add_r2) {
      const FineKey fk = ivox.toFineKey(p_world);
      auto& bucket = add_hits[fk];
      bucket.sum += p_world;
      bucket.count += 1;
      mergeFineSample(bucket.fine_samples, p_world);
    }

    if (r2 >= min_carve_hit_r2) {
      const FineKey ray_key = (p_world * kRayEndpointInvRes).array().floor().cast<int>();
      carve_ray_endpoints.try_emplace(ray_key, p_world);
    }
  }

  bool patch_changed = false;
  const float obs_z_min = static_cast<float>(cfg.obs_z_min);
  const float obs_z_max = static_cast<float>(cfg.obs_z_max);

  // Step A: Direct obstacle hits -> decay carve misses & accumulate static additions.
  for (const auto& [fk, bucket] : add_hits) {
    // Any direct hit in this sub-voxel resets/decays pending carve evidence.
    if (auto cit = carve_miss_.find(fk); cit != carve_miss_.end()) {
      cit->second.count = std::max(0, cit->second.count - 2);
      if (cit->second.count == 0) carve_miss_.erase(cit);
    }

    const V3 pt = bucket.sum / static_cast<scalar>(bucket.count);
    if (input.in_vehicle_zone && input.in_vehicle_zone(pt)) continue;

    const float h = static_cast<float>(pt.z()) - estimateGroundZ(static_cast<float>(pt.x()),
                                                                 static_cast<float>(pt.y()));
    if (h < obs_z_min || h > obs_z_max) continue;

    V3 existing_pt;
    bool existing_base = false;
    if (ivox.getSubVoxel(fk, existing_pt, &existing_base)) {
      if (auto ait = added_voxels_.find(fk); ait != added_voxels_.end()) {
        for (const auto& fs_pt : bucket.fine_samples) {
          mergeFineSample(ait->second.fine_points, fs_pt);
        }
      }
      continue;
    }

    auto [pit, inserted] = pending_add_.try_emplace(fk);
    PendingAdd& cand = pit->second;
    if (inserted) {
      cand.mean_pt = pt;
      cand.fine_points = bucket.fine_samples;
      cand.hits = 1;
      cand.misses = 0;
      cand.first_stamp = input.stamp;
      cand.last_stamp = input.stamp;
    } else {
      const scalar alpha = scalar(1.0) / static_cast<scalar>(cand.hits + 1);
      cand.mean_pt = cand.mean_pt * (scalar(1.0) - alpha) + pt * alpha;
      for (const auto& fs_pt : bucket.fine_samples) {
        mergeFineSample(cand.fine_points, fs_pt);
      }
      cand.hits += 1;
      cand.last_stamp = input.stamp;
    }

    const double age = input.stamp - cand.first_stamp;
    if (cand.hits >= cfg.confirm_hits && age >= cfg.confirm_min_age_sec &&
        cand.misses * 2 <= cand.hits) {
      if (ivox.upsertSubVoxel(fk, cand.mean_pt, /*is_base=*/false)) {
        AddedVoxel av;
        av.centroid = cand.mean_pt;
        av.fine_points = std::move(cand.fine_points);
        if (av.fine_points.empty()) av.fine_points.push_back(av.centroid);
        added_voxels_[fk] = std::move(av);
        patch_changed = true;
      }
      pending_add_.erase(pit);
    }
  }

  // Step B: Real-return ray traversal -> free-space carving of removed static objects.
  const float ray_radius2 = static_cast<float>(cfg.carve_ray_radius * cfg.carve_ray_radius);
  const float step_len = std::max(0.12f, 0.75f * sub_resolution_);
  std::unordered_set<FineKey, FineKeyHash> missed_this_kf;
  missed_this_kf.reserve(512);

  for (const auto& [ray_key, p_hit] : carve_ray_endpoints) {
    const V3 delta = p_hit - sensor_origin;
    const float d_hit = static_cast<float>(delta.norm());
    if (d_hit < min_carve_hit_r) continue;

    const V3 dir = delta / static_cast<scalar>(d_hit);
    const float t_end = std::min(d_hit - carve_margin, max_carve_r);
    if (t_end <= min_r) continue;

    FineKey prev_fk(std::numeric_limits<int>::max(), 0, 0);
    for (float t = min_r; t <= t_end; t += step_len) {
      const V3 sample = sensor_origin + dir * static_cast<scalar>(t);
      const FineKey fk = ivox.toFineKey(sample);
      if (fk == prev_fk) continue;
      prev_fk = fk;

      if (protected_carve_keys.count(fk)) continue;

      // Penalize unconfirmed transient candidates traversed by a behind-hit ray
      // (at most once per keyframe).
      const bool first_visit_in_kf = missed_this_kf.insert(fk).second;
      if (!first_visit_in_kf) continue;

      if (auto pit = pending_add_.find(fk); pit != pending_add_.end()) {
        pit->second.misses += 1;
        if (pit->second.misses >= 3 && pit->second.misses > pit->second.hits) {
          pending_add_.erase(pit);
        }
      }

      V3 vox_pt;
      bool is_base = false;
      if (!ivox.getSubVoxel(fk, vox_pt, &is_base)) continue;

      if (input.in_vehicle_zone && input.in_vehicle_zone(vox_pt)) continue;

      // L0 Anchor Protection: ground and upper walls in L0 are permanently locked.
      const float h = static_cast<float>(vox_pt.z()) -
                      estimateGroundZ(static_cast<float>(vox_pt.x()), static_cast<float>(vox_pt.y()));
      if (is_base && (h < obs_z_min || h > obs_z_max)) continue;

      // Require the ray to pass tightly near the actual stored sub-voxel centroid.
      const V3 rel = vox_pt - sensor_origin;
      const float proj = static_cast<float>(rel.dot(dir));
      if (proj <= min_r || proj >= d_hit - carve_margin) continue;
      const float perp2 = static_cast<float>((rel - dir * static_cast<scalar>(proj)).squaredNorm());
      if (perp2 > ray_radius2) continue;

      auto [mit, inserted] = carve_miss_.try_emplace(fk);
      CarveMiss& cm = mit->second;
      if (inserted) {
        cm.first_origin = sensor_origin;
        cm.max_baseline = 0.0f;
        cm.count = 1;
        cm.last_stamp = input.stamp;
      } else {
        const float baseline = static_cast<float>((sensor_origin - cm.first_origin).norm());
        cm.max_baseline = std::max(cm.max_baseline, baseline);
        cm.count += 1;
        cm.last_stamp = input.stamp;
      }

      const int need_misses = is_base ? cfg.carve_confirm_misses : cfg.carve_added_misses;
      if (cm.count >= need_misses && cm.max_baseline >= static_cast<float>(cfg.carve_min_baseline)) {
        ivox.eraseSubVoxel(fk);
        if (added_voxels_.erase(fk) > 0) {
          patch_changed = true;
        }
        if (is_base || l0_centroids_.count(fk)) {
          removed_subvoxels_.insert(fk);
          patch_changed = true;
        }
        carve_miss_.erase(mit);
      }
    }
  }

  if (patch_changed && dirty_since_time_ < 0.0) {
    dirty_since_time_ = input.stamp;
  }
  return patch_changed;
}

void DynamicMapUpdater::computeDirtySets(
    std::unordered_set<FineKey, FineKeyHash>& newly_occupied,
    std::unordered_set<FineKey, FineKeyHash>& newly_cleared) const {
  newly_occupied.clear();
  newly_cleared.clear();

  for (const auto& [fk, vox] : added_voxels_) {
    if (!committed_added_keys_.count(fk)) {
      newly_occupied.insert(fk);
    }
  }
  for (const auto& fk : removed_subvoxels_) {
    if (!committed_removed_keys_.count(fk) && !added_voxels_.count(fk)) {
      newly_cleared.insert(fk);
    }
  }
  for (const auto& fk : committed_added_keys_) {
    if (!added_voxels_.count(fk)) {
      newly_cleared.insert(fk);
    }
  }
}

std::size_t DynamicMapUpdater::computeMaxClusterSize(
    const std::unordered_set<FineKey, FineKeyHash>& a,
    const std::unordered_set<FineKey, FineKeyHash>& b) const {
  std::unordered_set<FineKey, FineKeyHash> all = a;
  all.insert(b.begin(), b.end());
  if (all.empty()) return 0;

  std::unordered_set<FineKey, FineKeyHash> visited;
  visited.reserve(all.size());
  std::size_t max_cluster = 0;

  std::queue<FineKey> q;
  for (const auto& seed : all) {
    if (!visited.insert(seed).second) continue;
    std::size_t cluster = 0;
    q.push(seed);
    while (!q.empty()) {
      const FineKey cur = q.front();
      q.pop();
      ++cluster;
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            if (dx == 0 && dy == 0 && dz == 0) continue;
            const FineKey nb(cur.x() + dx, cur.y() + dy, cur.z() + dz);
            if (all.count(nb) && visited.insert(nb).second) {
              q.push(nb);
            }
          }
        }
      }
    }
    max_cluster = std::max(max_cluster, cluster);
  }
  return max_cluster;
}

void DynamicMapUpdater::maybeScheduleCommit(double now_sec, bool frame_still) {
  const auto& cfg = options_.map_update;
  if (!cfg.enable || !base_map_ || base_map_->empty()) return;
  if (commit_running_.load()) return;
  {
    std::lock_guard<std::mutex> lock(result_mutex_);
    if (pending_result_) return;
  }

  std::unordered_set<FineKey, FineKeyHash> newly_occupied;
  std::unordered_set<FineKey, FineKeyHash> newly_cleared;
  computeDirtySets(newly_occupied, newly_cleared);
  const std::size_t total_dirty = newly_occupied.size() + newly_cleared.size();
  if (total_dirty == 0) {
    dirty_since_time_ = -1.0;
    return;
  }
  if (dirty_since_time_ < 0.0) dirty_since_time_ = now_sec;

  const bool meets_total = static_cast<int>(total_dirty) >= cfg.min_changed_voxels;
  const bool meets_cluster =
      meets_total ||
      static_cast<int>(computeMaxClusterSize(newly_occupied, newly_cleared)) >= cfg.min_cluster_voxels;
  if (!meets_total && !meets_cluster) return;

  if (now_sec - last_commit_time_ < cfg.commit_cooldown_sec) return;
  if (!frame_still && (now_sec - dirty_since_time_) < cfg.max_commit_defer_sec) return;

  joinCommitThread();

  CommitJob job;
  job.generation = ++generation_;
  job.map_dir = map_dir_;
  job.map_name = map_name_;
  job.options = options_;
  job.base_map = base_map_;
  job.ground_grid = ground_grid_;
  job.sub_inv_resolution = sub_inv_resolution_;
  job.added_voxels = added_voxels_;
  job.removed_subvoxels = removed_subvoxels_;
  job.newly_occupied = std::move(newly_occupied);
  job.newly_cleared = std::move(newly_cleared);
  job.rebuild_reloc_targets = true;

  inflight_added_snapshot_ = added_voxels_;
  inflight_removed_snapshot_ = removed_subvoxels_;
  last_commit_time_ = now_sec;
  commit_running_.store(true);

  LOG(INFO) << " ---> [MapUpdate] launching background commit (gen=" << job.generation
            << ", dirty_add=" << job.newly_occupied.size()
            << ", dirty_carve=" << job.newly_cleared.size()
            << ", total_added=" << job.added_voxels.size()
            << ", total_carved=" << job.removed_subvoxels.size() << ")";

  commit_thread_ = std::thread([this, job = std::move(job)]() {
    CommitResult res = executeCommitJob(job);
    {
      std::lock_guard<std::mutex> lock(result_mutex_);
      pending_result_ = std::make_unique<CommitResult>(std::move(res));
    }
    commit_running_.store(false);
  });
}

bool DynamicMapUpdater::pollCommitResult(CommitResult& out) {
  std::unique_ptr<CommitResult> ready;
  {
    std::lock_guard<std::mutex> lock(result_mutex_);
    if (!pending_result_) return false;
    ready = std::move(pending_result_);
  }
  joinCommitThread();

  if (ready->generation != generation_) {
    // Stale commit superseded by resetToOrigin; ignore.
    return false;
  }
  if (!ready->success) {
    LOG(WARNING) << " ---> [MapUpdate] background commit failed (gen=" << ready->generation << ")";
    return false;
  }

  committed_added_keys_.clear();
  for (const auto& [fk, vox] : inflight_added_snapshot_) {
    committed_added_keys_.insert(fk);
  }
  committed_removed_keys_ = inflight_removed_snapshot_;
  inflight_added_snapshot_.clear();
  inflight_removed_snapshot_.clear();

  std::unordered_set<FineKey, FineKeyHash> rem_occ, rem_clr;
  computeDirtySets(rem_occ, rem_clr);
  if (rem_occ.empty() && rem_clr.empty()) {
    dirty_since_time_ = -1.0;
  }

  out = std::move(*ready);
  return true;
}

bool DynamicMapUpdater::flushSyncIfNeeded() {
  const auto& cfg = options_.map_update;
  if (!cfg.enable || !base_map_ || base_map_->empty()) return false;

  joinCommitThread();
  {
    std::lock_guard<std::mutex> lock(result_mutex_);
    if (pending_result_ && pending_result_->success && pending_result_->generation == generation_) {
      committed_added_keys_.clear();
      for (const auto& [fk, vox] : inflight_added_snapshot_) {
        committed_added_keys_.insert(fk);
      }
      committed_removed_keys_ = inflight_removed_snapshot_;
      pending_result_.reset();
    }
  }

  std::unordered_set<FineKey, FineKeyHash> newly_occupied;
  std::unordered_set<FineKey, FineKeyHash> newly_cleared;
  computeDirtySets(newly_occupied, newly_cleared);
  const std::size_t total_dirty = newly_occupied.size() + newly_cleared.size();
  if (total_dirty == 0) return false;

  const bool meets_total = static_cast<int>(total_dirty) >= cfg.min_changed_voxels;
  const bool meets_cluster =
      meets_total ||
      static_cast<int>(computeMaxClusterSize(newly_occupied, newly_cleared)) >= cfg.min_cluster_voxels;
  if (!meets_total && !meets_cluster) return false;

  CommitJob job;
  job.generation = ++generation_;
  job.map_dir = map_dir_;
  job.map_name = map_name_;
  job.options = options_;
  job.base_map = base_map_;
  job.ground_grid = ground_grid_;
  job.sub_inv_resolution = sub_inv_resolution_;
  job.added_voxels = added_voxels_;
  job.removed_subvoxels = removed_subvoxels_;
  job.newly_occupied = std::move(newly_occupied);
  job.newly_cleared = std::move(newly_cleared);
  job.rebuild_reloc_targets = false;  // Shutdown / standby flush: only disk files needed.

  LOG(INFO) << " ---> [MapUpdate] synchronous flush on standby/exit (dirty=" << total_dirty << ")";
  const CommitResult res = executeCommitJob(job);
  if (!res.success) return false;

  committed_added_keys_.clear();
  for (const auto& [fk, vox] : added_voxels_) committed_added_keys_.insert(fk);
  committed_removed_keys_ = removed_subvoxels_;
  dirty_since_time_ = -1.0;
  return true;
}

CloudPtr DynamicMapUpdater::synthesizePersistentMap(const CommitJob& job) {
  CloudPtr out(new PointCloudType());
  if (!job.base_map) return out;

  std::size_t extra_pts = 0;
  for (const auto& [fk, vox] : job.added_voxels) {
    extra_pts += std::max<std::size_t>(1, vox.fine_points.size());
  }
  out->points.reserve(job.base_map->size() + extra_pts);

  for (const auto& pt : job.base_map->points) {
    const V3 v(pt.x, pt.y, pt.z);
    if (!v.allFinite()) continue;
    const FineKey fk = (v * job.sub_inv_resolution).array().floor().cast<int>();
    if (job.removed_subvoxels.count(fk)) continue;
    out->points.push_back(pt);
  }

  for (const auto& [fk, vox] : job.added_voxels) {
    if (!vox.fine_points.empty()) {
      for (const auto& fp : vox.fine_points) {
        PointType p;
        p.x = static_cast<float>(fp.x());
        p.y = static_cast<float>(fp.y());
        p.z = static_cast<float>(fp.z());
        p.intensity = 100.0f;
        out->points.push_back(p);
      }
    } else {
      PointType p;
      p.x = static_cast<float>(vox.centroid.x());
      p.y = static_cast<float>(vox.centroid.y());
      p.z = static_cast<float>(vox.centroid.z());
      p.intensity = 100.0f;
      out->points.push_back(p);
    }
  }

  out->width = static_cast<uint32_t>(out->points.size());
  out->height = 1;
  out->is_dense = true;
  return out;
}

bool DynamicMapUpdater::ensureOriginBackup(const std::string& map_dir,
                                           const std::string& map_name) {
  const fs::path patch_dir = patchDirPath(map_dir);
  std::error_code ec;
  fs::create_directories(patch_dir, ec);
  if (ec) {
    LOG(ERROR) << " ---> [MapUpdate] failed to create backup dir: " << patch_dir.string();
    return false;
  }

  const fs::path cur_pcd = fs::path(map_dir) / map_name;
  const fs::path orig_pcd = originPcdPath(map_dir, map_name);
  if (!fs::exists(orig_pcd, ec)) {
    if (!copyFileAtomic(cur_pcd, orig_pcd, patch_dir)) {
      LOG(ERROR) << " ---> [MapUpdate] failed to back up origin PCD to " << orig_pcd.string();
      return false;
    }
    LOG(INFO) << " ---> [MapUpdate] backed up factory origin map to " << orig_pcd.string();
  }

  const fs::path cur_pgm = fs::path(map_dir) / "map.pgm";
  const fs::path orig_pgm = patch_dir / "map.origin.pgm";
  if (fs::exists(cur_pgm, ec) && !fs::exists(orig_pgm, ec)) {
    if (!copyFileAtomic(cur_pgm, orig_pgm, patch_dir)) {
      LOG(WARNING) << " ---> [MapUpdate] failed to back up map.origin.pgm";
    }
  }

  const fs::path cur_yaml = fs::path(map_dir) / "map.yaml";
  const fs::path orig_yaml = patch_dir / "map.origin.yaml";
  if (fs::exists(cur_yaml, ec) && !fs::exists(orig_yaml, ec)) {
    if (!copyFileAtomic(cur_yaml, orig_yaml, patch_dir)) {
      LOG(WARNING) << " ---> [MapUpdate] failed to back up map.origin.yaml";
    }
  }
  return true;
}

bool DynamicMapUpdater::writePcdAtomic(const std::string& map_dir,
                                       const std::string& target_path,
                                       const PointCloudType& cloud) {
  const fs::path patch_dir = patchDirPath(map_dir);
  std::error_code ec;
  fs::create_directories(patch_dir, ec);
  if (ec) return false;

  const fs::path tmp_pcd = patch_dir / ".global.pcd.tmp";
  if (pcl::io::savePCDFileBinary(tmp_pcd.string(), cloud) != 0) {
    fs::remove(tmp_pcd, ec);
    return false;
  }
  fs::rename(tmp_pcd, fs::path(target_path), ec);
  if (ec) {
    fs::copy_file(tmp_pcd, fs::path(target_path), fs::copy_options::overwrite_existing, ec);
    fs::remove(tmp_pcd, ec);
    if (ec) return false;
  }
  return true;
}

bool DynamicMapUpdater::syncPgmMap(const CommitJob& job, const PointCloudType& updated_map,
                                   std::string& out_yaml_path) {
  const auto& pgm_cfg = job.options.map_update.pgm;
  if (!pgm_cfg.enable) return false;

  const fs::path dir(job.map_dir);
  const fs::path patch_dir = patchDirPath(job.map_dir);
  const fs::path pgm_path = dir / "map.pgm";
  const fs::path yaml_path = dir / "map.yaml";

  const float z_min = static_cast<float>(pgm_cfg.z_min);
  const float z_max = static_cast<float>(pgm_cfg.z_max);
  const float sub_res = 1.0f / job.sub_inv_resolution;

  std::error_code ec;
  if (fs::exists(pgm_path, ec) && fs::exists(yaml_path, ec)) {
    MapYamlMeta meta;
    PgmImage img;
    if (!parseMapYaml(yaml_path, meta) || !readPgmFile(pgm_path, img)) {
      LOG(WARNING) << " ---> [MapUpdate] failed to parse existing map.pgm/map.yaml; skipping PGM sync";
      return false;
    }

    const uint8_t occ_val = (meta.negate == 1) ? 254 : 0;
    const uint8_t free_val = (meta.negate == 1) ? 0 : 254;
    const double inv_res = 1.0 / meta.resolution;

    // 1. Clear pixels inside newly carved 3D obstacle footprints (preserving GIMP virtual walls elsewhere).
    for (const auto& fk : job.newly_cleared) {
      const double x0 = fk.x() * sub_res;
      const double x1 = (fk.x() + 1) * sub_res;
      const double y0 = fk.y() * sub_res;
      const double y1 = (fk.y() + 1) * sub_res;

      const int ix0 = std::max(0, static_cast<int>(std::floor((x0 - meta.origin_x) * inv_res)));
      const int ix1 = std::min(img.width - 1, static_cast<int>(std::floor((x1 - 1e-5 - meta.origin_x) * inv_res)));
      const int iy0 = std::max(0, static_cast<int>(std::floor((y0 - meta.origin_y) * inv_res)));
      const int iy1 = std::min(img.height - 1, static_cast<int>(std::floor((y1 - 1e-5 - meta.origin_y) * inv_res)));

      for (int iy = iy0; iy <= iy1; ++iy) {
        const int row = img.height - 1 - iy;
        for (int ix = ix0; ix <= ix1; ++ix) {
          img.pixels[static_cast<std::size_t>(row) * img.width + ix] = free_val;
        }
      }
    }

    // 2. Build 2D dirty XY neighborhood around newly cleared & newly occupied sub-voxels,
    //    then re-project all persistent obstacle points (z_min <= z - z_ground <= z_max) to occupied.
    std::unordered_set<Eigen::Vector2i, Grid2DHash> dirty_xy;
    for (const auto& fk : job.newly_cleared) {
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          dirty_xy.emplace(fk.x() + dx, fk.y() + dy);
    }
    for (const auto& fk : job.newly_occupied) {
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          dirty_xy.emplace(fk.x() + dx, fk.y() + dy);
    }

    for (const auto& pt : updated_map.points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
      const Eigen::Vector2i k2(static_cast<int>(std::floor(pt.x * job.sub_inv_resolution)),
                               static_cast<int>(std::floor(pt.y * job.sub_inv_resolution)));
      if (!dirty_xy.count(k2)) continue;
      const float h = pt.z - job.ground_grid.query(pt.x, pt.y);
      if (h < z_min || h > z_max) continue;

      const int ix = static_cast<int>(std::floor((pt.x - meta.origin_x) * inv_res));
      const int iy = static_cast<int>(std::floor((pt.y - meta.origin_y) * inv_res));
      if (ix < 0 || ix >= img.width || iy < 0 || iy >= img.height) continue;
      const int row = img.height - 1 - iy;
      img.pixels[static_cast<std::size_t>(row) * img.width + ix] = occ_val;
    }

    if (!writePgmAtomic(patch_dir, pgm_path, img)) return false;
    out_yaml_path = yaml_path.string();
    return true;
  }

  if (!pgm_cfg.create_if_missing || updated_map.empty()) return false;

  // Generate fresh map.pgm + map.yaml when absent.
  const double res = std::max(0.01, pgm_cfg.resolution);
  const double inv_res = 1.0 / res;
  double min_x = std::numeric_limits<double>::max();
  double max_x = std::numeric_limits<double>::lowest();
  double min_y = std::numeric_limits<double>::max();
  double max_y = std::numeric_limits<double>::lowest();
  for (const auto& pt : updated_map.points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y)) continue;
    min_x = std::min<double>(min_x, pt.x);
    max_x = std::max<double>(max_x, pt.x);
    min_y = std::min<double>(min_y, pt.y);
    max_y = std::max<double>(max_y, pt.y);
  }
  if (min_x >= max_x || min_y >= max_y) return false;

  constexpr double kPad = 1.0;
  const double origin_x = min_x - kPad;
  const double origin_y = min_y - kPad;
  const int width = static_cast<int>(std::ceil((max_x + kPad - origin_x) * inv_res));
  const int height = static_cast<int>(std::ceil((max_y + kPad - origin_y) * inv_res));
  if (width <= 0 || height <= 0 || width > 20000 || height > 20000) return false;

  PgmImage img;
  img.width = width;
  img.height = height;
  img.pixels.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 205);

  // Mark observed floor neighborhood as free (254), then obstacle band as occupied (0).
  for (const auto& pt : updated_map.points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
    const float h = pt.z - job.ground_grid.query(pt.x, pt.y);
    const int ix = static_cast<int>(std::floor((pt.x - origin_x) * inv_res));
    const int iy = static_cast<int>(std::floor((pt.y - origin_y) * inv_res));
    if (ix < 0 || ix >= width || iy < 0 || iy >= height) continue;
    if (std::abs(h) <= 0.20f) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int nx = ix + dx;
          const int ny = iy + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          auto& px = img.pixels[static_cast<std::size_t>(height - 1 - ny) * width + nx];
          if (px == 205) px = 254;
        }
      }
    }
  }
  for (const auto& pt : updated_map.points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
    const float h = pt.z - job.ground_grid.query(pt.x, pt.y);
    if (h < z_min || h > z_max) continue;
    const int ix = static_cast<int>(std::floor((pt.x - origin_x) * inv_res));
    const int iy = static_cast<int>(std::floor((pt.y - origin_y) * inv_res));
    if (ix < 0 || ix >= width || iy < 0 || iy >= height) continue;
    img.pixels[static_cast<std::size_t>(height - 1 - iy) * width + ix] = 0;
  }

  if (!writePgmAtomic(patch_dir, pgm_path, img)) return false;
  if (!writeMapYamlAtomic(patch_dir, yaml_path, res, origin_x, origin_y)) return false;
  out_yaml_path = yaml_path.string();
  return true;
}

DynamicMapUpdater::CommitResult DynamicMapUpdater::executeCommitJob(const CommitJob& job) {
  const auto t0 = std::chrono::steady_clock::now();
  CommitResult res;
  res.generation = job.generation;
  res.added_voxels = job.added_voxels.size();
  res.removed_voxels = job.removed_subvoxels.size();

  res.updated_map = synthesizePersistentMap(job);
  if (!res.updated_map || res.updated_map->empty()) {
    return res;
  }

  if (!ensureOriginBackup(job.map_dir, job.map_name)) {
    return res;
  }

  const std::string target_pcd = (fs::path(job.map_dir) / job.map_name).string();
  if (!writePcdAtomic(job.map_dir, target_pcd, *res.updated_map)) {
    LOG(ERROR) << " ---> [MapUpdate] failed to atomically write updated PCD: " << target_pcd;
    return res;
  }

  if (job.rebuild_reloc_targets) {
    res.ndt.reset(new pcl::NormalDistributionsTransform<PointType, PointType>());
    res.ndt->setTransformationEpsilon(1e-4);
    res.ndt->setMaximumIterations(25);
    res.ndt->setResolution(1.0);
    res.ndt->setInputTarget(res.updated_map);

    res.genz = std::make_unique<GenzRefiner>(job.options.genz);
    res.fc = std::make_unique<FcGate>(job.options.fc);
    if (!res.genz->SetTarget(res.updated_map) || !res.fc->SetTarget(res.updated_map)) {
      LOG(ERROR) << " ---> [MapUpdate] failed to rebuild GenZ/FC targets";
      return res;
    }

    if (job.options.cold_start.enable) {
      res.kiss = std::make_unique<KissGlobalMatcher>(job.options.kiss);
      if (!res.kiss->SetTarget(res.updated_map)) {
        res.kiss.reset();
      }
    }
  }

  res.pgm_updated = syncPgmMap(job, *res.updated_map, res.map_yaml_path);
  res.success = true;

  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t0).count();
  LOG(INFO) << " ---> [MapUpdate] commit gen=" << job.generation << " finished in " << ms
            << " ms (points=" << res.updated_map->size()
            << ", pgm_updated=" << (res.pgm_updated ? "true" : "false") << ")";
  return res;
}

bool DynamicMapUpdater::resetToOrigin(CloudPtr& restored_map, std::string& out_yaml_path,
                                      bool& pgm_restored, std::string& message) {
  pgm_restored = false;
  out_yaml_path.clear();

  // Cancel/supersede any running background commit and wait for it to finish cleanly.
  ++generation_;
  joinCommitThread();
  {
    std::lock_guard<std::mutex> lock(result_mutex_);
    pending_result_.reset();
  }

  const fs::path patch_dir = patchDirPath(map_dir_);
  const fs::path orig_pcd = originPcdPath(map_dir_, map_name_);
  const fs::path target_pcd = fs::path(map_dir_) / map_name_;
  std::error_code ec;

  if (fs::exists(orig_pcd, ec)) {
    if (!copyFileAtomic(orig_pcd, target_pcd, patch_dir)) {
      message = "Failed to copy origin PCD backup back to " + target_pcd.string();
      return false;
    }

    CloudPtr loaded(new PointCloudType());
    if (pcl::io::loadPCDFile<PointType>(target_pcd.string(), *loaded) == -1) {
      message = "Failed to reload restored origin PCD " + target_pcd.string();
      return false;
    }
    std::vector<int> nan_idx;
    loaded->is_dense = false;
    pcl::removeNaNFromPointCloud(*loaded, *loaded, nan_idx);
    if (loaded->empty()) {
      message = "Restored origin PCD is empty";
      return false;
    }

    const fs::path orig_pgm = patch_dir / "map.origin.pgm";
    const fs::path orig_yaml = patch_dir / "map.origin.yaml";
    const fs::path target_pgm = fs::path(map_dir_) / "map.pgm";
    const fs::path target_yaml = fs::path(map_dir_) / "map.yaml";
    if (fs::exists(orig_pgm, ec)) {
      pgm_restored = copyFileAtomic(orig_pgm, target_pgm, patch_dir);
    }
    if (fs::exists(orig_yaml, ec)) {
      if (copyFileAtomic(orig_yaml, target_yaml, patch_dir)) {
        out_yaml_path = target_yaml.string();
      }
    }

    setBaseMap(loaded);
    restored_map = loaded;
    message = "Restored global.pcd" + std::string(pgm_restored ? " and map.pgm" : "") +
              " from patch/*.origin.*";
  } else {
    if (!base_map_ || base_map_->empty()) {
      message = "Base map is not loaded";
      return false;
    }
    restored_map.reset(new PointCloudType(*base_map_));
    message = "No disk commit yet; cleared in-memory L1 patches to factory base map";
  }

  added_voxels_.clear();
  removed_subvoxels_.clear();
  pending_add_.clear();
  carve_miss_.clear();
  committed_added_keys_.clear();
  committed_removed_keys_.clear();
  inflight_added_snapshot_.clear();
  inflight_removed_snapshot_.clear();
  has_last_kf_ = false;
  dirty_since_time_ = -1.0;

  return true;
}

CloudPtr DynamicMapUpdater::buildPatchDebugCloud() const {
  CloudPtr cloud(new PointCloudType());
  cloud->points.reserve(added_voxels_.size() + removed_subvoxels_.size());

  for (const auto& [fk, vox] : added_voxels_) {
    PointType p;
    p.x = static_cast<float>(vox.centroid.x());
    p.y = static_cast<float>(vox.centroid.y());
    p.z = static_cast<float>(vox.centroid.z());
    p.intensity = 1.0f;
    cloud->points.push_back(p);
  }
  for (const auto& fk : removed_subvoxels_) {
    if (added_voxels_.count(fk)) continue;
    V3 pos = (fk.cast<scalar>() + V3::Constant(scalar(0.5))) * sub_resolution_;
    if (auto it = l0_centroids_.find(fk); it != l0_centroids_.end()) {
      pos = it->second;
    }
    PointType p;
    p.x = static_cast<float>(pos.x());
    p.y = static_cast<float>(pos.y());
    p.z = static_cast<float>(pos.z());
    p.intensity = -1.0f;
    cloud->points.push_back(p);
  }

  cloud->width = static_cast<uint32_t>(cloud->points.size());
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

}  // namespace LI2Sup
