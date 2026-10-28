
#include "lio/super_lio_reloc.h"

#include <sys/resource.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>

#include <chrono>
#include <limits>


using namespace BASIC;

namespace LI2Sup{

inline bool calc_plane_coeff(const int N, const std::array<V3, 5>& points, std::array<double, 4>& abcd)
{
  Eigen::Vector3d normvec;
  if (N == 5) {
    Eigen::Matrix<double, 5, 3> A;
    Eigen::Matrix<double, 5, 1> b;
    for (int j = 0; j < 5; j++) {
      A.row(j) = points[j].cast<double>();
      b(j) = -1.0;
    }
    normvec = A.colPivHouseholderQr().solve(b);
  }
  else {
    Eigen::Matrix<double, 4, 3> A;
    Eigen::Matrix<double, 4, 1> b;

    for (int j = 0; j < N; j++) {
      A.row(j) = points[j].cast<double>();
      b(j) = -1.0;
    }
    normvec = A.colPivHouseholderQr().solve(b);
  }

  double n = normvec.norm();
  if (n < 1e-6f) return false;

  abcd[3] = 1.0 / n;
  normvec *= abcd[3];
  abcd[0] = normvec[0];
  abcd[1] = normvec[1];
  abcd[2] = normvec[2];
  
  for (int i = 0; i < N; ++i) {
    const V3& p = points[i];
    auto dist = abcd[0] * p(0) + abcd[1] * p(1) + abcd[2] * p(2) + abcd[3];
    if (std::abs(dist) > 0.1) return false;
  }
  return true;
}


inline bool compute_error(
  const std::array<double, 4>& abcd, const V3& point, 
  const float length, scalar& error)
{
  error = abcd[0] * point[0] + abcd[1] * point[1] + abcd[2] * point[2] + abcd[3];
  return length > 81 * error * error;
}


void SuperLIOReLoc::init(){
  ivox_.reset(new OctVoxMapType(OctVoxMapType::Options{g_ivox_resolution, g_ivox_capacity}));
  kf_.reset(new ESKF());
  data_wrapper_->setESKF(kf_);
  
  scan_undistort_full_.reset(new PointCloudType());
  ds_undistort_.reset(new PointCloudType());
  world_pc_.reset(new PointCloudType());
  ds_world_.reset(new PointCloudType());
  point_map_.reset(new PointCloudType());
  init_obs_data_.reset(new PointCloudType());
  
  points_world_v3_.reserve(21000);
  abcd_vec_.resize(20000);
  effect_knn_idxs_.resize(20000);
  voxel_grid_fliter_.setLeafSize(g_voxel_fliter_size);

  LOG(INFO) << GREEN << " ---> [SuperLIO]: initialized." << RESET;

  auto start_time = std::chrono::high_resolution_clock::now();
  const bool map_loaded = SuperLIOReLoc::map_init();
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  if (!map_loaded) {
    LOG(ERROR) << RED << " ---> [SuperLIO]: Map init failed." << RESET;
    g_flag_run = false;
    return;
  }
  LOG(INFO) << GREEN << " ---> [SuperLIO]: Map init success. Time: "
            << duration.count() << " ms." << RESET;

  if (data_wrapper_->localization()) {
    const auto& options = data_wrapper_->relocOptions();
    start_time = std::chrono::high_resolution_clock::now();
    ndt_.reset(new pcl::NormalDistributionsTransform<PointType, PointType>());
    ndt_->setTransformationEpsilon(1e-4);
    ndt_->setMaximumIterations(25);
    ndt_->setResolution(1.0);
    ndt_->setInputTarget(point_map_);
    genz_ = std::make_unique<GenzRefiner>(options.genz);
    fc_ = std::make_unique<FcGate>(options.fc);
    if (!genz_->SetTarget(point_map_) || !fc_->SetTarget(point_map_)) {
      LOG(ERROR) << RED << " ---> [Reloc]: GenZ/FC target init failed." << RESET;
      g_flag_run = false;
      return;
    }
    if (options.cold_start.enable) {
      kiss_ = std::make_unique<KissGlobalMatcher>(options.kiss);
      if (!kiss_->SetTarget(point_map_)) kiss_.reset();
    }
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    LOG(INFO) << GREEN << " ---> [Reloc]: targets ready in " << duration.count()
              << " ms; cold start " << (!options.cold_start.enable ? "off" :
                                        options.origin_prior.enable ? (kiss_ ? "origin prior + KISS" : "origin prior")
                                                                    : (kiss_ ? "KISS" : "off"))
              << RESET;
  }

  state_fn_ = &SuperLIOReLoc::stateWaitKFInit;
  // The localization map is already loaded into ivox_.  Mark the initial
  // activation as consumed so SuperLIO::process() does not immediately call
  // Reset() and erase the map on its first timer tick.
  was_active_ = data_wrapper_->is_active();
}


void SuperLIOReLoc::Reset() {
  auto fixed_map = point_map_;
  auto fixed_index = ivox_;
  point_map_.reset(new PointCloudType());
  SuperLIO::Reset();
  point_map_ = fixed_map;
  ivox_ = fixed_index;
  init_frame_count_ = 0;
  init_obs_data_->clear();
  has_init_pose_ = false;
  cold_start_attempts_ = 0;
  moving_windows_ = 0;
  zupt_active_ = false;
  hold_set_ = false;
  verify_left_ = 0;
  verify_failed_ = false;
  if (data_wrapper_->localization()) data_wrapper_->setLocState(5);
}

bool SuperLIOReLoc::coldStartArmed() const {
  const auto& options = data_wrapper_->relocOptions();
  const int max_attempts = options.cold_start.max_attempts;
  return options.cold_start.enable && (kiss_ || options.origin_prior.enable) &&
         (max_attempts == 0 || cold_start_attempts_ < max_attempts);
}

void SuperLIOReLoc::process() {
  if (data_wrapper_->localization() && map_loaded_) {
    VV3 vehicle;
    scalar yaw = 0;
    if (data_wrapper_->takeVehicleUpdate(vehicle, yaw)) {
      if (vehicle.empty())
        LOG(INFO) << YELLOW << " ---> [Vehicle] cloud expired, removed from the map" << RESET;
      setVehicle(std::move(vehicle), yaw);
    }
    if (data_wrapper_->takeVehicleLayerClear() && layer_points_ > 0) {
      LOG(INFO) << YELLOW << " ---> [Vehicle] map update: " << layer_points_ << " cells dropped" << RESET;
      rebuildFixedIndex(vehicle_points_);
      layer_points_ = 0;
    }
  }
  if (data_wrapper_->localization() && data_wrapper_->is_active()) {
    if (flg_get_init_guess_ && kf_->init_) Reset();
    // Once tracking is lost, require a validated new initial pose.
    if (data_wrapper_->locState() == 4 && kf_->init_) Reset();
    // A pose that failed its verification window is discarded the same way.
    if (verify_failed_) Reset();
  }
  SuperLIO::process();
}

SuperLIOReLoc::ImuStats SuperLIOReLoc::imuStats(int count, const V3& mean_gyro, const V3& mean_acce,
                                                const V3& m2_gyro, const V3& m2_acce) {
  ImuStats stats;
  if (count < 2 || !mean_acce.allFinite() || mean_acce.norm() < 1e-3) return stats;
  stats.gyro_mean = mean_gyro.norm();
  stats.gyro_std = std::sqrt(m2_gyro.sum() / count);
  // Raw accelerometer units vary by driver (g on Livox): scale to m/s^2.
  stats.acc_std = std::sqrt(m2_acce.sum() / count) * g_gravity_norm / mean_acce.norm();
  stats.valid = std::isfinite(stats.gyro_mean) && std::isfinite(stats.gyro_std) && std::isfinite(stats.acc_std);
  return stats;
}

bool SuperLIOReLoc::imuStill(const ImuStats& stats) const {
  const auto& still = data_wrapper_->relocOptions().still;
  return stats.valid && stats.gyro_mean <= still.max_gyro_mean && stats.gyro_std <= still.max_gyro_std &&
         stats.acc_std <= still.max_acc_std;
}

static const char* cmdStateName(ROSWrapper::CmdState state) {
  return state == ROSWrapper::CmdState::kIdle ? "idle" : state == ROSWrapper::CmdState::kMoving ? "moving" : "unknown";
}

bool SuperLIOReLoc::initWindowStill(int imu_count, const V3& mean_gyro, const V3& mean_acce) {
  const auto& options = data_wrapper_->relocOptions();
  if (!options.still_gate) return true;
  const auto stats = imuStats(imu_count, mean_gyro, mean_acce, init_m2_gyro_, init_m2_acce_);
  // Let the existing gravity check reject a window without usable IMU data.
  if (!stats.valid) return true;
  const auto cmd = data_wrapper_->cmdState();
  const bool still_ok = cmd != ROSWrapper::CmdState::kMoving && imuStill(stats);
  const auto& still = options.still;
  if (still_ok) {
    LOG(INFO) << GREEN << " ---> [Reloc] init window still: cmd " << cmdStateName(cmd) << ", gyro_mean="
              << stats.gyro_mean << " gyro_std=" << stats.gyro_std << " rad/s, acc_std=" << stats.acc_std
              << " m/s^2" << RESET;
    moving_windows_ = 0;
  } else if (moving_windows_++ % 10 == 0) {
    LOG(WARNING) << RED << " ---> [Reloc] device moving, init window discarded (" << moving_windows_
                 << " so far): cmd " << cmdStateName(cmd) << ", gyro_mean=" << stats.gyro_mean << " (max "
                 << still.max_gyro_mean << "), gyro_std=" << stats.gyro_std << " (max " << still.max_gyro_std
                 << ") rad/s, acc_std=" << stats.acc_std << " (max " << still.max_acc_std << ") m/s^2" << RESET;
  }
  return still_ok;
}

void SuperLIOReLoc::onMeasure() {
  const auto& options = data_wrapper_->relocOptions();
  frame_still_ = false;
  if (!data_wrapper_->localization() || !kf_->init_) {
    zupt_active_ = false;
    hold_set_ = false;
    return;
  }
  int count = 0;
  V3 mean_gyro = V3::Zero(), mean_acce = V3::Zero(), m2_gyro = V3::Zero(), m2_acce = V3::Zero();
  for (const auto& imu : measures_.imu) {
    ++count;
    const V3 d_gyro = imu.gyr - mean_gyro;
    const V3 d_acce = imu.acc - mean_acce;
    mean_gyro += d_gyro / count;
    mean_acce += d_acce / count;
    m2_gyro += d_gyro.cwiseProduct(imu.gyr - mean_gyro);
    m2_acce += d_acce.cwiseProduct(imu.acc - mean_acce);
  }
  const auto stats = imuStats(count, mean_gyro, mean_acce, m2_gyro, m2_acce);
  // Unlike the init gate, ZUPT never runs on IMU alone: it needs an idle command.
  const auto cmd = data_wrapper_->cmdState();
  frame_still_ = cmd == ROSWrapper::CmdState::kIdle && imuStill(stats);
  const bool still = options.zupt.enable && frame_still_;
  if (still) kf_->SetZeroVelocityObs(options.zupt.velocity_sigma);
  if (still != zupt_active_) {
    zupt_active_ = still;
    const auto v = kf_->GetNavState().v;
    LOG(INFO) << (still ? GREEN : YELLOW) << " ---> [ZUPT] " << (still ? "on" : "off") << ": cmd "
              << cmdStateName(cmd) << ", gyro_mean=" << stats.gyro_mean << " gyro_std=" << stats.gyro_std
              << " rad/s, acc_std=" << stats.acc_std << " m/s^2, |v|=" << v.norm() << " m/s" << RESET;
  }
  holdPose(cmd == ROSWrapper::CmdState::kIdle);
}

void SuperLIOReLoc::holdPose(bool cmd_idle) {
  const auto& hold = data_wrapper_->relocOptions().hold;
  // Only on a verified, tracked pose: holding an unverified one would mask a
  // wrong acceptance.
  if (!hold.enable || !cmd_idle || data_wrapper_->locState() != 2) {
    if (hold_set_) {
      LOG(INFO) << YELLOW << " ---> [Hold] released after " << hold_scans_ << " scans; max offset from the anchor "
                << hold_max_offset_ << " m, " << hold_max_rot_deg_ << " deg" << RESET;
    }
    hold_set_ = false;
    return;
  }
  // IMU-vetoed scans (arm, door, someone leaning on it) skip the hold, anchor kept.
  if (!frame_still_) return;
  const auto state = kf_->GetNavState();
  const scalar offset = (state.p - hold_p_).norm();
  const scalar rot_deg = (hold_R_.inverse() * state.R).log_vee().norm() * 180.0 / M_PI;
  if (hold_set_ && (offset > hold.max_offset || rot_deg > hold.max_rot_deg)) {
    LOG(WARNING) << YELLOW << " ---> [Hold] " << offset << " m, " << rot_deg
                 << " deg from the anchor: the robot moved, anchoring again" << RESET;
    hold_set_ = false;
  }
  if (!hold_set_) {
    hold_set_ = true;
    hold_R_ = state.R;
    hold_p_ = state.p;
    hold_scans_ = 0;
    hold_max_offset_ = hold_max_rot_deg_ = 0;
    LOG(INFO) << GREEN << " ---> [Hold] anchored at p=[" << state.p.transpose() << "]" << RESET;
  } else {
    hold_max_offset_ = std::max(hold_max_offset_, offset);
    hold_max_rot_deg_ = std::max(hold_max_rot_deg_, rot_deg);
  }
  ++hold_scans_;
  kf_->SetPoseObs(hold_R_, hold_p_, hold.sigma_rot_deg * M_PI / 180.0, hold.sigma_pos);
}


bool SuperLIOReLoc::verifyScan(const NavState& state, std::size_t matched) {
  const auto& verify = data_wrapper_->relocOptions().verify;
  const std::size_t scan_points = ds_undistort_->size();
  const double ratio = scan_points > 0 ? static_cast<double>(matched) / scan_points : 0.0;
  // Drift only means something while the robot has stood still since acceptance.
  verify_all_still_ = verify_all_still_ && frame_still_;
  const double drift = (state.p - verify_anchor_).norm();
  const double speed = state.v.norm();
  verify_min_ratio_ = std::min(verify_min_ratio_, ratio);
  verify_max_drift_ = std::max(verify_max_drift_, drift);
  verify_max_speed_ = std::max(verify_max_speed_, speed);
  LOG(INFO) << " ---> [Verify] scan " << verify.frames - verify_left_ + 1 << "/" << verify.frames << ": match "
            << matched << "/" << scan_points << " (" << ratio << "), drift " << drift << " m"
            << (verify_all_still_ ? " still" : "") << ", |v| " << speed << " m/s";
  const char* reason = ratio < verify.min_match_ratio                          ? "match ratio"
                       : verify_all_still_ && drift > verify.max_still_drift   ? "drift while still"
                       : speed > verify.max_speed                              ? "speed"
                                                                               : nullptr;
  if (!reason) return true;
  LOG(WARNING) << RED << " ---> [Verify] failed (" << reason << "): match " << ratio << " (min "
               << verify.min_match_ratio << "), drift " << drift << " m (max " << verify.max_still_drift
               << (verify_all_still_ ? ", still" : ", moving: not checked") << "), |v| " << speed << " m/s (max "
               << verify.max_speed << "); pose discarded, initializing again" << RESET;
  return false;
}


bool SuperLIOReLoc::map_init(){
  if(map_loaded_) return true;

  std::string map_name = g_save_map_dir + "/" + g_map_name;
  if(pcl::io::loadPCDFile<PointType>(map_name, *point_map_) == -1){
    LOG(ERROR) << RED << " ---> Load map failed. File: " << map_name << RESET;
    return false;
  }

  std::vector<int> useless_indices;
  point_map_->is_dense = false;
  pcl::removeNaNFromPointCloud(*point_map_, *point_map_, useless_indices);

  if (point_map_->empty()) return false;

  if (data_wrapper_->localization()) {
    rebuildFixedIndex({});
  } else {
    VV3 point_map_v3;
    point_map_v3.reserve(point_map_->size());
    for (const auto& point : *point_map_) point_map_v3.emplace_back(point.x, point.y, point.z);
    ivox_->insert(point_map_v3);
  }

  LOG(INFO) << GREEN << " ---> Load map success. File: " << map_name << RESET;
  LOG(INFO) << GREEN << " ---> Map size: " << point_map_->size() << RESET;
  ivox_->printInfo();

  map_loaded_ = true;

  data_wrapper_->set_global_map(point_map_);
  data_wrapper_->set_initial_data(re_init_pose_, flg_get_init_guess_);
  return true;
}


void SuperLIOReLoc::rebuildFixedIndex(const VV3& extra) {
  const auto start = std::chrono::steady_clock::now();
  VV3 points;
  points.reserve(point_map_->size() + extra.size());
  for (const auto& point : *point_map_) points.emplace_back(point.x, point.y, point.z);
  points.insert(points.end(), extra.begin(), extra.end());
  // No LRU eviction in a fixed map, even with a small configured capacity.
  // Headroom for the vehicle-side map update, which never evicts.
  const auto capacity = std::max(static_cast<std::size_t>(g_ivox_capacity), 2 * points.size() + 100000);
  ivox_.reset(new OctVoxMapType(OctVoxMapType::Options{g_ivox_resolution, capacity}));
  ivox_->insert(points);
  if (extra.empty()) return;
  V3 lo = extra.front(), hi = extra.front();
  for (const auto& p : extra) {
    lo = lo.cwiseMin(p);
    hi = hi.cwiseMax(p);
  }
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
  LOG(INFO) << GREEN << " ---> [Vehicle] " << extra.size() << " points added to the map, x[" << lo.x() << ", "
            << hi.x() << "] y[" << lo.y() << ", " << hi.y() << "] z[" << lo.z() << ", " << hi.z() << "] ("
            << ms.count() << " ms)" << RESET;
}


bool SuperLIOReLoc::kf_init(){
  const int need_init_frames = 10;
  int& imu_cout = init_imu_count_;
  int& init_frame_count = init_frame_count_;
  V3& mean_gyro = init_mean_gyro_;
  V3& mean_acce = init_mean_acce_;

  /// get init guess from ROS topic.
  if(flg_get_init_guess_){
    imu_cout = 0;
    init_frame_count = 0;
    init_obs_data_->clear();
    mean_gyro = V3::Zero();
    mean_acce = V3::Zero();
    flg_get_init_guess_ = false;
    has_init_pose_ = true;
    if (data_wrapper_->localization()) data_wrapper_->setLocState(1);
    return false;
  }

  if (data_wrapper_->localization() && !has_init_pose_) {
    if (!coldStartArmed()) return false;
    if (data_wrapper_->locState() != 1) {
      LOG(INFO) << YELLOW << " ---> [Reloc] cold start: keep still, accumulating" << RESET;
      data_wrapper_->setLocState(1);
    }
  }

  CloudPtr point_cloud_pcl = CloudPtr(new PointCloudType());
  for(std::size_t i = 0; i < measures_.lidar.pc->size(); i++){
    auto p = measures_.lidar.pc->points[i];
    PointType point;
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    point.intensity = p.intensity;
    point_cloud_pcl->points.push_back(point);
  }

  if(init_frame_count < need_init_frames){
    *init_obs_data_ += *point_cloud_pcl;
  }
  init_frame_count++;

  for(auto& imu: measures_.imu){
    imu_cout ++;
    if (imu_cout == 1) {
      init_m2_gyro_.setZero();
      init_m2_acce_.setZero();
    }
    const V3 d_gyro = imu.gyr - mean_gyro;
    const V3 d_acce = imu.acc - mean_acce;
    mean_gyro += d_gyro / imu_cout;
    mean_acce += d_acce / imu_cout;
    init_m2_gyro_ += d_gyro.cwiseProduct(imu.gyr - mean_gyro);
    init_m2_acce_ += d_acce.cwiseProduct(imu.acc - mean_acce);
  }

  if(imu_cout < 20){
    return false;
  }

  if(init_frame_count < need_init_frames){
    return false;
  }

  // The filter starts at zero velocity: never initialize from a moving window.
  if (data_wrapper_->localization() && !initWindowStill(imu_cout, mean_gyro, mean_acce)) {
    imu_cout = 0;
    init_frame_count = 0;
    init_obs_data_->clear();
    mean_gyro = V3::Zero();
    mean_acce = V3::Zero();
    return false;
  }

  LOG(INFO) << YELLOW << " ---> INIT start... obs_data size: " << init_obs_data_->size() << " target size: " << point_map_->size() << RESET;

  if (!mean_acce.allFinite() || mean_acce.norm() < 1e-3) return false;

  V3 gravity = - mean_acce * g_gravity_norm / mean_acce.norm();
  V3 ref_gravity(0, 0, - g_gravity_norm);
  M3 init_rot = Quat::FromTwoVectors(gravity, ref_gravity).toRotationMatrix();
  V3 n = init_rot.col(0);
  double yaw = atan2(n(1), n(0));

  M3 R_yaw_inv = Eigen::AngleAxis<scalar>(-yaw, V3::UnitZ()).toRotationMatrix(); 
  M3 rot = R_yaw_inv * init_rot;

  M3 init_guess_R_ = re_init_pose_.R_ * rot;
  V3 init_guess_t_ = re_init_pose_.t_;
  M4 init_guess_T = M4::Identity();
  init_guess_T.block<3, 3>(0, 0) = init_guess_R_;
  init_guess_T.block<3, 1>(0, 3) = init_guess_t_;


  if (data_wrapper_->localization()) {
    const bool manual = has_init_pose_;
    const bool aligned = alignToMap(manual, rot, init_guess_T);
    data_wrapper_->resetInputWatchdog();
    if (!aligned) {
      imu_cout = 0;
      init_frame_count = 0;
      init_obs_data_->clear();
      mean_gyro = V3::Zero();
      mean_acce = V3::Zero();
      if (!manual && !coldStartArmed()) {
        LOG(WARNING) << RED << " ---> [Reloc] cold start gave up after " << cold_start_attempts_
                     << " attempts; waiting for /initialpose" << RESET;
        data_wrapper_->setLocState(5);
      }
      return false;
    }
  } else {
    pcl::PointCloud<pcl::PointXYZI>::Ptr tmp_src(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*init_obs_data_, *tmp_src, g_lidar_imu.matrix().cast<float>());

    pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
    ndt.setTransformationEpsilon(1e-4);
    ndt.setEuclideanFitnessEpsilon(1e-4);
    ndt.setMaximumIterations(25);
    ndt.setResolution(1.0);
    ndt.setInputTarget(point_map_);

    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
    icp.setMaxCorrespondenceDistance(4.0);
    icp.setMaximumIterations(40);
    icp.setTransformationEpsilon(1e-4);
    icp.setEuclideanFitnessEpsilon(1e-4);
    icp.setRANSACIterations(0);
    icp.setInputTarget(point_map_);

    ndt.setInputSource(tmp_src);
    icp.setInputSource(tmp_src);

    pcl::PointCloud<pcl::PointXYZI>::Ptr unused_result(new pcl::PointCloud<pcl::PointXYZI>());
    ndt.align(*unused_result, init_guess_T.matrix().cast<float>());
    icp.align(*unused_result, ndt.getFinalTransformation());

    if (!ndt.hasConverged() || !icp.hasConverged() ||
        !icp.getFinalTransformation().allFinite() || !std::isfinite(icp.getFitnessScore()) ||
        icp.getFitnessScore() > 1.5)
    // if (icp.hasConverged() == false)
    {
      /// reset init state.
      imu_cout = 0;
      init_frame_count = 0;
      init_obs_data_->clear();
      mean_gyro = V3::Zero();
      mean_acce = V3::Zero();
      LOG(INFO) << RED << " ---> Global ICP Converged Fail! FitnessScore: " << icp.getFitnessScore() << RESET;
      return false;
    } else{
      init_guess_T = icp.getFinalTransformation().cast<scalar>();
      LOG(INFO) << GREEN << " ---> Global ICP Converged Succeed! FitnessScore: " << icp.getFitnessScore() << RESET;
    }
  }

  LOG(INFO) << GREEN << "\n" << init_guess_T << RESET;

  ESKF::Options options;
  options.gyro_var_ = g_imu_ng;
  options.acce_var_ = g_imu_na;
  options.bias_gyro_var_ = g_imu_nbg;
  options.bias_acce_var_ = g_imu_nba;
  options.num_iterations_ = g_kf_max_iterations;
  options.quit_eps_ = g_kf_quit_eps;

  float imu_scale = g_gravity_norm / mean_acce.norm();
  kf_->SetInitialConditions(options, mean_gyro, V3::Zero(), imu_scale, ref_gravity);
  auto state = kf_->GetSysState();
  /// The horizontal initial state of the imu in the robot coordinate system.
  state.R = SO3(init_guess_T.block<3, 3>(0, 0));
  state.p = init_guess_T.block<3, 1>(0, 3);
  state.timestamp = -1.0;
  kf_->SetX(state);
  sys_init_pose_ = kf_->GetSE3();

  {
    init_obs_data_->clear();
    data_wrapper_->set_initial_data(re_init_pose_, flg_get_init_guess_, true);
  }

  if (data_wrapper_->localization()) {
    // Not GOOD yet: the first scans after the alignment stall must confirm the pose.
    const auto& verify = data_wrapper_->relocOptions().verify;
    verify_left_ = verify.frames;
    verify_anchor_ = state.p;
    verify_all_still_ = true;
    verify_min_ratio_ = 1.0;
    verify_max_drift_ = verify_max_speed_ = 0.0;
    if (verify_left_ > 0)
      LOG(INFO) << YELLOW << " ---> [Verify] tracking " << verify_left_ << " scans before reporting GOOD" << RESET;
    data_wrapper_->setLocState(verify_left_ > 0 ? 1 : 2);
  }
  return true;
}


bool SuperLIOReLoc::alignToMap(bool manual, const M3& level_rot, M4& T_map_imu) {
  const auto& options = data_wrapper_->relocOptions();
  const auto start = std::chrono::steady_clock::now();
  auto elapsed = [&start]() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
  };

  CloudPtr src_imu(new PointCloudType());
  pcl::transformPointCloud(*init_obs_data_, *src_imu, g_lidar_imu.matrix().cast<float>());
  CloudPtr query(new PointCloudType());
  {
    const float leaf = static_cast<float>(options.genz.source_voxel);
    pcl::VoxelGrid<PointType> grid;
    grid.setLeafSize(leaf, leaf, leaf);
    grid.setInputCloud(src_imu);
    grid.filter(*query);
  }

  Eigen::Matrix4d T;
  Selection selection = Selection::kNone;
  if (manual) {
    // /initialpose denotes map<-lidar, while the query is in IMU.
    selection = selectCandidate(
        query, {(re_init_pose_ * g_lidar_imu.inverse()).matrix().cast<double>()},
        options.fc.max_delta_trans_manual, options.fc.max_delta_rot_deg_manual, 1, "initialpose", T);
  } else {
    ++cold_start_attempts_;
    Eigen::Matrix4d T_level_imu = Eigen::Matrix4d::Identity();
    T_level_imu.block<3, 3>(0, 0) = level_rot.cast<double>();

    const auto& origin = options.origin_prior;
    if (origin.enable) {
      std::vector<Eigen::Matrix4d> guesses;
      for (double yaw = 0; yaw < 360.0 - 1e-6; yaw += origin.yaw_step_deg) {
        Eigen::Matrix4d T_map_level = Eigen::Matrix4d::Identity();
        T_map_level.block<3, 3>(0, 0) =
            Eigen::AngleAxisd(yaw * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        T_map_level.block<3, 1>(0, 3) = Eigen::Vector3d(origin.x, origin.y, origin.z);
        guesses.push_back(T_map_level * T_level_imu);
      }
      selection = selectCandidate(query, guesses, origin.max_delta_trans,
                                  origin.yaw_step_deg / 2 + 10.0, 1, "origin", T);
    }
    // An ambiguous scene stays ambiguous for KISS: retry with the next window.
    if (selection == Selection::kNone && kiss_) {
      CloudPtr level(new PointCloudType());
      pcl::transformPointCloud(*src_imu, *level, T_level_imu.cast<float>());
      std::vector<Eigen::Matrix4d> guesses;
      for (const auto& kiss : kiss_->Match(level))
        if (kiss.valid) guesses.push_back(kiss.T_map_query * T_level_imu);
      selection = selectCandidate(query, guesses, options.fc.max_delta_trans_kiss,
                                  options.fc.max_delta_rot_deg_kiss,
                                  static_cast<std::size_t>(options.kiss.min_agreeing_candidates),
                                  "kiss", T);
    }
  }

  if (selection != Selection::kAccepted) {
    LOG(WARNING) << RED << " ---> [FC-Init] " << (manual ? "initialpose" : "cold start attempt ")
                 << (manual ? std::string() : std::to_string(cold_start_attempts_))
                 << " failed (" << elapsed() << " ms)" << RESET;
    return false;
  }
  LOG(INFO) << GREEN << " ---> [FC-Init] accepted (" << elapsed() << " ms)" << RESET;
  T_map_imu = T.cast<scalar>();
  return true;
}


SuperLIOReLoc::Selection SuperLIOReLoc::selectCandidate(
    const CloudPtr& query, const std::vector<Eigen::Matrix4d>& guesses, double max_delta_trans,
    double max_delta_rot_deg, std::size_t min_passed, const char* label, Eigen::Matrix4d& T_map_imu) {
  const auto& options = data_wrapper_->relocOptions();
  struct Passed {
    Eigen::Matrix4d T;
    FcGate::Report report;
  };
  std::vector<Passed> passed;
  for (std::size_t i = 0; i < guesses.size(); ++i) {
    // NDT widens the basin before GenZ-ICP: hand-placed, origin and KISS
    // guesses can be a metre or more and 10-20 degrees off.
    Eigen::Matrix4d refined = guesses[i];
    CloudPtr unused(new PointCloudType());
    ndt_->setInputSource(query);
    ndt_->align(*unused, guesses[i].cast<float>());
    const Eigen::Matrix4d coarse = ndt_->getFinalTransformation().cast<double>();
    if (ndt_->hasConverged() && coarse.allFinite()) refined = coarse;
    FcGate::Report report;
    if (genz_->Refine(query, refined, refined))
      report = fc_->Check(query, guesses[i], refined, max_delta_trans, max_delta_rot_deg);
    const Eigen::Vector3d t = refined.block<3, 1>(0, 3);
    LOG(INFO) << (report.pass ? GREEN : RED) << " ---> [FC-Init] " << label << " candidate "
              << i + 1 << "/" << guesses.size() << (report.pass ? " PASS" : " REJECT")
              << " t=[" << t.transpose() << "]: " << report.Describe() << RESET;
    if (report.pass) passed.push_back({refined, report});
  }
  if (passed.empty()) return Selection::kNone;

  const auto best = std::max_element(passed.begin(), passed.end(), [](const auto& a, const auto& b) {
    return a.report.wall_inlier_ratio < b.report.wall_inlier_ratio;
  });
  for (const auto& other : passed) {
    const double dt = (other.T.block<3, 1>(0, 3) - best->T.block<3, 1>(0, 3)).norm();
    const double dr = Eigen::AngleAxisd(Eigen::Matrix3d(
        best->T.block<3, 3>(0, 0).transpose() * other.T.block<3, 3>(0, 0))).angle() * 180.0 / M_PI;
    if (dt > options.fc.ambiguity_trans || dr > options.fc.ambiguity_rot_deg) {
      LOG(WARNING) << RED << " ---> [FC-Init] " << label << ": ambiguous, passing candidates "
                   << dt << " m / " << dr << " deg apart" << RESET;
      return Selection::kAmbiguous;
    }
  }
  if (passed.size() < min_passed) {
    LOG(WARNING) << RED << " ---> [FC-Init] " << label << ": " << passed.size()
                 << " passing candidate(s), need " << min_passed << " that agree" << RESET;
    return Selection::kNone;
  }
  T_map_imu = best->T;
  return Selection::kAccepted;
}


void SuperLIOReLoc::setVehicle(VV3 points, scalar yaw) {
  vehicle_points_ = std::move(points);
  rebuildFixedIndex(vehicle_points_);  // also drops a previous vehicle's map update
  layer_points_ = 0;
  if (vehicle_points_.empty()) return;
  vehicle_cos_ = std::cos(yaw);
  vehicle_sin_ = std::sin(yaw);
  region_lo_ = V3::Constant(std::numeric_limits<scalar>::max());
  region_hi_ = V3::Constant(std::numeric_limits<scalar>::lowest());
  for (const auto& p : vehicle_points_) {
    const V3 box(vehicle_cos_ * p.x() + vehicle_sin_ * p.y(), -vehicle_sin_ * p.x() + vehicle_cos_ * p.y(), p.z());
    region_lo_ = region_lo_.cwiseMin(box);
    region_hi_ = region_hi_.cwiseMax(box);
  }
  // Doors swing out sideways and stand taller than the modeled band.
  const scalar margin = data_wrapper_->relocOptions().vehicle.update.margin;
  region_lo_ -= V3(margin, margin, 0.1);
  region_hi_ += V3(margin, margin, 0.5);
}

bool SuperLIOReLoc::inVehicleRegion(const V3& p) const {
  const V3 box(vehicle_cos_ * p.x() + vehicle_sin_ * p.y(), -vehicle_sin_ * p.x() + vehicle_cos_ * p.y(), p.z());
  return (box.array() >= region_lo_.array()).all() && (box.array() <= region_hi_.array()).all();
}

void SuperLIOReLoc::updateVehicleLayer() {
  const auto& update = data_wrapper_->relocOptions().vehicle.update;
  const bool collecting = update.enable && !vehicle_points_.empty() && data_wrapper_->vehicleCollecting();
  if (collecting != layer_collecting_) {
    layer_collecting_ = collecting;
    LOG(INFO) << GREEN << " ---> [Vehicle] map update " << (collecting ? "started" : "stopped") << ": "
              << layer_points_ << " cells so far" << RESET;
  }
  // Only from a tracked pose: a pose that is off would write the error into the
  // map. Standing still adds no new view, only whatever the pose drifts.
  if (!collecting || data_wrapper_->locState() != 2 || frame_still_) return;
  const SE3 pose = kf_->GetSE3();
  const scalar min_range2 = update.min_range * update.min_range;
  VV3 points;
  points.reserve(ds_undistort_->size());
  for (std::size_t i = 0; i < ds_undistort_->size(); ++i) {
    const V3& body = points_body_v3_[i];
    if (body.squaredNorm() < min_range2) continue;
    const V3 world = pose.R_ * body + pose.t_;
    if (inVehicleRegion(world)) points.push_back(world);
  }
  layer_points_ += ivox_->insertNew(points);
}

void SuperLIOReLoc::UpdateMap() {
  if (data_wrapper_->localization()) {
    updateVehicleLayer();
    return;
  }
  if(!g_update_map) return;
  
  static int __update_delay = 100;
  if(__update_delay > 0){
    __update_delay--;
    std::cout << "Update map Delay: " << 100 - __update_delay << " %" << std::endl;
    return;
  }

  const size_t ptsize = ds_undistort_->size();
  if (ptsize == 0) return;
  
  last_pose_ = kf_->GetSE3();
  points_world_v3_.resize(ptsize);
  
  const auto R = last_pose_.R_;
  const auto t = last_pose_.t_;
  
  for (size_t i = 0; i < ptsize; ++i) {
    const auto& pt = points_body_v3_[i];
    points_world_v3_[i] = R * pt + t;
  }
  
  ivox_->insert(points_world_v3_);

}


void SuperLIOReLoc::Output() {
  auto state = kf_->GetNavState();
  if (data_wrapper_->localization()) {
    std::size_t matched = 0;
    for (std::size_t i = 0; i < effect_knn_num_; ++i)
      if (effect_mask_[effect_knn_idxs_[i]]) ++matched;
    if (!state.p.allFinite() || !state.R.R_.allFinite() || matched < 10) {
      if (data_wrapper_->locState() != 4) {
        LOG(WARNING) << RED << " ---> [Loc] LOST: "
                     << (state.p.allFinite() && state.R.R_.allFinite() ? "" : "non-finite state, ")
                     << "matched " << matched << "/" << effect_knn_num_ << " points (min 10), p=["
                     << state.p.transpose() << "]" << RESET;
      }
      data_wrapper_->setLocState(4);
      return;
    }
    if (verify_left_ > 0) {
      // Publish nothing until the window passes: downstream must not act on it.
      if (!verifyScan(state, matched)) {
        verify_left_ = 0;
        verify_failed_ = true;
        return;
      }
      if (--verify_left_ > 0) return;
      LOG(INFO) << GREEN << " ---> [Verify] passed: min match " << verify_min_ratio_ << ", max drift "
                << verify_max_drift_ << " m" << (verify_all_still_ ? " (still)" : " (moving)")
                << ", max |v| " << verify_max_speed_ << " m/s" << RESET;
    }
    data_wrapper_->setLocState(2);
  }
  data_wrapper_->pub_odom(state);

  Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
  transformation.block<3, 3>(0, 0) = state.R.R_.cast<float>();
  transformation.block<3, 1>(0, 3) = state.p.cast<float>();

  CloudPtr world_pc(new PointCloudType());
  
  if(g_visual_map){
    static int count = -1;
    count++;
    if(count % g_pub_step != 0){
      return;
    }
    count = 0;
    if(g_visual_dense){
      pcl::transformPointCloud(*scan_undistort_full_, *world_pc, transformation);
      data_wrapper_->pub_cloud_world(world_pc, state.timestamp);
    }else{
      pcl::transformPointCloud(*ds_undistort_, *world_pc, transformation);
      data_wrapper_->pub_cloud_world(world_pc, state.timestamp);
    }
  }

}



} // namespace END.
