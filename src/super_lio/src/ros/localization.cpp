#include "ros/ROSWrapper.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace LI2Sup {
using namespace BASIC;

void ROSWrapper::setupLocalization() {
  const auto map_dir = declare_parameter<std::string>("system.map_path", "");
  const auto pcd = declare_parameter<std::string>("fixed_map.global_pcd", "");
  const auto map_file = map_dir.empty() ? std::filesystem::path(pcd)
                                      : std::filesystem::path(map_dir) / "global.pcd";
  if (map_file.empty() || !std::filesystem::is_regular_file(map_file))
    throw std::runtime_error("Fixed map missing: provide --map_path <directory containing global.pcd> or fixed_map.global_pcd");
  const auto absolute = std::filesystem::absolute(map_file);
  g_save_map_dir = absolute.parent_path().string();
  g_map_name = absolute.filename().string();
  map_frame_ = declare_parameter<std::string>("common.map_frame_id", "map");
  loc_lidar_frame_ = declare_parameter<std::string>("system.lidar_frame_id", "livox_frame");
  base_frame_ = declare_parameter<std::string>("system.base_frame_id", "livox_frame");
  level_frame_ = declare_parameter<std::string>("system.level_frame_id", "level_frame");
  if (map_frame_.empty() || loc_lidar_frame_.empty() || base_frame_.empty() ||
      level_frame_.empty() || map_frame_ == base_frame_ || map_frame_ == loc_lidar_frame_ ||
      level_frame_ == loc_lidar_frame_ || level_frame_ == base_frame_ || level_frame_ == map_frame_)
    throw std::runtime_error("Localization TF frame names must form a valid tree");
  const double x = declare_parameter<double>("system.base_to_lidar_x", 0.0);
  const double y = declare_parameter<double>("system.base_to_lidar_y", 0.0);
  const double z = declare_parameter<double>("system.base_to_lidar_z", 0.0);
  const double roll = declare_parameter<double>("system.base_to_lidar_roll", 0.0);
  const double pitch = declare_parameter<double>("system.base_to_lidar_pitch", 0.0);
  const double yaw = declare_parameter<double>("system.base_to_lidar_yaw", 0.0);
  const M3 rotation = (Eigen::AngleAxis<scalar>(yaw, V3::UnitZ()) *
                       Eigen::AngleAxis<scalar>(pitch, V3::UnitY()) *
                       Eigen::AngleAxis<scalar>(roll, V3::UnitX())).toRotationMatrix();
  const SE3 base_lidar(rotation, V3(x, y, z));
  if (!base_lidar.matrix().allFinite()) throw std::runtime_error("Invalid base-to-lidar extrinsic");
  if (base_frame_ == loc_lidar_frame_ &&
      (V3(x, y, z).norm() > 1e-6 || (rotation - M3::Identity()).norm() > 1e-6))
    throw std::runtime_error("Identical base/lidar frames require identity extrinsic");
  lidar_base_ = base_lidar.inverse();
  publish_tf_ = declare_parameter<bool>("runtime.publish_tf", true);
  publish_odom_ = declare_parameter<bool>("runtime.publish_odom", true);
  publish_path_ = declare_parameter<bool>("runtime.publish_path", true);
  imu_timeout_ = declare_parameter<double>("runtime.imu_timeout_sec", 1.0);
  lidar_timeout_ = declare_parameter<double>("runtime.lidar_timeout_sec", 2.0);
  auto& r = reloc_options_;
  r.cold_start.enable = declare_parameter<bool>("reloc.cold_start.enable", r.cold_start.enable);
  r.cold_start.max_attempts = declare_parameter<int>("reloc.cold_start.max_attempts", r.cold_start.max_attempts);
  r.still.cmd_vel_topic = declare_parameter<std::string>("still.cmd_vel_topic", r.still.cmd_vel_topic);
  r.still.cmd_vel_stamped = declare_parameter<bool>("still.cmd_vel_stamped", r.still.cmd_vel_stamped);
  r.still.cmd_hold_sec = declare_parameter<double>("still.cmd_hold_sec", r.still.cmd_hold_sec);
  r.still.max_gyro_mean = declare_parameter<double>("still.max_gyro_mean", r.still.max_gyro_mean);
  r.still.max_gyro_std = declare_parameter<double>("still.max_gyro_std", r.still.max_gyro_std);
  r.still.max_acc_std = declare_parameter<double>("still.max_acc_std", r.still.max_acc_std);
  r.still_gate = declare_parameter<bool>("reloc.still_gate", r.still_gate);
  r.verify.frames = declare_parameter<int>("reloc.verify.frames", r.verify.frames);
  r.verify.min_match_ratio = declare_parameter<double>("reloc.verify.min_match_ratio", r.verify.min_match_ratio);
  r.verify.max_still_drift = declare_parameter<double>("reloc.verify.max_still_drift", r.verify.max_still_drift);
  r.verify.max_speed = declare_parameter<double>("reloc.verify.max_speed", r.verify.max_speed);
  r.zupt.enable = declare_parameter<bool>("zupt.enable", r.zupt.enable);
  r.zupt.velocity_sigma = declare_parameter<double>("zupt.velocity_sigma", r.zupt.velocity_sigma);
  r.hold.enable = declare_parameter<bool>("hold.enable", r.hold.enable);
  r.hold.sigma_pos = declare_parameter<double>("hold.sigma_pos", r.hold.sigma_pos);
  r.hold.sigma_rot_deg = declare_parameter<double>("hold.sigma_rot_deg", r.hold.sigma_rot_deg);
  r.hold.max_offset = declare_parameter<double>("hold.max_offset", r.hold.max_offset);
  r.hold.max_rot_deg = declare_parameter<double>("hold.max_rot_deg", r.hold.max_rot_deg);
  r.vehicle.enable = declare_parameter<bool>("vehicle.enable", r.vehicle.enable);
  r.vehicle.cloud_topic = declare_parameter<std::string>("vehicle.cloud_topic", r.vehicle.cloud_topic);
  r.vehicle.box_topic = declare_parameter<std::string>("vehicle.box_topic", r.vehicle.box_topic);
  r.vehicle.timeout_sec = declare_parameter<double>("vehicle.timeout_sec", r.vehicle.timeout_sec);
  r.vehicle.min_points = declare_parameter<int>("vehicle.min_points", r.vehicle.min_points);
  auto& u = r.vehicle.update;
  u.enable = declare_parameter<bool>("vehicle.update.enable", u.enable);
  u.start_topic = declare_parameter<std::string>("vehicle.update.start_topic", u.start_topic);
  u.stop_topic = declare_parameter<std::string>("vehicle.update.stop_topic", u.stop_topic);
  u.clear_topic = declare_parameter<std::string>("vehicle.update.clear_topic", u.clear_topic);
  u.margin = declare_parameter<double>("vehicle.update.margin", u.margin);
  u.min_range = declare_parameter<double>("vehicle.update.min_range", u.min_range);
  r.origin_prior.enable = declare_parameter<bool>("reloc.origin_prior.enable", r.origin_prior.enable);
  r.origin_prior.x = declare_parameter<double>("reloc.origin_prior.x", r.origin_prior.x);
  r.origin_prior.y = declare_parameter<double>("reloc.origin_prior.y", r.origin_prior.y);
  r.origin_prior.z = declare_parameter<double>("reloc.origin_prior.z", r.origin_prior.z);
  r.origin_prior.yaw_step_deg = declare_parameter<double>("reloc.origin_prior.yaw_step_deg", r.origin_prior.yaw_step_deg);
  r.origin_prior.max_delta_trans = declare_parameter<double>("reloc.origin_prior.max_delta_trans", r.origin_prior.max_delta_trans);
  r.kiss.voxel_sizes = declare_parameter<std::vector<double>>("reloc.kiss.voxel_sizes", r.kiss.voxel_sizes);
  r.kiss.use_quatro = declare_parameter<bool>("reloc.kiss.use_quatro", r.kiss.use_quatro);
  r.kiss.target_pre_voxel = declare_parameter<double>("reloc.kiss.target_pre_voxel", r.kiss.target_pre_voxel);
  r.kiss.max_target_points = static_cast<std::size_t>(declare_parameter<int>(
      "reloc.kiss.max_target_points", static_cast<int>(r.kiss.max_target_points)));
  r.kiss.min_rotation_inliers = static_cast<std::size_t>(declare_parameter<int>(
      "reloc.kiss.min_rotation_inliers", static_cast<int>(r.kiss.min_rotation_inliers)));
  r.kiss.min_final_inliers = static_cast<std::size_t>(declare_parameter<int>(
      "reloc.kiss.min_final_inliers", static_cast<int>(r.kiss.min_final_inliers)));
  r.kiss.min_agreeing_candidates = declare_parameter<int>("reloc.kiss.min_agreeing_candidates", r.kiss.min_agreeing_candidates);
  r.kiss.max_tilt_deg = declare_parameter<double>("reloc.kiss.max_tilt_deg", r.kiss.max_tilt_deg);
  r.genz.map_voxel = declare_parameter<double>("reloc.genz.map_voxel", r.genz.map_voxel);
  r.genz.max_points_per_voxel = declare_parameter<int>("reloc.genz.max_points_per_voxel", r.genz.max_points_per_voxel);
  r.genz.planarity_threshold = declare_parameter<double>("reloc.genz.planarity_threshold", r.genz.planarity_threshold);
  r.genz.source_voxel = declare_parameter<double>("reloc.genz.source_voxel", r.genz.source_voxel);
  r.genz.max_iterations = declare_parameter<int>("reloc.genz.max_iterations", r.genz.max_iterations);
  r.genz.convergence = declare_parameter<double>("reloc.genz.convergence", r.genz.convergence);
  r.genz.coarse_max_corr = declare_parameter<double>("reloc.genz.coarse_max_corr", r.genz.coarse_max_corr);
  r.genz.fine_max_corr = declare_parameter<double>("reloc.genz.fine_max_corr", r.genz.fine_max_corr);
  r.fc.map_voxel = declare_parameter<double>("reloc.fc.map_voxel", r.fc.map_voxel);
  r.fc.inlier_dist = declare_parameter<double>("reloc.fc.inlier_dist", r.fc.inlier_dist);
  r.fc.min_inlier_ratio = declare_parameter<double>("reloc.fc.min_inlier_ratio", r.fc.min_inlier_ratio);
  r.fc.wall_band_min = declare_parameter<double>("reloc.fc.wall_band_min", r.fc.wall_band_min);
  r.fc.wall_band_max = declare_parameter<double>("reloc.fc.wall_band_max", r.fc.wall_band_max);
  r.fc.min_wall_inlier_ratio = declare_parameter<double>("reloc.fc.min_wall_inlier_ratio", r.fc.min_wall_inlier_ratio);
  r.fc.min_coverage = declare_parameter<double>("reloc.fc.min_coverage", r.fc.min_coverage);
  r.fc.min_sector_points = declare_parameter<int>("reloc.fc.min_sector_points", r.fc.min_sector_points);
  r.fc.max_rmse = declare_parameter<double>("reloc.fc.max_rmse", r.fc.max_rmse);
  r.fc.max_delta_trans_manual = declare_parameter<double>("reloc.fc.max_delta_trans_manual", r.fc.max_delta_trans_manual);
  r.fc.max_delta_rot_deg_manual = declare_parameter<double>("reloc.fc.max_delta_rot_deg_manual", r.fc.max_delta_rot_deg_manual);
  r.fc.max_delta_trans_kiss = declare_parameter<double>("reloc.fc.max_delta_trans_kiss", r.fc.max_delta_trans_kiss);
  r.fc.max_delta_rot_deg_kiss = declare_parameter<double>("reloc.fc.max_delta_rot_deg_kiss", r.fc.max_delta_rot_deg_kiss);
  r.fc.ambiguity_trans = declare_parameter<double>("reloc.fc.ambiguity_trans", r.fc.ambiguity_trans);
  r.fc.ambiguity_rot_deg = declare_parameter<double>("reloc.fc.ambiguity_rot_deg", r.fc.ambiguity_rot_deg);
  const bool kiss_voxels_ok = !r.kiss.voxel_sizes.empty() &&
      std::all_of(r.kiss.voxel_sizes.begin(), r.kiss.voxel_sizes.end(), [](double v) { return v >= 0.05; });
  if (r.cold_start.max_attempts < 0 || !kiss_voxels_ok ||
      r.still.cmd_hold_sec < 0 || r.still.max_gyro_mean <= 0 || r.still.max_gyro_std <= 0 ||
      r.still.max_acc_std <= 0 || r.zupt.velocity_sigma <= 0 || r.verify.frames < 0 ||
      r.hold.sigma_pos <= 0 || r.hold.sigma_rot_deg <= 0 || r.hold.max_offset <= 0 || r.hold.max_rot_deg <= 0 ||
      (r.vehicle.enable && (r.vehicle.cloud_topic.empty() || r.vehicle.box_topic.empty() ||
                            r.vehicle.timeout_sec <= 0 || r.vehicle.min_points < 1)) ||
      (r.vehicle.enable && u.enable && (u.start_topic.empty() || u.stop_topic.empty() ||
                                        u.clear_topic.empty() || u.margin < 0 || u.min_range < 0)) ||
      r.verify.min_match_ratio < 0 || r.verify.min_match_ratio > 1 ||
      r.verify.max_still_drift <= 0 || r.verify.max_speed <= 0 ||
      r.origin_prior.yaw_step_deg <= 0 || r.origin_prior.yaw_step_deg > 360 ||
      r.origin_prior.max_delta_trans <= 0 || r.kiss.min_agreeing_candidates < 1 || r.genz.map_voxel <= 0 ||
      r.genz.max_points_per_voxel < 1 || r.genz.max_iterations < 1 || r.genz.coarse_max_corr <= 0 ||
      r.genz.fine_max_corr <= 0 || r.fc.map_voxel <= 0 || r.fc.inlier_dist <= 0 ||
      r.fc.wall_band_min >= r.fc.wall_band_max || r.fc.min_sector_points < 1)
    throw std::runtime_error("Invalid reloc.* parameters");
  if (g_filter_rate < 1 || g_pub_step < 1 || g_ivox_resolution <= 0 ||
      g_voxel_fliter_size <= 0 || g_kf_max_iterations < 1 ||
      imu_timeout_ <= 0 || lidar_timeout_ <= 0)
    throw std::runtime_error("Invalid localization sensor/filter/watchdog parameters");
  loc_state_pub_ = create_publisher<std_msgs::msg::Int32>("hikari_loc/loc_state", 10);
  ndt_status_pub_ = create_publisher<std_msgs::msg::Int32>("hikari_loc/ndt_status", 10);
  loc_status_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("hikari_loc/status", 10);
  loc_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("hikari_loc/loc_status", 10);
  global_map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/pcdmap", rclcpp::QoS(1).reliable().transient_local());
  loc_status_timer_ = create_wall_timer(std::chrono::milliseconds(200), [this]() { publishLocStatus(); });
  setupCmdVel();
  setupVehicleCloud();
}

void ROSWrapper::setupCmdVel() {
  const auto& still = reloc_options_.still;
  if (still.cmd_vel_topic.empty()) {
    LOG(WARNING) << YELLOW << " ---> [Still] no still.cmd_vel_topic: init gate uses IMU only, ZUPT off" << RESET;
    return;
  }
  // Best effort matches both reliable and best-effort publishers.
  const auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
  // Idle from startup: a robot nobody has commanded yet is standing.
  last_cmd_motion_ = now().seconds() - still.cmd_hold_sec;
  auto on_cmd = [this](const geometry_msgs::msg::Twist& twist) {
    constexpr double kZero = 1e-3;
    if (std::abs(twist.linear.x) > kZero || std::abs(twist.linear.y) > kZero ||
        std::abs(twist.angular.z) > kZero)
      last_cmd_motion_ = now().seconds();
  };
  if (still.cmd_vel_stamped)
    cmd_vel_stamped_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
        still.cmd_vel_topic, qos,
        [on_cmd](geometry_msgs::msg::TwistStamped::ConstSharedPtr msg) { on_cmd(msg->twist); });
  else
    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        still.cmd_vel_topic, qos, [on_cmd](geometry_msgs::msg::Twist::ConstSharedPtr msg) { on_cmd(*msg); });
  LOG(INFO) << GREEN << " ---> [Still] commanded velocity from " << still.cmd_vel_topic
            << ", hold " << still.cmd_hold_sec << " s" << RESET;
}

ROSWrapper::CmdState ROSWrapper::cmdState() const {
  const std::size_t publishers = cmd_vel_sub_ ? cmd_vel_sub_->get_publisher_count()
                               : cmd_vel_stamped_sub_ ? cmd_vel_stamped_sub_->get_publisher_count() : 0;
  // twist_mux goes quiet when idle, so silence counts as zero command; but with
  // nobody on the topic (wrong name, mux down) there is no evidence at all.
  if (publishers == 0) return CmdState::kUnknown;
  return now().seconds() - last_cmd_motion_ >= reloc_options_.still.cmd_hold_sec ? CmdState::kIdle
                                                                                 : CmdState::kMoving;
}

void ROSWrapper::setupVehicleCloud() {
  const auto& vehicle = reloc_options_.vehicle;
  if (!vehicle.enable) return;
  // Latched on the modeling side: a restarted localization still gets the last run.
  const auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
  vehicle_cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      vehicle.cloud_topic, qos, [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        vehicle_cloud_msg_ = msg;
        tryAcceptVehicle();
      });
  vehicle_box_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      vehicle.box_topic, qos, [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) {
        vehicle_box_stamp_ = rclcpp::Time(msg->header.stamp).seconds();
        const auto& q = msg->pose.orientation;
        vehicle_box_yaw_ = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
        tryAcceptVehicle();
      });
  LOG(INFO) << GREEN << " ---> [Vehicle] cloud from " << vehicle.cloud_topic << " (with " << vehicle.box_topic
            << "), kept " << vehicle.timeout_sec << " s" << RESET;
  const auto& update = vehicle.update;
  if (!update.enable) return;
  // Best effort matches both reliable and best-effort publishers.
  const auto command_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
  auto on_command = [this](const char* name, auto apply) {
    return [this, name, apply](std_msgs::msg::Bool::ConstSharedPtr msg) {
      if (!msg->data) return;
      apply();
      LOG(INFO) << GREEN << " ---> [Vehicle] " << name << ": map update "
                << (vehicle_collecting_ ? "collecting" : "stopped") << RESET;
    };
  };
  // A new start also drops whatever an unfinished previous docking collected.
  vehicle_start_sub_ = create_subscription<std_msgs::msg::Bool>(
      update.start_topic, command_qos, on_command(update.start_topic.c_str(), [this]() {
        vehicle_layer_clear_ = true;
        vehicle_collecting_ = true;
      }));
  vehicle_stop_sub_ = create_subscription<std_msgs::msg::Bool>(
      update.stop_topic, command_qos, on_command(update.stop_topic.c_str(), [this]() {
        vehicle_collecting_ = false;
      }));
  vehicle_clear_sub_ = create_subscription<std_msgs::msg::Bool>(
      update.clear_topic, command_qos, on_command(update.clear_topic.c_str(), [this]() {
        vehicle_collecting_ = false;
        vehicle_layer_clear_ = true;
      }));
  LOG(INFO) << GREEN << " ---> [Vehicle] map update: " << update.start_topic << " -> " << update.stop_topic
            << ", cleared on " << update.clear_topic << ", margin " << update.margin << " m, min range "
            << update.min_range << " m" << RESET;
}

void ROSWrapper::tryAcceptVehicle() {
  // Both are published in one modeling run with the same stamp; a cloud without
  // its box comes from a failed run and is never used.
  constexpr double kPairTolerance = 0.5;  // s
  if (!vehicle_cloud_msg_ || vehicle_box_stamp_ < 0) return;
  const double stamp = rclcpp::Time(vehicle_cloud_msg_->header.stamp).seconds();
  if (stamp == vehicle_checked_stamp_ || std::abs(stamp - vehicle_box_stamp_) > kPairTolerance) return;
  vehicle_checked_stamp_ = stamp;
  const auto& vehicle = reloc_options_.vehicle;
  const double age = now().seconds() - stamp;
  if (vehicle_cloud_msg_->header.frame_id != map_frame_) {
    LOG(WARNING) << YELLOW << " ---> [Vehicle] cloud ignored: frame " << vehicle_cloud_msg_->header.frame_id
                 << ", expected " << map_frame_ << RESET;
    return;
  }
  if (age > vehicle.timeout_sec) {
    LOG(INFO) << YELLOW << " ---> [Vehicle] cloud ignored: modeled " << age << " s ago (timeout "
              << vehicle.timeout_sec << " s)" << RESET;
    return;
  }
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::fromROSMsg(*vehicle_cloud_msg_, cloud);
  VV3 points;
  points.reserve(cloud.size());
  for (const auto& p : cloud)
    if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) points.emplace_back(p.x, p.y, p.z);
  if (points.size() < static_cast<std::size_t>(vehicle.min_points)) {
    LOG(WARNING) << YELLOW << " ---> [Vehicle] cloud ignored: " << points.size() << " points (min "
                 << vehicle.min_points << ")" << RESET;
    return;
  }
  vehicle_pending_.swap(points);
  vehicle_pending_yaw_ = vehicle_box_yaw_;
  vehicle_pending_set_ = true;
  vehicle_map_stamp_ = stamp;
}

bool ROSWrapper::takeVehicleUpdate(VV3& points, scalar& yaw) {
  if (vehicle_pending_set_) {
    points.swap(vehicle_pending_);
    yaw = vehicle_pending_yaw_;
    vehicle_pending_.clear();
    vehicle_pending_set_ = false;
    return true;
  }
  if (vehicle_map_stamp_ < 0 || now().seconds() - vehicle_map_stamp_ <= reloc_options_.vehicle.timeout_sec)
    return false;
  vehicle_map_stamp_ = -1;
  points.clear();
  return true;
}

void ROSWrapper::publishLocalization(const NavState& state) {
  const builtin_interfaces::msg::Time stamp = rclcpp::Time(static_cast<int64_t>(state.timestamp * 1e9));
  const SE3 map_lidar = SE3(state.R, state.p) * g_lidar_imu;
  const Quat q(map_lidar.R_);
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = map_frame_;
  odom.child_frame_id = loc_lidar_frame_;
  odom.pose.pose.position.x = map_lidar.t_.x();
  odom.pose.pose.position.y = map_lidar.t_.y();
  odom.pose.pose.position.z = map_lidar.t_.z();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();
  // Same convention as hikari_loclite: world IMU velocity rotated into lidar.
  const V3 velocity = map_lidar.R_.transpose() * state.v;
  odom.twist.twist.linear.x = velocity.x();
  odom.twist.twist.linear.y = velocity.y();
  odom.twist.twist.linear.z = velocity.z();
  if (publish_odom_) pub_odom_->publish(odom);
  if (publish_path_ && (last_path_stamp_ < 0 || state.timestamp - last_path_stamp_ >= 0.1)) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = odom.header;
    pose.pose = odom.pose.pose;
    path_.header = odom.header;
    path_.poses.push_back(pose);
    if (path_.poses.size() > 5000) path_.poses.erase(path_.poses.begin());
    pub_path_->publish(path_);
    last_path_stamp_ = state.timestamp;
  }
  if (last_output_stamp_ >= 0 && state.timestamp > last_output_stamp_)
    fps_ = 1.0 / (state.timestamp - last_output_stamp_);
  last_output_stamp_ = state.timestamp;
  if (!publish_tf_) return;
  auto send = [&](const std::string& parent, const std::string& child, const SE3& pose) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = parent;
    tf.child_frame_id = child;
    tf.transform.translation.x = pose.t_.x();
    tf.transform.translation.y = pose.t_.y();
    tf.transform.translation.z = pose.t_.z();
    const Quat rotation(pose.R_);
    tf.transform.rotation.x = rotation.x();
    tf.transform.rotation.y = rotation.y();
    tf.transform.rotation.z = rotation.z();
    tf.transform.rotation.w = rotation.w();
    tf_broadcaster_->sendTransform(tf);
  };
  send(map_frame_, base_frame_, map_lidar * lidar_base_);
  // Preserve lidar yaw while removing roll/pitch in the gravity-aligned map.
  const scalar yaw = std::atan2(map_lidar.R_(1, 0), map_lidar.R_(0, 0));
  const M3 map_level = Eigen::AngleAxis<scalar>(yaw, V3::UnitZ()).toRotationMatrix();
  send(loc_lidar_frame_, level_frame_, SE3(M3(map_lidar.R_.transpose() * map_level), V3::Zero()));
}

void ROSWrapper::publishLocStatus() {
  const double time = now().seconds();
  const double imu_age = last_imu_received_ < 0 ? -1 : std::max(0.0, time - last_imu_received_);
  const double lidar_age = last_lidar_received_ < 0 ? -1 : std::max(0.0, time - last_lidar_received_);
  if (!is_active()) loc_state_ = 5;
  else if (loc_state_ == 2 && (imu_age < 0 || lidar_age < 0 ||
           imu_age > imu_timeout_ || lidar_age > lidar_timeout_)) {
    loc_state_ = 4;
    LOG(WARNING) << RED << " ---> [Loc] LOST: sensor timeout, imu_age=" << imu_age << " s (limit " << imu_timeout_
                 << "), lidar_age=" << lidar_age << " s (limit " << lidar_timeout_ << ")" << RESET;
  }
  std_msgs::msg::Int32 state;
  state.data = loc_state_;
  loc_state_pub_->publish(state);
  state.data = 0;
  ndt_status_pub_->publish(state);
  std_msgs::msg::Float32MultiArray status;
  status.layout.dim.resize(1);
  status.layout.dim[0].label = "state;ndt_conf;imu_age_s;lidar_age_s;fps;in_map";
  status.layout.dim[0].size = status.layout.dim[0].stride = 6;
  // PCL fitness is not hikari's NDT confidence: explicitly report unavailable.
  status.data = {static_cast<float>(loc_state_), -1.f, static_cast<float>(imu_age),
                static_cast<float>(lidar_age), loc_state_ == 2 ? static_cast<float>(fps_) : 0.f,
                loc_state_ == 2 ? 1.f : 0.f};
  loc_status_pub_->publish(status);
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = map_frame_;
  marker.header.stamp = now();
  marker.ns = "loc_status";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::msg::Marker::ADD;
  if (!path_.poses.empty()) marker.pose.position = path_.poses.back().pose.position;
  marker.pose.position.z += 2;
  marker.pose.orientation.w = 1;
  marker.scale.z = 1.5;
  marker.color.a = 1;
  marker.color.r = loc_state_ == 2 ? 0 : 1;
  marker.color.g = loc_state_ == 2 ? 1 : 0;
  marker.text = loc_state_ == 2 ? "GOOD" : loc_state_ == 1 ? "INIT" :
                loc_state_ == 4 ? "LOST" : "WAIT FOR /initialpose";
  loc_marker_pub_->publish(marker);
}
}  // namespace LI2Sup
