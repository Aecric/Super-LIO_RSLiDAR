#include "ros/ROSWrapper.h"
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
           imu_age > imu_timeout_ || lidar_age > lidar_timeout_)) loc_state_ = 4;
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
