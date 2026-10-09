
#ifndef ROSWRAPPER_HPP_
#define ROSWRAPPER_HPP_

#include <map>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tuple>
#include <deque>
#include <atomic>
#include <vector>
#include <execution>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/callback_group.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <pcl_conversions/pcl_conversions.h>

/// msgs
#include "super_lio/msg/cloud_pose.hpp"
#include "super_lio/msg/cloud_pose2.hpp"
#include "super_lio_loop_msgs/msg/keyframe.hpp"
#include "super_lio_loop_msgs/srv/get_corrected_poses.hpp"


#include "lio/params.h"
#include "basic/alias.h"
#include "basic/logs.h"
#include "basic/Manifold.h"
#include "common/ds.h"

#include "lio/ESKF.h"
#include "OctVoxMap/OctVoxMap.hpp"


namespace LI2Sup{

void LoadParamFromRos(rclcpp::Node& node);

void livox2pcl(const livox_ros_driver2::msg::CustomMsg::SharedPtr& msg, BASIC::CloudPtr& point_cloud);

class ROSWrapper : public rclcpp::Node {
public:
  explicit ROSWrapper(const rclcpp::NodeOptions& options = rclcpp::NodeOptions(), bool localization = false);
  ~ROSWrapper(){};
  using Ptr = std::shared_ptr<ROSWrapper>;
  bool sync_measure(MeasureGroup&);

  void setESKF(ESKF::Ptr& eskf) { eskf_ = eskf;}

  void clear(){
    lidar_buffer_.clear();
    imu_buffer_.clear();
    lidar_pushed_ = false;
    last_timestamp_imu_ = -1.0;
    last_timestamp_lidar_ = -1.0;
    path_.poses.clear();
    last_output_stamp_ = last_path_stamp_ = -1;
    fps_ = 0;
    last_path_point_ = BASIC::V3(0, 0, -100);
  }

  /// False while the node is on standby: sensor subscriptions are torn down, so
  /// nothing is deserialised and nothing is processed.
  bool is_active() const { return active_.load(std::memory_order_acquire); }

  bool localization() const { return localization_; }
  void setLocState(int state) { loc_state_ = state; }
  int locState() const { return loc_state_; }
  void pub_odom(const NavState&);

  void pub_cloud_world(const BASIC::CloudPtr& pc, double time);
  void pub_cloud2planner(const BASIC::CloudPtr& pc, double time);
  void pub_cloud_world_pose(const BASIC::CloudPtr& pc, 
                            const NavState& state);
  void pub_cloud_body_pose(const BASIC::CloudPtr& pc, 
                           const NavState& state);
  void pub_cloud_body_pose( const BASIC::VV3& pc_body,
                            const NavState& state);
  /// Undistorted scan expressed in the LiDAR frame at scan-end, plus the
  /// matching world<-lidar pose on an identical stamp. Downstream nodes that
  /// need motion-compensated points in the robot's own TF tree (target
  /// modeling, ROI cropping) consume this pair instead of the raw scan.
  void pub_cloud_body_odom(const BASIC::CloudPtr& body_imu_cloud,
                           const NavState& state);
  void pub_processing_time(double time, double current_time, double mean_time, double std_time);
  void pub_keyframe(
    std::uint32_t id, const NavState & state, const BASIC::CloudPtr & body_cloud);
  bool request_corrected_poses(
    std::vector<std::uint32_t> & ids,
    std::vector<geometry_msgs::msg::Pose> & poses,
    double timeout_seconds);

  void set_global_map(const BASIC::CloudPtr& global_map);

  void set_initial_data(BASIC::SE3& init_pose, bool& flg_get_init_guess, bool flg_finish_init = false);

  rclcpp::CallbackGroup::SharedPtr getSensorCallbackGroup() {
    return cb_sensor_;
  }

private:
  void imuHandler(const sensor_msgs::msg::Imu::SharedPtr msg);
  void livoxHandler(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg);
  void stdMsgHandler(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  void setupLocalization();
  void publishLocalization(const NavState& state);
  void publishLocStatus();
  bool localization_ = false;
  int loc_state_ = 5;
  std::string map_frame_ = "map", loc_lidar_frame_ = "livox_frame";
  std::string base_frame_ = "livox_frame", level_frame_ = "level_frame";
  BASIC::SE3 lidar_base_;
  bool publish_tf_ = true, publish_odom_ = true, publish_path_ = true;
  double last_imu_received_ = -1, last_lidar_received_ = -1;
  double last_output_stamp_ = -1, last_path_stamp_ = -1, fps_ = 0;
  double imu_timeout_ = 1, lidar_timeout_ = 2;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr loc_state_pub_, ndt_status_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr loc_status_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr loc_marker_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_pub_;
  rclcpp::TimerBase::SharedPtr loc_status_timer_, global_map_timer_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr init_pose_sub_;
  void setupParams();
  void setupIO();
  void createSensorSubscriptions();
  void destroySensorSubscriptions();
  void setActive(bool active);

private:
  rclcpp::CallbackGroup::SharedPtr cb_sensor_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_lidar_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar_std_;

  std::deque<IMUData>   imu_buffer_;
  std::deque<LidarData> lidar_buffer_;
  bool lidar_pushed_ = false;
  double last_timestamp_imu_ = -1.0;
  double last_timestamp_lidar_ = -1.0;

  ESKF::Ptr eskf_{nullptr};

  nav_msgs::msg::Path path_;
  geometry_msgs::msg::PoseStamped msg2uav_;
  sensor_msgs::msg::PointCloud2 global_map_msg_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  BASIC::V3 last_path_point_ = BASIC::V3(0, 0, -100);

  /// frame_id of the most recent incoming scan; re-used as the frame_id of the
  /// undistorted body cloud so it lands in the robot's existing TF tree.
  std::string lidar_frame_id_ = "lidar";

  std::atomic<bool> active_{true};
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr set_active_srv_;

/// output.
private:
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;       /// lidar fre --> IMU frame
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_imu_odom_;   /// IMU fre   --> IMU frame
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_robo_odom_;  /// IMU fre   --> Robot frame
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_world_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_body_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_body_odom_;
  rclcpp::Publisher<super_lio_loop_msgs::msg::Keyframe>::SharedPtr
    pub_keyframe_;
  rclcpp::Client<super_lio_loop_msgs::srv::GetCorrectedPoses>::SharedPtr
    corrected_pose_client_;
};

} // namespace END.

#endif
