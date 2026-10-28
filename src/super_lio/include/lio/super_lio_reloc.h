

#ifndef SUPER_LIO_RELOCATION_H_
#define SUPER_LIO_RELOCATION_H_


#include <memory>

#include <pcl/registration/ndt.h>

#include "super_lio.h"
#include "lio/reloc/fc_gate.h"
#include "lio/reloc/genz_refiner.h"
#include "lio/reloc/kiss_global_matcher.h"


namespace LI2Sup{

class SuperLIOReLoc : public SuperLIO {
public:
  SuperLIOReLoc(){
    BASIC::V3 init_t = BASIC::V3(g_init_px, g_init_py, g_init_pz);
    Eigen::Matrix3d init_R = (Eigen::AngleAxisd(g_init_yaw / 180 * M_PI, Eigen::Vector3d::UnitZ()) *
                              Eigen::AngleAxisd(g_init_pitch /180 * M_PI, Eigen::Vector3d::UnitY()) *
                              Eigen::AngleAxisd(g_init_roll / 180 * M_PI, Eigen::Vector3d::UnitX())).toRotationMatrix();
                
    BASIC::M3 init_R2 = init_R.cast<BASIC::scalar>();             
      
    re_init_pose_ = BASIC::SE3(BASIC::SO3(init_R2), init_t);
  };
  ~SuperLIOReLoc(){};

  void init() override;
  void Reset() override;
  void process() override;

private:
  bool kf_init() override;
  bool map_init() override;
  void UpdateMap() override;
  void Output() override;

  /// Localization mode: KISS (cold start) or NDT (/initialpose) -> GenZ-ICP -> FC.
  /// level_rot rotates the IMU frame into the gravity-levelled frame.
  bool alignToMap(bool manual, const BASIC::M3& level_rot, BASIC::M4& T_map_imu);
  enum class Selection { kAccepted, kNone, kAmbiguous };
  /// Verify every T_map_imu guess (NDT coarse -> GenZ-ICP -> FC) and pick the
  /// best passing one. kNone also when fewer than min_passed pass.
  Selection selectCandidate(const BASIC::CloudPtr& query, const std::vector<Eigen::Matrix4d>& guesses,
                            double max_delta_trans, double max_delta_rot_deg, std::size_t min_passed,
                            const char* label, Eigen::Matrix4d& T_map_imu);
  bool coldStartArmed() const;

  /// Standstill evidence over a set of IMU samples (Welford sums per axis).
  struct ImuStats {
    double gyro_mean = 0, gyro_std = 0, acc_std = 0;  // rad/s, rad/s, m/s^2
    bool valid = false;
  };
  static ImuStats imuStats(int count, const BASIC::V3& mean_gyro, const BASIC::V3& mean_acce,
                           const BASIC::V3& m2_gyro, const BASIC::V3& m2_acce);
  bool imuStill(const ImuStats& stats) const;
  /// Init window: command not moving (unknown falls back to IMU) and IMU still.
  bool initWindowStill(int imu_count, const BASIC::V3& mean_gyro, const BASIC::V3& mean_acce);
  /// ZUPT for this scan while tracking: command idle and the scan's IMU still.
  void onMeasure() override;
  /// Pose hold for this scan (hold.*): anchor while tracking and still.
  void holdPose(bool cmd_idle);
  /// One scan of the post-acceptance window (reloc.verify); false discards the pose.
  bool verifyScan(const NavState& state, std::size_t matched);
  /// Localization: rebuild the search index from the fixed map plus extra
  /// points in the map frame (the modeled vehicle; empty = fixed map only).
  void rebuildFixedIndex(const BASIC::VV3& extra);
  /// Vehicle cloud accepted (non-empty) or expired (empty): map and update region.
  void setVehicle(BASIC::VV3 points, BASIC::scalar yaw);
  /// Map update beside the vehicle (vehicle.update.*), one scan.
  void updateVehicleLayer();
  bool inVehicleRegion(const BASIC::V3& p) const;

private:
  bool map_loaded_ = false;
  int init_frame_count_ = 0;
  bool has_init_pose_ = false;
  BASIC::CloudPtr init_obs_data_;
  bool flg_get_init_guess_ = false;
  BASIC::SE3 re_init_pose_;

  std::unique_ptr<KissGlobalMatcher> kiss_;
  std::unique_ptr<GenzRefiner> genz_;
  std::unique_ptr<FcGate> fc_;
  pcl::NormalDistributionsTransform<BASIC::PointType, BASIC::PointType>::Ptr ndt_;
  int cold_start_attempts_ = 0;
  // Welford sums of squared deviations over the init window (per axis).
  BASIC::V3 init_m2_gyro_ = BASIC::V3::Zero();
  BASIC::V3 init_m2_acce_ = BASIC::V3::Zero();
  int moving_windows_ = 0;
  bool zupt_active_ = false;
  bool frame_still_ = false;      // current scan: command idle and IMU still
  // Pose hold anchor and how far the scans tried to pull away from it.
  bool hold_set_ = false;
  BASIC::SO3 hold_R_;
  BASIC::V3 hold_p_ = BASIC::V3::Zero();
  int hold_scans_ = 0;
  double hold_max_offset_ = 0, hold_max_rot_deg_ = 0;
  // Post-acceptance verification window.
  int verify_left_ = 0;
  bool verify_failed_ = false;
  bool verify_all_still_ = true;
  BASIC::V3 verify_anchor_ = BASIC::V3::Zero();
  double verify_min_ratio_ = 1.0, verify_max_drift_ = 0.0, verify_max_speed_ = 0.0;
  // Modeled vehicle in the map and the region its map update may fill: the
  // footprint in the box frame (yaw), grown by vehicle.update.margin.
  BASIC::VV3 vehicle_points_;
  BASIC::scalar vehicle_cos_ = 1, vehicle_sin_ = 0;
  BASIC::V3 region_lo_ = BASIC::V3::Zero(), region_hi_ = BASIC::V3::Zero();
  BASIC::VV3 layer_cloud_;   // points the map update stored, one per sub-voxel filled
  bool layer_collecting_ = false;
};

} // namespace END.

#endif


