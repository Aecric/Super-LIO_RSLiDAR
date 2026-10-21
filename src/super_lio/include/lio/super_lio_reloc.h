

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
};

} // namespace END.

#endif


