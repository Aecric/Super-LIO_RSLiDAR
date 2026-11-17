#ifndef SUPER_LIO_RELOC_OPTIONS_H_
#define SUPER_LIO_RELOC_OPTIONS_H_

#include <cstddef>
#include <string>
#include <vector>

namespace LI2Sup {

/// Fixed-map initialization pipeline (localization mode only):
///   /initialpose : guess                      -> NDT coarse -> GenZ-ICP -> FC
///   cold start   : 0. last GOOD pose, if there was one (after LOST)
///                                              -> NDT coarse -> GenZ-ICP -> FC
///                  1. if it does not pass: map-origin prior (yaw sweep)
///                                              -> NDT coarse -> GenZ-ICP -> FC
///                  2. only if no origin hypothesis passes:
///                     KISS-Matcher Top-K guesses -> NDT coarse -> GenZ-ICP -> FC
/// All poses are T_map_imu; the query cloud is the accumulated init window.
struct RelocOptions {
  struct ColdStart {
    // Run KISS global registration whenever no /initialpose is pending
    // (startup, after LOST, after re-activation).
    bool enable = true;
    // Failed attempts before giving up and waiting for /initialpose. 0 = never.
    int max_attempts = 0;
  } cold_start;

  /// Standstill detection (params still.*), shared by the init gate and ZUPT.
  /// Still = the commanded velocity has been zero for cmd_hold_sec AND the IMU
  /// stats are within limits. The robot only moves when commanded, so cmd_vel is
  /// the primary signal; the loose IMU limits only veto uncommanded motion
  /// (pushed, bumped). IMU alone cannot tell smooth constant-velocity driving
  /// from standing still on a quiet chassis.
  struct Still {
    // Final command to the chassis (twist_mux output). Empty: no command
    // source, the init gate falls back to the IMU check and ZUPT is off.
    std::string cmd_vel_topic = "/cmd_vel";
    bool cmd_vel_stamped = false;  // geometry_msgs/TwistStamped instead of Twist
    double cmd_hold_sec = 1.0;     // braking / coasting margin after the last non-zero command
    double max_gyro_mean = 0.05;   // rad/s, includes the gyro bias
    double max_gyro_std = 0.03;    // rad/s, norm of per-axis std
    double max_acc_std = 0.25;     // m/s^2, norm of per-axis std
  } still;

  /// Initialization assumes zero velocity (cold start and /initialpose): an init
  /// window that is not still is discarded and accumulated again.
  bool still_gate = true;

  /// After acceptance the pose is tracked for `frames` scans before GOOD is
  /// reported (nothing is published meanwhile). The alignment stalls the
  /// executor for seconds, so these scans are the backlog from the stall: a
  /// pose that was wrong, or went stale because the robot moved during the
  /// stall, fails here and initialization starts over. 0 disables.
  struct Verify {
    int frames = 10;
    double min_match_ratio = 0.3;      // matched / downsampled scan points
    double max_still_drift = 0.15;     // m from the accepted pose, while still throughout
    double max_speed = 2.0;            // m/s
  } verify;

  /// Zero-velocity update while tracking and still: v = 0 observation per scan.
  /// Keeps a weakly constrained direction (corridor axis) from integrating
  /// velocity noise into position drift while the robot stands.
  struct Zupt {
    bool enable = true;
    double velocity_sigma = 0.02;  // m/s, std of the zero-velocity observation
  } zupt;

  /// Pose hold while tracking and standing (params hold.*): the pose of the
  /// first still scan is held by a strong pose observation. Unmapped things
  /// moving right next to the sensor while parked (the arm in the door, people)
  /// otherwise pull the pose, which ZUPT (velocity only) cannot stop. Scans the
  /// IMU vetoes (arm, door) skip the hold but keep the anchor; any command
  /// releases it.
  struct Hold {
    bool enable = true;
    double sigma_pos = 0.01;       // m
    double sigma_rot_deg = 0.2;    // deg
    // Beyond this from the anchor the robot really moved (pushed during vetoed
    // scans): re-anchor instead of pulling it back.
    double max_offset = 0.10;      // m
    double max_rot_deg = 3.0;      // deg
  } hold;

  /// Vehicle cloud from target modeling (params vehicle.*), added to the fixed
  /// map while fresh. The vehicle parks beside the docking points but is not in
  /// the map, and its long side runs along the driving direction: unmatched, its
  /// points pull the pose sideways while the map has nothing to hold it along
  /// the vehicle. In the map, its side and ends constrain both directions.
  struct Vehicle {
    bool enable = true;
    std::string cloud_topic = "/tm/debug_registered_cloud";  // map frame, latched
    // Published only for a successful run; the cloud topic also carries the raw
    // accumulation of a failed one, so a cloud is used only with a box of the
    // same stamp.
    std::string box_topic = "/tm/box_pose";
    double timeout_sec = 1800.0;   // removed this long after the modeling stamp
    int min_points = 200;
    /// Map update beside the vehicle (vehicle.update.*). The door is opened only
    /// for the docking, after modeling: while approaching, scan points around
    /// the vehicle fill empty map cells so the open door is in the map when the
    /// robot stops in front of it. Collection ends on arrival, before the arm
    /// reaches in, and the points are dropped once the robot is back.
    struct Update {
      bool enable = true;
      std::string start_topic = "/driverseat_in";        // std_msgs/Bool true: collect
      std::string stop_topic = "/driverseat_in_done";    // true: stop collecting, keep
      std::string clear_topic = "/driverseat_out_done";  // true: drop the points
      double margin = 1.2;      // m around the vehicle footprint (door swing)
      double min_range = 1.0;   // m from the sensor: robot body and arm never enter
    } update;
  } vehicle;

  /// After LOST the robot is normally still where it was last tracked (a sensor
  /// dropout while parked), which can be far from the map origin and in a view
  /// KISS cannot solve. The cold start therefore first tries the last GOOD pose
  /// of this run (position and yaw; roll/pitch from gravity).
  struct LastPose {
    bool enable = true;
    double max_delta_trans = 1.0;       // FC: allowed distance from the last pose
    double max_delta_rot_deg = 15.0;    // FC: allowed rotation from the last pose
  } last_pose;

  /// Devices usually power up near where the map was started (map origin), so
  /// the cold start first tries that position with a set of yaw hypotheses.
  struct OriginPrior {
    bool enable = true;
    double x = 0.0, y = 0.0, z = 0.0;   // IMU position in the map
    double yaw_step_deg = 45.0;         // hypotheses 0, step, 2*step, ...
    double max_delta_trans = 3.0;       // FC: allowed distance from the prior
  } origin_prior;

  struct Kiss {
    // One KISS-Matcher run per voxel size (FPFH scale); each valid solution is a
    // candidate. A static single-view query gives few correspondences, so the
    // inlier counts cannot tell right from wrong: GenZ-ICP + FC decide.
    std::vector<double> voxel_sizes{0.25, 0.3, 0.4};
    // The query is gravity levelled: solve yaw + translation only (Quatro).
    bool use_quatro = true;
    double target_pre_voxel = 0.2;    // map downsample before caching; <=0 disables
    std::size_t max_target_points = 800000;
    std::size_t min_rotation_inliers = 0;
    std::size_t min_final_inliers = 0;
    // A single KISS candidate passing FC may be a lucky fit in a degenerate
    // scene: require this many passing candidates that agree.
    int min_agreeing_candidates = 2;
    // The fixed map is gravity aligned, so a valid solution is (almost) pure yaw.
    double max_tilt_deg = 10.0;
  } kiss;

  struct Genz {
    double map_voxel = 0.3;            // static voxel hash built from the fixed map
    int max_points_per_voxel = 1;
    double planarity_threshold = 0.1;
    double source_voxel = 0.2;         // query downsample before registration
    int max_iterations = 100;
    double convergence = 1e-4;
    double coarse_max_corr = 1.0;      // two passes: coarse then fine
    double fine_max_corr = 0.3;
  } genz;

  /// FC: false-positive barrier for a refined candidate. Every check must pass.
  /// Allowed delta from the guess: /initialpose -> *_manual, origin prior ->
  /// origin_prior.max_delta_trans + half the yaw step, KISS -> *_kiss.
  struct Fc {
    double map_voxel = 0.1;            // KD-tree map resolution
    double inlier_dist = 0.3;
    // Measured on a same-session map: true matches 1.0 / 1.0 / 0.05 m, the
    // strongest wrong ones 0.875 / 0.77 / 0.10 m (inlier / wall / rmse).
    double min_inlier_ratio = 0.85;
    // Floors and ceilings match under any yaw; the wall band (relative to the
    // sensor height) carries the yaw/translation evidence.
    double wall_band_min = -0.5;
    double wall_band_max = 1.0;
    double min_wall_inlier_ratio = 0.8;
    // Fraction of observed 45-degree sectors (wall band) that have inliers.
    double min_coverage = 0.5;
    int min_sector_points = 10;
    double max_rmse = 0.08;            // over inliers
    // Allowed motion from the coarse guess to the refined pose.
    double max_delta_trans_manual = 5.0;
    double max_delta_rot_deg_manual = 45.0;
    double max_delta_trans_kiss = 3.0;
    double max_delta_rot_deg_kiss = 30.0;
    // Two passing candidates further apart than this are ambiguous (symmetric
    // scene): the whole attempt is rejected, including the KISS fallback.
    double ambiguity_trans = 1.0;
    double ambiguity_rot_deg = 10.0;
  } fc;

  /// Automatic scene-change map update (params map_update.*).
  /// Three-layer architecture:
  ///   L0: immutable base map anchors (ground and high walls locked)
  ///   L1: persistent static additions + carved base tombstones
  ///   L2: transient vehicle.* and driverseat points (never persisted)
  struct MapUpdate {
    bool enable = true;
    // Keyframe & tracking quality gate before feeding observations to L1.
    double kf_trans_thresh = 0.25;         // m, min translation between update keyframes
    double kf_rot_thresh_deg = 5.0;        // deg, min rotation between update keyframes
    double min_match_ratio = 0.45;         // matched / downsampled scan points
    int min_matched_points = 30;           // minimum matched plane points
    double min_base_ratio = 0.50;          // fraction of matched points supported by L0 base map
    int min_base_matched = 20;             // minimum L0-supported matched points
    double max_speed = 1.5;                // m/s
    double max_angular_vel = 0.6;          // rad/s
    bool pause_during_driverseat = true;   // freeze L1 while /driverseat_in is collecting
    double vehicle_exclude_margin = 0.5;   // m, extra exclusion buffer around active vehicle box

    // Adaptive 2D local ground Z estimation & vertical anchor protection band
    // (heights are relative to estimated local ground Z, never assuming z = 0).
    double ground_grid_res = 1.5;          // m, XY cell size for ground Z grid
    double max_floor_variation = 1.5;      // m, max floor height step / range from global floor mode
    double obs_z_min = 0.10;               // m above local ground: below this L0 is locked ground
    double obs_z_max = 2.00;               // m above local ground: above this L0 is locked upper wall

    // Range limits and multi-keyframe evidence for addition & free-space carving.
    double min_range = 1.0;                // m from LiDAR optical center
    double max_add_range = 8.0;            // m, max range to confirm new static objects
    double max_carve_range = 9.0;          // m, max ray traversal distance for carving
    int confirm_hits = 5;                  // distinct keyframes required to confirm an added sub-voxel
    double confirm_min_age_sec = 2.0;      // s, min time span from first hit to reject moving objects
    int carve_confirm_misses = 8;          // distinct keyframes required to carve an L0 sub-voxel
    int carve_added_misses = 5;            // distinct keyframes required to carve an L1-added sub-voxel
    double carve_margin = 0.35;            // m, safety margin before the distant hit point
    double carve_ray_radius = 0.12;        // m, max point-to-ray perpendicular distance for a miss
    double carve_min_baseline = 0.25;      // m, min sensor translation spread across carving rays
    double pending_ttl_sec = 30.0;         // s, expire unconfirmed candidates not seen recently

    // Large-update commit trigger (persists global.pcd, syncs reloc targets + map.pgm).
    int min_changed_voxels = 40;           // total uncommitted 0.25 m sub-voxels (added + carved)
    int min_cluster_voxels = 16;           // OR single connected dirty cluster size (~1 m^2 object)
    double commit_cooldown_sec = 30.0;     // s, minimum interval between background commits
    double max_commit_defer_sec = 20.0;    // s, wait up to this long for standstill before committing

    // 2D occupancy grid (map.pgm + map.yaml) synchronization.
    struct Pgm {
      bool enable = true;
      bool create_if_missing = true;       // generate map.pgm + map.yaml if absent on commit
      double resolution = 0.05;            // m/px when creating a new map.pgm
      double z_min = 0.10;                 // m above local ground: obstacle band lower bound
      double z_max = 1.50;                 // m above local ground: obstacle band upper bound
      std::string reload_service = "/map_server/load_map";  // empty disables Nav2 service call
    } pgm;
  } map_update;
};

}  // namespace LI2Sup

#endif
