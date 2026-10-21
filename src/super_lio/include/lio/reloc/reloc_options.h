#ifndef SUPER_LIO_RELOC_OPTIONS_H_
#define SUPER_LIO_RELOC_OPTIONS_H_

#include <cstddef>
#include <vector>

namespace LI2Sup {

/// Fixed-map initialization pipeline (localization mode only):
///   /initialpose : guess                      -> NDT coarse -> GenZ-ICP -> FC
///   cold start   : 1. map-origin prior (yaw sweep) -> NDT coarse -> GenZ-ICP -> FC
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
};

}  // namespace LI2Sup

#endif
