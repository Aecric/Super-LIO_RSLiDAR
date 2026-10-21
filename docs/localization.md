# 固定地图纯定位模式

该模式使用 Super-LIO 的 ESKF / OctVoxMap 跟踪已有 PCD 地图，入口、核心输出 topic 和 TF 坐标含义参照 [hikari_loclite](https://github.com/Aecric/hikari_loclite)。建图仍使用原有 `super_lio_node`。纯定位入口强制关闭地图更新、地图保存、关键帧发布和回环后端，即使配置误开这些开关也不会修改固定地图。

## 构建与运行

在 Super-LIO 仓库根目录执行：

```bash
source /opt/ros/<ros_distro>/setup.bash
colcon build --base-paths src --packages-up-to super_lio --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

ros2 run super_lio run_loclite_online \
  --config "$(ros2 pkg prefix --share super_lio)/config/loclite_livox.yaml" \
  --map_path /path/to/map_dir
```

冷启动依赖的 KISS-Matcher 与 GenZ-ICP 已 vendored 在 `src/super_lio/thirdparty/`；首次 CMake 配置会从 GitHub 下载 KISS-Matcher 的依赖 ROBIN（含 PMC / xenium），需要联网。

与 hikari_loclite 的在线命令相比，仅包名改成 `super_lio`；`--config <yaml>`、`--config=<yaml>`、`--map_path <dir>` 和 `--map_path=<dir>` 均支持。配置参数请放在 `--ros-args` 之前。地图目录应包含 `global.pcd`。

```bash
ros2 launch super_lio loclite.launch.py \
  config:=/path/to/loclite_livox.yaml \
  map_path:=/path/to/map_dir
```

地图路径优先级：CLI / launch 在的 `map_path` → 配置 `system.map_path` 下的 `global.pcd` → `fixed_map.global_pcd`。路径以当前工作目录解析，相对路径不会拼接源码目录；部署建议使用绝对路径。地图不存在、为空或加载失败时进程非零退出。

配置采用 **Super-LIO 的 ROS 2 YAML 格式**（`/**: ros__parameters:`），不能直接传入 hikari_loclite 的原始 YAML。使用本包 `config/loclite_livox.yaml`，按传感器调整 `lio.sensor.*`、`lio.ros.*`、`lio.extrinsic.lidar_imu`。后者表示 IMU ← LiDAR，平移单位为米，旋转矩阵沿用 Super-LIO 原有排列约定。

## 初始化与恢复

初始化在设备**静止**时累积 10 帧点云与 IMU（重力拉平），对每个候选位姿执行 NDT 粗配 → [GenZ-ICP](https://github.com/cocel-postech/genz-icp) 精配 → **FC 假阳性屏障**，通过后开始跟踪。候选来源：

1. **`/initialpose`**（RViz 2D Pose Estimate）：给出 **map ← lidar_frame** 初值，随时可发，会打断正在进行的冷启动。
2. **冷启动**（没有初值时自动进行：启动、LOST 之后、重新启用之后）：
   - 先假设设备在**地图原点（建图起点）**附近：在 `reloc.origin_prior.{x,y,z}`（默认 0）按 `yaw_step_deg`（默认 45°，共 8 个）枚举 yaw 假设；
   - 原点假设都未通过 FC 时，再用 [KISS-Matcher](https://github.com/MIT-SPARK/KISS-Matcher) 对整张固定图做无先验全局配准（Quatro 4-DoF，每个 `reloc.kiss.voxel_sizes` 出一个候选）。

FC 对精配结果逐项检查，全部通过才接受：0.3 m 内点率 ≥ 0.85、墙面带（传感器高度 −0.5～+1.0 m，排除地面和天花板）内点率 ≥ 0.8、有观测的 45° 扇区覆盖率 ≥ 0.5、内点 RMSE ≤ 0.08 m、相对候选初值的位移/转角不超限。若多个候选都通过但彼此相距 > 1 m 或 > 10°，判为对称歧义，整个窗口放弃（不再退到 KISS）并用下一窗口重试；KISS 还要求至少 `reloc.kiss.min_agreeing_candidates`（默认 2）个候选通过且一致，避免退化场景里单个候选碰巧对上。冷启动失败会持续重试（`reloc.cold_start.max_attempts`，0 = 不限），运动中累积的点云通常过不了 FC，因此设备需静止等待；状态为 1（INIT）。日志前缀 `[FC-Init]` 会列出每个候选的各项指标。

默认阈值来自一条 Mid360 bag 与同一次建图的地图（真匹配 1.0 / 1.0 / 0.05 m，最强错匹配 0.875 / 0.77 / 0.10 m）；跨天、场景有变化的地图会让真匹配内点率下降，请用实际地图 + bag 复核后再调 `reloc.fc.*`。整图 KISS 的目标点数上限为 `reloc.kiss.max_target_points`（0.2 m 降采样后），超出时只用原点先验。

```bash
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
  '{header: {frame_id: map}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}'
```

`/initialpose` 的 z 和完整四元数都会保留；非法数值、零四元数、非地图 frame 会被拒绝。运行中可再次发送初值，暂停输出并重新验证。`lio.relocation.init_pose` 是旧重定位入口的配置，纯定位模式使用 `/initialpose`。

`~/set_active` 服务可停用/启用传感器订阅（`loclite.launch.py` 启动时节点名为 `super_lio_loc`，即 `/super_lio_loc/set_active`；`ros2 run` 直接启动时为 `/super_lio/set_active`，与同样用 `ros2 run` 启动的建图节点同名，同时运行时需用 `--ros-args -r __node:=<name>` 改名）；重新启用后重新发送 `/initialpose`，固定地图仍保留。有效匹配不足或输入掉线后停止发布跟踪位姿，转入等待初值恢复流程。

## 输出约定

| Topic | 类型 | 内容 |
|---|---|---|
| `/hikari_loc/odom` | `nav_msgs/msg/Odometry` | map ← lidar_frame；线速度在 lidar_frame 中表达 |
| `/hikari_loc/path` | `nav_msgs/msg/Path` | map 下雷达轨迹；最小间隔 0.1 s，最多 5000 点，重新初始化时清空 |
| `/pcdmap` | `sensor_msgs/msg/PointCloud2` | map 下固定地图，可靠、transient-local、深度 1，支持晚订阅 |
| `/hikari_loc/loc_state` | `std_msgs/msg/Int32` | 与 hikari_loclite 同一枚举编号 |
| `/hikari_loc/ndt_status` | `std_msgs/msg/Int32` | 固定 0，单层 NDT |
| `/hikari_loc/loc_status` | `visualization_msgs/msg/Marker` | 定位状态文字 |
| `/hikari_loc/status` | `std_msgs/msg/Float32MultiArray` | `[state, ndt_conf, imu_age_s, lidar_age_s, fps, in_map]` |

状态编号：0 未初始化、1 初始化中、2 GOOD、3 DEGRADED、4 LOST、5 WAIT_FOR_INITIALPOSE。本实现使用 1/2/4/5；GOOD 表示已通过初始化配准且跟踪有足够匹配，不代表 hikari_loclite 的完整稳定性/退化判定。`ndt_conf=-1` 表示不可用，不能把 PCL 的 ICP fitness 当成 hikari_loclite 的 NDT 置信度。未收到输入时 age 为 -1。状态以 5 Hz 发布；默认 IMU / 雷达掉线阈值为 1 / 2 秒。

纯定位不发布旧 `/lio/*` 输出；SC 专用调试 topic、手动全局重定位服务及 `run_loclite_offline` 不在此模式中。可使用普通 rosbag 回放验证（从设备静止的时刻开始回放即可触发冷启动）：

```bash
ros2 launch super_lio loclite.launch.py map_path:=/path/to/map_dir use_sim_time:=true
# 另一个终端；初始化时保持对应 bag 段静止，冷启动自动进行，也可发送 /initialpose
ros2 bag play /path/to/bag --clock
```

## TF

默认配置：

```text
map → livox_frame → level_frame
```

- `common.map_frame_id` 默认 `map`。
- `system.lidar_frame_id` 默认 `livox_frame`，也是 odom 的 child frame，不随输入点云 header 自动改变。
- `system.base_frame_id` 默认 `livox_frame`。若改成 `base_link`，动态 TF 为 `map → base_link`，计算式为 `T_map_lidar × inverse(T_base_lidar)`。外参由 `system.base_to_lidar_{x,y,z,roll,pitch,yaw}` 指定，角度单位为弧度。请由机器人已有 TF 发布器提供与该外参一致的 `base_link → livox_frame` 静态变换。
- `livox_frame → level_frame` 是零平移旋转，去掉 roll/pitch、保留地图中的 yaw。这里按滤波后的雷达姿态直接计算，未移植 hikari_loclite 的独立重力低通滤波器。
- TF、odom 使用雷达帧末时间戳。此模式不做 IMU 高频 TF 外推。重初始化和 LOST 期间不发布新位姿。

## 验证

构建并加载环境后，在独立 ROS domain 运行合成地图集成测试：

```bash
ROS_DOMAIN_ID=173 python3 src/super_lio/test/localization_smoke.py
```

测试覆盖 CLI 错误处理、固定地图晚订阅、等待初值、`/initialpose` 的 NDT/GenZ-ICP/FC 初始化、原点先验冷启动、雷达位姿/TF/状态接口、重复初值、停用/重新启用和地图文件不变。真实场景中的精度、动态干扰与长时间漂移仍需用对应地图和 rosbag 评估。
