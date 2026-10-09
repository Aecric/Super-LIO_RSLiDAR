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

与 hikari_loclite 的在线命令相比，仅包名改成 `super_lio`；`--config <yaml>`、`--config=<yaml>`、`--map_path <dir>` 和 `--map_path=<dir>` 均支持。配置参数请放在 `--ros-args` 之前。地图目录应包含 `global.pcd`。

```bash
ros2 launch super_lio loclite.launch.py \
  config:=/path/to/loclite_livox.yaml \
  map_path:=/path/to/map_dir
```

地图路径优先级：CLI / launch 在的 `map_path` → 配置 `system.map_path` 下的 `global.pcd` → `fixed_map.global_pcd`。路径以当前工作目录解析，相对路径不会拼接源码目录；部署建议使用绝对路径。地图不存在、为空或加载失败时进程非零退出。

配置采用 **Super-LIO 的 ROS 2 YAML 格式**（`/**: ros__parameters:`），不能直接传入 hikari_loclite 的原始 YAML。使用本包 `config/loclite_livox.yaml`，按传感器调整 `lio.sensor.*`、`lio.ros.*`、`lio.extrinsic.lidar_imu`。后者表示 IMU ← LiDAR，平移单位为米，旋转矩阵沿用 Super-LIO 原有排列约定。

## 初始化与恢复

启动后处于 `WAIT_FOR_INITIALPOSE`。保持设备静止，通过 RViz 的 2D Pose Estimate 或 `/initialpose` 给出 **map ← lidar_frame** 初值。初始化累积 10 帧点云与 IMU，执行 NDT + ICP 验证，成功后开始跟踪。初始位置需要接近真实位置；此模式不包含 KISS / Scan Context 无初值全局搜索。

```bash
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
  '{header: {frame_id: map}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}'
```

`/initialpose` 的 z 和完整四元数都会保留；非法数值、零四元数、非地图 frame 会被拒绝。运行中可再次发送初值，暂停输出并重新验证。`lio.relocation.init_pose` 是旧重定位入口的配置，纯定位模式使用 `/initialpose`。

`/super_lio/set_active` 服务可停用/启用传感器订阅；重新启用后重新发送 `/initialpose`，固定地图仍保留。有效匹配不足或输入掉线后停止发布跟踪位姿，转入等待初值恢复流程。

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

纯定位不发布旧 `/lio/*` 输出；SC 专用调试 topic、手动全局重定位服务及 `run_loclite_offline` 不在此模式中。可使用普通 rosbag 回放验证：

```bash
ros2 launch super_lio loclite.launch.py map_path:=/path/to/map_dir use_sim_time:=true
# 另一个终端；初始化时保持对应 bag 段静止，并发送 /initialpose
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

测试覆盖 CLI 错误处理、固定地图晚订阅、等待初值、真实 NDT/ICP 初始化、雷达位姿/TF/状态接口、重复初值、停用/重新启用和地图文件不变。真实场景中的精度、动态干扰与长时间漂移仍需用对应地图和 rosbag 评估。
