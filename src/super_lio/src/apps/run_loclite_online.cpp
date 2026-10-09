#include <iostream>
#include <string>
#include <vector>
#include "lio/super_lio_reloc.h"

int main(int argc, char** argv) {
  std::string config, map_path;
  std::vector<std::string> args{argv[0]};
  try {
    bool ros_args = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--ros-args") ros_args = true;
      if (!ros_args && (arg == "--help" || arg == "-h")) {
        std::cout << "Usage: ros2 run super_lio run_loclite_online --config <ROS parameter YAML> "
                     "[--map_path <directory containing global.pcd>] [--ros-args ...]\n";
        return 0;
      }
      if (!ros_args && (arg == "--config" || arg == "--map_path" ||
          arg.rfind("--config=", 0) == 0 || arg.rfind("--map_path=", 0) == 0)) {
        const auto equal = arg.find('=');
        const auto flag = arg.substr(0, equal);
        std::string value;
        if (equal != std::string::npos) value = arg.substr(equal + 1);
        else if (i + 1 < argc) value = argv[++i];
        if (value.empty() || value.rfind("--", 0) == 0)
          throw std::runtime_error("Missing value for " + flag);
        (flag == "--config" ? config : map_path) = value;
      } else args.push_back(arg);
    }
    if (config.empty()) throw std::runtime_error("--config is required (use config/loclite_livox.yaml)");
    // Load the file first; ordinary ROS command-line overrides follow it.
    args.insert(args.begin() + 1, {"--ros-args", "--params-file", config, "--"});
    if (!map_path.empty()) args.insert(args.end(), {"--ros-args", "-p", "system.map_path:=" + map_path});
    std::vector<char*> raw;
    for (auto& arg : args) raw.push_back(arg.data());
    rclcpp::init(static_cast<int>(raw.size()), raw.data());
    LI2Sup::g_flag_run = true;
    auto wrapper = std::make_shared<LI2Sup::ROSWrapper>(rclcpp::NodeOptions(), true);
    auto lio = std::make_shared<LI2Sup::SuperLIOReLoc>();
    lio->setROSWrapper(wrapper);
    lio->init();
    if (!LI2Sup::g_flag_run) throw std::runtime_error("Failed to load fixed map");
    auto timer = wrapper->create_wall_timer(std::chrono::milliseconds(2),
        [lio]() { lio->process(); }, wrapper->getSensorCallbackGroup());
    rclcpp::spin(wrapper);
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "super_lio localization: " << error.what() << '\n';
    if (rclcpp::ok()) rclcpp::shutdown();
    return 1;
  }
}
