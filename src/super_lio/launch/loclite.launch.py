"""Fixed-map localization with hikari_loclite-compatible command arguments."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_node(context):
    arguments = ['--config', LaunchConfiguration('config').perform(context)]
    map_path = LaunchConfiguration('map_path').perform(context)
    if map_path:
        arguments += ['--map_path', map_path]
    return [Node(
        package='super_lio', executable='run_loclite_online', output='screen',
        arguments=arguments,
        parameters=[{'use_sim_time': ParameterValue(
            LaunchConfiguration('use_sim_time'), value_type=bool)}])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=os.path.join(
            get_package_share_directory('super_lio'), 'config', 'loclite_livox.yaml')),
        DeclareLaunchArgument('map_path', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        OpaqueFunction(function=launch_node),
    ])
