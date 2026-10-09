#!/usr/bin/env python3
"""ROS integration smoke test. Run after sourcing a built super_lio workspace.

Creates a synthetic fixed map and stationary Livox/IMU stream in a temporary
folder. No robot or real map is needed. Use an isolated ROS_DOMAIN_ID.
"""
import math
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest

import rclpy
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from geometry_msgs.msg import PoseWithCovarianceStamped
from livox_ros_driver2.msg import CustomMsg, CustomPoint
from nav_msgs.msg import Odometry, Path as RosPath
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Int32, Float32MultiArray
from std_srvs.srv import SetBool
from tf2_msgs.msg import TFMessage


class LocalizationSmoke(unittest.TestCase):
    def test_launch_map_override(self):
        share = Path(get_package_share_directory('super_lio'))
        with tempfile.TemporaryDirectory(prefix='super-lio-launch-test-') as directory:
            config = Path(directory) / 'loclite.yaml'
            config.write_text((share / 'config/loclite_livox.yaml').read_text().replace(
                'system.map_path: ""', 'system.map_path: "/nonexistent/config-map"'))
            (Path(directory) / 'global.pcd').write_text(
                'VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n'
                'TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA ascii\n1 2 3 4\n')
            with tempfile.TemporaryFile(mode='w+') as output:
                process = subprocess.Popen([
                    'ros2', 'launch', 'super_lio', 'loclite.launch.py',
                    'config:='+str(config), 'map_path:='+directory,
                    'use_sim_time:=true'], stdout=output, stderr=output)
                try:
                    time.sleep(3)
                    self.assertIsNone(process.poll())
                finally:
                    process.send_signal(signal.SIGINT)
                    process.wait(timeout=10)
                output.seek(0)
                text = output.read()
                self.assertIn('Map init success', text)
                self.assertNotIn('process has died', text)

    def test_fixed_map_session(self):
        executable = str(Path(get_package_prefix('super_lio')) / 'lib/super_lio/run_loclite_online')
        config = str(Path(get_package_share_directory('super_lio')) / 'config/loclite_livox.yaml')
        self.assertEqual(subprocess.run([executable, '--help'], capture_output=True).returncode, 0)
        self.assertNotEqual(subprocess.run([executable, '--config'], capture_output=True).returncode, 0)
        self.assertNotEqual(subprocess.run([executable, '--config', config, '--map_path',
                                           '/nonexistent/super-lio-map'], capture_output=True).returncode, 0)
        with tempfile.TemporaryDirectory(prefix='super-lio-loc-test-') as directory:
            # Three offset planes constrain all six pose dimensions.
            points = []
            for a in range(-20, 21):
                for b in range(-10, 21):
                    points.extend([(4.0, a * .13, b * .13),
                                   (a * .13, 3.5, b * .13),
                                   (a * .13, b * .13, -.8)])
            yaw = .3
            c, s = math.cos(yaw), math.sin(yaw)
            world = [(c*x-s*y+2, s*x+c*y-1, z+.6) for x, y, z in points]
            map_file = Path(directory) / 'global.pcd'
            map_file.write_text('VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n'
                                'TYPE F F F F\nCOUNT 1 1 1 1\n'
                                f'WIDTH {len(world)}\nHEIGHT 1\nPOINTS {len(world)}\nDATA ascii\n' +
                                ''.join(f'{x} {y} {z} 10\n' for x, y, z in world))
            original = map_file.read_bytes()
            log = open(Path(directory) / 'node.log', 'w+')
            process = subprocess.Popen([executable, '--config='+config, '--map_path='+directory,
                                       '--ros-args', '-p', 'system.base_frame_id:=base_link',
                                       '-p', 'system.base_to_lidar_x:=0.4'],
                                       stdout=log, stderr=log)
            rclpy.init()
            node = rclpy.create_node('localization_smoke')
            odoms, paths, transforms, states, maps, rich = [], [], [], [], [], []
            subscriptions = [
                node.create_subscription(Odometry, '/hikari_loc/odom', odoms.append, 10),
                node.create_subscription(RosPath, '/hikari_loc/path', paths.append, 10),
                node.create_subscription(TFMessage, '/tf', lambda msg: transforms.extend(msg.transforms), 100),
                node.create_subscription(Int32, '/hikari_loc/loc_state', lambda msg: states.append(msg.data), 10),
                node.create_subscription(Float32MultiArray, '/hikari_loc/status', rich.append, 10),
                node.create_subscription(PointCloud2, '/pcdmap', maps.append,
                    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
            ]
            imu_pub = node.create_publisher(Imu, '/livox/imu', 50)
            lidar_pub = node.create_publisher(CustomMsg, '/livox/lidar', 10)
            initial_pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 1)

            def spin_for(seconds, feed=False):
                end = time.monotonic() + seconds
                counter = 0
                while time.monotonic() < end:
                    self.assertIsNone(process.poll(), 'localization node exited')
                    if feed:
                        stamp = node.get_clock().now().to_msg()
                        imu = Imu()
                        imu.header.stamp = stamp
                        imu.linear_acceleration.z = 9.7946
                        imu_pub.publish(imu)
                        if counter % 5 == 0:
                            scan = CustomMsg()
                            scan.header.stamp = stamp
                            scan.header.frame_id = 'raw_sensor_frame'
                            scan.point_num = len(points)
                            scan.points = [CustomPoint(x=float(x), y=float(y), z=float(z),
                                reflectivity=10, tag=0, line=0,
                                offset_time=int(i * 50_000_000 / len(points)))
                                for i, (x, y, z) in enumerate(points)]
                            lidar_pub.publish(scan)
                        counter += 1
                    rclpy.spin_once(node, timeout_sec=.01)
                    time.sleep(.01)

            def initial():
                pose = PoseWithCovarianceStamped()
                pose.header.frame_id = 'map'
                pose.pose.pose.position.x = 2.
                pose.pose.pose.position.y = -1.
                pose.pose.pose.position.z = .6
                pose.pose.pose.orientation.z = math.sin(yaw/2)
                pose.pose.pose.orientation.w = math.cos(yaw/2)
                initial_pub.publish(pose)

            try:
                spin_for(2)
                self.assertIn(5, states)
                self.assertTrue(maps, 'late subscriber must receive transient-local map')
                self.assertEqual(maps[-1].header.frame_id, 'map')
                spin_for(.5, feed=True)
                self.assertFalse(odoms, 'must wait for initialpose')
                initial()
                spin_for(6, feed=True)
                self.assertTrue(odoms, 'registration should succeed')
                odom = odoms[-1]
                self.assertEqual((odom.header.frame_id, odom.child_frame_id), ('map', 'livox_frame'))
                self.assertAlmostEqual(odom.pose.pose.position.x, 2., delta=.15)
                self.assertAlmostEqual(odom.pose.pose.position.y, -1., delta=.15)
                self.assertAlmostEqual(odom.pose.pose.position.z, .6, delta=.15)
                self.assertTrue(paths)
                self.assertEqual(paths[-1].poses[-1].header.frame_id, 'map')
                self.assertIn(('map', 'base_link'), [(t.header.frame_id, t.child_frame_id) for t in transforms])
                base_tf = [t for t in transforms if t.child_frame_id == 'base_link'][-1]
                self.assertAlmostEqual(base_tf.transform.translation.x, 2. - .4*c, delta=.15)
                self.assertAlmostEqual(base_tf.transform.translation.y, -1. - .4*s, delta=.15)
                self.assertIn(('livox_frame', 'level_frame'), [(t.header.frame_id, t.child_frame_id) for t in transforms])
                self.assertNotIn('world', [t.header.frame_id for t in transforms])
                self.assertEqual(len(rich[-1].data), 6)
                self.assertEqual(rich[-1].data[1], -1.)
                self.assertIn(2, states)
                count = len(odoms)
                initial()
                spin_for(4, feed=True)
                self.assertGreater(len(odoms), count, 'repeated initialpose must reacquire')
                client = node.create_client(SetBool, '/super_lio/set_active')
                self.assertTrue(client.wait_for_service(timeout_sec=2))
                for active in [False, True]:
                    future = client.call_async(SetBool.Request(data=active))
                    rclpy.spin_until_future_complete(node, future, timeout_sec=2)
                    self.assertTrue(future.result().success)
                    spin_for(.3)
                count = len(odoms)
                initial()
                spin_for(4, feed=True)
                self.assertGreater(len(odoms), count, 'reactivation must preserve fixed map')
                spin_for(2.5)
                self.assertTrue(4 in states[-15:] or 5 in states[-15:], 'watchdog must leave GOOD')
                self.assertEqual(map_file.read_bytes(), original)
                self.assertEqual(list(Path(directory).glob('*.pcd')), [map_file])
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read()[-12000:])
                raise
            finally:
                node.destroy_node()
                rclpy.shutdown()
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()


if __name__ == '__main__':
    unittest.main()
