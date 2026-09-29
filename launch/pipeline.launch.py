"""Replay with backpressure, or process an external live PointCloud2 topic."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def value(name, kind=str):
    return ParameterValue(LaunchConfiguration(name), value_type=kind)


def finished(event, context):
    if event.returncode != 0:
        raise RuntimeError(f'ROS node exited with code {event.returncode}')
    return [EmitEvent(event=Shutdown(reason='Processing finished'))]


def generate_launch_description():
    defaults = dict(input='', bag_topic='', cloud_topic='/lidar_points',
                    result_topic='/obstacle_detector/result', frame_id='',
                    rate_hz='10.0', max_frames='0', timeout_sec='300.0',
                    output_jsonl='', threads='3', start_loader='true',
                    input_reliability='reliable', max_gap_sec='0.25')
    arguments = [DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()]
    processor = Node(package='obstacle_detector_ros', executable='obstacle_processor',
                     name='obstacle_processor', output='screen', parameters=[{
                         'cloud_topic': value('cloud_topic'), 'result_topic': value('result_topic'),
                         'output_jsonl': value('output_jsonl'), 'threads': value('threads', int),
                         'input_reliability': value('input_reliability'),
                         'max_gap_sec': value('max_gap_sec', float)}])
    loader = Node(package='obstacle_detector_ros', executable='lidar_loader',
                  name='lidar_loader', output='screen', condition=IfCondition(LaunchConfiguration('start_loader')),
                  parameters=[{'input': value('input'), 'bag_topic': value('bag_topic'),
                               'cloud_topic': value('cloud_topic'), 'result_topic': value('result_topic'),
                               'frame_id': value('frame_id'), 'rate_hz': value('rate_hz', float),
                               'max_frames': value('max_frames', int),
                               'timeout_sec': value('timeout_sec', float)}])
    return LaunchDescription(arguments + [
        RegisterEventHandler(OnProcessExit(target_action=loader, on_exit=finished)),
        RegisterEventHandler(OnProcessExit(target_action=processor, on_exit=finished)),
        processor, loader])
