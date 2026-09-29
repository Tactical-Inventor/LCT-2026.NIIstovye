"""These tests require native ROS 2; Docker's `test` target runs all of them."""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy', reason='Native ROS 2 required; run docker build --target test .')
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.serialization import serialize_message, deserialize_message
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String
from obstacle_detector_ros.conversion import cloud_arrays, header_dict
from obstacle_detector_ros.loader_node import LidarLoader
from obstacle_detector_ros.processor_node import ObstacleProcessor
from obstacle_detector_ros.ros_support import cloud_qos, make_cloud, result_qos


def overrides(**values):
    return [Parameter(k, value=v) for k,v in values.items()]


def spin_until(executor, predicate, seconds=30):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
    assert predicate(), 'Timed out waiting for ROS transport'


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.setenv('ROS_DOMAIN_ID', str(20 + os.getpid() % 150))
    rclpy.init()
    executor = SingleThreadedExecutor()
    nodes = []
    def add(node):
        nodes.append(node)
        executor.add_node(node)
        return node
    try:
        yield executor, add
    finally:
        executor.shutdown()
        for node in reversed(nodes):
            node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize('float64', [False, True])
def test_native_cdr_and_ring_roundtrip(sample_xyz, float64):
    message = make_cloud(sample_xyz, Header(frame_id='lidar'), np.array([1,2,3,4], np.uint16), float64=float64)
    actual = deserialize_message(serialize_message(message), PointCloud2)
    xyz, ring, removed = cloud_arrays(actual)
    np.testing.assert_array_equal(xyz, sample_xyz)
    np.testing.assert_array_equal(ring, [1,2,3,4])
    assert removed == 0


@pytest.mark.parametrize('reliability', ['reliable', 'best_effort'])
def test_processor_topics_error_recovery_and_jsonl(graph, tmp_path, sample_xyz, reliability):
    executor, add = graph
    prefix = '/test_' + uuid.uuid4().hex
    output = tmp_path/'results.jsonl'
    add(ObstacleProcessor(parameter_overrides=overrides(
        cloud_topic=prefix+'/cloud', result_topic=prefix+'/result',
        corridor_topic=prefix+'/corridor', obstacles_topic=prefix+'/obstacles',
        output_jsonl=str(output), input_reliability=reliability)))
    peer = add(Node('test_observer'))
    results, corridors, obstacles = [], [], []
    pub = peer.create_publisher(PointCloud2, prefix+'/cloud', cloud_qos(reliability))
    peer.create_subscription(String, prefix+'/result', lambda m: results.append(json.loads(m.data)), result_qos())
    peer.create_subscription(PointCloud2, prefix+'/corridor', corridors.append, cloud_qos())
    peer.create_subscription(PointCloud2, prefix+'/obstacles', obstacles.append, cloud_qos())
    spin_until(executor, lambda: pub.get_subscription_count() == 1)
    header = Header(frame_id='lidar')
    header.stamp.sec = 7
    invalid = make_cloud(sample_xyz, header)
    invalid.fields = []
    pub.publish(invalid)
    spin_until(executor, lambda: len(results) == len(corridors) == len(obstacles) == 1)
    assert results[0]['processing_ok'] is False and results[0]['status'] == 'CAUTION'
    assert results[0]['path_clear'] is None and obstacles[0].width == 0
    header.stamp.nanosec = 100_000_000
    pub.publish(make_cloud(sample_xyz, header))
    spin_until(executor, lambda: len(results) == len(corridors) == len(obstacles) == 2)
    assert results[1]['processing_ok'] is True and results[1]['status'] == 'CAUTION'
    assert results[1]['header'] == header_dict(header)
    assert header_dict(corridors[1].header) == header_dict(header)
    assert [json.loads(line) for line in output.read_text().splitlines()] == results
    # A late result subscriber gets the latest decision through transient-local QoS.
    late = []
    peer.create_subscription(String, prefix+'/result', late.append, result_qos())
    spin_until(executor, lambda: bool(late))
    assert json.loads(late[-1].data) == results[-1]


def test_loader_waits_for_each_result_including_last(graph, tmp_path, bag_factory):
    executor, add = graph
    bag, _ = bag_factory(tmp_path/'source.db3')
    prefix = '/test_' + uuid.uuid4().hex
    loader = add(LidarLoader(parameter_overrides=overrides(
        input=str(bag), cloud_topic=prefix+'/cloud', result_topic=prefix+'/result', rate_hz=100.0)))
    peer = add(Node('test_acknowledger'))
    clouds = []
    peer.create_subscription(PointCloud2, prefix+'/cloud', clouds.append, cloud_qos())
    ack = peer.create_publisher(String, prefix+'/result', result_qos())
    spin_until(executor, lambda: len(clouds) == 1 and ack.get_subscription_count() == 1)
    for _ in range(15):
        executor.spin_once(timeout_sec=0.02)
    assert len(clouds) == 1 and not loader.done
    for index in range(3):
        spin_until(executor, lambda: len(clouds) == index+1)
        assert header_dict(clouds[index].header)['timestamp_ns'] == 1_000_000_000 + index*100_000_000
        assert clouds[index].header.frame_id == 'test_lidar'
        assert not loader.done
        ack.publish(String(data=json.dumps(dict(header=header_dict(clouds[index].header), processing_ok=True))))
    spin_until(executor, lambda: loader.done)
    assert len(clouds) == 3


@pytest.mark.parametrize('source_kind', ['db3', 'npz'])
def test_installed_launch_replays_all_frames_and_exits(tmp_path, bag_factory, sample_xyz, source_kind):
    if source_kind == 'db3':
        source, _ = bag_factory(tmp_path/'source.db3')
        expected = 3
    else:
        source = tmp_path/'source.npz'
        np.savez(source, xyz=sample_xyz, ring=np.arange(len(sample_xyz), dtype=np.uint16))
        expected = 1
    output = tmp_path/'results.jsonl'
    env = dict(os.environ, ROS_DOMAIN_ID=str(20 + os.getpid() % 150))
    process = subprocess.run([
        'ros2', 'launch', 'obstacle_detector_ros', 'pipeline.launch.py',
        f'input:={source}', f'output_jsonl:={output}', 'timeout_sec:=60.0'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=100)
    assert process.returncode == 0, process.stdout + process.stderr
    results = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(results) == expected
    assert [r['frame_index'] for r in results] == list(range(expected))
    assert all(r['processing_ok'] and r['status']=='CAUTION' for r in results)
    if source_kind == 'db3':
        assert [r['header']['timestamp_ns'] for r in results] == [1_000_000_000,1_100_000_000,1_200_000_000]


def test_installed_launch_returns_failure_for_missing_input(tmp_path):
    process = subprocess.run([
        'ros2', 'launch', 'obstacle_detector_ros', 'pipeline.launch.py',
        f'input:={tmp_path / "missing.db3"}'], cwd=tmp_path,
        capture_output=True, text=True, timeout=30)
    assert process.returncode != 0, process.stdout + process.stderr
