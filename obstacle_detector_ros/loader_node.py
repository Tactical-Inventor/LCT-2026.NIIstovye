"""Load DB3/NPY/NPZ clouds and publish native sensor_msgs/PointCloud2."""
import json
import math
import time
import numpy as np
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String
from obstacle_detector.io import FrameSource
from .conversion import header_dict
from .ros_support import cloud_qos, make_cloud, parameter, result_qos, run_node


class LidarLoader(Node):
    def __init__(self, **kwargs):
        super().__init__('lidar_loader', **kwargs)
        path = parameter(self, 'input', '')
        if not path:
            raise ValueError('Set input to a DB3 file, bag directory, NPY or NPZ file')
        topic = parameter(self, 'bag_topic', '')
        cloud_topic = parameter(self, 'cloud_topic', '/lidar_points')
        result_topic = parameter(self, 'result_topic', '/obstacle_detector/result')
        self.frame_id = parameter(self, 'frame_id', '')
        rate = parameter(self, 'rate_hz', 10.0)
        self.timeout = parameter(self, 'timeout_sec', 300.0)
        limit = parameter(self, 'max_frames', 0)
        self.wait_result = parameter(self, 'wait_for_result', True)
        if not math.isfinite(rate) or rate <= 0 or not math.isfinite(self.timeout) or self.timeout <= 0 or limit < 0:
            raise ValueError('rate_hz and timeout_sec must be positive; max_frames >= 0')
        self.period = 1.0 / rate
        self.publisher = self.create_publisher(PointCloud2, cloud_topic, cloud_qos())
        self.result_topic = result_topic
        self.result_subscription = self.create_subscription(String, result_topic, self.on_result, result_qos())
        self.source = FrameSource(path, topic=topic or None)
        self.count = min(len(self.source), limit) if limit else len(self.source)
        self.index = 0
        self.pending = None
        self.done = False
        self.wait_started = time.monotonic()
        self.next_publish = self.wait_started
        self.timer = self.create_timer(min(self.period, 0.05), self.tick)
        self.get_logger().info(f'Loaded {self.count} clouds; publishing to {cloud_topic}')

    def on_result(self, message):
        if self.pending is None:
            return
        try:
            result = json.loads(message.data)
        except (ValueError, TypeError):
            return
        if not isinstance(result, dict) or result.get('header') != self.pending:
            return
        if result.get('processing_ok') is not True:
            raise RuntimeError(f"Processor rejected frame {self.index-1}: {result.get('error')}")
        self.pending = None
        self.wait_started = time.monotonic()

    def tick(self):
        now = time.monotonic()
        if self.pending is not None:
            if now - self.wait_started > self.timeout:
                raise TimeoutError(f'No processing result for frame {self.index-1} within {self.timeout}s')
            return
        if self.index >= self.count:
            self.get_logger().info(f'Completed {self.index} clouds')
            self.timer.cancel()
            self.done = True
            return
        connected = self.publisher.get_subscription_count() > 0
        if self.wait_result:
            connected = connected and self.count_publishers(self.result_topic) > 0
        if not connected:
            if now - self.wait_started > self.timeout:
                raise TimeoutError('No compatible cloud subscriber / result publisher discovered')
            return
        if now < self.next_publish:
            return
        if self.source.single is None:
            stamp, raw = self.source.read_serialized(self.index)
            message = deserialize_message(raw, PointCloud2)
            # Match the recording timestamp used by the offline detector.
            message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        else:
            frame = self.source.read(self.index)
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            message = make_cloud(frame.xyz, header, frame.ring,
                                 float64=np.issubdtype(frame.xyz.dtype, np.float64), allow_nonfinite=True)
        message.header.frame_id = self.frame_id or message.header.frame_id or 'lidar'
        if self.wait_result:
            self.pending = header_dict(message.header)
        self.publisher.publish(message)
        self.index += 1
        self.wait_started = now
        self.next_publish = now + self.period

    def destroy_node(self):
        self.source.close()
        return super().destroy_node()


def main(args=None):
    run_node(LidarLoader, args)
