"""Subscribe to clouds, run the existing sequential detector, publish results."""
from pathlib import Path
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from obstacle_detector import Config, StreamProcessor
from .conversion import SequenceGuard, cloud_arrays, header_dict, json_text, result_dict
from .ros_support import cloud_qos, make_cloud, parameter, result_qos, run_node


class ObstacleProcessor(Node):
    def __init__(self, **kwargs):
        super().__init__('obstacle_processor', **kwargs)
        config = Config(threads=parameter(self, 'threads', 3),
                        rise_m=parameter(self, 'rise_m', 0.10),
                        min_unique=parameter(self, 'min_unique', 3),
                        self_return_m=parameter(self, 'self_return_m', 2.5))
        self.processor = StreamProcessor(config)
        self.sequence = SequenceGuard(parameter(self, 'max_gap_sec', 0.25))
        self.index = 0
        cloud_topic = parameter(self, 'cloud_topic', '/lidar_points')
        qos = cloud_qos(parameter(self, 'input_reliability', 'reliable'),
                        parameter(self, 'queue_depth', 10))
        self.results = self.create_publisher(String,
            parameter(self, 'result_topic', '/obstacle_detector/result'), result_qos())
        self.corridor = self.create_publisher(PointCloud2,
            parameter(self, 'corridor_topic', '/obstacle_detector/corridor'), cloud_qos())
        self.obstacles = self.create_publisher(PointCloud2,
            parameter(self, 'obstacles_topic', '/obstacle_detector/obstacles'), cloud_qos())
        self.subscription = self.create_subscription(PointCloud2, cloud_topic, self.on_cloud, qos)
        self.output = None
        output_path = parameter(self, 'output_jsonl', '')
        if output_path:
            path = Path(output_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.output = path.open('x', encoding='utf-8')
        self.get_logger().info(f'Ready for PointCloud2 on {cloud_topic}; coordinates: forward -Y, up +Z')

    def on_cloud(self, message):
        reset_reason = self.sequence.update(message.header)
        if reset_reason:
            self.processor.reset()
            self.get_logger().warning(f'Temporal state reset: {reset_reason}')
        try:
            xyz, ring, removed = cloud_arrays(message)
            if not len(xyz):
                raise ValueError('The cloud has no finite XYZ points')
            result = self.processor.process(xyz, ring, header_dict(message.header)['timestamp_ns'])
            payload = result_dict(result, message.header, self.index,
                                  removed_points=removed, reset_reason=reset_reason)
            corridor = make_cloud(result.region.points, message.header)
            chunks = [t.observed_points for t in result.blocking if len(t.observed_points)]
            points = np.concatenate(chunks) if chunks else np.empty((0, 3))
            obstacles = make_cloud(points, message.header)
        except Exception as exc:
            # A failed frame cannot keep evidence accumulated before the gap.
            self.processor.reset()
            payload = dict(schema_version='1.0', header=header_dict(message.header),
                           frame_index=self.index, processing_ok=False, status='CAUTION',
                           path_clear=None, distance_m=None, error=str(exc))
            corridor = make_cloud(np.empty((0, 3)), message.header)
            obstacles = make_cloud(np.empty((0, 3)), message.header)
            self.get_logger().error(f'Frame {self.index}: {exc}')
        text = json_text(payload)
        if self.output is not None:
            self.output.write(text + '\n')
            self.output.flush()
        self.corridor.publish(corridor)
        self.obstacles.publish(obstacles)
        # This result also acknowledges replay, after all outputs are ready.
        self.results.publish(String(data=text))
        self.index += 1

    def destroy_node(self):
        if self.output is not None:
            self.output.close()
        return super().destroy_node()


def main(args=None):
    run_node(ObstacleProcessor, args)
