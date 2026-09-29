"""Native ROS helpers; imports stay out of the offline CLI."""
from copy import deepcopy
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from .conversion import packed_cloud


def parameter(node, name, default):
    return node.declare_parameter(name, default, ParameterDescriptor(read_only=True)).value


def cloud_qos(reliability='reliable', depth=10):
    if reliability not in ('reliable', 'best_effort') or depth < 1:
        raise ValueError('Use reliability reliable/best_effort and queue_depth >= 1')
    return QoSProfile(depth=depth, reliability=(ReliabilityPolicy.RELIABLE
        if reliability == 'reliable' else ReliabilityPolicy.BEST_EFFORT))


def result_qos():
    return QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


def make_cloud(xyz, header, ring=None, *, float64=False, allow_nonfinite=False):
    data, step = packed_cloud(xyz, ring, float64=float64, allow_nonfinite=allow_nonfinite)
    width = 8 if float64 else 4
    code = PointField.FLOAT64 if float64 else PointField.FLOAT32
    fields = [PointField(name=name, offset=i*width, datatype=code, count=1)
              for i, name in enumerate(('x', 'y', 'z'))]
    if ring is not None:
        fields.append(PointField(name='ring', offset=3*width, datatype=PointField.UINT16, count=1))
    return PointCloud2(header=deepcopy(header), height=1, width=len(xyz), fields=fields,
                       is_bigendian=False, point_step=step, row_step=step*len(xyz),
                       data=data, is_dense=bool(np.isfinite(xyz).all()))


def run_node(factory, args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = factory()
        while rclpy.ok() and not getattr(node, 'done', False):
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().error(str(exc))
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
