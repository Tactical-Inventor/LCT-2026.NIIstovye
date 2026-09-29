import sqlite3
from types import SimpleNamespace
import numpy as np
import pytest
from rosbags.typesys import Stores, get_typestore


@pytest.fixture
def typestore():
    return get_typestore(Stores.ROS2_HUMBLE)


def header(stamp=1_000_000_000, frame_id='lidar'):
    sec, nanosec = divmod(stamp, 1_000_000_000)
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=frame_id)


@pytest.fixture
def sample_xyz():
    # A valid but deliberately insufficient scene must return CAUTION, not CLEAR.
    return np.array([[0., -5., 0.], [1., -6., 0.], [-1., -6., 0.], [0., -7., 0.]], np.float32)


@pytest.fixture
def bag_factory(typestore, sample_xyz):
    def create(path, stamps=(1_000_000_000, 1_100_000_000, 1_200_000_000), xyz=None):
        xyz = sample_xyz if xyz is None else xyz
        types = typestore.types
        fields = [types['sensor_msgs/msg/PointField'](name=k, offset=i*4, datatype=7, count=1)
                  for i, k in enumerate(('x', 'y', 'z'))]
        source_header = types['std_msgs/msg/Header'](
            stamp=types['builtin_interfaces/msg/Time'](sec=42, nanosec=17), frame_id='test_lidar')
        message = types['sensor_msgs/msg/PointCloud2'](
            header=source_header, height=1, width=len(xyz), fields=fields,
            is_bigendian=False, point_step=12, row_step=12*len(xyz),
            data=np.frombuffer(np.asarray(xyz, '<f4').tobytes(), dtype=np.uint8), is_dense=True)
        raw = bytes(typestore.serialize_cdr(message, message.__msgtype__))
        with sqlite3.connect(str(path)) as db:
            db.execute('CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT)')
            db.execute('CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)')
            db.execute("INSERT INTO topics VALUES (1, '/recorded_points', 'sensor_msgs/msg/PointCloud2', 'cdr')")
            db.executemany('INSERT INTO messages VALUES (?,1,?,?)',
                           [(i+1, stamp, raw) for i, stamp in enumerate(stamps)])
        return path, raw
    return create
