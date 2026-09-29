"""Optional file adapters. ROS installation is not required to read ROS2 bags."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import numpy as np


@dataclass(frozen=True)
class Frame:
    xyz: np.ndarray
    ring: np.ndarray | None = None
    timestamp_ns: int | None = None
    frame_id: str | None = None


def decode_pointcloud(raw, typestore):
    msg = typestore.deserialize_cdr(raw, "sensor_msgs/msg/PointCloud2")
    return decode_pointcloud_message(msg)


def decode_pointcloud_message(msg):
    """Decode a native ROS or rosbags message, including organized row padding."""
    codes = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    endian = ">" if msg.is_bigendian else "<"
    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.datatype not in codes or f.count < 1:
            raise ValueError("Unsupported PointField layout")
        dtype = np.dtype(endian + codes[f.datatype])
        names.append(f.name)
        formats.append((dtype, (f.count,)) if f.count > 1 else dtype)
        offsets.append(f.offset)
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PointCloud2 lacks XYZ fields")
    dtype = np.dtype(dict(names=names, formats=formats, offsets=offsets, itemsize=msg.point_step))
    if msg.row_step < msg.width * msg.point_step or len(msg.data) < msg.row_step * msg.height:
        raise ValueError("Invalid PointCloud2 buffer size")
    if not msg.width or not msg.height:
        return Frame(np.empty((0, 3), np.float32), frame_id=msg.header.frame_id)
    records = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                         strides=(msg.row_step, msg.point_step))
    xyz = np.column_stack([records[k].reshape(-1) for k in ("x", "y", "z")])
    if xyz.shape != (msg.width * msg.height, 3):
        raise ValueError("XYZ fields must be scalar")
    xyz = xyz.astype(np.float64 if xyz.dtype.itemsize == 8 else np.float32, copy=False)
    ring = records["ring"].reshape(-1).copy() if "ring" in names else None
    return Frame(xyz, ring, frame_id=msg.header.frame_id)


class FrameSource:
    """Random access to one cloud or a time-sorted, possibly split SQLite bag.

    SQLite inputs are opened read-only. Only message indices reside in memory.
    A topic is mandatory when several PointCloud2 topics exist.
    """
    def __init__(self, path, *, topic=None):
        self.path = Path(path).resolve()
        self.connections = []
        self.rows = []
        self.typestore = None
        self.single = None
        self.topic = None
        try:
            if self.path.suffix.lower() == ".db3" or self.path.is_dir():
                files = sorted(self.path.glob("*.db3")) if self.path.is_dir() else [self.path]
                if not files:
                    raise ValueError("No .db3 files found")
                from rosbags.typesys import Stores, get_typestore
                self.typestore = get_typestore(Stores.ROS2_HUMBLE)
                available = set()
                for file in files:
                    db = sqlite3.connect(file.as_uri() + "?mode=ro", uri=True)
                    self.connections.append(db)
                    available.update(r[0] for r in db.execute(
                        "SELECT name FROM topics WHERE type='sensor_msgs/msg/PointCloud2' AND serialization_format='cdr'"))
                if topic is None:
                    if len(available) != 1:
                        raise ValueError(f"Select --topic from {sorted(available)}")
                    topic = next(iter(available))
                if topic not in available:
                    raise ValueError(f"PointCloud2 topic not found: {topic}")
                self.topic = topic
                for part, db in enumerate(self.connections):
                    self.rows.extend((stamp, part, ident) for stamp, ident in db.execute(
                        "SELECT m.timestamp,m.id FROM messages m JOIN topics t ON m.topic_id=t.id "
                        "WHERE t.name=? AND t.type='sensor_msgs/msg/PointCloud2' AND t.serialization_format='cdr'", (topic,)))
                self.rows.sort()
                if not self.rows:
                    raise ValueError("No PointCloud2 messages found")
            else:
                suffix = self.path.suffix.lower()
                if suffix == ".npy":
                    self.single = Frame(np.load(self.path, allow_pickle=False))
                elif suffix == ".npz":
                    with np.load(self.path, allow_pickle=False) as data:
                        if "xyz" not in data:
                            raise ValueError("NPZ requires key xyz, optional ring")
                        self.single = Frame(data["xyz"], data["ring"] if "ring" in data else None)
                else:
                    raise ValueError("Supported inputs: NPY, NPZ, DB3 or a bag directory")
        except Exception:
            self.close()
            raise

    def __len__(self):
        return 1 if self.single is not None else len(self.rows)

    def read(self, index=0):
        if not 0 <= index < len(self):
            raise IndexError(f"frame {index} outside 0..{len(self)-1}")
        if self.single is not None:
            return self.single
        stamp, raw = self.read_serialized(index)
        frame = decode_pointcloud(raw, self.typestore)
        return Frame(frame.xyz, frame.ring, stamp, frame.frame_id)

    def read_serialized(self, index=0):
        """Return recording timestamp and original CDR bytes for ROS publication."""
        if self.single is not None:
            raise ValueError("Serialized messages are available only for DB3 inputs")
        if not 0 <= index < len(self.rows):
            raise IndexError(f"frame {index} outside 0..{len(self.rows)-1}")
        stamp, part, ident = self.rows[index]
        raw = self.connections[part].execute(
            "SELECT data FROM messages WHERE id=?", (ident,)).fetchone()[0]
        return stamp, raw

    def close(self):
        for db in self.connections:
            db.close()
        self.connections.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
