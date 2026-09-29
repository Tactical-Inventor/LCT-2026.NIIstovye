"""Optional regression on a real recording; no recording is bundled with the repo."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import numpy as np
import scipy
import numba
from obstacle_detector import StreamProcessor
from obstacle_detector.io import FrameSource
from obstacle_detector_ros.conversion import SequenceGuard, cloud_arrays, result_dict, json_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    direct, transported = StreamProcessor(), StreamProcessor()
    guard = SequenceGuard()
    rows = []
    with FrameSource(args.input) as source:
        for index in range(min(args.frames, len(source))):
            frame = source.read(index)
            stamp, raw = source.read_serialized(index)
            message = source.typestore.deserialize_cdr(raw, 'sensor_msgs/msg/PointCloud2')
            message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
            xyz, ring, removed = cloud_arrays(message)
            assert removed == 0, 'Use a finite-cloud recording for exact offline parity'
            np.testing.assert_array_equal(xyz, frame.xyz)
            np.testing.assert_array_equal(ring, frame.ring)
            assert guard.update(message.header) is None, 'Recording has a sequence discontinuity'
            a = direct.process(frame.xyz.astype(np.float64), frame.ring, stamp)
            b = transported.process(xyz, ring, stamp)
            expected = result_dict(a, message.header, index)
            actual = result_dict(b, message.header, index)
            expected.pop('timing_ms')
            actual.pop('timing_ms')
            assert json_text(actual) == json_text(expected), f'Result differs at frame {index}'
            rows.append(actual)
            if index == 0 or (index + 1) % 10 == 0:
                print(f'{index+1} frames: parity OK, {a.status}', flush=True)
    report = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  numba=numba.__version__, frames=len(rows), exact_transport_parity=True,
                  status_counts=dict(Counter(r['status'] for r in rows)), results=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
