import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from obstacle_detector import StreamProcessor
from obstacle_detector.io import FrameSource, decode_pointcloud_message
from obstacle_detector_ros.conversion import (
    SequenceGuard, cloud_arrays, json_text, packed_cloud, result_dict)
from conftest import header


@pytest.mark.parametrize('endian', ['<', '>'])
def test_organized_cloud_padding_endianness_and_ring(endian):
    records = np.zeros((2, 2), dtype=[('x', endian+'f4'), ('y', endian+'f4'),
                                     ('z', endian+'f4'), ('ring', endian+'u2')])
    records['x'] = [[1, 2], [3, 4]]
    records['y'] = -records['x']
    records['ring'] = [[5, 6], [7, 8]]
    data = b''.join(row.tobytes() + b'padding!' for row in records)
    fields = [NS(name=k, offset=i*4, datatype=7, count=1) for i, k in enumerate(('x','y','z'))]
    fields.append(NS(name='ring', offset=12, datatype=4, count=1))
    message = NS(header=header(), width=2, height=2, point_step=14, row_step=36,
                 fields=fields, data=data, is_bigendian=endian=='>')
    xyz, ring, removed = cloud_arrays(message)
    np.testing.assert_array_equal(xyz[:, 0], [1,2,3,4])
    np.testing.assert_array_equal(xyz[:, 1], [-1,-2,-3,-4])
    np.testing.assert_array_equal(ring, [5,6,7,8])
    assert removed == 0
    message.data = data[:-1]
    with pytest.raises(ValueError, match='buffer size'):
        cloud_arrays(message)


def test_nan_removal_preserves_ring_alignment():
    fields = [NS(name=k, offset=i*4, datatype=7, count=1) for i,k in enumerate(('x','y','z'))]
    fields.append(NS(name='ring', offset=12, datatype=4, count=1))
    data = np.zeros(3, dtype=[('xyz','<f4',(3,)), ('ring','<u2')])
    data['xyz'] = [[1,2,3], [np.nan,2,3], [4,5,6]]
    data['ring'] = [1,2,3]
    message = NS(header=header(), width=3, height=1, point_step=14, row_step=42,
                 fields=fields, data=data.tobytes(), is_bigendian=False)
    xyz, ring, removed = cloud_arrays(message)
    np.testing.assert_array_equal(xyz, [[1,2,3], [4,5,6]])
    np.testing.assert_array_equal(ring, [1,3])
    assert removed == 1


@pytest.mark.parametrize('ring', [None, np.array([0,65535], np.uint16)])
def test_packed_cloud_round_trip(ring):
    xyz = np.array([[1,-2,3], [4,-5,6]], dtype=float)
    data, step = packed_cloud(xyz, ring)
    fields = [NS(name=k, offset=i*4, datatype=7, count=1) for i,k in enumerate(('x','y','z'))]
    if ring is not None:
        fields.append(NS(name='ring', offset=12, datatype=4, count=1))
    message = NS(header=header(), width=2, height=1, point_step=step, row_step=2*step,
                 fields=fields, data=data, is_bigendian=False)
    actual = decode_pointcloud_message(message)
    np.testing.assert_array_equal(actual.xyz, xyz)
    np.testing.assert_array_equal(actual.ring, ring)


@pytest.mark.parametrize('ring', [np.array([-1]), np.array([65536]), np.array([1.5]), np.array([1,2])])
def test_reject_invalid_output_ring(ring):
    with pytest.raises(ValueError, match='ring'):
        packed_cloud(np.zeros((1,3)), ring)


@pytest.mark.parametrize('xyz', [np.zeros(3), np.array([[np.nan,0,0]]), np.array([[1e100,0,0]])])
def test_reject_invalid_output_xyz(xyz):
    with pytest.raises(ValueError):
        packed_cloud(xyz)


def test_float64_input_is_not_rounded_to_float32():
    xyz = np.array([[1.00000000001, -10., 2.]], dtype=np.float64)
    raw, step = packed_cloud(xyz, np.array([42]), float64=True)
    assert step == 26
    decoded = np.frombuffer(raw, dtype=[('xyz','<f8',(3,)), ('ring','<u2')])
    np.testing.assert_array_equal(decoded['xyz'], xyz)
    assert decoded['ring'][0] == 42


def test_input_nan_can_reach_processor_for_filtering():
    xyz = np.array([[1., np.nan, np.inf]])
    raw, step = packed_cloud(xyz, allow_nonfinite=True)
    assert step == 12
    np.testing.assert_array_equal(np.frombuffer(raw, '<f4').reshape(-1,3), xyz)


def test_sequence_resets_for_frame_change_clock_rewind_and_missing_frames():
    guard = SequenceGuard()
    assert guard.update(header()) is None
    assert guard.update(header(1_100_000_000)) is None
    assert guard.update(header(1_400_000_000)) == 'timestamp_gap'
    assert guard.update(header(1_000_000_000)) == 'timestamp_moved_backwards'
    assert guard.update(header(1_100_000_000, 'other')) == 'frame_id_changed'
    assert SequenceGuard(0).max_gap_ns == 0


@pytest.mark.parametrize('gap', [-1, float('nan'), float('inf')])
def test_invalid_gap(gap):
    with pytest.raises(ValueError):
        SequenceGuard(gap)


def test_strict_json_numpy_and_unknown_numbers():
    text = json_text({'a':np.int64(7), 'b':np.bool_(True), 'c':np.array([np.nan, np.inf, 3.])})
    assert json.loads(text) == {'a':7, 'b':True, 'c':[None,None,3.]}
    assert 'NaN' not in text and 'Infinity' not in text


def test_db3_serialized_round_trip_and_split_timestamp_order(tmp_path, bag_factory, typestore, sample_xyz):
    _, raw = bag_factory(tmp_path/'part_b.db3', (1_200_000_000,))
    bag_factory(tmp_path/'part_a.db3', (1_100_000_000, 1_000_000_000))
    with FrameSource(tmp_path) as source:
        assert len(source) == 3 and source.topic == '/recorded_points'
        assert [source.read_serialized(i)[0] for i in range(3)] == [1_000_000_000,1_100_000_000,1_200_000_000]
        assert source.read_serialized(0)[1] == raw
        message = typestore.deserialize_cdr(raw, 'sensor_msgs/msg/PointCloud2')
        xyz, _, _ = cloud_arrays(message)
        np.testing.assert_array_equal(xyz, source.read(0).xyz)
        np.testing.assert_array_equal(xyz, sample_xyz)
        with pytest.raises(IndexError):
            source.read_serialized(3)


def test_npy_adapter_remains_usable(tmp_path, sample_xyz):
    path = tmp_path/'frame.npy'
    np.save(path, sample_xyz)
    with FrameSource(path) as source:
        np.testing.assert_array_equal(source.read().xyz, sample_xyz)
        with pytest.raises(ValueError, match='only for DB3'):
            source.read_serialized()


def test_real_detector_result_is_serializable_and_never_clear_without_route(sample_xyz):
    result = StreamProcessor().process(sample_xyz.astype(float))
    payload = json.loads(json_text(result_dict(result, header(), 0)))
    assert payload['processing_ok'] is True
    assert payload['status'] == 'CAUTION'
    assert payload['path_clear'] is None
    assert payload['route_available'] is False
    assert payload['header']['timestamp_ns'] == 1_000_000_000
