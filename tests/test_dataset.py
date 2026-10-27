import numpy as np
import pytest

from signalwatch.dataset import SENSORS, parse_csv_bytes


def csv_bytes(count=100, duplicate=False, labels=True):
    header = ['datetime'] + [sensor['key'] for sensor in SENSORS]
    if labels:
        header += ['anomaly']
    rows = [';'.join(header)]
    for i in range(count):
        stamp = i if not duplicate or i != 50 else 49
        row = [f'2020-01-01T00:{stamp // 60:02d}:{stamp % 60:02d}'] + ['1.0'] * 8
        if labels:
            row += ['0']
        rows.append(';'.join(row))
    return '\n'.join(rows).encode()


def test_optional_labels_and_strict_timestamps():
    record = parse_csv_bytes(csv_bytes(labels=False), 'upload-a', 'Local CSV')
    assert record['labels'] is None
    assert record['values'].shape == (100, 8)
    assert np.all(np.diff(record['elapsed']) == 1)
    with pytest.raises(ValueError, match='increasing'):
        parse_csv_bytes(csv_bytes(duplicate=True), 'x', 'x')


def test_reject_small_invalid_and_nonfinite():
    with pytest.raises(ValueError, match='100'):
        parse_csv_bytes(csv_bytes(count=99), 'x', 'x')
    with pytest.raises(ValueError, match='finite'):
        parse_csv_bytes(csv_bytes().replace(b'1.0', b'nan', 1), 'x', 'x')
    with pytest.raises(ValueError):
        parse_csv_bytes(b'secret\nprivate', 'x', 'x')


def test_mapping_binary_labels_and_size_limit():
    raw=csv_bytes().replace(b'Current',b'amps')
    record=parse_csv_bytes(raw,'x','Mapped',{'Current':'amps'})
    assert record['values'].shape==(100,8)
    with pytest.raises(ValueError,match='binary'):
        parse_csv_bytes(csv_bytes().replace(b';0\n',b';2\n',1),'x','x')
    with pytest.raises(ValueError,match='5 MiB'):
        parse_csv_bytes(b'x'*(5*1024*1024+1),'x','x')


def test_gap_detection_and_cadence_uses_initial_warmup_only():
    from datetime import datetime,timedelta
    lines=csv_bytes(count=200).decode().splitlines()
    for i in range(200):
        seconds=i if i<100 else 100+(i-100)*10
        parts=lines[i+1].split(';')
        parts[0]=(datetime(2020,1,1)+timedelta(seconds=seconds)).isoformat()
        lines[i+1]=';'.join(parts)
    record=parse_csv_bytes('\n'.join(lines).encode(),'x','x')
    prefix=parse_csv_bytes('\n'.join(lines[:101]).encode(),'x','x')
    assert record['cadence_seconds']==prefix['cadence_seconds']==1
    assert record['gap_before'][101:].all()
    np.testing.assert_array_equal(record['gap_before'][:100],prefix['gap_before'])
