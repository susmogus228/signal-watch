"""Pinned SKAB recordings and strict, local-only CSV ingestion."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np

SNAPSHOT = 'b2c0d46c2971dcbfe71e26087b6d231998bb91c2'
SENSORS = [
    {'key': 'Accelerometer1RMS', 'label': 'Acceleration 1 RMS', 'unit': 'g'},
    {'key': 'Accelerometer2RMS', 'label': 'Acceleration 2 RMS', 'unit': 'g'},
    {'key': 'Current', 'label': 'Motor current', 'unit': 'A'},
    {'key': 'Pressure', 'label': 'Pressure', 'unit': 'bar'},
    {'key': 'Temperature', 'label': 'Motor temperature', 'unit': 'deg C'},
    {'key': 'Thermocouple', 'label': 'Liquid temperature', 'unit': 'deg C'},
    {'key': 'Voltage', 'label': 'Voltage', 'unit': 'V'},
    {'key': 'Volume Flow RateRMS', 'label': 'Flow rate RMS', 'unit': 'L/min'},
]


def parse_csv_bytes(raw: bytes, recording_id: str, name: str, mapping: dict | None = None) -> dict:
    if len(raw) > 5 * 1024 * 1024:
        raise ValueError('CSV exceeds the 5 MiB limit.')
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('CSV must use UTF-8 encoding.') from exc
    first = text.splitlines()[0] if text.splitlines() else ''
    delimiter = ';' if first.count(';') > first.count(',') else ','
    reader = csv.DictReader(io.StringIO(text, newline=None), delimiter=delimiter)
    headers = reader.fieldnames or []
    if len(headers) != len(set(headers)):
        raise ValueError('CSV headers must be unique.')
    mapping = mapping or {}
    columns = {sensor['key']: mapping.get(sensor['key'], sensor['key']) for sensor in SENSORS}
    columns['timestamp'] = mapping.get('timestamp', 'datetime' if 'datetime' in headers else 'timestamp')
    for key in ['anomaly', 'changepoint']:
        candidate = mapping.get(key, key)
        if candidate in headers:
            columns[key] = candidate
        elif key in mapping:
            raise ValueError(f'Mapped {key} column is missing.')
    if any(value not in headers for value in columns.values()):
        raise ValueError('CSV requires a timestamp and all eight mapped sensor columns.')
    if len(set(columns.values())) != len(columns):
        raise ValueError('Each mapped field must use a different column.')
    values, dates, labels, changes = [], [], [], []
    for row in reader:
        if not row or all(value is None or str(value).strip() == '' for value in row.values()):
            continue
        if len(values) >= 100_000:
            raise ValueError('CSV exceeds the 100000 row limit.')
        if None in row or any(row.get(value) is None for value in columns.values()):
            raise ValueError('CSV rows have inconsistent column counts.')
        try:
            dates.append(datetime.fromisoformat(row[columns['timestamp']].strip().replace('Z', '+00:00')))
            values.append([float(row[columns[sensor['key']]]) for sensor in SENSORS])
            for key, target in [('anomaly', labels), ('changepoint', changes)]:
                if key in columns:
                    label = float(row[columns[key]])
                    if label not in [0., 1.]:
                        raise ValueError('Labels must be binary 0 or 1.')
                    target.append(bool(label))
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('CSV requires valid ISO timestamps, numeric sensors and binary labels.') from exc
    if len(values) < 100:
        raise ValueError('CSV requires at least 100 rows.')
    matrix = np.asarray(values, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError('All sensor values must be finite numbers.')
    try:
        elapsed = np.array([(stamp - dates[0]).total_seconds() for stamp in dates])
    except TypeError as exc:
        raise ValueError('Timestamps must use a consistent timezone.') from exc
    intervals = np.diff(elapsed)
    if not np.isfinite(elapsed).all() or np.any(intervals <= 0):
        raise ValueError('Timestamps must be strictly increasing.')
    # Cadence is established during warm-up, not from future upload rows.
    cadence = float(np.median(intervals[:30]))
    gaps = np.r_[False, intervals > cadence * 3]
    return {'id': recording_id, 'name': name, 'group': 'local', 'split': 'upload',
            'values': matrix, 'timestamps': [stamp.isoformat() for stamp in dates],
            'elapsed': elapsed, 'labels': np.array(labels, bool) if labels else None,
            'changepoints': np.array(changes, bool) if changes else None,
            'gap_before': gaps, 'cadence_seconds': cadence,
            'sha256': hashlib.sha256(raw).hexdigest()}


def manifest(root: Path) -> dict:
    return json.loads((Path(root) / 'data/manifest.json').read_text(encoding='utf-8'))


def recording_summary(record: dict) -> dict:
    return {key: record[key] for key in ['id', 'name', 'group', 'split', 'cadence_seconds', 'sha256']} | {
        'rows': len(record['values']), 'duration_seconds': float(record['elapsed'][-1]) if len(record['elapsed']) else 0.,
        'labelled': record['labels'] is not None}


def load_builtin(root: Path, recording_id: str) -> dict:
    source = manifest(root)
    entries = {entry['id']: entry for entry in source['recordings']}
    if recording_id == 'normal_audit':
        entry = entries['anomaly-free']
        raw = (Path(root) / entry['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Pinned built-in recording hash mismatch.')
        record = parse_csv_bytes(raw, recording_id, 'Held-out normal audit')
        split = json.loads((Path(root) / 'artifacts/split_manifest.json').read_text(encoding='utf-8'))
        start, end = split['normal_blocks']['audit']
        for key in ['values', 'timestamps', 'elapsed', 'labels', 'changepoints', 'gap_before']:
            if record[key] is not None:
                record[key] = record[key][start:end]
        record['elapsed'] = record['elapsed'] - record['elapsed'][0]
        record['gap_before'][0] = False
        record['labels'] = np.zeros(len(record['values']), bool)
        record.update(group='anomaly-free', split='test')
        return record
    if recording_id not in entries or recording_id == 'anomaly-free':
        raise KeyError('Unknown public recording ID.')
    entry = entries[recording_id]
    raw = (Path(root) / entry['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('Pinned built-in recording hash mismatch.')
    record = parse_csv_bytes(raw, recording_id, entry['name'])
    record.update(group=entry['group'], split=entry['split'])
    return record


def catalog(root: Path) -> list[dict]:
    entries = manifest(root)['recordings']
    output = [{key: entry[key] for key in ['id','name','group','split','rows','duration_seconds','labelled','cadence_seconds','sha256']}
              for entry in entries if entry['id'] != 'anomaly-free']
    if (Path(root) / 'artifacts/split_manifest.json').exists():
        output.append(recording_summary(load_builtin(root, 'normal_audit')))
    return output
