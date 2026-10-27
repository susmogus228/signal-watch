"""Fetch only pinned public SKAB CSV files and preserve hashes and notices."""
from __future__ import annotations
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signalwatch.dataset import SNAPSHOT, parse_csv_bytes


def main():
    base = f'https://raw.githubusercontent.com/waico/SKAB/{SNAPSHOT}/'
    paths = ['data/anomaly-free/anomaly-free.csv']
    paths += [f'data/{group}/{number}.csv' for group, numbers in [('other',range(1,15)),('valve1',range(16)),('valve2',range(4))] for number in numbers]
    def download(path):
        destination = ROOT / 'data/raw' / Path(path).relative_to('data')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(urlopen(base + path, timeout=45).read())
        raw = destination.read_bytes()
        group = Path(path).parent.name
        recording_id = 'anomaly-free' if group == 'anomaly-free' else f'{group}-{Path(path).stem}'
        record = parse_csv_bytes(raw, recording_id, f'{group.title()} / recording {Path(path).stem}')
        number = int(Path(path).stem) if group != 'anomaly-free' else -1
        split = 'normal_source' if number < 0 else ('validation' if number % 3 == 0 else 'test')
        dt = record['elapsed'][1:] - record['elapsed'][:-1]
        return {'id': recording_id,'name':record['name'],'group':group,'split':split,
                'path':destination.relative_to(ROOT).as_posix(),'url':base+path,
                'sha256':record['sha256'],'bytes':len(raw),'rows':len(record['values']),
                'duration_seconds':float(record['elapsed'][-1]),'cadence_seconds':record['cadence_seconds'],
                'labelled':record['labels'] is not None,'anomaly_samples':int(record['labels'].sum()) if record['labels'] is not None else 0,
                'gap_count':int(record['gap_before'].sum()),'min_interval_seconds':float(dt.min()),'max_interval_seconds':float(dt.max()),
                'missing_sensor_values':0}
    entries = list(ThreadPoolExecutor(max_workers=6).map(download, paths))
    notices=[]
    for upstream, local in [('LICENSE','data/SKAB-LICENSE'),('data/README.md','data/SKAB-DATA-README.md'),('README.md','data/SKAB-README.md')]:
        raw=urlopen(base+upstream,timeout=45).read()
        (ROOT/local).write_bytes(raw)
        notices.append({'url':base+upstream,'path':local,'sha256':hashlib.sha256(raw).hexdigest()})
    result={'dataset':'SKAB','snapshot':SNAPSHOT,'license':'GPL-3.0 upstream notice retained; no project source license selected',
            'split_rule':'Disturbed recording numeric ID divisible by 3: validation; other IDs: final test. Defined before scoring.',
            'recordings':entries,'notices':notices,'total_csv_bytes':sum(e['bytes'] for e in entries)}
    (ROOT/'data/manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'recordings':len(entries),'bytes':result['total_csv_bytes'],'rows':sum(e['rows'] for e in entries),'gaps':sum(e['gap_count'] for e in entries)}))


if __name__ == '__main__':
    main()
