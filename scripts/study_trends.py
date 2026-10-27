"""Run the frozen learned-trend study after independent pre-fit review."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signalwatch.ml_trends import run_study

if __name__ == '__main__':
    report = run_study(ROOT)
    selected = report['selected']
    print(json.dumps({'status': report['status'], 'actual_hgb_fits': report['actual_hgb_fits'],
                      'selected': None if selected is None else {key: selected[key] for key in ['arm', 'threshold', 'metrics', 'lost_event_ids', 'gained_event_ids', 'progressive_subset']},
                      'model_sha256': report.get('model_sha256'), 'study_seconds': report['study_seconds']}))
