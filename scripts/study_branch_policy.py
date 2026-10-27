"""Execute the frozen one-candidate branch policy study after pre-fit review."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signalwatch.ml_branch_policy import run_study

if __name__ == '__main__':
    report = run_study(ROOT)
    candidate = report['candidate']
    print(json.dumps({'status': report['status'], 'metrics': candidate['metrics'],
                      'lost_event_ids': candidate['lost_event_ids'], 'gained_event_ids': candidate['gained_event_ids'],
                      'model_fits': report['model_fits'], 'study_seconds': report['study_seconds']}))
