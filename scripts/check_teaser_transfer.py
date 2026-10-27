"""Evaluate the one frozen candidate on the separately collected SKAB teaser."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from signalwatch.teaser_transfer import run_transfer

if __name__=='__main__':
    report=run_transfer(ROOT)
    print(json.dumps({'candidate_enabled':report['candidate_enabled'],
                      'methods':{name:{key:metric[key] for key in ['detected_source_interventions',
                          'unannotated_background_alert_starts','assumed_background_alert_seconds']}
                                 for name,metric in report['methods'].items()}}))
