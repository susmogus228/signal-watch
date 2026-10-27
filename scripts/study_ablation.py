"""Run one frozen development-only controlled ablation; never deploys."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from signalwatch.ml_ablation import run_study

if __name__=='__main__':
    report=run_study(ROOT)
    print(json.dumps({'status':report['status'],'fits':report['actual_hgb_fits'],
                      'best_all':None if report['best_all'] is None else {k:v for k,v in report['best_all'].items() if k not in ['per_recording','group_slices']},
                      'best_new':None if report['best_new'] is None else {k:v for k,v in report['best_new'].items() if k not in ['per_recording','group_slices']},
                      'study_seconds':report['study_seconds']}))
