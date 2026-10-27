"""Run the approved bounded nested-calibration development experiment."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from signalwatch.ml_calibrated import run_study

if __name__=='__main__':
    report=run_study(ROOT)
    print(json.dumps({'status':report['status'],'promote':report['promote'],
                      'selected':{key:value for key,value in report['selected'].items() if key!='methods'} if report['selected'] else None,
                      'model_sha256':report.get('model_sha256'),'study_seconds':report['study_seconds']}))
