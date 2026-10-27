"""Run the four-candidate weighted supervised development study."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from signalwatch.ml_supervised_weighted import run_study

if __name__=='__main__':
    report=run_study(ROOT)
    print(json.dumps({'status':report['status'],'selected':report['selected'],'study_seconds':report['study_seconds']}))
