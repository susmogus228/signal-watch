"""Run the fixed v2 development study without opening legacy test runs."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from signalwatch.ml_v2 import run_study

if __name__=='__main__':
    result=run_study(ROOT)
    print(json.dumps({'status':result['status'],'selected':result['selected'],'study_seconds':result['study_seconds']}))
