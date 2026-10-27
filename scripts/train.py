"""Reproduce the fixed study; never include local uploads in training."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from signalwatch.ml import train_bundle

if __name__=='__main__':
    result=train_bundle(ROOT)
    print(json.dumps({key:result[key] for key in ['algorithm','training_rows','parameters','model_sha256','training_seconds']}))
