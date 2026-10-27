"""Report one frozen final trend model without changing active artifacts."""
import json
import sys
import argparse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from signalwatch.trend_diagnostics import run_diagnostics,STUDIES

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',choices=list(STUDIES),default='artifacts/trend-development')
    args=parser.parse_args()
    result=run_diagnostics(ROOT,args.study)
    print(json.dumps({'retrospective':result['retrospective']['metrics'],
                      'teaser':{key:result['reused_teaser']['candidate'][key] for key in
                                ['detected_source_interventions','unannotated_background_alert_starts','assumed_background_alert_seconds']}}))
