"""Fast exploration of already evaluated, hash-verified OOF operating points."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from signalwatch.ml_v2 import select_candidate


def compare_saved_hybrid(root: Path,bundle: dict,budget: float) -> dict:
    if not .25<=budget<=10.:
        raise ValueError('Budget must be between 0.25 and 10 false onsets per normal hour.')
    report=bundle['supervised_bundle']['study_report']
    path=Path(root)/'artifacts/hybrid-development/candidate_audits.json'
    raw=path.read_bytes();expected=bundle.get('metadata',{}).get('development_audits_sha256')
    if not expected or hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('Saved development candidate evidence hash mismatch.')
    points=json.loads(raw)
    for point in points:
        metrics=point['metrics'];fraction=metrics.get('normal_alert_fraction');rate=metrics.get('false_episodes_per_hour')
        point['feasible']=fraction is not None and rate is not None and fraction<=.05 and rate<=budget
    selected=select_candidate(points)
    if selected is None:
        return {'scope':report['scope'],'budget':budget,'feasible':False,
                'note':'No feasible saved development operating point; active artifacts are unchanged.'}
    methods=copy.deepcopy(report['methods']);methods['learned']=selected['metrics']
    return {'scope':report['scope'],'budget':budget,'feasible':True,'selected':selected,
            'thresholds':dict(bundle['thresholds']),'methods':methods,'calibration':{'learned':selected},
            'evidence_sha256':expected,
            'note':'Selects from previously evaluated grouped OOF development operating points. No inference, model fitting or retrospective audit is repeated.'}
