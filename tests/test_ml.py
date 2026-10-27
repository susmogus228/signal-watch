import numpy as np

from signalwatch.ml import score_recording


class Predictor:
    def predict(self, features):
        return features[:, :8]


def fixture_record(n=150):
    rng = np.random.default_rng(7)
    return {'values': rng.normal(size=(n, 8)), 'gap_before': np.zeros(n, bool)}


def bundle():
    return {'model': Predictor(), 'window': 10, 'center': np.zeros(8),
            'scale': np.ones(8), 'residual_scale': np.ones(8),
            'thresholds': {'learned': 3., 'robust': 3., 'persistence': 3.}}


def test_future_prefix_invariance_and_gap_reset():
    record = fixture_record()
    full = score_recording(bundle(), record)
    short = {key: value[:80] for key, value in record.items()}
    prefix = score_recording(bundle(), short)
    for method in full['scores']:
        np.testing.assert_allclose(full['scores'][method][:80], prefix['scores'][method], equal_nan=True)
    record['gap_before'][70] = True
    after_gap = score_recording(bundle(), record)
    assert np.isnan(after_gap['scores']['learned'][70:80]).all()
    isolated = {key: value[70:] for key, value in record.items()}
    isolated['gap_before'][0] = False
    fresh = score_recording(bundle(), isolated)
    np.testing.assert_allclose(after_gap['scores']['learned'][70:], fresh['scores']['learned'], equal_nan=True)


def test_empty_and_warmup_only_prefixes_are_valid():
    for count in [0,1,10]:
        output=score_recording(bundle(),fixture_record(count))
        assert output['contributions'].shape==(count,8)
        for values in output['scores'].values():
            assert len(values)==count
            assert np.isnan(values).all()


def test_real_artifact_prefix_invariance_and_labels_are_not_features():
    from pathlib import Path
    from signalwatch.ml import load_bundle
    from signalwatch.dataset import load_builtin
    root=Path(__file__).resolve().parents[1]
    if not (root/'artifacts/model.joblib').exists():
        import pytest
        pytest.skip('Run scripts/train.py for the real-artifact integration check.')
    model=load_bundle(root)
    record=load_builtin(root,'valve1-1')
    full=score_recording(model,record)
    short=dict(record)
    for key in ['values','elapsed','gap_before','labels']:
        short[key]=record[key][:200].copy()
    short['labels']=~short['labels']
    prefix=score_recording(model,short)
    for method in full['scores']:
        np.testing.assert_allclose(full['scores'][method][:200],prefix['scores'][method],equal_nan=True)
