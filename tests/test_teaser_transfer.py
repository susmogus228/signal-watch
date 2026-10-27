import csv
import io
import zipfile
import hashlib
from datetime import datetime,timedelta

import numpy as np
import pytest

from signalwatch.dataset import SENSORS
from signalwatch.teaser_transfer import parse_long_csv,annotation_masks,transfer_metrics,load_teaser_archive,run_transfer


def long_csv(change=None):
    rows=[]
    start=datetime(2019,7,8,18,38,30)
    for i in range(100):
        for j,s in enumerate(SENSORS):
            rows.append({'datetime':str(start+timedelta(seconds=i)),'id':s['key'],'value':str(i+j/10),'index':str(i*8+j)})
    if change:change(rows)
    output=io.StringIO();w=csv.DictWriter(output,fieldnames=['datetime','id','value','index'],delimiter=';')
    w.writeheader();w.writerows(rows)
    return output.getvalue().encode()


def test_exact_pivot_uses_sensor_keys_and_preserves_values_without_database_index():
    rec=parse_long_csv(long_csv())
    assert rec['values'].shape==(100,8)
    np.testing.assert_allclose(rec['values'][7],np.arange(8)/10+7)
    assert rec['labels'] is None
    assert rec['timestamps'][0]=='2019-07-08T18:38:30'
    assert rec['cadence_seconds']==1.


@pytest.mark.parametrize('change',[
    lambda rows:rows.append(dict(rows[0])),
    lambda rows:rows.pop(),
    lambda rows:rows[0].update(id='unknown'),
    lambda rows:rows[0].update(value='nan'),
    lambda rows:rows[0].update(datetime='bad-date'),
])
def test_invalid_mapping_or_values_never_silently_interpolate(change):
    with pytest.raises(ValueError):parse_long_csv(long_csv(change))


def test_source_annotations_use_half_open_envelopes_and_uncertain_transitions():
    times=['18:39:21','18:39:22','18:42:32','18:44:36','18:45:48','18:45:49','18:46:51']
    masks=annotation_masks({'timestamps':['2019-07-08T'+t for t in times]})
    assert masks['envelope'].tolist()==[False,True,False,True,True,True,False]
    assert masks['core'].tolist()==[False,True,False,False,False,True,False]
    assert masks['uncertain'].tolist()==[False,False,False,True,True,False,False]
    assert len(masks['events'])==4


def test_preexisting_alert_is_not_a_new_hit_and_latencies_use_source_clock():
    rec=parse_long_csv(long_csv())
    score=np.zeros(100);score[50:60]=2.
    metrics=transfer_metrics(rec,score)
    event=metrics['events'][0]
    assert event['status']=='missed'
    assert event['preexisting_alert_at_first_observed_row']
    score[50:52]=0.
    metrics=transfer_metrics(rec,score)
    assert metrics['events'][0]['status']=='detected'
    assert metrics['events'][0]['delay_from_appearance_start_seconds']==0.
    assert metrics['events'][1]['status']=='unobservable'


def test_gap_resets_actual_episodes_without_splitting_four_source_interventions():
    rec=parse_long_csv(long_csv());rec['gap_before'][55]=True
    score=np.zeros(100);score[50:60]=2.
    metrics=transfer_metrics(rec,score)
    assert metrics['events'][0]['status']=='detected'
    assert metrics['events'][0]['delay_from_appearance_start_seconds']==3.
    assert len(metrics['events'])==4


def test_archive_requires_exact_source_identity_before_parsing(tmp_path):
    path=tmp_path/'source.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('SKAB teaser.csv',long_csv());z.writestr('testbed.png',b'test-fixture')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(load_teaser_archive(path,digest)['values'])==100
    with pytest.raises(ValueError,match='hash'):
        load_teaser_archive(path,'0'*64)


def test_completed_transfer_refuses_another_model_inference(tmp_path):
    target=tmp_path/'outputs/teaser-transfer';target.mkdir(parents=True)
    (target/'report.json').write_text('{}')
    with pytest.raises(ValueError,match='already completed'):
        run_transfer(tmp_path)

