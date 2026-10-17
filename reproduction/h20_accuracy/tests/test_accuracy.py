import copy
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import evaluate_accuracy as ev


def test_author_and_official_hotpot_metrics_are_distinct():
    assert ev.scores('hotpotqa','yes indeed','yes') == {'em':0.,'f1':2/3,'official_answer_f1':0.}
    assert ev.scores('hotpotqa','The Nile','Nile')['f1']==1.
    assert ev.scores('multinews','','a document')['rougeL']==0.
    assert ev.scores('conversation','cats running','cat run')['rougeL']==1.


def fixture(monkeypatch,tmp_path):
    rows=[{'index':i,'example_sha256':f'hash{i}','gold':'yes','receiver_tokens':100,
           'sender_tokens':90,'shared_blocks_64':1} for i in range(2)]
    meta={'version':'fixture'}
    monkeypatch.setattr(ev,'prepare',lambda task:(rows,meta))
    protocol={'task':'hotpotqa','limit':0,'data':meta,'mode':'mapflow','expected_reuse_layers':1}
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    preds=[]
    for row in rows:
        rid=f'request-{row["index"]}'
        preds.append({**row,'request_id':rid,'prediction':'yes','scores':ev.scores('hotpotqa','yes','yes'),
            'finish_reason':'stop','source_response':{'usage':{'prompt_tokens':90}},
            'response':{'usage':{'prompt_tokens':100},'choices':[{'message':{'content':'yes'},'finish_reason':'stop'}]},
            'reuse':{'reused_layers':1,'events':[{'layer':3,'reused':True,'kernel_completed':True,
                'nnz':5,'requests':[[rid,100]]}]}})
    return preds


def save(path,preds):
    (path/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in preds))


def test_complete_audit(monkeypatch,tmp_path):
    preds=fixture(monkeypatch,tmp_path);save(tmp_path,preds)
    assert ev.audit(tmp_path)['n']==2


@pytest.mark.parametrize('corruption',['duplicate','wrong_score','wrong_prompt','wrong_text','fake_reuse','wrong_request'])
def test_audit_rejects_corrupted_evidence(monkeypatch,tmp_path,corruption):
    preds=fixture(monkeypatch,tmp_path)
    if corruption=='duplicate':preds[1]=copy.deepcopy(preds[0])
    elif corruption=='wrong_score':preds[0]['scores']['f1']=.5
    elif corruption=='wrong_prompt':preds[0]['response']['usage']['prompt_tokens']=101
    elif corruption=='wrong_text':preds[0]['response']['choices'][0]['message']['content']='no'
    elif corruption=='fake_reuse':preds[0]['reuse']['events'][0]['reused']=False
    elif corruption=='wrong_request':preds[0]['reuse']['events'][0]['requests']=[['someone-else',100]]
    save(tmp_path,preds)
    with pytest.raises(AssertionError):ev.audit(tmp_path)
