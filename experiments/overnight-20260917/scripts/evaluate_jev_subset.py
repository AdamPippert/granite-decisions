"""Fixed 60-request Jev comparison; never used to train or select checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from benchmark_decisions import jev,distribution,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--key-file',required=True);p.add_argument('--out',required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=a.resume)
    rows=[json.loads(s) for s in Path(a.data).read_text().splitlines()]
    selected=[]
    for family in ('banking77','boolq','synthetic_score'):
        candidates=[r for r in rows if r['family']==family]
        selected+=sorted(candidates,key=lambda r:hashlib.sha256(('jev-fixed-v1-'+r['id']).encode()).hexdigest())[:20]
    if len(selected)!=60:raise ValueError('expected 20 records per family')
    key=Path(a.key_file).read_text().strip()
    result=[json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()] if a.resume else []
    planned={r['id']:r for r in selected}
    if len({r['id'] for r in result})!=len(result):raise ValueError('duplicate saved rows')
    for r in result:
        row=planned[r['id']]
        if r['state_question_hash']!=digest([row['state'],row['question']]):raise ValueError('resume identity mismatch')
    done={r['id'] for r in result}
    with (out/'predictions.jsonl').open('a' if a.resume else 'x') as f:
        for row in selected:
            if row['id'] in done:continue
            q=row['question'];start=time.perf_counter()
            response=jev(row['state'],{'decision':q},'jev-latest',key)
            error=None
            try:probs=distribution(q,response['answers']['decision'])
            except (ValueError,KeyError,TypeError) as exc:probs=None;error=str(exc)
            if q['type']=='noul':target=int(row['label'])
            elif q['type']=='score':target=row['label']
            else:target=sorted(q['criteria']).index(row['label'])
            entry={'id':row['id'],'family':row['family'],'target':target,'probabilities':probs,
                'correct':max(range(len(probs)),key=probs.__getitem__)==target if probs else None,
                'rejection':error,
                'elapsed_ms':(time.perf_counter()-start)*1000,'state_question_hash':digest([row['state'],q]),
                'response':response}
            result.append(entry);f.write(json.dumps(entry)+'\n');f.flush()
            print(row['id'],flush=True)
    summary={'selection':'20 per family ordered by sha256(jev-fixed-v1-ID), fixed before predictions',
        'purpose':'held-out evaluator only; never training labels or checkpoint selection',
        'test_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),'n':len(result),
        'families':{}}
    for family in ('banking77','boolq','synthetic_score'):
        valid=[r for r in result if r['family']==family and r['correct'] is not None]
        summary['families'][family]={'requested':20,'valid_n':len(valid),'rejected':20-len(valid),
            'accuracy_on_valid_responses':sum(r['correct'] for r in valid)/len(valid) if valid else None}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
