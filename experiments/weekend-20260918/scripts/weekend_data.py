"""CPU-only data contract and sampling for the frozen weekend v3 corpus."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

from granite_decisions.contracts import label_index, labels

SPLITS=('train','validation','calibration','test','challenge')


def check_deadlines(train,finish,now=None):
    parsed=[datetime.fromisoformat(x) for x in (train,finish)]
    if any(x.tzinfo is None for x in parsed):raise ValueError('deadlines require timezone')
    a,b=[x.timestamp() for x in parsed]
    if not (time.time() if now is None else now)<a<b:raise ValueError('deadlines must be future and ordered')
    return a,b


def load_data(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('version')!='weekend-v3' or manifest.get('experimental_training_ready') is not True:
        raise ValueError('requires frozen approved weekend-v3 manifest')
    data={};ids=set();groups={};prompts={}
    for split in SPLITS:
        raw=(directory/(split+'.jsonl')).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest['files'][split+'.jsonl']:
            raise ValueError('dataset digest mismatch: '+split)
        rows=[json.loads(line) for line in raw.splitlines()]
        if not rows or len(rows)!=manifest['counts'][split]:raise ValueError('split count mismatch')
        for row in rows:
            label_index(row['question'],row['label'])
            if len(labels(row['question']))<2:raise ValueError('requires at least two options')
            if row['split']!=split or row['id'] in ids:raise ValueError('split or duplicate ID')
            ids.add(row['id'])
            prompt=json.dumps([row['state'],row['question']],sort_keys=True)
            for key,seen in ((row['group'],groups),(prompt,prompts)):
                if key in seen and seen[key]!=split:raise ValueError('cross-split leakage')
                seen[key]=split
        data[split]=rows
    families={r['family'] for r in data['train']}
    if len(families)!=manifest['families'] or len(families)!=24:raise ValueError('expected all 24 families')
    for split in ('validation','calibration','test'):
        if {r['family'] for r in data[split]}!=families:raise ValueError('family coverage mismatch')
    challenge={r['family'] for r in data['challenge']}
    if challenge & families or len(challenge)!=manifest['challenge_families']:raise ValueError('challenge family overlap/count')
    if sum(map(len,data.values()))!=manifest['rows']:raise ValueError('total count mismatch')
    return manifest,data


def grouped_training(encoded):
    result=defaultdict(lambda:defaultdict(list))
    for entry in encoded:result[entry[0]['family']][entry[0]['group']].append(entry)
    return {f:tuple(tuple(rows) for _,rows in sorted(groups.items())) for f,groups in sorted(result.items())}


def sample_update(by_family,rng):
    """One sample per family; group size cannot bias scenario selection."""
    return [rng.choice(rng.choice(groups)) for groups in by_family.values()]


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);args=p.parse_args()
    manifest,data=load_data(args.data)
    print(json.dumps({'status':'passed','counts':manifest['counts'],
        'families':sorted({r['family'] for r in data['train']}),'dataset_sha256':manifest['files']},indent=2))
