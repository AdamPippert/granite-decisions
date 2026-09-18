"""CPU-only context preflight using the pinned Granite tokenizer and real prompt."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from corpus import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backend',type=Path,default=ROOT/'experiments/overnight-20260917/src/granite_decisions/general.py')
    a=p.parse_args()
    from transformers import AutoTokenizer
    spec=importlib.util.spec_from_file_location('granite_decisions.general',a.backend)
    backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
    tokenizer=AutoTokenizer.from_pretrained(backend.MODEL_ID,revision=backend.REVISION,trust_remote_code=False)
    report={'model':backend.MODEL_ID,'revision':backend.REVISION,'prompt_version':backend.PROMPT_VERSION,'gpu_used':False,'splits':{}}
    for split in ('train','validation','calibration','test','challenge'):
        path=a.data/(split+'.jsonl');rows=[json.loads(s) for s in path.read_text().splitlines()]
        lengths=sorted(len(tokenizer.apply_chat_template(backend.messages(r['state'],r['question']),tokenize=True,add_generation_prompt=True)) for r in rows)
        report['splits'][split]={'n':len(rows),'max':max(lengths),'median':lengths[len(lengths)//2],'p95':lengths[int(len(lengths)*.95)],'over_4096':sum(n>4096 for n in lengths),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    with a.out.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    if any(s['over_4096'] for s in report['splits'].values()):raise ValueError('context overflow; no truncation allowed')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
