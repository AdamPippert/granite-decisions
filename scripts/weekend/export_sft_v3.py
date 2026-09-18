"""Export the native rows into conversational prompt/completion pairs.

Uses the actual numeric-code inference prompt. Only prompt/completion belong in
model input; provenance, group, valid codes and native labels are audit metadata.
This export does not implement family-balanced categorical training loss.
"""
import argparse
import importlib.util
import json
from pathlib import Path
from corpus import ROOT,canonical
from granite_decisions.contracts import label_index,labels


def load_messages(path):
    spec=importlib.util.spec_from_file_location('granite_decisions.general',path)
    backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
    return backend.messages


def export(row,messages):
    code=label_index(row['question'],row['label'])
    prompt=messages(row['state'],row['question'])
    return {'id':row['id'],'group':row['group'],'family':row['family'],'split':row['split'],
      'prompt':prompt,'completion':[{'role':'assistant','content':str(code)}],
      'valid_codes':[str(i) for i in range(len(labels(row['question'])))],
      'native_label_json':canonical(row['label']),'source_ids':row['source_ids'],'license':row['license']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backend',type=Path,default=ROOT/'experiments/overnight-20260917/src/granite_decisions/general.py')
    a=p.parse_args();messages=load_messages(a.backend);rows=[json.loads(s) for s in a.data.read_text().splitlines()]
    with a.out.open('x') as f:
        for row in rows:f.write(canonical(export(row,messages))+'\n')

if __name__=='__main__':main()
