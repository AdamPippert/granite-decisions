"""Export heterogeneous typed decisions as reversible flat JSON for HF Viewer."""
import argparse
import json
from pathlib import Path
from corpus import canonical


def flatten(row):
    return {'id':row['id'],'family':row['family'],'group':row['group'],'split':row['split'],
            'type':row['question']['type'],'state_json':canonical(row['state']),
            'question_json':canonical(row['question']),'label_json':canonical(row['label']),
            'source_ids':row['source_ids'],'license':row['license'],
            'metadata_json':canonical({k:v for k,v in row.items() if k not in {'id','family','group','split','state','question','label','source_ids','license'}})}


def restore(row):
    result={k:row[k] for k in ('id','family','group','split','source_ids','license')}
    result.update(json.loads(row['metadata_json']))
    for field in ('state','question','label'):result[field]=json.loads(row[field+'_json'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=[json.loads(s) for s in a.data.read_text().splitlines()]
    exported=[flatten(row) for row in rows]
    if [restore(row) for row in exported]!=rows:raise ValueError('export roundtrip mismatch')
    with a.out.open('x') as f:f.write(''.join(canonical(row)+'\n' for row in exported))

if __name__=='__main__':main()
