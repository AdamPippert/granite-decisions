"""Freeze the validated experimental dataset, preserving all evaluation rows.

Jev below-threshold groups are excluded from training without changing labels.
A freeze is not evidence of real-world model quality or independent human review.
"""
import argparse
import hashlib
import json
from pathlib import Path
from corpus import canonical,digest
from expanded_corpus import SPLITS,validate
from refine_v3 import select
from quality_gate import interpret


def read_rows(path):return [json.loads(s) for s in path.read_text().splitlines()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_quarantines(rows,judgments):
    planned={r['id']:r for r in select(rows)}
    if len(judgments)!=len(planned) or {r['id'] for r in judgments}!=set(planned):raise ValueError('incomplete or duplicate audit')
    groups=set()
    for r in judgments:
        row=planned[r['id']]
        if r['input_sha256']!=digest(row) or r['group']!=row['group']:raise ValueError('audit identity mismatch')
        try:observed=interpret(row,r['response'])
        except (KeyError,ValueError,TypeError):observed={'pass':False}
        if type(r['pass']) is not bool or r['pass']!=observed['pass']:raise ValueError('saved judgment disagrees with raw response')
        if not r['pass']:groups.add(row['group'])
    return groups


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--sources',type=Path,required=True);p.add_argument('--tokens',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();data={s:read_rows(a.candidates/(s+'.jsonl')) for s in SPLITS};sources={s['id']:s for s in json.loads(a.sources.read_text())}
    validate(data,sources)
    summary=json.loads((a.audit/'summary.json').read_text())
    if summary['data_sha256']!=sha(a.candidates/'train.jsonl') or summary['sources_sha256']!=sha(a.sources):raise ValueError('audit data/source digest mismatch')
    mutations=read_rows(a.audit/'mutation-controls.jsonl')
    if len(mutations)!=480 or any(not r['issues'] for r in mutations):raise ValueError('mutation gate failed')
    groups=audit_quarantines(data['train'],read_rows(a.audit/'judgments.jsonl'))
    if sorted(groups)!=summary['quarantine_groups']:raise ValueError('summary quarantine mismatch')
    tokens=json.loads(a.tokens.read_text())
    for split in SPLITS:
        check=tokens['splits'][split]
        if check['sha256']!=sha(a.candidates/(split+'.jsonl')) or check['n']!=len(data[split]) or check['max']>4096 or check['over_4096']!=0:raise ValueError('tokenizer preflight mismatch')
    quarantined=[r for r in data['train'] if r['group'] in groups]
    data['train']=[r for r in data['train'] if r['group'] not in groups]
    report=validate(data,sources)
    a.out.mkdir(parents=True,exist_ok=False)
    for split,rows in data.items():(a.out/(split+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in rows))
    (a.out/'quarantined-train.jsonl').write_text(''.join(canonical(r)+'\n' for r in quarantined))
    report.update(experimental_training_ready=True,independent_human_review=False,quarantined_rows=len(quarantined),quarantined_groups=sorted(groups),
      curation_scope='96 preselected training records audited by Jev; all records checked deterministically. Whole flagged training groups quarantined. Evaluation never filtered by Jev.',
      sources_sha256=sha(a.sources),tokenizer_check_sha256=sha(a.tokens),judge_summary_sha256=sha(a.audit/'summary.json'),
      files={s+'.jsonl':sha(a.out/(s+'.jsonl')) for s in SPLITS},quarantine_sha256=sha(a.out/'quarantined-train.jsonl'),
      generation_manifest_sha256=sha(a.candidates/'manifest.json'),
      training_contract={'input_fields':['state','question'],'target_field':'label','metadata_never_prompted':['audit','source_ids','family','split','group','provenance'],
        'suggested_sampling':'uniform family, then uniform scenario group, then uniform variant','suggested_loss':'categorical cross-entropy divided by log(number of options), averaged across families','max_prompt_tokens_checked':4096,'base_revision':tokens['revision'],
        'warning':'The historical overnight trainer hardcodes three old families. Do not pass this corpus to it unchanged; use a trainer that consumes all manifest families and normalizes accumulation correctly.'})
    (a.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','training_contract')},indent=2))

if __name__=='__main__':main()
