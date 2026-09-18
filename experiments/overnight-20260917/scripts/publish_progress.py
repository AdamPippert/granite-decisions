"""Collect bounded overnight evidence and publish an explicitly qualified analysis."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT=Path('/home/adam/granite-decisions-overnight')
REMOTE='/var/home/adam/Development/granite-decisions/experiments/overnight-20260917'
SSH=['ssh','-F','/dev/null','-i','/home/adam/.ssh/homelab-hosts','-o','IdentitiesOnly=yes',
     '-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=15',
     'adam@hyde-fedora.tail4df14e.ts.net']
FORGE_SSH='ssh -F /dev/null -i /home/adam/.ssh/homelab-hosts -o UserKnownHostsFile=/home/adam/granite-decisions-overnight/forgejo-known-hosts -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o BatchMode=yes'
GH_SSH='ssh -F /dev/null -i /home/adam/.ssh/github-adampippert -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o BatchMode=yes'
FORGE='ssh://git@superrouter.tail4df14e.ts.net:2223/adam/granite-decisions.git'
GH='git@github.com:AdamPippert/granite-decisions.git'


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def analyze(run):
    report=read(run/'report.json')
    best=read(run/'selection-frozen.json',read(run/'best.json'))
    base=read(run/'baseline-metrics.json')
    status=read(run/'status.json',{})
    events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()] if (run/'events.jsonl').exists() else []
    updates=[e for e in events if e.get('phase')=='train']
    trials={e['trial']:e['step'] for e in updates}
    lines=['# Overnight progress — September 18, 2026','',
        'Automated analysis of saved experiment evidence. Personal project by Adam Pippert.',
        'Scheduled cutoff: 05:00 America/Los_Angeles. This report does not deploy or publish adapter weights.','',
        f"Run status: **{'complete' if report and report.get('status')=='complete' else 'incomplete'}**. Last recorded phase: `{status.get('phase','unavailable')}`.",
        f'Completed optimizer updates: **{len(updates)}**. Trial progress: `{json.dumps(trials,sort_keys=True)}`.','',
        '## Selection evidence','',
        'The fixed objective is the equal-family mean of candidate cross entropy divided by log(option count). '
        'Selection used validation only, with a guard against more than a 3 percentage point accuracy regression in any family. '
        'Calibration and test data did not select checkpoints.']
    if best and base:
        selected=best['validation'];delta=base['objective']-selected['objective']
        lines+=['',f"Selected candidate: `{best['trial']}`, step {best['step']}. Validation objective: "
                f"{base['objective']:.6f} → {selected['objective']:.6f} (decrease {delta:.6f})."]
        if not best.get('adapter'):lines+=['No adapter satisfied the predeclared improvement and accuracy guardrails; the untuned baseline was retained.']
        lines+=['','| Family | Baseline validation accuracy | Selected validation accuracy |','|---|---:|---:|']
        for f,v in base['families'].items():
            lines+=[f"| {f} | {v['accuracy']:.2%} | {selected['families'][f]['accuracy']:.2%} |"]
    if report and report.get('status')=='complete':
        ev=report['evaluation']
        lines+=['','## Held-out test evidence','',
                '| Family | n | Baseline accuracy | Selected accuracy | Baseline calibrated NLL | Selected calibrated NLL |',
                '|---|---:|---:|---:|---:|---:|']
        for f,a in ev['baseline']['calibrated']['families'].items():
            b=ev['selected']['calibrated']['families'][f]
            lines+=[f"| {f} | {a['n']} | {a['accuracy']:.2%} | {b['accuracy']:.2%} | {a['nll']:.4f} | {b['nll']:.4f} |"]
        lines+=['','Temperatures were fitted on the separate calibration split. Full Brier, calibration-error, '
                'and confidence-threshold coverage measurements are in `evidence/report.json`. '
                'These measurements are descriptive; this small experiment does not establish statistical significance or general Jev equivalence.']
    else:
        lines+=['','Final held-out evaluation did not complete. Saved validation changes are provisional; '
                'no held-out improvement claim is made. Check the saved status and completed validation artifacts before continuing.']
    jev_dir=ROOT/'jev-evaluation'
    jev_summary=read(jev_dir/'summary.json')
    if jev_summary:
        lines+=['','## Independent Jev comparison','',
            'Jev evaluated a fixed 60-example test subset (20 per family). These outputs were never '
            'training labels or checkpoint-selection inputs. Invalid probability responses are rejected, '
            'not renormalized; paired accuracy includes only valid responses, so exclusions may bias the comparison. '
            'Rejection counts are in `jev-evaluation/summary.json`. This small sample is descriptive.']
        baseline_test=read(run/'baseline-test.json')
        selected_test=read(run/'selected-test.json',baseline_test if best and not best.get('adapter') else None)
        if baseline_test and selected_test:
            source_manifest=read(run/'data-manifest.json')
            if jev_summary['test_sha256']!=source_manifest['files']['test.jsonl']:
                raise ValueError('Jev/Granite test identity mismatch')
            jev_rows=[json.loads(s) for s in (jev_dir/'predictions.jsonl').read_text().splitlines()]
            maps=[{r['id']:r for r in rows} for rows in [baseline_test,selected_test]]
            lines+=['','| Family | n | Jev accuracy | Untuned Granite accuracy | Selected Granite accuracy |',
                    '|---|---:|---:|---:|---:|']
            for family in ('banking77','boolq','synthetic_score'):
                subset=[r for r in jev_rows if r['family']==family and r['correct'] is not None];values=[]
                if not subset:
                    lines+=[f'| {family} | 0 | unavailable | unavailable | unavailable |'];continue
                for lookup in maps:
                    pairs=[(r,lookup[r['id']]) for r in subset]
                    if any(a['target']!=b['target'] or a['family']!=b['family'] for a,b in pairs):
                        raise ValueError('Jev/Granite row identity mismatch')
                    values.append(sum(max(range(len(b['logits'])),key=b['logits'].__getitem__)==b['target'] for a,b in pairs)/len(pairs))
                accuracy=sum(r['correct'] for r in subset)/len(subset)
                lines+=[f'| {family} | {len(subset)} | {accuracy:.2%} | {values[0]:.2%} | {values[1]:.2%} |']
        else:
            lines+=['Paired Granite test predictions are incomplete; see `jev-evaluation/summary.json` for Jev-only results.']
    lines+=['','## Scope and provenance','',
        '7,164 training records: 3,068 BANKING77, 2,048 BoolQ, and 2,048 original Score examples. '
        'Validation, calibration, and test each contain 666 records. The base model revision is pinned. '
        'Public benchmark exposure during the base model’s pretraining is unknown.','',
        '- [BANKING77](https://huggingface.co/datasets/PolyAI/banking77): CC-BY-4.0; Casanueva et al. (2020).',
        '- [BoolQ](https://huggingface.co/datasets/google/boolq): CC-BY-SA-3.0; Clark et al. (2019).',
        '- [Original Score dataset](https://huggingface.co/datasets/adampippert/granite-decisions-synthetic): MIT, `score_interval_v1` configuration.',
        '','Source URLs, immutable revisions, content hashes, transformations, and split policy are in `data-manifest.json`. '
        'External dataset content is not republished here or relicensed as MIT. Fullcollar remains deferred.','',
        '## Next decision','',
        'Inspect per-family test regressions and calibration before promoting an adapter. '
        'Expand the independent evaluation to user-relevant decision schemas and adversarial inputs. '
        'Further training should be a new experiment with a fresh selection plan; do not repeatedly select against this test split.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if not args.dry_run and datetime.now(timezone.utc)<datetime.fromisoformat('2026-09-18T12:01:00+00:00'):
        raise RuntimeError('Publication is scheduled after the 5 AM Pacific cutoff')
    if not args.dry_run:
        account=json.loads(subprocess.check_output(['/usr/bin/uvx','hf','auth','whoami','--format','json'],text=True,timeout=60))
        if account.get('user')!='adampippert':raise RuntimeError('Personal HF account identity check failed')
    with tempfile.TemporaryDirectory(prefix='granite-progress-') as tmp:
        tmp=Path(tmp);archive=tmp/'evidence.tar.gz'
        with archive.open('wb') as f:
            subprocess.run(SSH+[f"tar -czf - --exclude='run/checkpoints' -C {REMOTE} run"],stdout=f,check=True,timeout=120)
        with tarfile.open(archive) as tar:tar.extractall(tmp,filter='data')
        run=tmp/'run';summary=analyze(run)
        (ROOT/'PROGRESS-preview.md').write_text(summary)
        if args.dry_run:
            print(summary);return
        # Clone the current remote main, preserving any changes made since v0.1.0.
        repo=tmp/'repo'
        subprocess.run(['git','-c','core.sshCommand='+FORGE_SSH,'clone','--branch','main',FORGE,str(repo)],check=True,timeout=120)
        dest=repo/'experiments/overnight-20260917';dest.mkdir(parents=True,exist_ok=True)
        for folder in ['scripts','src','tests','reviews','jev-evaluation']:
            if (ROOT/folder).exists():shutil.copytree(ROOT/folder,dest/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
        for name in ['PLAN.md','source-lock.json','source-manifest.json','README.md']:
            shutil.copy2(ROOT/name,dest/name)
        shutil.copy2(ROOT/'data-ready/manifest.json',dest/'data-manifest.json')
        shutil.copytree(run,dest/'evidence',dirs_exist_ok=True)
        (dest/'PROGRESS.md').write_text(summary)
        for f in dest.rglob('*'):
            if f.is_file() and (f.suffix in ['.safetensors','.gguf','.bin'] or f.name in ['token','.env']):
                raise RuntimeError('Unexpected private or model-weight artifact '+str(f))
        def git(*argv):
            return subprocess.check_output(['git','-C',str(repo),*argv],text=True).strip()
        git('config','user.name','Adam Pippert')
        git('config','user.email','5215461+AdamPippert@users.noreply.github.com')
        git('add','experiments/overnight-20260917')
        git('diff','--cached','--check')
        git('commit','-m','Record bounded overnight Granite training experiment and evaluated progress')
        commit=git('rev-parse','HEAD')
        publication={'commit':commit,'created_at':datetime.now(timezone.utc).isoformat(),'forgejo':False,'github':False,'huggingface':False}
        record=ROOT/'publication-result.json'
        record.write_text(json.dumps(publication,indent=2)+'\n')
        git('-c','core.sshCommand='+FORGE_SSH,'push','origin','HEAD:main')
        publication['forgejo']=True;record.write_text(json.dumps(publication,indent=2)+'\n')
        git('-c','core.sshCommand='+GH_SSH,'push',GH,'HEAD:main')
        publication['github']=True;record.write_text(json.dumps(publication,indent=2)+'\n')
        subprocess.run(['/usr/bin/uvx','hf','upload','adampippert/granite-decisions',str(dest),
            'experiments/overnight-20260917','--type','model','--commit-message','Publish analyzed overnight experiment progress'],check=True,timeout=300)
        publication['huggingface']=True;record.write_text(json.dumps(publication,indent=2)+'\n')
        print(json.dumps(publication))


if __name__=='__main__':main()
