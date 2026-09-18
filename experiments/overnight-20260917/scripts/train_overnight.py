"""Bounded, validation-selected multi-task LoRA experiments on local ROCm."""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
from granite_decisions.contracts import label_index, labels
from granite_decisions.general import GeneralBackend, MODEL_ID, REVISION, PROMPT_VERSION
from prepare_overnight_data import validate_splits

FAMILIES = ('banking77', 'boolq', 'synthetic_score')
STOP = False


def stop_requested(signum, frame):
    global STOP
    STOP = True


def atomic_json(path, value):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def probabilities(logits, temperature=1.0):
    z=np.asarray(logits,dtype=float)/temperature
    z-=z.max()
    p=np.exp(z)
    return p/p.sum()


def categorical_nll(logits, target, temperature=1.0):
    z=np.asarray(logits,dtype=float)/temperature
    return float(np.logaddexp.reduce(z)-z[target])


def metrics(predictions, temperatures=None):
    result={}
    for family in FAMILIES:
        rows=[r for r in predictions if r['family']==family]
        if not rows: raise ValueError('missing evaluation family '+family)
        nll=[];acc=[];brier=[];conf=[];mae=[]
        for row in rows:
            p=probabilities(row['logits'],(temperatures or {}).get(family,1.0))
            target=row['target']; one=np.eye(len(p))[target]
            nll.append(categorical_nll(row['logits'],target,(temperatures or {}).get(family,1.0)))
            acc.append(int(p.argmax()==target)); conf.append(float(p.max()))
            brier.append(float(((p-one)**2).sum()))
            mae.append(abs(float(np.dot(np.arange(len(p)),p))-target))
        selected=[i for i,c in enumerate(conf) if c>=0.8]
        ece=0.0
        for lo in np.arange(0,1,0.1):
            bucket=[i for i,c in enumerate(conf) if lo<=c<lo+0.1 or (lo>0.89 and c==1)]
            if bucket:
                ece+=len(bucket)/len(rows)*abs(np.mean([acc[i] for i in bucket])-np.mean([conf[i] for i in bucket]))
        result[family]={'n':len(rows),'nll':float(np.mean(nll)),
            'normalized_nll':float(np.mean([v/math.log(len(r['logits'])) for v,r in zip(nll,rows)])),
            'accuracy':float(np.mean(acc)), 'brier':float(np.mean(brier)), 'ece_10_bins':float(ece),
            'coverage_at_0_8':len(selected)/len(rows),
            'accuracy_at_0_8':float(np.mean([acc[i] for i in selected])) if selected else None}
        if family=='synthetic_score':result[family]['expected_score_mae']=float(np.mean(mae))
    return {'families':result,'objective':float(np.mean([v['normalized_nll'] for v in result.values()]))}


def eligible(candidate, baseline):
    return all(candidate['families'][f]['accuracy']>=baseline['families'][f]['accuracy']-0.03 for f in FAMILIES)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--train-until',required=True,help='Timezone-aware ISO timestamp')
    p.add_argument('--finish-until',required=True)
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--pilot',action='store_true')
    args=p.parse_args()
    train_until=datetime.fromisoformat(args.train_until).timestamp()
    finish_until=datetime.fromisoformat(args.finish_until).timestamp()
    if train_until<=time.time() or finish_until<=train_until or args.steps<1 or args.eval_every<1:
        p.error('invalid deadlines or step counts')
    if datetime.fromisoformat(args.train_until).tzinfo is None or datetime.fromisoformat(args.finish_until).tzinfo is None:
        p.error('deadlines must include timezone')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    data_dir=Path(args.data)
    manifest=json.loads((data_dir/'manifest.json').read_text())
    data={}
    # Dataset validation can inspect split membership; no test labels enter model selection.
    for split in ('train','validation','calibration','test'):
        raw=(data_dir/(split+'.jsonl')).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest['files'][split+'.jsonl']:
            raise ValueError('dataset digest mismatch '+split)
        data[split]=[json.loads(x) for x in raw.splitlines()]
    validate_splits(data)
    for split,rows in data.items():
        for r in rows:label_index(r['question'],r['label'])
    atomic_json(out/'data-manifest.json',manifest)
    signal.signal(signal.SIGTERM,stop_requested)
    signal.signal(signal.SIGINT,stop_requested)
    import torch
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    torch.manual_seed(11)
    backend=GeneralBackend(max_tokens=4096,batch_size=1)

    def encode(rows):
        return [(r,backend.tokens(r['state'],r['question']),label_index(r['question'],r['label'])) for r in rows]

    if args.pilot:
        for s in data:data[s]=[r for f in FAMILIES for r in [x for x in data[s] if x['family']==f][:2]]
    encoded={s:encode(data[s]) for s in ('train','validation')}
    lengths={s:{'min':min(len(e[1]) for e in rows),'max':max(len(e[1]) for e in rows),'n':len(rows)} for s,rows in encoded.items()}
    atomic_json(out/'token-lengths.json',lengths)
    backend.model=get_peft_model(backend.model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,
        target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
    backend.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    backend.model.enable_input_require_grads()
    params=[v for v in backend.model.parameters() if v.requires_grad]
    initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(backend.model).items()}

    def log(event):
        event={'utc':datetime.now(timezone.utc).isoformat(),**event}
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(event,allow_nan=False)+'\n')
        atomic_json(out/'status.json',event)
        print(json.dumps(event,allow_nan=False),flush=True)

    def evaluate(examples, name):
        backend.model.eval(); result=[]
        with torch.no_grad():
            for start in range(0,len(examples),3):
                if time.time()>=finish_until: raise TimeoutError('final evaluation deadline reached')
                batch=examples[start:start+3]
                scores=backend.forward([(tokens,len(labels(r['question']))) for r,tokens,_ in batch])
                for (r,tokens,target),z in zip(batch,scores):
                    result.append({'id':r['id'],'family':r['family'],'target':target,'logits':z.cpu().tolist()})
                if len(result)%99==0:
                    atomic_json(out/(name+'.partial.json'),result)
                    log({'phase':'evaluate','name':name,'processed':len(result),'total':len(examples)})
        atomic_json(out/(name+'.json'),result)
        return metrics(result),result

    # Numerical batch check across different lengths and candidate counts, before optimization.
    examples=[next(e for e in encoded['validation'] if e[0]['family']==f) for f in FAMILIES]
    backend.model.eval()
    with torch.no_grad():
        inputs=[(e[1],len(labels(e[0]['question']))) for e in examples]
        singles=[backend.forward([x])[0].cpu().numpy() for x in inputs]
        batch=[x.cpu().numpy() for x in backend.forward(inputs)]
    errors=[float(np.abs(probabilities(a)-probabilities(b)).max()) for a,b in zip(singles,batch)]
    if max(errors)>0.03:raise RuntimeError('batch probability parity exceeds 0.03: '+str(errors))
    log({'phase':'gpu_checks_passed','batch_probability_errors':errors,'token_lengths':lengths,
         'trainable_parameters':sum(v.numel() for v in params),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__})
    baseline,_=evaluate(encoded['validation'],'baseline-validation')
    atomic_json(out/'baseline-metrics.json',baseline)
    best={'adapter':None,'validation':baseline,'trial':'untuned','step':0}
    atomic_json(out/'best.json',best)
    log({'phase':'baseline_complete','metrics':baseline})
    grid=[(1e-5,17),(3e-5,17),(1e-5,29),(3e-5,29)]
    if args.pilot:grid=grid[:1]
    for trial_index,(lr,seed) in enumerate(grid):
        if STOP or time.time()>=train_until:break
        trial=f'lr{lr:g}-seed{seed}'
        set_peft_model_state_dict(backend.model,initial)
        torch.manual_seed(seed)
        rng=random.Random(seed)
        by_family={f:[e for e in encoded['train'] if e[0]['family']==f] for f in FAMILIES}
        optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=0.01)
        steps=2 if args.pilot else args.steps
        trial_best=baseline['objective']; stale=0;start=time.perf_counter()
        for step in range(1,steps+1):
            if STOP or time.time()>=train_until:break
            backend.model.train();optimizer.zero_grad(set_to_none=True)
            warmup=max(1,min(20,steps//10))
            scale=min(1.0,step/warmup)*(0.1+0.9*(1+math.cos(math.pi*step/steps))/2)
            for group in optimizer.param_groups:group['lr']=lr*scale
            losses=[]
            # Two examples/family/update: gradients represent the equal family mixture.
            for family in FAMILIES:
                for _ in range(2):
                    r,tokens,target=rng.choice(by_family[family]);k=len(labels(r['question']))
                    z=backend.forward([(tokens,k)])[0]
                    loss=torch.nn.functional.cross_entropy(z[None],torch.tensor([target],device='cuda'))/math.log(k)
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                    (loss/6).backward();losses.append(float(loss.detach()))
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True))
            optimizer.step()
            log({'phase':'train','trial':trial,'step':step,'loss':sum(losses)/6,'gradient_norm':norm,
                 'learning_rate':lr*scale,'seconds_per_update':(time.perf_counter()-start)/step})
            if step%args.eval_every==0 or step==steps or STOP or time.time()>=train_until:
                current,_=evaluate(encoded['validation'],trial+f'-step{step}-validation')
                allowed=eligible(current,baseline)
                # Always preserve evaluated checkpoints for audit; unique names avoid overwriting best.
                dest=out/'checkpoints'/(trial+f'-step{step}')
                dest.mkdir(parents=True,exist_ok=False)
                backend.model.save_pretrained(dest,safe_serialization=True)
                atomic_json(dest/'decision_manifest.json',{'model':MODEL_ID,'revision':REVISION,
                    'prompt_version':PROMPT_VERSION,'trial':trial,'step':step,'data_hashes':manifest['files']})
                # Actual saved-weight reload validation on one held validation record.
                probe=encoded['validation'][0];inp=[(probe[1],len(labels(probe[0]['question'])))]
                with torch.no_grad():expected=backend.forward(inp)[0].clone()
                set_peft_model_state_dict(backend.model,load_file(str(dest/'adapter_model.safetensors')))
                with torch.no_grad():restored=backend.forward(inp)[0]
                torch.testing.assert_close(restored,expected,rtol=1e-4,atol=1e-4)
                changed=max(float((v.detach().cpu()-initial[k]).abs().max()) for k,v in get_peft_model_state_dict(backend.model).items())
                if changed<=0:raise RuntimeError('adapter did not change')
                if allowed and current['objective']<best['validation']['objective']-1e-5:
                    best={'adapter':str(dest.resolve()),'validation':current,'trial':trial,'step':step}
                    atomic_json(out/'best.json',best)
                if current['objective']<trial_best-1e-4:trial_best=current['objective'];stale=0
                else:stale+=1
                log({'phase':'validation','trial':trial,'step':step,'eligible':allowed,'metrics':current,
                     'best_trial':best['trial'],'reload':'passed','max_adapter_delta':changed})
                if stale>=3:break
        del optimizer
        if args.pilot:
            atomic_json(out/'pilot.json',{'status':'passed','baseline':baseline,'best':best,'batch_probability_errors':errors})
            log({'phase':'pilot_complete'});return

    # Freeze selection before reading calibration/test through the model.
    atomic_json(out/'selection-frozen.json',best)
    log({'phase':'selection_frozen','selection':best})
    from scipy.optimize import minimize_scalar
    eval_encoded={s:encode(data[s]) for s in ('calibration','test')}
    reports={}
    for name,adapter in [('baseline',None),('selected',best['adapter'])]:
        if name=='selected' and adapter is None:
            reports[name]=reports['baseline'];continue
        set_peft_model_state_dict(backend.model,load_file(str(Path(adapter)/'adapter_model.safetensors')) if adapter else initial)
        _,cal=evaluate(eval_encoded['calibration'],name+'-calibration')
        temperatures={}
        for family in FAMILIES:
            subset=[r for r in cal if r['family']==family]
            def objective(logtemp):
                t=math.exp(logtemp)
                return float(np.mean([categorical_nll(r['logits'],r['target'],t) for r in subset]))
            opt=minimize_scalar(objective,bounds=(-3,3),method='bounded')
            if not opt.success or not math.isfinite(opt.fun):raise RuntimeError('calibration fitting failed')
            temperatures[family]=math.exp(float(opt.x))
        uncal,test=evaluate(eval_encoded['test'],name+'-test')
        reports[name]={'uncalibrated':uncal,'calibrated':metrics(test,temperatures),'temperatures':temperatures}
        atomic_json(out/(name+'-evaluation.json'),reports[name])
    final={'status':'complete','selection':best,'evaluation':reports,'no_automatic_weight_publication':True,
        'calibration_scope':'Experimental family-level temperatures; not a guarantee for novel question schemas.',
        'limitations':'Limited public benchmark subsets; unknown base-model pretraining contamination; synthetic Score data not real-world validation.',
        'completed_at':datetime.now(timezone.utc).isoformat(),'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
    atomic_json(out/'report.json',final)
    log({'phase':'complete','report':str(out/'report.json')})


if __name__=='__main__':main()
