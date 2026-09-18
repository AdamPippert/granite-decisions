"""Optional supervised LoRA over single-token candidate decisions on ROCm.

This minimizes categorical log loss over valid options, not proprietary RLCD.
The base weights stay frozen. Adapter artifacts remain separate from GGUF/head
bundles. This tool does not deploy, convert, calibrate, or claim quality gains.
"""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import time

from granite_decisions.contracts import digest, label_index, labels, questions
from granite_decisions.prompting import decision_messages
from granite_decisions.training import read_dataset, check_disjoint

MODEL = 'ibm-granite/granite-4.1-3b'
REVISION = 'c0650403e44e78ec0262dab1c90914c65b196c4e'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--questions',required=True)
    p.add_argument('--train',required=True)
    p.add_argument('--validation',required=True,help='Validation only; keep final test separate')
    p.add_argument('--out',required=True)
    p.add_argument('--revision',default=REVISION)
    p.add_argument('--steps',type=int,default=3,help='Default is a plumbing smoke, not a full training run')
    p.add_argument('--validation-limit',type=int,default=6,help='Number of state/question pairs')
    p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=17)
    args=p.parse_args()
    if args.steps<1 or args.validation_limit<1 or args.max_tokens<1 or not math.isfinite(args.learning_rate) or args.learning_rate<=0:
        p.error('steps, validation-limit, max-tokens and learning-rate must be positive')
    if not re.fullmatch('[0-9a-f]{40}',args.revision): p.error('revision must be an immutable 40-character commit')
    out=Path(args.out)
    if out.exists(): p.error('output exists; choose a new directory')
    qs=questions(json.loads(Path(args.questions).read_text()))
    train,valid=[read_dataset(path,qs) for path in (args.train,args.validation)]
    check_disjoint(train,valid)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('ROCm PyTorch GPU required; CPU fallback is not a GPU validation')
    torch.manual_seed(args.seed); random.seed(args.seed)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=args.revision,trust_remote_code=False)
    code_ids=[tokenizer.encode(c,add_special_tokens=False) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    if any(len(x)!=1 for x in code_ids) or len({x[0] for x in code_ids})!=26:
        raise RuntimeError('single-token unique candidate codes required')
    def examples(rows):
        result=[]
        for row in rows:
            for key,q in qs.items():
                tokens=tokenizer.apply_chat_template(decision_messages(row['state'],q),tokenize=True,add_generation_prompt=True)
                if len(tokens)>args.max_tokens: raise ValueError('context overflow; refusing silent truncation')
                result.append((tokens,[x[0] for x in code_ids[:len(labels(q))]],label_index(q,row['labels'][key])))
        return result
    training=examples(train); validation=examples(valid)
    random.shuffle(training)
    random.shuffle(validation)
    validation=validation[:args.validation_limit]
    out.mkdir(parents=True,exist_ok=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=args.revision,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',trust_remote_code=False).to('cuda')
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,
        target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    parameters=[v for v in model.parameters() if v.requires_grad]
    before={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate)
    def logits(example):
        ids,candidates,_=example
        input_ids=torch.tensor([ids],device='cuda')
        scores=model(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),use_cache=False,logits_to_keep=1).logits[0,-1].float()
        return scores[torch.tensor(candidates,device='cuda')]
    def evaluate():
        model.eval()
        with torch.no_grad():
            values=[float(torch.nn.functional.cross_entropy(logits(e)[None],torch.tensor([e[2]],device='cuda'))) for e in validation]
        return sum(values)/len(values)
    start=time.perf_counter()
    initial=evaluate()
    losses=[]; gradients=[]
    model.train()
    for step in range(args.steps):
        e=training[step%len(training)]
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(logits(e)[None],torch.tensor([e[2]],device='cuda'))
        if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
        if float(norm)<=0: raise RuntimeError('no adapter gradient')
        optimizer.step()
        losses.append(float(loss.detach())); gradients.append(float(norm))
        print(json.dumps({'step':step+1,'loss':losses[-1],'gradient_norm':gradients[-1]}),flush=True)
    final=evaluate()
    state=get_peft_model_state_dict(model)
    delta=max(float((state[k].detach().cpu()-v).abs().max()) for k,v in before.items())
    if delta<=0: raise RuntimeError('adapter weights did not change')
    model.save_pretrained(out/'adapter',safe_serialization=True)
    tokenizer.save_pretrained(out/'adapter')
    # Verify serialized adapter restoration, not merely file existence.
    with torch.no_grad(): expected=logits(validation[0]).detach().clone()
    set_peft_model_state_dict(model,load_file(str(out/'adapter'/'adapter_model.safetensors')))
    with torch.no_grad(): restored=logits(validation[0]).detach()
    torch.testing.assert_close(restored,expected,rtol=1e-4,atol=1e-4)
    report={'status':'passed','method':'supervised_candidate_cross_entropy_lora','quality_benchmark':False,
        'model':MODEL,'revision':args.revision,'seed':args.seed,'steps':args.steps,'learning_rate':args.learning_rate,
        'trainable_parameters':sum(v.numel() for v in parameters),'training_losses':losses,'gradient_norms':gradients,
        'max_adapter_delta':delta,'adapter_reload':'passed','validation_pairs':len(validation),
        'validation_nll_before':initial,'validation_nll_after':final,'calibration':'not_fitted',
        'data_hashes':{'train':digest(train),'validation':digest(valid),'questions':digest(qs)},
        'gpu':torch.cuda.get_device_name(0),'hip':torch.version.hip,
        'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','peft','accelerate']},
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.perf_counter()-start}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
