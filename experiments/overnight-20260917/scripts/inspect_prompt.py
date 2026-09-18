import json
from pathlib import Path
from granite_decisions.general import GeneralBackend
from granite_decisions.prompting import decision_messages
from granite_decisions.contracts import labels
b=GeneralBackend();t=b.torch
rows=[json.loads(s) for s in Path('data-ready/validation.jsonl').read_text().splitlines()]
for family in ['banking77','boolq','synthetic_score']:
 r=next(r for r in rows if r['family']==family);ids=b.tokens(r['state'],r['question'])
 with t.no_grad():
  x=t.tensor([ids],device='cuda');z=b.model(input_ids=x,logits_to_keep=1).logits[0,-1].float()
  vals,indices=z.topk(8)
  pred=b.model.generate(input_ids=x,max_new_tokens=12,do_sample=False,pad_token_id=b.tokenizer.eos_token_id)
 print(json.dumps({'family':family,'label':r['label'],'tail':b.tokenizer.decode(ids[-30:]),'top':[(b.tokenizer.decode([int(i)]),float(v)) for i,v in zip(indices,vals)],'generated':b.tokenizer.decode(pred[0,len(ids):]),'candidate_probs':t.softmax(z[b.code_ids[:len(labels(r['question']))]],-1).tolist()}),flush=True)
