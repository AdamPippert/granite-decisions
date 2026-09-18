from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/weekend'))
from expanded_corpus import generate,RESEARCH,validate
from refine_v3 import semantic_controls,select,judge_row,judge_control,collect
from freeze_v3 import audit_quarantines
from corpus import digest
from export_sft_v3 import export,load_messages
from granite_decisions.contracts import label_index


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=generate();cls.sources={r['id']:r for r in json.loads((RESEARCH/'expanded-v3/sources.json').read_text())}

    def test_sft_export_uses_runtime_prompt_and_typed_target(self):
        backend=Path(__file__).resolve().parents[1]/'experiments/overnight-20260917/src/granite_decisions/general.py'
        messages=load_messages(backend)
        for row in select(self.data['train']):
            out=export(row,messages)
            self.assertEqual(out['prompt'],messages(row['state'],row['question']))
            self.assertEqual(out['completion'],[{'role':'assistant','content':str(label_index(row['question'],row['label']))}])
            self.assertIn(out['completion'][0]['content'],out['valid_codes'])
            self.assertNotIn('oracle_trace',json.dumps(out['prompt']))

    def test_controls_balanced_and_train_only_selection(self):
        controls=semantic_controls();self.assertEqual(len(controls),80)
        self.assertEqual(sum(r['expected_valid'] for r in controls),40)
        self.assertEqual(len({r['defect'] for r in controls}),10)
        self.assertEqual(len(select(self.data['train'])),96)
        with self.assertRaises(ValueError):select(self.data['test'])
        with self.assertRaises(ValueError):select(self.data['challenge'])

    def test_gold_and_provenance_never_sent(self):
        row=self.data['train'][0];seen=[]
        def call(state,questions):seen.append((state,questions));raise RuntimeError('offline')
        result=judge_row(row,self.sources,call)
        self.assertFalse(result['pass']);self.assertEqual(len(seen),1)
        self.assertEqual(set(seen[0][0]),{'policy','case_record'})
        self.assertNotIn('audit',seen[0][0]);self.assertNotIn('label',seen[0][0])
        bad=deepcopy(row);bad['label']='broken';seen.clear()
        self.assertFalse(judge_row(bad,self.sources,call)['pass']);self.assertEqual(seen,[])

    def test_semantic_controls_fail_closed_on_malformed_response(self):
        row=semantic_controls()[0]
        for response in ({},{'answers':{'quality':None}},{'answers':{'quality':{'type':'noul','noul':float('nan')}}}):
            self.assertFalse(judge_control(row,lambda s,q:response)['valid_response'])

    def test_resume_rejects_changed_input_without_calling(self):
        row=semantic_controls()[0]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'audit.jsonl'
            path.write_text(json.dumps({'id':row['id'],'input_sha256':'wrong'})+'\n')
            with self.assertRaisesRegex(ValueError,'mismatch'):collect(path,[row],lambda _:self.fail('must not call API'),True)

    def test_freeze_rejects_incomplete_and_forged_audit(self):
        rows=self.data['train'];selected=select(rows)
        with self.assertRaisesRegex(ValueError,'incomplete'):audit_quarantines(rows,[])
        judgments=[{'id':r['id'],'group':r['group'],'input_sha256':digest(r),'pass':True,'response':{}} for r in selected]
        with self.assertRaisesRegex(ValueError,'raw response'):audit_quarantines(rows,judgments)

    def test_branch_coverage_gate_catches_missing_exception(self):
        data=deepcopy(self.data)
        data['validation']=[r for r in data['validation'] if not (r['audit']['policy_family']=='vendor' and r['audit']['policy_index']==1 and r['audit']['oracle_trace']=='rule_1')]
        with self.assertRaisesRegex(ValueError,'branch coverage'):validate(data,self.sources)

if __name__=='__main__':unittest.main()
