import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from granite_decisions.prompting import decision_messages


def load_script(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

b=load_script('benchmark_decisions')


class BenchmarkTests(unittest.TestCase):
    def test_choice_order_matches_training_targets(self):
        q={'type':'choice','instructions':'Select','criteria':{'z':'last','a':'first'}}
        self.assertEqual(b.distribution(q,{'type':'choice','probabilities':{'z':.1,'a':.9}}),[.9,.1])
        options=json.loads(decision_messages({},q)[0]['content'].split('\n',1)[1])['options']
        self.assertEqual([(v['code'],v['option']) for v in options],[('A','a'),('B','z')])

    def test_invalid_distributions_rejected(self):
        q={'type':'choice','criteria':{'a':'','b':''}}
        for probs in ({'a':1},{'a':.9,'b':.9},{'a':float('nan'),'b':.1},{'a':-.1,'b':1.1}):
            with self.assertRaises(ValueError): b.distribution(q,{'type':'choice','probabilities':probs})

    def test_noul_is_not_confidence(self):
        self.assertEqual(b.distribution({'type':'noul'},{'type':'noul','noul':.2}),[.8,.2])

    def test_compare_checks_identity_and_uses_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            qs={'q':{'type':'noul','instructions':'True?'}}
            rows=[{'id':'r1','state':'input','labels':{'q':True}}]
            (root/'q.json').write_text(json.dumps(qs))
            (root/'data.jsonl').write_text(json.dumps(rows[0])+'\n')
            record={'id':'r1','state_sha256':b.digest('input'),'questions_sha256':b.digest(qs),'dataset_sha256':b.digest(rows),
                    'elapsed_ms':1,'response':{'model':'test','answers':{'q':{'type':'noul','noul':.1}}}}
            result=root/'jev.jsonl'; result.write_text(json.dumps(record)+'\n')
            args=argparse.Namespace(questions=root/'q.json',data=root/'data.jsonl',results=[result],out=root/'report.json')
            b.compare(args)
            report=json.loads(args.out.read_text())
            self.assertEqual(report['providers']['jev']['fields']['q']['accuracy'],0)
            record['state_sha256']='wrong'
            result.write_text(json.dumps(record)+'\n')
            with self.assertRaises(ValueError): b.compare(args)
            record['state_sha256']=b.digest('input')
            result.write_text((json.dumps(record)+'\n')*2)
            with self.assertRaises(ValueError): b.compare(args)
