"""Training requirements: balanced objective, split integrity and deadline behavior."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/weekend-20260918/scripts'))
# Trainer must use the exact frozen runtime that was used for token validation.
import granite_decisions
runtime=str(ROOT/'experiments/overnight-20260917/src/granite_decisions')
if runtime not in granite_decisions.__path__:granite_decisions.__path__.append(runtime)
from weekend_data import load_data,check_deadlines,grouped_training,sample_update
from train_weekend import metrics,eligible,categorical_nll

DATA=ROOT/'research/weekend-20260918/expanded-v3/data'


class WeekendTrainingTests(unittest.TestCase):
    def test_frozen_data_all_families_and_challenge(self):
        manifest,data=load_data(DATA)
        self.assertEqual(len(data['train']),5740)
        self.assertEqual(len({r['family'] for r in data['train']}),24)
        self.assertEqual(len({r['family'] for r in data['challenge']}),6)

    def test_tampered_dataset_fails_before_gpu_load(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            (d/'manifest.json').write_bytes((DATA/'manifest.json').read_bytes())
            (d/'train.jsonl').write_bytes((DATA/'train.jsonl').read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'digest mismatch'):load_data(d)

    def test_macro_nll_not_dominated_by_large_family(self):
        rows=[dict(family='a',target=0,logits=[0,0]),dict(family='b',target=0,logits=[0]*5)]
        self.assertAlmostEqual(metrics(rows)['objective'],1.0)
        rows[1]['logits']=[3,0,0,0,0]
        self.assertAlmostEqual(metrics(rows)['objective'],metrics(rows+[rows[1]]*20)['objective'])
        self.assertAlmostEqual(categorical_nll([1000,-1000],1),2000)

    def test_each_family_guard_applies_to_new_families(self):
        base={'families':{str(i):{'accuracy':.8} for i in range(24)}}
        candidate=deepcopy(base);candidate['families']['23']['accuracy']=.76
        self.assertFalse(eligible(candidate,base))
        candidate['families']['23']['accuracy']=.78
        self.assertTrue(eligible(candidate,base))

    def test_groups_not_rows_determine_sampling_probability(self):
        encoded=[({'family':f,'group':g},[],0) for f in ('a','b') for g in ('small','large') for _ in range(1 if g=='small' else 50)]
        grouped=grouped_training(encoded);rng=random.Random(7);small=0
        for _ in range(2000):
            batch=sample_update(grouped,rng)
            self.assertEqual({r['family'] for r,_,_ in batch},{'a','b'})
            small+=sum(r['group']=='small' for r,_,_ in batch)
        self.assertTrue(1800<small<2200)

    def test_deadlines_timezone_order_and_expiry(self):
        for train,finish in [('2026-09-20T15:30:00','2026-09-20T16:55:00-07:00'),('2026-09-20T17:30:00-07:00','2026-09-20T16:55:00-07:00')]:
            with self.assertRaises(ValueError):check_deadlines(train,finish,now=0)
        a,b=check_deadlines('2026-09-20T15:30:00-07:00','2026-09-20T16:55:00-07:00',now=0)
        self.assertEqual(b-a,85*60)
        with self.assertRaises(ValueError):check_deadlines('2026-09-20T15:30:00-07:00','2026-09-20T16:55:00-07:00',now=b)

    def test_challenge_metrics_use_own_families(self):
        result=metrics([dict(family='unseen',kind='score',target=2,logits=[0,0,20])])
        self.assertEqual(result['families']['unseen']['accuracy'],1)
        self.assertLess(result['families']['unseen']['expected_score_mae'],1e-6)


class SupervisorTests(unittest.TestCase):
    def test_start_window_rejects_early_and_late(self):
        from datetime import datetime
        from supervise import validate_start
        c=json.loads((ROOT/'experiments/weekend-20260918/schedule.json').read_text())
        start=datetime.fromisoformat(c['start']).timestamp()
        validate_start(c,start);validate_start(c,start+899)
        for t in (start-1,start+901):
            with self.assertRaisesRegex(ValueError,'start window'):validate_start(c,t)

    def test_inventory_rejects_changed_code_and_escaping_path(self):
        import hashlib
        from supervise import verify_inventory
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'code.py';p.write_text('original')
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            (d/'hashes').write_text(digest+'  code.py\n')
            verify_inventory(d,'hashes');p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_inventory(d,'hashes')
            (d/'hashes').write_text(digest+'  ../code.py\n')
            with self.assertRaisesRegex(ValueError,'escapes'):verify_inventory(d,'hashes')

    def test_deadline_kills_child_ignoring_term(self):
        import time
        from supervise import run_child
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);start=time.time()
            result=run_child([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'],start+1,d/'log',d/'status',d/'cancel',grace=.2)
            self.assertEqual(result['reason'],'deadline')
            self.assertEqual(result['exit_code'],-9)
            self.assertLess(time.time()-start,3)

    def test_cancel_and_nonzero_exit_are_not_success(self):
        import time
        from supervise import run_child
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);(d/'cancel').touch()
            result=run_child([sys.executable,'-c','import time;time.sleep(30)'],time.time()+10,d/'log',d/'status',d/'cancel',grace=.2)
            self.assertEqual(result['reason'],'cancelled');self.assertEqual(result['phase'],'stopped')
            (d/'cancel').unlink()
            result=run_child([sys.executable,'-c','raise SystemExit(7)'],time.time()+10,d/'log',d/'status',d/'cancel',grace=.2)
            self.assertEqual(result['exit_code'],7);self.assertEqual(result['phase'],'stopped')

    def test_normal_completion_preserves_log_and_status(self):
        import time
        from supervise import run_child
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            result=run_child([sys.executable,'-c','print("finished")'],time.time()+10,d/'log',d/'status',d/'cancel')
            self.assertEqual(result['phase'],'finished');self.assertEqual(result['exit_code'],0)
            self.assertIn('finished',(d/'log').read_text())
            self.assertEqual(json.loads((d/'status').read_text()),result)
