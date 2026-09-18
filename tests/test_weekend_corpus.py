"""Policy acceptance examples written independently of generator sampling."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/weekend'))
from corpus import catalog, oracle, generate, validate, validate_row, RESEARCH
from quality_gate import controls, run, select, interpret
from granite_decisions.contracts import questions
from export_hf import flatten, restore


class WeekendCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=generate();cls.sources={s['id']:s for s in json.loads((RESEARCH/'sources.json').read_text())}

    def test_manual_decision_examples(self):
        # Literal requirements examples; expected results do not call generator.
        examples=[
         ('release',{'budget_remaining_minutes':10,'rollback_tested':True},10,'release'),
         ('release',{'budget_remaining_minutes':9,'rollback_tested':True},10,'hold'),
         ('restore',{'restore_minutes':10,'restore_test_verified':False},10,'test_restore'),
         ('restore',{'restore_minutes':11,'restore_test_verified':True},10,'improve_recovery'),
         ('vendor',{'assurance_age_days':11,'mandatory_control_met':False},10,'reject'),
         ('vendor',{'assurance_age_days':10,'mandatory_control_met':True},10,'accept'),
         ('experiment',{'lift_lower_bound_basis_points':11,'assignment_valid':False,'guardrail_passed':False},10,'fix_measurement'),
         ('experiment',{'lift_lower_bound_basis_points':10,'assignment_valid':True,'guardrail_passed':True},10,'launch'),
         ('pilot',{'cost_usd':10,'reversible':True},10,'pilot'),
         ('pilot',{'cost_usd':10,'reversible':False},10,'escalate'),
         ('build_buy',{'buy_cost_usd':10,'product_meets_requirements':True,'internal_capacity_verified':True},10,'buy'),
         ('build_buy',{'buy_cost_usd':11,'product_meets_requirements':True,'internal_capacity_verified':False},10,'defer'),
         ('supplier',{'concentration_percent':11,'alternative_qualified':False},10,'qualify_alternative'),
         ('supplier',{'concentration_percent':10,'alternative_qualified':False},10,'retain'),
         ('budget',{'total_cost_usd':11,'cost_inventory_complete':False},10,'complete_inventory'),
         ('budget',{'total_cost_usd':10,'cost_inventory_complete':True},10,'approve'),
         ('communications',{'minutes_since_update':10,'scope_verified':True,'cause_claim_supported':True},10,True),
         ('cost_evidence',{'unallocated_cost_usd':10,'periods_aligned':True,'indirect_cost_included':False},10,False),
         ('security_priority',{'affected_assets':10,'active_exploitation_verified':False,'internet_exposed':True},10,2),
         ('security_priority',{'affected_assets':9,'active_exploitation_verified':True,'internet_exposed':False},10,3),
         ('recovery_priority',{'outage_minutes':9,'critical_dependency':True,'workaround_verified':False},10,3),
         ('recovery_priority',{'outage_minutes':9,'critical_dependency':False,'workaround_verified':True},10,0),
        ]
        for family,facts,threshold,expected in examples:
            with self.subTest(family=family,facts=facts):self.assertEqual(oracle(catalog()[family],facts,threshold)[0],expected)

    def test_unknown_is_not_false_and_conflict_precedes_other_rules(self):
        spec=catalog()['release']
        self.assertEqual(oracle(spec,{'budget_remaining_minutes':0,'rollback_tested':None},10)[0],'needs_info')
        self.assertEqual(oracle(spec,{'budget_remaining_minutes':0,'rollback_tested':False},10,True)[0],'needs_info')
        with self.assertRaises(ValueError):oracle(catalog()['security_priority'],{},10)

    def test_contract_and_provenance_all_rows(self):
        report=validate(self.data,self.sources)
        self.assertEqual((report['rows'],report['groups']),(2304,576))
        for rows in self.data.values():
            for r in rows:questions({'decision':r['question']})

    def test_counterfactuals_and_injection_variants_stay_together(self):
        for rows in self.data.values():
            groups={}
            for r in rows:groups.setdefault(r['group'],{})[r['audit']['variant']]=r
            for g in groups.values():
                self.assertEqual(set(g),{'base','irrelevant','boundary','uncertain'})
                self.assertEqual(g['base']['label'],g['irrelevant']['label'])
                self.assertNotEqual(g['base']['state']['evidence'],g['boundary']['state']['evidence'])

    def test_reject_cross_split_policy_leakage_even_with_new_ids(self):
        bad=deepcopy(self.data);row=deepcopy(bad['train'][0]);row.update(id='different',group='different',split='test');bad['test'].append(row)
        with self.assertRaisesRegex(ValueError,'leakage'):validate(bad,self.sources)

    def test_missing_label_coverage_rejected(self):
        bad=deepcopy(self.data)
        bad['validation']=[r for r in bad['validation'] if not (r['family']=='it_business_communications' and r['label'] is True)]
        with self.assertRaisesRegex(ValueError,'label coverage'):validate(bad,self.sources)

    def test_deterministic_mutation_controls(self):
        selected=select(self.data['train'],2)
        for defect,row in controls(selected):
            with self.subTest(defect=defect,id=row['id']):self.assertTrue(validate_row(row,self.sources))

    def test_no_gold_sent_and_errors_quarantined(self):
        row=self.data['train'][0];seen=[]
        def failed(state,qs):
            seen.append((state,qs));raise RuntimeError('simulated transport failure')
        result=run([row],self.sources,failed)
        self.assertFalse(result[0]['pass']);self.assertEqual(len(seen),1)
        self.assertNotIn('label',seen[0][0]);self.assertNotIn('audit',seen[0][0])
        bad=deepcopy(row);bad['label']='bogus';seen.clear()
        self.assertFalse(run([bad],self.sources,failed)[0]['pass']);self.assertFalse(seen)

    def test_hf_export_preserves_typed_labels(self):
        for rows in self.data.values():
            for row in rows:self.assertEqual(restore(flatten(row)),row)

    def test_handwritten_challenge_contract_and_sources(self):
        rows=[json.loads(line) for line in (RESEARCH/'challenge.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),24)
        self.assertEqual(len({r['id'] for r in rows}),24)
        for row in rows:
            questions({'decision':row['question']})
            self.assertEqual(row['split'],'challenge')
            self.assertTrue(all(self.sources[s]['eligible_as_inspiration'] for s in row['source_ids']))
            self.assertFalse(row['human_reviewed'])

    def test_invalid_judge_distribution_rejected(self):
        row=next(r for r in self.data['train'] if r['question']['type']=='choice')
        response={'answers':{'decision':{'type':'choice','probabilities':{k:0.9 for k in row['question']['criteria']}},'quality':{'type':'noul','noul':1.0}}}
        with self.assertRaises(ValueError):interpret(row,response)
        with self.assertRaisesRegex(ValueError,'training split'):select(self.data['test'],2)

if __name__=='__main__':unittest.main()
