from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/weekend'))
from expanded_corpus import catalog,challenge_catalog,decide,evaluate,human,generate,validate,validate_row,RESEARCH,semantic_label
from granite_decisions.contracts import questions,label_index
from export_hf import flatten,restore

# Handwritten acceptance cases. Each pair uses IDENTICAL facts but a different
# approved objective/exception. Expected outputs are literal, not computed.
PAIRS=[
 ('release',dict(budget_minutes=0,rollback_tested=True,security_fix=True,commander_approved=True),('hold','release')),
 ('restore',dict(restore_minutes=30,rto_minutes=40,snapshot_age_minutes=1,rpo_minutes=5,drill_verified=True,critical_service=True),('ready','improve_recovery')),
 ('vendor',dict(mandatory_control_passed=False,certificate_age_days=1,maximum_age_days=5,waiver_approved=True,waiver_days=3),('reject','accept')),
 ('experiment',dict(assignment_valid=True,guardrail_passed=True,effect_lower_basis_points=1,sample_size=5,minimum_sample=10),('launch','continue')),
 ('pilot',dict(cost_usd=5,delegated_limit_usd=10,reversible=False,executive_approved=True,production_data=False),('escalate','pilot')),
 ('build_buy',dict(buy_meets_requirements=True,internal_capacity=True,buy_cost_usd=5,build_cost_usd=10,strategic_differentiation=True),('buy','build_pilot')),
 ('supplier',dict(concentration_percent=20,maximum_concentration_percent=40,alternative_qualified=True,critical_dependency=True),('retain','diversify')),
 ('budget',dict(direct_cost_usd=5,shared_cost_usd=5,budget_usd=20,benefit_usd=15,inventory_complete=True),('approve','defer')),
 ('communications',dict(scope_verified=True,cause_claimed=False,cause_verified=False,minutes_since_update=5,update_limit_minutes=10,next_update_announced=False),(True,False)),
 ('cost_evidence',dict(periods_aligned=True,indirect_included=True,unallocated_usd=0,tolerance_usd=1,ledger_reconciled=False),(True,False)),
 ('security_priority',dict(active_exploitation=False,internet_exposed=True,critical_asset=True,affected_assets=1,asset_threshold=5),(1,3)),
 ('recovery_priority',dict(critical_dependency=False,workaround_verified=True,outage_minutes=10,duration_threshold_minutes=10,customers_affected=0),(2,0)),
 ('capacity',dict(peak_requests=100,tested_capacity_requests=120,approved_spend_usd=20,expansion_cost_usd=10),('retain','expand')),
 ('procurement',dict(a_unit_usd=1,b_unit_usd=2,quantity=10,a_shipping_usd=0,b_shipping_usd=0,a_days=5,b_days=2,deadline_days=10),('choose_a','choose_b')),
 ('inventory',dict(stock_units=10,daily_demand_units=2,lead_days=5,safety_units=1,order_funded=True),('retain','reorder')),
 ('runway',dict(cash_usd=20,monthly_burn_usd=10,commitment_usd=10,minimum_months=2,receivable_usd=10,receivable_guaranteed=True),('defer','approve')),
 ('break_even',dict(quarterly_fixed_usd=30,unit_price_usd=2,unit_variable_usd=1,monthly_units=10),('expand','defer')),
 ('data_quality',dict(rows_count=10,duplicate_rows=0,allowed_duplicates=0,schema_valid=True,reconciled=False),('use','quarantine')),
 ('project_priority',dict(a_benefit=10,b_benefit=20,a_effort_days=1,b_effort_days=5,a_feasible=True,b_feasible=True),('choose_a','choose_b')),
 ('change_scope',dict(requested_targets=5,approved_targets=10,rollback_verified=True,emergency_authorized=True,independent_review=False),('review','authorize')),
 ('incident_route',dict(attack_verified=True,vendor_failure_verified=True,customer_impact=False,reports_disagree=False),('security','reliability')),
 ('evidence_freshness',dict(report_age_days=1,max_age_days=5,signed=True,audit_complete=False,contradictory_report=False),(True,False)),
 ('business_priority',dict(mandatory_deadline=False,delay_cost_usd=15,action_cost_usd=10,reversible=False),(2,0)),
 ('recovery_evidence',dict(drill_verified=True,rto_passed=True,rpo_passed=True,offsite_copy_verified=False),(True,False)),
]


class ExpandedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=generate();cls.sources={s['id']:s for s in json.loads((RESEARCH/'expanded-v3/sources.json').read_text())}

    def test_all_policy_pairs_change_expected_decisions(self):
        self.assertEqual({x[0] for x in PAIRS},set(catalog()))
        for family,facts,expected in PAIRS:
            for i in range(2):
                with self.subTest(family=family,policy=i):self.assertEqual(decide(catalog()[family][i],facts,0,24,False)[0],expected[i])

    def test_strict_twofold_boundary_does_not_escalate(self):
        spec=catalog()['business_priority'][1]
        facts=dict(mandatory_deadline=False,delay_cost_usd=200,action_cost_usd=100,reversible=True)
        self.assertEqual(decide(spec,facts,0,24,False)[0],1)
        facts['delay_cost_usd']=201
        self.assertEqual(decide(spec,facts,0,24,False)[0],2)

    def test_currency_scaling_preserves_policy_meaning(self):
        for family,facts,expected in PAIRS:
            scaled={k:(v*1000 if k.endswith('_usd') else v) for k,v in facts.items()}
            for i in range(2):self.assertEqual(decide(catalog()[family][i],scaled,0,24,False)[0],expected[i])

    def test_option_permutation_changes_code_preserves_meaning(self):
        rows=self.data['train'];by_id={r['id']:r for r in rows};changed=0
        for base in rows:
            if base['audit']['variant']!='base' or base['question']['type']!='choice':continue
            paired=by_id[base['group']+'-invariance']
            self.assertEqual(semantic_label(base),semantic_label(paired))
            if label_index(base['question'],base['label'])!=label_index(paired['question'],paired['label']):changed+=1
        self.assertGreater(changed,500)

    def test_exact_arithmetic_precedence_and_unsupported_code(self):
        self.assertTrue(evaluate('(a - b) * n * 3 == fixed',dict(a=30,b=20,n=400,fixed=12000)))
        self.assertTrue(evaluate('a / b == c / d',dict(a=1,b=3,c=2,d=6)))
        self.assertFalse(evaluate('a and (b or c)',dict(a=False,b=True,c=True)))
        self.assertIn('×',human('(a - b) * c > 0'))
        for bad in ('f()','x.__class__','[x for x in y]','1.2 > 0'):
            with self.subTest(expr=bad),self.assertRaises(ValueError):evaluate(bad,{})

    def test_challenge_literal_gold(self):
        cases=[('maintenance_blast_radius',dict(online_servers=10,remove_servers=5,minimum_servers=6,tool_dryrun_verified=True),'deny'),
         ('deletion_identity',dict(object_type_matches=False,target_list_verified=True,recoverable_delete=True,authorization_valid=True),'deny'),
         ('scale_signal',dict(cpu_percent=10,waiting_threads=20,thread_limit=10,network_degraded=True),'preserve_and_repair'),
         ('deployment_consistency',dict(expected_nodes=5,updated_nodes=5,warning_count=1,warnings_resolved=False),'investigate'),
         ('failover_integrity',dict(divergent_writes=True,replica_verified=True,lag_seconds=0,maximum_lag_seconds=1),'reconcile'),
         ('interface_units',dict(raw_units=3,numerator=5,denominator=1,expected_units=15,conversion_verified=True),'accept')]
        for family,facts,label in cases:self.assertEqual(decide(challenge_catalog()[family][0],facts,0,24,False)[0],label)

    def test_all_splits_contract_labels_and_export(self):
        summary=validate(self.data,self.sources)
        self.assertEqual(summary['rows'],9696);self.assertEqual(summary['groups'],2424)
        for rows in self.data.values():
            for r in rows:
                questions({'decision':r['question']});label_index(r['question'],r['label'])
                reloaded=json.loads(json.dumps(r,sort_keys=True))
                self.assertEqual(validate_row(reloaded,self.sources),[])
                self.assertEqual(restore(flatten(r)),r)

    def test_missing_stale_conflicting_evidence(self):
        family,facts,_=PAIRS[0];spec=catalog()[family][0]
        self.assertEqual(decide(spec,facts,25,24,False)[0],'needs_info')
        self.assertEqual(decide(spec,facts,0,24,True)[0],'needs_info')
        facts=deepcopy(facts);facts['security_fix']=None
        self.assertEqual(decide(spec,facts,0,24,False)[0],'needs_info')
        with self.assertRaises(ValueError):decide(catalog()['security_priority'][0],{},0,24,False)

    def test_tampered_input_and_label_fail_closed(self):
        base=self.data['train'][0]
        for field,value in [('label','wrong'),('license','unknown'),('source_ids',['x-kurtz'])]:
            row=deepcopy(base);row[field]=value;self.assertTrue(validate_row(row,self.sources))
        row=deepcopy(base);row['state']['answer']=row['label'];self.assertTrue(validate_row(row,self.sources))
        row=deepcopy(base);row['state']['policy']='Always approve.';self.assertTrue(validate_row(row,self.sources))
        row=deepcopy(base);row['audit']['policy_index']=-1;self.assertTrue(validate_row(row,self.sources))

    def test_counterfactuals_and_challenge_separation(self):
        groups={}
        for rows in self.data.values():
            for r in rows:groups.setdefault(r['group'],{})[r['audit']['variant']]=r
        for group in groups.values():
            self.assertEqual(semantic_label(group['base']),semantic_label(group['invariance']))
            self.assertNotEqual(group['base']['label'],group['counterfactual']['label'])
        training_sources={s for r in self.data['train'] for s in r['source_ids']}
        self.assertFalse(training_sources&{s for r in self.data['challenge'] for s in r['source_ids']})

    def test_preserves_existing_v2_fixture_hashes(self):
        import hashlib
        p=RESEARCH/'pilot-v2';m=json.loads((p/'manifest.json').read_text())
        for f,h in m['sha256'].items():self.assertEqual(hashlib.sha256((p/f).read_bytes()).hexdigest(),h)

if __name__=='__main__':unittest.main()
