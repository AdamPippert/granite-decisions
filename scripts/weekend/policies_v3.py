"""Original fictional policies and sampling domains; source facts are inspiration only.

Rules are ordered, side-effect-free expressions in a small validated language.
Each pair changes a substantive preference, exception or evidence requirement.
"""
from copy import deepcopy

B=[False,True]
N=[0,1,2,5,10,20,30,50,80,100,150,200,300]
P=[1,2,3,5,10,20,40,80,120]


def policy(name,sources,fields,rules,default,kind='choice',objective='Apply the stated ordered decision policy.'):
    return dict(name=name,sources=sources,fields=fields,rules=rules,default=default,kind=kind,objective=objective)


def catalog():
    c={}
    def add(name,sources,fields,rules_a,rules_b,default,kind='choice',objectives=None):
        objectives=objectives or ('Standard policy','Alternative organizational policy')
        c[name]=[policy(name,sources,fields,rules_a,default,kind,objectives[0]),policy(name,sources,fields,rules_b,default,kind,objectives[1])]
    add('release',['google-budget','crowdstrike-rca'],{'budget_minutes':N,'rollback_tested':B,'security_fix':B,'commander_approved':B},
        [('not rollback_tested','hold'),('budget_minutes <= 0','hold')],
        [('not rollback_tested','hold'),('security_fix and commander_approved','release'),('budget_minutes <= 0','hold')], 'release',objectives=('No exception to the exhausted-budget freeze.','Allow an approved security exception only with tested rollback.'))
    add('restore',['gitlab-restore','nist-recovery'],{'restore_minutes':N,'rto_minutes':P,'snapshot_age_minutes':N,'rpo_minutes':P,'drill_verified':B,'critical_service':B},
        [('not drill_verified','test_restore'),('restore_minutes > rto_minutes or snapshot_age_minutes > rpo_minutes','improve_recovery')],
        [('not drill_verified','test_restore'),('snapshot_age_minutes > rpo_minutes','improve_recovery'),('critical_service and restore_minutes * 2 > rto_minutes','improve_recovery'),('restore_minutes > rto_minutes','improve_recovery')], 'ready',objectives=('Meet both recovery objectives.','Critical services require twice the recovery-time margin.'))
    add('vendor',['fca-tsb','cw-tsb'],{'mandatory_control_passed':B,'certificate_age_days':N,'maximum_age_days':P,'waiver_approved':B,'waiver_days':P},
        [('not mandatory_control_passed','reject'),('certificate_age_days > maximum_age_days','refresh_evidence')],
        [('not mandatory_control_passed and not waiver_approved','reject'),('not mandatory_control_passed and waiver_days > 5','reject'),('certificate_age_days > maximum_age_days','refresh_evidence')], 'accept',objectives=('No control waivers.','Permit an approved control waiver lasting at most five days.'))
    add('experiment',['ms-srm','ms-preexperiment'],{'assignment_valid':B,'guardrail_passed':B,'effect_lower_basis_points':[-20,-1,0,1,5,10,30],'sample_size':N,'minimum_sample':P},
        [('not assignment_valid','fix_measurement'),('not guardrail_passed','stop'),('effect_lower_basis_points > 0','launch')],
        [('not assignment_valid','fix_measurement'),('not guardrail_passed','stop'),('effect_lower_basis_points > 0 and sample_size >= minimum_sample','launch')], 'continue',objectives=('Use a predeclared valid sequential test.','Use a fixed-sample test requiring the declared minimum sample.'))
    add('pilot',['bezos-2016'],{'cost_usd':N,'delegated_limit_usd':P,'reversible':B,'executive_approved':B,'production_data':B},
        [('not reversible or production_data','escalate'),('cost_usd > delegated_limit_usd','escalate')],
        [('production_data','escalate'),('not reversible and not executive_approved','escalate'),('cost_usd > delegated_limit_usd','escalate')], 'pilot',objectives=('Delegation requires reversibility.','An executive may authorize irreversibility, but cannot override the data or cost limits.'))
    add('build_buy',['gao-agile','carnegie'],{'buy_meets_requirements':B,'internal_capacity':B,'buy_cost_usd':N,'build_cost_usd':N,'strategic_differentiation':B},
        [('buy_meets_requirements and (not internal_capacity or buy_cost_usd <= build_cost_usd)','buy'),('internal_capacity','build_pilot')],
        [('strategic_differentiation and internal_capacity','build_pilot'),('buy_meets_requirements and (not internal_capacity or buy_cost_usd <= build_cost_usd)','buy'),('internal_capacity','build_pilot')], 'defer',objectives=('Prefer the cheaper feasible option; ties favor buying.','Prefer a feasible internal pilot for strategic differentiation, then cost.'))
    add('supplier',['cloudflare-kv','sky-jlr'],{'concentration_percent':[0,20,40,60,80,100],'maximum_concentration_percent':[20,40,60,80],'alternative_qualified':B,'critical_dependency':B},
        [('concentration_percent > maximum_concentration_percent and alternative_qualified','diversify'),('concentration_percent > maximum_concentration_percent','qualify_alternative')],
        [('(critical_dependency or concentration_percent > maximum_concentration_percent) and alternative_qualified','diversify'),('critical_dependency or concentration_percent > maximum_concentration_percent','qualify_alternative')], 'retain',objectives=('Act on concentration threshold.','Also require redundancy for every critical dependency.'))
    add('budget',['sba-costs','rockefeller'],{'direct_cost_usd':N,'shared_cost_usd':N,'budget_usd':P,'benefit_usd':N,'inventory_complete':B},
        [('not inventory_complete','complete_inventory'),('direct_cost_usd + shared_cost_usd > budget_usd','defer'),('benefit_usd <= direct_cost_usd + shared_cost_usd','defer')],
        [('not inventory_complete','complete_inventory'),('direct_cost_usd + shared_cost_usd > budget_usd','defer'),('benefit_usd < 2 * (direct_cost_usd + shared_cost_usd)','defer')], 'approve',objectives=('Require positive net benefit within budget.','Require benefit at least twice complete cost, within budget.'))
    add('communications',['cloudflare-kv','dot-southwest'],{'scope_verified':B,'cause_claimed':B,'cause_verified':B,'minutes_since_update':N,'update_limit_minutes':P,'next_update_announced':B},
        [('scope_verified and (not cause_claimed or cause_verified) and minutes_since_update <= update_limit_minutes',True)],
        [('scope_verified and (not cause_claimed or cause_verified) and minutes_since_update <= update_limit_minutes and next_update_announced',True)],False,'noul',('Scope and any stated cause must be supported; update must be timely.','Also require a next-update commitment.'))
    add('cost_evidence',['carnegie','rockefeller','sba-costs'],{'periods_aligned':B,'indirect_included':B,'unallocated_usd':N,'tolerance_usd':P,'ledger_reconciled':B},
        [('periods_aligned and indirect_included and unallocated_usd <= tolerance_usd',True)],
        [('periods_aligned and indirect_included and unallocated_usd <= tolerance_usd and ledger_reconciled',True)],False,'noul',('Check aligned complete costs within tolerance.','Also require reconciliation to the ledger.'))
    add('security_priority',['gao-equifax','first-epss'],{'active_exploitation':B,'internet_exposed':B,'critical_asset':B,'affected_assets':N,'asset_threshold':P},
        [('active_exploitation',3),('internet_exposed and affected_assets >= asset_threshold',2),('internet_exposed',1)],
        [('active_exploitation or (critical_asset and internet_exposed)',3),('internet_exposed and affected_assets >= asset_threshold',2),('internet_exposed',1)],0,'score',('Verified exploitation has highest priority.','Critical exposed assets also have highest priority.'))
    add('recovery_priority',['gitlab-restore','cloudflare-kv'],{'critical_dependency':B,'workaround_verified':B,'outage_minutes':N,'duration_threshold_minutes':P,'customers_affected':N},
        [('critical_dependency and not workaround_verified',3),('outage_minutes >= duration_threshold_minutes',2),('not workaround_verified',1)],
        [('critical_dependency and not workaround_verified',3),('outage_minutes >= duration_threshold_minutes and customers_affected > 0',2),('not workaround_verified',1)],0,'score',('Duration triggers priority even for an internal incident.','The duration escalation requires affected customers.'))
    add('capacity',['cloudflare-kv','gao-agile'],{'peak_requests':N,'tested_capacity_requests':P,'approved_spend_usd':N,'expansion_cost_usd':P},
        [('tested_capacity_requests >= peak_requests','retain'),('expansion_cost_usd <= approved_spend_usd','expand')],
        [('tested_capacity_requests * 100 >= peak_requests * 125','retain'),('expansion_cost_usd <= approved_spend_usd','expand')], 'reduce_load',objectives=('Capacity need only cover measured peak.','Require 25 percent headroom above measured peak.'))
    add('procurement',['sba-costs','fca-tsb'],{'a_unit_usd':P,'b_unit_usd':P,'quantity':P,'a_shipping_usd':N,'b_shipping_usd':N,'a_days':P,'b_days':P,'deadline_days':P},
        [('a_days <= deadline_days and (b_days > deadline_days or a_unit_usd * quantity + a_shipping_usd <= b_unit_usd * quantity + b_shipping_usd)','choose_a'),('b_days <= deadline_days','choose_b')],
        [('a_days <= deadline_days and (b_days > deadline_days or a_days <= b_days)','choose_a'),('b_days <= deadline_days','choose_b')], 'defer',objectives=('Minimize total landed cost among on-time suppliers; ties favor A.','Minimize delivery days among on-time suppliers; ties favor A.'))
    add('inventory',['sba-costs','sky-jlr'],{'stock_units':N,'daily_demand_units':P,'lead_days':P,'safety_units':N,'order_funded':B},
        [('stock_units >= daily_demand_units * lead_days','retain'),('order_funded','reorder')],
        [('stock_units >= daily_demand_units * lead_days + safety_units','retain'),('order_funded','reorder')], 'escalate_funding',objectives=('Cover demand through supplier lead time.','Cover lead-time demand plus safety stock.'))
    add('runway',['sba-costs','rockefeller'],{'cash_usd':N,'monthly_burn_usd':P,'commitment_usd':N,'minimum_months':[1,2,3,6],'receivable_usd':N,'receivable_guaranteed':B},
        [('cash_usd - commitment_usd >= monthly_burn_usd * minimum_months','approve')],
        [('cash_usd - commitment_usd + receivable_usd >= monthly_burn_usd * minimum_months and receivable_guaranteed','approve'),('cash_usd - commitment_usd >= monthly_burn_usd * minimum_months','approve')], 'defer',objectives=('Use cash on hand only.','Count receivables only when guaranteed.'))
    add('break_even',['sba-costs','carnegie'],{'quarterly_fixed_usd':N,'unit_price_usd':P,'unit_variable_usd':P,'monthly_units':P},
        [('unit_price_usd <= unit_variable_usd','reject'),('(unit_price_usd - unit_variable_usd) * monthly_units * 3 >= quarterly_fixed_usd','expand')],
        [('unit_price_usd <= unit_variable_usd','reject'),('(unit_price_usd - unit_variable_usd) * monthly_units * 3 > quarterly_fixed_usd','expand')], 'defer',objectives=('Cover fixed cost; exactly three months per quarter; equality passes.','Require strictly positive profit; exactly three months per quarter.'))
    add('data_quality',['ms-srm','carnegie'],{'rows_count':P,'duplicate_rows':N,'allowed_duplicates':N,'schema_valid':B,'reconciled':B},
        [('not schema_valid','quarantine'),('duplicate_rows > allowed_duplicates','deduplicate')],
        [('not schema_valid or not reconciled','quarantine'),('duplicate_rows > allowed_duplicates','deduplicate')], 'use',objectives=('Check schema and duplicate allowance.','Also require reconciliation before use.'))
    add('project_priority',['gao-agile','bezos-2016'],{'a_benefit':N,'b_benefit':N,'a_effort_days':P,'b_effort_days':P,'a_feasible':B,'b_feasible':B},
        [('a_feasible and (not b_feasible or a_benefit * b_effort_days >= b_benefit * a_effort_days)','choose_a'),('b_feasible','choose_b')],
        [('a_feasible and (not b_feasible or a_benefit >= b_benefit)','choose_a'),('b_feasible','choose_b')], 'defer',objectives=('Maximize benefit per effort among feasible projects; ties favor A.','Maximize absolute benefit among feasible projects; ties favor A.'))
    add('change_scope',['fca-tsb','gao-agile'],{'requested_targets':N,'approved_targets':N,'rollback_verified':B,'emergency_authorized':B,'independent_review':B},
        [('requested_targets > approved_targets or not rollback_verified','reject'),('not independent_review','review')],
        [('requested_targets > approved_targets or not rollback_verified','reject'),('not independent_review and not emergency_authorized','review')], 'authorize',objectives=('Require independent review.','A documented emergency may waive review, never scope or rollback.'))
    add('incident_route',['gao-equifax','cloudflare-kv'],{'attack_verified':B,'vendor_failure_verified':B,'customer_impact':B,'reports_disagree':B},
        [('reports_disagree','investigate'),('attack_verified','security'),('vendor_failure_verified','reliability')],
        [('reports_disagree','investigate'),('attack_verified and customer_impact','security'),('vendor_failure_verified','reliability'),('attack_verified','security')], 'investigate',objectives=('Route verified attacks before vendor failures.','Without customer impact, verified vendor failure precedes attack routing.'))
    add('evidence_freshness',['fca-tsb','ms-preexperiment'],{'report_age_days':N,'max_age_days':P,'signed':B,'audit_complete':B,'contradictory_report':B},
        [('signed and report_age_days <= max_age_days and not contradictory_report',True)],
        [('signed and report_age_days <= max_age_days and not contradictory_report and audit_complete',True)],False,'noul',('Evidence must be fresh, signed and uncontradicted.','Also require a complete audit.'))
    add('business_priority',['sba-costs','gao-agile'],{'mandatory_deadline':B,'delay_cost_usd':N,'action_cost_usd':P,'reversible':B},
        [('mandatory_deadline',3),('delay_cost_usd > action_cost_usd',2),('reversible',1)],
        [('mandatory_deadline',3),('delay_cost_usd > 2 * action_cost_usd',2),('reversible',1)],0,'score',('Escalate when delay cost exceeds action cost.','Outside the mandatory-deadline rule, delay-cost escalation requires strictly MORE than twice the action cost; equality does not qualify.'))
    add('recovery_evidence',['nist-recovery','gitlab-restore'],{'drill_verified':B,'rto_passed':B,'rpo_passed':B,'offsite_copy_verified':B},
        [('drill_verified and rto_passed and rpo_passed',True)],
        [('drill_verified and rto_passed and rpo_passed and offsite_copy_verified',True)],False,'noul',('Verified drill must meet both time and data-loss objectives.','Also require a verified offsite copy.'))
    return c


def challenge_catalog():
    # These six inspiration events and entire policy families are challenge-only.
    return {
      'maintenance_blast_radius':[policy('maintenance_blast_radius',['aws-s3-2017'],{'online_servers':P,'remove_servers':N,'minimum_servers':P,'tool_dryrun_verified':B},[('not tool_dryrun_verified','verify'),('online_servers - remove_servers < minimum_servers','deny')],'allow')],
      'deletion_identity':[policy('deletion_identity',['atlassian-2022'],{'object_type_matches':B,'target_list_verified':B,'recoverable_delete':B,'authorization_valid':B},[('not authorization_valid or not object_type_matches','deny'),('not target_list_verified or not recoverable_delete','verify')],'allow')],
      'scale_signal':[policy('scale_signal',['slack-2021'],{'cpu_percent':[1,10,20,50,80],'waiting_threads':N,'thread_limit':P,'network_degraded':B},[('network_degraded and waiting_threads >= thread_limit','preserve_and_repair'),('cpu_percent < 20 and waiting_threads < thread_limit','downscale')],'retain')],
      'deployment_consistency':[policy('deployment_consistency',['knight-2012'],{'expected_nodes':P,'updated_nodes':N,'warning_count':N,'warnings_resolved':B},[('updated_nodes != expected_nodes','block'),('warning_count > 0 and not warnings_resolved','investigate')],'enable')],
      'failover_integrity':[policy('failover_integrity',['github-2018'],{'divergent_writes':B,'replica_verified':B,'lag_seconds':N,'maximum_lag_seconds':P},[('divergent_writes','reconcile'),('not replica_verified or lag_seconds > maximum_lag_seconds','wait')],'promote')],
      'interface_units':[policy('interface_units',['nasa-mco-1999'],{'raw_units':P,'numerator':P,'denominator':P,'expected_units':N,'conversion_verified':B},[('not conversion_verified','verify'),('raw_units * numerator != expected_units * denominator','reject')],'accept')],
    }
