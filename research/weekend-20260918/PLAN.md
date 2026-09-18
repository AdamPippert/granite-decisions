# Weekend IT and business decision data plan

Prepared September 18, 2026 for Adam Pippert's personal Granite Decisions project.
Training window: **Friday September 18, 20:00 PDT – Sunday September 20, 17:00 PDT** (45 hours; UTC September 19 03:00 – September 21 00:00). This package prepares data and tests; it does not start or schedule GPU work.

## Deliverable and limits

The initial research ledger contains 27 sources/leads across regulators, incident postmortems, research, trade publications, news, leadership writing, autobiographies and X. Each entry records its URL, inspected section, access status, historical date where known, event grouping, original factual capsule and reuse boundary. **A lead is not an inspected source.** Three X posts were inaccessible (403/empty response); none supplies training text or labels. Modern articles and books are reference-only. Even Google's accessible SRE book carries CC BY-NC-ND terms; accessibility is not an open training license.

The accompanying **2,304-row original MIT synthetic pilot** covers 12 families, with 576 independent policy instances and four variants per instance. It contains 1,440 train, 288 validation, 288 calibration and 288 test rows. Original fictional rules and numbers determine labels. Sources inspire the decision structure; they do not supply an allegedly optimal historical action or license the whole source under MIT. There are also **24 manually authored challenge cases**, not independently human-reviewed and not used by the curation judge.

This is an executable pilot and an expansion plan, not a claim of extremely high real-world quality. Structured fields, short ordered rules and repeated templates make it easier than real documents. Thresholds and scenario groups are disjoint across splits, but rule templates and inspiration sources are shared. Six independent groups per family in each evaluation split are too few for strong performance claims. X coverage, international/non-English cases and modern leadership diversity remain gaps.

## Coverage and source rationale

| Decision family | Sources in ledger | What the examples test |
| --- | --- | --- |
| Release authorization | Google SRE; CrowdStrike RCA announcement | Budget boundary, rollback readiness, explicit exception in challenge |
| Recovery readiness | GitLab postmortem; NIST contingency publication | Verified restore, recovery time, missing evidence; RPO challenge |
| Vendor acceptance | FCA TSB findings; Computer Weekly | Required control, stale assurance, conflicting reports |
| Product experiments | Microsoft experimentation research | Assignment validity before effect, guardrails, threshold; confidence-interval challenge |
| Delegated pilots | Bezos shareholder letter | Reversibility, spending authority, escalation |
| Build/buy exploration | GAO Agile; Carnegie accounting narrative | Requirements, complete cost, capability; explicit fictional preference |
| Supplier concentration | Cloudflare; FCA; Sky News JLR | Dependency exposure, qualified alternatives, indirect supplier uncertainty |
| Budget approval | SBA; Rockefeller autobiography | Complete costs, cap, shared-cost and unit-conversion challenges |
| Incident communications | Cloudflare; DOT Southwest | Verified scope/cause and timely update; unknown cause is not an attack |
| Cost evidence | Carnegie; Rockefeller; SBA | Period alignment, hidden costs, completeness |
| Security priority | GAO Equifax; FIRST EPSS | Exploitation, exposure, affected assets under a stipulated rubric |
| Recovery priority | GitLab; Cloudflare | Critical dependencies, duration, verified workarounds |

The autobiographies are retrospective self-reports with survivor and author biases. They suggest questions about accounting, evidence and reversibility; they never define universal management truths. The security rubric is neither SSVC nor CVSS. EPSS is exploitation likelihood over 30 days, not severity or a complete business risk score. Historical regulator findings do not establish current law for another organization. All this context stays in metadata rather than leaking company names or outcomes into labels.

## Business-hours expansion sequence

1. **Finish evidence cards before expanding templates.** Turn 20–30 accessible incident/research/government sources into short original cards: contemporaneous information, unavailable information, constraints, competing objectives, disputed claims, outcome kept separate, URL/section/date and rights. Group syndicated stories and multiple reports of one incident under one event. Verify exact dates before temporal holdouts. Add procurement, capacity planning, service desk escalation, data quality, cash runway, inventory and change governance; do not extrapolate from famous leaders alone.
2. **Build a broader policy library.** Target 24–32 independently specified decision families and at least 40 distinct policy structures—not thousands of cosmetic versions of the same threshold. Include opposing objectives and equally plausible actions whose ranking changes only with the stated policy. Require needs_info/abstain when objectives or decisive facts are missing. Preserve explicit exceptions and precedence. Every numeric task specifies units, periods and inclusive/exclusive boundaries.
3. **Generate 8,000–12,000 candidate rows, quality permitting.** Aim for 2,000+ scenario roots, with two to four linked variants rather than treating variants as independent observations. Rough allocation: 45% IT operations/security/governance, 35% business/product/procurement, 20% cross-domain ambiguity and evidence quality. Use mixed prose/tables/event timelines, option reordering/renaming, negation, irrelevant context, stale evidence, contradiction, realistic failure paths and controlled prompt injection. Existing pilot rows may be retained only after passing the final gate. Volume is a ceiling, not a success criterion.
4. **Use executable references and independent adjudication.** For each policy, write a decision table or arithmetic verifier, then independently authored expected cases. At least 10% of candidate roots and all disagreements, severe-risk actions, novel templates and source-rights ambiguities need independent domain review before production-quality designation. Without that review, retain the explicit experimental label; do not manufacture a human signoff. Publish reviewer rubric and unresolved disagreements without private identity data.
5. **Split before judging or training.** Allocate approximately 70/10/10/10 by scenario/event/policy structure to train/validation/calibration/test. Keep every counterfactual, paraphrase and shared event together. Reserve genuinely new policy structures and at least six source events for a separate 300+ case challenge. Near-duplicate search and semantic audit complement exact hashes. Only training candidates go through Jev curation. Do not silently remove final-test examples that Jev gets wrong.
6. **Freeze by 19:30 PDT.** Publish source ledger, original dataset, generators, split manifest, checksums, quality report and version. Sources without suitable reuse rights remain reference-only links and short original factual notes. No copyrighted article, biography chapter or X post is relabeled as MIT. Freeze the dataset only if the deterministic gate has zero failures, every class is covered in each split, leakage checks pass, and pending human-review limitations are visible.

## Pass/fail gate and tests

`corpus.py` validates exact policy/question construction, field types, provenance eligibility, expected labels, oracle traces, label coverage and group/policy/semantic split separation. Corrupted source IDs, target leakage, missing policies, wrong labels and bad Boolean types fail closed.

`quality_gate.py` sends only model-visible state and questions to Jev; it withholds the target, oracle trace, source name and split. A candidate passes this advisory filter only if its predicted label matches the independent rule label, the top probability is at least 0.55, and its separately requested unambiguity probability is at least 0.8. Ties, malformed distributions and transport errors quarantine the record. No label is replaced with Jev's answer. These predeclared thresholds are initial operating points, not empirically calibrated guarantees.

The pilot tests include 22 literal policy acceptance cases across all families, unknown/conflict precedence, option ordering and irrelevant-instruction invariance by construction, cross-split leakage, every class in every split, source/label/policy corruption, blinded payloads and invalid Jev responses. The 24 separate challenge cases include RTO versus RPO, quarterly versus monthly cost, missing restore evidence, stale vendor evidence, unresolved SRM, confidence intervals, reversible versus irreversible spending, shared costs, uncertain policy announcements and hindsight bias. **Construction invariance is not yet demonstrated Granite model invariance.** Run all model variants on the same cases to measure it.

For the expanded corpus, calibrate the *semantic* judge with at least 100 independently adjudicated clean/ambiguous/contradictory cases; report false accept and false reject rates by defect. Current mutation controls exercise the deterministic validator and cannot estimate Jev's semantic false-accept rate. Keep curation decisions separate from final Jev-versus-Granite evaluation so agreement does not become a circular definition of quality.

## Weekend training and objective

| PDT window | Work and stopping condition |
| --- | --- |
| Fri 20:00–21:00 | Verify corpus hashes, ROCm health, disk/checkpoints, tokenizer limits and checkpoint reload; benchmark current published starter and best previous adapter on the frozen validation set. Smoke one update with loss/gradient and save/reload parity. |
| Fri 21:00–Sat 09:00 | Small pilot ablations: previous adapter vs base initialization; old/new data mixtures; conservative learning rates 1e-5 and 3e-5. Keep seeds/data order fixed for paired comparisons. Measure steps/sec before committing remaining budget. |
| Sat 09:00–21:00 | Train the best eligible configuration on the larger accepted corpus; evaluate fixed validation every 100 updates or 30 minutes. Early-stop after three checks without meaningful gain. No indefinite synthetic regeneration loop. |
| Sat 21:00–Sun 12:00 | Replicate winner with a second seed; inspect family failures. Any revised data becomes a separately versioned experiment with a new untouched test, not a patch trained against the old test. |
| Sun 12:00–15:00 | Stop exploratory runs, preserve checkpoints, calibrate on calibration only, compare base/previous/winner/Jev on the same untouched test and challenge. Bootstrap uncertainty by scenario/event group, not row. |
| Sun 15:00–17:00 | Analyze results and regression cases, update dataset/model cards, publish evidence and the best eligible artifact under personal adampippert. Hard-stop all training no later than 17:00; do not extend into the evening. |

Primary training loss: **mean across task families of categorical cross-entropy divided by log(number of labels)**. Noul has two labels; Choice and Score use their finite categorical domains. Sample a family uniformly, then a root uniformly, then a variant to avoid overcounting roots with many variants. Begin a paired ablation with 50% prior-task replay / 50% new roots versus 25% / 75%; preserve BANKING77, BoolQ and prior Score capabilities. These are proposed configurations, not newly implemented trainer options.

Select on validation proper loss, subject to guardrails: no existing-family accuracy drop above 3 percentage points, no increase in predefined severe false approvals, valid finite outputs, and at least baseline performance on needs_info detection. Report macro accuracy/F1, per-family confusion, calibration NLL/Brier, selective risk versus coverage, counterfactual consistency, option-order invariance and latency. Do not optimize directly for Jev agreement. If validation supports no gain, retain the previous adapter and publish the negative result. Retain optimizer, RNG, source/data hashes and base revision with each checkpoint. Sunday final test is evaluation, not another selection loop.

The prior run averaged roughly 10 seconds/update; use a fresh measured estimate because longer IT/business records can change throughput substantially. Reserve evaluation time and bound maximum updates accordingly. A hard wall-clock deadline must apply to the entire process group with graceful checkpointing before termination. No unattended training launcher is installed by this planning task.

## Reproduce

From repository root:

```sh
python scripts/weekend/corpus.py --out /tmp/granite-it-business-pilot
python -m unittest discover -s tests -v
python scripts/weekend/quality_gate.py \
  --data /tmp/granite-it-business-pilot/train.jsonl \
  --out /tmp/granite-quality-controls
# Optional bounded live judge: add --live --key-file /path/to/private/key
# Default limit is 24 requests; no secrets are stored in outputs.
```

The package uses the existing Jev API client and repository contracts. Full CPU tests require the repository's test/serve dependencies. `pilot-v2/manifest.json` binds the four splits and source ledger. `challenge.jsonl` is a separate handwritten fixture; it is deliberately not generated by the policy engine. Publish both the original generator and the synthetic rows. Preserve source/model licenses separately. Fullcollar is excluded from this work.
