# Granite Decisions IT/business corpus v3 — frozen for experimental training

Personal project of Adam Pippert. Prepared September 18, 2026 before the **08:00 America/Los_Angeles data deadline**. GPU training remains an evening task (20:00 Friday through 17:00 Sunday). These data scripts neither schedule nor start training.

## What is ready

**9,676 accepted original synthetic records, 2,419 scenario groups, 24 training families and 48 policy structures**, plus six challenge-only policy families within that total. Twenty additional rows in five training groups are quarantined and excluded. The 33-entry source ledger contains 26 accessible/inspected sources and seven unverified or blocked leads; only eligible entries supply topic inspiration. Three inaccessible X posts supply no training content. Source articles, book chapters and post text are not copied into the dataset.

| Split | Rows | Use |
| --- | ---: | --- |
| train | 5,740 | Optimization; 24 families |
| validation | 1,152 | Hyperparameters and checkpoint selection |
| calibration | 1,152 | Probability calibration after checkpoint selection |
| test | 1,152 | Final measurement only |
| challenge | 480 | Six new policy families/inspiration events; final measurement only |
| quarantined-train | 20 | Inspectable exclusions, **not training or test data** |

The earlier v2 release stays unchanged. V3 supersedes v2 for IT/business training: do not concatenate versions or mix their evaluation folds. This release is suitable for an **experimental fine-tuning run**, not a declaration of reliable autonomous business decision making. No independent human panel has adjudicated the examples and no new Granite performance result is claimed.

## Substantive expansion

The families are release authorization, restore readiness, vendor acceptance, experiments, delegated pilots, build/buy, supplier concentration, budgets, incident communications, cost evidence, security priority, recovery priority, capacity, procurement, inventory, cash runway, break-even, data quality, project priority, change scope, incident routing, evidence freshness, business priority and recovery evidence.

Each has two meaningfully different organizational policies, backed by a literal acceptance test in which **the same facts require different answers**. Examples: lowest landed cost versus earliest delivery; absolute benefit versus benefit per day; counting guaranteed receivables versus cash only; exact break-even versus strictly positive profit; security exceptions versus no release exceptions. Rules include arithmetic, ordered exceptions, multiple constraints, units, missing evidence, evidence age and conflict precedence. USD costs, budgets and benefits also vary together across four magnitude scales (1×, 10×, 100× and 1000×), with label-invariance tests. These are explicitly fictional policies, not claims that a historical leader or company used them or that they are universally optimal.

Each scenario has four linked variants: base, rendering/injection invariance, label-changing counterfactual, and uncertainty (or another complete-evidence contrast for Score). Evidence is rendered as prose, tables or bullet records. Approximately one quarter of base Choice roots use opaque option names; invariance variants rotate semantic meanings across opaque names so the correct numeric answer code can change. Unknown and stale information do not become false factual claims: Choice uses needs_info; Noul asks whether a proposition is *established*; Score records require complete evidence.

The 480 challenge rows use new sources and new policy families: [AWS maintenance blast radius](https://aws.amazon.com/message/41926/), [Atlassian deletion identifiers](https://www.atlassian.com/blog/how-we-build/post-incident-review-april-2022-outage), [Slack scaling signals](https://slack.engineering/slacks-outage-on-january-4th-2021/), [Knight Capital deployment controls](https://www.sec.gov/newsroom/press-releases/2013-222), [GitHub failover integrity](https://github.blog/news-insights/company-news/oct21-post-incident-analysis/) and [NASA interface units](https://science.nasa.gov/mission/mars-climate-orbiter/). No source prose or historical outcomes are model inputs. They inspire original fictional scenarios. The challenge shares the rendering engine and rule primitives with training, so it is not an independent real-world benchmark or a measure of pretraining contamination.

## Quality evidence and exclusions

Every record passes deterministic schema, type, provenance, exact rendered-input, symbolic-label and trace checks. Every rule branch and every label appears in each split for each applicable policy. All four variants stay together; policy instances and exact semantic examples do not cross splits. The six challenge inspiration events and policy structures are absent from all main splits. Main splits intentionally share policy structures and inspiration events.

A fixed, blinded **96-record training sample** went to Jev; no target, source name, oracle trace or split was sent. Its top prediction agreed on **95/96**. **91/96** passed the predeclared label-confidence ≥0.55 and clarity-confidence ≥0.8 gates. Three cases missed the label-confidence threshold, one missed the clarity threshold and one had a different top label. All five associated groups (20 rows) were quarantined, without relabeling. Their rule-derived labels remain intact for inspection. This conservative curation can remove difficult valid cases; it is not proof those cases are wrong. The other training rows are deterministically checked, not individually Jev-approved. Evaluation splits were never submitted to the curation judge or filtered against it.

The deterministic checker rejected **480/480 corrupted probes**. A separate semantic check used 40 authored ambiguous tasks and 40 explicitly repaired counterparts. At the fixed 0.8 clarity threshold, Jev accepted **0/40 ambiguous tasks** and rejected **4/40 valid repairs**, all involving a defined conflict-resolution outcome. These are controlled integration probes across ten defect patterns, not independent human-adjudicated real-world error rates. They justify keeping the judge advisory and preserving every exclusion.

The CPU tokenizer preflight used the actual inference prompt and pinned Granite revision `c0650403e44e78ec0262dab1c90914c65b196c4e`. All 9,696 candidates fit without truncation; maximum prompt length was **446 tokens**. The accepted SFT export also verifies the inference prompt as an exact token prefix and the intended numeric answer as the next token. No GPU was used.

Two issues caught during refinement are covered by regression tests: canonical JSON key sorting initially changed a rendered evidence string, and label-stratified sampling missed several exception branches in evaluation. Rendering is now order-stable, and generation stratifies on rule branches rather than only labels. Source incident dates are recorded separately from article publication/update dates. See `QUALITY_REPORT.json`, `data/manifest.json`, `tokenizer-check.json`, `sft-check.json` and `jev-audit/` for measured evidence.

## Training formats

`data/*.jsonl` uses the project's native per-record `state`, `question`, typed `label`, `family` and `group` schema. Only state and question enter a prompt; label is the target. **Never feed audit metadata, oracle traces, source IDs, split, group or a whole serialized row as model input.** The manifest lists exact file hashes.

`sft/*.jsonl` provides conversational `prompt`/`completion` pairs using the same numeric option codes as the inference backend. Feed only these two columns to a completion-only SFT trainer; the other columns support auditing and sampling. This format follows [TRL's documented conversational prompt-completion format](https://huggingface.co/docs/trl/sft_trainer#expected-dataset-type-and-format). Verify completion-only masking with the pinned local runtime before the evening launch. Standard SFT loss differs from the proposed family-balanced categorical loss; the export does not silently implement that objective.

Recommended native sampling: choose a family uniformly, then a scenario root uniformly, then a variant. Loss: categorical cross-entropy over valid options, divided by log(option count), averaged across families. Start with the planned 50% prior-task replay / 50% new-data ablation; retain existing source licenses and splits. Tune only on validation, fit calibration only afterward, and reserve both test and challenge for final reporting. Keep old/new per-family regression checks and severe false-approval counts alongside the mean loss.

**Integration boundary:** the historical overnight training script hardcodes BANKING77/BoolQ/Score and a six-example accumulation denominator. Do not point it at this corpus unchanged. The data are ready in native and standard SFT formats; the evening launcher still needs the appropriate all-family loader/sampler and its GPU smoke check. No existing trainer behavior was silently changed by this data task.

## Reproduce without a GPU

From the repository root, using new output directories:

```sh
python scripts/weekend/expanded_corpus.py \
  --sources research/weekend-20260918/expanded-v3/sources.json \
  --out /tmp/v3-candidates
python scripts/weekend/freeze_v3.py \
  --candidates /tmp/v3-candidates \
  --sources research/weekend-20260918/expanded-v3/sources.json \
  --audit research/weekend-20260918/expanded-v3/jev-audit \
  --tokens research/weekend-20260918/expanded-v3/tokenizer-check.json \
  --out /tmp/v3-frozen
python -m unittest discover -s tests -v
```

Freezing reuses hash-bound published judgments; it makes no API calls. `refine_v3.py` can perform a fresh explicitly bounded live audit with a private key file. `--resume` refuses mismatched record identities. `export_sft_v3.py` and `export_hf.py` reproduce the two sharing formats. Install repository test dependencies for the test suite; tokenizer checks additionally need the pinned Transformers version and Jinja2.

## License and limitations

Original code, policies, synthetic evidence and labels are MIT. Each linked source retains its own license; public accessibility is not permission to redistribute its prose as training data. IBM base weights remain under their separate Apache-2.0 license. No real customer records, personal X content, generated hidden reasoning or Jev-supplied training labels are included.

Limitations include English-only templated cases, simplified costs and operational constraints, shared rendering and rule vocabulary, small per-family evaluation groups, imperfect judge filtering and no independent domain panel. Report uncertainty clustered by scenario group. Success on these fixtures establishes neither real-world optimality nor Jev equivalence. Keep human review for consequential actions and evaluate on actual use-case policies before deployment.
