# Overnight progress — September 18, 2026

Automated analysis of saved experiment evidence. Personal project by Adam Pippert.
Scheduled cutoff: 05:00 America/Los_Angeles. This report does not deploy or publish adapter weights.

Run status: **complete**. Last recorded phase: `complete`.
Completed optimizer updates: **1482**. Trial progress: `{"lr1e-05-seed17": 400, "lr1e-05-seed29": 400, "lr3e-05-seed17": 400, "lr3e-05-seed29": 282}`.

## Selection evidence

The fixed objective is the equal-family mean of candidate cross entropy divided by log(option count). Selection used validation only, with a guard against more than a 3 percentage point accuracy regression in any family. Calibration and test data did not select checkpoints.

Selected candidate: `lr3e-05-seed29`, step 282. Validation objective: 3.556362 → 0.352832 (decrease 3.203530).

| Family | Baseline validation accuracy | Selected validation accuracy |
|---|---:|---:|
| banking77 | 41.56% | 58.44% |
| boolq | 85.16% | 87.11% |
| synthetic_score | 52.34% | 93.75% |

## Held-out test evidence

| Family | n | Baseline accuracy | Selected accuracy | Baseline calibrated NLL | Selected calibrated NLL |
|---|---:|---:|---:|---:|---:|
| banking77 | 154 | 42.86% | 64.94% | 2.8535 | 1.8766 |
| boolq | 256 | 82.03% | 84.77% | 0.4028 | 0.3324 |
| synthetic_score | 256 | 63.28% | 95.31% | 1.0612 | 0.1491 |

Temperatures were fitted on the separate calibration split. Full Brier, calibration-error, and confidence-threshold coverage measurements are in `evidence/report.json`. These measurements are descriptive; this small experiment does not establish statistical significance or general Jev equivalence.

## Independent Jev comparison

Jev evaluated a fixed 60-example test subset (20 per family). These outputs were never training labels or checkpoint-selection inputs. Invalid probability responses are rejected, not renormalized; paired accuracy includes only valid responses, so exclusions may bias the comparison. Rejection counts are in `jev-evaluation/summary.json`. This small sample is descriptive.

| Family | n | Jev accuracy | Untuned Granite accuracy | Selected Granite accuracy |
|---|---:|---:|---:|---:|
| banking77 | 18 | 88.89% | 72.22% | 77.78% |
| boolq | 20 | 80.00% | 70.00% | 70.00% |
| synthetic_score | 20 | 100.00% | 50.00% | 95.00% |

## Scope and provenance

7,164 training records: 3,068 BANKING77, 2,048 BoolQ, and 2,048 original Score examples. Validation, calibration, and test each contain 666 records. The base model revision is pinned. Public benchmark exposure during the base model’s pretraining is unknown.

- [BANKING77](https://huggingface.co/datasets/PolyAI/banking77): CC-BY-4.0; Casanueva et al. (2020).
- [BoolQ](https://huggingface.co/datasets/google/boolq): CC-BY-SA-3.0; Clark et al. (2019).
- [Original Score dataset](https://huggingface.co/datasets/adampippert/granite-decisions-synthetic): MIT, `score_interval_v1` configuration.

Source URLs, immutable revisions, content hashes, transformations, and split policy are in `data-manifest.json`. External dataset content is not republished here or relicensed as MIT. Fullcollar remains deferred.

## Next decision

Inspect per-family test regressions and calibration before promoting an adapter. Expand the independent evaluation to user-relevant decision schemas and adversarial inputs. Further training should be a new experiment with a fresh selection plan; do not repeatedly select against this test split.
