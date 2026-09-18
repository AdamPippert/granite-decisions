# Overnight experiment: 2026-09-17

Personal project; only adampippert HF namespace. Fullcollar deferred.
User authorized hyde GPU training until 05:00 America/Los_Angeles, 2026-09-18.
Training cutoff 04:30 Pacific (11:30 UTC); evaluation cutoff 04:55 Pacific; independent process hard stop at 05:00 Pacific. Analyze and push completed progress after the cutoff.

Objective: equal-weight mean of per-family candidate cross entropy divided by log(number of valid options), using balanced task sampling. One numeric candidate token per option, immutable Granite BF16 base, LoRA only. Compare against the untuned model under exactly the same new prompt and precision. This is supervised learning, not a claim to implement proprietary RLCD.

Selection: fixed four-trial grid (learning rates 1e-5/3e-5, seeds 17/29), up to 400 optimizer updates each, six samples/update, validation every 50 updates, three-evaluation patience. Select lower macro normalized validation NLL only if each family's validation accuracy stays within 3 percentage points of the untuned baseline. Retain baseline if no eligible improvement. Keep test labels out of optimization and selection; fit temperatures on calibration only after selection. Test once for baseline and selected candidate. These public benchmarks may have appeared in base-model pretraining.

Datasets: BANKING77 CC-BY-4.0, BoolQ CC-BY-SA-3.0, original interval Score fixtures MIT. Provenance includes URLs, upstream commits, content hashes, source row indices, and passage/rubric-group-disjoint split construction. Publish original synthetic data and generator; retain source-specific licenses, never label the whole training mixture MIT.

Acceptance: reproducible data; meaningful split and objective tests; GPU forward/backward; code reviewed with Jev; immutable baseline; saved best adapter/checkpoint metadata; graceful deadline and independent hard timeout; detached logs and status. Do not automatically publish or deploy weights based on training loss alone. Existing v0.1.0 release unchanged.

Data build succeeded: 7,164 training examples; 666 each validation/calibration/test. BoolQ original URL HTTP403; official google/boolq parquet mirror used, grouped by normalized passage because titles are absent. BANKING77 cap is up to40/class after duplicate removal (3,068 total). Six objective/data/prompt tests passed on hyde.

Operational record: initial pilot passed. Baseline-only run restarted for exact uncapped NLL; the next toolbox was externally terminated (exit143, OOMKilled=false), evidence preserved. Recovered service: granite-overnight-recovered-20260917.service. Eval batch3 passed GPU parity; partial predictions saved every99 records. Local timer granite-progress-20260918.timer runs reviewed publisher at05:02Pacific. 60 independent Jev requests completed, 58 valid and2 rejected; never used for selection. Final9 focused tests passed; final Jev reviews recorded in reviews/.
