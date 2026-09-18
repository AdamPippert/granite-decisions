# Weekend v3 training on hyde

The native v3 trainer uses all 24 IT/business families and the frozen Granite numeric-decision prompt. It is staged separately from the historical overnight experiment. Original code and synthetic data are MIT; IBM base weights retain Apache-2.0. This is Adam Pippert's personal project.

## Schedule

All times are America/Los_Angeles (PDT), September 2026.

| Event | Time |
|---|---|
| Start | Friday September 18, 20:00 |
| Stop optimizer updates | Sunday September 20, 15:30 |
| Finish evaluation | Sunday September 20, 16:55 |
| External cancellation timer | Sunday September 20, 16:59:45 |
| Absolute process-group hard stop | Sunday September 20, 17:00 |

Superrouter's existing persistent user manager dispatches the job over SSH. The job runs detached inside hyde's existing `granite-decisions-rocm` toolbox. It keeps running if SSH disconnects. Both machines must be powered on at launch; hyde must remain available throughout training. Start is allowed only within 15 minutes of the scheduled time. There is no automatic retry, resume, overwrite, or weight publication. A reboot or container shutdown interrupts the job; saved evaluated checkpoints remain available.

The four fixed trials use learning rates 1e-5 and 3e-5 with seeds 17 and 29, up to 2,500 optimizer updates each. Each trial starts from the pinned untuned base and fresh LoRA parameters, not the previous overnight adapter. The pilot measured approximately 13.2 seconds per 24-example update; full-validation overhead and convergence determine how many trials fit. Three stale validation checks stop a trial early; the experiment may finish before Sunday if all trials converge. The schedule does not promise to consume every available GPU hour.

## Objective and heldout boundaries

Each update samples one example from every family, choosing a scenario group uniformly and then a variant uniformly. The loss is candidate-token cross-entropy divided by the logarithm of the candidate count, averaged over all 24 families. LoRA rank 8, alpha 16, q/v projections, BF16 base, gradient checkpointing and clipping preserve the tested ROCm configuration.

Checkpoints are evaluated every 75 updates and at each completed trial's last step. Selection minimizes macro family normalized validation NLL, provided no family's validation accuracy falls more than three percentage points below the untuned baseline. The untuned baseline remains selected if no checkpoint qualifies. Incomplete accumulation at the training cutoff is discarded; an interrupted run may lose the unevaluated updates since the last checkpoint.

Selection is written to `selection-frozen.json` before calibration, test or challenge inference. Temperatures are fitted only on calibration records. Baseline and selected checkpoints are evaluated on test and challenge records; challenge families receive no unrelated family calibration. The integration pilot exercises small samples of these paths after its own selection; it is not an effectiveness benchmark. The main run never uses pilot metrics for selection.

The v3 dataset contains 5,740 training records, 1,152 records in each of validation/calibration/test, and 480 challenge records. No older dataset is concatenated into this run. Main splits share policy structures; challenge shares generation primitives. These are synthetic policy-following evaluations, not independent real-world evidence of Jev equivalence or retention of the older banking/BoolQ tasks.

## Numerical validation

The first GPU pilot rejected batched BF16 inference: the largest probability difference from single-record scoring was approximately 0.1814. The staged trainer therefore uses single-record scoring for training, validation and final evaluation, matching the default runtime. A repeatability check across all families must pass within 1e-6. This avoids changing the tolerance to hide the discrepancy. Optimizing batched serving is separate work.

The isolated final pilot must complete forward/backward updates, adapter-change verification, saved-weight reload, calibration, test and challenge evaluation before the scheduler can start. Runtime code, dataset, configuration, pilot evidence and installed package versions are verified again at launch. Model loading is offline at launch using the already cached immutable base revision `c0650403e44e78ec0262dab1c90914c65b196c4e`.

## Operate from superrouter

Status:

```sh
bash /home/adam/.local/share/granite-decisions/weekend-20260918/dispatch.sh status
systemctl --user list-timers 'granite-weekend-v3-*'
journalctl --user -u granite-weekend-v3-start.service
```

Cancel before launch by disabling the start timer. Cancel an active run by creating its checked cancellation marker through the dispatch script:

```sh
systemctl --user disable --now granite-weekend-v3-start.timer
bash /home/adam/.local/share/granite-decisions/weekend-20260918/dispatch.sh stop
```

The supervisor polls the marker and terminates the trainer's process group, escalating to KILL after at most five seconds. The independent absolute deadline works without superrouter. Cancellation is persistent; do not remove the marker or restart without deliberately arranging a new run directory and schedule.

Hyde artifacts are under `/var/home/adam/Development/granite-decisions/experiments/weekend-20260918`:

- `supervisor.log`, `supervisor-status.json`, `launch.json`: launch/error/deadline evidence.
- `training.log`, `run/events.jsonl`, `run/status.json`: live training progress.
- `run/checkpoints`, `run/best.json`: evaluated adapters and current validation selection.
- `run/report.json`, `progress-summary.json`: final report or completed progress after a stop/error.
- `RUNTIME_SHA256SUMS`, `SHA256SUMS`, `environment.json`: runtime and dataset preflight.

Saved adapters load through the frozen `GeneralBackend(adapter=...)` API. Family-level calibration in the experiment report is analytical evidence, not a deployable runtime calibration artifact. Review evaluation and source/adapter manifests before publishing or deploying weights.

## Reproduce checks

From the repository root:

```sh
uv run --with numpy --with scipy --with fastapi --with httpx python -m unittest discover -s tests -v
systemd-analyze --user verify experiments/weekend-20260918/systemd/*.service experiments/weekend-20260918/systemd/*.timer
bash -n experiments/weekend-20260918/scripts/dispatch.sh
```

For another host, copy the runtime and frozen dataset, adjust the explicit host paths and schedule, generate inventories, and run the short `--pilot` before arming timers. Do not invoke the historical three-family trainer on v3.
