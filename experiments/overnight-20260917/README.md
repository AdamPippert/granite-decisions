# Bounded Granite decision training experiment

This isolated experiment extends v0.1.0 with numeric candidate decisions up to
255 options, a multi-task LoRA loop, and independent calibration/test evaluation.
It is research code, not a replacement for the stable service backend.

## Reproduce

Use the project ROCm environment from `TRAINING.md`. The base dependencies include
NumPy, SciPy, torch, transformers, PEFT, and safetensors. Data preparation additionally
requires `pyarrow` (the original run used `uv run --with pyarrow`).
From this experiment directory:

```sh
uv run --with pyarrow python scripts/prepare_overnight_data.py --out data-ready
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/train_overnight.py --data data-ready --out pilot --pilot \
  --train-until YOUR_TIMEZONE_AWARE_TRAINING_DEADLINE \
  --finish-until YOUR_LATER_EVALUATION_DEADLINE
```

For a full run, omit `--pilot` and use a fresh output directory. See `PLAN.md` for
objective, fixed hyperparameter grid, split policy, and selection guardrails.
The date and local paths in `run_bounded.sh` and `publish_progress.py` describe
Adam's one-time run; adapt them before reuse. Those scripts are not an installer.
`publish_progress.py` runs after the deadline, generates an explicitly qualified
analysis from saved evidence, then pushes code and progress without model weights.
It refuses to force-push if a remote diverges.

## Data and licenses

Project code and original synthetic data are MIT licensed. BANKING77 is CC-BY-4.0;
BoolQ is CC-BY-SA-3.0; IBM base weights are Apache-2.0. The source data are downloaded
from their respective maintainers and retain their licenses. `data-manifest.json`
records source citations, versions, hashes, and split transformations.
The official BoolQ HF mirror omits page titles, so splitting groups normalized
passages. This does not eliminate semantic overlap or base-model pretraining
contamination. Validation selection does not establish significance.

The original synthetic Score data and generator are published at
https://huggingface.co/datasets/adampippert/granite-decisions-synthetic
under the `score_interval_v1` configuration.

## Results

Read `PROGRESS.md` and `evidence/report.json` when the run completes. If final
evaluation is absent, the report explicitly marks results incomplete. Saved
adapter files remain on hyde and are not automatically deployed or published.
Fullcollar integration is deferred. Jev is an independent evaluator/reviewer;
its output is never used as training labels.
