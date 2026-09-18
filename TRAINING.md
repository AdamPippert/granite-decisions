# Training and evaluation on Hyde

The prototype runs on Hyde's AMD Radeon 8060S (gfx1151). Frozen-head training,
native ROCm inference, a live Jev comparison, and a three-step Granite LoRA
training/save/reload test were exercised on September 17, 2026. This is a working
research baseline, not a model that matches Jev's quality or reproduces RLCD.

## Architecture and adaptation choices

| Path | What learns | Runtime | Supported use |
| --- | --- | --- | --- |
| Untuned candidate classifier | Nothing | Patched llama.cpp ROCm, Q4_K_M | New questions; sequential one-token candidate scoring |
| Frozen feature heads | Linear weights and per-head temperature | Granite feature extraction on ROCm, NumPy/SciPy head fitting on CPU | Registered questions; one state encoding shared by all heads |
| Optional LoRA | Attention q/v adapters, 2,621,440 parameters in the tested configuration | ROCm PyTorch, BF16 Granite safetensors | Supervised question-conditioned adaptation; separate experimental artifacts |

The optional LoRA script minimizes cross-entropy over valid single-token option
codes. It uses the same prompt builder as the native baseline. This is supervised
learning with a proper scoring objective, not a reconstruction of TypeSafe's
proprietary reinforcement-learning method. Neither normalized logits nor this
loss alone establishes calibration. LoRA does not automatically replace the
GGUF, change the running service, or recalibrate frozen heads.

## Fedora Atomic setup

Run these commands on Hyde from `~/Development/granite-decisions`:

```bash
bash deploy/setup-toolbox.sh
```

The script creates the dedicated `granite-decisions-rocm` Toolbx when absent,
installs Fedora development packages inside it, and runs `deploy/run-hyde.sh`.
The host kernel and amdgpu driver provide GPU access. No host package layering or
reboot is required. This is the tested Fedora Atomic path; the original native
scripts also work in a suitably provisioned development environment.

Run project commands inside that Toolbx:

```bash
toolbox enter granite-decisions-rocm
cd ~/Development/granite-decisions
.venv/bin/python -m unittest discover -s tests -v
```

The tested native backend is llama.cpp revision
`aa39d7a3e145a88202793a89462d65e94a5fc25f`, with the included probability patch.
Fedora ROCm HIP is 7.1.1. The model is IBM's pinned 3B Q4_K_M GGUF, SHA-256
`662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29`.
The native smoke test requires positive offload evidence: **41/41 layers** were
observed in both inference modes. It now requests verbosity 4 because the pinned
backend routes model-load messages to trace level. Device enumeration alone does
not pass the test.

`runs/requirements-heads.lock.txt`, `runs/requirements-finetune.lock.txt`, and
`runs/rpm-versions.txt` record the installed environments. The Python snapshots
are version inventories, not portable binary or system-package locks. Re-run
GPU validation on other machines. This was not tested on RHEL or OpenShift.

## Repeat the train/calibrate/test exercise

Inside Toolbx, use new output directories for each run:

```bash
.venv/bin/python scripts/make_smoke_data.py --out runs/smoke-data-new
.venv/bin/python scripts/run_training_smoke.py \
  --server .native/build-rocm/bin/llama-server \
  --gguf models/granite-4.1-3b-Q4_K_M.gguf \
  --data runs/smoke-data-new --out runs/training-smoke-new
```

The runner launches temporary loopback servers, checks layer offload, extracts
features, trains heads, fits temperatures on a separate calibration split,
saves/reloads the bundle, and collects trained and untuned predictions. It stops
its own servers even on failure. Ports 19091 and 19092 must be free.

The original synthetic fixture has 54 training, 54 calibration, and 54 test
states, with Choice, Noul, and Score labels assigned by construction. Templates
are grouped by split; vocabulary and subject matter still overlap. The fixtures
are original MIT-licensed integration examples, not representative workload data.
For actual use cases, replace them with reviewed, provenance-grouped datasets in
the same JSONL format. Keep evaluation data outside training and model selection.
The original README documents the generic `train` and `evaluate` commands.

## Live Jev baseline

Jev is a comparator, not ground truth. The benchmark reports each model's accuracy,
NLL, multiclass Brier score, 10-bin ECE, threshold coverage, selective error, score
MAE, and pairwise agreement against separately supplied labels. Its coverage is
based on the common probability/margin thresholds, not the native engine's
whole-request policy. Untuned Granite still abstains by default in application use.

On the machine holding the Jev credential, run:

```bash
python3 scripts/benchmark_decisions.py jev \
  --questions data/questions.json --data data/test.jsonl \
  --key-file /path/to/jev-key --max-requests 60 --out jev.jsonl
```

This mode needs only Python's standard library. It sends the explicitly supplied
states and questions to `https://api.typesafe.ai/v1/systemone`; it does not send
labels. It uses `jev-latest`, records the returned model, timestamps, usage and
raw responses, and stops on errors without automatic retries. The alias is not
an immutable model revision. Credentials stay on their original host. Set
`TYPESAFE_API_KEY` instead of `--key-file` if desired.

Copy the result JSONL to the project and compare inside Toolbx:

```bash
.venv/bin/python scripts/benchmark_decisions.py compare \
  --questions runs/smoke-data/questions.json --data runs/smoke-data/test.jsonl \
  --results runs/jev.jsonl runs/training-smoke/untuned.jsonl runs/training-smoke/heads.jsonl \
  --dataset-kind synthetic --out runs/comparison-new.json
```

The comparison rejects incomplete, duplicate, or mismatched datasets/questions.
Do not substitute Jev's labels for verified outcomes when estimating accuracy.
The original run used exactly 54 Jev calls with three questions per call.

## Observed synthetic results

All accuracy columns below use 54 test states. Latency is per whole request with
three questions, measured at the client; Jev includes network time from a different
host. These are integration-run timings, not controlled throughput comparisons.

| Model/path | Route accuracy | Python Noul accuracy | Scope accuracy | Median request latency |
| --- | ---: | ---: | ---: | ---: |
| Jev | 100% | 100% | 100% | 134.7 ms |
| Untuned Granite Q4_K_M | 98.1% | 50.0% | 92.6% | 223.1 ms |
| Frozen Granite heads | 61.1% | 77.8% | 72.2% | 28.8 ms |

The faster shared-feature path lost substantial quality on this tiny training
set. The untuned Boolean result needs investigation on reviewed examples before
use. These findings do not establish that quantization caused the errors or that
more training alone will fix them. Full distributions and scoring metrics are in
`runs/comparison.json` and the shareable validation snapshot.

## Optional Granite LoRA environment

Inside Toolbx:

```bash
bash deploy/setup-finetune.sh
.venv-finetune/bin/python scripts/finetune_lora.py \
  --questions runs/smoke-data/questions.json \
  --train runs/smoke-data/train.jsonl \
  --validation runs/smoke-data/calibration.jsonl \
  --steps 3 --out runs/lora-smoke-new
```

The setup pins PyTorch 2.11.0+rocm7.2, Transformers 4.57.6, PEFT 0.18.1, and
Accelerate 1.12.0 in `.venv-finetune`, separate from the inference environment.
It verifies a GPU matrix operation and backward pass. The training script pins
IBM safetensors revision `c0650403e44e78ec0262dab1c90914c65b196c4e`, rejects
context overflow and overlapping state/ID splits, verifies nonzero finite adapter
gradients and changed weights, and reloads the saved safetensors adapter to check
its logits. Downloads require several GB of storage and network access.

The three-step smoke passed with about **6.48 GiB** peak allocated GPU memory.
Its six-pair validation NLL **worsened from 0.0459 to 3.7917**. Do not promote that
adapter. This proves environment functionality, not effective fine-tuning. The
learning rate, data volume, validation coverage, and training schedule require
use-case-specific tuning. The test split was not used for LoRA optimization.
PyTorch used its available SDPA implementation; experimental AMD flash/memory
attention was not enabled.

For a real experiment, supply reviewed data, increase `--steps`, select a learning
rate using validation only, and expand `--validation-limit` to cover all validation
state/question pairs. Retain a separate calibration split and untouched test set.
The script saves PEFT adapters for Transformers. GGUF merge/conversion, quantized
parity checks, and calibration of a tuned candidate classifier are future work;
there is no tested adapter-to-serving promotion pipeline in this prototype.

## Handoff to IBM and Red Hat

The useful deliverable is the typed-decision interface, reproducible baselines,
training/evaluation tools, and inspectable evidence. IBM/Red Hat can replace the
schema, labels, representation, training recipe, and deployment packaging for
their use cases. No external publication or endorsement is implied.

Before making model-quality claims, run representative licensed/reviewed datasets
across multiple seeds and held-out organizations/time periods, include ambiguous
and out-of-domain inputs, and compare reliability/coverage at equal error targets.
Keep model selection, temperature fitting, and final evaluation separate. Treat
independent question evaluation, generalization to new schemas, shared-prefix
batching, and a calibrated tuned-backbone serving path as explicit research work.

Our code is MIT; IBM weights retain Apache-2.0; llama.cpp retains MIT. The share
archive contains source and synthetic evidence, not credentials, downloaded
weights, environments, or private operational data. Local recovery is available
from `runs/source-before-training-setup.tar.gz`. No existing service was changed.

## Primary references

- [Jev API and typed questions](https://docs.typesafe.ai/api)
- [Jev confidence semantics](https://docs.typesafe.ai/confidence)
- [IBM Granite 4.1 3B](https://huggingface.co/ibm-granite/granite-4.1-3b)
- [PEFT LoRA quicktour](https://huggingface.co/docs/peft/main/en/quicktour)
- [Fedora ROCm HIP development package](https://packages.fedoraproject.org/pkgs/rocclr/rocm-hip-devel/)
- [Qwen reference supplied for inspiration](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)

The Qwen reference describes an MLX parallel constrained-decoding implementation.
Its ideas about per-field candidate scoring informed the comparison framing; no
code was copied, and its advertised calibration and speed do not transfer here.
