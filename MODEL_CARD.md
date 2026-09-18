# Granite Decisions v0.1.0

## What this release is

A local typed-decision runtime and reproducible training recipe around
[IBM Granite 4.1 3B](https://huggingface.co/ibm-granite/granite-4.1-3b).
The code supports Choice distributions, ordered Score rubrics, and Boolean Noul
probabilities. It downloads official IBM weights; no new pretrained model or
production-quality fine-tuned checkpoint is included.

The intended audience is developers and researchers adapting local decision
models to their own labeled use cases. Decisions remain advisory. The default
runtime abstains on uncalibrated output, low probability, ambiguity, and designated
unknown labels.

## Model and training

- Base: `ibm-granite/granite-4.1-3b`, Apache-2.0.
- Inference baseline: official Q4_K_M GGUF, pinned download and checksum.
- Fast path: frozen Granite features plus independently fitted linear heads.
- Calibration: temperature fitting on a separate labeled split.
- Optional backbone adaptation: BF16 supervised candidate cross-entropy LoRA.
- Training and evaluation commands: [TRAINING.md](TRAINING.md).

The LoRA implementation is not proprietary RLCD. The reference interface is
inspired by [Jev](https://docs.typesafe.ai/introduction); this is not an affiliated
product or an exact TypeSafe SDK replacement.

## Data provenance

No third-party task dataset has been used to train the bundled smoke artifacts.
The synthetic integration data was constructed with deterministic templates and
labels in `scripts/make_smoke_data.py`. The 54-row train, 54-row calibration, and
54-row test splits are published under `validation/hyde-20260917/synthetic-data`.
Templates are split by group, but vocabulary and topics overlap. The generator
and synthetic dataset use this repository's MIT license. No Jev outputs were used
as training labels. Jev was evaluated separately on the same 54 test inputs.

IBM's pretraining and post-training data are outside this project's control;
consult the upstream model card. Future open-data experiments must record source
URLs, immutable versions or content hashes, applicable licenses, transformations,
and split provenance before training.

## Measured limitations

On the small synthetic test set, Jev achieved 100% on all three fields. Untuned
Granite scored 98.1% routing, 50.0% Boolean, and 92.6% scope accuracy. Frozen heads
scored 61.1%, 77.8%, and 72.2%, respectively. These are integration examples and
must not be presented as a representative benchmark. Complete evidence is in
`validation/hyde-20260917`.

The optional three-step LoRA smoke verified GPU gradients, weight updates, saving,
and reload. Its validation NLL worsened from 0.0459 to 3.7917. That checkpoint is
not included or recommended for use. No tuned adapter-to-GGUF promotion path has
yet been validated.

The native baseline evaluates questions sequentially and supports up to 26
options. Fast heads require registered question definitions. Novel-schema
calibration, out-of-domain reliability, and adversarial robustness are unproven.
Syntax validity is not judgment correctness; confidence is not authority to act.

## Hardware and reproducibility

Tested on AMD Radeon 8060S / gfx1151, Fedora Atomic 44 with a dedicated Toolbx.
ROCm layer offload was observed for 41/41 layers. Optional PyTorch uses ROCm 7.2.
RHEL, OpenShift, CUDA, and other GPUs have not been validated. CPU unit tests do
not establish GPU compatibility. See the validation records for exact versions.

## Licensing

Project code and original synthetic fixtures: MIT. IBM weights: Apache-2.0.
llama.cpp: MIT, with its license included beside the patch. Preserve upstream
licenses when redistributing weights or components. No credentials or model
weights are committed to this repository.
