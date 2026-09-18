# Contributing

Start with the CPU unit tests in README.md. GPU integration checks and training
commands are in TRAINING.md. Keep frozen evaluation data separate from training,
calibration, and checkpoint selection. Report failures and skipped checks.

For dataset contributions, include source URLs, authors/citations, license,
revision/content hashes, generation or transformation code, and split provenance.
Publish synthetic data with its generator, seed, labeling rules, and limitations.
Never commit credentials, private operational data, downloaded weights, or local
Python environments. Use a new artifact directory for each training run.

A model improvement needs held-out evidence against the untuned baseline and Jev,
with accuracy, calibration, coverage/error tradeoffs, and hardware-specific timing.
Do not infer general quality from smoke examples or promote a checkpoint merely
because its training run completed. Keep optional training dependencies isolated
from the lightweight inference/head-training package.
