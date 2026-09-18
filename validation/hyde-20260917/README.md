# Hyde evidence snapshot

Read ../../TRAINING.md for methodology, commands, measured results, and limitations.
Synthetic integration data only; not a representative model-quality benchmark.
54 train + 54 calibration + 54 test states, each with three questions.
Jev results came from 54 explicitly requested API calls on superrouter.
The native runtime retains its default uncalibrated abstention policy.
LoRA smoke passed mechanically but validation NLL worsened; do not promote it.
Model weights, adapter weights, environments and credentials are not included.
Python package snapshots are tested inventories, not cross-platform binary locks.
