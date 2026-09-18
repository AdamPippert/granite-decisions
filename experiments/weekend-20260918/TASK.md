# Weekend training setup

Objective: stage and validate all-family v3 ROCm LoRA training on hyde, then schedule Friday 2026-09-18 20:00 PDT through Sunday 2026-09-20 17:00 PDT. Personal project; no Fullcollar or automatic publication of weights.

Acceptance: verified frozen data and source hashes, all 24 training families sampled by scenario group and variant, validation-only checkpoint selection, untouched test/challenge until selection, CPU tests, bounded GPU pilot with saved-adapter reload, detached scheduled launcher and independent hard stop, logs and reproducible commands.

Plan: preserve historical trainer; build v3-specific runner from its tested ROCm backend; add preflight and schedule configuration; run tests and Jev review; stage on hyde; run short isolated pilot; arm tonight's launcher only after validation. Stop optimization Sunday 15:30, reserve evaluation through 16:55, terminate all training processes by 17:00 PDT. No need to consume the full window if trials converge early.

Current: implementation in progress. Frozen dataset commit 9770aef70f9da8d1986850c7422ee80dc78d52c8; HF commit c3ea84b06e16dabe754d49644682f80517b0461b. Host user has no sudo and no lingering user manager; investigate detached schedule reliability without changing account permissions.

Completed setup: native all-family runner, group/variant sampler, validation guard and baseline fallback, heldout postselection evaluation, bounded supervisor, pinned source/package inventories, and persistent superrouter start/stop timers. Start is 2026-09-18 20:00 PDT; hard stop is September 20 17:00 PDT. Fixed four-trial grid allows 2,500 updates per trial, with convergence stopping and absolute deadlines.

Validation: 97 CPU tests run; 94 passed, 3 optional Fullcollar skipped. Both shell scripts and all four systemd units validate. Initial batched GPU pilot failed (maximum probability delta 0.181423); corrected to single-record evaluation matching default serving. Final pilot completed two all-family updates, saved/reloaded weights, calibration/test/challenge inference, and reported exactly zero repeatability error across 24 families. Fresh-process adapter load reproduced all three probe logits exactly. Actual detached container supervisor killed a TERM-ignoring subprocess at its deadline. All18 dataset checksums and18 runtime source files match the frozen/local source; launch preflight passed. Superrouter already had lingering enabled; no account permissions changed.

Timers installed and enabled; systemd reports Friday20:00 and Sunday16:59:45 PDT triggers. The main `run` directory does not exist and the weekend run has not started. Watchdog is independent of SSH after detached launch. Both hosts must be up for dispatch; reboot/container shutdown interrupts without automatic retry. Checkpoints and progress persist. No automatic publication/deployment of trained weights. Pilot metrics are integration evidence, not model-quality claims.
