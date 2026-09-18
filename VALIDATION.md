# Hyde validation update — September 17, 2026

See [TRAINING.md](TRAINING.md) and `validation/hyde-20260917/` for the newly
observed ROCm tests, full head-training cycle, live Jev comparison, and optional
LoRA smoke. The prior CPU development record follows unchanged.

# Validation record — September 17, 2026

## Follow-up revision: TypeSafe contract and Hyde runner

- 49 tests passed and 3 optional Fullcollar tests skipped because Fullcollar is
  unavailable in the current environment: 52 discovered, no failures.
  Transcript: `validation/unit-tests-update.txt`. The original 44-test transcript
  remains intact as earlier evidence, not a claim about this later run.
- New coverage checks structured requests, array state, null Choice descriptions,
  the 10-level Score limit, model-name enforcement, and equal predictions under
  question renaming, subsets, and aliases. Unknown question content is rejected
  before model work. Device enumeration alone cannot satisfy the offload check.
- Python compilation, CLI discovery, structured-example validation, and shell
  syntax checks passed. The native C++ patch and model artifact are unchanged.
- `deploy/run-hyde.sh` now runs the native build and interface checks with
  retained step/server logs. Its GPU success gate requires a ROCm device and
  positive layer-offload log evidence for both serving modes. The runner has
  not been executed on Fedora/ROCm; its log parser is covered by unit tests.
- SSH using the supplied address and username returned `Network is unreachable`
  before authentication. No commands ran on Hyde and no files there changed.
- Real-model inference was not rerun for this revision. The CPU evidence below
  records the earlier source revision. New general-purpose decision weights,
  question-conditioned parallel inference, and Jev parity remain unimplemented;
  see `TYPESAFE-COMPATIBILITY.md`.

## Original CPU validation

- 44 tests passed with no skipped tests, including the actual Fullcollar 0.1.0
  Python contract. The Fullcollar source was read from the existing workspace;
  it was not modified. Transcript: `validation/unit-tests.txt`.
- Real IBM Granite 4.1 3B Q4_K_M inference through a locally compiled native
  llama.cpp CPU server passed for Choice, Score, and Noul. The application
  returned typed distributions and correctly abstained because the baseline
  was uncalibrated.
- Real Granite feature extraction returned four finite, L2-normalized vectors
  with 2,560 dimensions each. A tiny linear head was fitted to those real
  vectors and produced finite two-class logits. This tests the native feature
  and head-training interface; it is not a trained production classifier.
- The complete train/calibrate/test/save/reload flow passed on clearly labeled
  synthetic test features, including schema/runtime mismatch rejection and
  weight-integrity checks.
- The native C++ probability patch compiled and passed the real-model test.
  `git diff --check` and reverse-application validation passed.
- Python package editable installation, module import/CLI discovery, Python
  compilation, and shell-script syntax checks passed.

## Failure found and repaired

The initial real-model integration test failed: upstream llama.cpp returned an
out-of-grammar token in its post-sampling top probabilities. At the inspected
revision, grammar rejection sampling can skip masking the full distribution when
the sampled token is already valid. The client correctly rejected the incomplete
candidate set. The included patch applies the grammar first when post-sampling
probabilities are requested. The repaired path passed on the real model.

An initial attempt to contact local test servers across separate execution
contexts failed because those contexts had separate network namespaces. The
reproducible smoke script launches and tests its server subprocesses in the same
execution context. This was an environment issue, not a model-service failure.

## Exact reference configuration

| Component | Observed value |
| --- | --- |
| Model repository | `ibm-granite/granite-4.1-3b-GGUF` |
| Repository revision | `ab4701481089b58a082ef63cc1cee738887293ff` |
| Weight filename | `granite-4.1-3b-Q4_K_M.gguf` |
| Weight size | 2,099,501,664 bytes |
| Weight SHA-256 | `662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29` |
| llama.cpp revision | `aa39d7a3e145a88202793a89462d65e94a5fc25f` plus included patch |
| Build | GNU 13.3.0, Linux x86_64, CPU |
| Python | 3.12.14 |
| Native threads | 4 generation, 4 prompt processing |
| Context / microbatch | 4,096 / 4,096 tokens |
| Application input budget | 2,048 tokens; overflow rejected |
| Embedding pooling | Last token, L2 normalization |
| GPU devices | None in the reference environment |

The raw successful run is in `validation/native-cpu-smoke.json`. Its single
three-question baseline request took approximately 4.51 seconds after model load.
Four feature extractions plus a tiny head fit took approximately 1.46 seconds in
total. These measurements use different inputs and do not establish a speedup,
throughput curve, latency guarantee, or hardware comparison. They are CPU smoke
timings, not Hyde performance estimates. No model-quality percentage is claimed.

## Not yet validated

- Building or running on Hyde: the original hostname did not resolve; the later
  attempt with the supplied address returned `Network is unreachable`. No
  command ran on Hyde, and no files or services there were changed.
- Fedora/ROCm installation, driver version, actual GPU target, device permissions,
  successful layer offload, VRAM use, and thermal/performance behavior on Hyde.
- Domain quality, calibration under distribution shift, out-of-distribution
  detection, rare-event risk, or reliability on real Fullcollar traffic.
- Backbone fine-tuning, RLCD, or a new general-purpose trained model. This release
  trains supervised heads while keeping Granite frozen.
- Jev quality/speed/cost parity. No Jev API was called and no comparable workload
  benchmark was run.
- Speculative decoding and GPU/backend sampling with the probability patch.
  The included launchers do not enable these features.
- Production load, concurrency beyond the one-active-request service policy,
  multi-host deployment, or a public API.

The unit test transcript includes a deprecation warning from the installed
Starlette version's httpx test-client adapter. All HTTP tests still passed.

## Hyde acceptance procedure

`bash deploy/run-hyde.sh` automates steps 1–3 and retains the run evidence.
The domain-training and workflow checks in steps 4–5 require labeled data.

1. Run `deploy/hyde-native.sh inspect` and retain its host/ROCm/device output.
2. Run `deploy/hyde-native.sh build` in a new workspace. Inspect
   `llama-server --list-devices` and confirm the expected ROCm device.
3. Run `scripts/smoke_native.py` with its default GPU-layer setting and retain
   `hyde-smoke.json`. The smoke test now rejects absent/zero offload evidence;
   retain its server logs for inspection. A CPU run must
   be labeled explicitly with `--gpu-layers 0`.
4. Bind the live embedding runtime, train on verified domain labels, and use
   independent calibration/test data. Review selective error and coverage at
   prespecified thresholds, including errors for consequential routes.
5. Run in Fullcollar shadow mode, collect independently verified outcomes, and
   benchmark p50/p95 latency at representative input lengths. Choose activation
   criteria before inspecting the final acceptance set. Retain the old artifact
   for rollback. Serving selects a bundle explicitly; feedback cannot activate it.
