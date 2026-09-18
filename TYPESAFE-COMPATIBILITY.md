# TypeSafe reference and remaining work

Reviewed September 17, 2026 against the supplied
[introduction](https://docs.typesafe.ai/introduction),
[HTTP reference](https://docs.typesafe.ai/api), and primitive pages.
This project is an independent Granite implementation. It is an initial
foundation, not a completed open reproduction of Jev or a drop-in SDK endpoint.

## Checked interface

| Behavior | Current implementation |
| --- | --- |
| Request | `state`, `questions`, optional `model`; explicit model must match the configured Granite model |
| State | String, object, or array; JSON remains subject to depth and byte limits |
| Instructions | Nonempty string, object, or array, preserved as structured data |
| Choice | 2–255 named options in the contract and trained heads; baseline limited to 26 single-token codes |
| Choice descriptions | Null, nonempty string, object, or array; names can contain spaces and Unicode |
| Score | 2–10 ordered descriptions; returns the probability-weighted zero-based index and original description legend |
| Noul | Probability of true; optional true/false descriptions remain strings, following the HTTP reference |
| Question IDs | Response routing only; changing an ID does not select a different learned function |
| Registered questions | Subsets, reordered requests, and aliases reuse existing heads; new question content is rejected before feature extraction |
| Response | `model` and typed `answers`, plus local diagnostics, calibration, and advisory policy metadata |

The [Choice](https://docs.typesafe.ai/primitives/choice) and
[Score](https://docs.typesafe.ai/primitives/score) articles explicitly allow
structured descriptions, although some type rows in the HTTP reference are
narrower. This implementation follows those articles. It does not claim that
the TypeSafe SDK's response models accept the structured legends or our extra
metadata. No TypeSafe SDK compatibility test has been run.

Local limits: at most 64 questions, 128 characters per question ID or option
name, nonempty instructions/descriptions, 256 KiB JSON requests, depth 32, and
the bound runtime's token budget. Numeric/bool/null state roots are rejected.
Choice names, including null-description names, remain visible to the baseline
model. Question IDs are never put in the inference prompt.

## Architectural differences

The [introduction](https://docs.typesafe.ai/introduction) describes independent
parallel evaluation of runtime-supplied questions without text generation. The
current two backends each implement only part of that target:

| Requirement | Baseline | Trained heads |
| --- | --- | --- |
| New question definitions at runtime | Yes, within token and option limits | Requires training new heads |
| No output-token decoding | One constrained token per question | Yes |
| Shared state computation | No; one prompt per question | One state encoding for all requested heads |
| Independent questions | Separate prompts, evaluated sequentially | Separate learned functions on the same state vector |
| Independent Score-level evaluation | No; all levels are included in the prompt | Fixed categorical head, not separately conditioned level branches |
| Probabilities fitted on held-out data | No | Per-head temperature fitting |
| Nearly constant latency as questions increase | Not implemented | Not benchmarked; small head computations still grow with question count |

Neither backend implements Jev's architecture, sampler, RLCD, or broadly trained
decision weights. Schema-valid answers can still be wrong. The
[confidence documentation](https://docs.typesafe.ai/confidence) does not provide
a formula sufficient to establish numeric equivalence: this project explicitly
uses one minus normalized entropy. There is also no `usage` field compatible
with TypeSafe token accounting, no `jev-latest` alias, and no reproduction of
TypeSafe rate-limit or overload status codes.

## Next model milestone — proposed, not implemented

A closer open model needs a shared **question-conditioned** decision function,
trained across varied states, questions, and criteria. Training separate heads
on a fixed list of tasks cannot establish generalization to unseen questions.

One design to evaluate on Granite is to reuse a read-only state-prefix KV cache
and run isolated question/criterion branches. Native llama.cpp would expose
branch representations to learned decision heads without generating answer
tokens. Each branch must be unable to attend to other question branches. Score
branches would receive the level description without its index or neighboring
levels; indices would be applied only when computing the expected score in
ordinary code. This is a design proposal, not a claim about Jev's internals.

Acceptance requires tests for branch isolation, comparisons with full isolated
forward passes, a labeled multi-task training corpus, and evaluation that holds
out entire question definitions and domains as well as input states. Measure
calibration, accuracy, memory, and latency across input lengths and question
counts on Hyde. Shared-prefix caching alone does not establish flat latency.

The current CPU evidence validates native Granite inference and feature/head
plumbing. The Hyde runner supplies the next hardware check. It cannot supply
missing training data or establish Jev quality or speed parity.
