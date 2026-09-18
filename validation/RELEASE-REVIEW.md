# Initial release review

Jev Review evaluated the shared prompt, optional LoRA runner, paired benchmark,
and regression tests before the initial commit. The structured response is in
`jev-review-release.json`. It reported only low-severity, general concerns, with
no concrete failing code path. Those scores are advisory, not proof of quality.

Disposition: retain the focused implementation rather than introduce speculative
abstractions. Existing tests cover candidate ordering, incomplete probabilities,
duplicate/mismatched benchmark rows, API contracts, and train/test separation.
Real GPU tests cover layer offload and training/save/reload. Optional Fullcollar
coverage skips when that separate package is absent. The LoRA adapter's degraded
validation is documented and the checkpoint is excluded. Public release includes
only source, synthetic fixtures, and evidence; no credentials or model weights.

The new 255-option evaluator and larger experiments are not part of this initial
release. Subsequent changes require focused tests and new Jev reviews.
