"""Typed outputs and explicit advisory policy, separate from learned scores."""

import dataclasses
import time
import uuid
import numpy as np
from .contracts import ContractError, decision_request, digest, dumps, labels, questions, state, typed_answer
from .statistics import softmax


@dataclasses.dataclass(frozen=True)
class Policy:
    min_probability: float = 0.85
    min_margin: float = 0.10
    allow_uncalibrated: bool = False
    abstain_labels: tuple[str, ...] = ("unknown", "abstain")

    def __post_init__(self):
        if any(type(v) not in (int, float) or not 0 <= v <= 1 for v in (self.min_probability, self.min_margin)):
            raise ContractError("invalid_policy_threshold")
        if type(self.allow_uncalibrated) is not bool or not all(type(v) is str for v in self.abstain_labels):
            raise ContractError("invalid_policy")

    def as_dict(self):
        return {**dataclasses.asdict(self), "abstain_labels": list(self.abstain_labels)}


class DecisionEngine:
    def __init__(self, backend, policy=None, journal=None):
        self.backend, self.policy, self.journal = backend, policy or Policy(), journal

    def evaluate_request(self, body):
        model_id = self.backend.identity.get("runtime", {}).get("model_id", "unknown")
        return self.evaluate(*decision_request(body, model_id))

    def evaluate(self, value, question_map):
        value, qs = state(value), questions(question_map)
        start = time.perf_counter()
        logits, temperatures = self.backend.predict(value, qs)
        if set(logits) != set(qs) or set(temperatures) != set(qs):
            raise ContractError("backend_question_ids_mismatch")
        answers, diagnostics, blocked = {}, {}, []
        for key, q in qs.items():
            p = softmax(logits[key], temperatures[key])
            if p.ndim != 1:
                raise ContractError("expected_one_distribution_per_question")
            answers[key] = typed_answer(q, p)
            rank = np.sort(p)
            selected = labels(q)[int(p.argmax())]
            reasons = []
            if self.backend.calibration != "temperature_fitted_on_disjoint_data" and not self.policy.allow_uncalibrated:
                reasons.append("uncalibrated")
            if float(rank[-1]) < self.policy.min_probability:
                reasons.append("low_probability")
            if float(rank[-1] - rank[-2]) < self.policy.min_margin:
                reasons.append("ambiguous")
            if q["type"] == "choice" and selected in self.policy.abstain_labels:
                reasons.append("unknown_or_abstain_option")
            diagnostics[key] = {"selected_probability": float(rank[-1]), "margin": float(rank[-1] - rank[-2]), "reasons": reasons}
            if reasons:
                blocked.append(key)
        response = {"schema_version": "1", "decision_id": str(uuid.uuid4()),
                    "model": self.backend.identity.get("runtime", {}).get("model_id", "unknown"),
                    "backend": self.backend.identity, "schema_hash": digest(qs), "state_hash": digest(value),
                    "answers": answers, "diagnostics": diagnostics, "abstained_fields": blocked,
                    "status": "abstained" if blocked else "proposed", "authority": "advisory_only",
                    "calibration": self.backend.calibration,
                    "confidence_definition": "1_minus_normalized_entropy_not_accuracy",
                    "policy": self.policy.as_dict(), "latency_ms": (time.perf_counter() - start) * 1000}
        dumps(response)
        if self.journal is not None:
            self.journal.record(value, qs, response)
        return response
