"""Proper scoring rules, post-hoc temperature fitting, and selective metrics."""

import math
import numpy as np
from scipy.optimize import minimize_scalar
from .contracts import ContractError


def softmax(logits, temperature=1.0):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim not in (1, 2) or x.shape[-1] < 2 or not np.isfinite(x).all():
        raise ContractError("invalid_logits")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ContractError("invalid_temperature")
    shifted = (x - x.max(axis=-1, keepdims=True)) / temperature
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def validate_targets(logits, targets):
    x = np.asarray(logits, dtype=float)
    y = np.asarray(targets)
    if x.ndim != 2 or len(x) == 0 or y.shape != (len(x),) or y.dtype.kind not in "iu" or np.any(y < 0) or np.any(y >= x.shape[1]):
        raise ContractError("invalid_targets")
    softmax(x)
    return x, y.astype(int)


def nll(logits, targets, temperature=1.0):
    x, y = validate_targets(logits, targets)
    z = x / temperature
    m = z.max(axis=1)
    return float(np.mean(m + np.log(np.exp(z - m[:, None]).sum(axis=1)) - z[np.arange(len(y)), y]))


def fit_temperature(logits, targets):
    x, y = validate_targets(logits, targets)
    result = minimize_scalar(lambda log_t: nll(x, y, math.exp(log_t)), bounds=(math.log(0.05), math.log(20)), method="bounded")
    if not result.success or not math.isfinite(result.fun):
        raise ContractError("temperature_fit_failed")
    candidate = math.exp(result.x)
    return float(candidate if nll(x, y, candidate) <= nll(x, y) else 1.0)


def wilson_error_upper(errors, count, z=1.959963984540054):
    """Upper end of a two-sided 95% Wilson interval; no independence guarantee."""
    if not count:
        return None
    p = errors / count
    return float((p + z*z/(2*count) + z * math.sqrt(p*(1-p)/count + z*z/(4*count*count))) / (1 + z*z/count))


def metrics(logits, targets, temperature=1.0, threshold=0.85, min_margin=0.10):
    x, y = validate_targets(logits, targets)
    p = softmax(x, temperature)
    confidence, pred = p.max(axis=1), p.argmax(axis=1)
    correct = pred == y
    sorted_p = np.sort(p, axis=1)
    accepted = (confidence >= threshold) & (sorted_p[:, -1] - sorted_p[:, -2] >= min_margin)
    count = int(accepted.sum())
    errors = int((~correct & accepted).sum())
    ece, bins = 0.0, []
    bin_ids = np.minimum((confidence * 10).astype(int), 9)
    for i in range(10):
        mask = bin_ids == i
        if mask.any():
            acc, conf = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += float(mask.mean()) * abs(acc - conf)
            bins.append({"lower": i / 10, "upper": (i + 1) / 10, "n": int(mask.sum()), "accuracy": acc, "mean_probability": conf})
    return {"n": len(y), "accuracy": float(correct.mean()), "nll": nll(x, y, temperature),
            "brier": float(np.mean(np.sum((p - np.eye(p.shape[1])[y])**2, axis=1))),
            "ece_10_bins": ece, "reliability_bins": bins,
            "threshold": threshold, "min_margin": min_margin,
            "coverage": count / len(y), "accepted": count, "accepted_errors": errors,
            "selective_error": errors / count if count else None,
            "selective_error_wilson95_upper": wilson_error_upper(errors, count)}
