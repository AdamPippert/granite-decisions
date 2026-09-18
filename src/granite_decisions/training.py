"""Frozen Granite features, regularized categorical heads, disjoint calibration."""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from .contracts import ContractError, digest, dumps, label_index, labels, loads, question_signature, questions, state
from .llamacpp import file_sha256
from .statistics import fit_temperature, metrics, softmax


def read_dataset(path, question_map):
    records, ids, states = [], set(), set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = loads(line)
            if type(row) is not dict or set(row) != {"id", "state", "labels"}:
                raise ContractError("dataset_requires_id_state_labels")
            if type(row["id"]) is not str or not row["id"]:
                raise ContractError("dataset_id_required")
            state(row["state"])
            if type(row["labels"]) is not dict or set(row["labels"]) != set(question_map):
                raise ContractError("dataset_label_fields_mismatch")
            for key, q in question_map.items():
                label_index(q, row["labels"][key])
            sh = digest(row["state"])
            if row["id"] in ids or sh in states:
                raise ContractError("duplicate_dataset_id_or_state")
            ids.add(row["id"])
            states.add(sh)
            records.append(row)
    if not records:
        raise ContractError("empty_dataset")
    return records


def check_disjoint(*splits):
    ids, hashes = set(), set()
    for rows in splits:
        new_ids = {r["id"] for r in rows}
        new_hashes = {digest(r["state"]) for r in rows}
        if len(new_ids) != len(rows) or len(new_hashes) != len(rows) or ids & new_ids or hashes & new_hashes:
            raise ContractError("data_leakage_duplicate_id_or_state_across_splits")
        ids.update(new_ids)
        hashes.update(new_hashes)


def features(value):
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ContractError("invalid_features")
    return x


def fit_head(x, y, classes, l2=0.01):
    x = features(x)
    y = np.asarray(y, dtype=int)
    if y.shape != (len(x),) or set(y) != set(range(classes)):
        raise ContractError("every_training_class_requires_examples")
    if not np.isfinite(l2) or l2 <= 0:
        raise ContractError("l2_must_be_positive")
    width = x.shape[1]

    def objective(theta):
        weights = theta[:width*classes].reshape(width, classes)
        bias = theta[width*classes:]
        z = x @ weights + bias
        p = softmax(z)
        shifted = z - z.max(axis=1, keepdims=True)
        loss = np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y]) + l2 * np.sum(weights**2) / 2
        p[np.arange(len(y)), y] -= 1
        p /= len(y)
        gradient = np.concatenate(((x.T @ p + l2 * weights).ravel(), p.sum(axis=0)))
        return float(loss), gradient

    solution = minimize(objective, np.zeros(width*classes + classes), jac=True, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-10})
    if not solution.success or not np.isfinite(solution.x).all():
        raise ContractError("head_optimizer_did_not_converge")
    return solution.x[:width*classes].reshape(width, classes), solution.x[width*classes:]


def train_bundle(runtime, question_map, train, calibration, test, output, l2=0.01):
    qs = questions(question_map)
    if len({question_signature(q) for q in qs.values()}) != len(qs):
        raise ContractError("train_each_question_definition_once_alias_at_inference")
    check_disjoint(train, calibration, test)
    if len(train) < 20 or len(calibration) < 50 or len(test) < 50:
        raise ContractError("minimum_split_sizes_train20_calibration50_test50")
    # Require coverage in all splits; 50 is a mechanical floor, not adequacy proof.
    targets = []
    for split in (train, calibration, test):
        ys = {}
        for key, q in qs.items():
            ys[key] = np.array([label_index(q, row["labels"][key]) for row in split])
            if set(ys[key]) != set(range(len(labels(q)))):
                raise ContractError("every_class_required_in_each_split:" + key)
        targets.append(ys)
    if Path(output).exists():
        raise ContractError("candidate_output_already_exists")
    encoded = [features(runtime.encode([r["state"] for r in rows])) for rows in (train, calibration, test)]
    if any(len(x) != len(rows) for x, rows in zip(encoded, (train, calibration, test))):
        raise ContractError("feature_row_count_mismatch")
    if len({x.shape[1] for x in encoded}) != 1:
        raise ContractError("feature_dimensions_mismatch")
    arrays, heads, report = {}, {}, {}
    for i, (key, q) in enumerate(qs.items()):
        w, b = fit_head(encoded[0], targets[0][key], len(labels(q)), l2)
        cal_logits, test_logits = encoded[1] @ w + b, encoded[2] @ w + b
        temperature = fit_temperature(cal_logits, targets[1][key])
        arrays[f"w{i}"], arrays[f"b{i}"] = w, b
        heads[key] = {"weights": f"w{i}", "bias": f"b{i}", "labels": labels(q), "temperature": temperature}
        report[key] = {"calibration_before": metrics(cal_logits, targets[1][key]),
                       "calibration_after": metrics(cal_logits, targets[1][key], temperature),
                       "test_before": metrics(test_logits, targets[2][key]),
                       "test_after": metrics(test_logits, targets[2][key], temperature)}
    manifest = {"format": 1, "questions": qs, "schema_hash": digest(qs),
                "runtime_identity": runtime.identity, "feature_width": encoded[0].shape[1],
                "calibration": "temperature_fitted_on_disjoint_data", "heads": heads,
                "l2": l2, "promotion": "candidate_only",
                "data_hashes": {name: digest(rows) for name, rows in zip(("train", "calibration", "test"), (train, calibration, test))},
                "seen_ids": sorted({r["id"] for split in (train, calibration, test) for r in split}),
                "seen_state_hashes": sorted({digest(r["state"]) for split in (train, calibration, test) for r in split})}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".candidate-", dir=target.parent))
    try:
        np.savez_compressed(temporary / "heads.npz", **arrays)
        manifest["weights_sha256"] = file_sha256(temporary / "heads.npz")
        manifest["bundle_id"] = digest(manifest)
        (temporary / "manifest.json").write_text(dumps(manifest) + "\n")
        (temporary / "evaluation.json").write_text(dumps({"basis": "held_out_labeled_data", "fields": report}) + "\n")
        os.rename(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"bundle_id": manifest["bundle_id"], "output": str(target), "evaluation": report, "promotion": "candidate_only"}


class HeadBackend:
    def __init__(self, runtime, directory):
        self.runtime = runtime
        directory = Path(directory)
        self.manifest = loads((directory / "manifest.json").read_bytes(), limit=32 * 1024 * 1024)
        m = self.manifest
        bundle_id = m.get("bundle_id")
        if m.get("format") != 1 or digest({k: v for k, v in m.items() if k != "bundle_id"}) != bundle_id:
            raise ContractError("invalid_bundle_manifest")
        self.questions = questions(m["questions"])
        if digest(self.questions) != m["schema_hash"] or m["runtime_identity"] != runtime.identity:
            raise ContractError("model_runtime_or_schema_mismatch_retrain")
        if file_sha256(directory / "heads.npz") != m["weights_sha256"]:
            raise ContractError("head_weights_checksum_mismatch")
        self.arrays = {}
        with np.load(directory / "heads.npz", allow_pickle=False) as raw:
            for key, q in self.questions.items():
                head = m["heads"][key]
                w, b = np.array(raw[head["weights"]]), np.array(raw[head["bias"]])
                size = len(labels(q))
                if head["labels"] != labels(q) or w.shape != (m["feature_width"], size) or b.shape != (size,) or not np.isfinite(w).all() or not np.isfinite(b).all():
                    raise ContractError("invalid_head_shape_or_values")
                softmax(b, head["temperature"])
                self.arrays[key] = (w, b)
        if m.get("calibration") != "temperature_fitted_on_disjoint_data":
            raise ContractError("unsupported_calibration_status")
        self.identity = {"kind": "frozen-granite-heads-v1", "bundle_id": bundle_id, "runtime": runtime.identity}
        self.calibration = m["calibration"]
        self.by_question = {}
        for key, q in self.questions.items():
            signature = question_signature(q)
            previous = self.by_question.get(signature)
            if previous is not None:
                same = all(np.array_equal(a, b) for a, b in zip(self.arrays[key], self.arrays[previous]))
                if not same or m["heads"][key]["temperature"] != m["heads"][previous]["temperature"]:
                    raise ContractError("duplicate_question_heads_disagree_retrain")
            else:
                self.by_question[signature] = key

    def predict(self, value, question_map):
        selected = {}
        for key, q in questions(question_map).items():
            trained_key = self.by_question.get(question_signature(q))
            if trained_key is None:
                raise ContractError("untrained_question_retrain_or_use_baseline")
            selected[key] = trained_key
        x = features(self.runtime.encode([value]))
        if x.shape != (1, self.manifest["feature_width"]):
            raise ContractError("embedding_shape_changed")
        logits, temperatures = {}, {}
        for key, trained_key in selected.items():
            w, b = self.arrays[trained_key]
            logits[key] = (x @ w + b)[0]
            temperatures[key] = self.manifest["heads"][trained_key]["temperature"]
        return logits, temperatures
