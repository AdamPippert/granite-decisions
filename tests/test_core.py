import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
from granite_decisions.contracts import ContractError, digest, dumps, label_index, loads, questions, typed_answer
from granite_decisions.journal import Journal
from granite_decisions.runtime import DecisionEngine, Policy
from granite_decisions.statistics import fit_temperature, metrics, nll, softmax
from granite_decisions.training import HeadBackend, check_disjoint, fit_head, train_bundle

QS = {
    "route": {"type": "choice", "instructions": "Choose a route.", "criteria": {"a": "First route", "b": "Second route"}},
    "present": {"type": "noul", "instructions": "Is the feature present?"},
    "level": {"type": "score", "instructions": "Choose a level.", "criteria": ["Low", "High"]},
}


class FakeFeatures:
    identity = {"kind": "synthetic_test_features", "model_id": "synthetic-not-granite"}

    def __init__(self):
        self.calls = 0

    def encode(self, states):
        self.calls += 1
        return np.array([[1., 0.] if s["x"] else [0., 1.] for s in states])


class FakeBackend:
    identity = {"kind": "test", "runtime": {"model_id": "synthetic-test"}}
    calibration = "uncalibrated_conditional_token_probabilities"

    def predict(self, state, qs):
        return {k: np.array([0., 8.]) for k in qs}, {k: 1.0 for k in qs}


def split(prefix, count=60):
    return [{"id": f"{prefix}-{i}", "state": {"x": bool(i % 2), "record": f"{prefix}-{i}"},
             "labels": {"route": "a" if i % 2 else "b", "present": bool(i % 2), "level": i % 2}} for i in range(count)]


class Contracts(unittest.TestCase):
    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ContractError):
            loads('{"state":1,"state":2}')

    def test_nonfinite_values_rejected(self):
        for value in ('{"v":NaN}', '{"v":1e9999}'):
            with self.assertRaises(ContractError):
                loads(value)

    def test_depth_bounded(self):
        with self.assertRaises(ContractError):
            loads('[' * 40 + '0' + ']' * 40)

    def test_question_order_canonical(self):
        self.assertEqual(digest(questions(QS)), digest(questions(dict(reversed(list(QS.items()))))))

    def test_boolean_not_ordinal_integer(self):
        with self.assertRaises(ContractError):
            label_index(QS["level"], True)

    def test_boolean_label_strict(self):
        with self.assertRaises(ContractError):
            label_index(QS["present"], "true")

    def test_choice_output_cannot_escape_schema(self):
        self.assertEqual(typed_answer(QS["route"], [.1, .9])["choice"], "b")

    def test_ordinal_score_is_expectation(self):
        self.assertEqual(typed_answer(QS["level"], [.25, .75])["score"], .75)

    def test_bad_probability_mass_rejected(self):
        for probs in ([.7, .7], [float("nan"), .5], [-.1, 1.1], [1]):
            with self.assertRaises(ContractError):
                typed_answer(QS["route"], probs)

    def test_uniform_confidence_is_zero(self):
        self.assertAlmostEqual(typed_answer(QS["route"], [.5, .5])["confidence"], 0)

    def test_unknown_question_field_rejected(self):
        bad = copy.deepcopy(QS)
        bad["route"]["tool"] = "execute"
        with self.assertRaises(ContractError):
            questions(bad)

    def test_structured_rubric_roundtrips_and_keeps_level_order(self):
        q = {"type": "score", "instructions": ["Rate the request", {"dimension": "urgency"}],
             "criteria": [{"description": "No deadline"}, ["Explicit deadline", {"days": 1}]]}
        clean = questions({"q": q})["q"]
        answer = typed_answer(clean, [.2, .8])
        self.assertEqual(clean, q)
        self.assertEqual(answer["legend"], {"0": q["criteria"][0], "1": q["criteria"][1]})
        self.assertEqual(answer["score"], .8)

    def test_score_accepts_ten_levels_and_rejects_eleven(self):
        q = {"type": "score", "instructions": "Rate", "criteria": [str(i) for i in range(10)]}
        self.assertEqual(len(questions({"q": q})["q"]["criteria"]), 10)
        q["criteria"].append("ten")
        with self.assertRaises(ContractError):
            questions({"q": q})

    def test_choice_accepts_255_null_descriptions_and_rejects_256(self):
        q = {"type": "choice", "instructions": "Choose", "criteria": {str(i): None for i in range(255)}}
        self.assertEqual(len(questions({"q": q})["q"]["criteria"]), 255)
        q["criteria"]["255"] = None
        with self.assertRaises(ContractError):
            questions({"q": q})

    def test_invalid_structured_instructions_rejected(self):
        for instructions in (True, 7, None, "", {}, [], {"nested": float("inf")}):
            with self.assertRaises(ContractError):
                questions({"q": {"type": "noul", "instructions": instructions}})


class Calibration(unittest.TestCase):
    def test_large_logits_stable(self):
        np.testing.assert_allclose(softmax([10000., 10000.]), [.5, .5])

    def test_temperature_reduces_overconfident_nll(self):
        x = np.tile([0., 8.], (100, 1))
        y = np.array([0] * 20 + [1] * 80)
        t = fit_temperature(x, y)
        self.assertGreater(t, 1)
        self.assertLess(nll(x, y, t), nll(x, y))

    def test_zero_coverage_is_unknown_error_not_zero(self):
        report = metrics([[0., 0.], [0., 0.]], [0, 1])
        self.assertEqual(report["coverage"], 0)
        self.assertIsNone(report["selective_error"])
        self.assertIsNone(report["selective_error_wilson95_upper"])

    def test_zero_observed_errors_still_has_uncertainty(self):
        report = metrics(np.tile([0., 10.], (20, 1)), [1] * 20)
        self.assertGreater(report["selective_error_wilson95_upper"], 0.1)

    def test_train_fit_separates_signal(self):
        x = np.tile(np.eye(2), (50, 1))
        y = np.tile([0, 1], 50)
        w, b = fit_head(x, y, 2)
        self.assertTrue(np.array_equal((x @ w + b).argmax(axis=1), y))

    def test_identical_state_cannot_leak_under_new_id(self):
        a = {"id": "a", "state": {"text": "identical"}}
        b = {"id": "b", "state": {"text": "identical"}}
        with self.assertRaises(ContractError):
            check_disjoint([a], [b])


class Runtime(unittest.TestCase):
    def test_uncalibrated_abstains_despite_large_logits(self):
        result = DecisionEngine(FakeBackend()).evaluate({"x": 1}, QS)
        self.assertEqual(result["status"], "abstained")
        self.assertIn("uncalibrated", result["diagnostics"]["route"]["reasons"])

    def test_explicit_experiment_remains_advisory(self):
        result = DecisionEngine(FakeBackend(), Policy(allow_uncalibrated=True)).evaluate({}, QS)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["authority"], "advisory_only")

    def test_unknown_option_abstains(self):
        qs = {"route": {"type": "choice", "instructions": "Choose.", "criteria": {"a": "Route", "unknown": "Unknown"}}}
        result = DecisionEngine(FakeBackend(), Policy(allow_uncalibrated=True)).evaluate({}, qs)
        self.assertEqual(result["status"], "abstained")

    def test_bad_policy_fails(self):
        for value in (float("nan"), 1.1, True):
            with self.assertRaises(ContractError):
                Policy(min_probability=value)

    def test_wrong_distribution_dimensions_rejected(self):
        class Bad(FakeBackend):
            def predict(self, value, qs):
                return {k: np.ones((1, 2)) for k in qs}, {k: 1. for k in qs}
        with self.assertRaises(ContractError):
            DecisionEngine(Bad()).evaluate({}, QS)


class Bundle(unittest.TestCase):
    def test_train_reload_and_predict_all_heads_one_feature_call(self):
        runtime = FakeFeatures()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "candidate"
            result = train_bundle(runtime, QS, split("train"), split("cal"), split("test"), out)
            self.assertEqual(result["promotion"], "candidate_only")
            backend = HeadBackend(runtime, out)
            before = runtime.calls
            response = DecisionEngine(backend).evaluate({"x": True}, QS)
            self.assertEqual(runtime.calls, before + 1)
            self.assertEqual(response["answers"]["route"]["choice"], "a")
            self.assertGreater(response["answers"]["present"]["noul"], .9)
            self.assertGreater(response["answers"]["level"]["score"], .9)
            for requested in ({"renamed": QS["route"]},
                              {"level": QS["level"], "renamed": QS["route"]},
                              {"alias one": QS["route"], "alias two": QS["route"]}):
                before = runtime.calls
                subset = DecisionEngine(backend).evaluate({"x": True}, requested)
                self.assertEqual(runtime.calls, before + 1)
                self.assertEqual(set(subset["answers"]), set(requested))
                for name, q in requested.items():
                    original = "route" if q == QS["route"] else "level"
                    self.assertEqual(subset["answers"][name], response["answers"][original])
            changed = copy.deepcopy(QS)
            changed["route"]["instructions"] += " Changed meaning."
            before = runtime.calls
            with self.assertRaises(ContractError):
                backend.predict({"x": True}, changed)
            self.assertEqual(runtime.calls, before)
            bad_runtime = FakeFeatures()
            bad_runtime.identity = {"kind": "different_model"}
            with self.assertRaises(ContractError):
                HeadBackend(bad_runtime, out)
            with open(out / "heads.npz", "ab") as handle:
                handle.write(b"corruption")
            with self.assertRaises(ContractError):
                HeadBackend(runtime, out)

    def test_tiny_calibration_not_promoted_to_fitted(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ContractError):
                train_bundle(FakeFeatures(), QS, split("a"), split("b", 4), split("c"), Path(directory)/"out")

    def test_duplicate_question_definitions_cannot_learn_id_dependent_heads(self):
        runtime = FakeFeatures()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ContractError, "train_each_question_definition_once"):
                train_bundle(runtime, {"one": QS["route"], "two": QS["route"]},
                             split("train"), split("cal"), split("test"), Path(directory)/"out")
        self.assertEqual(runtime.calls, 0)


class Feedback(unittest.TestCase):
    def test_negative_history_retained_without_mutating_model(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory)/"log.sqlite", record_state=True)
            backend = FakeBackend()
            engine = DecisionEngine(backend, journal=journal)
            result = engine.evaluate({"text": "a"}, QS)
            verified = {"route": "a", "present": False, "level": 0}
            journal.feedback(result["decision_id"], "failure", verified, "reviewer", "Observed failure")
            journal.feedback(result["decision_id"], "success", verified, "reviewer", "Follow-up annotation")
            with journal._connect() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0], 2)
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute("DELETE FROM feedback")
            self.assertEqual(journal.export_verified()[0]["labels"], verified)
            self.assertEqual(backend.calibration, "uncalibrated_conditional_token_probabilities")

    def test_unknown_correction_withdraws_training_label(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory)/"log.sqlite", record_state=True)
            r = DecisionEngine(FakeBackend(), journal=journal).evaluate({}, QS)
            journal.feedback(r["decision_id"], "failure", {"route": "a", "present": False, "level": 0}, "r")
            journal.feedback(r["decision_id"], "unknown", {}, "r", "Label uncertain")
            self.assertEqual(journal.export_verified(), [])

    def test_default_does_not_save_state(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory)/"log.sqlite")
            result = DecisionEngine(FakeBackend(), journal=journal).evaluate({"private": "text"}, QS)
            journal.feedback(result["decision_id"], "failure", {"route": "a", "present": False, "level": 0}, "reviewer")
            self.assertEqual(journal.export_verified(), [])
            with journal._connect() as db:
                self.assertIsNone(db.execute("SELECT state FROM decisions").fetchone()[0])


if __name__ == "__main__":
    unittest.main()
