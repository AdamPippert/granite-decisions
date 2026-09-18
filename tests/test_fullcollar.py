import importlib.util
import unittest
from granite_decisions import DecisionEngine
from granite_decisions.fullcollar import GraniteProvider, shadow_engine
from test_core import FakeBackend, QS


@unittest.skipUnless(importlib.util.find_spec("fullcollar"), "optional Fullcollar installation unavailable")
class Fullcollar(unittest.TestCase):
    def test_typed_answers_pass_actual_fullcollar_contract(self):
        from fullcollar.contracts import validate_answers
        result = GraniteProvider(DecisionEngine(FakeBackend()), shadow_only=True).evaluate({}, QS)
        answers = validate_answers(QS, result)
        self.assertEqual(set(answers), set(QS))

    def test_default_provider_prevents_fullcollar_ignoring_abstention(self):
        from fullcollar.contracts import ProviderError
        with self.assertRaises(ProviderError):
            GraniteProvider(DecisionEngine(FakeBackend())).evaluate({}, QS)

    def test_shadow_factory_fixes_mode(self):
        from granite_decisions.contracts import ContractError
        with self.assertRaises(ContractError):
            shadow_engine(DecisionEngine(FakeBackend()), mode="active")
        self.assertEqual(shadow_engine(DecisionEngine(FakeBackend())).mode, "shadow")
