import unittest
from fastapi.testclient import TestClient
from granite_decisions.runtime import DecisionEngine
from granite_decisions.service import create_app
from test_core import FakeBackend, QS


class Service(unittest.TestCase):
    def test_auth_and_typed_response(self):
        client = TestClient(create_app(DecisionEngine(FakeBackend()), "test-only-key"))
        body = {"state": {"request": "test"}, "questions": QS}
        self.assertEqual(client.post("/v1/decisions", json=body).status_code, 401)
        response = client.post("/v1/decisions", json=body, headers={"Authorization": "Bearer test-only-key"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "abstained")

    def test_duplicate_key_rejected(self):
        client = TestClient(create_app(DecisionEngine(FakeBackend())))
        response = client.post("/v1/decisions", content='{"state":{},"state":{}}')
        self.assertEqual(response.status_code, 400)

    def test_body_limit(self):
        client = TestClient(create_app(DecisionEngine(FakeBackend())))
        self.assertEqual(client.post("/v1/decisions", content=b'x' * 262145).status_code, 400)

    def test_browser_origin_rejected(self):
        client = TestClient(create_app(DecisionEngine(FakeBackend())))
        self.assertEqual(client.get("/health", headers={"Origin": "https://example.com"}).status_code, 403)

    def test_request_cannot_change_policy(self):
        client = TestClient(create_app(DecisionEngine(FakeBackend())))
        response = client.post("/v1/decisions", json={"state": {}, "questions": QS, "allow_uncalibrated": True})
        self.assertEqual(response.status_code, 422)

    def test_documented_request_with_loaded_model_and_array_state(self):
        client = TestClient(create_app(DecisionEngine(FakeBackend())))
        response = client.post("/v1/systemone", json={
            "model": "synthetic-test", "state": [{"role": "user", "content": "Help"}],
            "questions": {"Support team?": {
                "type": "choice", "instructions": {"question": "Who handles this?", "context": ["Support"]},
                "criteria": {"Customer service": None, "Engineering": {"scope": ["Bugs", "Outages"]}},
            }},
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "synthetic-test")
        self.assertEqual(set(response.json()["answers"]["Support team?"]["probabilities"]), {"Customer service", "Engineering"})

    def test_request_cannot_switch_to_an_unloaded_model(self):
        class Counting(FakeBackend):
            calls = 0

            def predict(self, value, qs):
                self.calls += 1
                return super().predict(value, qs)
        backend = Counting()
        client = TestClient(create_app(DecisionEngine(backend)))
        for model in ("jev-latest", "../../other-model.gguf", None, 1):
            response = client.post("/v1/systemone", json={"model": model, "state": {}, "questions": QS})
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["error"], "requested_model_not_loaded")
        self.assertEqual(backend.calls, 0)
