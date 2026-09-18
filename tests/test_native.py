import copy
import unittest
import numpy as np
from granite_decisions.contracts import ContractError
from granite_decisions.llamacpp import BackendError, LlamaRuntime, NativeClient, PROMPT_VERSION, gpu_offload_evidence, props_fingerprint


class StubNative:
    def __init__(self):
        self.props = {"model_path": "/models/granite.gguf", "build_info": "test-build", "chat_template": "template", "total_slots": 1, "granite_decisions_grammar_probs": 1, "granite_decisions_pooling": 3,
                      "default_generation_settings": {"n_ctx": 4096}}
        self.calls, self.partial, self.truncated = [], False, False

    def request(self, path, body=None):
        self.calls.append((path, body))
        if path == "/props":
            return self.props
        if path == "/apply-template":
            return {"prompt": "test template"}
        if path == "/embedding":
            return [{"index": 0, "embedding": [[.6, .8]]}]
        if path == "/completion":
            top = [{"id": 65, "prob": .25}, {"id": 66, "prob": .75}]
            return {"truncated": self.truncated, "content": "irrelevant",
                    "completion_probabilities": [{"top_probs": top[:1] if self.partial else top}]}
        raise AssertionError(path)

    def tokens(self, text, **kwargs):
        return [ord(text)] if len(text) == 1 else [1, 2, 3]


def runtime(client, mode):
    return LlamaRuntime(client, {"format": 1, "mode": mode, "model_id": "ibm-granite/granite-4.1-3b", "prompt_version": PROMPT_VERSION,
                                 "server_fingerprint": props_fingerprint(client.props), "max_tokens": 2048})


class Native(unittest.TestCase):
    def test_gpu_device_listing_is_insufficient_offload_evidence(self):
        for log in ("ROCm0: AMD GPU", "load_tensors: offloaded 0/41 layers to GPU",
                    "load_tensors: offloaded 42/41 layers to GPU"):
            with self.assertRaises(BackendError):
                gpu_offload_evidence(log)
        self.assertEqual(gpu_offload_evidence("load_tensors: offloaded 41/41 layers to GPU")["offloaded_layers"], 41)

    def test_loopback_only(self):
        for url in ("http://hyde:8080", "http://1.2.3.4", "http://127.0.0.1/?key=x", "http://user:pass@127.0.0.1"):
            with self.assertRaises(ContractError):
                NativeClient(url)

    def test_last_pool_embedding(self):
        client = StubNative()
        x = runtime(client, "embedding").encode([{"hello": "world"}])
        np.testing.assert_allclose(x, [[.6, .8]])
        self.assertEqual(sum(p == "/embedding" for p, _ in client.calls), 1)

    def test_baseline_uses_all_candidate_probabilities_not_generated_text(self):
        client = StubNative()
        r = runtime(client, "baseline")
        q = {"decision": {"type": "noul", "instructions": "True?"}}
        result = r.baseline_logits({}, q)
        np.testing.assert_allclose(np.exp(result["decision"]), [.25, .75])
        payload = next(b for p, b in client.calls if p == "/completion")
        self.assertEqual(payload["n_predict"], 1)
        self.assertEqual(payload["temperature"], 1)
        self.assertEqual(payload["samplers"], ["temperature"])

    def test_missing_low_probability_candidate_rejected(self):
        client = StubNative()
        client.partial = True
        with self.assertRaises(BackendError):
            runtime(client, "baseline").baseline_logits({}, {"q": {"type": "noul", "instructions": "True?"}})

    def test_context_truncation_rejected(self):
        client = StubNative()
        client.truncated = True
        with self.assertRaises(BackendError):
            runtime(client, "baseline").baseline_logits({}, {"q": {"type": "noul", "instructions": "True?"}})

    def test_budget_exceeded_before_inference(self):
        client = StubNative()
        r = runtime(client, "embedding")
        client.tokens = lambda *a, **k: list(range(3000))
        with self.assertRaises(ContractError):
            r.encode([{}])
        self.assertFalse(any(p == "/embedding" for p, _ in client.calls))

    def test_runtime_change_rejected(self):
        client = StubNative()
        r = runtime(client, "embedding")
        client.props["build_info"] = "new-build"
        with self.assertRaises(BackendError):
            r.encode([{}])

    def test_unpatched_baseline_rejected(self):
        client = StubNative()
        del client.props["granite_decisions_grammar_probs"]
        with self.assertRaises(BackendError):
            runtime(client, "baseline")

    def test_wrong_pooling_rejected(self):
        client = StubNative()
        client.props["granite_decisions_pooling"] = 1
        with self.assertRaises(BackendError):
            runtime(client, "embedding")


if __name__ == "__main__":
    unittest.main()
