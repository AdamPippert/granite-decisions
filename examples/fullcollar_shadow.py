"""Requires Fullcollar 0.1.0 installed separately; no changes to its source."""
import json
from granite_decisions import DecisionEngine
from granite_decisions.fullcollar import shadow_engine
from granite_decisions.llamacpp import BaselineBackend, LlamaRuntime, NativeClient

with open("baseline-runtime.json") as handle:
    manifest = json.load(handle)
backend = BaselineBackend(LlamaRuntime(NativeClient("http://127.0.0.1:8092"), manifest))
engine = shadow_engine(DecisionEngine(backend))
print(json.dumps(engine.decide({
    "schema_version": "1", "request_id": "granite-shadow-001",
    "skill": "semantic-route", "state": {"request": "Find the documentation for llama.cpp pooling modes."}
}), indent=2))
