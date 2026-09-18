"""Native llama.cpp HTTP backend. Model execution remains in ROCm llama.cpp."""

import hashlib
import ipaddress
import math
import re
import string
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from .contracts import ContractError, MAX_BYTES, digest, dumps, labels, loads, state

from .prompting import decision_messages

PROMPT_VERSION = "granite-decisions-state-v1"
MODEL_ID = "ibm-granite/granite-4.1-3b"


class BackendError(RuntimeError):
    pass


def gpu_offload_evidence(log):
    """Require positive layer-offload evidence, not just device enumeration."""
    matches = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", log)
    if not matches:
        raise BackendError("gpu_layer_offload_not_observed_in_log")
    offloaded, total = map(int, matches[-1])
    if not 0 < offloaded <= total:
        raise BackendError("gpu_layer_offload_not_positive")
    return {"offloaded_layers": offloaded, "total_layers": total, "basis": "llama_server_load_log"}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise BackendError("llama_redirect_rejected")


class NativeClient:
    def __init__(self, url="http://127.0.0.1:8091", timeout=60, api_key=None):
        parsed = urllib.parse.urlsplit(url)
        try:
            local = ipaddress.ip_address(parsed.hostname or "").is_loopback
            _ = parsed.port
        except ValueError:
            local = False
        if not local or parsed.scheme != "http" or parsed.path not in ("", "/") or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ContractError("use_loopback_http_or_an_ssh_tunnel")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 600:
            raise ContractError("timeout_must_be_0_to_600_seconds")
        if api_key is not None and (type(api_key) is not str or not api_key or any(ord(c) < 33 or ord(c) > 126 for c in api_key)):
            raise ContractError("invalid_api_key")
        self.url, self.timeout, self.api_key = url.rstrip("/"), timeout, api_key
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, path, body=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        raw = None if body is None else dumps(body).encode()
        if raw is not None and len(raw) > MAX_BYTES * 4:
            raise ContractError("backend_request_too_large")
        req = urllib.request.Request(self.url + path, data=raw, headers=headers)
        try:
            with self.opener.open(req, timeout=self.timeout) as result:
                return loads(result.read(4 * MAX_BYTES + 1), 4 * MAX_BYTES)
        except urllib.error.HTTPError as exc:
            raise BackendError("llama_http_" + str(exc.code)) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise BackendError("llama_unavailable") from None
        except ContractError:
            raise BackendError("llama_invalid_json") from None

    def tokens(self, text, add_special=False, parse_special=False):
        raw = self.request("/tokenize", {"content": text, "add_special": add_special, "parse_special": parse_special})
        values = raw.get("tokens") if type(raw) is dict else None
        if type(values) is not list or not all(type(v) is int and v >= 0 for v in values):
            raise BackendError("invalid_tokenization")
        return values


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def props_fingerprint(props):
    if type(props) is not dict or not all(props.get(k) for k in ("model_path", "build_info")):
        raise BackendError("server_must_expose_model_path_and_build_info")
    return digest({k: props.get(k) for k in ("model_path", "model_alias", "build_info", "chat_template", "total_slots",
                                           "granite_decisions_grammar_probs", "granite_decisions_pooling",
                                           "granite_decisions_gpu_layers")})


def bind_runtime(client, model_file, model_id=MODEL_ID, mode="embedding", max_tokens=2048):
    """Run on the server host; bind actual GGUF bytes to its observed server."""
    if model_id not in {f"ibm-granite/granite-4.1-{size}b" for size in (3, 8, 30)}:
        raise ContractError("expected_granite_4_1_instruct_model_id")
    if mode not in {"embedding", "baseline"}:
        raise ContractError("unknown_backend_mode")
    props = client.request("/props")
    fingerprint = props_fingerprint(props)
    if mode == "baseline" and props.get("granite_decisions_grammar_probs") != 1:
        raise BackendError("baseline_requires_bundled_llama_probability_patch")
    if mode == "embedding" and props.get("granite_decisions_pooling") != 3:
        raise BackendError("embedding_requires_verified_last_pooling_from_bundled_server")
    actual = Path(model_file).resolve(strict=True)
    if actual != Path(props["model_path"]).resolve():
        raise ContractError("server_model_path_does_not_match_gguf")
    n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
    if type(max_tokens) is not int or max_tokens < 16 or type(n_ctx) is not int or max_tokens + 2 > n_ctx:
        raise ContractError("max_tokens_must_fit_server_slot_context")
    # The launcher fixes pooling=last. The native response shape is also checked.
    return {"format": 1, "model_id": model_id, "gguf_sha256": file_sha256(actual),
            "server_fingerprint": fingerprint, "build_info": props["build_info"],
            "mode": mode, "pooling": "last" if mode == "embedding" else None,
            "normalization": "l2" if mode == "embedding" else None,
            "max_tokens": max_tokens, "prompt_version": PROMPT_VERSION}


class LlamaRuntime:
    def __init__(self, client, manifest):
        self.client, self.manifest = client, loads(dumps(manifest))
        if self.manifest.get("format") != 1 or self.manifest.get("prompt_version") != PROMPT_VERSION:
            raise ContractError("unsupported_runtime_manifest")
        if self.manifest.get("mode") not in {"embedding", "baseline"}:
            raise ContractError("invalid_runtime_mode")
        if type(self.manifest.get("max_tokens")) is not int or self.manifest["max_tokens"] < 16:
            raise ContractError("invalid_context_budget")
        self.identity = {"kind": "llama.cpp", **self.manifest}
        self._codes = None
        self.check_identity()

    def check_identity(self):
        props = self.client.request("/props")
        if self.manifest["mode"] == "baseline" and props.get("granite_decisions_grammar_probs") != 1:
            raise BackendError("baseline_requires_bundled_llama_probability_patch")
        if self.manifest["mode"] == "embedding" and props.get("granite_decisions_pooling") != 3:
            raise BackendError("embedding_requires_verified_last_pooling_from_bundled_server")
        if props_fingerprint(props) != self.manifest["server_fingerprint"]:
            raise BackendError("server_changed_rebind_and_recalibrate")
        n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
        if type(n_ctx) is not int or n_ctx < self.manifest["max_tokens"] + 2:
            raise BackendError("server_context_changed")

    def _bounded_tokens(self, prompt, special=False):
        ids = self.client.tokens(prompt, add_special=True, parse_special=special)
        if not ids or len(ids) > self.manifest["max_tokens"]:
            raise ContractError("context_budget_exceeded_no_truncation")
        return ids

    def encode(self, states):
        if self.manifest["mode"] != "embedding":
            raise ContractError("embedding_runtime_required")
        self.check_identity()
        rows = []
        for value in states:
            state(value)
            text = "Represent this state for classification.\nSTATE:\n" + dumps(value) + "\nEND STATE."
            ids = self._bounded_tokens(text)
            response = self.client.request("/embedding", {"content": ids, "embd_normalize": 2})
            try:
                if type(response) is not list or len(response) != 1:
                    raise ValueError()
                vector = np.asarray(response[0]["embedding"], dtype=np.float64)
                if vector.ndim == 2 and vector.shape[0] == 1:
                    vector = vector[0]
                if vector.ndim != 1 or vector.size < 2 or not np.isfinite(vector).all():
                    raise ValueError()
                norm = np.linalg.norm(vector)
                if not np.isclose(norm, 1.0, atol=1e-3):
                    raise ValueError()
                rows.append(vector)
            except (KeyError, TypeError, ValueError):
                raise BackendError("expected_one_finite_l2_pooled_embedding_use_pooling_last") from None
        if not rows:
            raise ContractError("empty_batch")
        try:
            return np.stack(rows)
        except ValueError:
            raise BackendError("embedding_dimensions_changed") from None

    def baseline_logits(self, value, question_map):
        """One constrained output token per question; NOT the shared-head fast path."""
        if self.manifest["mode"] != "baseline":
            raise ContractError("baseline_runtime_required")
        self.check_identity()
        state(value)
        if self._codes is None:
            self._codes = []
            for code in string.ascii_uppercase:
                ids = self.client.tokens(code)
                if len(ids) != 1:
                    raise BackendError("baseline_requires_single_token_letter_codes")
                self._codes.append((code, ids[0]))
            if len({v for _, v in self._codes}) != 26:
                raise BackendError("letter_token_ids_not_unique")
        result = {}
        for key, q in question_map.items():
            names = labels(q)
            if len(names) > 26:
                raise ContractError("baseline_max_26_options_use_trained_heads_for_more")
            codes = self._codes[:len(names)]
            prompt = self.client.request("/apply-template", {
                "messages": decision_messages(value, q)}).get("prompt")
            if type(prompt) is not str:
                raise BackendError("chat_template_failed")
            ids = self._bounded_tokens(prompt, special=True)
            response = self.client.request("/completion", {
                "prompt": ids, "n_predict": 1, "temperature": 1.0,
                "samplers": ["temperature"], "top_k": 0, "top_p": 1.0, "min_p": 0.0,
                "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                "grammar": "root ::= " + " | ".join('"' + c + '"' for c, _ in codes),
                "n_probs": len(codes), "post_sampling_probs": True,
                "seed": 0, "cache_prompt": False, "stream": False})
            if response.get("truncated") is not False:
                raise BackendError("backend_truncation_or_missing_truncation_status")
            try:
                steps = response["completion_probabilities"]
                if len(steps) != 1:
                    raise ValueError()
                entries = steps[0]["top_probs"]
                by_id = {entry["id"]: float(entry["prob"]) for entry in entries}
                if len(by_id) != len(entries) or set(by_id) != {idx for _, idx in codes}:
                    raise ValueError()
                p = np.array([by_id[idx] for _, idx in codes])
                if not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1.0, atol=1e-5):
                    raise ValueError()
                result[key] = np.log(np.maximum(p, np.finfo(float).tiny))
            except (KeyError, TypeError, ValueError):
                raise BackendError("incomplete_candidate_probabilities_not_safe_to_infer_missing_scores") from None
        return result


class BaselineBackend:
    def __init__(self, runtime):
        self.runtime = runtime
        self.identity = {"kind": "one-token-baseline-v1", "runtime": runtime.identity}
        self.calibration = "uncalibrated_conditional_token_probabilities"

    def predict(self, value, question_map):
        return self.runtime.baseline_logits(value, question_map), {key: 1.0 for key in question_map}
