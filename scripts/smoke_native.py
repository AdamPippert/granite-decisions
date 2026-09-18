"""Launch real llama.cpp subprocesses and check Granite interfaces on one host.

This is an integration smoke test, not a decision-quality benchmark.
Defaults to ROCm offload for Hyde; --gpu-layers 0 is an explicit CPU reference.
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from granite_decisions import DecisionEngine
from granite_decisions.contracts import dumps
from granite_decisions.llamacpp import BackendError, BaselineBackend, LlamaRuntime, NativeClient, bind_runtime, gpu_offload_evidence
from granite_decisions.training import fit_head

parser = argparse.ArgumentParser()
parser.add_argument("--server", required=True)
parser.add_argument("--gguf", required=True)
parser.add_argument("--gpu-layers", type=int, default=99)
parser.add_argument("--out", required=True)
parser.add_argument("--log-dir", help="New directory for retained server logs (default: OUT.logs)")
args = parser.parse_args()
if Path(args.out).exists():
    raise SystemExit("Output already exists; choose a new report path.")
log_directory = Path(args.log_dir or (args.out + ".logs"))
log_directory.mkdir(parents=True, exist_ok=False)
server, model = str(Path(args.server).resolve()), str(Path(args.gguf).resolve())
report = {"test": "real_granite_native_interface_smoke", "gpu_layers_requested": args.gpu_layers, "quality_benchmark": False}
devices = subprocess.run([server, "--list-devices"], check=True, capture_output=True, text=True)
report["devices"] = devices.stdout + devices.stderr
if args.gpu_layers > 0 and not re.search(r"(?:ROCm|HIP)\d", report["devices"]):
    raise SystemExit("No ROCm device reported; refusing to call this a GPU test.")

try:
    for mode, port in (("baseline", 19092), ("embedding", 19091)):
        log_path = log_directory / (mode + ".log")
        command = [server, "--model", model, "--host", "127.0.0.1", "--port", str(port),
                   "--ctx-size", "4096", "--batch-size", "4096", "--ubatch-size", "4096",
                   "--verbosity", "4", "--cors-origins", "localhost",
                   "--parallel", "1", "--n-gpu-layers", str(args.gpu_layers),
                   "--threads", "4", "--threads-batch", "4", "--no-context-shift", "--no-webui"]
        if mode == "embedding":
            command += ["--embedding", "--pooling", "last"]
        with log_path.open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            client = NativeClient(f"http://127.0.0.1:{port}", timeout=120, api_key=os.environ.get("LLAMA_API_KEY"))
            deadline = time.monotonic() + 90
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("llama-server exited: " + log_path.read_text()[-3000:])
                try:
                    if client.request("/health").get("status") == "ok":
                        break
                except BackendError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("server did not become ready")
                time.sleep(.25)
            evidence = gpu_offload_evidence(log_path.read_text()) if args.gpu_layers > 0 else {"basis": "explicit_cpu_reference"}
            manifest = bind_runtime(client, model, mode=mode)
            runtime = LlamaRuntime(client, manifest)
            if mode == "baseline":
                qs = {
                    "route": {"type": "choice", "instructions": "What does the request ask to do?", "criteria": {
                        "code": "Write or fix code.", "retrieve": "Find documentation.", "unknown": "Cannot determine the request."}},
                    "python_mentioned": {"type": "noul", "instructions": "Does the request explicitly mention Python?"},
                    "complexity": {"type": "score", "instructions": "How complex is the request?", "criteria": ["Simple task", "Several dependent steps", "Open-ended research"]},
                }
                response = DecisionEngine(BaselineBackend(runtime)).evaluate({"request": "Write a Python function that adds two integers."}, qs)
                assert response["status"] == "abstained", "Uncalibrated scores must abstain by default"
                report[mode] = response
            else:
                data = [{"request": text} for text in (
                    "Write a Python function to add two integers.",
                    "Fix the Python function with a syntax error.",
                    "Find documentation for Python list methods.",
                    "Find the official API reference for Python dictionaries.")]
                start = time.perf_counter()
                x = runtime.encode(data)
                assert x.shape == (4, 2560), x.shape
                assert np.isfinite(x).all()
                np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-3)
                w, b = fit_head(x, [0, 0, 1, 1], 2)
                z = x @ w + b
                assert np.isfinite(z).all() and z.shape == (4, 2)
                report[mode] = {"runtime": manifest, "feature_shape": list(x.shape),
                                "l2_norms": np.linalg.norm(x, axis=1).tolist(),
                                "training_smoke_logits": z.tolist(), "calibration": "not_fitted",
                                "elapsed_ms_including_four_encodings_and_head_fit": (time.perf_counter() - start)*1000}
            report[mode]["offload_evidence"] = evidence
            print(mode + " interface passed", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
except BaseException as exc:
    report["status"] = "failed"
    report["error"] = str(exc)
    report["server_logs"] = str(log_directory)
    Path(args.out).write_text(dumps(report) + "\n")
    raise
report["status"] = "passed"
report["server_logs"] = str(log_directory)
Path(args.out).write_text(dumps(report) + "\n")
print(json.dumps({"status": "passed", "report": args.out}))
