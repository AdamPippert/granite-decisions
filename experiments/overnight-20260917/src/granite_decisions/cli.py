import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from .contracts import ContractError, digest, dumps, label_index, labels, loads, questions
from .journal import Journal
from .llamacpp import BackendError, BaselineBackend, LlamaRuntime, MODEL_ID, NativeClient, bind_runtime
from .runtime import DecisionEngine, Policy
from .statistics import metrics
from .training import HeadBackend, read_dataset, train_bundle


def read_json(path):
    return loads(Path(path).read_bytes())


def write_new(path, value):
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(dumps(value) + "\n")


def native(args):
    return LlamaRuntime(NativeClient(args.url, args.timeout, os.environ.get("LLAMA_API_KEY")), read_json(args.runtime))


def make_engine(args):
    model = native(args)
    backend = HeadBackend(model, args.bundle) if args.bundle else BaselineBackend(model)
    policy = Policy(args.threshold, args.margin, args.allow_uncalibrated)
    journal = Journal(args.journal, args.record_state) if args.journal else None
    return DecisionEngine(backend, policy, journal)


def add_native(parser):
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--timeout", type=float, default=60)


def add_engine(parser):
    add_native(parser)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--bundle")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--allow-uncalibrated", action="store_true")
    parser.add_argument("--journal")
    parser.add_argument("--record-state", action="store_true")


def evaluate_dataset(engine, rows, qs):
    logits = {k: [] for k in qs}
    timings, blocked = [], []
    for row in rows:
        start = time.perf_counter()
        result = engine.evaluate(row["state"], qs)
        timings.append((time.perf_counter() - start) * 1000)
        blocked.append(result["status"] == "abstained")
        for key, q in qs.items():
            answer = result["answers"][key]
            p = ([1 - answer["noul"], answer["noul"]] if q["type"] == "noul" else [answer["probabilities"][k] for k in labels(q)])
            logits[key].append(np.log(np.maximum(p, np.finfo(float).tiny)))
    reports = {}
    for key, q in qs.items():
        y = [label_index(q, r["labels"][key]) for r in rows]
        reports[key] = metrics(logits[key], y, threshold=engine.policy.min_probability, min_margin=engine.policy.min_margin)
        reports[key]["majority_class_accuracy"] = max(np.bincount(y)) / len(y)
        reports[key]["uniform_nll"] = float(np.log(len(labels(q))))
    seen = set(getattr(engine.backend, "manifest", {}).get("seen_state_hashes", []))
    overlap = sum(digest(r["state"]) in seen for r in rows)
    return {"n": len(rows), "backend": engine.backend.identity, "policy": engine.policy.as_dict(),
            "calibration": engine.backend.calibration, "fields": reports,
            "whole_request_coverage": 1 - sum(blocked)/len(blocked),
            "latency_ms": {"p50": float(np.percentile(timings, 50)), "p95": float(np.percentile(timings, 95)), "max": max(timings)},
            "overlap_with_training_bundle_data": overlap, "independent_of_bundle_data": overlap == 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local Granite 4.1 decisions on native llama.cpp")
    sub = parser.add_subparsers(dest="command", required=True)
    bind = sub.add_parser("bind", help="On server host, bind GGUF and server identity")
    add_native(bind)
    bind.add_argument("--gguf", required=True)
    bind.add_argument("--model-id", default=MODEL_ID)
    bind.add_argument("--mode", choices=["embedding", "baseline"], required=True)
    bind.add_argument("--max-tokens", type=int, default=2048)
    bind.add_argument("--out", required=True)
    infer = sub.add_parser("infer")
    add_engine(infer)
    infer.add_argument("--request", default="-")
    infer.add_argument("--jsonl", action="store_true", help="One request and response per line on stdin")
    train = sub.add_parser("train")
    add_native(train)
    train.add_argument("--runtime", required=True)
    train.add_argument("--questions", required=True)
    for name in ("train", "calibration", "test", "out"):
        train.add_argument("--" + name, required=True)
    train.add_argument("--l2", type=float, default=0.01)
    evaluate = sub.add_parser("evaluate")
    add_engine(evaluate)
    evaluate.add_argument("--questions", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--out", required=True)
    serve = sub.add_parser("serve")
    add_engine(serve)
    serve.add_argument("--port", type=int, default=8093)
    feedback = sub.add_parser("feedback")
    feedback.add_argument("--journal", required=True)
    feedback.add_argument("--event", required=True)
    export = sub.add_parser("export-feedback")
    export.add_argument("--journal", required=True)
    export.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "bind":
            result = bind_runtime(NativeClient(args.url, args.timeout, os.environ.get("LLAMA_API_KEY")), args.gguf, args.model_id, args.mode, args.max_tokens)
            write_new(args.out, result)
        elif args.command == "train":
            qs = questions(read_json(args.questions))
            splits = [read_dataset(getattr(args, name), qs) for name in ("train", "calibration", "test")]
            result = train_bundle(native(args), qs, *splits, args.out, args.l2)
        elif args.command == "feedback":
            event = read_json(args.event)
            if type(event) is not dict or set(event) != {"decision_id", "verdict", "labels", "reviewer", "note"}:
                raise ContractError("invalid_feedback_event_fields")
            result = Journal(args.journal).feedback(**event)
        elif args.command == "export-feedback":
            records = Journal(args.journal).export_verified()
            with open(args.out, "x") as handle:
                for record in records:
                    handle.write(dumps(record) + "\n")
            result = {"exported": len(records), "out": args.out}
        else:
            if args.record_state and not args.journal:
                raise ContractError("record_state_requires_journal")
            engine = make_engine(args)
            if args.command == "serve":
                import uvicorn
                from .service import create_app
                uvicorn.run(create_app(engine, os.environ.get("GRANITE_DECISIONS_API_KEY")), host="127.0.0.1", port=args.port)
                return 0
            if args.command == "evaluate":
                qs = questions(read_json(args.questions))
                result = evaluate_dataset(engine, read_dataset(args.data, qs), qs)
                write_new(args.out, result)
            elif args.jsonl:
                if args.request != "-":
                    raise ContractError("jsonl_uses_stdin")
                # readline is bounded so an unbroken huge input cannot consume RAM.
                while raw := sys.stdin.buffer.readline(262_146):
                    if len(raw) > 262_144:
                        raise ContractError("jsonl_line_too_large")
                    try:
                        body = loads(raw)
                        item = engine.evaluate_request(body)
                    except (ContractError, BackendError) as exc:
                        item = {"error": str(exc)}
                    print(dumps(item), flush=True)
                return 0
            else:
                body = loads(sys.stdin.buffer.read(262_145)) if args.request == "-" else read_json(args.request)
                result = engine.evaluate_request(body)
        print(dumps(result))
        return 0
    except (ContractError, BackendError) as exc:
        print(dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    except OSError as exc:
        print(dumps({"error": "filesystem_or_socket_error", "kind": type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
