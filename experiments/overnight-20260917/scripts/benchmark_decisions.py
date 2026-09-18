"""Paired, bounded Jev/Granite evaluation. Jev mode needs only Python stdlib.

Only explicitly supplied rows are sent to TypeSafe. Labels are never sent.
Responses bind to hashes of the complete dataset and questions; compare refuses
missing/duplicate rows or mismatches. API errors stop the run (no hidden retries).
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def distribution(q, answer):
    if answer.get('type') != q['type']:
        raise ValueError('answer type mismatch')
    if q['type'] == 'noul':
        p = answer['noul']
        if type(p) not in (float, int):
            raise ValueError('invalid noul')
        values = [1 - p, p]
    else:
        keys = sorted(q['criteria']) if q['type'] == 'choice' else [str(i) for i in range(len(q['criteria']))]
        probs = answer['probabilities']
        if set(probs) != set(keys):
            raise ValueError('incomplete probability distribution')
        values = [probs[k] for k in keys]
    if any(type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError('invalid probability')
    if not math.isclose(sum(values), 1, abs_tol=1e-5):
        raise ValueError('probabilities do not sum to one')
    return values


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def jev(state, questions, model, key):
    body = canonical({'model': model, 'state': state, 'questions': questions}).encode()
    req = urllib.request.Request('https://api.typesafe.ai/v1/systemone', body,
                                 {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Jev HTTP {exc.code}; stopped without retrying or logging credentials') from None
    except urllib.error.URLError:
        raise RuntimeError('Jev connection failed; stopped without retrying') from None


def inputs(args):
    qs = json.loads(Path(args.questions).read_text())
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('empty dataset or duplicate ids')
    if len({digest(r['state']) for r in rows}) != len(rows):
        raise ValueError('duplicate states')
    if any(set(r['labels']) != set(qs) for r in rows):
        raise ValueError('label fields mismatch')
    return qs, rows, {'questions_sha256': digest(qs), 'dataset_sha256': digest(rows)}


def collect(args):
    qs, rows, identity = inputs(args)
    if len(rows) > args.max_requests:
        raise ValueError('dataset exceeds --max-requests; no requests sent')
    if Path(args.out).exists():
        raise ValueError('output exists; choose a new path')
    if args.mode == 'jev':
        key = Path(args.key_file).read_text().strip() if args.key_file else os.environ.get('TYPESAFE_API_KEY', '')
        if not key or '\n' in key:
            raise ValueError('missing or invalid Jev key')
        call = lambda state: jev(state, qs, args.model, key)
    else:
        from granite_decisions.llamacpp import NativeClient, LlamaRuntime, BaselineBackend
        from granite_decisions.training import HeadBackend, read_dataset
        from granite_decisions.runtime import DecisionEngine
        read_dataset(args.data, qs)
        runtime = LlamaRuntime(NativeClient(args.url, timeout=120), json.loads(Path(args.runtime).read_text()))
        engine = DecisionEngine(HeadBackend(runtime, args.bundle) if args.bundle else BaselineBackend(runtime))
        call = lambda state: engine.evaluate(state, qs)
    with open(args.out, 'x') as handle:
        for row in rows:
            start = time.perf_counter()
            result = call(row['state'])
            elapsed = (time.perf_counter() - start) * 1000
            if set(result['answers']) != set(qs):
                raise ValueError('answer fields mismatch')
            for name, q in qs.items():
                distribution(q, result['answers'][name])
            record = {**identity, 'id': row['id'], 'state_sha256': digest(row['state']),
                      'provider': args.mode, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      'elapsed_ms': elapsed, 'response': result}
            handle.write(canonical(record) + '\n')
            handle.flush()
            print(f"{args.mode}: {row['id']} ({elapsed:.0f} ms)", flush=True)


def compare(args):
    import numpy as np
    from granite_decisions.contracts import label_index
    from granite_decisions.statistics import metrics
    qs, rows, identity = inputs(args)
    report = {'basis': 'supplied_labels_not_jev_as_ground_truth', **identity, 'n': len(rows),
              'latency_scope': 'end_to_end_client_wall_time; first request separate; network included for Jev',
              'dataset_kind': getattr(args, 'dataset_kind', 'unspecified'), 'providers': {}}
    predictions = {}
    for path in args.results:
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        by_id = {r['id']: r for r in records}
        if len(records) != len(by_id) or set(by_id) != {r['id'] for r in rows}:
            raise ValueError('incomplete, extra, or duplicate result rows')
        for row in rows:
            r = by_id[row['id']]
            if any(r[k] != v for k, v in identity.items()) or r['state_sha256'] != digest(row['state']):
                raise ValueError('dataset or question identity mismatch')
        name = Path(path).stem
        if name in report['providers']:
            raise ValueError('result filenames must have distinct stems')
        times = [by_id[r['id']]['elapsed_ms'] for r in rows]
        fields = {}
        predictions[name] = {}
        for key, q in qs.items():
            p = np.array([distribution(q, by_id[r['id']]['response']['answers'][key]) for r in rows])
            y = np.array([label_index(q, r['labels'][key]) for r in rows])
            pred = p.argmax(axis=1)
            predictions[name][key] = pred
            fields[key] = metrics(np.log(np.maximum(p, np.finfo(float).tiny)), y)
            fields[key]['test_majority_class_accuracy_reference'] = float(np.bincount(y).max()/len(y))
            if q['type'] == 'score':
                fields[key]['expected_score_mae'] = float(np.mean(abs(p @ np.arange(p.shape[1]) - y)))
        report['providers'][name] = {'fields': fields, 'models': sorted({r['response']['model'] for r in records}),
            'first_request_ms': times[0], 'all_requests_p50_ms': float(np.median(times)),
            'warm_p50_ms': float(np.median(times[1:])) if len(times) > 1 else None,
            'p95_ms': float(np.percentile(times,95)),
            'usage': {k: sum(r['response'].get('usage',{}).get(k,0) for r in records) for k in ('input_tokens','output_tokens')}}
    report['pairwise_agreement'] = {a + '__' + b: {k: float(np.mean(predictions[a][k] == predictions[b][k])) for k in qs}
        for i,a in enumerate(predictions) for b in list(predictions)[i+1:]}
    with open(args.out, 'x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['jev','native','compare'])
    parser.add_argument('--questions', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--key-file', help='Read on the credential host; never copied to output')
    parser.add_argument('--model', default='jev-latest')
    parser.add_argument('--max-requests', type=int, default=60)
    parser.add_argument('--url', default='http://127.0.0.1:8091')
    parser.add_argument('--runtime')
    parser.add_argument('--bundle')
    parser.add_argument('--results', nargs='+')
    parser.add_argument('--dataset-kind', choices=['synthetic','reviewed','unspecified'], default='unspecified')
    args = parser.parse_args()
    if args.mode == 'native' and not args.runtime:
        parser.error('native mode requires --runtime')
    if args.mode == 'compare' and not args.results:
        parser.error('compare mode requires --results')
    try:
        compare(args) if args.mode == 'compare' else collect(args)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(1, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
