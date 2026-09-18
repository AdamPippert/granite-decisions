"""Fetch cited open data; build deterministic, group-disjoint experiment splits."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import urllib.request

SPLITS = ('train', 'validation', 'calibration', 'test')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(text):
    return ' '.join(text.casefold().split())


def make_synthetic():
    """Original MIT fixtures; thresholds, wording, and states vary across splits."""
    result = {s: [] for s in SPLITS}
    templates = {
        'train': 'Assign the level containing the observed count.',
        'validation': 'Which level includes the count in the supplied record?',
        'calibration': 'Select the level whose interval contains the observed count.',
        'test': 'Using only the interval definitions, classify this count into a level.',
    }
    seen = set()
    for split, size in zip(SPLITS, (2048, 256, 256, 256)):
        rng = random.Random('granite-score-v1-' + split)
        while len(result[split]) < size:
            levels = rng.randint(3, 7)
            cuts = sorted(rng.sample(range(1, 1000), levels - 1))
            value = rng.randint(0, 1000)
            # Group by the entire rubric, rather than only a (value,rubric) pair.
            group = sha(json.dumps(cuts).encode())
            if group in seen:
                continue
            seen.add(group)
            criteria = [f'Count less than {cuts[0]}.']
            criteria += [f'Count at least {lo} and less than {hi}.' for lo, hi in zip(cuts, cuts[1:])]
            criteria += [f'Count at least {cuts[-1]}.']
            result[split].append(dict(id=f'score-{split}-{len(result[split])}',
                family='synthetic_score', group=group, state={'observed_count': value},
                question={'type':'score', 'instructions':templates[split], 'criteria':criteria},
                label=sum(value >= t for t in cuts), source='original-synthetic-score-v1',
                license='MIT'))
    return result


def validate_splits(data):
    ids, groups, examples, passages = set(), {}, {}, {}
    for split, rows in data.items():
        for r in rows:
            if r['id'] in ids:
                raise ValueError('duplicate record ID')
            ids.add(r['id'])
            key = (r['family'], r['group'])
            if key in groups and groups[key] != split:
                raise ValueError('cross-split source group leakage')
            groups[key] = split
            passage = r.get('passage_group')
            if passage in passages and passages[passage] != split:
                raise ValueError('cross-split passage leakage')
            if passage: passages[passage] = split
            content = sha(json.dumps([r['state'],r['question']],sort_keys=True).encode())
            if content in examples:
                raise ValueError('duplicate decision example')
            examples[content] = split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--source-cache', help='Previously downloaded immutable source files')
    p.add_argument('--synthetic-only', action='store_true', help='Generate original MIT Score records without downloads')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if args.synthetic_only:
        for split, rows in make_synthetic().items():
            (out/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return
    (out/'sources').mkdir()
    locks = json.loads((Path(__file__).resolve().parents[1]/'source-lock.json').read_text())
    sources = []

    def fetch(name, url, license, citation):
        cache=Path(args.source_cache)/name if args.source_cache else None
        raw = cache.read_bytes() if cache and cache.is_file() else urllib.request.urlopen(url, timeout=90).read()
        (out/'sources'/name).write_bytes(raw)
        sources.append(dict(file='sources/'+name, url=url, sha256=sha(raw),
                            license=license, citation=citation))
        return raw

    bank_repo = 'PolyAI-LDN/task-specific-datasets'
    bank_base = f'https://raw.githubusercontent.com/{bank_repo}/{locks[bank_repo]}/'
    bank_cite = 'Casanueva et al. (2020), Efficient Intent Detection with Dual Sentence Encoders, https://arxiv.org/abs/2003.04807'
    fetch('banking-LICENSE', bank_base+'LICENSE', 'CC-BY-4.0', bank_cite)
    fetch('banking-README.md', bank_base+'README.md', 'CC-BY-4.0', bank_cite)
    banks = {s:list(csv.DictReader(io.StringIO(fetch('banking-'+s+'.csv',
        bank_base+'banking_data/'+s+'.csv','CC-BY-4.0',bank_cite).decode()))) for s in ('train','test')}
    names = sorted({r['category'] for r in banks['train']})
    assert len(names) == 77
    q = dict(type='choice',instructions='Identify the banking intent expressed by the customer request.',
             criteria={name:name.replace('_',' ') for name in names})
    data = {s:[] for s in SPLITS}
    used = set()
    # Reserve official test examples before sampling any training-source split.
    reserved = {normalized(r['text']) for r in banks['test']}
    for label in names:
        candidates = []
        for i,r in enumerate(banks['train']):
            key = normalized(r['text'])
            if r['category'] == label and key not in reserved and key not in used:
                used.add(key); candidates.append((i,r))
        random.Random('bank-'+label).shuffle(candidates)
        for split, chunk in [('validation',candidates[:2]),('calibration',candidates[2:4]),('train',candidates[4:44])]:
            for i,r in chunk:
                data[split].append(dict(id=f'banking-train-{i}',family='banking77',
                    group=sha(normalized(r['text']).encode()),state={'request':r['text']},question=q,
                    label=label,source='sources/banking-train.csv',source_row=i,license='CC-BY-4.0'))
        tests = [(i,r) for i,r in enumerate(banks['test']) if r['category']==label]
        random.Random('bank-test-'+label).shuffle(tests)
        selected=0
        for i,r in tests:
            key=normalized(r['text'])
            if key in used: continue
            used.add(key)
            data['test'].append(dict(id=f'banking-test-{i}',family='banking77',
                group=sha(key.encode()),state={'request':r['text']},question=q,label=label,
                source='sources/banking-test.csv',source_row=i,license='CC-BY-4.0'))
            selected+=1
            if selected==2: break

    bool_repo='google-research-datasets/boolean-questions'
    bool_cite='Clark et al. (2019), BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions, https://arxiv.org/abs/1905.10044'
    fetch('boolq-README.md',f'https://raw.githubusercontent.com/{bool_repo}/{locks[bool_repo]}/README.md','CC-BY-SA-3.0',bool_cite)
    import pyarrow.parquet as pq
    bool_revision='35b264d03638db9f4ce671b711558bf7ff0f80d5'
    bools={}
    for source, hf_split in [('train','train'),('dev','validation')]:
        raw=fetch('boolq-'+source+'.parquet',
            f'https://huggingface.co/datasets/google/boolq/resolve/{bool_revision}/data/{hf_split}-00000-of-00001.parquet',
            'CC-BY-SA-3.0',bool_cite)
        bools[source]=pq.read_table(io.BytesIO(raw)).to_pylist()
        # Official HF mirror omits titles; use exact normalized passages as groups.
        for row in bools[source]: row['title']=row['passage']
    reserved_titles={normalized(r['title']) for r in bools['dev']}
    reserved_passages={normalized(r['passage']) for r in bools['dev']}
    seen_examples=set()
    for source in ('train','dev'):
        rows=list(enumerate(bools[source]));random.Random('boolq-'+source).shuffle(rows)
        counts={s:0 for s in SPLITS}
        limits={'train':2048,'validation':256,'calibration':256,'test':256}
        for i,r in rows:
            title=normalized(r['title']); passage=normalized(r['passage'])
            if source=='train' and (title in reserved_titles or passage in reserved_passages): continue
            # Assign all occurrences of a title to one split by a deterministic hash.
            bucket=int(sha(title.encode())[:8],16)%10
            split='test' if source=='dev' else ('validation' if bucket==0 else 'calibration' if bucket==1 else 'train')
            if counts[split]>=limits[split]: continue
            key=sha(json.dumps([r['question'],passage]).encode())
            if key in seen_examples: continue
            seen_examples.add(key)
            data[split].append(dict(id=f'boolq-{source}-{i}',family='boolq',group=sha(title.encode()),
                passage_group=sha(passage.encode()),state={'passage':r['passage']},
                question={'type':'noul','instructions':'Answer this yes/no question using the supplied passage: '+r['question']},
                label=r['answer'],source='sources/boolq-'+source+'.parquet',source_row=i,license='CC-BY-SA-3.0'))
            counts[split]+=1
    # Drop cross-title repeated passages if a passage has already been assigned elsewhere.
    passage_splits={}
    for split in reversed(SPLITS):
        clean=[]
        for r in data[split]:
            pg=r.get('passage_group')
            if pg and pg in passage_splits and passage_splits[pg]!=split: continue
            if pg: passage_splits[pg]=split
            clean.append(r)
        data[split]=clean
    synthetic=make_synthetic()
    for split in SPLITS:
        data[split]+=synthetic[split]
    validate_splits(data)
    from collections import Counter
    counts={s:dict(Counter(r['family'] for r in rows)) for s,rows in data.items()}
    for split in SPLITS:
        assert counts[split]['banking77']>=(2300 if split=='train' else 154),counts
        per_label=Counter(r['label'] for r in data[split] if r['family']=='banking77')
        assert len(per_label)==77 and min(per_label.values())>=(20 if split=='train' else 2),per_label
        assert counts[split]['boolq']>=200,counts
        (out/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data[split]))
    (out/'synthetic').mkdir()
    for split,rows in synthetic.items():
        (out/'synthetic'/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest={'format':1,'sources':sources,'upstream_commits':locks,'counts':counts,
        'split_policy':'BANKING77 stratified holdouts from official train; official test reserved. BoolQ normalized-passage-hash train splits (HF mirror omits titles); official dev reserved as test; cross-split passages removed. Synthetic rubric groups disjoint.',
        'files':{s+'.jsonl':sha((out/(s+'.jsonl')).read_bytes()) for s in SPLITS},
        'licenses':{'banking77':'CC-BY-4.0','boolq':'CC-BY-SA-3.0','synthetic_score':'MIT'},
        'pretraining_contamination':'Unknown; upstream model may have seen these public benchmarks.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
