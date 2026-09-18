"""Fetch a pinned official IBM GGUF; print its content hash for run records."""
import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

REVISION = "ab4701481089b58a082ef63cc1cee738887293ff"
NAME = "granite-4.1-3b-Q4_K_M.gguf"
EXPECTED_SHA256 = "662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29"
URL = f"https://huggingface.co/ibm-granite/granite-4.1-3b-GGUF/resolve/{REVISION}/{NAME}"

parser = argparse.ArgumentParser()
parser.add_argument("--directory", required=True)
parser.add_argument("--reuse", action="store_true", help="Reuse existing weights only after verifying the pinned checksum")
args = parser.parse_args()
destination = Path(args.directory)
destination.mkdir(parents=True, exist_ok=True)
path = destination / NAME
if path.exists():
    if not args.reuse:
        raise SystemExit("Destination already exists; use --reuse to verify it or choose a new directory.")
    with path.open("rb") as existing:
        actual = hashlib.file_digest(existing, "sha256").hexdigest()
    if actual != EXPECTED_SHA256:
        raise SystemExit("Existing weights differ from the pinned model; preserved unchanged.")
    print(f"{actual}  {path} (verified existing file)")
    raise SystemExit(0)
partial = destination / (NAME + ".part")
sha = hashlib.sha256()
try:
    with urllib.request.urlopen(URL, timeout=120) as response, partial.open("xb") as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
            sha.update(chunk)
    if sha.hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("GGUF checksum mismatch; retained .part file for inspection")
    os.rename(partial, path)
except BaseException:
    # A partial download is retained for inspection, never mistaken for a model.
    raise
print(f"{sha.hexdigest()}  {path}")
