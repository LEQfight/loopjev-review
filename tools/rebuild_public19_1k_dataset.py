"""Rebuild the frozen public19-1k-validation-v1 suite from pinned revisions.

Run on a machine with access to Hugging Face. Set HTTPS_PROXY if needed. The
script verifies every generated split against the canonical manifest before
copying that manifest into the output directory.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def install_windows_fcntl_stub() -> None:
    if os.name != "nt":
        return
    module = types.ModuleType("fcntl")
    module.LOCK_EX = 2
    module.flock = lambda *_args, **_kwargs: None
    sys.modules["fcntl"] = module


def digest(path: Path) -> str:
    import hashlib

    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    canonical_manifest_path = Path(args.canonical_manifest).resolve()
    output = Path(args.out).resolve()
    canonical = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")

    install_windows_fcntl_stub()
    from kev import suite
    from kev.data import ALL_REPOS, ALL_SOURCES, EVAL_ONLY, Source
    from huggingface_hub import HfApi as RealHfApi

    # CPython's C random.Random.__new__ rejects the second constructor argument
    # on some builds before Source.__init__ can consume it.
    Source.__new__ = staticmethod(lambda cls, *_args, **_kwargs: random.Random.__new__(cls))

    dataset_revisions = canonical["dataset_revisions"]
    base_revisions = canonical["base_revisions"]

    class PinnedHfApi(RealHfApi):
        def dataset_info(self, repo, revision=None, **_kwargs):
            resolved = dataset_revisions.get(repo)
            if resolved is None:
                raise ValueError(f"no pinned dataset revision for {repo}")
            return SimpleNamespace(sha=resolved)

        def model_info(self, repo, **_kwargs):
            try:
                return SimpleNamespace(sha=base_revisions[repo])
            except KeyError as error:
                raise ValueError(f"no pinned model revision for {repo}") from error

    source_names = (
        "banking77", "boolq", "agnews", "mnli", "sst5", "yelp", "trec",
        "dbpedia14", "amazon", "imdb", "arc", "openbookqa", "csqa", "mmlu",
        "emotion", "tweet_offensive", "qnli", "paws", "sciq",
    )
    sources = {name: ALL_SOURCES[name] for name in source_names}
    repos = {name: ALL_REPOS[name] for name in source_names}
    holdout = tuple(name for name in source_names if name in EVAL_ONLY)

    with patch("huggingface_hub.HfApi", PinnedHfApi):
        suite.freeze(
            output,
            train=900,
            calibration=100,
            development=80,
            test=80,
            seed=canonical["seed"],
            holdout=holdout,
            sources=sources,
            repos=repos,
        )

    expected = canonical["files"]
    for name, info in expected.items():
        actual = digest(output / name)
        if actual != info["sha256"]:
            raise SystemExit(f"{name}: generated {actual}, expected {info['sha256']}")
        print(f"verified {name}: {actual}")

    generated_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for field in ("trainable_sources", "eval_only_sources", "dataset_revisions", "base_revisions", "admission"):
        if generated_manifest.get(field) != canonical.get(field):
            raise SystemExit(f"manifest field mismatch: {field}")

    shutil.copyfile(canonical_manifest_path, output / "manifest.json")
    print(f"installed canonical manifest: {digest(output / 'manifest.json')}")


if __name__ == "__main__":
    main()
