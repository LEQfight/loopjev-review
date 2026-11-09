"""Create a deterministic, source-stratified 100-record suite sample."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


SPLIT_QUOTAS = {"train": 39, "development": 30, "test": 31}


def stable_order(record):
    return hashlib.sha256(record["_meta"]["id"].encode("utf-8")).hexdigest()


def allocate_sources(sources, quota):
    """One record per source, then a stable extra pass, for a fixed total."""
    if quota < len(sources):
        raise ValueError(f"quota {quota} cannot cover {len(sources)} sources")
    counts = {source: 1 for source in sources}
    remaining = quota - len(sources)
    for source in sources:
        if remaining <= 0:
            break
        counts[source] += 1
        remaining -= 1
    index = 0
    while remaining > 0:
        counts[sources[index % len(sources)]] += 1
        remaining -= 1
        index += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    suite = Path(args.suite).resolve()
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    selected = []
    split_counts = {}
    source_counts = {}
    for split, quota in SPLIT_QUOTAS.items():
        records = [
            json.loads(line)
            for line in (suite / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_source = collections.defaultdict(list)
        for record in records:
            by_source[record["_meta"]["source"]].append(record)
        sources = sorted(by_source)
        allocations = allocate_sources(sources, quota)

        chosen = []
        for source in sources:
            ordered = sorted(by_source[source], key=stable_order)
            records_for_source = ordered[: allocations[source]]
            for record in records_for_source:
                record["_meta"]["sample_partition"] = split
            chosen.extend(records_for_source)
        if len(chosen) != quota:
            raise AssertionError((split, len(chosen), quota))

        selected.extend(chosen)
        split_counts[split] = len(chosen)
        source_counts[split] = dict(allocations)

    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in selected),
        encoding="utf-8",
        newline="\n",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "name": "public19-1k-validation-v1-sample100",
        "kind": "review_sample",
        "warning": (
            "This is a small stratified sample for schema and task inspection. "
            "It is not the complete benchmark, not an evaluation-equivalent subset, "
            "and its metrics must not be compared with full-suite results."
        ),
        "records": len(selected),
        "questions": sum(len(record["questions"]) for record in selected),
        "split_counts": split_counts,
        "source_counts": source_counts,
        "sampling": (
            "Source-stratified deterministic selection: candidates are sorted by "
            "sha256(_meta.id), with one record per source before extra rounds."
        ),
        "sha256": digest,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
