# Dataset review sample

`public19-1k-validation-v1-sample100.jsonl` contains 100 deterministic,
source-stratified records from the Loop-JEV public19 multi-task suite.

This is intentionally only a schema and task-inspection sample. It is **not**
the full benchmark, not an evaluation-equivalent subset, and must not be used
to compare metrics with full-suite results.

## Composition

| Split | Records |
| --- | ---: |
| train | 39 |
| development | 30 |
| test | 31 |
| total | 100 |

The sample covers all 13 trainable sources. Development and test also cover
the six held-out sources: `mmlu`, `emotion`, `tweet_offensive`, `qnli`, `paws`,
and `sciq`.

Sampling is deterministic: within each split and source, candidate records are
ordered by `sha256(_meta.id)`, then one record per source is selected before
extra rounds.

## File contract

- Encoding: UTF-8.
- Format: one complete labelled request per JSONL line.
- Original record fields, `questions`, `label`, and `_meta` provenance are
  preserved.
- `_meta.sample_partition` identifies the sample source partition: train,
  development, or test.
- `_meta.split` remains the original source-dataset split name, such as
  `train`, `validation`, or `test`.
- The manifest records per-split/source counts and the file SHA-256.

## Provenance

The full suite is `loopjev-public19-1k-validation-v1`, seed `20261104`.
Canonical dataset and tokenizer revisions are recorded in the larger local
study's frozen manifest. The sample is provided for code review only.

The source examples come from public datasets. Their original dataset licenses
and terms continue to apply. Do not infer that this repository relicenses the
underlying text or labels.
