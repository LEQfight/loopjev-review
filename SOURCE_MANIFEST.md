# Source manifest

This repository is a source-only extraction from the local latest Loop-JEV
workspace at:

`D:\MyProgram\Python\jev_benchmark_engineered\loopjev\kev`

Extraction date: 2026-11-09.

The files below were copied byte-for-byte. Their line numbers therefore match
the local source used to write the review.

## Core source

| File | SHA-256 |
| --- | --- |
| `kev/model.py` | `5859ec9342af095fe42fe62eabe608b06fd75be0e465ad33f1bdf22121e567e4` |
| `kev/loop.py` | `940a24502d1390625543e7947be6130f81220b3fcef8513f1a03cdbb14bb9c3a` |
| `kev/train.py` | `ed6f6f9963366724d0add0d466e2e14c378587c1bbd2586af40af2b0ebe92836` |
| `kev/checkpoint.py` | `1d8de778c79455ad5bfff4a01bb8fc063ace4dbff0b6b6b404ce6c5f8d573d25` |
| `kev/full_ft.py` | `c3abef39cd9d486b3c075a37286afbcd1aa42a04c78a1b7b9b2e77b5a3963929` |
| `kev/api.py` | `a6f53af5354f8ea17915abc78360fdf291a5108ce39a53a2d9135d48c603ae58` |
| `kev/data.py` | `e77a2fffeb8ee7e05b118893be8e38b8358aef97c3ca2e143cdce208201b92f9` |
| `kev/suite.py` | `8e7d257a10b4704d7bf4f8ec2c74dddf26aa5a2a610ea78610360e6baf43eff5` |
| `kev/predictors.py` | `29263ae3297e5cde6cf03da5c5aa3ed377c9681c6a5a639ea4aa259acec7575b` |
| `kev/benchmark.py` | `ff81e91b4db58d58081384018d1b2388c98ef71078591fd06f6ac257f7a3fc3a` |
| `kev/metrics.py` | `167a586334cfcced4adcc197758535c3bcda832b855c151095ef2d1968cf8983` |
| `kev/device.py` | `bb882f59df46eae633d3b29ff0a939c000aa60a7d6d04ae73829b336f9bc9711` |
| `kev/shared_prefix.py` | `bc6dedbad734cca227f3ec6746223b3b03363b0c045aa9f2493d886b1828e71e` |
| `kev/contrastive.py` | `cbb979aa5d40265ad0e64695f94b281d91751fa405811ddfa8212fede111edcf` |
| `kev/composition.py` | `f335ed17e18e0a544893db5e22b9a059e6ce1b2e14dbb863ac7d7bbf8f3e0536` |

## Source tests

| File | SHA-256 |
| --- | --- |
| `tests/test_loop.py` | `da84b53530fbcf609817a1e20d41f26267f7dbab709751a59d3b05f3fd79b2cf` |
| `tests/test_loop_integration.py` | `66723b496608b881a1453708749af7f2fb48170a3a8a43686756238db4fcfa45` |
| `tests/test_multik_loss.py` | `755183af1465139314dc3535179206fd192c572c02e10a3f3ac84a57acbebcbd` |

## Exclusions

The package intentionally excludes model weights, adapters, checkpoints,
training/evaluation artifacts, SwanLab records, raw datasets, launch scripts,
server inventory, historical reports, notebooks, web playgrounds, and
unrelated upstream utilities.

## Upstream context

The source is based on the pinned Kev upstream used by the local study. The
formal experiments discussed in the larger workspace used the official
`jaredpalmer/kev-0.5b@v0.1` checkpoint as initialization, but that experiment
history is outside this review package.
