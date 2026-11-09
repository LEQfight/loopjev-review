"""Executable contracts for issues documented in README.md.

These tests do not modify the extracted implementation. They pin the expected
checkpoint-loading contract and make the current mismatch visible without
blocking collection.
"""
from types import SimpleNamespace

import pytest

from kev.checkpoint import Checkpoint, LoadOptions, Meta


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known finding: Checkpoint._full_torch does not pass meta.base_revision "
        "to DecisionModel, so a full-weight Ponder checkpoint reloads its frozen "
        "LM head from an unpinned base revision."
    ),
)
def test_full_weight_ponder_pins_base_revision(monkeypatch):
    checkpoint = Checkpoint.__new__(Checkpoint)
    checkpoint.path = "saved-full-weight-checkpoint"
    checkpoint.meta = Meta(
        base="Qwen/Qwen2.5-0.5B",
        base_revision="PINNED_REVISION",
        weights="full",
        loop_method="ponder",
        loop_steps=2,
        ponder_top_k=7,
    )

    seen = {}

    def fake_decision_model(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(checkpoint, "saved_dtype", lambda: "float32")
    monkeypatch.setattr("kev.checkpoint.DecisionModel", fake_decision_model)
    checkpoint._full_torch(None, "cpu", LoadOptions())

    assert seen["revision"] == "PINNED_REVISION"
