import contextlib
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from kev.train import Variant, batch_loss, parse_args, parse_multik_aux_probs, sample_multik_aux_step
from kev.model import DecisionModel


def args(**overrides):
    values = dict(
        shared_prefix=0,
        ord_w=0.0,
        label_smoothing=0.0,
        brier_w=0.0,
        focal_gamma=0.0,
        anchor_w=0.0,
        perm_kl=0.0,
        multik_aux_w=0.0,
        multik_aux_probs=(0.2, 0.3, 0.5),
        seed=20261104,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def variant():
    return Variant(
        rec={"questions": [{"qid": "q0", "qtype": "choice", "options": ["a", "b"], "label": 0}]},
        enc={"ids": [1, 2, 3]},
        request_id="record-0",
        source="unit",
    )


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Parameter(torch.tensor(0.25))
        self.aux = torch.nn.Parameter(torch.tensor(-0.4))
        self.loop_steps = 3
        self.calls = []

    def forward_batch(self, encs, shared_prefix=False, **kwargs):
        loop_steps = kwargs.get("loop_steps")
        self.calls.append(loop_steps)
        score = self.main + (self.aux if loop_steps is not None else 0.0)
        return [[torch.stack((score, -score))] for _ in encs]


def test_disabled_is_the_original_single_forward_and_loss():
    model = TinyModel()
    loss, terms = batch_loss(model, args(), [variant()], "cpu", {}, None, contextlib.nullcontext())
    expected = F.cross_entropy(torch.stack((model.main, -model.main))[None], torch.tensor([0]))
    assert model.calls == [None]
    assert torch.equal(loss, expected)
    assert terms["multik_aux"] == 0


def test_auxiliary_true_k_forward_backpropagates_without_mutating_main_depth():
    model = TinyModel()
    a = args(multik_aux_w=0.2)
    chosen = sample_multik_aux_step(a, [variant()], epoch=1)
    loss, terms = batch_loss(model, a, [variant()], "cpu", {}, None, contextlib.nullcontext(), epoch=1)
    loss.backward()
    assert chosen in (0, 1, 2)
    assert model.calls == [None, chosen]
    assert model.loop_steps == 3
    assert model.main.grad is not None and model.main.grad.abs() > 0
    assert model.aux.grad is not None and model.aux.grad.abs() > 0
    assert terms[f"multik_k{chosen}_n"] == 1
    assert terms["multik_aux_n"] == 1


def test_auxiliary_sampling_is_stable_and_probability_parser_is_strict():
    a = args(multik_aux_probs=(0.0, 0.0, 1.0))
    assert sample_multik_aux_step(a, [variant()], 2) == 2
    assert sample_multik_aux_step(a, [variant()], 2) == 2
    assert parse_multik_aux_probs("0.2,0.3,0.5") == [0.2, 0.3, 0.5]
    for bad in ("1,2", "1,2,3,4", "0,0,0", "1,nan,1", "1,-1,1"):
        with pytest.raises(ValueError):
            parse_multik_aux_probs(bad)


def test_zero_step_override_reaches_the_ordinary_row_path_instead_of_leaking_k3():
    class Shell:
        loop_steps = 3
        hybrid = False

        def rows_form(self, encs):
            return True

        def forward_rows_batch(self, encs, shared_prefix=False, loop_steps=None):
            return loop_steps

    assert DecisionModel.forward_batch(Shell(), [{}], loop_steps=0) == 0


@pytest.mark.parametrize(
    "extra",
    [
        ["--multik_aux_w", "-0.1"],
        ["--multik_aux_w", "0.2"],
        ["--loop_method", "ponder", "--loop_steps", "3", "--ponder_decide_only", "1",
         "--shared_prefix", "0", "--multik_aux_w", "0.2", "--multik_aux_probs", "1,2"],
    ],
)
def test_invalid_cli_combinations_fail_before_training(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(sys, "argv", ["kev.train", "--out", str(tmp_path / "new-run"), *extra])
    with pytest.raises(SystemExit):
        parse_args()

