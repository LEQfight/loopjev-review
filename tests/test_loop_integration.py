"""Weight-free tests for the LoopJev integration in Kev's model, training and checkpoint path.

Run: uv run python -m pytest tests/test_loop_integration.py -q
"""
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kev.checkpoint import Checkpoint, LoadOptions, Meta, read_meta, write_meta
from kev.model import SPECIAL, DecisionModel, rows_of

DIM, VOCAB = 8, 64


class CharTokenizer:
    """A deterministic character tokenizer: one id per character, plus the five Kev delimiters."""
    pad_token_id = 0

    def __init__(self):
        self.specials = {token: i + 1 for i, token in enumerate(SPECIAL)}

    def convert_tokens_to_ids(self, token):
        return self.specials.get(token, 0)

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[10 + (ord(c) % 40) for c in text] or [10])


class TinyBackbone(nn.Module):
    """A deterministic attention-only causal backbone that records every call's shape, positions and mask."""

    def __init__(self, hybrid=False):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.proj = nn.Linear(DIM, DIM, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.linspace(-0.9, 1.1, VOCAB * DIM).reshape(VOCAB, DIM))
            self.proj.weight.copy_(torch.eye(DIM) * 0.6 + torch.linspace(-0.2, 0.2, DIM * DIM).reshape(DIM, DIM))
        self.config = SimpleNamespace(hidden_size=DIM, layer_types=["linear_attention"] if hybrid else ["full_attention"])
        self.calls, self.lengths, self.positions, self.masks, self.hidden = 0, [], [], [], []
        self.embeds = []

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, attention_mask=None):
        assert (input_ids is None) != (inputs_embeds is None)
        assert position_ids is not None
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        self.calls += 1
        self.lengths.append(inputs_embeds.shape[1])
        self.positions.append(position_ids.detach().clone())
        self.masks.append(None if attention_mask is None else attention_mask.detach().clone())
        self.embeds.append(inputs_embeds.detach().clone())
        length = inputs_embeds.shape[1]
        causal = torch.tril(torch.ones(length, length, dtype=inputs_embeds.dtype))
        denom = torch.arange(1, length + 1, dtype=inputs_embeds.dtype)
        pooled = torch.einsum("ij,bjd->bid", causal, inputs_embeds) / denom[None, :, None]
        pos = position_ids.to(inputs_embeds.dtype).unsqueeze(-1)
        hidden = self.proj(inputs_embeds + 0.15 * pooled + 0.02 * pos)
        self.hidden.append(hidden.detach().clone())
        return SimpleNamespace(last_hidden_state=hidden)


def install_loader(monkeypatch, backbone, lm_head):
    """Patch the one causal-LM loader the constructor uses; returns the calls it made."""
    calls = []

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(model=backbone, lm_head=lm_head)

    monkeypatch.setattr("kev.model.AutoModelForCausalLM.from_pretrained", loader)
    return calls


def build(monkeypatch, method="kev", steps=0, hybrid=False, ponder_top_k=100, **ponder_options):
    tok = CharTokenizer()
    backbone = TinyBackbone(hybrid=hybrid)
    lm_head = nn.Linear(DIM, VOCAB, bias=False)
    calls = install_loader(monkeypatch, backbone, lm_head)
    model = DecisionModel("fake/tiny", tok, "cpu", head_dim=DIM, loop_method=method, loop_steps=steps,
                          ponder_top_k=ponder_top_k, **ponder_options)
    return model, tok, backbone, lm_head, calls


def make_record(questions, state="state"):
    return {"state": state,
            "questions": [{"instr": instr, "options": list(options), "label": 0} for instr, options in questions]}


def grad_reached(parameters):
    grads = [parameter.grad for parameter in parameters]
    return (bool(grads) and all(grad is not None and bool(torch.isfinite(grad).all()) for grad in grads)
            and any(bool(grad.abs().sum() > 0) for grad in grads))


@pytest.mark.parametrize("method", ["coconut", "ponder"])
def test_positive_steps_score_rows_and_carry_gradients(monkeypatch, method):
    steps = 2
    model, tok, backbone, lm_head, _ = build(monkeypatch, method, steps)
    recs = [make_record([("choose one", ["alpha", "beta", "gamma"]), ("pick", ["a", "b"])]),
            make_record([("select", ["x", "y", "z", "w"])], state="other state")]
    encs = [model.encode(tok, rec) for rec in recs]
    out = model.forward_batch(encs)

    # Nested logits with each question's own option count.
    assert [len(zs) for zs in out] == [len(enc["decide_idx"]) for enc in encs]
    for enc, zs in zip(encs, out):
        assert len(zs) == len(enc["opt_idx"])
        for option_idx, z in zip(enc["opt_idx"], zs):
            assert z.shape == (len(option_idx),)
            assert bool(torch.isfinite(z).all())

    # Every question is one causal row; each arm makes steps + 1 passes per row.
    row_specs, expected_lengths = [], []
    for enc in encs:
        state_ids, state_pos, rows = rows_of(enc)
        for row in rows:
            length = len(state_ids) + len(row["ids"])
            row_specs.append((state_ids, state_pos, row, length))
            expected_lengths.extend([length + i for i in range(steps + 1)] if method == "coconut" else [length] * (steps + 1))
    assert backbone.calls == len(row_specs) * (steps + 1)
    assert backbone.lengths == expected_lengths
    assert all(mask is not None and bool(torch.all(mask == 1)) for mask in backbone.masks)

    # Exact rows: state + branch, the existing positions, no block mask.
    for i, (state_ids, state_pos, row, _) in enumerate(row_specs):
        expected_positions = state_pos + row["pos"]
        first = backbone.positions[i * (steps + 1)][0].tolist()
        assert first[:len(expected_positions)] == expected_positions
        if method == "ponder":
            for j in range(1, steps + 1):
                assert backbone.positions[i * (steps + 1) + j][0].tolist() == expected_positions

    # Which pass each arm reads: coconut takes option ends from the ordinary first pass and the latent as query;
    # ponder takes both from the final pass.
    row_index = 0
    for enc, zs in zip(encs, out):
        state_ids, _, rows = rows_of(enc)
        for row, z in zip(rows, zs):
            base = row_index * (steps + 1)
            option_idx = torch.tensor([len(state_ids) + o for o in row["opts"]])
            if method == "coconut":
                first = backbone.hidden[base][0].float()
                final = backbone.hidden[base + steps][0].float()
                expected = model.head(final[-1], first[option_idx])
            else:
                final = backbone.hidden[base + steps][0].float()
                expected = model.head(final[len(state_ids) + row["decide"]], final[option_idx])
            assert torch.allclose(z, expected, atol=1e-6)
            row_index += 1

    model.zero_grad()
    sum(z.square().sum() for zs in out for z in zs).backward()
    assert grad_reached(list(backbone.parameters()))
    assert grad_reached(list(model.head.parameters()))
    if method == "ponder":
        assert all(parameter.grad is None for parameter in lm_head.parameters())   # the LM head is frozen


def test_zero_steps_delegates_to_the_existing_forward(monkeypatch):
    model, tok, backbone, _, _ = build(monkeypatch, "kev", 0)
    encs = [model.encode(tok, make_record([("choose one", ["alpha", "beta"]), ("pick", ["a", "b"])]))]
    baseline = model.forward_batch(encs)

    def boom(*args, **kwargs):
        raise AssertionError("the positive-step loop path must not run at steps=0")

    monkeypatch.setattr(model, "_loop_forward_batch", boom)
    for method in ("coconut", "ponder"):
        model.loop_method, model.loop_steps = method, 0
        before = backbone.calls
        out = model.forward_batch(encs)
        assert backbone.calls == before + 1
        assert all(torch.equal(a, b) for x, y in zip(baseline, out) for a, b in zip(x, y))
    assert getattr(model, "lm_head", None) is None   # ponder's zero-step path needs no vocabulary head

    rows = model.forward_rows_batch(encs)   # the row form delegates too
    model.loop_method, model.loop_steps = "kev", 0
    again = model.forward_rows_batch(encs)
    assert all(torch.equal(a, b) for x, y in zip(rows, again) for a, b in zip(x, y))


def test_ponder_loads_the_causal_lm_once_and_freezes_the_lm_head(monkeypatch):
    model, _, backbone, lm_head, calls = build(monkeypatch, "ponder", 1)
    assert len(calls) == 1 and calls[0][0][0] == "fake/tiny"
    assert model.lm is backbone and model.lm_head is lm_head
    assert model.ponder_top_k == 100
    assert all(not parameter.requires_grad for parameter in model.lm_head.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())


@pytest.mark.parametrize("top_k", [1, 3, 100])
def test_positive_ponder_passes_configured_top_k_to_the_recurrence(monkeypatch, top_k):
    import kev.loop as loop

    seen = []
    real = loop.ponder_forward

    def recording(*args, **kwargs):
        seen.append(kwargs.get("top_k"))
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "ponder_forward", recording)
    model, tok, _, _, _ = build(monkeypatch, "ponder", 1, ponder_top_k=top_k)
    assert model.ponder_top_k == top_k
    enc = model.encode(tok, make_record([("pick", ["a", "b"])]))
    model.forward_batch([enc])
    assert seen and all(value == top_k for value in seen)

    # A LoopConfig carries the same bound into the helper.
    config = loop.LoopConfig("ponder", 1, top_k)
    assert config.ponder_top_k == top_k and config.top_k == top_k


def test_positive_ponder_defaults_keep_legacy_recurrence_settings(monkeypatch):
    model, _, _, _, _ = build(monkeypatch, "ponder", 1)
    assert model.ponder_top_k == 100
    assert (model.ponder_scale_embeds, model.ponder_multiply_steps,
            model.ponder_full_vocab, model.ponder_preserve_options,
            model.ponder_decide_only, model.ponder_option_decide_only) == (False, False, False, False, False, False)


def test_positive_ponder_passes_official_style_settings_to_the_recurrence(monkeypatch):
    import kev.loop as loop

    seen = []
    real = loop.ponder_forward

    def recording(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "ponder_forward", recording)
    model, tok, _, _, _ = build(monkeypatch, "ponder", 2, ponder_full_vocab=True,
                                ponder_scale_embeds=True, ponder_multiply_steps=True)
    assert model.ponder_top_k is None
    assert (model.ponder_scale_embeds, model.ponder_multiply_steps,
            model.ponder_full_vocab) == (True, True, True)
    enc = model.encode(tok, make_record([("pick", ["a", "b"])]))
    model.forward_batch([enc])
    assert seen
    assert all(call["top_k"] is None and call["ponder_full_vocab"] is True
               and call["ponder_scale_embeds"] is True and call["ponder_multiply_steps"] is True
               for call in seen)


def test_positive_ponder_accepts_top_k_none_as_full_vocabulary(monkeypatch):
    model, _, _, _, _ = build(monkeypatch, "ponder", 1, ponder_top_k=None)
    assert model.ponder_top_k is None and model.ponder_full_vocab is True


def test_positive_ponder_preserve_options_adds_one_ordinary_pass(monkeypatch):
    model, tok, backbone, _, _ = build(monkeypatch, "ponder", 2, ponder_preserve_options=True)
    enc = model.encode(tok, make_record([("pick", ["a", "b"])]))
    model.forward_batch([enc])
    assert model.ponder_preserve_options is True
    assert backbone.calls == 4   # one ordinary option pass + two ponder updates + final pass


def test_positive_ponder_decide_only_uses_ordinary_options_and_final_decide_query(monkeypatch):
    model, tok, backbone, _, _ = build(monkeypatch, "ponder", 2, ponder_decide_only=True)
    enc = model.encode(tok, make_record([("pick", ["alpha", "beta", "gamma"])]))
    out = model.forward_batch([enc])

    state_ids, _, rows = rows_of(enc)
    row = rows[0]
    assert model.ponder_decide_only is True
    assert backbone.calls == 4   # ordinary pass + two decide-only updates + final pass
    ordinary = backbone.hidden[0][0].float()
    final = backbone.hidden[3][0].float()
    option_idx = torch.tensor([len(state_ids) + o for o in row["opts"]])
    query = final[len(state_ids) + row["decide"]]
    expected = model.head(query, ordinary[option_idx])
    assert torch.allclose(out[0][0], expected, atol=1e-6)
    assert not torch.equal(ordinary[len(state_ids) + row["decide"]], query)   # the decide state was pondered


def test_batched_ponder_rows_match_serial_rows(monkeypatch):
    model, tok, _, _, _ = build(monkeypatch, "ponder", 2, ponder_decide_only=True)
    encs = [model.encode(tok, make_record([("pick", ["a", "bb"]), ("select", ["x", "yy", "zzz"])])),
            model.encode(tok, make_record([("choose", ["short", "much longer"])], state="a different state"))]
    monkeypatch.setenv("KEV_LOOP_ROW_BATCH", "1")
    serial = model.forward_batch(encs)
    monkeypatch.setenv("KEV_LOOP_ROW_BATCH", "8")
    batched = model.forward_batch(encs)
    assert all(torch.allclose(a, b, atol=1e-6) for left, right in zip(serial, batched) for a, b in zip(left, right))
    model.zero_grad()
    sum(logit.square().sum() for record in batched for logit in record).backward()
    assert grad_reached(list(model.lm.parameters()))


def test_positive_ponder_decide_only_keeps_non_decide_embeddings_exact(monkeypatch):
    model, tok, backbone, _, _ = build(monkeypatch, "ponder", 2, ponder_decide_only=True)
    enc = model.encode(tok, make_record([("pick", ["alpha", "beta"])]))
    model.forward_batch([enc])
    state_ids, _, rows = rows_of(enc)
    row = rows[0]
    ids = torch.tensor([state_ids + row["ids"]])
    base = backbone.embed(ids)
    decide = len(state_ids) + row["decide"]
    non_decide = torch.ones_like(ids, dtype=torch.bool)
    non_decide[0, decide] = False

    assert backbone.calls == 4
    assert torch.equal(backbone.embeds[0], base)   # ordinary input_ids pass
    assert torch.equal(backbone.embeds[1], base)   # first recurrence starts at E0
    for index in (2, 3):                           # recurrence 2 + final pass
        assert torch.equal(backbone.embeds[index][non_decide], base[non_decide])
        assert not torch.equal(backbone.embeds[index][0, decide], base[0, decide])


def test_positive_ponder_decide_only_passes_flags_and_projects_only_the_decide_state(monkeypatch):
    import kev.loop as loop

    seen = []
    real = loop.ponder_forward

    def recording(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "ponder_forward", recording)
    model, tok, _, lm_head, _ = build(monkeypatch, "ponder", 2, ponder_decide_only=True,
                                      ponder_full_vocab=True, ponder_scale_embeds=True,
                                      ponder_multiply_steps=True)
    shapes = []
    hook = lm_head.register_forward_pre_hook(lambda module, args: shapes.append(tuple(args[0].shape)))
    try:
        enc = model.encode(tok, make_record([("pick", ["a", "b"])]))
        model.forward_batch([enc])
    finally:
        hook.remove()

    assert shapes == [(1, DIM), (1, DIM)]   # per recurrence, one [B, d] decide row, no [B, L, d]
    assert seen
    assert all(call["ponder_decide_only"] is True and call.get("decide_positions") is not None
               and call["top_k"] is None and call["ponder_full_vocab"] is True
               and call["ponder_scale_embeds"] is True and call["ponder_multiply_steps"] is True
               for call in seen)


def test_positive_ponder_decide_only_zero_steps_is_the_ordinary_forward(monkeypatch):
    model, tok, backbone, _, _ = build(monkeypatch, "ponder", 0, ponder_decide_only=True)
    assert model.ponder_decide_only is True and getattr(model, "lm_head", None) is None
    enc = model.encode(tok, make_record([("pick", ["a", "b"])]))
    model.ponder_decide_only = False
    baseline = model.forward_batch([enc])
    before = backbone.calls
    model.ponder_decide_only = True
    out = model.forward_batch([enc])
    assert backbone.calls == before + 1
    assert all(torch.equal(a, b) for x, y in zip(baseline, out) for a, b in zip(x, y))


def test_positive_ponder_option_decide_only_uses_final_option_and_decide_states(monkeypatch):
    model, tok, backbone, _, _ = build(monkeypatch, "ponder", 2, ponder_option_decide_only=True)
    enc = model.encode(tok, make_record([("pick", ["alpha", "beta", "gamma"])]))
    out = model.forward_batch([enc])

    state_ids, _, rows = rows_of(enc)
    row = rows[0]
    assert model.ponder_option_decide_only is True
    assert backbone.calls == 3   # no ordinary pass: two recurrences + the final pass
    final = backbone.hidden[2][0].float()
    option_idx = torch.tensor([len(state_ids) + o for o in row["opts"]])
    decide = len(state_ids) + row["decide"]
    expected = model.head(final[decide], final[option_idx])
    assert torch.allclose(out[0][0], expected, atol=1e-6)
    # The readout endpoints were actually pondered: the first recurrence starts at E0.
    assert bool((backbone.hidden[0][0, decide] != final[decide]).any())
    assert bool((backbone.hidden[0][0, option_idx] != final[option_idx]).any(dim=-1).all())


def test_positive_ponder_option_decide_only_keeps_nonselected_embeddings_exact(monkeypatch):
    model, tok, backbone, _, _ = build(monkeypatch, "ponder", 2, ponder_option_decide_only=True)
    enc = model.encode(tok, make_record([("pick", ["alpha", "beta", "gamma"])]))
    model.forward_batch([enc])

    state_ids, _, rows = rows_of(enc)
    row = rows[0]
    ids = torch.tensor([state_ids + row["ids"]])
    base = backbone.embed(ids)
    selected = torch.zeros_like(ids, dtype=torch.bool)
    selected[0, [len(state_ids) + option for option in row["opts"]]] = True
    selected[0, len(state_ids) + row["decide"]] = True
    nonselected = ~selected

    assert backbone.calls == 3
    for index in range(3):
        assert torch.equal(backbone.embeds[index][nonselected], base[nonselected])
    assert torch.equal(backbone.embeds[0][selected], base[selected])   # first recurrence starts at E0
    assert bool((backbone.embeds[1][selected] != base[selected]).any(dim=-1).all())
    assert bool((backbone.embeds[2][selected] != base[selected]).any(dim=-1).all())


def test_positive_ponder_option_decide_only_zero_steps_is_the_ordinary_forward(monkeypatch):
    model, tok, backbone, _, _ = build(monkeypatch, "ponder", 0, ponder_option_decide_only=True)
    assert model.ponder_option_decide_only is True and getattr(model, "lm_head", None) is None
    enc = model.encode(tok, make_record([("pick", ["a", "b"])]))
    model.ponder_option_decide_only = False
    baseline = model.forward_batch([enc])
    before = backbone.calls
    model.ponder_option_decide_only = True
    out = model.forward_batch([enc])
    assert backbone.calls == before + 1
    assert all(torch.equal(a, b) for x, y in zip(baseline, out) for a, b in zip(x, y))


def test_batched_option_decide_rows_match_serial_rows_with_variable_option_counts(monkeypatch):
    model, tok, _, _, _ = build(monkeypatch, "ponder", 2, ponder_option_decide_only=True)
    encs = [model.encode(tok, make_record([("pick", ["a", "bb"]), ("select", ["x", "yy", "zzz"])])),
            model.encode(tok, make_record([("choose", ["short", "much longer"])], state="a different state"))]
    monkeypatch.setenv("KEV_LOOP_ROW_BATCH", "1")
    serial = model.forward_batch(encs)
    monkeypatch.setenv("KEV_LOOP_ROW_BATCH", "8")
    batched = model.forward_batch(encs)
    assert all(torch.allclose(a, b, atol=1e-6) for left, right in zip(serial, batched) for a, b in zip(left, right))
    model.zero_grad()
    sum(logit.square().sum() for record in batched for logit in record).backward()
    assert grad_reached(list(model.lm.parameters()))
    assert grad_reached(list(model.head.parameters()))


def test_coconut_keeps_the_backbone_only_loader(monkeypatch):
    model, _, backbone, _, calls = build(monkeypatch, "coconut", 1)
    assert len(calls) == 1 and model.lm is backbone
    assert getattr(model, "lm_head", None) is None


def test_non_ponder_forward_ignores_ponder_top_k(monkeypatch):
    model, tok, _, _, _ = build(monkeypatch, "coconut", 1, ponder_top_k=1)
    enc = model.encode(tok, make_record([("pick", ["a", "b"])]))
    first = model.forward_batch([enc])
    model.ponder_top_k = 100
    second = model.forward_batch([enc])
    assert all(torch.equal(a, b) for x, y in zip(first, second) for a, b in zip(x, y))


def test_constructor_validates_the_loop_config(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an invalid loop config must be rejected before any model load")

    monkeypatch.setattr("kev.model.AutoModelForCausalLM.from_pretrained", forbidden)
    monkeypatch.setattr("kev.model.AutoModel.from_pretrained", forbidden)
    tok = CharTokenizer()
    with pytest.raises(ValueError, match="method"):
        DecisionModel("fake/tiny", tok, "cpu", loop_method="bogus", loop_steps=0)
    with pytest.raises(ValueError, match="nonnegative"):
        DecisionModel("fake/tiny", tok, "cpu", loop_method="coconut", loop_steps=-1)
    with pytest.raises(ValueError, match="requires steps=0"):
        DecisionModel("fake/tiny", tok, "cpu", loop_method="kev", loop_steps=1)
    with pytest.raises(ValueError, match="int"):
        DecisionModel("fake/tiny", tok, "cpu", loop_method="coconut", loop_steps=True)
    for top_k in (0, -1, True, 1.5, "7"):
        with pytest.raises(ValueError, match="ponder_top_k"):
            DecisionModel("fake/tiny", tok, "cpu", loop_method="ponder", loop_steps=1, ponder_top_k=top_k)
        with pytest.raises(ValueError, match="ponder_top_k"):
            DecisionModel("fake/tiny", tok, "cpu", ponder_top_k=top_k)
    for name in ("ponder_scale_embeds", "ponder_multiply_steps", "ponder_full_vocab",
                 "ponder_preserve_options", "ponder_decide_only", "ponder_option_decide_only"):
        for bad in (0, 1, 1.0, "true", None):
            with pytest.raises(ValueError, match=name):
                DecisionModel("fake/tiny", tok, "cpu", **{name: bad})
    with pytest.raises(ValueError, match="disagree"):
        DecisionModel("fake/tiny", tok, "cpu", loop_method="ponder", loop_steps=1,
                      ponder_top_k=7, ponder_full_vocab=True)
    with pytest.raises(ValueError, match="at most one"):
        DecisionModel("fake/tiny", tok, "cpu", loop_method="ponder", loop_steps=1,
                      ponder_decide_only=True, ponder_preserve_options=True)
    with pytest.raises(ValueError, match="at most one"):
        DecisionModel("fake/tiny", tok, "cpu", loop_method="ponder", loop_steps=1,
                      ponder_option_decide_only=True, ponder_decide_only=True)
    with pytest.raises(ValueError, match="at most one"):
        DecisionModel("fake/tiny", tok, "cpu", loop_method="ponder", loop_steps=1,
                      ponder_option_decide_only=True, ponder_preserve_options=True)
    with pytest.raises(ValueError, match="requires loop_method"):
        DecisionModel("fake/tiny", tok, "cpu", loop_method="coconut", loop_steps=1,
                      ponder_decide_only=True)
    with pytest.raises(ValueError, match="requires loop_method"):
        DecisionModel("fake/tiny", tok, "cpu", loop_method="coconut", loop_steps=1,
                      ponder_option_decide_only=True)


def test_positive_steps_reject_hybrid_backbones_and_option_isolation(monkeypatch):
    tok = CharTokenizer()
    backbone = TinyBackbone(hybrid=True)
    install_loader(monkeypatch, backbone, nn.Linear(DIM, VOCAB, bias=False))
    with pytest.raises(ValueError, match="attention-only"):
        DecisionModel("fake/tiny", tok, "cpu", head_dim=DIM, loop_method="coconut", loop_steps=1)
    with pytest.raises(ValueError, match="option_isolation"):
        DecisionModel("fake/tiny", tok, "cpu", option_isolation=True, loop_method="coconut", loop_steps=1)


def test_full_weight_ponder_loads_checkpoint_backbone_and_pinned_base_head(monkeypatch):
    tok = CharTokenizer()
    base_backbone = TinyBackbone()
    trained_backbone = TinyBackbone()
    lm_head = nn.Linear(DIM, VOCAB, bias=False)
    causal_calls = install_loader(monkeypatch, base_backbone, lm_head)
    backbone_calls = []

    def load_backbone(*args, **kwargs):
        backbone_calls.append((args, kwargs))
        return trained_backbone

    monkeypatch.setattr("kev.model.AutoModel.from_pretrained", load_backbone)
    model = DecisionModel("fake/tiny", tok, "cpu", weights="some/checkpoint", head_dim=DIM,
                          loop_method="ponder", loop_steps=1, ponder_decide_only=True)
    assert model.lm is trained_backbone and model.lm_head is lm_head
    assert causal_calls[0][0][0] == "fake/tiny"
    assert backbone_calls[0][0][0] == "some/checkpoint"
    assert all(not parameter.requires_grad for parameter in model.lm_head.parameters())


@pytest.mark.parametrize("method", ["coconut", "ponder"])
def test_positive_steps_reject_shared_prefix(monkeypatch, method):
    model, tok, _, _, _ = build(monkeypatch, method, 1)
    enc = model.encode(tok, make_record([("pick", ["a", "b"])]))
    with pytest.raises(ValueError, match="shared prefix"):
        model.forward_batch([enc], shared_prefix=True)
    with pytest.raises(ValueError, match="shared prefix"):
        model.forward_rows_batch([enc], shared_prefix=True)


def test_meta_loop_fields_default_round_trip_and_compatibility(tmp_path):
    old = Meta.from_dict({"base": "Qwen/tiny"})
    assert (old.loop_method, old.loop_steps) == ("kev", 0)
    assert old.ponder_top_k == 100
    assert (old.ponder_scale_embeds, old.ponder_multiply_steps, old.ponder_full_vocab,
            old.ponder_preserve_options, old.ponder_decide_only,
            old.ponder_option_decide_only) == (False, False, False, False, False, False)
    assert old.coconut_start_layer == 0
    assert (old.to_dict()["loop_method"], old.to_dict()["loop_steps"]) == ("kev", 0)
    assert old.to_dict()["ponder_top_k"] == 100
    assert old.to_dict()["ponder_decide_only"] is False
    assert old.to_dict()["ponder_option_decide_only"] is False
    assert old.to_dict()["coconut_start_layer"] == 0

    new = Meta(base="Qwen/tiny", loop_method="ponder", loop_steps=3, ponder_top_k=7,
               ponder_scale_embeds=True, ponder_multiply_steps=True, ponder_preserve_options=True)
    round_tripped = Meta.from_dict(new.to_dict())
    assert (round_tripped.loop_method, round_tripped.loop_steps, round_tripped.ponder_top_k) == ("ponder", 3, 7)
    assert (round_tripped.ponder_scale_embeds, round_tripped.ponder_multiply_steps,
            round_tripped.ponder_preserve_options) == (True, True, True)
    decide = Meta(base="Qwen/tiny", loop_method="ponder", loop_steps=2,
                  ponder_decide_only=True, ponder_full_vocab=True)
    decide_round_trip = Meta.from_dict(decide.to_dict())
    assert decide_round_trip.ponder_decide_only is True and decide_round_trip.ponder_full_vocab is True
    option = Meta(base="Qwen/tiny", loop_method="ponder", loop_steps=2,
                  ponder_option_decide_only=True, ponder_full_vocab=True)
    option_round_trip = Meta.from_dict(option.to_dict())
    assert option_round_trip.ponder_option_decide_only is True
    coconut = Meta(base="Qwen/tiny", loop_method="coconut", loop_steps=2, coconut_start_layer=16)
    coconut_round_trip = Meta.from_dict(coconut.to_dict())
    assert coconut_round_trip.coconut_start_layer == 16
    assert Meta.from_dict({"base": "Qwen/tiny"}).ponder_decide_only is False   # old checkpoints
    assert Meta.from_dict({"base": "Qwen/tiny"}).ponder_option_decide_only is False
    write_meta(tmp_path, new)
    read = read_meta(tmp_path)
    assert (read.loop_method, read.loop_steps, read.ponder_top_k) == ("ponder", 3, 7)
    loop_fields = {"loop_method", "loop_steps", "ponder_top_k", "ponder_scale_embeds",
                   "ponder_multiply_steps", "ponder_full_vocab", "ponder_preserve_options",
                   "ponder_decide_only", "ponder_option_decide_only", "coconut_start_layer"}
    assert loop_fields <= set(Meta.KNOWN)
    assert loop_fields <= set(Checkpoint.COMPAT_FIELDS)
    with pytest.raises(ValueError, match="at most one"):
        Meta(base="Qwen/tiny", ponder_decide_only=True, ponder_preserve_options=True)
    with pytest.raises(ValueError, match="at most one"):
        Meta(base="Qwen/tiny", loop_method="ponder", ponder_option_decide_only=True,
             ponder_decide_only=True)
    with pytest.raises(ValueError, match="at most one"):
        Meta(base="Qwen/tiny", loop_method="ponder", ponder_option_decide_only=True,
             ponder_preserve_options=True)
    with pytest.raises(ValueError, match="requires loop_method"):
        Meta(base="Qwen/tiny", ponder_option_decide_only=True)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="ponder_top_k"):
            Meta(base="Qwen/tiny", ponder_top_k=bad)


def test_checkpoint_passes_saved_ponder_top_k_to_the_model(monkeypatch):
    ck = Checkpoint.__new__(Checkpoint)
    ck.path = "saved"
    ck.meta = Meta(base="Qwen/tiny", loop_method="ponder", loop_steps=2, ponder_top_k=7,
                   ponder_scale_embeds=True, ponder_multiply_steps=True, ponder_decide_only=True)
    monkeypatch.setattr(ck, "saved_dtype", lambda: "float32")
    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("kev.checkpoint.DecisionModel", fake)
    model, merged = ck._full_torch(None, "cpu", LoadOptions())
    assert merged is True and isinstance(model, SimpleNamespace)
    assert seen["loop_method"] == "ponder" and seen["loop_steps"] == 2 and seen["ponder_top_k"] == 7
    assert seen["ponder_scale_embeds"] is True and seen["ponder_multiply_steps"] is True
    assert seen["ponder_preserve_options"] is False and seen["ponder_decide_only"] is True
    assert seen["ponder_option_decide_only"] is False
    assert seen["coconut_start_layer"] == 0


def test_checkpoint_passes_saved_option_decide_only_to_the_model(monkeypatch):
    ck = Checkpoint.__new__(Checkpoint)
    ck.path = "saved"
    ck.meta = Meta(base="Qwen/tiny", loop_method="ponder", loop_steps=2,
                   ponder_option_decide_only=True)
    monkeypatch.setattr(ck, "saved_dtype", lambda: "float32")
    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("kev.checkpoint.DecisionModel", fake)
    ck._full_torch(None, "cpu", LoadOptions())
    assert seen["ponder_option_decide_only"] is True
    assert seen["ponder_decide_only"] is False and seen["ponder_preserve_options"] is False


def test_mlx_rejects_loop_checkpoints():
    ck = Checkpoint.__new__(Checkpoint)
    ck.path = "loop-checkpoint"
    for method, steps in (("coconut", 0), ("ponder", 1)):
        ck.meta = Meta(base="Qwen/tiny", loop_method=method, loop_steps=steps)
        with pytest.raises(ValueError, match="MLX"):
            ck._load_mlx(None, LoadOptions())


def test_train_cli_accepts_zero_step_loops_and_rejects_bad_combinations(tmp_path, monkeypatch, capsys):
    import kev.train as train

    def parse(*argv):
        monkeypatch.setattr(sys, "argv", ["kev.train", *argv])
        return train.parse_args()

    accepted = parse("--loop_method", "coconut", "--loop_steps", "0", "--out", str(tmp_path / "ok"))
    assert (accepted.loop_method, accepted.loop_steps) == ("coconut", 0)
    assert accepted.coconut_start_layer == 0
    assert accepted.ponder_top_k == 100
    assert accepted.ponder_decide_only == 0
    assert accepted.ponder_option_decide_only == 0

    accepted = parse("--loop_method", "ponder", "--loop_steps", "1", "--ponder_top_k", "7", "--out", str(tmp_path / "top-k"))
    assert accepted.ponder_top_k == 7

    full = parse("--full_ft", "1", "--weights_dtype", "bf16", "--loop_method", "ponder", "--loop_steps", "3",
                 "--ponder_decide_only", "1", "--full_ft_offload", "0", "--out", str(tmp_path / "full-loop"))
    assert full.full_ft == 1 and full.shared_prefix == 0 and full.full_ft_offload == 0

    official = parse("--loop_method", "ponder", "--loop_steps", "3", "--ponder_scale_embeds", "1",
                     "--ponder_multiply_steps", "1", "--ponder_full_vocab", "1",
                     "--out", str(tmp_path / "official"))
    assert (official.ponder_scale_embeds, official.ponder_multiply_steps,
            official.ponder_full_vocab) == (1, 1, 1)
    assert official.ponder_decide_only == 0

    decide = parse("--loop_method", "ponder", "--loop_steps", "2", "--ponder_top_k", "3",
                   "--ponder_scale_embeds", "1", "--ponder_multiply_steps", "1",
                   "--ponder_decide_only", "1", "--out", str(tmp_path / "decide"))
    assert (decide.ponder_decide_only, decide.ponder_top_k, decide.ponder_scale_embeds,
            decide.ponder_multiply_steps) == (1, 3, 1, 1)
    assert "ponder_decide_only" in train.RESUME_KNOBS   # old full-weight resume points lack the key

    option = parse("--loop_method", "ponder", "--loop_steps", "2", "--ponder_top_k", "3",
                   "--ponder_scale_embeds", "1", "--ponder_multiply_steps", "1",
                   "--ponder_option_decide_only", "1", "--out", str(tmp_path / "option"))
    assert (option.ponder_option_decide_only, option.ponder_top_k, option.ponder_scale_embeds,
            option.ponder_multiply_steps) == (1, 3, 1, 1)
    assert option.ponder_decide_only == 0 and option.ponder_preserve_options == 0
    assert "ponder_option_decide_only" in train.RESUME_KNOBS
    assert "coconut_start_layer" in train.RESUME_KNOBS

    partial = parse("--loop_method", "coconut", "--loop_steps", "2",
                    "--coconut_start_layer", "16", "--out", str(tmp_path / "partial-coconut"))
    assert partial.coconut_start_layer == 16

    cases = [
        (("--loop_method", "coconut", "--loop_steps", "-1", "--out", str(tmp_path / "negative")), "nonnegative"),
        (("--ponder_top_k", "0", "--out", str(tmp_path / "top-k-zero")), "ponder_top_k"),
        (("--ponder_top_k", "-3", "--out", str(tmp_path / "top-k-negative")), "ponder_top_k"),
        (("--ponder_top_k", "x", "--out", str(tmp_path / "top-k-text")), "--ponder_top_k"),
        (("--loop_method", "coconut", "--loop_steps", "1", "--shared_prefix", "1", "--out", str(tmp_path / "prefix")), "shared_prefix"),
        (("--loop_method", "coconut", "--loop_steps", "1", "--option_isolation", "1", "--out", str(tmp_path / "isolated")), "option_isolation"),
        (("--loop_method", "ponder", "--loop_steps", "1", "--ponder_decide_only", "1",
          "--ponder_preserve_options", "1", "--out", str(tmp_path / "both-ablation")), "at most one"),
        (("--loop_method", "ponder", "--loop_steps", "1", "--ponder_option_decide_only", "1",
          "--ponder_decide_only", "1", "--out", str(tmp_path / "option-decide")), "at most one"),
        (("--loop_method", "ponder", "--loop_steps", "1", "--ponder_option_decide_only", "1",
          "--ponder_preserve_options", "1", "--out", str(tmp_path / "option-preserve")), "at most one"),
        (("--loop_method", "coconut", "--loop_steps", "1", "--ponder_decide_only", "1",
          "--out", str(tmp_path / "decide-coconut")), "ponder_decide_only"),
        (("--loop_method", "coconut", "--loop_steps", "1", "--ponder_option_decide_only", "1",
          "--out", str(tmp_path / "option-coconut")), "ponder_option_decide_only"),
        (("--loop_method", "ponder", "--loop_steps", "1", "--coconut_start_layer", "16",
          "--out", str(tmp_path / "ponder-layer")), "coconut_start_layer"),
    ]
    for argv, needle in cases:
        with pytest.raises(SystemExit):
            parse(*argv)
        assert needle in capsys.readouterr().err
