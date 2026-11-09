"""CPU tests for the bounded latent-loop core: config, exact baseline, recurrence, gradients.

Run: uv run python -m pytest tests/test_loop.py -q
"""
import dataclasses
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kev.loop import LoopConfig, coconut_forward, ponder_decide_forward, ponder_forward

VOCAB, DIM = 11, 6
IDS = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 5]])
POS = torch.arange(4).expand(2, 4).clone()
MASK = torch.ones_like(IDS)
DECIDE = torch.tensor([3, 3])   # the <decide> token is the last token of each row


class TinyCausalBackbone(nn.Module):
    """A deterministic attention-only causal backbone with fixed weights and a pass counter."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.proj = nn.Linear(DIM, DIM, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.linspace(-0.9, 1.1, VOCAB * DIM).reshape(VOCAB, DIM))
            self.proj.weight.copy_(torch.eye(DIM) * 0.6 + torch.linspace(-0.2, 0.2, DIM * DIM).reshape(DIM, DIM))
        self.calls = 0
        self.lengths = []
        self.masks = []
        self.embeds = []
        self.hidden = []

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, attention_mask=None):
        assert (input_ids is None) != (inputs_embeds is None)
        assert position_ids is not None
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        self.calls += 1
        self.lengths.append(inputs_embeds.shape[1])
        self.masks.append(attention_mask)
        self.embeds.append(inputs_embeds.detach().clone())
        length = inputs_embeds.shape[1]
        causal = torch.tril(torch.ones(length, length, dtype=inputs_embeds.dtype))
        denom = torch.arange(1, length + 1, dtype=inputs_embeds.dtype)
        pooled = torch.einsum("ij,bjd->bid", causal, inputs_embeds) / denom[None, :, None]
        pos = position_ids.to(inputs_embeds.dtype).unsqueeze(-1)
        hidden = self.proj(inputs_embeds + 0.15 * pooled + 0.02 * pos)
        self.hidden.append(hidden.detach().clone())
        return SimpleNamespace(last_hidden_state=hidden)


def make_fake():
    backbone = TinyCausalBackbone()
    head = nn.Linear(DIM, VOCAB, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.linspace(-1.2, 1.3, VOCAB * DIM).reshape(VOCAB, DIM))
    return backbone, head


def ordinary(backbone):
    return backbone(input_ids=IDS, position_ids=POS, attention_mask=MASK).last_hidden_state


def manual_ponder(backbone, head, steps, *, scale_embeds=False, multiply_steps=False, top_k=100):
    """Reference recurrence written independently of kev.loop, with the official initialization flags."""
    embedding = backbone.get_input_embeddings()
    base = embedding(IDS)
    weight = embedding.weight
    if scale_embeds:
        embed_scale = torch.sqrt(torch.tensor(DIM, dtype=base.dtype))
        base = base * embed_scale
        weight = weight * embed_scale
    current = base * steps if multiply_steps else base
    anchored = not (scale_embeds or multiply_steps)
    accumulated = None
    for _ in range(steps):
        hidden = backbone(inputs_embeds=current, position_ids=POS, attention_mask=MASK).last_hidden_state
        logits = head(hidden).float()
        if top_k is None:
            probs = torch.softmax(logits, dim=-1)
            expected = torch.matmul(probs.to(weight.dtype), weight)
        else:
            k = min(top_k, VOCAB)
            values, selected_ids = torch.topk(logits, k, dim=-1)
            probs = torch.softmax(values, dim=-1)
            selected = weight[selected_ids]
            expected = (probs.to(selected.dtype).unsqueeze(-1) * selected).sum(dim=-2)
        if anchored:
            accumulated = expected if accumulated is None else accumulated + expected
            current = base + accumulated
        else:
            current = current + expected
    return backbone(inputs_embeds=current, position_ids=POS, attention_mask=MASK).last_hidden_state


def manual_decide(backbone, head, steps, *, positions=DECIDE, scale_embeds=False,
                  multiply_steps=False, top_k=100):
    """Independent decide-only reference: ordinary pass, then only the decide row is pondered."""
    embedding = backbone.get_input_embeddings()
    first = ordinary(backbone)
    base = embedding(IDS)
    weight = embedding.weight
    rows = torch.arange(IDS.shape[0])
    decide_base = base[rows, positions]
    if scale_embeds:
        embed_scale = torch.sqrt(torch.tensor(DIM, dtype=base.dtype))
        decide_base = decide_base * embed_scale
        weight = weight * embed_scale
    decide_current = decide_base * steps if multiply_steps else decide_base
    anchored = not (scale_embeds or multiply_steps)
    accumulated = None
    for _ in range(steps):
        current = base.clone()
        current[rows, positions] = decide_current
        hidden = backbone(inputs_embeds=current, position_ids=POS, attention_mask=MASK).last_hidden_state
        logits = head(hidden[rows, positions]).float()
        if top_k is None:
            probs = torch.softmax(logits, dim=-1)
            expected = torch.matmul(probs.to(weight.dtype), weight)
        else:
            values, selected_ids = torch.topk(logits, min(top_k, VOCAB), dim=-1)
            probs = torch.softmax(values, dim=-1)
            selected = weight[selected_ids]
            expected = (probs.to(selected.dtype).unsqueeze(-1) * selected).sum(dim=-2)
        if anchored:
            accumulated = expected if accumulated is None else accumulated + expected
            decide_current = decide_base + accumulated
        else:
            decide_current = decide_current + expected
    final_current = base.clone()
    final_current[rows, positions] = decide_current
    final = backbone(inputs_embeds=final_current, position_ids=POS, attention_mask=MASK).last_hidden_state
    return first, final, final[rows, positions]


# Option endpoints plus <decide>, variable option counts per row. Both rows decide at position 3.
UPDATE = torch.zeros(2, 4, dtype=torch.bool)
UPDATE[0, [1, 2, 3]] = True
UPDATE[1, [0, 3]] = True


def manual_option_decide(backbone, head, steps, update_mask, *, scale_embeds=False,
                         multiply_steps=False, top_k=100):
    """Independent option+decision reference: no ordinary pass, only the mask rows are pondered."""
    embedding = backbone.get_input_embeddings()
    base = embedding(IDS)
    weight = embedding.weight
    selected_base = base[update_mask]
    if scale_embeds:
        embed_scale = torch.sqrt(torch.tensor(DIM, dtype=base.dtype))
        selected_base = selected_base * embed_scale
        weight = weight * embed_scale
    selected_current = selected_base * steps if multiply_steps else selected_base
    anchored = not (scale_embeds or multiply_steps)
    accumulated = None
    for _ in range(steps):
        current = base.clone()
        current[update_mask] = selected_current
        hidden = backbone(inputs_embeds=current, position_ids=POS, attention_mask=MASK).last_hidden_state
        logits = head(hidden[update_mask]).float()
        if top_k is None:
            probs = torch.softmax(logits, dim=-1)
            expected = torch.matmul(probs.to(weight.dtype), weight)
        else:
            values, selected_ids = torch.topk(logits, min(top_k, VOCAB), dim=-1)
            probs = torch.softmax(values, dim=-1)
            selected = weight[selected_ids]
            expected = (probs.to(selected.dtype).unsqueeze(-1) * selected).sum(dim=-2)
        if anchored:
            accumulated = expected if accumulated is None else accumulated + expected
            selected_current = selected_base + accumulated
        else:
            selected_current = selected_current + expected
    final_current = base.clone()
    final_current[update_mask] = selected_current
    return backbone(inputs_embeds=final_current, position_ids=POS, attention_mask=MASK).last_hidden_state


def test_loop_config_validates_and_is_immutable():
    assert LoopConfig().method == "kev" and LoopConfig().steps == 0
    assert LoopConfig().ponder_top_k == 100 and LoopConfig().top_k == 100
    assert LoopConfig("coconut", 0).passes == 1
    assert LoopConfig("ponder", 3).passes == 4
    assert LoopConfig("ponder", 3, 7).ponder_top_k == 7
    assert LoopConfig("ponder", 3, top_k=7).top_k == 7
    default = LoopConfig("ponder", 1)
    assert (default.ponder_scale_embeds, default.ponder_multiply_steps,
            default.ponder_full_vocab, default.ponder_top_k) == (False, False, False, 100)
    assert default.ponder_decide_only is False and default.passes == 2
    assert default.ponder_option_decide_only is False
    assert LoopConfig("ponder", 2, ponder_decide_only=True).passes == 4
    assert LoopConfig("ponder", 0, ponder_decide_only=True).passes == 1
    assert LoopConfig("ponder", 2, ponder_option_decide_only=True).passes == 3
    assert LoopConfig("ponder", 0, ponder_option_decide_only=True).passes == 1
    with pytest.raises(ValueError, match="ponder_decide_only"):
        LoopConfig("coconut", 1, ponder_decide_only=True)
    with pytest.raises(ValueError, match="ponder_option_decide_only"):
        LoopConfig("coconut", 1, ponder_option_decide_only=True)
    with pytest.raises(ValueError, match="coconut_start_layer"):
        LoopConfig("ponder", 1, coconut_start_layer=1)
    with pytest.raises(ValueError, match="nonnegative"):
        LoopConfig("coconut", 1, coconut_start_layer=-1)
    with pytest.raises(ValueError, match="int"):
        LoopConfig("coconut", 1, coconut_start_layer=True)
    with pytest.raises(ValueError, match="at most one"):
        LoopConfig("ponder", 1, ponder_decide_only=True, ponder_option_decide_only=True)
    assert LoopConfig("ponder", 1, None).ponder_full_vocab is True
    assert LoopConfig("ponder", 1, ponder_full_vocab=True).ponder_top_k is None
    official = LoopConfig.official_style(3)
    assert (official.method, official.steps, official.ponder_top_k) == ("ponder", 3, None)
    assert (official.ponder_scale_embeds, official.ponder_multiply_steps,
            official.ponder_full_vocab, official.ponder_decide_only) == (True, True, True, False)
    assert LoopConfig.official_style(2, ponder_decide_only=True).ponder_decide_only is True
    bounded = LoopConfig.official_style(2, top_k=7)
    assert (bounded.ponder_top_k, bounded.ponder_full_vocab) == (7, False)
    for method, steps in (("bogus", 0), ("kev", 1), ("coconut", -1), ("ponder", -1),
                          ("coconut", 1.5), ("coconut", True)):
        with pytest.raises(ValueError):
            LoopConfig(method, steps)
    for top_k in (0, -1, True, False, 1.5, "7"):
        with pytest.raises(ValueError):
            LoopConfig("ponder", 1, top_k)
        with pytest.raises(ValueError):
            LoopConfig("ponder", 1, top_k=top_k)
    with pytest.raises(ValueError, match="disagree"):
        LoopConfig("ponder", 1, 3, top_k=4)
    with pytest.raises(ValueError, match="disagree"):
        LoopConfig("ponder", 1, 7, ponder_full_vocab=True)
    for name in ("ponder_scale_embeds", "ponder_multiply_steps", "ponder_full_vocab",
                 "ponder_decide_only", "ponder_option_decide_only"):
        for bad in (0, 1, 1.0, "true", None):
            with pytest.raises(ValueError, match="bool"):
                LoopConfig("ponder", 1, **{name: bad})
    config = LoopConfig("coconut", 2)
    assert LoopConfig("coconut", 2, coconut_start_layer=16).coconut_start_layer == 16
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.steps = 3


def test_coconut_zero_steps_is_exact_one_pass():
    backbone, _ = make_fake()
    reference = ordinary(backbone)
    backbone.calls, backbone.lengths, backbone.masks = 0, [], []
    out = coconut_forward(backbone, IDS, POS, MASK, 0)
    assert backbone.calls == 1 and backbone.lengths == [4]
    assert out.passes == 1
    assert torch.equal(out.last_hidden_state, reference)
    assert torch.equal(out.token_hidden_states, reference)
    assert torch.equal(out.latent_hidden_state, reference[:, -1])


def test_ponder_zero_steps_is_exact_one_pass():
    backbone, head = make_fake()
    reference = ordinary(backbone)
    backbone.calls, backbone.lengths, backbone.masks = 0, [], []
    out = ponder_forward(backbone, head, IDS, POS, MASK, 0, top_k=1)   # K=0 ignores top_k: one ordinary input_ids pass
    assert backbone.calls == 1 and backbone.lengths == [4]
    assert out.passes == 1
    assert torch.equal(out.last_hidden_state, reference)


def test_coconut_grows_by_one_latent_per_step_and_makes_k_plus_one_passes():
    backbone, _ = make_fake()
    reference = ordinary(backbone)
    backbone.calls, backbone.lengths, backbone.masks = 0, [], []
    out = coconut_forward(backbone, IDS, POS, MASK, 2)
    assert backbone.calls == 3 and out.passes == 3
    assert backbone.lengths == [4, 5, 6]
    assert [tuple(mask.shape) for mask in backbone.masks] == [(2, 4), (2, 5), (2, 6)]
    assert all(bool(mask.all()) for mask in backbone.masks)
    assert out.last_hidden_state.shape == (2, 6, DIM)
    assert out.token_hidden_states.shape == (2, 4, DIM)
    assert torch.equal(out.token_hidden_states, reference)
    assert torch.equal(out.latent_hidden_state, out.last_hidden_state[:, -1])

    # Hand-computed recurrence: E0 plus the previous pass's last hidden state, one token at a time.
    embeds = backbone.get_input_embeddings()(IDS)
    positions, mask = POS, MASK
    latent = reference[:, -1]
    hidden = reference
    for _ in range(2):
        embeds = torch.cat((embeds, latent.unsqueeze(1)), dim=1)
        positions = torch.cat((positions, positions[:, -1:] + 1), dim=1)
        mask = torch.cat((mask, mask.new_ones((2, 1))), dim=1)
        hidden = backbone(inputs_embeds=embeds, position_ids=positions, attention_mask=mask).last_hidden_state
        latent = hidden[:, -1]
    assert torch.allclose(out.last_hidden_state, hidden, atol=1e-6)
    assert torch.allclose(out.latent_hidden_state, hidden[:, -1], atol=1e-6)


@pytest.mark.parametrize("top_k", [VOCAB, VOCAB + 7, 100])
def test_ponder_top_k_at_or_above_vocab_matches_full_vocabulary_reference(top_k):
    reference_backbone, reference_head = make_fake()
    base = reference_backbone.get_input_embeddings()(IDS)
    first = reference_backbone(inputs_embeds=base, position_ids=POS, attention_mask=MASK).last_hidden_state
    probs = torch.softmax(reference_head(first).float(), dim=-1)
    weight = reference_backbone.get_input_embeddings().weight
    assert probs.shape[-1] == VOCAB and bool((probs > 0).all())

    expected = torch.zeros_like(base)
    for token in range(VOCAB):
        expected = expected + probs[..., token].unsqueeze(-1) * weight[token]
    second = base + expected
    assert not torch.allclose(second, base)
    final = reference_backbone(inputs_embeds=second, position_ids=POS, attention_mask=MASK).last_hidden_state

    backbone, head = make_fake()
    out = ponder_forward(backbone, head, IDS, POS, MASK, 1, top_k=top_k)
    assert backbone.calls == 2 and backbone.lengths == [4, 4]
    assert out.passes == 2
    assert torch.allclose(out.last_hidden_state, final, atol=1e-6)


def test_ponder_k1_matches_manual_top_k_expected_embedding():
    k = 3
    reference_backbone, reference_head = make_fake()
    base = reference_backbone.get_input_embeddings()(IDS)
    first = reference_backbone(inputs_embeds=base, position_ids=POS, attention_mask=MASK).last_hidden_state
    logits = reference_head(first).float()
    values, selected_ids = torch.topk(logits, k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    weight = reference_backbone.get_input_embeddings().weight
    selected = weight[selected_ids]
    expected = (probs.unsqueeze(-1) * selected).sum(dim=-2)
    second = base + expected
    final = reference_backbone(inputs_embeds=second, position_ids=POS, attention_mask=MASK).last_hidden_state

    backbone, head = make_fake()
    out = ponder_forward(backbone, head, IDS, POS, MASK, 1, top_k=k)
    assert backbone.calls == 2 and backbone.lengths == [4, 4]
    assert torch.allclose(out.last_hidden_state, final, atol=1e-6)


def test_ponder_top_k_never_softmaxes_the_full_vocabulary(monkeypatch):
    widths = []
    original = torch.softmax

    def recording_softmax(input, dim, *args, **kwargs):
        widths.append(input.shape[-1])
        return original(input, dim=dim, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", recording_softmax)
    backbone, head = make_fake()
    ponder_forward(backbone, head, IDS, POS, MASK, 2, top_k=3)
    assert widths == [3, 3]


def test_ponder_full_vocab_softmaxes_the_full_vocabulary(monkeypatch):
    widths = []
    original = torch.softmax

    def recording_softmax(input, dim, *args, **kwargs):
        widths.append(input.shape[-1])
        return original(input, dim=dim, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", recording_softmax)
    backbone, head = make_fake()
    ponder_forward(backbone, head, IDS, POS, MASK, 2, ponder_full_vocab=True)
    assert widths == [VOCAB, VOCAB]


@pytest.mark.parametrize("scale_embeds", [False, True])
@pytest.mark.parametrize("multiply_steps", [False, True])
@pytest.mark.parametrize("top_k", [None, 3])
def test_ponder_explicit_initialization_matches_manual_recurrence(scale_embeds, multiply_steps, top_k):
    reference_backbone, reference_head = make_fake()
    expected = manual_ponder(reference_backbone, reference_head, 2, scale_embeds=scale_embeds,
                             multiply_steps=multiply_steps, top_k=top_k)

    backbone, head = make_fake()
    kwargs = {"ponder_full_vocab": True} if top_k is None else {"top_k": top_k}
    out = ponder_forward(backbone, head, IDS, POS, MASK, 2, ponder_scale_embeds=scale_embeds,
                         ponder_multiply_steps=multiply_steps, **kwargs)
    assert backbone.calls == 3 and backbone.lengths == [4, 4, 4] and out.passes == 3
    assert torch.allclose(out.last_hidden_state, expected, atol=1e-6)


def test_ponder_official_style_config_drives_the_named_recurrence():
    config = LoopConfig.official_style(2)
    reference_backbone, reference_head = make_fake()
    expected = manual_ponder(reference_backbone, reference_head, 2, scale_embeds=True,
                             multiply_steps=True, top_k=None)
    backbone, head = make_fake()
    out = ponder_forward(backbone, head, IDS, POS, MASK, config=config)
    assert backbone.calls == 3 and out.passes == 3
    assert torch.allclose(out.last_hidden_state, expected, atol=1e-6)


def test_ponder_default_config_keeps_unscaled_anchored_recurrence():
    config = LoopConfig("ponder", 2, top_k=3)
    assert (config.ponder_scale_embeds, config.ponder_multiply_steps,
            config.ponder_full_vocab) == (False, False, False)

    reference_backbone, reference_head = make_fake()
    expected = manual_ponder(reference_backbone, reference_head, 2, top_k=3)
    backbone, head = make_fake()
    out = ponder_forward(backbone, head, IDS, POS, MASK, config=config)
    assert backbone.calls == 3 and out.passes == 3
    assert torch.allclose(out.last_hidden_state, expected, atol=1e-6)


def test_ponder_zero_steps_ignores_official_style_settings():
    backbone, head = make_fake()
    reference = ordinary(backbone)
    backbone.calls, backbone.lengths, backbone.masks = 0, [], []
    out = ponder_forward(backbone, head, IDS, POS, MASK, 0, ponder_full_vocab=True,
                         ponder_scale_embeds=True, ponder_multiply_steps=True)
    assert backbone.calls == 1 and backbone.lengths == [4] and out.passes == 1
    assert torch.equal(out.last_hidden_state, reference)


def test_ponder_official_style_gradients_reach_backbone_head_and_embedding_rows():
    backbone, head = make_fake()
    out = ponder_forward(backbone, head, IDS, POS, MASK, 2, ponder_full_vocab=True,
                         ponder_scale_embeds=True, ponder_multiply_steps=True)
    out.last_hidden_state.square().mean().backward()
    _assert_gradients(list(backbone.parameters()) + list(head.parameters()))
    embedding_grad = backbone.get_input_embeddings().weight.grad
    assert embedding_grad is not None
    assert bool((embedding_grad.abs().sum(dim=-1) > 0).any())


def test_ponder_decide_only_default_false_is_the_unchanged_recurrence():
    backbone, head = make_fake()
    default = ponder_forward(backbone, head, IDS, POS, MASK, 2, top_k=3)
    backbone, head = make_fake()
    explicit = ponder_forward(backbone, head, IDS, POS, MASK, 2, top_k=3,
                              ponder_decide_only=False, decide_positions=DECIDE)
    assert torch.equal(default.last_hidden_state, explicit.last_hidden_state)
    assert default.passes == explicit.passes == 3
    assert default.first_hidden_state is None and default.decide_hidden_state is None
    assert default.token_hidden_states is default.last_hidden_state


@pytest.mark.parametrize("scale_embeds", [False, True])
@pytest.mark.parametrize("multiply_steps", [False, True])
@pytest.mark.parametrize("top_k", [None, 3])
def test_ponder_decide_only_matches_manual_recurrence(scale_embeds, multiply_steps, top_k):
    reference_backbone, reference_head = make_fake()
    first, final, query = manual_decide(reference_backbone, reference_head, 2, scale_embeds=scale_embeds,
                                        multiply_steps=multiply_steps, top_k=top_k)

    backbone, head = make_fake()
    kwargs = {"ponder_full_vocab": True} if top_k is None else {"top_k": top_k}
    out = ponder_forward(backbone, head, IDS, POS, MASK, 2, ponder_decide_only=True,
                         decide_positions=DECIDE, ponder_scale_embeds=scale_embeds,
                         ponder_multiply_steps=multiply_steps, **kwargs)
    assert backbone.calls == 4 and backbone.lengths == [4, 4, 4, 4] and out.passes == 4
    assert torch.allclose(out.token_hidden_states, first, atol=1e-6)
    assert torch.allclose(out.last_hidden_state, final, atol=1e-6)
    assert torch.allclose(out.decide_hidden_state, query, atol=1e-6)
    assert torch.allclose(out.query_hidden_state, query, atol=1e-6)


def test_ponder_decide_only_keeps_non_decide_embeddings_exact_and_projects_only_decide():
    backbone, head = make_fake()
    shapes = []
    hook = head.register_forward_pre_hook(lambda module, args: shapes.append(tuple(args[0].shape)))
    try:
        out = ponder_forward(backbone, head, IDS, POS, MASK, 2, top_k=3,
                             ponder_decide_only=True, decide_positions=DECIDE)
    finally:
        hook.remove()

    # lm_head sees the [B, d] decide states only, never [B, L, d] / [B, L, vocab].
    assert shapes == [(2, DIM), (2, DIM)]
    base = backbone.get_input_embeddings()(IDS)
    rows = torch.arange(IDS.shape[0])
    non_decide = torch.ones_like(IDS, dtype=torch.bool)
    non_decide[rows, DECIDE] = False
    # Pass 0 is the ordinary input_ids pass; 1-2 are recurrences; 3 is the final pass.
    assert torch.equal(backbone.embeds[0], base)
    assert torch.equal(backbone.embeds[1], base)   # first recurrence starts from E0, unscaled
    for index in (2, 3):
        assert torch.equal(backbone.embeds[index][non_decide], base[non_decide])
        assert not torch.equal(backbone.embeds[index][rows, DECIDE], base[rows, DECIDE])
    assert torch.equal(out.token_hidden_states, backbone.hidden[0])
    assert torch.allclose(out.decide_hidden_state, out.last_hidden_state[rows, DECIDE])


def test_ponder_decide_only_zero_steps_is_exact_one_pass():
    backbone, head = make_fake()
    reference = ordinary(backbone)
    backbone.calls, backbone.lengths, backbone.masks, backbone.embeds = 0, [], [], []
    out = ponder_forward(backbone, head, IDS, POS, MASK, 0, ponder_full_vocab=True,
                         ponder_scale_embeds=True, ponder_multiply_steps=True,
                         ponder_decide_only=True, decide_positions=DECIDE)
    assert backbone.calls == 1 and backbone.lengths == [4] and out.passes == 1
    assert torch.equal(out.last_hidden_state, reference)
    assert torch.equal(out.token_hidden_states, reference)
    assert torch.equal(out.decide_hidden_state, reference[torch.arange(2), DECIDE])


def test_ponder_decide_only_gradients_reach_the_backbone_and_selected_embedding_rows():
    k = 3
    backbone, head = make_fake()
    base = backbone.get_input_embeddings()(IDS)
    probe = base.clone()
    probe[torch.arange(2), DECIDE] = base[torch.arange(2), DECIDE]
    first = backbone(inputs_embeds=probe, position_ids=POS, attention_mask=MASK).last_hidden_state
    _, selected_ids = torch.topk(head(first[torch.arange(2), DECIDE]).float(), k, dim=-1)

    out = ponder_forward(backbone, head, IDS, POS, MASK, 2, top_k=k,
                         ponder_decide_only=True, decide_positions=DECIDE)
    out.decide_hidden_state.square().mean().backward()
    _assert_gradients(list(backbone.parameters()) + list(head.parameters()))
    embedding_grad = backbone.get_input_embeddings().weight.grad
    assert embedding_grad is not None
    selected = embedding_grad[selected_ids.reshape(-1)]
    assert bool((selected.abs().sum(dim=-1) > 0).any())


def test_ponder_decide_forward_matches_the_flag_and_validates_positions():
    backbone, head = make_fake()
    direct = ponder_decide_forward(backbone, head, IDS, POS, DECIDE, MASK, 2, top_k=3)
    backbone, head = make_fake()
    flagged = ponder_forward(backbone, head, IDS, POS, MASK, 2, top_k=3,
                             ponder_decide_only=True, decide_positions=DECIDE)
    assert direct.passes == flagged.passes == 4
    assert torch.allclose(direct.last_hidden_state, flagged.last_hidden_state, atol=1e-6)
    assert torch.allclose(direct.decide_hidden_state, flagged.decide_hidden_state, atol=1e-6)

    backbone, head = make_fake()
    with pytest.raises(ValueError, match="decide_positions"):
        ponder_decide_forward(backbone, head, IDS, POS, torch.tensor([1]), MASK, 2)
    with pytest.raises(ValueError, match="decide_positions"):
        ponder_decide_forward(backbone, head, IDS, POS, torch.tensor([0, 99]), MASK, 2)


@pytest.mark.parametrize("scale_embeds", [False, True])
@pytest.mark.parametrize("multiply_steps", [False, True])
@pytest.mark.parametrize("top_k", [None, 3])
def test_ponder_option_decide_only_matches_manual_recurrence(scale_embeds, multiply_steps, top_k):
    reference_backbone, reference_head = make_fake()
    expected = manual_option_decide(reference_backbone, reference_head, 2, UPDATE,
                                    scale_embeds=scale_embeds, multiply_steps=multiply_steps,
                                    top_k=top_k)

    backbone, head = make_fake()
    kwargs = {"ponder_full_vocab": True} if top_k is None else {"top_k": top_k}
    out = ponder_forward(backbone, head, IDS, POS, MASK, 2, ponder_option_decide_only=True,
                         update_mask=UPDATE, decide_positions=DECIDE,
                         ponder_scale_embeds=scale_embeds,
                         ponder_multiply_steps=multiply_steps, **kwargs)
    assert backbone.calls == 3 and backbone.lengths == [4, 4, 4] and out.passes == 3
    assert torch.allclose(out.last_hidden_state, expected, atol=1e-6)
    assert out.first_hidden_state is None
    assert out.token_hidden_states is out.last_hidden_state
    assert torch.allclose(out.decide_hidden_state, out.last_hidden_state[torch.arange(2), DECIDE], atol=1e-6)


def test_ponder_option_decide_only_zero_steps_is_exact_one_pass():
    backbone, head = make_fake()
    reference = ordinary(backbone)
    backbone.calls, backbone.lengths, backbone.masks = 0, [], []
    out = ponder_forward(backbone, head, IDS, POS, MASK, 0, ponder_full_vocab=True,
                         ponder_scale_embeds=True, ponder_multiply_steps=True,
                         ponder_option_decide_only=True, update_mask=UPDATE,
                         decide_positions=DECIDE)
    assert backbone.calls == 1 and backbone.lengths == [4] and out.passes == 1
    assert torch.equal(out.last_hidden_state, reference)
    assert out.first_hidden_state is None and out.decide_hidden_state is None

    # No recurrence, no mask needed: the flag is still exactly the ordinary pass.
    backbone, head = make_fake()
    reference = ordinary(backbone)
    out = ponder_forward(backbone, head, IDS, POS, MASK, 0, ponder_option_decide_only=True)
    assert out.passes == 1 and torch.equal(out.last_hidden_state, reference)


def test_ponder_option_decide_only_keeps_nonselected_embeddings_exact_and_projects_only_selected():
    backbone, head = make_fake()
    shapes = []
    hook = head.register_forward_pre_hook(lambda module, args: shapes.append(tuple(args[0].shape)))
    try:
        out = ponder_forward(backbone, head, IDS, POS, MASK, 2, top_k=3,
                             ponder_option_decide_only=True, update_mask=UPDATE,
                             decide_positions=DECIDE)
    finally:
        hook.remove()

    base = backbone.get_input_embeddings()(IDS)
    nonselected = ~UPDATE
    assert torch.equal(backbone.embeds[0][nonselected], base[nonselected])
    assert torch.equal(backbone.embeds[1][nonselected], base[nonselected])
    assert torch.equal(backbone.embeds[2][nonselected], base[nonselected])
    # The first recurrence starts from E0; later passes carry the selected expected embeddings.
    assert torch.equal(backbone.embeds[0][UPDATE], base[UPDATE])
    assert bool((backbone.embeds[1][UPDATE] != base[UPDATE]).any(dim=-1).all())
    assert bool((backbone.embeds[2][UPDATE] != base[UPDATE]).any(dim=-1).all())

    # lm_head sees the [N, d] selected states only, never [B, L, d] / [B, L, vocab].
    assert shapes == [(int(UPDATE.sum()), DIM), (int(UPDATE.sum()), DIM)]
    assert torch.allclose(out.decide_hidden_state, out.last_hidden_state[torch.arange(2), DECIDE], atol=1e-6)


def test_ponder_option_decide_only_gradients_reach_backbone_selected_logits_and_embedding_rows():
    k = 3
    backbone, head = make_fake()
    base = backbone.get_input_embeddings()(IDS)
    first = backbone(inputs_embeds=base, position_ids=POS, attention_mask=MASK).last_hidden_state
    _, selected_ids = torch.topk(head(first[UPDATE]).float(), k, dim=-1)

    out = ponder_forward(backbone, head, IDS, POS, MASK, 2, top_k=k,
                         ponder_option_decide_only=True, update_mask=UPDATE,
                         decide_positions=DECIDE)
    out.last_hidden_state.square().mean().backward()
    _assert_gradients(list(backbone.parameters()) + list(head.parameters()))
    embedding_grad = backbone.get_input_embeddings().weight.grad
    assert embedding_grad is not None
    selected = embedding_grad[selected_ids.reshape(-1)]
    assert bool((selected.abs().sum(dim=-1) > 0).any())


def test_ponder_option_decide_only_validates_the_update_mask():
    backbone, head = make_fake()
    with pytest.raises(ValueError, match="update_mask"):
        ponder_forward(backbone, head, IDS, POS, MASK, 1, ponder_option_decide_only=True,
                       decide_positions=DECIDE)
    with pytest.raises(ValueError, match="boolean"):
        ponder_forward(backbone, head, IDS, POS, MASK, 1, ponder_option_decide_only=True,
                       update_mask=UPDATE.long(), decide_positions=DECIDE)
    with pytest.raises(ValueError, match="match input_ids"):
        ponder_forward(backbone, head, IDS, POS, MASK, 1, ponder_option_decide_only=True,
                       update_mask=UPDATE[:, :3], decide_positions=DECIDE)

    padded_mask = MASK.clone()
    padded_mask[1, 3] = 0
    with pytest.raises(ValueError, match="padded"):
        ponder_forward(backbone, head, IDS, POS, padded_mask, 1, ponder_option_decide_only=True,
                       update_mask=UPDATE)

    without_decide = UPDATE.clone()
    without_decide[:, 3] = False
    with pytest.raises(ValueError, match="decide position"):
        ponder_forward(backbone, head, IDS, POS, MASK, 1, ponder_option_decide_only=True,
                       update_mask=without_decide, decide_positions=DECIDE)
    with pytest.raises(ValueError, match="ponder_option_decide_only"):
        ponder_forward(backbone, head, IDS, POS, MASK, 1, update_mask=UPDATE)


def _assert_gradients(parameters):
    grads = [parameter.grad for parameter in parameters]
    assert grads and all(grad is not None for grad in grads)
    assert all(bool(torch.isfinite(grad).all()) for grad in grads)
    assert any(bool(grad.abs().sum() > 0) for grad in grads)


def test_ponder_top_k_gradients_reach_selected_logits_and_embedding_rows():
    k = 3
    backbone, head = make_fake()
    base = backbone.get_input_embeddings()(IDS)
    first = backbone(inputs_embeds=base, position_ids=POS, attention_mask=MASK).last_hidden_state
    _, selected_ids = torch.topk(head(first).float(), k, dim=-1)

    out = ponder_forward(backbone, head, IDS, POS, MASK, 1, top_k=k)
    out.last_hidden_state.square().mean().backward()
    _assert_gradients(list(backbone.parameters()) + list(head.parameters()))
    embedding_grad = backbone.get_input_embeddings().weight.grad
    assert embedding_grad is not None
    selected = embedding_grad[selected_ids.reshape(-1)]
    assert bool((selected.abs().sum(dim=-1) > 0).any())


def test_coconut_gradients_reach_the_backbone():
    backbone, _ = make_fake()
    out = coconut_forward(backbone, IDS, POS, MASK, 2)
    assert backbone.calls == 3 and out.passes == 3
    (out.latent_hidden_state.square().mean() + out.token_hidden_states.square().mean()).backward()
    _assert_gradients(list(backbone.parameters()))


def test_ponder_gradients_reach_the_backbone_and_head():
    backbone, head = make_fake()
    out = ponder_forward(backbone, head, IDS, POS, MASK, 2)
    assert backbone.calls == 3 and backbone.lengths == [4, 4, 4] and out.passes == 3
    out.last_hidden_state.square().mean().backward()
    _assert_gradients(list(backbone.parameters()) + list(head.parameters()))


@pytest.mark.parametrize("method", ["coconut", "ponder"])
def test_padded_masks_are_rejected(method):
    padded = MASK.clone()
    padded[0, 2] = 0
    backbone, head = make_fake()
    with pytest.raises(ValueError, match="full" if method == "coconut" else "prefix"):
        if method == "coconut":
            coconut_forward(backbone, IDS, POS, padded, 1)
        else:
            ponder_forward(backbone, head, IDS, POS, padded, 1)


def test_ponder_accepts_exact_right_padding():
    ids, pos, mask = IDS.clone(), POS.clone(), MASK.clone()
    ids[1, 3] = 0
    pos[1, 3] = 0
    mask[1, 3] = 0
    backbone, head = make_fake()
    batched = ponder_forward(backbone, head, ids, pos, mask, 2, ponder_decide_only=True,
                             decide_positions=torch.tensor([3, 2]))
    backbone2, head2 = make_fake()
    single = ponder_forward(backbone2, head2, ids[1:2, :3], pos[1:2, :3], mask[1:2, :3], 2,
                            ponder_decide_only=True, decide_positions=torch.tensor([2]))
    assert torch.allclose(batched.first_hidden_state[1, :3], single.first_hidden_state[0], atol=1e-6)
    assert torch.allclose(batched.decide_hidden_state[1], single.decide_hidden_state[0], atol=1e-6)


def test_helpers_accept_loop_config():
    config = LoopConfig("coconut", 1)
    backbone, _ = make_fake()
    with_config = coconut_forward(backbone, IDS, POS, MASK, config=config)
    backbone, _ = make_fake()
    with_steps = coconut_forward(backbone, IDS, POS, MASK, 1)
    assert torch.allclose(with_config.last_hidden_state, with_steps.last_hidden_state)
    with pytest.raises(ValueError, match="coconut"):
        coconut_forward(backbone, IDS, POS, MASK, config=LoopConfig("ponder", 1))
