"""Bounded latent-loop forwards for LoopJev.

The helpers run an attention-only causal backbone (anything that returns
``last_hidden_state``) and never touch Kev's pointer head or call generation.
``LoopConfig`` names the arm, bounds the recurrence, and bounds Ponder's
vocabulary softmax; the forward variants are COCONUT, PonderLM, and the
separate decide-only and option+decision Ponder ablations. With
``K = config.steps`` the ordinary COCONUT/Ponder forwards make ``K + 1``
backbone passes; decide-only Ponder adds one ordinary pass (``K + 2``), while
option+decision Ponder stays at ``K + 1``. ``K == 0`` is exactly one ordinary
``input_ids`` pass so the baseline arm stays numerically identical.

Ponder has two explicitly named initializations. The defaults preserve
LoopJev's existing recurrence: unscaled embeddings, ``current = E0``, a
top-100 vocabulary softmax, and ``current = E0 + sum(expected)``. The opt-in
``ponder_scale_embeds`` / ``ponder_multiply_steps`` flags reproduce the
official positive-step initialization (``E0 * sqrt(hidden)`` and, optionally,
``E0 * steps``), and ``ponder_full_vocab`` selects the official
full-vocabulary softmax instead of the top-k approximation. All three default
off; see ``LoopConfig.official_style`` for the named combination.

``ponder_decide_only`` is a separate, explicitly experimental ablation: the
first pass is ordinary, only the ``<decide>`` input embedding is pondered, and
the pointer head reads ordinary option states with the final pondered
``<decide>`` state as its query. It is not official PonderLM fidelity; see
``kev/docs/loopjev-decide-only.md``. It defaults off and cannot be combined
with ``ponder_preserve_options``.

``ponder_option_decide_only`` is the readout-endpoint ablation: no ordinary
pass, and only the option-endpoint and ``<decide>`` input embeddings are
pondered, so the pointer query and option keys both come from the final pass.
State and every other input embedding stay exactly ``E0`` on every pass. It
requires a boolean ``update_mask``, defaults off, needs ``method='ponder'``, and
is mutually exclusive with ``ponder_decide_only`` and
``ponder_preserve_options``; see ``kev/docs/loopjev-option-decide-only.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

METHODS = ("kev", "coconut", "ponder")
DEFAULT_PONDER_TOP_K = 100


@dataclass(frozen=True, slots=True, init=False)
class LoopConfig:
    """The arm, its nonnegative number of latent steps, and Ponder's recurrence.

    ``kev`` is the unmodified forward and therefore requires ``steps=0``.
    ``coconut`` and ``ponder`` allow zero steps (which is again the ordinary
    forward) and any positive bound. ``ponder_top_k`` bounds the vocabulary
    softmax of the Ponder recurrence (``top_k`` is accepted as an alias); it is
    validated for every arm even though only ``ponder`` reads it.
    ``ponder_full_vocab`` (or ``ponder_top_k=None``) selects the full-vocabulary
    softmax instead. ``ponder_scale_embeds`` scales the input embeddings and the
    interpolation weight by ``sqrt(hidden_size)``; ``ponder_multiply_steps``
    initializes the running embeddings at ``E0 * steps``. Both initialization
    flags default false and, together with full vocabulary, are needed to
    represent the official positive-step path. They are explicit settings, not
    a claim that a Qwen checkpoint carries the official checkpoint flags.

    ``ponder_decide_only`` (default false) is the decision-aware ablation: an
    ordinary first pass, then the same initialization/update settings applied
    only to the ``<decide>`` embedding. It requires ``method='ponder'`` and is
    not part of the official path.

    ``ponder_option_decide_only`` (default false) is the readout-endpoint
    ablation: no ordinary pass, and the same initialization/update settings
    apply only to the option endpoints and ``<decide>`` positions selected by a
    boolean ``update_mask``. It requires ``method='ponder'`` and cannot be
    combined with ``ponder_decide_only``.
    """
    method: str
    steps: int
    ponder_top_k: int | None
    ponder_scale_embeds: bool
    ponder_multiply_steps: bool
    ponder_full_vocab: bool
    ponder_decide_only: bool
    ponder_option_decide_only: bool
    coconut_start_layer: int

    def __init__(self, method: str = "kev", steps: int = 0,
                 ponder_top_k: int | None = DEFAULT_PONDER_TOP_K, *, top_k: int | None = None,
                 ponder_scale_embeds: bool = False, ponder_multiply_steps: bool = False,
                 ponder_full_vocab: bool = False, ponder_decide_only: bool = False,
                 ponder_option_decide_only: bool = False, coconut_start_layer: int = 0):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise ValueError(f"steps must be an int, got {type(steps).__name__}")
        if steps < 0:
            raise ValueError(f"steps must be nonnegative, got {steps}")
        if method == "kev" and steps != 0:
            raise ValueError("method 'kev' requires steps=0")
        for name, value in (("ponder_scale_embeds", ponder_scale_embeds),
                            ("ponder_multiply_steps", ponder_multiply_steps),
                            ("ponder_full_vocab", ponder_full_vocab),
                            ("ponder_decide_only", ponder_decide_only),
                            ("ponder_option_decide_only", ponder_option_decide_only)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool, got {type(value).__name__}")
        if ponder_decide_only and method != "ponder":
            raise ValueError("ponder_decide_only requires method='ponder'")
        if ponder_option_decide_only and method != "ponder":
            raise ValueError("ponder_option_decide_only requires method='ponder'")
        if isinstance(coconut_start_layer, bool) or not isinstance(coconut_start_layer, int):
            raise ValueError(f"coconut_start_layer must be an int, got {type(coconut_start_layer).__name__}")
        if coconut_start_layer < 0:
            raise ValueError(f"coconut_start_layer must be nonnegative, got {coconut_start_layer}")
        if coconut_start_layer and method != "coconut":
            raise ValueError("coconut_start_layer requires method='coconut'")
        if ponder_decide_only and ponder_option_decide_only:
            raise ValueError("ponder_decide_only and ponder_option_decide_only are separate ablations; enable at most one")
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int):
                raise ValueError(f"top_k must be an int, got {type(top_k).__name__}")
            if top_k <= 0:
                raise ValueError(f"top_k must be positive, got {top_k}")
            if ponder_top_k is not None and ponder_top_k != DEFAULT_PONDER_TOP_K and ponder_top_k != top_k:
                raise ValueError("top_k and ponder_top_k disagree")
            if ponder_full_vocab:
                raise ValueError("top_k and ponder_full_vocab disagree")
            ponder_top_k = top_k
        if ponder_full_vocab:
            if ponder_top_k is not None and ponder_top_k != DEFAULT_PONDER_TOP_K:
                raise ValueError("ponder_top_k and ponder_full_vocab disagree")
            ponder_top_k = None
        if ponder_top_k is not None:
            if isinstance(ponder_top_k, bool) or not isinstance(ponder_top_k, int):
                raise ValueError(f"ponder_top_k must be an int, got {type(ponder_top_k).__name__}")
            if ponder_top_k <= 0:
                raise ValueError(f"ponder_top_k must be positive, got {ponder_top_k}")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "ponder_top_k", ponder_top_k)
        object.__setattr__(self, "ponder_scale_embeds", ponder_scale_embeds)
        object.__setattr__(self, "ponder_multiply_steps", ponder_multiply_steps)
        object.__setattr__(self, "ponder_full_vocab", ponder_top_k is None)
        object.__setattr__(self, "ponder_decide_only", ponder_decide_only)
        object.__setattr__(self, "ponder_option_decide_only", ponder_option_decide_only)
        object.__setattr__(self, "coconut_start_layer", coconut_start_layer)

    @classmethod
    def official_style(cls, steps: int, *, top_k: int | None = None,
                       ponder_decide_only: bool = False) -> "LoopConfig":
        """Named official-style positive-step configuration.

        Sets the released initialization/update settings explicitly:
        ``scale_embeds``, ``mutiply_pondering_steps`` (the upstream key's
        spelling), and ``current += expected`` each step. The default
        ``top_k=None`` also selects the official full-vocabulary softmax; pass
        a positive ``top_k`` to keep the cheaper approximation with the same
        initialization. ``ponder_decide_only`` selects the separate ablation
        with the same initialization; it is not official. Qwen2.5 checkpoints
        do not carry the official checkpoint flags, so this is an explicit
        emulation, not evidence that a Qwen checkpoint was trained with them.
        """
        if top_k is None:
            return cls("ponder", steps, ponder_scale_embeds=True, ponder_multiply_steps=True,
                       ponder_full_vocab=True, ponder_decide_only=ponder_decide_only)
        return cls("ponder", steps, top_k, ponder_scale_embeds=True, ponder_multiply_steps=True,
                   ponder_decide_only=ponder_decide_only)

    @property
    def top_k(self) -> int | None:
        """Alias for ``ponder_top_k`` (``None`` = full vocabulary)."""
        return self.ponder_top_k

    @property
    def passes(self) -> int:
        """Backbone passes the arm's forward makes.

        ``steps + 1`` ordinarily. Positive-step decide-only Ponder makes one
        extra ordinary pass first, so it makes ``steps + 2``; at zero steps it
        is still the exact one-pass ordinary forward. Option+decision Ponder
        makes ``steps + 1``: no ordinary pass, then the final pass.
        """
        if self.ponder_decide_only and self.steps > 0:
            return self.steps + 2
        return self.steps + 1


@dataclass(frozen=True, slots=True)
class CoconutOutput:
    """COCONUT result.

    last_hidden_state: the final pass, including the appended latent tokens.
    token_hidden_states: the ordinary first pass, i.e. the original tokens'
        hidden states (Kev's option-end states are read from these).
    latent_hidden_state: the final pass's final position, the latent query.
    passes: number of backbone forwards (``steps + 1``).
    """
    last_hidden_state: torch.Tensor
    token_hidden_states: torch.Tensor
    latent_hidden_state: torch.Tensor
    passes: int

    @property
    def original_hidden_states(self) -> torch.Tensor:
        return self.token_hidden_states

    @property
    def final_hidden_state(self) -> torch.Tensor:
        return self.last_hidden_state

    @property
    def query_hidden_state(self) -> torch.Tensor:
        return self.latent_hidden_state

    @property
    def num_passes(self) -> int:
        return self.passes

    @property
    def pass_count(self) -> int:
        return self.passes


@dataclass(frozen=True, slots=True)
class PonderOutput:
    """PonderLM result: the final pass's hidden states and its pass count.

    ``first_hidden_state`` is the ordinary first pass in decide-only mode
    (its state and option states are the pointer head's keys); it is ``None``
    for the ordinary Ponder recurrence and for option+decision Ponder, whose
    ``token_hidden_states`` alias stays the final pass exactly as before.
    ``decide_hidden_state`` is the final pass's ``<decide>`` state in decide-only
    or option+decision mode when a decide position was supplied, else ``None``.
    """
    last_hidden_state: torch.Tensor
    passes: int
    first_hidden_state: torch.Tensor | None = None
    decide_hidden_state: torch.Tensor | None = None

    @property
    def hidden_state(self) -> torch.Tensor:
        return self.last_hidden_state

    @property
    def final_hidden_state(self) -> torch.Tensor:
        return self.last_hidden_state

    @property
    def token_hidden_states(self) -> torch.Tensor:
        """The ordinary first pass when there is one, else the final pass (legacy Ponder)."""
        return self.last_hidden_state if self.first_hidden_state is None else self.first_hidden_state

    @property
    def original_hidden_states(self) -> torch.Tensor:
        return self.token_hidden_states

    @property
    def ordinary_hidden_states(self) -> torch.Tensor:
        return self.token_hidden_states

    @property
    def query_hidden_state(self) -> torch.Tensor:
        return self.last_hidden_state[:, -1, :] if self.decide_hidden_state is None else self.decide_hidden_state

    @property
    def num_passes(self) -> int:
        return self.passes

    @property
    def pass_count(self) -> int:
        return self.passes


def _resolve_steps(steps: Any, config: LoopConfig | None, K: int | None, method: str) -> int:
    """Read ``steps`` (or a LoopConfig / K alias) for one helper and validate it."""
    if config is not None:
        if not isinstance(config, LoopConfig):
            raise ValueError("config must be a LoopConfig")
        if config.method != method:
            raise ValueError(f"{method} helper needs a {method!r} LoopConfig, got {config.method!r}")
        if K is not None and K != config.steps:
            raise ValueError("K and config.steps disagree")
        if steps not in (0, config.steps):
            raise ValueError("steps and config.steps disagree")
        return config.steps
    if isinstance(steps, LoopConfig):
        if steps.method != method:
            raise ValueError(f"{method} helper needs a {method!r} LoopConfig, got {steps.method!r}")
        steps = steps.steps
    if K is not None:
        if isinstance(K, bool) or not isinstance(K, int):
            raise ValueError(f"K must be an int, got {type(K).__name__}")
        if steps not in (0, K):
            raise ValueError("steps and K disagree")
        steps = K
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ValueError(f"steps must be an int, got {type(steps).__name__}")
    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}")
    return steps


def _resolve_top_k(top_k: Any, ponder_top_k: Any, config: LoopConfig | None,
                   steps: Any, full_vocab: bool = False) -> int | None:
    """Read the Ponder softmax bound from an explicit alias, a LoopConfig, or the default.

    ``None`` means the full-vocabulary softmax. ``full_vocab=True`` is the
    separately named option and must not be combined with an explicit top-k.
    """
    if not isinstance(full_vocab, bool):
        raise ValueError(f"ponder_full_vocab must be a bool, got {type(full_vocab).__name__}")
    for name, value in (("top_k", top_k), ("ponder_top_k", ponder_top_k)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if top_k is not None and ponder_top_k is not None and top_k != ponder_top_k:
        raise ValueError("top_k and ponder_top_k disagree")
    explicit = top_k if top_k is not None else ponder_top_k
    source = config if config is not None else (steps if isinstance(steps, LoopConfig) else None)
    source_top_k = None if source is None else source.ponder_top_k
    if full_vocab:
        if explicit is not None:
            raise ValueError("full vocabulary and top_k disagree")
        if source is not None and source_top_k is not None:
            raise ValueError("full vocabulary and config.ponder_top_k disagree")
        return None
    if source is not None:
        if explicit is not None and explicit != source_top_k:
            raise ValueError("top_k and config.ponder_top_k disagree")
        return source_top_k
    if explicit is None:
        return DEFAULT_PONDER_TOP_K
    return explicit


def _resolve_ponder_flags(scale_embeds: Any, multiply_steps: Any, full_vocab: Any,
                          decide_only: Any, option_decide_only: Any, config: LoopConfig | None,
                          steps: Any) -> tuple[bool, bool, bool, bool, bool]:
    """Read the explicit Ponder initialization/vocabulary/readout flags (defaults all false)."""
    for name, value in (("ponder_scale_embeds", scale_embeds),
                        ("ponder_multiply_steps", multiply_steps),
                        ("ponder_full_vocab", full_vocab),
                        ("ponder_decide_only", decide_only),
                        ("ponder_option_decide_only", option_decide_only)):
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{name} must be a bool, got {type(value).__name__}")
    source = config if config is not None else (steps if isinstance(steps, LoopConfig) else None)

    def pick(name: str, explicit: bool | None, attr: str) -> bool:
        if source is None:
            return False if explicit is None else explicit
        configured = getattr(source, attr)
        if explicit is not None and explicit != configured:
            raise ValueError(f"{name} and config.{attr} disagree")
        return configured

    flags = (pick("ponder_scale_embeds", scale_embeds, "ponder_scale_embeds"),
             pick("ponder_multiply_steps", multiply_steps, "ponder_multiply_steps"),
             pick("ponder_full_vocab", full_vocab, "ponder_full_vocab"),
             pick("ponder_decide_only", decide_only, "ponder_decide_only"),
             pick("ponder_option_decide_only", option_decide_only, "ponder_option_decide_only"))
    if flags[3] and flags[4]:
        raise ValueError("ponder_decide_only and ponder_option_decide_only are separate ablations; enable at most one")
    return flags


def _resolve_update_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor | None,
                         update_mask: Any) -> torch.Tensor:
    """Validate the boolean ``[B, L]`` positions the option+decision recurrence updates.

    The mask must select at least one position in every row and must never select
    a padded (masked) position. It is returned as bool on ``input_ids``'s device.
    """
    if update_mask is None:
        raise ValueError("ponder_option_decide_only requires a boolean update_mask [B, L]")
    mask = torch.as_tensor(update_mask, device=input_ids.device)
    if tuple(mask.shape) != tuple(input_ids.shape):
        raise ValueError(f"update_mask must match input_ids {tuple(input_ids.shape)}, got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise ValueError(f"update_mask must be a boolean tensor, got {mask.dtype}")
    if attention_mask is not None:
        padded = mask & (attention_mask == 0)
        if bool(padded.any()):
            raise ValueError("update_mask must not select padded positions")
    if not bool(mask.any()):
        raise ValueError("update_mask must select at least one position")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("update_mask must select at least one position in every row")
    return mask


def _resolve_decide_positions(input_ids: torch.Tensor,
                              decide_positions: torch.Tensor | None) -> torch.Tensor:
    """One ``<decide>`` position per row as a long tensor on the input's device.

    ``None`` means the last token of each row: the layout ``rows_of`` builds
    for one question (state + branch, ``<decide>`` last). Callers with another
    layout pass an explicit ``[B]`` tensor.
    """
    batch, length = input_ids.shape
    if decide_positions is None:
        return torch.full((batch,), length - 1, dtype=torch.long, device=input_ids.device)
    positions = torch.as_tensor(decide_positions, dtype=torch.long, device=input_ids.device)
    if positions.ndim == 0 and batch == 1:
        positions = positions.reshape(1)
    if positions.ndim == 2 and tuple(positions.shape) == (batch, 1):
        positions = positions[:, 0]
    if positions.ndim != 1 or positions.shape[0] != batch:
        raise ValueError(f"decide_positions must be [B] matching input_ids {tuple(input_ids.shape)}, "
                         f"got {tuple(positions.shape)}")
    if bool((positions < 0).any()) or bool((positions >= length).any()):
        raise ValueError(f"decide_positions must be in [0, {length}), got {positions.tolist()}")
    return positions


def _check_rows(input_ids: torch.Tensor, position_ids: torch.Tensor,
                attention_mask: torch.Tensor | None) -> None:
    """Rows must be equal-length and unpadded: a full mask only, no silent last-position guess."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
    if tuple(position_ids.shape) != tuple(input_ids.shape):
        raise ValueError(f"position_ids must match input_ids {tuple(input_ids.shape)}, got {tuple(position_ids.shape)}")
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids rows must be nonempty")
    if attention_mask is None:
        return
    if tuple(attention_mask.shape) != tuple(input_ids.shape):
        raise ValueError(f"attention_mask must match input_ids {tuple(input_ids.shape)}, got {tuple(attention_mask.shape)}")
    if not bool(torch.all(attention_mask == 1)):
        raise ValueError("attention_mask must be full (all ones); padded rows are not supported")


def _check_ponder_rows(input_ids: torch.Tensor, position_ids: torch.Tensor,
                       attention_mask: torch.Tensor | None) -> None:
    """Ponder accepts ordinary equal rows and right-padded batches.

    Unlike COCONUT, Ponder never appends a latent token after each row's real end, so masked right padding is exact. An
    interior hole is rejected: every row must be a nonempty prefix of ones followed only by zeros.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
    if tuple(position_ids.shape) != tuple(input_ids.shape):
        raise ValueError(f"position_ids must match input_ids {tuple(input_ids.shape)}, got {tuple(position_ids.shape)}")
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids rows must be nonempty")
    if attention_mask is None:
        return
    if tuple(attention_mask.shape) != tuple(input_ids.shape):
        raise ValueError(f"attention_mask must match input_ids {tuple(input_ids.shape)}, got {tuple(attention_mask.shape)}")
    active = attention_mask != 0
    lengths = active.sum(1)
    if bool((lengths == 0).any()):
        raise ValueError("attention_mask rows must contain at least one real token")
    expected = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] < lengths[:, None]
    if not bool(torch.equal(active, expected)):
        raise ValueError("attention_mask must be an all-ones prefix followed by right padding")


def _embedding_module(backbone: Any) -> torch.nn.Module:
    getter = getattr(backbone, "get_input_embeddings", None)
    if callable(getter):
        module = getter()
        if module is not None:
            return module
    module = getattr(backbone, "embed_tokens", None)
    if module is None:
        inner = getattr(backbone, "model", None)
        module = getattr(inner, "embed_tokens", None)
    if module is None:
        raise TypeError("backbone does not expose input embeddings (get_input_embeddings / embed_tokens)")
    return module


def _input_embeddings(backbone: Any, input_ids: torch.Tensor) -> torch.Tensor:
    return _embedding_module(backbone)(input_ids)


def _embedding_weight(backbone: Any) -> torch.Tensor:
    weight = getattr(_embedding_module(backbone), "weight", None)
    if weight is None:
        raise TypeError("backbone input embeddings have no weight")
    return weight


def _forward(backbone: Any, *, input_ids: torch.Tensor | None = None,
             inputs_embeds: torch.Tensor | None = None, position_ids: torch.Tensor,
             attention_mask: torch.Tensor | None) -> torch.Tensor:
    kwargs = {"position_ids": position_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if input_ids is not None:
        kwargs["input_ids"] = input_ids
    else:
        kwargs["inputs_embeds"] = inputs_embeds
    out = backbone(**kwargs)
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        raise TypeError("backbone must return an object with last_hidden_state")
    return hidden


def _resolve_coconut_start_layer(value: int | None, config: LoopConfig | None,
                                 steps: Any) -> int:
    source = config if config is not None else (steps if isinstance(steps, LoopConfig) else None)
    configured = 0 if source is None else source.coconut_start_layer
    if value is None:
        return configured
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"coconut_start_layer must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"coconut_start_layer must be nonnegative, got {value}")
    if source is not None and value != configured:
        raise ValueError("coconut_start_layer and config.coconut_start_layer disagree")
    return value


def _decoder_stack(backbone: Any) -> tuple[Any, Any]:
    layers = getattr(backbone, "layers", None)
    norm = getattr(backbone, "norm", None)
    if layers is None:
        inner = getattr(backbone, "model", None)
        layers = getattr(inner, "layers", None)
        norm = getattr(inner, "norm", None)
    if layers is None or norm is None:
        raise TypeError("partial-depth COCONUT needs a decoder backbone exposing layers and norm")
    return layers, norm


def _deep_coconut_hidden(backbone: Any, hidden_states: torch.Tensor,
                         position_ids: torch.Tensor,
                         attention_mask: torch.Tensor | None,
                         start_layer: int) -> torch.Tensor:
    """Run one pass through decoder layers ``[start_layer, num_layers)``.

    The input is already the hidden-state sequence after a full ordinary pass.
    This helper mirrors the Qwen decoder's mask construction and rotary setup;
    it intentionally has no token embeddings or prefix layers.
    """
    layers, norm = _decoder_stack(backbone)
    if start_layer >= len(layers):
        raise ValueError(f"coconut_start_layer {start_layer} must be below {len(layers)}")
    try:
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    except ImportError as error:
        raise TypeError("partial-depth COCONUT needs transformers.masking_utils") from error
    mask_kwargs = {
        "config": backbone.config,
        "inputs_embeds": hidden_states,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
    if bool(getattr(backbone, "has_sliding_layers", False)):
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    position_embeddings = backbone.rotary_emb(hidden_states, position_ids)
    layer_types = getattr(backbone.config, "layer_types", None) or ["full_attention"] * len(layers)
    for index in range(start_layer, len(layers)):
        hidden_states = layers[index](
            hidden_states,
            attention_mask=causal_mask_mapping[layer_types[index]],
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
    return norm(hidden_states)


def coconut_forward(backbone: Any, input_ids: torch.Tensor, position_ids: torch.Tensor,
                    attention_mask: torch.Tensor | None = None, steps: int = 0, *,
                    config: LoopConfig | None = None, K: int | None = None,
                    coconut_start_layer: int | None = None) -> CoconutOutput:
    """COCONUT row forward.

    The first pass is the ordinary ``input_ids`` pass. Each of the ``steps``
    recurrent passes appends the previous pass's final hidden state as the
    embedding of one new latent token, extends positions and the (full) mask,
    and reruns the whole row with ``inputs_embeds``. Gradients stay attached
    through every pass. Returns the ordinary pass's token states, the final
    pass, the final latent query, and the ``steps + 1`` pass count.

    ``steps`` may also be a LoopConfig; ``config`` and ``K`` are keyword
    equivalents for integrations that carry them explicitly. ``coconut_start_layer``
    selects an experimental partial-depth recurrence: the ordinary full first
    pass supplies token states, then each latent step reruns only decoder layers
    ``[start_layer, num_layers)`` on the growing hidden sequence. Zero is the
    standard full-backbone COCONUT recurrence.
    """
    K = _resolve_steps(steps, config, K, "coconut")
    start_layer = _resolve_coconut_start_layer(coconut_start_layer, config, steps)
    _check_rows(input_ids, position_ids, attention_mask)

    first = _forward(backbone, input_ids=input_ids, position_ids=position_ids,
                     attention_mask=attention_mask)
    if K == 0:
        return CoconutOutput(first, first, first[:, -1, :], 1)

    hidden = first
    positions = position_ids
    mask = attention_mask
    if start_layer == 0:
        embeds = _input_embeddings(backbone, input_ids)
        for _ in range(K):
            latent = hidden[:, -1, :]
            if latent.dtype != embeds.dtype:
                latent = latent.to(embeds.dtype)
            embeds = torch.cat((embeds, latent.unsqueeze(1)), dim=1)
            positions = torch.cat((positions, positions[:, -1:] + 1), dim=1)
            if mask is not None:
                mask = torch.cat((mask, mask.new_ones((mask.shape[0], 1))), dim=1)
            hidden = _forward(backbone, inputs_embeds=embeds, position_ids=positions,
                              attention_mask=mask)
    else:
        for _ in range(K):
            latent = hidden[:, -1, :]
            if latent.dtype != hidden.dtype:
                latent = latent.to(hidden.dtype)
            hidden = torch.cat((hidden, latent.unsqueeze(1)), dim=1)
            positions = torch.cat((positions, positions[:, -1:] + 1), dim=1)
            if mask is not None:
                mask = torch.cat((mask, mask.new_ones((mask.shape[0], 1))), dim=1)
            hidden = _deep_coconut_hidden(backbone, hidden, positions, mask, start_layer)
    return CoconutOutput(hidden, first, hidden[:, -1, :], K + 1)


def ponder_forward(backbone: Any, lm_head: Any, input_ids: torch.Tensor,
                   position_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                   steps: int = 0, top_k: int | None = None, *,
                   config: LoopConfig | None = None, K: int | None = None,
                   ponder_top_k: int | None = None,
                   ponder_scale_embeds: bool | None = None,
                   ponder_multiply_steps: bool | None = None,
                   ponder_full_vocab: bool | None = None,
                   ponder_decide_only: bool | None = None,
                   ponder_option_decide_only: bool | None = None,
                   update_mask: torch.Tensor | None = None,
                   decide_positions: torch.Tensor | None = None) -> PonderOutput:
    """PonderLM expected-embedding forward with explicit initialization settings.

    ``E0`` is the input embedding. With the defaults (all flags false) the
    recurrence is LoopJev's existing one: ``current = E0``, a top-100
    vocabulary softmax, and ``current = E0 + sum(expected)``. Each of the
    ``steps`` updates runs the backbone on the current embedding and projects
    every position through ``lm_head`` in float32. Only the
    ``top_k = min(configured, vocab)`` largest logits are kept (``torch.topk``),
    only those are softmaxed, and the expected embedding is the
    probability-weighted sum of their rows in the backbone's input-embedding
    weight.

    ``ponder_scale_embeds`` multiplies ``E0`` and the interpolation weight by
    ``sqrt(hidden_size)``. ``ponder_multiply_steps`` initializes
    ``current = E0 * steps``. Together with ``ponder_full_vocab`` (the official
    full-vocabulary softmax, instead of top-k) they represent the official
    positive-step path: optional scaling, optional multiply-by-steps
    initialization, then ``current = current + expected`` each step. One final
    backbone pass runs on the last embedding. Nothing is detached: gradients
    reach the backbone, the projected logits, the scaled interpolation weight,
    and the selected embedding rows.

    ``ponder_decide_only=True`` selects the decision-aware ablation instead: an
    ordinary ``input_ids`` pass, then the same initialization/update settings
    applied only to the ``<decide>`` embedding at ``decide_positions`` (one per
    row; ``None`` = the last token of each row). Every other input embedding
    stays exactly the original token embedding on every pass, only the
    ``<decide>`` hidden state is projected through ``lm_head``, and one final
    pass supplies the query. This is not official PonderLM and is incompatible
    with ``ponder_preserve_options`` (the model layer rejects both together).

    ``ponder_option_decide_only=True`` selects the readout-endpoint ablation:
    there is no ordinary pass. ``update_mask`` is a boolean ``[B, L]`` selecting
    the option endpoints and the ``<decide>`` position of every row (padded
    positions are rejected). Only those input embeddings are initialized and
    updated by the same scale/multiply/expected-embedding rules; every other
    embedding, the state included, stays exactly ``E0`` on every pass.
    ``lm_head`` sees only the selected hidden states (``[N, d]``), and the final
    pass supplies both the pointer query and the option keys. It requires
    ``method='ponder'`` and cannot be combined with ``ponder_decide_only``.

    ``steps`` may also be a LoopConfig; ``config`` and ``K`` are keyword
    equivalents for integrations that carry them explicitly. ``top_k`` (or its
    alias ``ponder_top_k``) sets the bound and must agree with a LoopConfig's
    ``ponder_top_k`` when one is given; otherwise the default is 100. Explicit
    ``ponder_*`` flags must agree with a LoopConfig's when one is given. Use
    ``ponder_full_vocab=True`` for the full-vocabulary softmax; passing
    ``ponder_top_k=None`` to this helper still means "unspecified", because
    ``None`` is also the sentinel for an omitted top-k argument.
    """
    K = _resolve_steps(steps, config, K, "ponder")
    scale_embeds, multiply_steps, full_vocab, decide_only, option_decide_only = _resolve_ponder_flags(
        ponder_scale_embeds, ponder_multiply_steps, ponder_full_vocab,
        ponder_decide_only, ponder_option_decide_only, config, steps)
    top_k = _resolve_top_k(top_k, ponder_top_k, config, steps, full_vocab)
    _check_ponder_rows(input_ids, position_ids, attention_mask)
    if update_mask is not None and not option_decide_only:
        raise ValueError("update_mask requires ponder_option_decide_only=True")
    positions = None
    if decide_only:
        positions = _resolve_decide_positions(input_ids, decide_positions)
    elif option_decide_only and decide_positions is not None:
        positions = _resolve_decide_positions(input_ids, decide_positions)
    selected = None
    if option_decide_only:
        if update_mask is None and K > 0:
            raise ValueError("ponder_option_decide_only requires a boolean update_mask [B, L]")
        if update_mask is not None:
            selected = _resolve_update_mask(input_ids, attention_mask, update_mask)
            if positions is not None:
                rows = torch.arange(selected.shape[0], device=selected.device)
                if not bool(selected[rows, positions].all()):
                    raise ValueError("update_mask must select every decide position")

    if K == 0:
        hidden = _forward(backbone, input_ids=input_ids, position_ids=position_ids,
                          attention_mask=attention_mask)
        if decide_only:
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            return PonderOutput(hidden, 1, first_hidden_state=hidden,
                                decide_hidden_state=hidden[rows, positions])
        return PonderOutput(hidden, 1)

    if decide_only:
        return _ponder_decide_recurrence(
            backbone, lm_head, input_ids, position_ids, attention_mask, positions,
            steps=K, top_k=top_k, scale_embeds=scale_embeds,
            multiply_steps=multiply_steps)

    if option_decide_only:
        result = _ponder_option_decide_recurrence(
            backbone, lm_head, input_ids, position_ids, attention_mask, selected,
            steps=K, top_k=top_k, scale_embeds=scale_embeds,
            multiply_steps=multiply_steps)
        if positions is None:
            return result
        rows = torch.arange(result.last_hidden_state.shape[0], device=result.last_hidden_state.device)
        return PonderOutput(result.last_hidden_state, result.passes,
                            decide_hidden_state=result.last_hidden_state[rows, positions])

    base = _input_embeddings(backbone, input_ids)
    weight = _embedding_weight(backbone)
    if scale_embeds:
        embed_scale = torch.sqrt(torch.tensor(base.shape[-1], dtype=base.dtype,
                                              device=base.device))
        base = base * embed_scale
        weight = weight * embed_scale
    current = base * K if multiply_steps else base
    anchored = not (scale_embeds or multiply_steps)   # preserve the legacy default's exact arithmetic
    accumulated = None
    for _ in range(K):
        hidden = _forward(backbone, inputs_embeds=current, position_ids=position_ids,
                          attention_mask=attention_mask)
        logits = lm_head(hidden).float()
        if top_k is None:
            # Official use_topk=False path: probabilities over the full vocabulary.
            probs = torch.softmax(logits, dim=-1)
            expected = torch.matmul(probs.to(weight.dtype), weight)
        else:
            k = min(top_k, logits.shape[-1])
            selected_logits, selected_ids = torch.topk(logits, k, dim=-1)
            probs = torch.softmax(selected_logits, dim=-1)
            selected = weight[selected_ids]   # [..., k, d]; autograd scatters into exactly these rows
            expected = (probs.to(selected.dtype).unsqueeze(-1) * selected).sum(dim=-2)
        if anchored:
            accumulated = expected if accumulated is None else accumulated + expected
            current = base + accumulated
        else:
            current = current + expected
    final = _forward(backbone, inputs_embeds=current, position_ids=position_ids,
                     attention_mask=attention_mask)
    return PonderOutput(final, K + 1)


def _ponder_decide_recurrence(backbone: Any, lm_head: Any, input_ids: torch.Tensor,
                              position_ids: torch.Tensor, attention_mask: torch.Tensor | None,
                              decide_positions: torch.Tensor, *, steps: int, top_k: int | None,
                              scale_embeds: bool, multiply_steps: bool) -> PonderOutput:
    """The decide-only recurrence shared by ``ponder_forward`` and ``ponder_decide_forward``.

    One ordinary ``input_ids`` pass supplies the state/option states. The
    original embedding is read once; each recurrent pass is a clone of it with
    only the ``<decide>`` row replaced. ``lm_head`` is applied to that row's
    hidden state alone ([B, vocab], never [B, L, vocab]). The final pass's
    ``<decide>`` hidden state is the pointer query. Nothing is detached.
    """
    first = _forward(backbone, input_ids=input_ids, position_ids=position_ids,
                     attention_mask=attention_mask)
    base = _input_embeddings(backbone, input_ids)
    weight = _embedding_weight(backbone)
    rows = torch.arange(base.shape[0], device=base.device)
    decide_base = base[rows, decide_positions]
    if scale_embeds:
        embed_scale = torch.sqrt(torch.tensor(base.shape[-1], dtype=base.dtype,
                                              device=base.device))
        decide_base = decide_base * embed_scale
        weight = weight * embed_scale
    decide_current = decide_base * steps if multiply_steps else decide_base
    anchored = not (scale_embeds or multiply_steps)   # same arithmetic as the ordinary recurrence
    accumulated = None

    def current_embeds():
        """A fresh embedding tensor: non-decide rows are bit-identical to E0."""
        current = base.clone()
        current[rows, decide_positions] = decide_current.to(base.dtype)
        return current

    for _ in range(steps):
        hidden = _forward(backbone, inputs_embeds=current_embeds(), position_ids=position_ids,
                          attention_mask=attention_mask)
        decide_hidden = hidden[rows, decide_positions]
        logits = lm_head(decide_hidden).float()
        if top_k is None:
            # Official use_topk=False path: probabilities over the full vocabulary.
            probs = torch.softmax(logits, dim=-1)
            expected = torch.matmul(probs.to(weight.dtype), weight)
        else:
            k = min(top_k, logits.shape[-1])
            selected_logits, selected_ids = torch.topk(logits, k, dim=-1)
            probs = torch.softmax(selected_logits, dim=-1)
            selected = weight[selected_ids]   # [B, k, d]; autograd scatters into exactly these rows
            expected = (probs.to(selected.dtype).unsqueeze(-1) * selected).sum(dim=-2)
        if anchored:
            # Legacy default: E0 + sum(expected), applied only to the decide row.
            accumulated = expected if accumulated is None else accumulated + expected
            decide_current = decide_base + accumulated
        else:
            # Official-style: current += expected, applied only to the decide row.
            decide_current = decide_current + expected
    final = _forward(backbone, inputs_embeds=current_embeds(), position_ids=position_ids,
                     attention_mask=attention_mask)
    return PonderOutput(final, steps + 2, first_hidden_state=first,
                        decide_hidden_state=final[rows, decide_positions])


def _ponder_option_decide_recurrence(backbone: Any, lm_head: Any, input_ids: torch.Tensor,
                                     position_ids: torch.Tensor,
                                     attention_mask: torch.Tensor | None,
                                     update_mask: torch.Tensor, *, steps: int,
                                     top_k: int | None, scale_embeds: bool,
                                     multiply_steps: bool) -> PonderOutput:
    """The option+decision recurrence: no ordinary pass, selected readout endpoints only.

    ``update_mask`` is a boolean ``[B, L]`` selecting exactly the option endpoints
    and the ``<decide>`` position of every row. Every nonselected input embedding
    is a clone of the original token embedding on every pass. Only the selected
    rows' hidden states go through ``lm_head`` (``[N, vocab]``, never
    ``[B, L, vocab]``), and the expected embeddings are scattered back to those
    selected positions. Nothing is detached.
    """
    base = _input_embeddings(backbone, input_ids)
    weight = _embedding_weight(backbone)
    selected_base = base[update_mask]
    if scale_embeds:
        embed_scale = torch.sqrt(torch.tensor(base.shape[-1], dtype=base.dtype,
                                              device=base.device))
        selected_base = selected_base * embed_scale
        weight = weight * embed_scale
    selected_current = selected_base * steps if multiply_steps else selected_base
    anchored = not (scale_embeds or multiply_steps)   # same arithmetic as the ordinary recurrence
    accumulated = None

    def current_embeds():
        """A fresh embedding tensor: every nonselected row is bit-identical to E0."""
        current = base.clone()
        current[update_mask] = selected_current.to(base.dtype)
        return current

    for _ in range(steps):
        hidden = _forward(backbone, inputs_embeds=current_embeds(), position_ids=position_ids,
                          attention_mask=attention_mask)
        selected_hidden = hidden[update_mask]
        logits = lm_head(selected_hidden).float()
        if top_k is None:
            # Official use_topk=False path: probabilities over the full vocabulary.
            probs = torch.softmax(logits, dim=-1)
            expected = torch.matmul(probs.to(weight.dtype), weight)
        else:
            k = min(top_k, logits.shape[-1])
            selected_logits, selected_ids = torch.topk(logits, k, dim=-1)
            probs = torch.softmax(selected_logits, dim=-1)
            selected = weight[selected_ids]   # [N, k, d]; autograd scatters into exactly these rows
            expected = (probs.to(selected.dtype).unsqueeze(-1) * selected).sum(dim=-2)
        if anchored:
            accumulated = expected if accumulated is None else accumulated + expected
            selected_current = selected_base + accumulated
        else:
            selected_current = selected_current + expected
    final = _forward(backbone, inputs_embeds=current_embeds(), position_ids=position_ids,
                     attention_mask=attention_mask)
    return PonderOutput(final, steps + 1)


def ponder_decide_forward(backbone: Any, lm_head: Any, input_ids: torch.Tensor,
                          position_ids: torch.Tensor, decide_positions: torch.Tensor | None = None,
                          attention_mask: torch.Tensor | None = None, steps: int = 0,
                          top_k: int | None = None, *, config: LoopConfig | None = None,
                          K: int | None = None, ponder_top_k: int | None = None,
                          ponder_scale_embeds: bool | None = None,
                          ponder_multiply_steps: bool | None = None,
                          ponder_full_vocab: bool | None = None) -> PonderOutput:
    """Explicit entry point for the decide-only Ponder ablation.

    Equivalent to ``ponder_forward(..., ponder_decide_only=True, ...)``; this
    name always selects decide-only, so a ``config`` only supplies its steps,
    top-k, and initialization flags. ``decide_positions`` is one position per
    row (``None`` = each row's last token).
    """
    K = _resolve_steps(steps, config, K, "ponder")
    scale_embeds, multiply_steps, full_vocab, _, _ = _resolve_ponder_flags(
        ponder_scale_embeds, ponder_multiply_steps, ponder_full_vocab, None, None, config, steps)
    top_k = _resolve_top_k(top_k, ponder_top_k, config, steps, full_vocab)
    _check_ponder_rows(input_ids, position_ids, attention_mask)
    positions = _resolve_decide_positions(input_ids, decide_positions)
    if K == 0:
        hidden = _forward(backbone, input_ids=input_ids, position_ids=position_ids,
                          attention_mask=attention_mask)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return PonderOutput(hidden, 1, first_hidden_state=hidden,
                            decide_hidden_state=hidden[rows, positions])
    return _ponder_decide_recurrence(
        backbone, lm_head, input_ids, position_ids, attention_mask, positions,
        steps=K, top_k=top_k, scale_embeds=scale_embeds, multiply_steps=multiply_steps)


# Explicit names for call sites that describe the mechanism rather than the arm.
coconut_row_forward = coconut_forward
ponder_lm_forward = ponder_forward

__all__ = ["LoopConfig", "CoconutOutput", "PonderOutput", "coconut_forward",
           "ponder_forward", "ponder_decide_forward", "coconut_row_forward",
           "ponder_lm_forward", "METHODS", "DEFAULT_PONDER_TOP_K"]
