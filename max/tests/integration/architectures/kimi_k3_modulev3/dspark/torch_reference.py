# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Plain-torch reference of the published Kimi-K3-DSpark drafter forward.

Re-expresses TorchSpec's ``K3DSparkModel`` (``torchspec/models/draft/
dflash.py`` backbone, ``torchspec/models/draft/dspark.py`` heads, the MLA
attention ``K3DSparkMLAAttention`` borrows from ``draft/deepseek_eagle.py``,
and the YaRN tables from ``draft/llama3_eagle.py``) as an explicit,
single-request CPU oracle for the K3 ModuleV3 spec-decode lanes. One forward
drafts a whole block diffusion-style:

- The block embeds ``[anchor, mask x block_size-1]`` and every slot predicts
  the token one position past its own — TorchSpec supervises all
  ``block_size`` slots with the anchor seeding slot 0
  (``torchspec/models/dspark.py:134-136``), so a 7-row block drafts 7 tokens.
- Attention is bidirectional inside the block (DFlash's mask runs with
  ``is_causal=False``); the context taps are visible only strictly before
  the anchor's position (``torchspec/models/draft/dflash.py:70``).
- The Markov head adds a bigram bias conditioned on the previous token; at
  inference the chain threads each slot's greedy pick into the next slot's
  bias, exactly the loop ``deepseekV41/dspark_drafter.py::draft`` runs in
  MAX.

Loads validate every checkpoint tensor name and shape against
:func:`expected_state_dict_shapes`, so a load doubles as a format audit. The
confidence head (adaptive block length — out of scope for the fixed 7-token
blocks) is validated but never computed. Runs in float32 on CPU by default.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F

DEFAULT_BLOCK_SIZE = 7

__all__ = [
    "DSparkReferenceConfig",
    "K3DSparkReference",
    "RopeSpec",
    "expected_state_dict_shapes",
]


def _int(config: Mapping[str, object], key: str) -> int:
    value = config[key]
    assert isinstance(value, int) and not isinstance(value, bool), (
        f"dspark config {key!r}: expected int, got {value!r}"
    )
    return value


def _int_or(config: Mapping[str, object], key: str, default: int) -> int:
    return _int(config, key) if key in config else default


def _float_or(
    config: Mapping[str, object], key: str, default: float | None
) -> float:
    if key not in config:
        assert default is not None, f"dspark config is missing {key!r}"
        return default
    value = config[key]
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"dspark config {key!r}: expected number, got {value!r}"
    )
    return float(value)


def _bool_or(config: Mapping[str, object], key: str, default: bool) -> bool:
    value = config.get(key, default)
    assert isinstance(value, bool), f"dspark config {key!r}: {value!r}"
    return value


@dataclass(frozen=True)
class RopeSpec:
    """The rotary tables' parameters, parsed from the rope config."""

    dim: int
    base: float
    yarn: bool
    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float
    mscale: float
    mscale_all_dim: float


@dataclass(frozen=True)
class DSparkReferenceConfig:
    """The drafter geometry the reference needs, parsed from ``config.json``.

    Field names follow the published checkpoint's keys; ``target_hidden_size``
    is the target model's width (the taps' width) and ``num_taps`` is the
    tap count, parsed from ``num_target_layers`` — not fixed at five.
    """

    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_head_dim: int
    rms_norm_eps: float
    vocab_size: int
    mask_token_id: int
    num_taps: int
    target_hidden_size: int
    markov_rank: int
    enable_confidence_head: bool
    confidence_head_with_markov: bool
    block_size: int
    rope: RopeSpec

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, object]
    ) -> DSparkReferenceConfig:
        q_lora_rank = _int_or(config, "q_lora_rank", 1536)
        assert q_lora_rank > 0, (
            "q_lora_rank=None (the direct q_proj path) is not exercised by "
            "any published K3 DSpark checkpoint"
        )
        qk_nope = _int(config, "qk_nope_head_dim")
        qk_rope = _int(config, "qk_rope_head_dim")
        markov_rank = _int_or(config, "markov_rank", 0)
        with_markov = _bool_or(config, "confidence_head_with_markov", True)
        enable_confidence = _bool_or(config, "enable_confidence_head", False)
        assert not (enable_confidence and with_markov and markov_rank == 0), (
            "confidence_head_with_markov=True requires a Markov head "
            "(markov_rank > 0)"
        )
        hidden_size = _int(config, "hidden_size")
        return cls(
            hidden_size=hidden_size,
            intermediate_size=_int(config, "intermediate_size"),
            num_layers=_int(config, "num_hidden_layers"),
            num_heads=_int(config, "num_attention_heads"),
            q_lora_rank=q_lora_rank,
            kv_lora_rank=_int(config, "kv_lora_rank"),
            qk_nope_head_dim=qk_nope,
            qk_rope_head_dim=qk_rope,
            v_head_dim=_int(config, "v_head_dim"),
            qk_head_dim=qk_nope + qk_rope,
            rms_norm_eps=_float_or(config, "rms_norm_eps", 1e-6),
            vocab_size=_int(config, "vocab_size"),
            mask_token_id=_int(config, "mask_token_id"),
            num_taps=_int_or(config, "num_target_layers", 5),
            target_hidden_size=_int_or(
                config, "target_hidden_size", hidden_size
            ),
            markov_rank=markov_rank,
            enable_confidence_head=enable_confidence,
            confidence_head_with_markov=with_markov,
            block_size=_int_or(config, "block_size", DEFAULT_BLOCK_SIZE),
            rope=_rope_spec(config, qk_rope),
        )


def _plain_rope_spec(dim: int, base: float) -> RopeSpec:
    """The un-interpolated spec: raw frequencies, no YaRN correction."""
    return RopeSpec(
        dim=dim,
        base=base,
        yarn=False,
        factor=1.0,
        original_max_position_embeddings=0,
        beta_fast=32.0,
        beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=0.0,
    )


def _rope_spec(config: Mapping[str, object], rope_dim: int) -> RopeSpec:
    params_obj = config.get("rope_parameters")
    if params_obj is None:
        params_obj = config.get("rope_scaling")
    base = _float_or(config, "rope_theta", 10000.0)
    if params_obj is None:
        return _plain_rope_spec(rope_dim, base)
    assert isinstance(params_obj, dict), (
        f"rope_parameters: expected a dict, got {params_obj!r}"
    )
    params: Mapping[str, object] = params_obj
    base = _float_or(
        config, "rope_theta", _float_or(params, "rope_theta", 10000.0)
    )
    rope_type = params.get("rope_type", params.get("type"))
    if rope_type in (None, "default"):
        return _plain_rope_spec(rope_dim, base)
    if rope_type != "yarn":
        raise NotImplementedError(
            f"rope_type {rope_type!r}: only 'yarn' is published for K3 DSpark"
        )
    return RopeSpec(
        dim=rope_dim,
        base=base,
        yarn=True,
        factor=_float_or(params, "factor", None),
        original_max_position_embeddings=_int(
            params, "original_max_position_embeddings"
        ),
        beta_fast=_float_or(params, "beta_fast", 32.0),
        beta_slow=_float_or(params, "beta_slow", 1.0),
        mscale=_float_or(params, "mscale", 1.0),
        mscale_all_dim=_float_or(params, "mscale_all_dim", 0.0),
    )


def _yarn_get_mscale(scale: float, mscale: float) -> float:
    """``yarn_get_mscale`` from ``torchspec/models/draft/llama3_eagle.py``."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_inv_freq(spec: RopeSpec) -> torch.Tensor:
    """YaRN-blended inverse frequencies, one per rotated pair.

    ``LlamaYarnRotaryEmbedding._set_cos_sin_cache``: fast pairs keep the raw
    table, the slowest pairs interpolate by 1/factor, and the correction
    range (``yarn_find_correction_range``) blends between them.
    """
    dim = spec.dim
    exponent = torch.arange(0, dim, 2, dtype=torch.float32) / dim
    # TorchSpec's exact expressions: the reciprocal and the factor inside
    # it, not a negative exponent or a later divide. b**(-e) and b**e then
    # 1/f drift by ~1 ulp, which grows into visible cos/sin error once a
    # far position multiplies it up (1.2e-3 at position 60000).
    freq_extra = 1.0 / (spec.base**exponent)
    freq_inter = 1.0 / (spec.factor * spec.base**exponent)

    def correction_dim(rotations: float) -> float:
        return (
            dim
            * math.log(
                spec.original_max_position_embeddings
                / (rotations * 2 * math.pi)
            )
            / (2 * math.log(spec.base))
        )

    low = max(math.floor(correction_dim(spec.beta_fast)), 0)
    high = min(math.ceil(correction_dim(spec.beta_slow)), dim - 1)
    # yarn_linear_ramp_mask bumps the ceiling by 0.001 at a singular range.
    ceiling = high if low < high else low + 0.001
    ramp = torch.clamp(
        (torch.arange(dim // 2, dtype=torch.float32) - low) / (ceiling - low),
        0.0,
        1.0,
    )
    return freq_inter * ramp + freq_extra * (1.0 - ramp)


def _apply_interleaved_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Interleaved-pair RoPE, the DeepSeek MLA convention.

    Consecutive pairs ``(2i, 2i+1)`` rotate by the i-th table angle
    (``_apply_rope_interleaved`` in ``torchspec/models/draft/dspark.py``).
    ``cos``/``sin`` carry one entry per pair and broadcast over ``x``.
    """
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin), dim=-1
    )
    return rotated.flatten(-2)


def expected_state_dict_shapes(
    config: Mapping[str, object],
) -> dict[str, tuple[int, ...]]:
    """Every checkpoint tensor the reference consumes, with its exact shape.

    Keys match the published ``Inferact/Kimi-K3-DSpark`` safetensors names,
    so callers can audit a checkpoint's mapping outright.
    """
    cfg = DSparkReferenceConfig.from_mapping(config)
    shapes: dict[str, tuple[int, ...]] = {
        "context_proj.weight": (
            cfg.hidden_size,
            cfg.num_taps * cfg.target_hidden_size,
        ),
        "context_norm.weight": (cfg.hidden_size,),
        "embed_tokens.weight": (cfg.vocab_size, cfg.hidden_size),
        "final_norm.weight": (cfg.hidden_size,),
    }
    for i in range(cfg.num_layers):
        p = f"layers.{i}."
        shapes[p + "self_attn.q_a_proj.weight"] = (
            cfg.q_lora_rank,
            cfg.hidden_size,
        )
        shapes[p + "self_attn.q_a_layernorm.weight"] = (cfg.q_lora_rank,)
        shapes[p + "self_attn.q_b_proj.weight"] = (
            cfg.num_heads * cfg.qk_head_dim,
            cfg.q_lora_rank,
        )
        shapes[p + "self_attn.kv_a_proj_with_mqa.weight"] = (
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            cfg.hidden_size,
        )
        shapes[p + "self_attn.kv_a_layernorm.weight"] = (cfg.kv_lora_rank,)
        shapes[p + "self_attn.kv_b_proj.weight"] = (
            cfg.num_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            cfg.kv_lora_rank,
        )
        shapes[p + "self_attn.o_proj.weight"] = (
            cfg.hidden_size,
            cfg.num_heads * cfg.v_head_dim,
        )
        shapes[p + "input_layernorm.weight"] = (cfg.hidden_size,)
        shapes[p + "post_attention_layernorm.weight"] = (cfg.hidden_size,)
        shapes[p + "mlp.gate_proj.weight"] = (
            cfg.intermediate_size,
            cfg.hidden_size,
        )
        shapes[p + "mlp.up_proj.weight"] = (
            cfg.intermediate_size,
            cfg.hidden_size,
        )
        shapes[p + "mlp.down_proj.weight"] = (
            cfg.hidden_size,
            cfg.intermediate_size,
        )
    if cfg.markov_rank > 0:
        shapes["markov_head.markov_w1.weight"] = (
            cfg.vocab_size,
            cfg.markov_rank,
        )
        shapes["markov_head.markov_w2.weight"] = (
            cfg.vocab_size,
            cfg.markov_rank,
        )
    if cfg.enable_confidence_head:
        # AcceptRatePredictor's input is the hidden state, plus the previous
        # token's markov_w1 row when confidence_head_with_markov is set.
        conf_in = cfg.hidden_size
        if cfg.confidence_head_with_markov:
            conf_in += cfg.markov_rank
        shapes["confidence_head.proj.weight"] = (1, conf_in)
        shapes["confidence_head.proj.bias"] = (1,)
    return shapes


class _LayerWeights(NamedTuple):
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor
    q_a_proj: torch.Tensor
    q_a_layernorm: torch.Tensor
    q_b_proj: torch.Tensor
    kv_a_proj_with_mqa: torch.Tensor
    kv_a_layernorm: torch.Tensor
    kv_b_proj: torch.Tensor
    o_proj: torch.Tensor


class K3DSparkReference:
    """The published K3 DSpark drafter as an explicit torch forward.

    ``draft_logits`` scores one block with a teacher-forced Markov bias;
    ``greedy_block`` draws the block the way inference does, threading each
    slot's greedy pick into the next slot's bias. Float32 CPU by default;
    every tensor is validated on load.
    """

    def __init__(
        self,
        config: Mapping[str, object],
        weights: Mapping[str, torch.Tensor],
        lm_head: torch.Tensor,
        *,
        block_size: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.cfg = DSparkReferenceConfig.from_mapping(config)
        self.dtype = dtype
        self.block_size = (
            self.cfg.block_size if block_size is None else int(block_size)
        )

        expected = expected_state_dict_shapes(config)
        missing = sorted(set(expected) - set(weights))
        unexpected = sorted(set(weights) - set(expected))
        if missing or unexpected:
            raise ValueError(
                f"checkpoint tensors do not match the reference: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for name, shape in expected.items():
            tensor = weights[name]
            assert isinstance(tensor, torch.Tensor), (
                f"{name}: expected a torch.Tensor"
            )
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name}: expected shape {shape}, got {tuple(tensor.shape)}"
                )
        self._w = {name: weights[name].to(dtype) for name in sorted(expected)}

        assert isinstance(lm_head, torch.Tensor), "lm_head: expected a tensor"
        if tuple(lm_head.shape) != (self.cfg.vocab_size, self.cfg.hidden_size):
            raise ValueError(
                f"lm_head: expected [vocab={self.cfg.vocab_size}, "
                f"hidden={self.cfg.hidden_size}], got {tuple(lm_head.shape)}"
            )
        # The published drafter carries no lm_head; it borrows its target's.
        self.lm_head_weight = lm_head.to(dtype)

        self._layers = [
            self._layer_weights(i) for i in range(self.cfg.num_layers)
        ]

        spec = self.cfg.rope
        if spec.yarn:
            self._inv_freq = _yarn_inv_freq(spec)
            attn_mscale = _yarn_get_mscale(spec.factor, spec.mscale_all_dim)
            self._rope_mscale = (
                _yarn_get_mscale(spec.factor, spec.mscale) / attn_mscale
            )
            self.softmax_scale = attn_mscale**2 / math.sqrt(
                self.cfg.qk_head_dim
            )
        else:
            self._inv_freq = 1.0 / (
                spec.base
                ** (
                    torch.arange(0, spec.dim, 2, dtype=torch.float32) / spec.dim
                )
            )
            self._rope_mscale = 1.0
            self.softmax_scale = 1.0 / math.sqrt(self.cfg.qk_head_dim)

    def _layer_weights(self, i: int) -> _LayerWeights:
        p = f"layers.{i}."
        return _LayerWeights(
            input_layernorm=self._w[p + "input_layernorm.weight"],
            post_attention_layernorm=self._w[
                p + "post_attention_layernorm.weight"
            ],
            gate_proj=self._w[p + "mlp.gate_proj.weight"],
            up_proj=self._w[p + "mlp.up_proj.weight"],
            down_proj=self._w[p + "mlp.down_proj.weight"],
            q_a_proj=self._w[p + "self_attn.q_a_proj.weight"],
            q_a_layernorm=self._w[p + "self_attn.q_a_layernorm.weight"],
            q_b_proj=self._w[p + "self_attn.q_b_proj.weight"],
            kv_a_proj_with_mqa=self._w[
                p + "self_attn.kv_a_proj_with_mqa.weight"
            ],
            kv_a_layernorm=self._w[p + "self_attn.kv_a_layernorm.weight"],
            kv_b_proj=self._w[p + "self_attn.kv_b_proj.weight"],
            o_proj=self._w[p + "self_attn.o_proj.weight"],
        )

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """``DFlashRMSNorm``: fp32 variance, cast back, then scale."""
        hidden = x.to(torch.float32)
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + self.cfg.rms_norm_eps)
        return weight * hidden.to(x.dtype)

    def _rope_cos_sin(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-pair cos/sin at the given positions, YaRN-mscale included."""
        freqs = positions.to(torch.float32)[:, None] * self._inv_freq[None, :]
        return freqs.cos() * self._rope_mscale, freqs.sin() * self._rope_mscale

    def _context_feature(self, context_hidden: torch.Tensor) -> torch.Tensor:
        """``extract_context_feature`` on pre-concatenated taps: one shared
        projection + norm feeding every layer's context-KV path."""
        projected = F.linear(
            context_hidden.to(self.dtype), self._w["context_proj.weight"]
        )
        return self._rms_norm(projected, self._w["context_norm.weight"])

    def _attention_mask(
        self,
        ctx_positions: torch.Tensor,
        anchor_position: int,
        block: int,
    ) -> torch.Tensor:
        """Which KV rows each block row may attend to: ``[block, T+block]``.

        The DFlash block-causal rules for one valid block
        (``torchspec/models/draft/dflash.py:70-86``): context rows strictly
        before the anchor's position, plus the whole block bidirectionally
        (``is_causal=False`` on the DFlash mask).
        """
        ctx_cols = ctx_positions < anchor_position
        return torch.cat(
            [
                ctx_cols.tile((block, 1)),
                torch.ones((block, block), dtype=torch.bool),
            ],
            dim=1,
        )

    def _attention(
        self,
        lw: _LayerWeights,
        block_normed: torch.Tensor,
        ctx_feature: torch.Tensor,
        draft_positions: torch.Tensor,
        ctx_positions: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        """One MLA layer's dual-source attention
        (``K3DSparkMLAAttention.forward``): Q from the block stream only,
        K/V from the context taps and the block through shared projections,
        interleaved RoPE on the rope side only, V never rotated.
        """
        cfg = self.cfg
        block = block_normed.shape[0]
        total = ctx_feature.shape[0] + block

        q = F.linear(block_normed, lw.q_a_proj)
        q = self._rms_norm(q, lw.q_a_layernorm)
        q = F.linear(q, lw.q_b_proj).view(block, cfg.num_heads, cfg.qk_head_dim)
        q_nope = q[..., : cfg.qk_nope_head_dim]
        q_rope = q[..., cfg.qk_nope_head_dim :]

        kv_in = torch.cat([ctx_feature, block_normed], dim=0)
        kv_a = F.linear(kv_in, lw.kv_a_proj_with_mqa)
        latent = self._rms_norm(
            kv_a[..., : cfg.kv_lora_rank], lw.kv_a_layernorm
        )
        k_rope = kv_a[..., cfg.kv_lora_rank :]
        kv_b = F.linear(latent, lw.kv_b_proj).view(
            total, cfg.num_heads, cfg.qk_nope_head_dim + cfg.v_head_dim
        )
        k_nope = kv_b[..., : cfg.qk_nope_head_dim]
        v = kv_b[..., cfg.qk_nope_head_dim :]

        full_positions = torch.cat([ctx_positions, draft_positions])
        q_cos, q_sin = self._rope_cos_sin(draft_positions)
        q_rope = _apply_interleaved_rope(
            q_rope, q_cos[:, None, :], q_sin[:, None, :]
        )
        k_cos, k_sin = self._rope_cos_sin(full_positions)
        # One shared rope head, MQA-style, broadcast across the heads.
        k_rope = _apply_interleaved_rope(k_rope, k_cos, k_sin)

        q_full = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat(
            [k_nope, k_rope[:, None, :].expand(*k_nope.shape[:2], -1)],
            dim=-1,
        )

        scores = torch.einsum("xhd,yhd->hxy", q_full, k_full)
        scores = scores * self.softmax_scale
        scores = scores.masked_fill(~allowed, float("-inf"))
        probs = scores.softmax(dim=-1)
        out = torch.einsum("hxy,yhd->xhd", probs, v)
        return F.linear(
            out.reshape(block, cfg.num_heads * cfg.v_head_dim), lw.o_proj
        )

    def _decoder_layer(
        self,
        lw: _LayerWeights,
        block_hidden: torch.Tensor,
        ctx_feature: torch.Tensor,
        draft_positions: torch.Tensor,
        ctx_positions: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-norm sandwich: ln -> attn -> +res; ln -> SwiGLU -> +res."""
        residual = block_hidden
        x = self._rms_norm(block_hidden, lw.input_layernorm)
        x = self._attention(
            lw, x, ctx_feature, draft_positions, ctx_positions, allowed
        )
        hidden = residual + x

        residual = hidden
        x = self._rms_norm(hidden, lw.post_attention_layernorm)
        x = F.linear(
            F.silu(F.linear(x, lw.gate_proj)) * F.linear(x, lw.up_proj),
            lw.down_proj,
        )
        return residual + x

    def _draft_hidden(
        self,
        context_hidden: torch.Tensor,
        context_positions: torch.Tensor,
        anchor_token: int,
        anchor_position: int,
    ) -> torch.Tensor:
        """Runs the block through the stack and the final norm.

        Returns ``[block_size, hidden]``: row j sits at position
        ``anchor_position + j`` (``_create_position_ids``) and predicts the
        token at ``anchor_position + j + 1``.
        """
        cfg = self.cfg
        ctx_positions = torch.as_tensor(
            context_positions, dtype=torch.long
        ).reshape(-1)
        if context_hidden.ndim != 2 or tuple(ctx_positions.shape) != (
            context_hidden.shape[0],
        ):
            raise ValueError(
                f"context_hidden {tuple(context_hidden.shape)} and "
                f"context_positions {tuple(ctx_positions.shape)} must be "
                f"[T_ctx, {cfg.num_taps * cfg.target_hidden_size}] and [T_ctx]"
            )
        assert context_hidden.shape[1] == cfg.num_taps * cfg.target_hidden_size
        assert 0 <= anchor_token < cfg.vocab_size, (
            f"anchor_token {anchor_token} outside [0, {cfg.vocab_size})"
        )
        assert anchor_position >= 0

        ctx_feature = self._context_feature(context_hidden)
        # _create_noise_embed: the anchor token seeds block row 0, mask
        # tokens fill the rest.
        block_ids = torch.full(
            (self.block_size,), cfg.mask_token_id, dtype=torch.long
        )
        block_ids[0] = anchor_token
        hidden = self._w["embed_tokens.weight"][block_ids]
        draft_positions = torch.arange(
            anchor_position, anchor_position + self.block_size
        )
        allowed = self._attention_mask(
            ctx_positions, anchor_position, self.block_size
        )
        for lw in self._layers:
            hidden = self._decoder_layer(
                lw,
                hidden,
                ctx_feature,
                draft_positions,
                ctx_positions,
                allowed,
            )
        return self._rms_norm(hidden, self._w["final_norm.weight"])

    def _base_logits(
        self,
        context_hidden: torch.Tensor,
        context_positions: torch.Tensor,
        anchor_token: int,
        anchor_position: int,
    ) -> torch.Tensor:
        """The target's LM head over the block's final hidden states."""
        hidden = self._draft_hidden(
            context_hidden, context_positions, anchor_token, anchor_position
        )
        return F.linear(hidden, self.lm_head_weight)

    def markov_bias(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """``VanillaMarkov``'s bigram bias: ``markov_w2(markov_w1[prev])``.

        Returns ``[N, vocab]`` for ``[N]`` previous-token ids.
        """
        if self.cfg.markov_rank == 0:
            raise ValueError(
                "this drafter has no markov head (markov_rank == 0)"
            )
        ids = torch.as_tensor(prev_tokens).reshape(-1).to(torch.long)
        embeddings = self._w["markov_head.markov_w1.weight"][ids]
        return F.linear(embeddings, self._w["markov_head.markov_w2.weight"])

    def draft_logits(
        self,
        context_hidden: torch.Tensor,
        context_positions: torch.Tensor,
        anchor_token: int,
        anchor_position: int,
        prev_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scores one draft block, with the Markov bias teacher-forced.

        Args:
            context_hidden: ``[T_ctx, num_taps * target_hidden]`` target
                taps, concatenated in ``target_layer_ids`` order — the order
                ``extract_context_feature`` concatenates them in.
            context_positions: ``[T_ctx]`` absolute position of each tap row.
            anchor_token: The committed token that seeds the block; its
                embedding is row 0's input.
            anchor_position: The anchor's absolute position. Block rows sit
                at ``anchor_position .. +block_size-1`` and only context
                rows strictly before this position are visible.
            prev_tokens: ``[block_size]`` previous-token ids for the bias,
                slot j's being the token slot j-1 drafted (the anchor seeds
                slot 0 — ``torchspec/models/dspark.py:169``). ``None``
                returns the unbiased base logits.

        Returns:
            ``[block_size, vocab]`` logits in ``self.dtype``; row j
            predicts the token at ``anchor_position + j + 1``.
        """
        logits = self._base_logits(
            context_hidden, context_positions, anchor_token, anchor_position
        )
        if prev_tokens is None:
            return logits
        prev = torch.as_tensor(prev_tokens).reshape(-1).to(torch.long)
        if tuple(prev.shape) != (self.block_size,):
            raise ValueError(
                f"prev_tokens: expected [{self.block_size}], got "
                f"{tuple(prev.shape)}"
            )
        return logits + self.markov_bias(prev)

    def greedy_block(
        self,
        context_hidden: torch.Tensor,
        context_positions: torch.Tensor,
        anchor_token: int,
        anchor_position: int,
    ) -> torch.Tensor:
        """Draws the block's greedy draft tokens, bias applied slot by slot.

        Mirrors inference (``deepseekV41/dspark_drafter.py::draft``): slot
        j's logits get the bigram bias of the token drawn at slot j-1, the
        anchor seeding slot 0.

        Returns:
            ``[block_size]`` int64 draft ids.
        """
        base = self._base_logits(
            context_hidden, context_positions, anchor_token, anchor_position
        )
        drafted = []
        prev = torch.tensor(anchor_token, dtype=torch.long)
        for slot in range(self.block_size):
            step = base[slot] + self.markov_bias(prev)[0]
            prev = step.argmax()
            drafted.append(prev)
        return torch.stack(drafted)
