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
# Ported to ModuleV3 from
# max/python/max/pipelines/architectures/dflash2_qwen3_5/model_config.py
# at main 4df8111afba.
"""Config for the DFlash2 draft of a Qwen3.5 target, on ModuleV3.

The draft HF config is the DFlash ``dflash_config`` shape plus four keys
DFlash never had (``conv_kernel_size``, ``conv_group_size``,
``selector_rank``, ``selector_top_k``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib.config import SpeculativeConfig
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta

__all__ = [
    "DRAFT_SLIDING_WINDOW",
    "DFlash2Qwen3_5Config",
    "Dflash2DraftHFConfig",
    "dflash2_draft_width",
    "draft_head_dim",
    "parse_dflash2_draft_hf_config",
]

DRAFT_SLIDING_WINDOW = 2048
"""Positions the drafter can attend to, from the checkpoint's config.

The draft KV leaf is bounded at this rather than at the target's context: the
drafter's mask is windowed, so nothing older is ever read, and the leaf's cost
becomes a per-slot constant instead of scaling with ``--max-length``.
"""

_SLIDING = "sliding_attention"
_FULL = "full_attention"


@dataclass(frozen=True)
class Dflash2DraftHFConfig:
    """The ``dflash_config`` block of a DFlash2 draft checkpoint."""

    mask_token_id: int
    target_layer_ids: list[int]
    block_size: int
    num_target_layers: int | None
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int

    def draft_width(self, speculative: SpeculativeConfig) -> int:
        """Returns ``block_size - 1``, refusing a flag that disagrees.

        The width is not recoverable from the MEF, so the export records it
        alongside and the engine checks its own ``--num-speculative-tokens``
        against it at boot. An overridden width would compile a graph at the
        trained block and declare another.

        Raises:
            ValueError: If ``num_speculative_tokens`` is set and is not
                ``block_size - 1``.
        """
        expected = self.block_size - 1
        requested = speculative.num_speculative_tokens
        if requested is not None and requested != expected:
            raise ValueError(
                f"DFlash2 was trained at block_size={self.block_size}, so it"
                f" drafts exactly {expected} tokens per step, but"
                f" --num-speculative-tokens is {requested}. The block width is"
                " structural (the drafter's convolution zeroes its t-1 tap at"
                " block-local position 0 and the selector walks a fixed"
                " tensor), so this is a hard error rather than an override."
            )
        return expected


def _get(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _required_int(dflash_config: Any, name: str) -> int:
    value = _get(dflash_config, name)
    if value is None:
        raise ValueError(
            f"DFlash2 draft HF config is missing ``dflash_config.{name}``."
        )
    return int(value)


def parse_dflash2_draft_hf_config(
    huggingface_config: Any,
) -> Dflash2DraftHFConfig:
    """Reads the DFlash2 drafter's structural constants off its HF config.

    Raises:
        ValueError: If a required key is missing, if the checkpoint declares no
            ``block_size``, or if it asks for an embedding/logit rescale this
            graph does not implement.
    """
    dflash_config = _get(huggingface_config, "dflash_config")
    if dflash_config is None:
        raise ValueError(
            "DFlash2 draft HF config is missing ``dflash_config``."
        )
    mask_token_id = _required_int(dflash_config, "mask_token_id")
    target_layer_ids = _get(dflash_config, "target_layer_ids")
    if not isinstance(target_layer_ids, (list, tuple)) or not target_layer_ids:
        raise ValueError(
            "DFlash2 dflash_config.target_layer_ids must be a non-empty list"
            f" of ints, got {target_layer_ids!r}."
        )
    block_size = _get(
        dflash_config, "block_size", _get(huggingface_config, "block_size")
    )
    if block_size is None:
        raise ValueError(
            "DFlash2 draft HF config declares no ``dflash_config.block_size``."
            " The block width is structural — the drafter's convolution zeroes"
            " its t-1 tap at block-local position 0 — so it cannot be supplied"
            " by a flag."
        )
    num_target_layers = _get(
        dflash_config,
        "num_target_layers",
        _get(huggingface_config, "num_target_layers"),
    )

    # The reference multiplies the borrowed embedding and the borrowed head's
    # logits by these; this graph applies neither.
    for name in ("input_embedding_scale", "output_multiplier"):
        scale = _get(dflash_config, name, 1.0)
        if float(scale) != 1.0:
            raise ValueError(
                f"DFlash2 draft declares dflash_config.{name}={scale}; this"
                " graph borrows the target's embedding and head unscaled and"
                " implements neither rescale."
            )

    return Dflash2DraftHFConfig(
        mask_token_id=mask_token_id,
        target_layer_ids=[int(i) for i in target_layer_ids],
        block_size=int(block_size),
        num_target_layers=(
            None if num_target_layers is None else int(num_target_layers)
        ),
        conv_kernel_size=_required_int(dflash_config, "conv_kernel_size"),
        conv_group_size=_required_int(dflash_config, "conv_group_size"),
        selector_rank=_required_int(dflash_config, "selector_rank"),
        selector_top_k=_required_int(dflash_config, "selector_top_k"),
    )


def dflash2_draft_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """Returns the width the DFlash2 draft checkpoint was trained for.

    Registered as the architecture's ``checkpoint_draft_width`` so that an
    omitted ``--num-speculative-tokens`` resolves to the graph's width, which
    the scheduler's draft-token handoff reads off the config.
    """
    del target_huggingface_config
    if draft_huggingface_config is None:
        raise ValueError("DFlash2 requires a draft model.")
    return parse_dflash2_draft_hf_config(draft_huggingface_config).draft_width(
        speculative
    )


def draft_head_dim(huggingface_config: Any) -> int:
    """Returns ``head_dim``, or ``hidden_size // num_attention_heads``."""
    head_dim = getattr(huggingface_config, "head_dim", None)
    if head_dim is not None:
        return int(head_dim)
    return int(
        huggingface_config.hidden_size // huggingface_config.num_attention_heads
    )


def _layer_windows(
    huggingface_config: Any, num_layers: int
) -> tuple[int | None, ...]:
    """Returns each layer's attention window, ``None`` for a full layer.

    Without ``layer_types``, every layer takes ``sliding_window``.

    Raises:
        ValueError: If ``layer_types`` names a kind this drafter lacks, or
            sliding layers have no window.
    """
    window = getattr(huggingface_config, "sliding_window", None)
    layer_types: Sequence[str] | None = (
        getattr(huggingface_config, "layer_types", None) or None
    )
    if layer_types is None:
        return (window,) * num_layers
    if len(layer_types) != num_layers:
        raise ValueError(
            "DFlash2 layer_types must have one entry per draft layer."
            f" Got {len(layer_types)} entries for {num_layers} layers."
        )
    unknown = sorted(set(layer_types) - {_SLIDING, _FULL})
    if unknown:
        raise ValueError(
            f"DFlash2 draft has unsupported layer_types {unknown};"
            f" expected only {_SLIDING!r} or {_FULL!r}."
        )
    if _SLIDING in layer_types and window is None:
        raise ValueError(
            "DFlash2 sliding_attention layers require a sliding_window on the"
            " draft config."
        )
    return tuple(window if kind == _SLIDING else None for kind in layer_types)


@dataclass(frozen=True, kw_only=True)
class DFlash2Qwen3_5Config:
    """The drafter's geometry, and what its pairing reads beside it."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    attention_scale: float
    layer_windows: tuple[int | None, ...]
    """Each layer's attention window, ``None`` for a full layer."""
    vocab_size: int
    max_seq_len: int
    """The longest position the RoPE table covers."""
    kv_params: KVCacheParams
    dflash: Dflash2DraftHFConfig
    dtype: DType = DType.bfloat16

    @property
    def num_hidden_layers(self) -> int:
        """Returns the drafter's layer count."""
        return len(self.layer_windows)

    @property
    def mask_token_id(self) -> int:
        """Returns the token that fills the block's drafted slots."""
        return self.dflash.mask_token_id

    @property
    def target_layer_ids(self) -> list[int]:
        """Returns the target layers the draft reads."""
        return self.dflash.target_layer_ids

    @property
    def block_size(self) -> int:
        """Returns the block the draft was trained at."""
        return self.dflash.block_size

    @property
    def num_target_layers(self) -> int | None:
        """Returns the target depth the draft was trained against.

        ``None`` when the checkpoint names no depth.
        """
        return self.dflash.num_target_layers

    @classmethod
    def from_huggingface_config(
        cls,
        huggingface_config: Any,
        *,
        kv_params: KVCacheParams,
        max_seq_len: int,
    ) -> DFlash2Qwen3_5Config:
        """Reads the drafter config, refusing a convention it lacks.

        Raises:
            ValueError: If the checkpoint is quantized, clips its QKV, scales
                its RoPE, repeats a tap, or names a layer kind or window this
                drafter lacks.
        """
        hf = huggingface_config
        for name in ("quantization_config", "clip_qkv"):
            if getattr(hf, name, None) is not None:
                raise ValueError(
                    f"The DFlash2 drafter does not support {name}; got"
                    f" {getattr(hf, name)!r}."
                )
        rope = getattr(hf, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError(
                f"The DFlash2 drafter supports only default RoPE; got {rope!r}."
            )
        dflash = parse_dflash2_draft_hf_config(hf)
        layers = dflash.target_layer_ids
        if len(set(layers)) != len(layers):
            raise ValueError(
                "DFlash2 target_layer_ids must be distinct: the capture fires"
                " once per tapped layer, so a duplicate yields one fewer"
                f" column than fc expects. Got {layers}."
            )
        head_dim = draft_head_dim(hf)
        return cls(
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=get_rope_theta(hf),
            attention_scale=getattr(
                hf, "attention_multiplier", math.sqrt(1.0 / float(head_dim))
            ),
            layer_windows=_layer_windows(hf, hf.num_hidden_layers),
            vocab_size=hf.vocab_size,
            max_seq_len=max_seq_len,
            kv_params=kv_params,
            dflash=dflash,
        )
