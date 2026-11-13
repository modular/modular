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
"""Configuration of the Beam DFlash2 drafter, read from its ``config.json``.

The ``dflash_config`` block is the one the Qwen3.5 DFlash2 drafter reads, so
its parser is reused. What differs is the block: the checkpoint was trained
at 16 and is served at ``K + 1``, the width vLLM and SGLang serve it at, so
the convolution's block-local axis and the attention's block are both the
serving block.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig, SpeculativeConfig
from max.pipelines.speculative import DrafterExport

from ..dflash2_qwen3_5.model_config import (
    Dflash2DraftHFConfig,
    parse_dflash2_draft_hf_config,
)

__all__ = [
    "SERVING_BLOCK_SIZE",
    "DFlash2BeamConfig",
    "DFlash2BeamDraftConfig",
    "dflash2_beam_draft_width",
    "drafter_kv_params",
]

SERVING_BLOCK_SIZE = 8
"""The block the drafter's model card serves it at: the anchor and 7 drafts.

It is the default when ``--num-speculative-tokens`` is unset; any width
below the trained block serves at ``K + 1``.
"""

ARCHITECTURE = "DFlash2BeamDraftModel"


def _as_dict(huggingface_config: Any) -> Mapping[str, Any]:
    if isinstance(huggingface_config, Mapping):
        return huggingface_config
    return huggingface_config.to_dict()


def _require(source: Mapping[str, Any], key: str, expected: object) -> None:
    if (value := source.get(key)) != expected:
        raise ValueError(
            f"Beam DFlash2: expected {key}={expected!r}, got {value!r}."
        )


def dflash2_beam_draft_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """Returns how many drafts a step proposes and verifies.

    The block is ``K + 1`` wide, so any ``K`` below the trained block is
    defined; unset serves the model card's block of 8.

    Raises:
        ValueError: If the width does not fit below the trained block.
    """
    del target_huggingface_config
    if draft_huggingface_config is None:
        raise ValueError("Beam DFlash2 requires a draft model.")
    trained = parse_dflash2_draft_hf_config(draft_huggingface_config).block_size
    width = speculative.num_speculative_tokens
    if width is None:
        return SERVING_BLOCK_SIZE - 1
    if not 1 <= width < trained:
        raise ValueError(
            f"Beam DFlash2 drafts 1 to {trained - 1} tokens, below the block"
            f" of {trained} it was trained at; got --num-speculative-tokens"
            f" {width}."
        )
    return width


def drafter_kv_params(
    huggingface_config: Any,
    pipeline_config: PipelineConfig,
    devices: list[DeviceRef],
    kv_cache_config: KVCacheConfig,
) -> KVCacheParams:
    """The drafter's context cache: its own geometry, always BF16.

    It is windowed like the drafter's attention. Under speculation it carries
    the block's width as its draft count, since the block writes one more
    row than the verify scores.
    """
    config = _as_dict(huggingface_config)
    speculative = pipeline_config.speculative
    return kv_cache_config.to_params(
        # The drafter fills this cache with its own unquantized projections,
        # whatever the target's cache dtype.
        dtype=DType.bfloat16,
        n_kv_heads=int(config["num_key_value_heads"]),
        head_dim=int(config["head_dim"]),
        num_layers=int(config["num_hidden_layers"]),
        devices=devices,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        speculative_method=(
            speculative.speculative_method if speculative else None
        ),
        num_draft_tokens=speculative.draft_width + 1 if speculative else 0,
        window_size=int(config["sliding_window"]),
    )


@dataclass(frozen=True, kw_only=True)
class DFlash2BeamConfig:
    """The drafter's geometry and the conventions it was trained with."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    """Keys a block query attends to on either side, in positions."""
    vocab_size: int
    final_logit_softcapping: float | None
    """The cap on the borrowed head's logits, as Beam's own."""
    dflash: Dflash2DraftHFConfig
    block_size: int
    """The serving block: the anchor and ``K`` drafts. The convolution's
    block-local axis."""
    max_seq_len: int
    devices: list[DeviceRef]
    kv_params: KVCacheParams
    dtype: DType = DType.bfloat16

    @property
    def target_layer_ids(self) -> list[int]:
        """Returns the target layers whose outputs feed ``fc``.

        Id ``k`` is the residual stream after layer ``k``.
        """
        return list(self.dflash.target_layer_ids)

    @property
    def mask_token_id(self) -> int:
        """Returns the token every drafted slot of the block embeds."""
        return self.dflash.mask_token_id

    @classmethod
    def from_huggingface_config(
        cls,
        huggingface_config: Any,
        *,
        block_size: int,
        devices: list[DeviceRef],
        kv_params: KVCacheParams,
        max_seq_len: int,
    ) -> DFlash2BeamConfig:
        """Reads the drafter config, refusing a convention this module lacks.

        Args:
            huggingface_config: The draft checkpoint's config.
            block_size: The serving block, ``K + 1``.
            devices: The devices the drafter shards its heads over.
            kv_params: The drafter's KV cache parameters.
            max_seq_len: The longest position the RoPE table must cover.

        Raises:
            ValueError: If the config describes a drafter this module would
                compute differently, or the block outgrows the trained one.
        """
        config = _as_dict(huggingface_config)
        _require(config, "architectures", [ARCHITECTURE])
        _require(config, "is_causal", False)
        _require(config, "hidden_act", "silu")
        _require(config, "attention_bias", False)
        _require(config, "tie_word_embeddings", False)
        num_layers = int(config["num_hidden_layers"])
        _require(config, "layer_types", ["sliding_attention"] * num_layers)
        if config.get("rope_scaling") is not None:
            raise ValueError("Beam DFlash2: rope_scaling is not supported.")
        rope = config.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError(
                f"Beam DFlash2 supports only default RoPE, got {rope!r}."
            )
        # Newer transformers fold the top-level key into rope_parameters.
        rope_theta = rope.get("rope_theta", config.get("rope_theta"))
        if rope_theta is None:
            raise ValueError("Beam DFlash2: the config sets no rope_theta.")
        dflash = parse_dflash2_draft_hf_config(huggingface_config)
        if not 1 < block_size <= dflash.block_size:
            raise ValueError(
                f"Beam DFlash2 serves a block of 2 to {dflash.block_size}, the"
                f" block it was trained at; got {block_size}."
            )
        softcap = config["dflash_config"].get("final_logit_softcapping")
        return cls(
            hidden_size=int(config["hidden_size"]),
            num_attention_heads=int(config["num_attention_heads"]),
            num_key_value_heads=int(config["num_key_value_heads"]),
            head_dim=int(config["head_dim"]),
            intermediate_size=int(config["intermediate_size"]),
            num_hidden_layers=num_layers,
            rms_norm_eps=float(config["rms_norm_eps"]),
            rope_theta=float(rope_theta),
            sliding_window=int(config["sliding_window"]),
            vocab_size=int(config["vocab_size"]),
            final_logit_softcapping=(
                float(softcap) if softcap is not None else None
            ),
            dflash=dflash,
            block_size=block_size,
            max_seq_len=max_seq_len,
            devices=list(devices),
            kv_params=kv_params,
        )


@dataclass(frozen=True)
class DFlash2BeamDraftConfig:
    """The drafter module's config, and what the pairing reads beside it."""

    module: DFlash2BeamConfig
    sampleable_vocab_size: int
    """The tokenizer's size. The target's head rows from here up are padding
    with live weights, so neither the verify nor the draft samples them."""
    drafter_export: DrafterExport
    """What the graph bakes in of the drafter, for an export's sidecar."""

    @property
    def hidden_size(self) -> int:
        """Returns the drafter's hidden size."""
        return self.module.hidden_size

    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary the drafter shares with its target."""
        return self.module.vocab_size

    @property
    def block_size(self) -> int:
        """Returns the serving block."""
        return self.module.block_size

    @property
    def mask_token_id(self) -> int:
        """Returns the token every drafted slot of the block embeds."""
        return self.module.mask_token_id

    @property
    def target_layer_ids(self) -> list[int]:
        """Returns the target layers the drafter reads."""
        return self.module.target_layer_ids

    @property
    def num_target_layers(self) -> int | None:
        """Returns the target depth the drafter was trained against."""
        return self.module.dflash.num_target_layers
