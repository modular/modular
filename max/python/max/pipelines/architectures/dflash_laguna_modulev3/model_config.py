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
"""Configuration of the Laguna DFlash drafter, read from its ``config.json``.

The drafter is causal within its block, so a block of ``K + 1`` rows computes
the first ``K + 1`` rows of the 16-row block it was trained at. Any ``K``
below that block is served at ``K + 1``.
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
from max.pipelines.speculative._dflash import parse_dflash_draft_hf_config

__all__ = [
    "SERVING_BLOCK_SIZE",
    "DFlashLagunaConfig",
    "DFlashLagunaDraftConfig",
    "dflash_laguna_draft_width",
    "drafter_kv_params",
]

SERVING_BLOCK_SIZE = 8
"""The block the target's model card serves the drafter at: the anchor and
7 drafts. The default when ``--num-speculative-tokens`` is unset."""

CHECKPOINT_ARCHITECTURE = "DFlashLagunaForCausalLM"
ARCHITECTURE = CHECKPOINT_ARCHITECTURE


def _as_dict(huggingface_config: Any) -> Mapping[str, Any]:
    if isinstance(huggingface_config, Mapping):
        return huggingface_config
    return huggingface_config.to_dict()


def _require(source: Mapping[str, Any], key: str, expected: object) -> None:
    if (value := source.get(key)) != expected:
        raise ValueError(
            f"Laguna DFlash: expected {key}={expected!r}, got {value!r}."
        )


def dflash_laguna_draft_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """Returns how many drafts a step proposes and verifies.

    Raises:
        ValueError: If the width does not fit below the trained block.
    """
    del target_huggingface_config
    if draft_huggingface_config is None:
        raise ValueError("Laguna DFlash requires a draft model.")
    trained = parse_dflash_draft_hf_config(draft_huggingface_config).block_size
    width = speculative.num_speculative_tokens
    if width is None:
        return SERVING_BLOCK_SIZE - 1
    if trained is None or not 1 <= width < trained:
        raise ValueError(
            f"Laguna DFlash drafts 1 to {trained and trained - 1} tokens,"
            f" below the block of {trained} it was trained at; got"
            f" --num-speculative-tokens {width}."
        )
    return width


def drafter_kv_params(
    huggingface_config: Any,
    pipeline_config: PipelineConfig,
    devices: list[DeviceRef],
    kv_cache_config: KVCacheConfig,
) -> KVCacheParams:
    """The drafter's context cache: its own geometry, always BF16.

    It is windowed like the drafter's attention, and under speculation its
    draft count is the ``K + 1`` rows the next block writes past the context:
    its anchor and ``K`` mask slots.
    """
    config = _as_dict(huggingface_config)
    speculative = pipeline_config.speculative
    return kv_cache_config.to_params(
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
class DFlashLagunaConfig:
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
    """Keys a block query attends to, itself included."""
    vocab_size: int
    block_size: int
    """The serving block: the anchor and ``K`` drafts."""
    trained_block_size: int
    mask_token_id: int
    target_layer_ids: list[int]
    """The target layers whose residual streams feed ``fc``, in order."""
    num_target_layers: int | None
    max_seq_len: int
    devices: list[DeviceRef]
    kv_params: KVCacheParams

    @classmethod
    def from_huggingface_config(
        cls,
        huggingface_config: Any,
        *,
        block_size: int,
        devices: list[DeviceRef],
        kv_params: KVCacheParams,
        max_seq_len: int,
    ) -> DFlashLagunaConfig:
        """Reads the drafter config, refusing a convention this module lacks.

        Args:
            huggingface_config: The draft checkpoint's config.
            block_size: The serving block, ``K + 1``.
            devices: The device the drafter runs on.
            kv_params: The drafter's KV cache parameters.
            max_seq_len: The longest position the RoPE table must cover.

        Raises:
            ValueError: If the config describes a drafter this module would
                compute differently, or the block outgrows the trained one.
        """
        config = _as_dict(huggingface_config)
        _require(config, "architectures", [CHECKPOINT_ARCHITECTURE])
        _require(config, "hidden_act", "silu")
        _require(config, "attention_bias", False)
        _require(config, "gating", "per-head")
        _require(config, "num_experts", 0)
        num_layers = int(config["num_hidden_layers"])
        _require(config, "layer_types", ["sliding_attention"] * num_layers)
        _require(config.get("dflash_config") or {}, "causal", True)
        # Newer transformers fold the top-level rope keys into
        # rope_parameters.
        rope = config.get("rope_parameters") or {}
        rotary = rope.get(
            "partial_rotary_factor", config.get("partial_rotary_factor")
        )
        if (
            config.get("rope_scaling") is not None
            or rope.get("rope_type", "default") != "default"
            or rotary not in (None, 1.0)
        ):
            raise ValueError(
                f"Laguna DFlash supports only default full-head RoPE, got {rope!r}."
            )
        rope_theta = rope.get("rope_theta", config.get("rope_theta"))
        if rope_theta is None:
            raise ValueError("Laguna DFlash: the config sets no rope_theta.")
        vocab_size = int(config["vocab_size"])
        _require(config, "draft_vocab_size", vocab_size)
        dflash = parse_dflash_draft_hf_config(config)
        trained = dflash.block_size
        if trained is None or not 1 < block_size <= trained:
            raise ValueError(
                f"Laguna DFlash serves a block of 2 to {trained}, the block"
                f" it was trained at; got {block_size}."
            )
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
            vocab_size=vocab_size,
            block_size=block_size,
            trained_block_size=trained,
            mask_token_id=dflash.mask_token_id,
            target_layer_ids=list(dflash.target_layer_ids),
            num_target_layers=dflash.num_target_layers,
            max_seq_len=max_seq_len,
            devices=list(devices),
            kv_params=kv_params,
        )


@dataclass(frozen=True)
class DFlashLagunaDraftConfig:
    """The drafter module's config, and what the pairing reads beside it."""

    module: DFlashLagunaConfig
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
        return self.module.num_target_layers
