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
"""Configuration of the MiMo-V2 DFlash drafter, read from ``dflash/config.json``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams


def _get(source: Mapping[str, object], key: str) -> object:
    if key not in source:
        raise ValueError(f"DFlash MiMo-V2: config has no {key!r}.")
    return source[key]


def _int(source: Mapping[str, object], key: str) -> int:
    value = _get(source, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"DFlash MiMo-V2: {key} must be an int, got {value!r}."
        )
    return value


def _float(source: Mapping[str, object], key: str) -> float:
    value = _get(source, key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"DFlash MiMo-V2: {key} must be a number, got {value!r}."
        )
    return float(value)


def _require(source: Mapping[str, object], key: str, expected: object) -> None:
    if (value := _get(source, key)) != expected:
        raise ValueError(
            f"DFlash MiMo-V2: expected {key}={expected!r}, got {value!r}."
        )


@dataclass(kw_only=True)
class DFlashMiMoV2Config:
    """The drafter's geometry and the conventions its checkpoint was trained with."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    sliding_window: int
    """Per query, a key is visible iff ``key + sliding_window > query``."""
    block_size: int
    target_layer_ids: list[int]
    """Target decoder layers whose outputs feed ``fc``, as numbered in
    ``config.json``: id ``k`` is the residual stream after layer ``k``."""
    mask_token_id: int
    attention_value_scale: float
    """Scales context and block V alike, as vLLM does."""
    max_seq_len: int
    devices: list[DeviceRef]
    kv_params: KVCacheParams
    dtype: DType = DType.bfloat16

    @property
    def rotary_dim(self) -> int:
        """Leading dims of each head that RoPE rotates, NeoX style."""
        return int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_dflash_config(
        cls,
        config: Mapping[str, object],
        *,
        devices: Sequence[DeviceRef],
        kv_params: KVCacheParams,
        max_seq_len: int,
    ) -> DFlashMiMoV2Config:
        """Reads the drafter config, refusing any convention this module lacks.

        Args:
            config: The parsed ``dflash/config.json``.
            devices: The devices the drafter runs on, one shard each.
            kv_params: The drafter's KV cache parameters.
            max_seq_len: The longest position the RoPE table must cover.

        Returns:
            The drafter configuration.

        Raises:
            ValueError: If the config describes a drafter this module would
                compute differently.
        """
        dflash = _get(config, "dflash_config")
        if not isinstance(dflash, Mapping):
            raise ValueError("DFlash MiMo-V2: dflash_config is not a mapping.")

        _require(config, "architectures", ["DFlashDraftModel"])
        _require(config, "is_causal", False)
        _require(config, "use_sliding_window", True)
        _require(config, "hidden_act", "silu")
        _require(config, "attention_bias", False)
        _require(config, "tie_word_embeddings", False)
        _require(config, "add_swa_attention_sink_bias", True)
        _require(dflash, "attention_sink_bias", True)
        for key in ("rope_scaling", "rope_parameters"):
            if config.get(key) is not None:
                raise ValueError(f"DFlash MiMo-V2: {key} is not supported.")

        num_layers = _int(config, "num_hidden_layers")
        _require(config, "layer_types", ["sliding_attention"] * num_layers)
        head_dim = _int(config, "head_dim")
        _require(config, "v_head_dim", head_dim)
        block_size = _int(config, "block_size")
        _require(dflash, "block_size", block_size)
        hidden_size = _int(config, "hidden_size")
        _require(config, "target_hidden_size", hidden_size)

        target_layer_ids = _get(dflash, "target_layer_ids")
        num_target_layers = _int(config, "num_target_layers")
        if (
            not isinstance(target_layer_ids, list)
            or not target_layer_ids
            or not all(
                isinstance(i, int) and 0 <= i < num_target_layers
                for i in target_layer_ids
            )
        ):
            raise ValueError(
                f"DFlash MiMo-V2: target_layer_ids {target_layer_ids!r} are not"
                f" all layers of a {num_target_layers}-layer target."
            )

        partial_rotary_factor = _float(config, "partial_rotary_factor")
        if (head_dim * partial_rotary_factor) % 2:
            raise ValueError(
                f"DFlash MiMo-V2: partial_rotary_factor {partial_rotary_factor}"
                f" does not rotate an even number of head_dim {head_dim}."
            )

        return cls(
            hidden_size=hidden_size,
            num_attention_heads=_int(config, "num_attention_heads"),
            num_key_value_heads=_int(config, "num_key_value_heads"),
            head_dim=head_dim,
            intermediate_size=_int(config, "intermediate_size"),
            num_hidden_layers=num_layers,
            rms_norm_eps=_float(config, "rms_norm_eps"),
            rope_theta=_float(config, "rope_theta"),
            partial_rotary_factor=partial_rotary_factor,
            sliding_window=_int(config, "sliding_window"),
            block_size=block_size,
            target_layer_ids=list(target_layer_ids),
            mask_token_id=_int(dflash, "mask_token_id"),
            attention_value_scale=_float(dflash, "attention_value_scale"),
            max_seq_len=max_seq_len,
            devices=list(devices),
            kv_params=kv_params,
        )
