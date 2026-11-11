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
"""Tiny MiMo-V2 and DFlash drafter configs, and the base graphs built from
them, for tests that need no weights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.type import Shape
from max.graph.weights import WeightData
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.pipelines.architectures.dflash_mimo_v2 import DFlashMiMoV2Config
from max.pipelines.architectures.mimo_v2.model_config import (
    FULL,
    SLIDING,
    MiMoV2Config,
    attention_head_dim,
    layer_types,
)
from transformers.configuration_utils import PretrainedConfig

PAGE_SIZE = 128


def weight_data(name: str, dtype: DType, array: np.ndarray) -> WeightData:
    buffer = Buffer.from_numpy(np.ascontiguousarray(array))
    if dtype != DType.from_numpy(array.dtype):
        buffer = buffer.view(dtype)
    return WeightData(buffer, name, dtype, Shape(array.shape))


def tiny_target_hf(num_layers: int = 3, experts: int = 32) -> PretrainedConfig:
    """A small MiMo-V2 in Xiaomi's config: full attention with a
    dense MLP, then SWA-128 with sinks and MoE, then full attention and MoE.

    The router kernel runs a thread per expert in whole warps, so the expert
    count is a multiple of 32.
    """
    pattern = [0] + [1, 0] * ((num_layers - 1) // 2)
    pattern += [1] * (num_layers - len(pattern))
    moe = [0] + [1] * (num_layers - 1)
    return PretrainedConfig(
        vocab_size=1024,
        hidden_size=512,
        num_hidden_layers=num_layers,
        layernorm_epsilon=1e-6,
        hybrid_layer_pattern=pattern,
        moe_layer_freq=moe,
        num_attention_heads=8,
        swa_num_attention_heads=8,
        num_key_value_heads=4,
        swa_num_key_value_heads=8,
        head_dim=192,
        swa_head_dim=192,
        v_head_dim=128,
        swa_v_head_dim=128,
        partial_rotary_factor=0.334,
        rope_theta=10000000.0,
        swa_rope_theta=10000.0,
        sliding_window=128,
        attention_value_scale=0.707,
        add_full_attention_sink_bias=False,
        add_swa_attention_sink_bias=True,
        intermediate_size=1024,
        moe_intermediate_size=512,
        n_routed_experts=experts,
        num_experts_per_tok=8,
        n_shared_experts=None,
        norm_topk_prob=True,
        routed_scaling_factor=None,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_group=1,
        topk_group=1,
        hidden_act="silu",
        attention_bias=False,
        tie_word_embeddings=False,
        attention_projection_layout="fused_qkv",
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "store_dtype": "mxfp4",
            "mxfp4_block_size": 32,
        },
    )


def tiny_dflash_config(target_layer_ids: Sequence[int]) -> dict[str, Any]:
    """A small drafter's ``config.json``, as ``from_dflash_config`` reads it."""
    return {
        "architectures": ["DFlashDraftModel"],
        "hidden_size": 512,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "v_head_dim": 128,
        "partial_rotary_factor": 0.5,
        "block_size": 8,
        "dflash_config": {
            "target_layer_ids": list(target_layer_ids),
            "mask_token_id": 1000,
            "block_size": 8,
            "attention_value_scale": 0.612,
            "attention_sink_bias": True,
        },
        "layer_types": ["sliding_attention"] * 2,
        "sliding_window": 1024,
        "use_sliding_window": True,
        "is_causal": False,
        "num_target_layers": 3,
        "target_hidden_size": 512,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "add_swa_attention_sink_bias": True,
        "tie_word_embeddings": False,
    }


def target_kv_params(
    hf: PretrainedConfig,
    devices: Sequence[DeviceRef],
    *,
    num_draft_tokens: int | None = None,
) -> MultiKVCacheParams:
    """The target's ``{sliding, full}`` tree; with ``num_draft_tokens`` the
    speculative graph's, whose groups also take a draft dispatch buffer."""
    groups = layer_types(hf)

    def group(heads: int, layers: int, window: int | None) -> MHAKVCacheParams:
        return MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=heads,
            head_dim=attention_head_dim(hf),
            num_layers=layers,
            devices=list(devices),
            page_size=PAGE_SIZE,
            window_size=window,
            speculative_method=None if num_draft_tokens is None else "dflash",
            num_draft_tokens=num_draft_tokens or 0,
        )

    return MultiKVCacheParams.from_params(
        {
            SLIDING: group(
                hf.swa_num_key_value_heads,
                groups.count(SLIDING),
                hf.sliding_window,
            ),
            FULL: group(hf.num_key_value_heads, groups.count(FULL), None),
        }
    )


def drafter_kv_params(
    dflash: Mapping[str, Any],
    devices: Sequence[DeviceRef],
    *,
    speculative: bool,
) -> MHAKVCacheParams:
    """The drafter's context cache, the tail group of every tree with one."""
    block = int(dflash["block_size"])
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=int(dflash["num_key_value_heads"]),
        head_dim=int(dflash["head_dim"]),
        num_layers=int(dflash["num_hidden_layers"]),
        devices=list(devices),
        page_size=PAGE_SIZE,
        window_size=int(dflash["sliding_window"]),
        speculative_method="dflash" if speculative else None,
        num_draft_tokens=block if speculative else 0,
    )


@dataclass
class Model:
    """A target, a drafter and their weights, at any device count."""

    hf: PretrainedConfig
    dflash: dict[str, Any]
    target_state: dict[str, WeightData]
    draft_state: dict[str, WeightData]
    mask_embedding: WeightData
    max_seq_len: int

    def target(
        self, devices: Sequence[DeviceRef], *, speculative: bool
    ) -> MiMoV2Config:
        """The target's config; the speculative graph's KV groups carry the
        block's draft count."""
        block = int(self.dflash["block_size"])
        return MiMoV2Config.from_huggingface_config(
            self.hf,
            devices=list(devices),
            kv_params=target_kv_params(
                self.hf,
                devices,
                num_draft_tokens=block if speculative else None,
            ),
            max_seq_len=self.max_seq_len + 4 * block,
        )

    def drafter(
        self, devices: Sequence[DeviceRef], *, speculative: bool
    ) -> DFlashMiMoV2Config:
        return DFlashMiMoV2Config.from_dflash_config(
            self.dflash,
            devices=list(devices),
            kv_params=drafter_kv_params(
                self.dflash, devices, speculative=speculative
            ),
            max_seq_len=self.max_seq_len + 4 * int(self.dflash["block_size"]),
        )


def tiny_configs(max_seq_len: int = 4096) -> Model:
    """The tiny target and drafter without weights, enough to build graphs."""
    hf = tiny_target_hf()
    dflash = tiny_dflash_config(list(range(hf.num_hidden_layers)))
    hidden = np.zeros(dflash["hidden_size"], np.uint16)
    return Model(
        hf=hf,
        dflash=dflash,
        target_state={},
        draft_state={},
        mask_embedding=weight_data("mask_embedding", DType.bfloat16, hidden),
        max_seq_len=max_seq_len,
    )
