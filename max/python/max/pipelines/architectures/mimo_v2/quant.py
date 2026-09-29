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
"""Quantization scheme for the MiMo-V2.6-Flash NVFP4 export.

``ProCreations/MiMo-V2.6-Flash-RL-NVFP4`` is a lossless transcode of Xiaomi's
checkpoint, whose routed experts are MXFP4 and whose dense linears are FP8
with 128x128 block scales. The export declares modelopt ``MIXED_PRECISION``
with only the routed experts quantized (``W4A16_NVFP4``, group 16), and
stores every dense linear as F32 or BF16. MAX serves Xiaomi's format: the
weight adapter turns the experts back into MXFP4 and the F32 dense linears
back into FP8, each exactly. The shared parser refuses ``MIXED_PRECISION``, so
the checks and the configs live here.

``exclude_modules`` is not read. It is empty although no dense linear is
quantized, so only ``quantized_layers`` says what is.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from transformers.configuration_utils import PretrainedConfig

MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

QKV_LAYOUT = "global_q_k_v"
"""The only ``conversion_metadata.qkv_layout`` the adapter can invert."""

FP8_BLOCK = 128
MXFP4_BLOCK = 32
NVFP4_GROUP_SIZE = 16
_NVFP4_ALGO = "W4A16_NVFP4"


@dataclass(frozen=True)
class MiMoV2QuantScheme:
    """How each MiMo-V2 module is stored after weight adaptation.

    ``o_proj``, the router, the embeddings, ``lm_head``, norms and sinks are
    unquantized and have no config.
    """

    dense: QuantConfig
    """FP8 E4M3 with 128x128 block scales, for every ``qkv_proj`` and dense
    MLP, with dynamic 1x128 activation scales."""

    experts: QuantConfig
    """MXFP4 routed experts: E2M1 codes with E8M0 scales per 32 elements."""


def moe_layers(config: PretrainedConfig) -> list[int]:
    """Returns the decoder layers whose MLP is the routed-expert MoE."""
    return [i for i, is_moe in enumerate(config.moe_layer_freq) if is_moe]


def routed_expert_modules(config: PretrainedConfig) -> set[str]:
    """Returns the checkpoint name of every routed-expert projection."""
    return {
        f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
        for layer in moe_layers(config)
        for expert in range(config.n_routed_experts)
        for proj in MLP_PROJECTIONS
    }


def validate_checkpoint_config(config: PretrainedConfig) -> None:
    """Raises unless ``config`` describes the NVFP4 export.

    Args:
        config: The checkpoint's top-level Hugging Face config.

    Raises:
        ValueError: If the quantization metadata describes anything other
            than W4A16 NVFP4 routed experts with everything else unquantized,
            or the QKV layout is not the export's global order.
    """
    quant: Mapping[str, Any] = (
        getattr(config, "quantization_config", None) or {}
    )
    method, algo = quant.get("quant_method"), quant.get("quant_algo")
    if (method, algo) != ("modelopt", "MIXED_PRECISION"):
        raise ValueError(
            f"MiMo-V2: quantization_config is quant_method={method!r}, "
            f"quant_algo={algo!r}; only NVFP4 exports (modelopt "
            "MIXED_PRECISION) load for now."
        )
    if quant.get("kv_cache_quant_algo") is not None:
        raise ValueError(
            "MiMo-V2: kv_cache_quant_algo is "
            f"{quant['kv_cache_quant_algo']!r}; KV-cache scales are not wired."
        )

    metadata = getattr(config, "conversion_metadata", None) or {}
    layout = metadata.get("qkv_layout")
    if layout != QKV_LAYOUT:
        raise ValueError(
            f"MiMo-V2: conversion_metadata.qkv_layout is {layout!r}; the "
            f"weight adapter only inverts {QKV_LAYOUT!r}. Reading qkv_proj in "
            "the wrong row order gives wrong attention without an error."
        )

    declared = quant.get("quantized_layers") or {}
    expected = routed_expert_modules(config)
    if unexpected := sorted(set(declared) - expected):
        raise ValueError(
            f"MiMo-V2: quantized_layers names {len(unexpected)} module(s) "
            f"that are not routed-expert projections, e.g. {unexpected[:3]}. "
            "Only the routed experts are quantized in this export."
        )
    if missing := sorted(expected - set(declared)):
        raise ValueError(
            f"MiMo-V2: quantized_layers lists {len(declared)} of the "
            f"{len(expected)} routed-expert projections; missing e.g. "
            f"{missing[:3]}."
        )
    for module, entry in declared.items():
        if not isinstance(entry, Mapping) or (
            entry.get("quant_algo"),
            entry.get("group_size"),
        ) != (_NVFP4_ALGO, NVFP4_GROUP_SIZE):
            raise ValueError(
                f"MiMo-V2: quantized_layers[{module!r}] is {entry!r}; "
                f"expected quant_algo {_NVFP4_ALGO!r} with group_size "
                f"{NVFP4_GROUP_SIZE}."
            )


def parse_quant_scheme(config: PretrainedConfig) -> MiMoV2QuantScheme:
    """Validates the checkpoint config and returns the adapted scheme.

    Args:
        config: The checkpoint's top-level Hugging Face config.

    Returns:
        The quantization of the adapted weights.

    Raises:
        ValueError: If :func:`validate_checkpoint_config` rejects the config.
    """
    validate_checkpoint_config(config)
    moe = set(moe_layers(config))
    all_layers = set(range(config.num_hidden_layers))
    return MiMoV2QuantScheme(
        dense=QuantConfig(
            input_scale=InputScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                origin=ScaleOrigin.DYNAMIC,
                dtype=DType.float32,
                block_size=(1, FP8_BLOCK),
            ),
            weight_scale=WeightScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                dtype=DType.float32,
                block_size=(FP8_BLOCK, FP8_BLOCK),
            ),
            mlp_quantized_layers=all_layers - moe,
            # qkv_proj only; o_proj is BF16 in every layer.
            attn_quantized_layers=all_layers,
            embedding_output_dtype=DType.bfloat16,
            format=QuantFormat.BLOCKSCALED_FP8,
        ),
        experts=QuantConfig(
            input_scale=InputScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                origin=ScaleOrigin.DYNAMIC,
                dtype=DType.float32,
                block_size=(1, MXFP4_BLOCK),
            ),
            weight_scale=WeightScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                dtype=DType.float8_e8m0fnu,
                block_size=(1, MXFP4_BLOCK),
            ),
            mlp_quantized_layers=moe,
            attn_quantized_layers=set(),
            embedding_output_dtype=DType.bfloat16,
            format=QuantFormat.MXFP4,
        ),
    )
