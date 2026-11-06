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
"""Quantization scheme of Xiaomi's MiMo-V2.6-Flash checkpoint.

The routed experts are MXFP4, and every ``qkv_proj`` and dense MLP is FP8
E4M3 with 128x128 block scales. ``o_proj``, the router, the embeddings,
``lm_head``, norms and sinks are BF16 or F32.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from max.dtype import DType
from max.nn.quant_config import MXFP4_BLOCK_SIZE, QuantConfig
from transformers.configuration_utils import PretrainedConfig

FP8_BLOCK = 128
MXFP4_BLOCK = MXFP4_BLOCK_SIZE

_CHECKPOINT_QUANTIZATION: dict[str, object] = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [FP8_BLOCK, FP8_BLOCK],
    "store_dtype": "mxfp4",
    "mxfp4_block_size": MXFP4_BLOCK,
}
"""The ``quantization_config`` fields the checkpoint is read by."""


@dataclass(frozen=True)
class MiMoV2QuantScheme:
    """How each MiMo-V2 module is quantized.

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


def parse_quant_scheme(config: PretrainedConfig) -> MiMoV2QuantScheme:
    """Validates the checkpoint's quantization and returns its scheme.

    Args:
        config: The checkpoint's top-level Hugging Face config.

    Returns:
        The quantization of each module.

    Raises:
        ValueError: If ``quantization_config`` describes anything other than
            Xiaomi's FP8 dense linears and MXFP4 experts.
    """
    quant: Mapping[str, object] = (
        getattr(config, "quantization_config", None) or {}
    )
    for key, value in _CHECKPOINT_QUANTIZATION.items():
        if quant.get(key) != value:
            raise ValueError(
                f"MiMo-V2: quantization_config.{key} is {quant.get(key)!r}; "
                f"only {value!r} is implemented."
            )
    moe = set(moe_layers(config))
    all_layers = set(range(config.num_hidden_layers))
    return MiMoV2QuantScheme(
        dense=QuantConfig.blockscaled_fp8(
            mlp_quantized_layers=all_layers - moe,
            # qkv_proj only; o_proj is BF16 in every layer.
            attn_quantized_layers=all_layers,
            block_size=FP8_BLOCK,
            embedding_output_dtype=DType.bfloat16,
        ),
        experts=QuantConfig.mxfp4(
            mlp_quantized_layers=moe,
            embedding_output_dtype=DType.bfloat16,
        ),
    )
