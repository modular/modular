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
"""Which Laguna layers are NVFP4 is a property of the checkpoint, not of its
``quantization_config``: the released checkpoints write their ``ignore`` list
as regexes, which MAX's fnmatch-glob matcher silently mis-reads. So
``_resolve_nvfp4_quant_config`` derives the per-layer sets from the weights,
and these tests pin the two shapes that exist -- S-2.1's mixed precision and
M.1's uniform NVFP4 -- against state dicts of dtype/shape-only stubs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from max.dtype import DType
from max.graph import Shape
from max.graph.weights import WeightData
from max.pipelines.architectures.laguna.model import LagunaModel

HIDDEN = 64
MOE_DIM = 32
PACKED_HIDDEN = HIDDEN // 2
GROUPS = HIDDEN // 16
NUM_EXPERTS = 2
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _weight(shape: tuple[int, ...], dtype: DType) -> WeightData:
    """A stub carrying only the dtype and shape the resolver reads."""
    return WeightData(
        data=np.zeros(shape, dtype=np.uint8),
        name="stub",
        dtype=dtype,
        shape=Shape(shape),
    )


def _bf16(state_dict: dict[str, WeightData], prefix: str) -> None:
    for projection in PROJECTIONS:
        state_dict[f"{prefix}.{projection}.weight"] = _weight(
            (MOE_DIM, HIDDEN), DType.bfloat16
        )


def _nvfp4(state_dict: dict[str, WeightData], prefix: str) -> None:
    for projection in PROJECTIONS:
        base = f"{prefix}.{projection}"
        state_dict[f"{base}.weight"] = _weight(
            (MOE_DIM, PACKED_HIDDEN), DType.uint8
        )
        state_dict[f"{base}.weight_scale"] = _weight(
            (MOE_DIM, GROUPS), DType.float8_e4m3fn
        )
        state_dict[f"{base}.weight_scale_2"] = _weight((), DType.float32)
        state_dict[f"{base}.input_scale"] = _weight((), DType.float32)


def _state_dict(
    mlp_layer_types: Sequence[str],
    *,
    quantized_experts: Sequence[int],
    quantized_dense: Sequence[int] = (),
    quantized_shared_experts: bool,
) -> dict[str, WeightData]:
    """Builds a post-adapter state dict for a model of the given shape."""
    state_dict: dict[str, WeightData] = {
        "embed_tokens.weight": _weight((128, HIDDEN), DType.bfloat16),
    }
    for i, mlp_type in enumerate(mlp_layer_types):
        state_dict[f"layers.{i}.self_attn.q_proj.weight"] = _weight(
            (HIDDEN, HIDDEN), DType.bfloat16
        )
        if mlp_type == "dense":
            fill = _nvfp4 if i in quantized_dense else _bf16
            fill(state_dict, f"layers.{i}.mlp")
            continue
        fill = _nvfp4 if i in quantized_experts else _bf16
        for expert in range(NUM_EXPERTS):
            fill(state_dict, f"layers.{i}.mlp.experts.{expert}")
        shared_fill = _nvfp4 if quantized_shared_experts else _bf16
        shared_fill(state_dict, f"layers.{i}.mlp.shared_experts")
    return state_dict


def _resolve(
    mlp_layer_types: Sequence[str], state_dict: Mapping[str, WeightData]
) -> Any:
    """Runs the resolver over stand-ins for the two configs it reads."""
    num_layers = len(mlp_layer_types)
    model_config = SimpleNamespace(
        quant_config=None,
        mlp_layer_types=list(mlp_layer_types),
        num_hidden_layers=num_layers,
        attn_dtype=DType.bfloat16,
        dtype=DType.uint8,
        unquantized_dtype=DType.bfloat16,
    )
    LagunaModel._resolve_nvfp4_quant_config(
        SimpleNamespace(num_hidden_layers=num_layers),
        model_config,  # type: ignore[arg-type]
        state_dict,
    )
    return model_config.quant_config


S21_MLP_TYPES = ("dense", "sparse", "sparse", "sparse")


def test_mixed_precision_experts() -> None:
    """S-2.1's shape: a bf16 dense layer, NVFP4 experts in the middle, bf16
    experts at the end, and bf16 shared experts throughout."""
    quant_config = _resolve(
        S21_MLP_TYPES,
        _state_dict(
            S21_MLP_TYPES,
            quantized_experts=(1, 2),
            quantized_shared_experts=False,
        ),
    )

    assert quant_config is not None
    assert quant_config.is_nvfp4
    assert quant_config.mlp_quantized_layers == {1, 2}
    assert quant_config.attn_quantized_layers == set()
    assert quant_config.shared_experts_weight_dtype == DType.bfloat16
    assert quant_config.embedding_output_dtype == DType.bfloat16


def test_uniformly_quantized_mlps() -> None:
    """M.1's shape: every MLP is NVFP4, dense prefix and shared experts too."""
    quant_config = _resolve(
        S21_MLP_TYPES,
        _state_dict(
            S21_MLP_TYPES,
            quantized_experts=(1, 2, 3),
            quantized_dense=(0,),
            quantized_shared_experts=True,
        ),
    )

    assert quant_config is not None
    assert quant_config.mlp_quantized_layers == {0, 1, 2, 3}
    assert quant_config.attn_quantized_layers == set()
    assert quant_config.shared_experts_weight_dtype is None


def test_bfloat16_checkpoint_stays_unquantized() -> None:
    quant_config = _resolve(
        S21_MLP_TYPES,
        _state_dict(
            S21_MLP_TYPES,
            quantized_experts=(),
            quantized_shared_experts=False,
        ),
    )

    assert quant_config is None


def test_block_scales_without_global_scales_are_rejected() -> None:
    state_dict = _state_dict(
        S21_MLP_TYPES,
        quantized_experts=(1, 2),
        quantized_shared_experts=False,
    )
    del state_dict["layers.1.mlp.experts.0.gate_proj.input_scale"]

    with pytest.raises(ValueError, match="but not input_scale"):
        _resolve(S21_MLP_TYPES, state_dict)
