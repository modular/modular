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

"""Block-scaled FP8 tests for Qwen3.5 ModuleV3.

``Qwen/Qwen3.8-27B-FP8`` ships ``quant_method: "fp8"``: one
``float8_e4m3fn`` weight per projection with an inverse scale per 128x128
block in ``weight_scale_inv``, against dynamically quantized activations.
The modelopt parser reads a per-module ``quantized_layers`` map and cannot
describe that layout, so the scheme parser has a separate arm. These tests
pin what it accepts and what it refuses, from the checkpoint's own scale
names; the weight layout they imply is covered by serving the checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.graph.weights import WeightData
from max.nn.common_layers.mesh_axis import TP
from max.nn.quant_config import QuantFormat, ScaleGranularity, ScaleOrigin
from max.pipelines.architectures.qwen3_5_modulev3.layers.sharding import (
    check_block_split,
    check_segment_split,
)
from max.pipelines.architectures.qwen3_5_modulev3.quantization import (
    parse_quant_scheme,
)
from max.tensor.sharding import DeviceMesh

NUM_LAYERS = 4
# Layer 3 is full attention; 0-2 are linear attention (Gated DeltaNet).
_FULL_ATTN_LAYERS = (3,)


def _hf_quant_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    config.update(overrides)
    return config


def _quantized_modules(layer: int) -> list[str]:
    """The projections Qwen3.8's checkpoint quantizes in layer ``layer``."""
    modules = [
        f"layers.{layer}.mlp.{proj}"
        for proj in ("gate_proj", "up_proj", "down_proj")
    ]
    if layer in _FULL_ATTN_LAYERS:
        modules += [
            f"layers.{layer}.self_attn.{proj}"
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
    else:
        modules += [
            f"layers.{layer}.linear_attn.{proj}"
            for proj in ("in_proj_qkv", "in_proj_z", "out_proj")
        ]
    return modules


def _state_dict() -> dict[str, WeightData]:
    """The tensors the scheme parser reads: the embedding and the block scales."""
    weights = {
        "embed_tokens.weight": WeightData.from_numpy(
            np.zeros((8, 4), dtype=np.float32), "embed_tokens.weight"
        )
    }
    for layer in range(NUM_LAYERS):
        for module in _quantized_modules(layer):
            name = f"{module}.weight_scale_inv"
            weights[name] = WeightData.from_numpy(
                np.zeros((1, 1), dtype=np.float32), name
            )
    return weights


def test_block_fp8_reads_both_module_sets() -> None:
    scheme = parse_quant_scheme(_hf_quant_config(), _state_dict(), NUM_LAYERS)
    assert scheme is not None

    # The layout has one format, so one config serves both module sets.
    assert scheme.mlp is scheme.attn
    assert scheme.mlp is not None
    assert scheme.mlp.format == QuantFormat.BLOCKSCALED_FP8
    assert scheme.mlp.weight_scale.granularity == ScaleGranularity.BLOCK
    assert scheme.mlp.weight_scale.block_size == (128, 128)
    # The block matmul takes fp32 scales, so the loader widens the
    # checkpoint's bf16 ones.
    assert scheme.mlp.weight_scale.dtype == DType.float32
    assert scheme.mlp.input_scale.granularity == ScaleGranularity.BLOCK
    assert scheme.mlp.input_scale.origin == ScaleOrigin.DYNAMIC
    assert scheme.mlp.input_scale.block_size == (1, 128)

    every_layer = frozenset(range(NUM_LAYERS))
    assert scheme.mlp_layers == every_layer
    assert scheme.attn_layers == every_layer
    assert not scheme.quantize_lm_head
    for layer in range(NUM_LAYERS):
        assert scheme.mlp_config(layer) is scheme.mlp
        assert scheme.attn_config(layer) is scheme.attn
    assert scheme.compute_dtype == DType.float32


def test_block_fp8_requires_dynamic_activations() -> None:
    with pytest.raises(ValueError, match="activation_scheme='dynamic'"):
        parse_quant_scheme(
            _hf_quant_config(activation_scheme="static"),
            _state_dict(),
            NUM_LAYERS,
        )


def test_block_fp8_requires_128_wide_blocks() -> None:
    with pytest.raises(
        ValueError, match=r"wired for weight_block_size \[128, 128\]"
    ):
        parse_quant_scheme(
            _hf_quant_config(weight_block_size=[128, 64]),
            _state_dict(),
            NUM_LAYERS,
        )


def test_block_fp8_rejects_a_layer_quantized_in_part() -> None:
    state = _state_dict()
    del state["layers.0.mlp.gate_proj.weight_scale_inv"]
    with pytest.raises(ValueError, match="partially quantized"):
        parse_quant_scheme(_hf_quant_config(), state, NUM_LAYERS)


def test_block_fp8_rejects_a_scale_outside_a_unit() -> None:
    # The MoE router and the GDN's in_proj_a/in_proj_b have no projection in
    # the scheme, so a scaled one is a checkpoint this path cannot read.
    state = _state_dict()
    name = "layers.0.mlp.gate.weight_scale_inv"
    state[name] = WeightData.from_numpy(
        np.zeros((1, 1), dtype=np.float32), name
    )
    with pytest.raises(ValueError, match="partially quantized"):
        parse_quant_scheme(_hf_quant_config(), state, NUM_LAYERS)


def test_block_fp8_rejects_a_scale_past_the_last_layer() -> None:
    state = _state_dict()
    name = f"layers.{NUM_LAYERS}.mlp.gate_proj.weight_scale_inv"
    state[name] = WeightData.from_numpy(
        np.zeros((1, 1), dtype=np.float32), name
    )
    with pytest.raises(ValueError, match=f"declares {NUM_LAYERS} layers"):
        parse_quant_scheme(_hf_quant_config(), state, NUM_LAYERS)


def _tp_mesh(tp: int) -> DeviceMesh:
    """A ``tp``-way tensor-parallel mesh, simulated on the host."""
    return DeviceMesh(devices=(CPU(),) * tp, mesh_shape=(tp,), axis_names=(TP,))


def test_block_fp8_segments_split_in_whole_blocks_per_device() -> None:
    # 512 rows over 2 devices is 256 rows, two blocks, each.
    check_segment_split([512, 256], 128, _tp_mesh(2))
    # One device keeps the whole segment, so it only has to be whole blocks.
    check_segment_split([128, 256], 128, _tp_mesh(1))
    # 128 rows is one block, but over 2 devices each holds half of it.
    with pytest.raises(ValueError, match="not a whole number of blocks"):
        check_segment_split([512, 128], 128, _tp_mesh(2))
    with pytest.raises(ValueError, match="not a whole number of blocks"):
        check_segment_split([192], 128, _tp_mesh(1))


def test_block_fp8_k_split_refuses_a_partial_block() -> None:
    # `(None, TP)` splits the input columns, which the scales also block.
    check_block_split([256, 512], (128, 128), (None, TP), _tp_mesh(2))
    with pytest.raises(ValueError, match="axis 1"):
        check_block_split([256, 384], (128, 128), (None, TP), _tp_mesh(2))
    # The output rows take the same rule under `(TP, None)`.
    with pytest.raises(ValueError, match="axis 0"):
        check_block_split([384, 256], (128, 128), (TP, None), _tp_mesh(2))
    # Nothing is split on one device, so nothing is refused.
    check_block_split([200, 300], (128, 128), (None, TP), _tp_mesh(1))


def test_block_fp8_split_check_accepts_a_mesh_without_a_tp_axis() -> None:
    """A single-device mesh names only its default axis, never TP."""
    mesh = DeviceMesh(devices=(CPU(),), mesh_shape=(1,), axis_names=("_",))
    check_block_split([200, 300], (128, 128), (None, TP), mesh)
    check_block_split([200, 300], (128, 128), (TP, None), mesh)
