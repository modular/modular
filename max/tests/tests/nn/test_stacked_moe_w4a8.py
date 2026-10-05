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
"""CPU checks for StackedMoE's W4A8 MXFP4 experts and the sigmoid router.

The W4A8 kernels run only on SM100, so these check what a CPU can: the
declared weights, that tensor parallelism hands each device the interleave of
its own row-major scale slice, and the graph the forward builds.
"""

from __future__ import annotations

import dataclasses
import functools
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, ShardingStrategy, TensorType
from max.nn import kernels
from max.nn.moe import (
    GateUpFormat,
    SigmoidTopKRouter,
    StackedMoE,
    interleaved_block_scales_shape,
    quant_strategy,
    stacked_moe,
)
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

EXPERTS, TOP_K, HIDDEN, WIDTH = 4, 2, 256, 512
SCALES = ("_gate_up_scale", "_down_scale")


def _mxfp4(scale_dtype: DType = DType.float8_e8m0fnu) -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 32),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=scale_dtype,
            block_size=(1, 32),
        ),
        mlp_quantized_layers={0},
        attn_quantized_layers=set(),
        format=QuantFormat.MXFP4,
    )


def _moe(
    device: DeviceRef | None = None,
    scale_dtype: DType = DType.float8_e8m0fnu,
    moe_dim: int = WIDTH,
) -> StackedMoE:
    return StackedMoE(
        devices=[device or DeviceRef.CPU()],
        hidden_dim=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_token=TOP_K,
        moe_dim=moe_dim,
        gate_cls=functools.partial(SigmoidTopKRouter, norm_topk_prob=True),
        quant_config=_mxfp4(scale_dtype),
        router_dtype=DType.float32,
        mxfp8_activations=True,
    )


def _interleave(scales: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """``[E, N, K/32]`` scales, each where the SM100 grouped matmul reads it
    (``set_scale_factor`` in ``fp4_utils.mojo``)."""
    experts, rows, cols = scales.shape
    out = np.zeros(
        (experts, *interleaved_block_scales_shape(rows, cols)), np.uint8
    )
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    out[:, r // 128, c // 4, r % 32, (r % 128) // 32, c % 4] = scales
    return out


def test_declares_interleaved_expert_stacks() -> None:
    weights = _moe().raw_state_dict()
    shapes = {name: list(weight.shape) for name, weight in weights.items()}

    assert shapes == {
        "experts.gate_up_proj": [EXPERTS, 2 * WIDTH, HIDDEN // 2],
        "experts.gate_up_proj_scale": [
            EXPERTS,
            2 * WIDTH // 128,
            HIDDEN // 128,
            32,
            4,
            4,
        ],
        "experts.down_proj": [EXPERTS, HIDDEN, WIDTH // 2],
        "experts.down_proj_scale": [
            EXPERTS,
            HIDDEN // 128,
            WIDTH // 128,
            32,
            4,
            4,
        ],
        "gate.gate_score.weight": [EXPERTS, HIDDEN],
        "gate.e_score_correction_bias": [EXPERTS],
    }
    assert weights["gate.gate_score.weight"].dtype == DType.float32
    assert weights["gate.e_score_correction_bias"].dtype == DType.float32


@pytest.mark.parametrize("num_devices", [2, 4])
def test_each_shard_is_the_interleave_of_its_slice(num_devices: int) -> None:
    rng = np.random.default_rng(0)
    row_major = {
        "_gate_up_scale": rng.integers(
            0, 255, (EXPERTS, 2 * WIDTH, HIDDEN // 32), dtype=np.uint8
        ),
        "_down_scale": rng.integers(
            0, 255, (EXPERTS, HIDDEN, WIDTH // 32), dtype=np.uint8
        ),
    }
    names = {
        "_gate_up_scale": "experts.gate_up_proj_scale",
        "_down_scale": "experts.down_proj_scale",
    }
    # The CPU has no E8M0 type, and slicing moves bytes whatever their type.
    moe = _moe(scale_dtype=DType.uint8)
    moe.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices)
    state = {
        names[attr]: Buffer.from_numpy(_interleave(scales))
        for attr, scales in row_major.items()
    }
    moe.load_state_dict(state, weight_alignment=1, strict=False)
    shards = moe.shard([DeviceRef.CPU()] * num_devices)

    with Graph("scale_shards") as graph:
        graph.output(
            *(getattr(shard, attr) for attr in SCALES for shard in shards)
        )
    outputs = (
        InferenceSession(devices=[CPU()])
        .load(graph, weights_registry=state)
        .execute()
    )

    for i, attr in enumerate(SCALES):
        scales = row_major[attr]
        for rank in range(num_devices):
            if attr == "_gate_up_scale":
                # The rank's gate rows, then its up rows.
                rows = WIDTH // num_devices
                gate = scales[:, rank * rows : (rank + 1) * rows]
                up = scales[:, WIDTH + rank * rows : WIDTH + (rank + 1) * rows]
                want = np.concatenate([gate, up], axis=1)
            else:
                cols = scales.shape[2] // num_devices
                want = scales[:, :, rank * cols : (rank + 1) * cols]
            got = outputs[i * num_devices + rank]
            assert isinstance(got, Buffer)
            np.testing.assert_array_equal(
                got.to_numpy(), _interleave(want), err_msg=f"{attr} {rank}"
            )


def test_shards_keep_the_router_and_its_correction_bias() -> None:
    moe = _moe()
    moe.sharding_strategy = ShardingStrategy.tensor_parallel(2)
    shards = moe.shard([DeviceRef.CPU(), DeviceRef.CPU()])

    for shard in shards:
        assert isinstance(shard.gate, SigmoidTopKRouter)
        assert shard.moe_dim == WIDTH // 2
        assert shard.mxfp8_activations
        assert shard.router_dtype == DType.float32
        assert shard.gate.e_score_correction_bias.dtype == DType.float32


def test_tensor_parallel_needs_whole_granules() -> None:
    moe = _moe(moe_dim=128)
    with pytest.raises(ValueError, match="scale granules"):
        moe.sharding_strategy = ShardingStrategy.tensor_parallel(2)


def test_mxfp8_activations_need_mxfp4_weights() -> None:
    with pytest.raises(ValueError, match="MXFP4"):
        StackedMoE(
            devices=[DeviceRef.CPU()],
            hidden_dim=HIDDEN,
            num_experts=EXPERTS,
            num_experts_per_token=TOP_K,
            moe_dim=WIDTH,
            gate_cls=SigmoidTopKRouter,
            quant_config=dataclasses.replace(
                _mxfp4(), format=QuantFormat.NVFP4
            ),
            mxfp8_activations=True,
        )


@pytest.mark.parametrize("rows, cols", [(64, 4), (128, 2)])
def test_interleaved_shape_needs_whole_granules(rows: int, cols: int) -> None:
    with pytest.raises(ValueError, match="interleave granules"):
        interleaved_block_scales_shape(rows, cols)


def test_forward_refuses_non_sm100() -> None:
    moe = _moe(DeviceRef.GPU(0))
    with (
        mock.patch.object(
            stacked_moe, "accelerator_architecture_name", return_value="gfx950"
        ),
        Graph(
            "w4a8_refused",
            input_types=[
                TensorType(DType.bfloat16, ["tokens", HIDDEN], DeviceRef.GPU(0))
            ],
        ) as graph,
        pytest.raises(ValueError, match="SM100"),
    ):
        moe(graph.inputs[0].tensor)


def test_forward_builds_the_w4a8_graph() -> None:
    """The forward's two grouped matmuls take one expert slot per expert and
    the step's routed row count, and the quantize fuses the row gather."""
    moe = _moe(DeviceRef.GPU(0))
    moe.sharding_strategy = ShardingStrategy.tensor_parallel(1)
    (shard,) = moe.shard([DeviceRef.GPU(0)])
    with (
        mock.patch.object(
            stacked_moe, "accelerator_architecture_name", return_value="sm_100a"
        ),
        mock.patch.object(kernels, "_is_sm10x_gpu", return_value=True),
        mock.patch.object(
            quant_strategy,
            "grouped_matmul_block_scaled",
            wraps=quant_strategy.grouped_matmul_block_scaled,
        ) as matmul,
        mock.patch.object(
            quant_strategy,
            "grouped_quantize_dynamic_block_scaled",
            wraps=quant_strategy.grouped_quantize_dynamic_block_scaled,
        ) as quantize,
        Graph(
            "w4a8",
            input_types=[
                TensorType(DType.bfloat16, ["tokens", HIDDEN], DeviceRef.GPU(0))
            ],
        ) as graph,
    ):
        out = shard(graph.inputs[0].tensor)
        graph.output(out)

    assert out.dtype == DType.bfloat16
    assert list(out.shape) == [graph.inputs[0].tensor.shape[0], HIDDEN]
    assert matmul.call_count == 2
    for call in matmul.call_args_list:
        # Element 1 of the usage stats is the loop bound over expert slots.
        assert int(call.args[6].shape[0]) == EXPERTS
        assert call.kwargs["estimated_total_m"] is not None
    assert quantize.call_count == 2
    gate_up_quantize, down_quantize = quantize.call_args_list
    assert gate_up_quantize.kwargs["indices"] is not None
    assert down_quantize.kwargs["indices"] is None
    for call in quantize.call_args_list:
        assert call.kwargs["scales_type"] == DType.float8_e8m0fnu
        assert call.kwargs["out_type"] == DType.float8_e4m3fn
        assert call.kwargs["sf_vector_size"] == 32


def test_interleaved_gate_up_splits_axiswise() -> None:
    moe = StackedMoE(
        devices=[DeviceRef.CPU()],
        hidden_dim=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_token=TOP_K,
        moe_dim=WIDTH,
        gate_cls=SigmoidTopKRouter,
        gate_up_format=GateUpFormat.INTERLEAVED,
        quant_config=_mxfp4(),
        mxfp8_activations=True,
    )
    moe.sharding_strategy = ShardingStrategy.tensor_parallel(2)
    with Graph("interleaved_shards"):
        shards = moe.shard([DeviceRef.CPU(), DeviceRef.CPU()])
        shapes = [list(s._gate_up_scale.shape) for s in shards]

    assert shapes == [[EXPERTS, WIDTH // 128, HIDDEN // 128, 32, 4, 4]] * 2
