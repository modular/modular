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
"""The non-EP FP8 MoE builds without a host read of its usage stats on SM100.

`moe_create_indices` leaves the expert usage stats on the device, and the
grouped FP8 op takes them as host scalars, so the layer used to copy them back
and sync the stream. On SM100 the persistent kernel never reads them, so the
layer passes shape-derived bounds instead and the graph has no
device-to-host transfer. The grouped matmul itself still copies device stats it
is handed, because their active-group count may be below the slot count.

The tests run on a virtual SM100 device, as a CPU worker precompiling a B200
graph does. Virtual-device mode latches process-wide at the first device
creation, so the knobs are set at import.
"""

from __future__ import annotations

from max.driver import (
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)

set_virtual_device_api("cuda")
set_virtual_device_target_arch("sm_100a")
set_virtual_device_count(1)

import pytest
import torch
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef, Graph, Shape, TensorType
from max.graph.weights import WeightData
from max.nn.legacy import kernels
from max.nn.legacy.moe import MoEQuantized
from max.nn.legacy.moe.quant_strategy import Fp8Strategy
from max.nn.legacy.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

_E = 8
_K = 256
_N = 256
_M = 64
# A host read of the device stats; the weights' host-to-device transfers differ.
_D2H = 'to <"cpu"'
_FP8_OP = "mo.grouped.matmul.dynamic.scaled.fp8"


def _fp8_config() -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 128),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=(128, 128),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.BLOCKSCALED_FP8,
    )


def _moe_graph_text() -> str:
    moe = MoEQuantized(
        devices=[DeviceRef.GPU()],
        hidden_dim=256,
        num_experts=4,
        num_experts_per_token=2,
        moe_dim=128,
        dtype=DType.float8_e4m3fn,
        quant_config=_fp8_config(),
    )
    weights: dict[str, WeightData | torch.Tensor] = {
        "gate.gate_score.weight": torch.zeros(4, 256, dtype=torch.bfloat16)
    }
    for expert in range(4):
        for name, (out, inp) in {
            "gate_proj": (128, 256),
            "up_proj": (128, 256),
            "down_proj": (256, 128),
        }.items():
            key = f"experts.{expert}.{name}"
            weights[f"{key}.weight"] = WeightData(
                Buffer.from_dlpack(
                    torch.zeros(out, inp, dtype=torch.uint8)
                ).view(DType.float8_e4m3fn),
                f"{key}.weight",
                DType.float8_e4m3fn,
                Shape((out, inp)),
            )
            weights[f"{key}.weight_scale"] = torch.ones(out // 128, inp // 128)
    moe.load_state_dict(weights)
    with Graph(
        "fp8_moe_usage_stats",
        input_types=[
            TensorType(DType.bfloat16, ["tokens", 256], DeviceRef.GPU())
        ],
    ) as graph:
        graph.output(moe(graph.inputs[0].tensor))
    return str(graph)


def _grouped_matmul_graph_text(stats_device: DeviceRef) -> str:
    gpu = DeviceRef.GPU(0)
    with Graph(
        "fp8_grouped_matmul_usage_stats",
        input_types=[
            TensorType(DType.float8_e4m3fn, [_M, _K], gpu),
            TensorType(DType.float32, [_K // 128, _M], gpu),
            TensorType(DType.float8_e4m3fn, [_E, _N, _K], gpu),
            TensorType(DType.float32, [_E, _N // 128, _K // 128], gpu),
            TensorType(DType.uint32, [_E + 1], gpu),
            TensorType(DType.int32, [_E], gpu),
            TensorType(DType.uint32, [2], stats_device),
        ],
    ) as graph:
        hidden, scales, weight, weight_scales, starts, ids, stats = (
            v.tensor for v in graph.inputs
        )
        graph.output(
            Fp8Strategy(_fp8_config(), DType.float8_e4m3fn).grouped_matmul(
                weight,
                weight_scales,
                expert_inputs=(hidden, scales, starts, ids, stats),
            )
        )
    return str(graph)


def test_moe_built_for_a_virtual_sm100_needs_no_transfer() -> None:
    """The SM100 choice follows the compile target, not a visible GPU."""
    assert kernels._is_sm10x_gpu()
    text = _moe_graph_text()
    assert _FP8_OP in text
    assert _D2H not in text


def test_moe_needs_no_transfer_on_sm100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: True)
    text = _moe_graph_text()
    assert _FP8_OP in text
    assert _D2H not in text


def test_moe_copies_the_stats_to_the_host_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: False)
    text = _moe_graph_text()
    assert _FP8_OP in text
    assert _D2H in text


@pytest.mark.parametrize("sm10x", [True, False])
def test_grouped_matmul_copies_device_stats_it_is_handed(
    monkeypatch: pytest.MonkeyPatch, sm10x: bool
) -> None:
    """Device stats may report fewer groups than slots, so they are read."""
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: sm10x)
    assert _D2H in _grouped_matmul_graph_text(DeviceRef.GPU(0))


@pytest.mark.parametrize("sm10x", [True, False])
def test_host_stats_pass_through_untouched(
    monkeypatch: pytest.MonkeyPatch, sm10x: bool
) -> None:
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: sm10x)
    text = _grouped_matmul_graph_text(DeviceRef.CPU())
    assert _FP8_OP in text
    assert _D2H not in text
