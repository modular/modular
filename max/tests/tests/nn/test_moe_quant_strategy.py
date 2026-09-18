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

"""Checks for the operand normalization the fused SwiGLU kernels share.

Every caller that reaches those kernels hands them the same two easily
mis-derived operands: the per-expert SiLU-output scale, which the kernel
consumes INVERTED, and the usage stats, which it binds as a host scalar. A
caller that re-derives either one instead of going through
``prepare_swiglu_operands`` gets numerically wrong output with no error
anywhere -- shape, dtype and device all still check out.

The normalizer reads only device and expert count, so these build the operand
tuple from plain float32/uint32 rather than the packed NVFP4 dtypes the kernel
itself takes. CPU only.
"""

from __future__ import annotations

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.moe.quant_strategy import NvMxf4f8Strategy
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

_N_EXPERTS = 4


def _nvfp4_strategy() -> NvMxf4f8Strategy:
    config = QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 16),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=(1, 16),
        ),
        mlp_quantized_layers={0},
        attn_quantized_layers={0},
        format=QuantFormat.NVFP4,
    )
    return NvMxf4f8Strategy(config, DType.bfloat16)


def _zeros(shape: list[int], dtype: DType, device: DeviceRef) -> TensorValue:
    return ops.constant(
        np.zeros(shape, dtype=dtype.to_numpy()), dtype=dtype, device=device
    )


def _expert_inputs(
    device: DeviceRef, stats_device: DeviceRef | None = None
) -> tuple[TensorValue, ...]:
    """The six-tuple shape every fused SwiGLU entry point takes."""
    return (
        _zeros([8, 16], DType.float32, device),
        _zeros([8, 1], DType.float32, device),
        _zeros([_N_EXPERTS + 1], DType.uint32, device),
        _zeros([_N_EXPERTS], DType.uint32, device),
        _zeros([_N_EXPERTS], DType.int32, device),
        _zeros([2], DType.uint32, stats_device or device),
    )


def test_prepare_swiglu_operands_inverts_the_silu_output_scale(
    session: InferenceSession,
) -> None:
    """The kernel consumes 1/s, not s.

    Executed rather than inspected: a caller that forwards the raw scale is
    wrong by a factor of s squared while every static check still passes.
    """
    raw = np.array([0.5, 2.0, 4.0, 0.25], dtype=np.float32)
    cpu = DeviceRef.CPU()

    with Graph(
        "prepare_swiglu_operands_inversion",
        input_types=[TensorType(DType.float32, [_N_EXPERTS], device=cpu)],
    ) as graph:
        (input_scales,) = (inp.tensor for inp in graph.inputs)
        prepared = _nvfp4_strategy().prepare_swiglu_operands(
            _expert_inputs(cpu),
            expert_scales=_zeros([_N_EXPERTS], DType.float32, cpu),
            input_scales=input_scales,
        )
        assert prepared.c_input_scales is not None
        graph.output(prepared.c_input_scales)

    result = session.load(graph).execute(Buffer.from_numpy(raw))[0]
    assert isinstance(result, Buffer)
    np.testing.assert_allclose(result.to_numpy(), 1.0 / raw, rtol=1e-6)


def test_prepare_swiglu_operands_puts_usage_stats_on_the_host() -> None:
    """The active-expert count binds as a host scalar.

    Dispatch leaves the stats on the device, so a caller that forwards them
    unchanged feeds a device tensor to a host operand.
    """
    gpu = DeviceRef.GPU()
    with Graph("prepare_swiglu_operands_host_stats", input_types=[]):
        prepared = _nvfp4_strategy().prepare_swiglu_operands(
            _expert_inputs(gpu),
            expert_scales=_zeros([_N_EXPERTS], DType.float32, gpu),
            input_scales=_zeros([_N_EXPERTS], DType.float32, gpu),
        )
        assert prepared.usage_stats.device == DeviceRef.CPU()


def test_prepare_swiglu_operands_places_scales_on_the_hidden_device() -> None:
    """Both scale tensors follow the activations, wherever they were built."""
    gpu = DeviceRef.GPU()
    cpu = DeviceRef.CPU()
    with Graph("prepare_swiglu_operands_scale_device", input_types=[]):
        prepared = _nvfp4_strategy().prepare_swiglu_operands(
            _expert_inputs(gpu),
            expert_scales=_zeros([_N_EXPERTS], DType.float32, cpu),
            input_scales=_zeros([_N_EXPERTS], DType.float32, cpu),
        )
        assert prepared.c_input_scales is not None
        assert prepared.expert_scales is not None
        assert prepared.c_input_scales.device == gpu
        assert prepared.expert_scales.device == gpu
