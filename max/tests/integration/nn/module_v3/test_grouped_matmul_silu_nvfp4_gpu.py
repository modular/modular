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
"""Checks that the two SwiGLU routes in ``quant_ops.grouped_matmul_silu``
produce the same NVFP4 down-projection output.

A float32 SwiGLU output takes the fused quantized SiLU, because the down
matmul quantizes from bf16 only. A bf16 output skips it and leaves the down
matmul to quantize, with the same static per-expert input scale. Models ship
on both routes, so moving between them must not move their numerics. Which
route each output takes is pinned on CPU by ``test_quant_ops.py``.
"""

from __future__ import annotations

import numpy as np
from max import driver
from max.driver import CPU, Accelerator, LaunchTraceEntry
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.common_layers import quant_ops
from max.nn.common_layers.quant_tensor import NVFP4Activation, NVFP4Tensor
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.tensor import Tensor
from max.tensor.executor import CompilingExecutor, set_default_executor
from max.tensor.realization_context import lazy

_NVFP4_BLOCK = (1, 16)
_HIDDEN = 256
_MOE_DIM = 128
_EXPERTS = 8

# NVFP4's largest FP4 magnitude and the largest E4M3 block scale.
_NVFP4_MAX = 6.0
_E4M3_MAX = 448.0

# The E2M1 value of each 4-bit code; a byte packs two, low nibble first.
_FP4_VALUES = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
)


def _nvfp4_quant_config() -> QuantConfig:
    """A modelopt NVFP4 config, mirroring ``parse_quant_config``'s output."""
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=_NVFP4_BLOCK,
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=_NVFP4_BLOCK,
        ),
        mlp_quantized_layers={0},
        attn_quantized_layers={0},
        format=QuantFormat.NVFP4,
    )


def _round_to_bf16(x: np.ndarray) -> np.ndarray:
    """Rounds float32 values to the nearest bfloat16, ties to even."""
    bits = x.astype(np.float32).view(np.uint32)
    bits = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (bits & np.uint32(0xFFFF0000)).view(np.float32)


def _down_weight(codes: np.ndarray, input_scale: float) -> NVFP4Tensor:
    """Expert-stacked NVFP4 down weights, as ``QuantizedMoE`` stacks them.

    ``codes`` are the packed FP4 values. ``input_scale`` is the checkpoint's
    static activation scale, which both quantization routes divide the
    activation by before packing it.
    """
    device = Accelerator()
    return NVFP4Tensor(
        weight=Tensor(codes, device=device),
        # A unit block scale keeps the weight values the FP4 codes themselves,
        # so the comparison is not dominated by the block-scale rounding.
        weight_scale=Tensor.ones(
            (_EXPERTS, _HIDDEN, _MOE_DIM // _NVFP4_BLOCK[1]),
            dtype=DType.float32,
            device=device,
        ).cast(DType.float8_e4m3fn),
        weight_scale_2=Tensor.ones(
            (_EXPERTS,), dtype=DType.float32, device=CPU()
        ),
        input_scale=Tensor.full(
            (_EXPERTS,), input_scale, dtype=DType.float32, device=CPU()
        ),
        block_size=_NVFP4_BLOCK,
    )


def _indices() -> tuple[Tensor, Tensor, Tensor]:
    """One token per expert: group offsets, expert ids and scale offsets.

    ``moe_create_indices`` gives every group its own scale tile; with one
    token each they all start inside tile 0, so the offsets have to separate
    them or the groups overwrite each other's scales.
    """
    device = Accelerator()
    return (
        Tensor(np.arange(_EXPERTS + 1, dtype=np.uint32), device=device),
        Tensor(np.arange(_EXPERTS, dtype=np.int32), device=device),
        Tensor(np.arange(_EXPERTS, dtype=np.uint32), device=device),
    )


def _down_projections(
    gate_up_out: np.ndarray, codes: np.ndarray, input_scale: float
) -> tuple[Tensor, Tensor]:
    """Stages both routes' down projections of ``gate_up_out`` as one graph.

    The graph compiles once and touches the device only after that compile,
    which is what lets precompile_mefs record it on a CPU build action. Only
    the returned outputs outlive the lazy scope, so the intermediates stay
    inside the graph rather than becoming outputs of it.
    """
    with set_default_executor(CompilingExecutor()), lazy():
        config = _nvfp4_quant_config()
        starts, ids, offsets = _indices()
        down = _down_weight(codes, input_scale)
        # A graph input rather than a constant, so the compiler cannot fold
        # the SwiGLU away; the transfer is an op of the graph, since an eager
        # copy would allocate on the device before the compile.
        x = Tensor.from_graph_value(
            ops.transfer_to(
                TensorValue(Tensor.from_dlpack(gate_up_out)),
                DeviceRef.from_device(Accelerator()),
            )
        ).cast(DType.bfloat16)
        plain = quant_ops.grouped_silu(x, starts)
        assert isinstance(plain, Tensor)
        fused = quant_ops.grouped_silu(
            x, starts, down, config, scales_offset=offsets
        )
        assert isinstance(fused, NVFP4Activation)
        from_fused, from_bf16 = (
            quant_ops.grouped_matmul(
                activation, down, starts, ids, None, scales_offset=offsets
            )
            .cast(DType.float32)
            .to(CPU())
            for activation in (fused, plain)
        )
        return from_fused, from_bf16


def test_fused_and_bf16_swiglu_agree_through_the_down_matmul() -> None:
    """Both SwiGLU routes land on the same down-projection output.

    They quantize the same activations with the same static per-expert scale;
    the bf16 route just rounds to bfloat16 on the way, which is finer than
    the FP4 step it then quantizes to. The trace checks that both SiLU
    kernels and both down matmuls ran rather than being folded away.
    """
    gate_up_out = _round_to_bf16(
        np.random.default_rng(3)
        .standard_normal((_EXPERTS, 2 * _MOE_DIM))
        .astype(np.float32)
    )
    gate, up = gate_up_out[:, :_MOE_DIM], gate_up_out[:, _MOE_DIM:]
    swiglu = gate / (1 + np.exp(-gate)) * up
    codes = np.random.default_rng(2).integers(
        0, 256, size=(_EXPERTS, _HIDDEN, _MOE_DIM // 2), dtype=np.uint8
    )

    # modelopt's static activation scale: both routes divide by it before
    # packing, so it has to leave the activation inside NVFP4's range
    # (``_NVFP4_MAX`` FP4 steps times the E4M3 block scale) as a real
    # checkpoint's would. A scale that saturates would compare the two
    # kernels' clamping, not their quantization.
    input_scale = float(np.abs(_round_to_bf16(swiglu)).max()) / (
        _NVFP4_MAX * _E4M3_MAX
    )

    fused_out, plain_out = _down_projections(gate_up_out, codes, input_scale)
    # Reading one output runs the whole graph, inside the trace.
    with driver.launch_trace() as entries:
        from_fused: np.ndarray = np.from_dlpack(fused_out)
    from_bf16: np.ndarray = np.from_dlpack(plain_out)
    kernels = [
        e.name
        for e in entries
        if e.kind == LaunchTraceEntry.OperationKind.KERNEL_LAUNCH
    ]
    fused_silus = [k for k in kernels if k.startswith("ep_fused_silu_nvfp4")]
    plain_silus = [k for k in kernels if k.startswith("ep_fused_silu_bfloat16")]
    # The SM100 grouped matmul is named for its MMA kind, ``gmm``.
    matmuls = [k for k in kernels if "_gmm_" in k]
    # Only the bf16 route's activation is quantized by its down matmul.
    quantizes = [k for k in kernels if "grouped_quantize" in k]
    assert len(fused_silus) == 1, kernels
    assert len(plain_silus) == 1, kernels
    assert len(matmuls) == 2, kernels
    assert len(quantizes) == 1, kernels

    weights = _FP4_VALUES[
        np.stack([codes & 0xF, codes >> 4], axis=-1).reshape(
            _EXPERTS, _HIDDEN, _MOE_DIM
        )
    ]
    exact = np.einsum("ek,enk->en", swiglu.astype(np.float64), weights)
    scale = np.abs(from_fused).max()
    assert scale > 0, "degenerate inputs: both routes produced zeros"
    gap = np.linalg.norm(from_fused - from_bf16)
    error = np.linalg.norm(from_fused - exact)
    print(
        "\nmax |fused - bf16| = "
        f"{np.abs(from_fused - from_bf16).max():.6g}, "
        f"max |fused| = {scale:.6g}; "
        f"|fused - bf16| / |fused - exact| = {gap / error:.3g}"
    )
    # The routes quantize the same values with the same scale. They differ
    # only where the bf16 route's rounding moves a value across an FP4
    # rounding boundary, which a few of the activations do, and that moves
    # the output far less than FP4 quantization itself does. A route that
    # quantized differently would add an error as large as the first.
    assert gap <= 0.5 * error
