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
"""Which SwiGLU ``quant_ops.grouped_matmul_silu`` picks for an NVFP4 down
projection, outside the EP dispatch, and what it costs.

A bf16 SwiGLU output does not need the fused quantized SiLU — the down matmul
quantizes bf16 itself, with the same static per-expert scale — so the two
routes have to agree, and the bf16 one has to be taken: the fused kernel gives
each expert group a share of the GPU's SMs, so it only builds for a device
holding fewer experts than the GPU has SMs. A float32 SwiGLU output does need
it, because the down matmul quantizes from bf16 only.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator
from max.dtype import DType
from max.experimental.nn.common_layers import quant_ops
from max.experimental.nn.common_layers.quant_tensor import (
    NVFP4Activation,
    NVFP4Tensor,
    QuantAwareTensor,
)
from max.experimental.tensor import Tensor
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

_NVFP4_BLOCK = (1, 16)
_HIDDEN = 256
_MOE_DIM = 128

# Past any current GPU's SM count, which is what the fused quantized SiLU
# divides among the expert groups: 148 on a B200, 160 on a B300.
_MANY_EXPERTS = 256
_FEW_EXPERTS = 8

# NVFP4's largest FP4 magnitude and the largest E4M3 block scale.
_NVFP4_MAX = 6.0
_E4M3_MAX = 448.0


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


def _stacked_weight(
    experts: int,
    rows: int,
    cols: int,
    seed: int | None = None,
    input_scale: float = 1.0,
) -> NVFP4Tensor:
    """Expert-stacked NVFP4 weights, as ``QuantizedMoE`` stacks them.

    ``seed`` fills the packed FP4 codes pseudo-randomly; without it the
    weights are zero, which is enough for the tests that only compile.
    ``input_scale`` is the checkpoint's static activation scale, which both
    quantization routes divide the activation by before packing it.
    """
    device = Accelerator()
    shape = (experts, rows, cols // 2)
    if seed is None:
        data = Tensor.zeros(shape, dtype=DType.uint8, device=device)
    else:
        codes = np.random.default_rng(seed).integers(
            0, 256, size=shape, dtype=np.uint8
        )
        data = Tensor.from_dlpack(codes).to(device)
    return NVFP4Tensor(
        weight=data,
        # A unit block scale keeps the weight values the FP4 codes themselves,
        # so the comparison is not dominated by the block-scale rounding.
        weight_scale=Tensor.ones(
            (experts, rows, cols // _NVFP4_BLOCK[1]),
            dtype=DType.float32,
            device=device,
        ).cast(DType.float8_e4m3fn),
        weight_scale_2=Tensor.ones(
            (experts,), dtype=DType.float32, device=CPU()
        ),
        input_scale=input_scale
        * Tensor.ones((experts,), dtype=DType.float32, device=CPU()),
        block_size=_NVFP4_BLOCK,
    )


def _indices(experts: int) -> tuple[Tensor, Tensor, Tensor]:
    """One token per expert: group offsets, expert ids and scale offsets.

    ``moe_create_indices`` gives every group its own scale tile; with one
    token each they all start inside tile 0, so the offsets have to separate
    them or the groups overwrite each other's scales.
    """
    device = Accelerator()
    return (
        Tensor.from_dlpack(np.arange(experts + 1, dtype=np.uint32)).to(device),
        Tensor.from_dlpack(np.arange(experts, dtype=np.int32)).to(device),
        Tensor.from_dlpack(np.arange(experts, dtype=np.uint32)).to(device),
    )


def _grouped_matmul_silu(
    experts: int, accum_dtype: DType | None
) -> QuantAwareTensor:
    """Routes one zero token per expert through the gate/up matmul and SwiGLU."""
    device = Accelerator()
    starts, ids, offsets = _indices(experts)
    return quant_ops.grouped_matmul_silu(
        Tensor.zeros((experts, _HIDDEN), dtype=DType.bfloat16, device=device),
        _stacked_weight(experts, 2 * _MOE_DIM, _HIDDEN),
        _stacked_weight(experts, _HIDDEN, _MOE_DIM),
        starts,
        ids,
        None,
        _nvfp4_quant_config(),
        scales_offset=offsets,
        accum_dtype=accum_dtype,
    )


def _realize(tensor: Tensor) -> np.ndarray:
    """Reading a result compiles and runs the pending eager graph."""
    return np.from_dlpack(tensor.cast(DType.float32).to(CPU()))


@pytest.mark.parametrize("experts", [_FEW_EXPERTS, _MANY_EXPERTS])
def test_bf16_swiglu_leaves_the_down_matmul_to_quantize(experts: int) -> None:
    """At any expert count the bf16 SwiGLU stays out of the fused kernel.

    With ``_MANY_EXPERTS`` the fused quantized SiLU cannot be built at all,
    so emitting it fails the graph compiler's
    ``num_sms must be >= number of expert groups`` constraint.
    """
    out = _grouped_matmul_silu(experts, accum_dtype=None)
    assert isinstance(out, Tensor)
    assert out.dtype == DType.bfloat16
    _realize(out)


def test_fused_and_bf16_swiglu_agree_through_the_down_matmul() -> None:
    """Both SwiGLU routes land on the same down-projection output.

    They quantize the same activations with the same static per-expert scale;
    the bf16 route just rounds to bfloat16 on the way, which is finer than
    the FP4 step it then quantizes to. The models that move from the fused
    route to the bf16 one (DeepSeek-V3 and Kimi-K2.5 on NVFP4 off the EP
    dispatch) therefore keep their numerics.
    """
    device = Accelerator()
    experts = _FEW_EXPERTS
    config = _nvfp4_quant_config()
    starts, ids, offsets = _indices(experts)
    gate_up = _stacked_weight(experts, 2 * _MOE_DIM, _HIDDEN, seed=1)
    tokens = (
        Tensor.from_dlpack(
            np.random.default_rng(3)
            .standard_normal((experts, _HIDDEN))
            .astype(np.float32)
        )
        .to(device)
        .cast(DType.bfloat16)
    )

    gate_up_out = quant_ops.grouped_matmul(
        tokens, gate_up, starts, ids, None, scales_offset=offsets
    )
    plain = quant_ops.grouped_silu(gate_up_out, starts)
    assert isinstance(plain, Tensor)
    swiglu = _realize(plain)

    # modelopt's static activation scale: both routes divide by it before
    # packing, so it has to leave the activation inside NVFP4's range
    # (``_NVFP4_MAX`` FP4 steps times the E4M3 block scale) as a real
    # checkpoint's would. A scale that saturates would compare the two
    # kernels' clamping, not their quantization.
    down = _stacked_weight(
        experts,
        _HIDDEN,
        _MOE_DIM,
        seed=2,
        input_scale=float(np.abs(swiglu).max()) / (_NVFP4_MAX * _E4M3_MAX),
    )
    fused = quant_ops.grouped_silu(
        gate_up_out, starts, down, config, scales_offset=offsets
    )
    assert isinstance(fused, NVFP4Activation)

    from_fused = _realize(
        quant_ops.grouped_matmul(
            fused, down, starts, ids, None, scales_offset=offsets
        )
    )
    from_bf16 = _realize(
        quant_ops.grouped_matmul(
            plain, down, starts, ids, None, scales_offset=offsets
        )
    )

    scale = np.abs(from_fused).max()
    assert scale > 0, "degenerate inputs: both routes produced zeros"
    max_abs = np.abs(from_fused - from_bf16).max()
    print(
        f"\nSwiGLU max {np.abs(swiglu).max():.6g}; down-projection output: "
        f"max |fused - bf16| = {max_abs:.6g}, max |fused| = {scale:.6g}, "
        f"relative = {max_abs / scale:.3g}"
    )
    # Both routes quantize the same values with the same scale, so the only
    # difference is the bf16 route's rounding on the way in, far below the
    # FP4 step it then quantizes to. A route that quantized differently
    # would land orders of magnitude above this.
    assert max_abs <= 0.01 * scale
