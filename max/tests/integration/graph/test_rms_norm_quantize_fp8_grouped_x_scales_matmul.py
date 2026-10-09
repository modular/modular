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
"""The ``x_scales=`` pre-quantized-activation path of ``quantized_matmul``.

``rms_norm_quantize_dynamic_scaled_float8`` (see
``test_rms_norm_quantize_fp8_grouped.py``) produces an activation that is
already FP8 plus its blockwise scales, meant to feed a GEMM directly without
re-quantizing. That consumer side -- ``quantized_matmul(..., x_scales=...)``
-- has no caller in this stack yet (the real callers, DeepSeek-V3.2's
``sparse_mla.py`` and ``indexer.py``, land two PRs later), so this test
drives it directly: the same ``quantized_matmul`` call is fed once with the
fused kernel's ``(activation, scales)`` and once with the separate
``ops.rms_norm`` + ``quantize_dynamic_scaled_float8`` composite's, and the
two GEMM outputs must match. A regression in the ``x_scales`` wiring (a
dropped kwarg, a transposed scale axis, an aliased tensor) either fails loudly
at graph build -- ``_matmul_float8`` asserts ``x.dtype == weight.dtype`` and
``x_scales.dtype == weight_scale.dtype``, and ``dynamic_scaled_matmul``
asserts the scale shapes agree on the K dimension -- or drives the output away
from the float32 reference, which the cosine check below catches.
"""

import numpy as np
import pytest
import torch
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import (
    quantize_dynamic_scaled_float8,
    rms_norm_quantize_dynamic_scaled_float8,
)
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.quant_ops import quantized_matmul

_GROUP = 128
_EPS = 1e-6
_HIDDEN = 2048
_N_OUT = 256
_INPUT_SPEC = InputScaleSpec(
    granularity=ScaleGranularity.BLOCK,
    origin=ScaleOrigin.DYNAMIC,
    dtype=DType.float32,
    block_size=(1, _GROUP),
)
_WEIGHT_SPEC = WeightScaleSpec(
    granularity=ScaleGranularity.BLOCK,
    dtype=DType.float32,
    block_size=(_GROUP, _GROUP),
)
_QUANT_CONFIG = QuantConfig(
    input_scale=_INPUT_SPEC,
    weight_scale=_WEIGHT_SPEC,
    mlp_quantized_layers=set(),
    attn_quantized_layers=set(),
    format=QuantFormat.BLOCKSCALED_FP8,
)


def _quantize_weight_blockwise(
    weight_f32: torch.Tensor, block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blockwise (``block`` x ``block``) FP8 quantization of a ``[N, K]``
    weight: one scale per tile, ``dequant = fp8.float() * scale``."""
    n, k = weight_f32.shape
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    tiled = weight_f32.reshape(n // block, block, k // block, block)
    block_max = tiled.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = (block_max / fp8_max).float()
    q = (tiled / scale[:, None, :, None]).clamp(-fp8_max, fp8_max)
    return q.to(torch.float8_e4m3fn).reshape(n, k), scale


def _fp8_buffer(tensor: torch.Tensor) -> Buffer:
    """Wraps an FP8 tensor for dlpack; float8 tensors cannot dlpack directly
    (same trick as ``test_ep_moe_fp8.py`` / ``test_mlp_float8_gpu.py``)."""
    return Buffer.from_dlpack(tensor.contiguous().view(torch.uint8)).view(
        DType.float8_e4m3fn
    )


@pytest.mark.parametrize("rows", [1, 6])
@pytest.mark.parametrize("multiply_before_cast", [False, True])
def test_x_scales_matmul_matches_unfused_quantize_then_matmul(
    session: InferenceSession,
    rows: int,
    multiply_before_cast: bool,
) -> None:
    generator = torch.Generator().manual_seed(0)
    # Per-row gain and a per-group outlier so every group scale is distinct,
    # matching test_rms_norm_quantize_fp8_grouped.py's fixture.
    gain = torch.arange(1, rows + 1).reshape(rows, 1).float()
    x_torch = torch.randn(rows, _HIDDEN, generator=generator) * gain
    x_torch[:, ::_GROUP] *= 9.0
    x_torch = x_torch.bfloat16()
    w_torch = (torch.randn(_HIDDEN, generator=generator) * 0.5 + 1.0).bfloat16()

    weight_f32 = torch.randn(_N_OUT, _HIDDEN, generator=generator) * 0.5
    weight_fp8, weight_scale = _quantize_weight_blockwise(weight_f32, _GROUP)

    x_type = TensorType(DType.bfloat16, [rows, _HIDDEN], device=DeviceRef.GPU())
    w_type = TensorType(DType.bfloat16, [_HIDDEN], device=DeviceRef.GPU())
    weight_type = TensorType(
        DType.float8_e4m3fn, [_N_OUT, _HIDDEN], device=DeviceRef.GPU()
    )
    weight_scale_type = TensorType(
        DType.float32,
        [_N_OUT // _GROUP, _HIDDEN // _GROUP],
        device=DeviceRef.GPU(),
    )
    with Graph(
        "rms_norm_quantize_x_scales_matmul",
        input_types=[x_type, w_type, weight_type, weight_scale_type],
    ) as g:
        x, w, weight, weight_scale_v = (v.tensor for v in g.inputs)

        q_fus, s_fus = rms_norm_quantize_dynamic_scaled_float8(
            x,
            w,
            _EPS,
            _INPUT_SPEC,
            _WEIGHT_SPEC,
            multiply_before_cast=multiply_before_cast,
            out_type=DType.float8_e4m3fn,
            scales_type=DType.float32,
        )
        out_fus = quantized_matmul(
            q_fus, weight, weight_scale_v, None, _QUANT_CONFIG, x_scales=s_fus
        )

        normed = ops.rms_norm(
            x, w, _EPS, multiply_before_cast=multiply_before_cast
        )
        q_sep, s_sep = quantize_dynamic_scaled_float8(
            normed,
            _INPUT_SPEC,
            _WEIGHT_SPEC,
            out_type=DType.float8_e4m3fn,
            scales_type=DType.float32,
        )
        out_sep = quantized_matmul(
            q_sep, weight, weight_scale_v, None, _QUANT_CONFIG, x_scales=s_sep
        )

        g.output(
            ops.cast(out_fus, DType.float32),
            ops.cast(out_sep, DType.float32),
        )

    model = session.load(g)
    device = model.input_devices[0]
    outputs = model(
        Buffer.from_dlpack(x_torch).to(device),
        Buffer.from_dlpack(w_torch).to(device),
        _fp8_buffer(weight_fp8).to(device),
        Buffer.from_dlpack(weight_scale).to(device),
    )
    out_fus_np, out_sep_np = (
        o.to_numpy() for o in outputs if isinstance(o, Buffer)
    )

    # Same matmul call, same weight; the only difference is whether (q, s)
    # came from the fused kernel or the separate composite. FP8 mismatches
    # between the two (~0.1% of elements, one ULP, per the sibling test)
    # average out over the K=2048 reduction.
    np.testing.assert_allclose(out_fus_np, out_sep_np, rtol=2e-2, atol=2e-2)

    normed_ref = torch.nn.functional.rms_norm(
        x_torch.float(), (_HIDDEN,), w_torch.float(), _EPS
    )
    weight_deq = weight_fp8.float() * weight_scale.repeat_interleave(
        _GROUP, dim=0
    ).repeat_interleave(_GROUP, dim=1)
    ref = (normed_ref @ weight_deq.T).numpy()

    for out_np in (out_fus_np, out_sep_np):
        cos = (out_np * ref).sum() / np.sqrt(
            (out_np * out_np).sum() * (ref * ref).sum()
        )
        assert cos >= 0.99, f"cosine {cos} vs the f32 matmul reference"
