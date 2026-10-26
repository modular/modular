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
"""MXFP8 weight quantization and matmul numerics, on NVIDIA SM100.

The CPU tests only trace these paths. Here the quantizer and matmul run, so a
mistake in the scale layout shows up as wrong numbers rather than passing as
a well-formed graph. On SM100 the quantize kernel returns its scales in the
padded rank-5 SF-atom layout, which the quantizer flattens back to the
checkpoint's rank-2 ``[N, K / 32]`` and the matmul re-interleaves.
"""

from __future__ import annotations

import pytest
import torch
from max.driver import Accelerator, accelerator_api, accelerator_count
from max.dtype import DType
from max.experimental.nn.common_layers import quant_ops
from max.experimental.nn.common_layers.quant_tensor import (
    MXFP8Quantizer,
    MXFP8Tensor,
)
from max.experimental.tensor import Tensor
from test_common.graph_utils import is_b100_b200

_MX_BLOCK = 32

#: Half an E4M3 ulp, relative to the value: three mantissa bits.
#:
#: This bounds every element only because nothing saturates. The kernel's
#: ``compute_mxfp8_block_scale`` rounds ``amax / 448`` *up* to a power of two
#: (``cvt.rp.ue8m0x2.f32`` on SM100), so every scaled value fits E4M3's 448.
#: The OCP reference scale rounds down and clamps the top of the block; if the
#: kernel ever switches to it, this bound fails and needs a clamp term.
_E4M3_REL = 2.0**-4

#: E4M3's smallest subnormal step, in units of the block scale.
_E4M3_SUBNORMAL_STEP = 2.0**-9


def _skip_if_not_supported() -> None:
    if accelerator_count() == 0:
        pytest.skip("requires a GPU")
    if accelerator_api() == "hip":
        pytest.skip("the MXFP8 block-scaled MMA is NVIDIA-only")
    if not is_b100_b200():
        pytest.skip("the MXFP8 block-scaled MMA requires SM100")


def _scaled_weight(n: int, k: int, seed: int) -> torch.Tensor:
    """A bf16 ``[n, k]`` weight whose 32-wide blocks differ by powers of two.

    Gaussian values alone give most blocks the same E8M0 scale, so a scale
    landing on the wrong row or block would often still decode correctly.
    Giving every block its own exponent makes any misplaced scale off by at
    least a factor of two.
    """
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(n, k, generator=generator)
    exponents = torch.randint(
        -4, 5, (n, k // _MX_BLOCK), generator=generator
    ).float()
    return (values * (2.0**exponents).repeat_interleave(_MX_BLOCK, dim=-1)).to(
        torch.bfloat16
    )


def _dequantize(weight: MXFP8Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the float32 weight a container encodes, and its block scales.

    Both leaves are cast on device: ``float8_e8m0fnu`` is exponent-only, so
    its float value *is* the scale, and casting to an integer type would
    discard the encoding rather than reveal it.
    """
    values = torch.from_dlpack(weight.weight.cast(DType.float32)).cpu()
    scales = torch.from_dlpack(weight.weight_scale.cast(DType.float32)).cpu()
    return values * scales.repeat_interleave(_MX_BLOCK, dim=-1), scales


def test_mxfp8_quantizer_round_trips_the_weight() -> None:
    """Dequantizing the stored weight recovers the bf16 input.

    ``N = 192`` is not a multiple of 128 and ``K / 32 = 5`` is not a multiple
    of 4, so the kernel's rank-5 scales are padded on both axes and the
    quantizer has to strip the padding, not just reorder.
    """
    _skip_if_not_supported()
    n, k = 192, 160
    w_host = _scaled_weight(n, k, seed=31)

    quantizer = MXFP8Quantizer(Tensor.from_dlpack(w_host).to(Accelerator()))
    # Eager first use: the quantizer's functional ops run outside any graph.
    quantized = quantizer.quant_tensor

    assert list(quantized.weight.shape) == [n, k]
    assert list(quantized.weight_scale.shape) == [n, k // _MX_BLOCK]

    got, scales = _dequantize(quantized)
    expected = w_host.float()
    per_element_scale = scales.repeat_interleave(_MX_BLOCK, dim=-1)
    # Round-to-nearest within E4M3's range, plus the subnormal step at the
    # bottom of it.
    bound = (
        _E4M3_REL * expected.abs() + _E4M3_SUBNORMAL_STEP * per_element_scale
    )
    assert ((got - expected).abs() <= bound).all(), (
        (got - expected).abs().max().item(),
        bound.max().item(),
    )


def test_mxfp8_matmul_matches_a_dequantized_reference() -> None:
    """``matmul`` on an MXFP8 weight agrees with dequantize-then-matmul.

    The reference uses the weight this stack actually produced, so the only
    slack is the activation quantization and the bf16 output. ``K`` is a
    multiple of 128 as the SM100 kernel requires; ``N`` is not, so the weight
    scales still go through the padded interleave.
    """
    _skip_if_not_supported()
    m, n, k = 8, 192, 384
    device = Accelerator()
    generator = torch.Generator().manual_seed(17)
    x_host = torch.randn(m, k, generator=generator).to(torch.bfloat16)
    w_host = _scaled_weight(n, k, seed=19)

    weight = MXFP8Quantizer(Tensor.from_dlpack(w_host).to(device))
    x = Tensor.from_dlpack(x_host).to(device)

    out = quant_ops.matmul(x, weight)
    got = torch.from_dlpack(out.cast(DType.float32)).cpu()

    w_dequantized, _ = _dequantize(weight.quant_tensor)
    expected = x_host.float() @ w_dequantized.T

    # Each activation is off by at most half an ulp, as for the weight, plus
    # the subnormal step. The scale is below ``2 * amax / 448``, so that step
    # is under ``amax * 2**-16``.
    x_float = x_host.float()
    x_blocks = x_float.abs().view(m, k // _MX_BLOCK, _MX_BLOCK)
    x_floor = (x_blocks.amax(-1) * 2.0**-16).repeat_interleave(
        _MX_BLOCK, dim=-1
    )
    x_err = _E4M3_REL * x_float.abs() + x_floor
    bound = (
        x_err @ w_dequantized.abs().T
        + 2.0**-8 * expected.abs()  # bf16 output rounding
        + 1e-3
    )

    assert ((got - expected).abs() <= bound).all(), (
        (got - expected).abs().max().item(),
        bound.max().item(),
    )
