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
"""MXFP4 weight quantization and matmul numerics, on AMD CDNA.

MXFP4 has no NVIDIA matmul -- the SM100 block-scaled kernel takes ``uint8``
operands only with ``float8_e4m3fn`` scales -- so this is the only place its
numerics run. Presubmit has no AMD job, so this lands in postsubmit; run it
by hand with ``--config=remote-mi355`` before relying on it.

The reference dequantizes the weight this stack actually produced, rather
than assuming how the kernel picks block scales. That makes the comparison
sensitive to the packing and scale layout -- nibble order, scale alignment,
a transposed operand -- which is where a porting bug would sit, while
leaving only the activation quantization as slack in the bound.
"""

from __future__ import annotations

import pytest
import torch
from max.driver import Accelerator, accelerator_count
from max.dtype import DType
from max.experimental.nn.common_layers import quant_ops
from max.experimental.nn.common_layers.quant_tensor import MXFP4Quantizer
from max.experimental.tensor import Tensor

_M, _N, _K = 32, 64, 128
_MX_BLOCK = 32

#: ``e2m1`` code -> value. Low three bits are magnitude, bit 3 is the sign.
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_LUT = torch.tensor(_E2M1 + [-v for v in _E2M1], dtype=torch.float32)

#: Largest ``e2m1`` magnitude; one block's relative resolution is half the
#: gap below it, so a value rounds by at most ``1/6`` of the block's own max.
_E2M1_MAX = 6.0


def _dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Reconstructs ``[N, K]`` floats from packed codes and block scales.

    Args:
        packed: ``uint8`` codes, ``[N, K // 2]``, two ``e2m1`` values per byte.
        scales: Block scales as float32, ``[N, K // 32]``. ``float8_e8m0fnu``
            is exponent-only, so its *value* is the scale -- there is no
            bias to decode, and casting it to an integer type would discard
            the encoding rather than reveal it.

    Returns:
        The dequantized weight as float32.
    """
    low = _E2M1_LUT[(packed & 0x0F).long()]
    high = _E2M1_LUT[(packed >> 4).long()]
    # Interleave the nibble pair back into K order: byte i holds elements
    # 2i (low) and 2i+1 (high).
    values = torch.stack([low, high], dim=-1).reshape(packed.shape[0], -1)
    return values * scales.repeat_interleave(_MX_BLOCK, dim=-1)


@pytest.mark.skipif(accelerator_count() == 0, reason="requires a GPU")
def test_mxfp4_matmul_matches_a_dequantized_reference() -> None:
    """``matmul`` on an MXFP4 weight agrees with dequantize-then-matmul."""
    device = Accelerator()
    generator = torch.Generator().manual_seed(17)
    x_host = torch.randn(_M, _K, generator=generator).to(torch.bfloat16)
    w_host = torch.randn(_N, _K, generator=generator).to(torch.bfloat16)

    weight = MXFP4Quantizer(Tensor.from_dlpack(w_host).to(device))
    x = Tensor.from_dlpack(x_host).to(device)

    out = quant_ops.matmul(x, weight)
    # Reading a result realizes the pending eager graph.
    got = torch.from_dlpack(out.cast(DType.float32)).cpu()

    # Eager first use: the quantizer runs its functional ops outside any
    # graph, with no `ensure_context` needed.
    quantized = weight.quant_tensor
    packed = torch.from_dlpack(quantized.weight).cpu()
    scales = torch.from_dlpack(quantized.weight_scale.cast(DType.float32)).cpu()
    w_dequantized = _dequantize(packed, scales)
    assert w_dequantized.shape == (_N, _K)

    expected = x_host.float() @ w_dequantized.T

    # Only the activations are still approximated. Each contributes at most
    # `1/6` of its block's max, and the errors are independent, so bound the
    # sum rather than each term.
    x_blocks = x_host.float().view(_M, _K // _MX_BLOCK, _MX_BLOCK)
    x_slack = (x_blocks.abs().amax(-1) / _E2M1_MAX).sum(-1, keepdim=True)
    bound = x_slack * w_dequantized.abs().amax(-1).unsqueeze(0) * _MX_BLOCK

    assert ((got - expected).abs() <= bound + 1e-2).all(), (
        (got - expected).abs().max().item(),
        bound.max().item(),
    )


@pytest.mark.skipif(accelerator_count() == 0, reason="requires a GPU")
def test_mxfp4_quantized_weight_has_the_stored_layout() -> None:
    """The packed weight halves K and carries one E8M0 scale per 32 inputs."""
    device = Accelerator()
    generator = torch.Generator().manual_seed(23)
    w_host = torch.randn(_N, _K, generator=generator).to(torch.bfloat16)

    weight = MXFP4Quantizer(Tensor.from_dlpack(w_host).to(device))
    quantized = weight.quant_tensor

    assert quantized.weight.dtype == DType.uint8
    assert list(quantized.weight.shape) == [_N, _K // 2]
    assert quantized.weight_scale.dtype == DType.float8_e8m0fnu
    assert list(quantized.weight_scale.shape) == [_N, _K // _MX_BLOCK]
