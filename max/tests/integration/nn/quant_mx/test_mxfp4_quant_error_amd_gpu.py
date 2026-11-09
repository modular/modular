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
"""In-graph MXFP4 quantization-error measurement, on AMD CDNA.

MXFP4 has no NVIDIA matmul, so AMD is the one place its whole path runs:
quantize, the ``mxfp4_dequant`` op that measures, and the matmul that
serves. Two checks, as for the other formats: the printed
``[sum_sq_error, sum_sq_source, max_abs_error]`` matches a host decode of
the leaves the graph returned (and the dequant op is bit-identical to that
decode), and the matmul agrees with ``x @ dequant(w).T``.

Presubmit has no AMD job, so this lands in postsubmit; run it by hand with
``--config=remote-mi355``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from max import driver
from max import functional as F
from max.driver import Accelerator, accelerator_count
from max.driver.buffer import load_max_buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.engine.api import PrintStyle
from max.nn.common_layers import quant_ops
from max.nn.common_layers.quant_tensor import MXFP4Quantizer
from max.nn.kernels import mxfp4_dequant
from max.nn.module import Module, module_dataclass
from max.tensor import Tensor
from max.tensor.tensor import TensorType

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="requires a GPU"
)

_MX_BLOCK = 32
_N, _K = 192, 384
# The matmul shapes the AMD kernel is already exercised with.
_MM_M, _MM_N, _MM_K = 32, 64, 128
_E2M1_MAX = 6.0
# The dump is named after the weight, which ``compile`` names after its path.
_LABEL = "weight"

#: ``e2m1`` code -> value. Low three bits are magnitude, bit 3 is the sign.
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_LUT = torch.tensor(_E2M1 + [-v for v in _E2M1], dtype=torch.float32)


def _scaled_weight(n: int, k: int, seed: int) -> torch.Tensor:
    """A bf16 weight whose 32-wide blocks differ by powers of two."""
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(n, k, generator=generator)
    exponents = torch.randint(
        -4, 5, (n, k // _MX_BLOCK), generator=generator
    ).float()
    return (values * (2.0**exponents).repeat_interleave(_MX_BLOCK, dim=-1)).to(
        torch.bfloat16
    )


def _decode(packed: Tensor, scales: Tensor) -> torch.Tensor:
    """Host float32 weight from MXFP4 leaves.

    Byte i holds elements 2i (low nibble) and 2i+1 (high nibble). The E8M0
    scale's float value *is* the scale, so a device cast reads it.
    """
    codes = torch.from_dlpack(packed).cpu()
    block_scales = torch.from_dlpack(scales.cast(DType.float32)).cpu()
    low = _E2M1_LUT[(codes & 0x0F).long()]
    high = _E2M1_LUT[(codes >> 4).long()]
    values = torch.stack([low, high], dim=-1).reshape(codes.shape[0], -1)
    return values * block_scales.repeat_interleave(_MX_BLOCK, dim=-1)


def _find_dump(directory: Path) -> Path:
    matches = sorted(directory.glob(f"*quant_error*{_LABEL}*.max"))
    assert matches, sorted(p.name for p in directory.iterdir())
    return matches[0]


@module_dataclass
class _Probe(Module[[Tensor], tuple[Tensor, ...]]):
    weight: Tensor

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        quantizer = MXFP4Quantizer(self.weight, report_error=True)
        quantized = quantizer.quant_tensor
        restored = mxfp4_dequant(
            quantized.weight, quantized.weight_scale, out_type=DType.float32
        )
        return quantized.weight, quantized.weight_scale, restored


@module_dataclass
class _MatmulAndDequant(Module[[Tensor], tuple[Tensor, ...]]):
    weight: Tensor

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        quantizer = MXFP4Quantizer(self.weight)
        quantized = quantizer.quant_tensor
        restored = mxfp4_dequant(
            quantized.weight, quantized.weight_scale, out_type=DType.float32
        )
        return (
            quant_ops.matmul(x, quantizer),
            restored,
            quantized.weight,
            quantized.weight_scale,
        )


def _compile(
    model_cls: Callable[..., Module[[Tensor], tuple[Tensor, ...]]],
    w_host: torch.Tensor,
    device: driver.Device,
    session: InferenceSession,
    rows: int = 1,
) -> Any:
    with F.lazy():
        model = model_cls(
            weight=Tensor.zeros(list(w_host.shape), dtype=DType.bfloat16)
        )
    weight = Tensor.from_dlpack(w_host).to(device)
    return model.compile(
        TensorType(DType.bfloat16, [rows, w_host.shape[1]], device=device),
        weights={"weight": weight.driver_tensor},
        session=session,
    )


def test_in_graph_error_matches_host_dequantize(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    device = Accelerator()
    w_host = _scaled_weight(_N, _K, seed=41)

    session = InferenceSession(devices=[device])
    session.set_debug_print_options(
        style=PrintStyle.BINARY_MAX_CHECKPOINT, output_directory=tmp_path
    )
    compiled = _compile(_Probe, w_host, device, session)
    packed, scales, restored = compiled(
        Tensor.zeros([1, _K], dtype=DType.bfloat16, device=device)
    )

    assert list(packed.shape) == [_N, _K // 2]
    assert list(scales.shape) == [_N, _K // _MX_BLOCK]

    host = _decode(packed, scales)
    # The measuring op and the host decode read the same bytes identically.
    assert torch.equal(torch.from_dlpack(restored).cpu(), host)

    source = w_host.float().double()
    error = host.double() - source
    expected = np.array(
        [
            (error**2).sum().item(),
            (source**2).sum().item(),
            error.abs().max().item(),
        ]
    )
    got = load_max_buffer(_find_dump(tmp_path)).to_numpy().astype(np.float64)
    assert got.shape == (3,)
    np.testing.assert_allclose(got, expected, rtol=1e-4)

    # Pytest hides a passing test's output; put the numbers in the test log.
    rel_error = (got[0] / got[1]) ** 0.5
    with capsys.disabled():
        print(
            f"\nMXFP4 in-graph error ({_N}x{_K}): "
            f"sum_sq_error={got[0]:.6e} sum_sq_source={got[1]:.6e} "
            f"rel_error={rel_error:.4e} "
            f"snr_db={-20 * np.log10(rel_error):.2f} "
            f"max_abs={got[2]:.4e} (host recompute: "
            f"sum_sq_error={expected[0]:.6e})"
        )
    # 2-bit mantissa blocks: lossy, but not destroyed.
    assert 0 < got[0] / got[1] < 0.2


def test_dequant_op_matches_matmul_decode() -> None:
    """``x @ dequant(w).T`` agrees with the MXFP4 matmul on the same weight.

    The measurement decodes with ``mxfp4_dequant``; serving decodes inside
    the AMD matmul. Only the activations are still approximated, so the bound
    is the one the existing MXFP4 matmul test uses: each activation block
    contributes at most ``1/6`` of its max, summed across blocks.
    """
    device = Accelerator()
    generator = torch.Generator().manual_seed(17)
    x_host = torch.randn(_MM_M, _MM_K, generator=generator).to(torch.bfloat16)
    w_host = _scaled_weight(_MM_N, _MM_K, seed=19)

    compiled = _compile(
        _MatmulAndDequant,
        w_host,
        device,
        InferenceSession(devices=[device]),
        rows=_MM_M,
    )
    out, restored, packed, scales = compiled(
        Tensor.from_dlpack(x_host).to(device)
    )

    w_deq = _decode(packed, scales)
    assert torch.equal(torch.from_dlpack(restored).cpu(), w_deq)

    expected = x_host.float() @ w_deq.T
    x_blocks = x_host.float().view(_MM_M, _MM_K // _MX_BLOCK, _MX_BLOCK)
    x_slack = (x_blocks.abs().amax(-1) / _E2M1_MAX).sum(-1, keepdim=True)
    bound = x_slack * w_deq.abs().amax(-1).unsqueeze(0) * _MX_BLOCK

    got = torch.from_dlpack(out.cast(DType.float32)).cpu()
    assert ((got - expected).abs() <= bound + 1e-2).all(), (
        (got - expected).abs().max().item(),
        bound.max().item(),
    )
