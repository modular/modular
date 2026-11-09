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
"""In-graph MXFP8 quantization-error measurement, on SM100.

Same checks as the NVFP4 error tests: the printed
``[sum_sq_error, sum_sq_source, max_abs_error]`` must match a host
recomputation from the quantized leaves the graph returned, and the dequant
op that measures must agree with what the block-scaled matmul computes.
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
from max.driver import Accelerator, accelerator_api, accelerator_count
from max.driver.buffer import load_max_buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.engine.api import PrintStyle
from max.nn.common_layers import quant_ops
from max.nn.common_layers.quant_tensor import MXFP8Quantizer
from max.nn.kernels import mxfp8_dequant
from max.nn.module import Module, module_dataclass
from max.tensor import Tensor
from max.tensor.tensor import TensorType
from test_common.graph_utils import is_b100_b200

_MX_BLOCK = 32
_M, _N, _K = 8, 192, 384
# The dump is named after the weight, which ``compile`` names after its path.
_LABEL = "weight"

# E4M3 relative rounding, and its smallest subnormal step in block-scale units.
_E4M3_REL = 2.0**-4
_E4M3_SUBNORMAL_STEP = 2.0**-9


def _skip_if_not_supported() -> None:
    if accelerator_count() == 0:
        pytest.skip("requires a GPU")
    if accelerator_api() == "hip":
        pytest.skip("the MXFP8 block-scaled MMA is NVIDIA-only")
    if not is_b100_b200():
        pytest.skip("the MXFP8 block-scaled MMA requires SM100")


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


def _decode(values: Tensor, scales: Tensor) -> torch.Tensor:
    """Host float32 weight from MXFP8 leaves (casts happen on device)."""
    v = torch.from_dlpack(values.cast(DType.float32)).cpu()
    s = torch.from_dlpack(scales.cast(DType.float32)).cpu()
    return v * s.repeat_interleave(_MX_BLOCK, dim=-1)


def _find_dump(directory: Path) -> Path:
    matches = sorted(directory.glob(f"*quant_error*{_LABEL}*.max"))
    assert matches, sorted(p.name for p in directory.iterdir())
    return matches[0]


@module_dataclass
class _Probe(Module[[Tensor], tuple[Tensor, ...]]):
    weight: Tensor

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        quantizer = MXFP8Quantizer(self.weight, report_error=True)
        quantized = quantizer.quant_tensor
        return quantized.weight, quantized.weight_scale


@module_dataclass
class _MatmulAndDequant(Module[[Tensor], tuple[Tensor, ...]]):
    weight: Tensor

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        quantizer = MXFP8Quantizer(self.weight)
        quantized = quantizer.quant_tensor
        restored = mxfp8_dequant(
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
    rows: int,
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
    _skip_if_not_supported()
    device = Accelerator()
    w_host = _scaled_weight(_N, _K, seed=31)

    session = InferenceSession(devices=[device])
    session.set_debug_print_options(
        style=PrintStyle.BINARY_MAX_CHECKPOINT, output_directory=tmp_path
    )
    compiled = _compile(_Probe, w_host, device, session, rows=1)
    values, scales = compiled(
        Tensor.zeros([1, _K], dtype=DType.bfloat16, device=device)
    )

    restored = _decode(values, scales).double()
    source = w_host.float().double()
    error = restored - source
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
            f"\nMXFP8 in-graph error ({_N}x{_K}): "
            f"sum_sq_error={got[0]:.6e} sum_sq_source={got[1]:.6e} "
            f"rel_error={rel_error:.4e} "
            f"snr_db={-20 * np.log10(rel_error):.2f} "
            f"max_abs={got[2]:.4e} (host recompute: "
            f"sum_sq_error={expected[0]:.6e})"
        )
    assert 0 < got[0] / got[1] < 0.05


def test_dequant_op_matches_matmul_decode() -> None:
    """``x @ dequant(w).T`` agrees with the block-scaled matmul.

    The measurement decodes with ``mxfp8_dequant``; serving decodes inside
    the matmul. Activations are still quantized dynamically, so the slack is
    their rounding propagated through the weight, plus bf16 output rounding.
    """
    _skip_if_not_supported()
    device = Accelerator()
    generator = torch.Generator().manual_seed(17)
    x_host = torch.randn(_M, _K, generator=generator).to(torch.bfloat16)
    w_host = _scaled_weight(_N, _K, seed=19)

    compiled = _compile(
        _MatmulAndDequant,
        w_host,
        device,
        InferenceSession(devices=[device]),
        rows=_M,
    )
    out, restored, values, scales = compiled(
        Tensor.from_dlpack(x_host).to(device)
    )

    restored_host = torch.from_dlpack(restored).cpu()
    # Both decodes of the same bytes: the op and the plain device casts.
    assert torch.equal(restored_host, _decode(values, scales))

    x = x_host.float()
    w = restored_host.float()
    expected = x @ w.T
    block_amax = (
        x.view(_M, _K // _MX_BLOCK, _MX_BLOCK).abs().amax(-1, keepdim=True)
    )
    # The kernel rounds the block scale up to a power of two, at most 2x
    # amax / 448.
    subnormal = (
        (_E4M3_SUBNORMAL_STEP * 2.0 * block_amax / 448.0)
        .expand(-1, -1, _MX_BLOCK)
        .reshape(_M, _K)
    )
    dx = _E4M3_REL * x.abs() + subnormal
    bound = dx @ w.abs().T + 2.0**-7 * (x.abs() @ w.abs().T) + 1e-3

    got = torch.from_dlpack(out.cast(DType.float32)).cpu()
    assert ((got - expected).abs() <= bound).all(), (
        (got - expected).abs().max().item(),
        bound.max().item(),
    )
