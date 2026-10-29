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

"""The ModuleV3 GLM-5.3-Flash mHC write-back, against float64.

The real model's dimensions, ``hidden_size`` 4096 and ``hc_mult`` 4, run on
``mhc_post`` on AMD. A ``hidden_size`` of 1536 is not a whole number of the
kernel's bfloat16 blocks, so it covers the batched-matmul fallback there.
"""

from __future__ import annotations

import pytest
import torch
from max.driver import CPU, Accelerator, Device, accelerator_api
from max.dtype import DType
from max.pipelines.architectures.glm5_next_modulev3.layers.decoder import (
    write_back,
)
from max.tensor import Tensor, compilation
from max.tensor.compilation import CompiledCallable
from max.tensor.sharding import TensorLayout

HIDDEN_SIZE = 4096
HC_MULT = 4

BF16_ULP = 2.0**-8
"""One bfloat16 ulp relative to 1.0."""

_WriteBack = CompiledCallable[[Tensor, Tensor, Tensor, Tensor], Tensor]


FALLBACK_HIDDEN_SIZE = 1536
"""Not a whole number of ``mhc_post``'s narrowest bfloat16 block, 1024 wide."""


def _layouts(
    device: Device, hc_mult: int, hidden_size: int
) -> tuple[TensorLayout, ...]:
    return (
        TensorLayout(DType.bfloat16, ["rows", hc_mult, hidden_size], device),
        TensorLayout(DType.bfloat16, ["rows", hidden_size], device),
        TensorLayout(DType.float32, ["rows", hc_mult], device),
        TensorLayout(DType.float32, ["rows", hc_mult, hc_mult], device),
    )


@pytest.fixture(scope="module")
def compiled() -> dict[int, tuple[Device, _WriteBack]]:
    """One compile per hidden size; the row count stays symbolic."""
    device = Accelerator()
    return {
        hidden_size: (
            device,
            compilation.compile(write_back)(
                *_layouts(device, HC_MULT, hidden_size)
            ),
        )
        for hidden_size in (HIDDEN_SIZE, FALLBACK_HIDDEN_SIZE)
    }


@pytest.mark.parametrize(
    ("hc_mult", "hidden_size", "supported"),
    [
        (HC_MULT, HIDDEN_SIZE, True),
        (HC_MULT, FALLBACK_HIDDEN_SIZE, False),
        (3, HIDDEN_SIZE, False),
    ],
)
def test_write_back_falls_back_outside_the_kernel_limits(
    hc_mult: int, hidden_size: int, supported: bool
) -> None:
    """Shapes ``mhc_post`` rejects build the batched-matmul path instead."""
    graph = str(
        compilation.stage(write_back)(
            *_layouts(Accelerator(), hc_mult, hidden_size)
        )
    )
    uses_kernel = supported and accelerator_api() == "hip"
    assert ("mo.mhc.post" in graph) == uses_kernel
    assert ("matmul" in graph) != uses_kernel


@pytest.mark.parametrize("hidden_size", [HIDDEN_SIZE, FALLBACK_HIDDEN_SIZE])
@pytest.mark.parametrize("rows", [1, 17])
def test_write_back_matches_float64(
    compiled: dict[int, tuple[Device, _WriteBack]],
    hidden_size: int,
    rows: int,
) -> None:
    """``streams' = post * y + comb^T @ streams``, landing in bfloat16.

    The gates are drawn as the mapping produces them: ``post`` in ``[0, 2]``
    and ``comb`` column-stochastic. Against the expression evaluated in float64
    and rounded once, MAX is held to two bfloat16 ulps at the peak magnitude
    and to no further from the truth than the three-rounding bfloat16
    reference.
    """
    device, run = compiled[hidden_size]
    generator = torch.Generator().manual_seed(rows)
    streams = torch.randn(
        rows, HC_MULT, hidden_size, generator=generator
    ).bfloat16()
    y = torch.randn(rows, hidden_size, generator=generator).bfloat16()
    post = 2 * torch.rand(rows, HC_MULT, generator=generator)
    comb = torch.randn(rows, HC_MULT, HC_MULT, generator=generator).softmax(
        dim=-2
    )

    out = torch.from_dlpack(
        run(
            Tensor(streams.cuda(), device=device),
            Tensor(y.cuda(), device=device),
            Tensor(post.cuda(), device=device),
            Tensor(comb.cuda(), device=device),
        ).to(CPU())
    )
    assert tuple(out.shape) == (rows, HC_MULT, hidden_size)

    exact = (
        post.double().unsqueeze(-1) * y.double().unsqueeze(-2)
        + torch.matmul(comb.double().transpose(-1, -2), streams.double())
    ).bfloat16()
    reference = post.bfloat16().unsqueeze(-1) * y.unsqueeze(-2) + torch.matmul(
        comb.bfloat16().transpose(-1, -2), streams
    )

    atol = 2 * BF16_ULP * exact.abs().max().item()
    torch.testing.assert_close(out.float(), exact.float(), rtol=0.0, atol=atol)
    max_error = (out.float() - exact.float()).abs().max().item()
    reference_error = (reference.float() - exact.float()).abs().max().item()
    assert max_error <= 1.5 * reference_error, (
        f"MAX write-back is {max_error:.3e} from the float64 result while the "
        f"reference is {reference_error:.3e}"
    )
