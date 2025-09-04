# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Minimal ULP test for fused multiply-add semantics.

This test checks that computing y = a * b + c in bf16 matches a PyTorch
reference within <= 1 ULP.
"""

from __future__ import annotations

import pytest
import torch
from max.driver import accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from torch.utils.dlpack import from_dlpack


def _bf16_ordered_uint(x: torch.Tensor) -> torch.Tensor:
    """Map bf16 values to a monotone-ordered unsigned 16-bit integer space."""
    xb = x.to(torch.bfloat16).cpu()
    bits = xb.view(torch.int16)
    is_negative = bits < 0
    bits_u = bits.to(torch.int32) & 0xFFFF
    ordered = torch.where(is_negative, 0xFFFF - bits_u, bits_u + 0x8000)
    return (ordered & 0xFFFF).to(torch.int32)


def _ulp_distance_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Count representable bf16 values between a and b (elementwise).

    References:
    https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    https://marc-b-reynolds.github.io/math/2019/05/14/FloatCmp.html
    """
    oa = _bf16_ordered_uint(a)
    ob = _bf16_ordered_uint(b)
    return (oa - ob).abs()


def test_multiply_add_correctness(session: InferenceSession) -> None:
    """Tests that multiply+add is within 1 ULP of PyTorch reference."""
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    torch.manual_seed(42)

    # Shape chosen to mirror prior tests while staying light-weight.
    shape = (1, 257, 1024)

    itype = TensorType(
        dtype=DType.bfloat16, shape=list(shape), device=DeviceRef.GPU(0)
    )
    with Graph("fma", input_types=(itype, itype, itype)) as graph:
        a, b, c = (inp.tensor for inp in graph.inputs)
        y = ops.add(ops.mul(a, b), c)
        graph.output(y)

    compiled = session.load(graph)

    a = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    c = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    # Execute MAX graph and convert to torch.
    out = compiled.execute(a, b, c)
    y_max = from_dlpack(out[0]).float().cpu()

    # PyTorch bf16 reference.
    y_ref = (a * b + c).float().cpu()

    ulps = _ulp_distance_bf16(y_max, y_ref)
    max_ulps = int(ulps.max().item())

    print(f"  Max ULP distance: {max_ulps}")

    if max_ulps > 1:
        idx = int(ulps.argmax().item())
        coords = tuple(
            i.item() for i in torch.unravel_index(torch.tensor(idx), ulps.shape)
        )
        y_max_v = y_max.view(-1)[idx].item()
        y_ref_v = y_ref.view(-1)[idx].item()
        print("  Worst element:")
        print(f"    position: {coords}")
        print(f"    MAX:  {y_max_v:+.8e}")
        print(f"    Torch:  {y_ref_v:+.8e}")
        print(f"    ULPs: {max_ulps}")

    assert max_ulps <= 1, f"expected <= 1 ULP, got {max_ulps}"
