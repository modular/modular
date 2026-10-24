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
"""End-to-end tests for `ops.matmul` with an `out_dtype` wider than its inputs.

A router GEMM multiplies bfloat16 activations by bfloat16 weights and keeps
the float32 accumulator. The kernel writes float32 directly, so the result
matches a float32 product of the bfloat16-rounded inputs.
"""

from __future__ import annotations

import max.driver as md
import numpy as np
import numpy.typing as npt
import pytest
from max.driver import accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

_Array = npt.NDArray[np.float32]


def _round_to_bf16(x: _Array) -> _Array:
    """Rounds float32 values to the nearest bfloat16, ties to even."""
    bits = x.view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
@pytest.mark.parametrize(
    "lhs_shape, rhs_shape",
    [
        ((32, 256), (256, 128)),
        ((4, 32, 256), (4, 256, 128)),
    ],
    ids=["2d", "batched"],
)
def test_bf16_inputs_f32_output(
    lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]
) -> None:
    device = DeviceRef.GPU()
    rng = np.random.default_rng(0)
    lhs_np = rng.standard_normal(lhs_shape).astype(np.float32)
    rhs_np = rng.standard_normal(rhs_shape).astype(np.float32)

    with Graph(
        "matmul_bf16_to_f32",
        input_types=[
            TensorType(DType.float32, lhs_shape, device=device),
            TensorType(DType.float32, rhs_shape, device=device),
        ],
    ) as graph:
        lhs, rhs = (v.tensor for v in graph.inputs)
        out = ops.matmul(
            ops.cast(lhs, DType.bfloat16),
            ops.cast(rhs, DType.bfloat16),
            out_dtype=DType.float32,
        )
        assert out.dtype == DType.float32
        graph.output(out)

    accelerator = md.Accelerator()
    model = InferenceSession(devices=[accelerator]).load(graph)
    (result,) = model.execute(
        md.Buffer.from_numpy(lhs_np).to(accelerator),
        md.Buffer.from_numpy(rhs_np).to(accelerator),
    )
    assert isinstance(result, md.Buffer)

    expected = _round_to_bf16(lhs_np) @ _round_to_bf16(rhs_np)
    np.testing.assert_allclose(
        result.to(md.CPU()).to_numpy(), expected, rtol=1e-3, atol=1e-3
    )
