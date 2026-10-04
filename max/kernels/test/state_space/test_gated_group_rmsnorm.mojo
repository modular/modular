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
"""CPU test of the fused gated group-RMSNorm reading its gate through a load function.

The gate is a column slice of a wider tensor, as the in-projection's gate is in
Nemotron-H, and is read in place.
"""

from std.math import rsqrt
from std.sys import size_of

from layout import Coord, TileTensor, row_major
from nn.activations import silu
from state_space.gated_group_rmsnorm import (
    gated_group_rmsnorm_cpu,
)
from std.testing import assert_almost_equal


def main() raises:
    test_cpu_contiguous_gate()
    test_cpu_strided_bf16_gate()


def run_cpu[
    y_dtype: DType, gate_dtype: DType
](n_rows: Int, num_groups: Int, group_size: Int, gate_stride: Int) raises:
    var eps = Float32(1e-5)
    var intermediate = num_groups * group_size
    var y_heap = List(length=n_rows * intermediate, fill=Scalar[y_dtype](0))
    var gate_heap = List(
        length=n_rows * gate_stride, fill=Scalar[gate_dtype](0)
    )
    var weight_heap = List(length=intermediate, fill=Float32(0))
    var out_heap = List(length=n_rows * intermediate, fill=Scalar[y_dtype](0))

    for i in range(len(y_heap)):
        y_heap[i] = Scalar[y_dtype](Float32(((i * 7) % 101) - 50) * 0.05)
    for n in range(n_rows):
        for col in range(intermediate):
            var lin = n * intermediate + col
            gate_heap[n * gate_stride + col] = Scalar[gate_dtype](
                Float32(((lin * 13) % 97) - 48) * 0.06
            )
    for c in range(intermediate):
        weight_heap[c] = Float32(((c * 5) % 41) + 10) * 0.05

    var y_t = TileTensor(y_heap, row_major(Coord(n_rows, intermediate)))
    var gate_t = TileTensor(gate_heap, row_major(Coord(n_rows, gate_stride)))
    var weight_t = TileTensor(weight_heap, row_major(Coord(intermediate)))
    var out_t = TileTensor(out_heap, row_major(Coord(n_rows, intermediate)))

    def gate_fn[
        width: Int, alignment: Int
    ](n: Int, col: Int) {var gate_t} -> SIMD[gate_dtype, width]:
        return gate_t.load[
            width=width, alignment=alignment * size_of[gate_dtype]()
        ]((n, col))

    gated_group_rmsnorm_cpu[y_dtype, gate_dtype](
        out_t,
        y_t,
        gate_fn,
        weight_t,
        n_rows,
        num_groups,
        group_size,
        eps,
    )

    for n in range(n_rows):
        for g in range(num_groups):
            var base = g * group_size
            var m2 = Float32(0)
            for j in range(group_size):
                var yv = y_heap[n * intermediate + base + j].cast[.float32]()
                var gv = gate_heap[n * gate_stride + base + j].cast[.float32]()
                var gated = yv * silu(gv)
                m2 += gated * gated
            var nf = rsqrt(m2 / Float32(group_size) + eps)
            for j in range(group_size):
                var idx = n * intermediate + base + j
                var yv = y_heap[idx].cast[.float32]()
                var gv = gate_heap[n * gate_stride + base + j].cast[.float32]()
                var t_in = (yv * silu(gv) * nf).cast[y_dtype]()
                var expected = (
                    weight_heap[base + j] * t_in.cast[.float32]()
                ).cast[y_dtype]()
                assert_almost_equal(
                    expected, out_heap[idx], rtol=2e-2, atol=1e-2
                )


def test_cpu_contiguous_gate() raises:
    run_cpu[.bfloat16, DType.float32](3, 4, 100, 400)


def test_cpu_strided_bf16_gate() raises:
    """The gate is a column slice of a wider tensor, at an odd row stride."""
    run_cpu[.bfloat16, DType.bfloat16](3, 4, 100, 517)
