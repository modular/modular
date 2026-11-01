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

from std.math import inf, rsqrt
from std.sys import simd_width_of, size_of

from layout import MixedLayout, TileTensor, row_major
from nn.activations import silu
from state_space.gated_group_rmsnorm import (
    gated_group_rmsnorm_cpu,
)
from std.testing import assert_almost_equal, assert_equal


def main() raises:
    test_cpu_contiguous_gate()
    test_cpu_strided_bf16_gate()
    test_cpu_vector_tails()
    test_cpu_strided_elements()


def run_cpu[
    y_dtype: DType, gate_dtype: DType
](
    n_rows: Int,
    num_groups: Int,
    group_size: Int,
    gate_stride: Int,
    inner_stride: Int = 1,
) raises:
    var eps = Float32(1e-5)
    var intermediate = num_groups * group_size
    var y_heap = List(
        length=n_rows * intermediate * inner_stride, fill=Scalar[y_dtype](0)
    )
    var gate_heap = List(
        length=n_rows * gate_stride, fill=Scalar[gate_dtype](0)
    )
    var weight_heap = List(length=intermediate * inner_stride, fill=Float32(0))
    var out_heap = List(
        length=n_rows * intermediate * inner_stride, fill=inf[y_dtype]()
    )

    for i in range(n_rows * intermediate):
        y_heap[i * inner_stride] = Scalar[y_dtype](
            Float32(((i * 7) % 101) - 50) * 0.05
        )
    for n in range(n_rows):
        for col in range(intermediate):
            var lin = n * intermediate + col
            gate_heap[n * gate_stride + col] = Scalar[gate_dtype](
                Float32(((lin * 13) % 97) - 48) * 0.06
            )
    for c in range(intermediate):
        weight_heap[c * inner_stride] = Float32(((c * 5) % 41) + 10) * 0.05

    var y_t = TileTensor(
        y_heap,
        MixedLayout(
            (n_rows, intermediate), (intermediate * inner_stride, inner_stride)
        ),
    )
    var gate_t = TileTensor(gate_heap, row_major(n_rows, gate_stride))
    var weight_t = TileTensor(
        weight_heap, MixedLayout((intermediate,), (inner_stride,))
    )
    var out_t = TileTensor(
        out_heap,
        MixedLayout(
            (n_rows, intermediate), (intermediate * inner_stride, inner_stride)
        ),
    )

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
                var yv = y_heap[
                    (n * intermediate + base + j) * inner_stride
                ].cast[.float32]()
                var gv = gate_heap[n * gate_stride + base + j].cast[.float32]()
                var gated = yv * silu(gv)
                m2 += gated * gated
            var nf = rsqrt(m2 / Float32(group_size) + eps)
            for j in range(group_size):
                var idx = n * intermediate + base + j
                var yv = y_heap[idx * inner_stride].cast[.float32]()
                var gv = gate_heap[n * gate_stride + base + j].cast[.float32]()
                var t_in = (yv * silu(gv) * nf).cast[y_dtype]()
                var expected = (
                    weight_heap[(base + j) * inner_stride]
                    * t_in.cast[.float32]()
                ).cast[y_dtype]()
                assert_almost_equal(
                    expected, out_heap[idx * inner_stride], rtol=1e-6, atol=1e-6
                )

    for i in range(n_rows * intermediate):
        for padding in range(1, inner_stride):
            assert_equal(out_heap[i * inner_stride + padding], inf[y_dtype]())


def test_cpu_contiguous_gate() raises:
    run_cpu[.bfloat16, .float32](3, 4, 100, 400)


def test_cpu_strided_bf16_gate() raises:
    """The gate is a column slice of a wider tensor, at an odd row stride."""
    run_cpu[.bfloat16, .bfloat16](3, 4, 100, 517)


def test_cpu_vector_tails() raises:
    comptime width = simd_width_of[DType.float32]()
    for group_size in [1, width - 1, width, width + 1, 2 * width + 3]:
        run_cpu[.float32, .float32](3, 3, group_size, 3 * group_size + 5)
        run_cpu[.float16, .float16](3, 3, group_size, 3 * group_size + 5)
        run_cpu[.bfloat16, .float32](3, 3, group_size, 3 * group_size + 5)


def test_cpu_strided_elements() raises:
    comptime width = simd_width_of[DType.float32]()
    run_cpu[.float32, .float32](2, 3, 2 * width + 1, 6 * width + 7, 2)
