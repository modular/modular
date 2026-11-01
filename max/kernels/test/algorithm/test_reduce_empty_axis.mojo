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
"""CPU half of the empty-reduce-axis regression, for argmax.

`algorithm.cpu.rowwise`'s `launch` used to return before writing anything
when the reduced axis was empty. A reduce-shaped body still owns
`product(shape \\ axis)` outputs, each of which must be written. The output
here is pre-filled with `-1`, which no valid index is, so a skipped write is
caught.

Sum, max, min and mean compile to generated rowwise kernels, so their
empty-axis cases live in the graph-level property test,
`//max/tests/integration/graph:test_rowwise_reductions_fuzz`.
"""

from builtin_kernels.rowwise_reductions import reduce_argmax
from std.testing import TestSuite, assert_equal
from std.utils.coord import Coord


def _run_reduce_argmax_inner[
    dtype: DType = .float32
](num_rows: Int, axis_size: Int) raises -> List[Int64]:
    """`reduce_argmax` over a `[num_rows, axis_size]` input, axis=1."""
    var input_buf = List(length=num_rows * axis_size, fill=Scalar[dtype](2))
    var input_ptr = input_buf.unsafe_ptr()
    var output_buf = List(length=num_rows, fill=Int64(-1))
    var output_ptr = output_buf.unsafe_ptr()

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var input_ptr, var axis_size} -> SIMD[dtype, width]:
        return input_ptr.unsafe_load[width=width](
            Int(coords[0]) * axis_size + Int(coords[1])
        )

    @inline(.always)
    def output_fn[
        width: SIMDLength
    ](coords: Coord, val: SIMD[.int64, width]) {var output_ptr}:
        output_ptr.unsafe_store[width=width](Int(coords[0]), val)

    reduce_argmax[dtype, target="cpu", reduce_dim=1](
        input_fn, output_fn, Coord(num_rows, axis_size)
    )
    return output_buf^


def test_reduce_argmax_empty_axis_inner() raises:
    """`argmax` over an empty axis: no candidate ever wins, so the
    `ArgMax` monoid's `Int64.MAX` identity index gets clamped to `0` by
    `_in_range_index` — the same contract `test_arg_reduce_nan.mojo`
    already establishes for an all-NaN row (`_in_range_index`'s own
    docstring names "an empty row" as one of the cases it exists for)."""
    var out = _run_reduce_argmax_inner(num_rows=2, axis_size=0)
    assert_equal(len(out), 2)
    assert_equal(out[0], Int64(0))
    assert_equal(out[1], Int64(0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
