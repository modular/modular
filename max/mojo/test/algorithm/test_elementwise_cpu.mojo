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

from max.algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    _start_indices_of_nth_element,
    elementwise,
)
from max.gpu.host import DeviceContext
from std.testing import assert_equal
from std.testing import TestSuite

from std.utils.coord import ComptimeInt, Coord, coord_to_index_list
from std.utils.index import IndexList, Index


def _linear_index[
    rank: Int
](coords: IndexList[rank], shape: IndexList[rank]) -> Int:
    """Convert multi-dimensional coordinates to linear index (row-major)."""
    var linear_idx = 0
    var stride = 1

    comptime for i in reversed(range(rank)):
        linear_idx += coords[i] * stride
        stride *= shape[i]
    return linear_idx


def test_elementwise() raises:
    var ctx = DeviceContext(api="cpu")

    def run_elementwise[
        numelems: Int,
        outer_rank: Int,
        shape: IndexList[outer_rank],
    ](ctx: DeviceContext) raises:
        var memory1 = Array[Float32, numelems](fill={})
        var buffer1 = Span(memory1)

        var memory2 = Array[Float32, numelems](fill={})
        var buffer2 = Span(memory2)

        var memory3 = Array[Float32, numelems](fill={})
        var out_buffer = Span(memory3)

        var x: Float32 = 1.0
        for i in range(numelems):
            buffer1.unsafe_ptr()[unsafe_offset=i] = 2.0
            buffer2.unsafe_ptr()[unsafe_offset=i] = x
            out_buffer.unsafe_ptr()[unsafe_offset=i] = 0.0
            x += 1.0

        @inline(.always)
        def func[
            simd_width: Int, alignment: Int = 1
        ](idx: Coord) {var buffer1, var buffer2, var out_buffer}:
            var index = rebind[IndexList[outer_rank]](coord_to_index_list(idx))
            var linear_idx = _linear_index(index, shape)
            var in1 = buffer1.unsafe_ptr().unsafe_load[width=simd_width](
                linear_idx
            )
            var in2 = buffer2.unsafe_ptr().unsafe_load[width=simd_width](
                linear_idx
            )
            out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
                linear_idx, in1 * in2
            )

        elementwise[simd_width=1](
            func,
            Coord(shape),
            ctx,
        )

        for i2 in range(min(numelems, 64)):
            assert_equal(
                out_buffer.unsafe_ptr().unsafe_offset(i2).unsafe_load(),
                Float32(2 * (i2 + 1)),
            )

    run_elementwise[16, 1, Index(16)](ctx)
    run_elementwise[16, 2, Index(4, 4)](ctx)
    run_elementwise[16, 3, Index(4, 2, 2)](ctx)
    run_elementwise[32, 4, Index(4, 2, 2, 2)](ctx)
    run_elementwise[32, 5, Index(4, 2, 1, 2, 2)](ctx)
    run_elementwise[131072, 2, Index(1024, 128)](ctx)


def test_elementwise_implicit_runtime() raises:
    var ctx = DeviceContext(api="cpu")
    var vector_stack = Array[Int, 20](fill={})
    var vector = Span(vector_stack)

    for i in range(len(vector)):
        vector.unsafe_ptr()[unsafe_offset=i] = Int(i)

    @inline(.always)
    def func[simd_width: Int, alignment: Int = 1](idx: Coord) {var vector}:
        vector.unsafe_ptr()[unsafe_offset=idx[0].value()] = 42

    elementwise[simd_width=1](func, Coord(20), ctx)

    for i in range(len(vector)):
        assert_equal(vector.unsafe_ptr()[unsafe_offset=i], 42)


def test_indices_conversion() raises:
    var shape = IndexList[4](3, 4, 5, 6)
    assert_equal(
        _get_start_indices_of_nth_subvolume[0](10, shape),
        IndexList[4](0, 0, 1, 4),
    )
    assert_equal(
        _get_start_indices_of_nth_subvolume[1](10, shape),
        IndexList[4](0, 2, 0, 0),
    )
    assert_equal(
        _get_start_indices_of_nth_subvolume[2](10, shape),
        IndexList[4](2, 2, 0, 0),
    )
    assert_equal(
        _get_start_indices_of_nth_subvolume[3](2, shape),
        IndexList[4](2, 0, 0, 0),
    )
    assert_equal(
        _get_start_indices_of_nth_subvolume[4](0, shape),
        IndexList[4](0, 0, 0, 0),
    )


def _assert_start_indices_match[
    rank: Int, use_32bit: Bool
](coord_shape: Coord, index_shape: IndexList[rank], n_max: Int) raises:
    """Compares `_start_indices_of_nth_element` against the `IndexList` helper.
    """
    for n in range(n_max):
        var got_0 = _start_indices_of_nth_element[
            rank, type_of(coord_shape).ParamListType, use_32bit, 0
        ](n, coord_shape)
        var want_0 = _get_start_indices_of_nth_subvolume[0](n, index_shape)
        var got_1 = _start_indices_of_nth_element[
            rank, type_of(coord_shape).ParamListType, use_32bit, 1
        ](n, coord_shape)
        var want_1 = _get_start_indices_of_nth_subvolume[1](n, index_shape)
        for i in range(rank):
            assert_equal(got_0[i], want_0[i])
            assert_equal(got_1[i], want_1[i])


def test_start_indices_equivalence() raises:
    # All-dynamic extents take the runtime `divmod` path in both widths.
    var dynamic_shape = Coord(3, 4, 5, 6)
    var index_shape = IndexList[4](3, 4, 5, 6)
    _assert_start_indices_match[4, True](dynamic_shape, index_shape, 60)
    _assert_start_indices_match[4, False](dynamic_shape, index_shape, 60)

    # Mixed static/dynamic extents take the compile-time divisor fold.
    var mixed_shape = Coord(ComptimeInt[3](), 4, ComptimeInt[5](), 6)
    _assert_start_indices_match[4, True](mixed_shape, index_shape, 60)
    _assert_start_indices_match[4, False](mixed_shape, index_shape, 60)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
