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

from algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    elementwise,
)
from testing import assert_equal, assert_true
from testing import TestSuite

from utils.index import IndexList, Index


fn _linear_index[
    rank: Int
](coords: IndexList[rank], shape: IndexList[rank]) -> Int:
    """Convert multi-dimensional coordinates to linear index (row-major)."""
    var linear_idx = 0
    var stride = 1

    comptime for i in reversed(range(rank)):
        linear_idx += coords[i] * stride
        stride *= shape[i]
    return linear_idx


def test_elementwise():
    def run_elementwise[
        numelems: Int,
        outer_rank: Int,
        is_blocking: Bool,
        shape: IndexList[outer_rank],
    ]():
        var memory1 = InlineArray[Float32, numelems](uninitialized=True)
        var buffer1 = Span[Float32](memory1)

        var memory2 = InlineArray[Float32, numelems](uninitialized=True)
        var buffer2 = Span[Float32](memory2)

        var memory3 = InlineArray[Float32, numelems](uninitialized=True)
        var out_buffer = Span[Float32](memory3)

        var x: Float32 = 1.0
        for i in range(numelems):
            buffer1.unsafe_ptr()[i] = 2.0
            buffer2.unsafe_ptr()[i] = x
            out_buffer.unsafe_ptr()[i] = 0.0
            x += 1.0

        @always_inline
        @__copy_capture(buffer1, buffer2, out_buffer, shape)
        @parameter
        fn func[
            simd_width: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]):
            var index = rebind[IndexList[outer_rank]](idx)
            var linear_idx = _linear_index(index, shape)
            var in1 = buffer1.unsafe_ptr().load[width=simd_width](linear_idx)
            var in2 = buffer2.unsafe_ptr().load[width=simd_width](linear_idx)
            out_buffer.unsafe_ptr().store[width=simd_width](
                linear_idx, in1 * in2
            )

        elementwise[func, simd_width=1, use_blocking_impl=is_blocking](
            shape,
        )

        for i2 in range(min(numelems, 64)):
            assert_equal(
                (out_buffer.unsafe_ptr() + i2).load(), Float32(2 * (i2 + 1))
            )

    run_elementwise[16, 1, False, Index(16)]()
    run_elementwise[16, 1, True, Index(16)]()
    run_elementwise[16, 2, False, Index(4, 4)]()
    run_elementwise[16, 2, True, Index(4, 4)]()
    run_elementwise[16, 3, False, Index(4, 2, 2)]()
    run_elementwise[16, 3, True, Index(4, 2, 2)]()
    run_elementwise[32, 4, False, Index(4, 2, 2, 2)]()
    run_elementwise[32, 4, True, Index(4, 2, 2, 2)]()
    run_elementwise[32, 5, False, Index(4, 2, 1, 2, 2)]()
    run_elementwise[32, 5, True, Index(4, 2, 1, 2, 2)]()
    run_elementwise[131072, 2, False, Index(1024, 128)]()
    run_elementwise[131072, 2, True, Index(1024, 128)]()


def test_elementwise_implicit_runtime():
    var vector_stack = InlineArray[Scalar[DType.int], 20](uninitialized=True)
    var vector = Span[Scalar[DType.int]](vector_stack)

    for i in range(len(vector)):
        vector.unsafe_ptr()[i] = Scalar[DType.int](i)

    @always_inline
    @__copy_capture(vector)
    @parameter
    fn func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        vector.unsafe_ptr()[idx[0]] = 42

    elementwise[func, simd_width=1](20)

    for i in range(len(vector)):
        assert_equal(vector.unsafe_ptr()[i], 42)


def test_indices_conversion():
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


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
