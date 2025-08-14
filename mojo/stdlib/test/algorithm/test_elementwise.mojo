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

from algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    elementwise,
)
from buffer import NDBuffer
from buffer.dimlist import DimList

from utils.index import IndexList
from testing import assert_true, assert_equal


def test_elementwise():
    def run_elementwise[
        numelems: Int, outer_rank: Int, is_blocking: Bool
    ](dims: DimList):
        var memory1 = InlineArray[Float32, numelems](uninitialized=True)
        var buffer1 = NDBuffer[DType.float32, outer_rank](
            memory1.unsafe_ptr(), dims
        )

        var memory2 = InlineArray[Float32, numelems](uninitialized=True)
        var buffer2 = NDBuffer[DType.float32, outer_rank](
            memory2.unsafe_ptr(), dims
        )

        var memory3 = InlineArray[Float32, numelems](uninitialized=True)
        var out_buffer = NDBuffer[DType.float32, outer_rank](
            memory3.unsafe_ptr(), dims
        )

        var x: Float32 = 1.0
        for i in range(numelems):
            buffer1.data[i] = 2.0
            buffer2.data[i] = Float32(x.value)
            out_buffer.data[i] = 0.0
            x += 1.0

        @always_inline
        @parameter
        fn func[
            simd_width: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]):
            var index = rebind[IndexList[outer_rank]](idx)
            var in1 = buffer1.load[width=simd_width](index)
            var in2 = buffer2.load[width=simd_width](index)
            out_buffer.store[width=simd_width](index, in1 * in2)

        elementwise[func, simd_width=1, use_blocking_impl=is_blocking](
            rebind[IndexList[outer_rank]](out_buffer.get_shape()),
        )

        for i2 in range(min(numelems, 64)):
            assert_equal(out_buffer.data.offset(i2).load(), 2 * (i2 + 1))

    run_elementwise[16, 1, False](DimList(16))
    run_elementwise[16, 1, True](DimList(16))
    run_elementwise[16, 2, False](DimList(4, 4))
    run_elementwise[16, 2, True](DimList(4, 4))
    run_elementwise[16, 3, False](DimList(4, 2, 2))
    run_elementwise[16, 3, True](DimList(4, 2, 2))
    run_elementwise[32, 4, False](DimList(4, 2, 2, 2))
    run_elementwise[32, 4, True](DimList(4, 2, 2, 2))
    run_elementwise[32, 5, False](DimList(4, 2, 1, 2, 2))
    run_elementwise[32, 5, True](DimList(4, 2, 1, 2, 2))
    run_elementwise[131072, 2, False](DimList(1024, 128))
    run_elementwise[131072, 2, True](DimList(1024, 128))


def test_elementwise_implicit_runtime():
    var vector_stack = InlineArray[Scalar[DType.index], 20](uninitialized=True)
    var vector = NDBuffer[DType.index, 1, _, 20](vector_stack)

    for i in range(len(vector)):
        vector[i] = i

    @always_inline
    @parameter
    fn func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        vector[idx[0]] = 42

    elementwise[func, simd_width=1](20)

    for i in range(len(vector)):
        assert_equal(vector[i], 42)


def test_indices_conversion():
    var shape = IndexList[4](3, 4, 5, 6)
    assert_equal(
        _get_start_indices_of_nth_subvolume[0](10, shape), (0, 0, 1, 4)
    )
    assert_equal(
        _get_start_indices_of_nth_subvolume[1](10, shape), (0, 2, 0, 0)
    )
    assert_equal(
        _get_start_indices_of_nth_subvolume[2](10, shape), (2, 2, 0, 0)
    )
    assert_equal(_get_start_indices_of_nth_subvolume[3](2, shape), (2, 0, 0, 0))
    assert_equal(_get_start_indices_of_nth_subvolume[4](0, shape), (0, 0, 0, 0))


def main():
    test_elementwise()
    test_elementwise_implicit_runtime()
    test_indices_conversion()
