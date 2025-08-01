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

from algorithm import vectorize
from buffer import NDBuffer
from memory import memcmp

from testing import assert_equal


def test_vectorize():
    # Create a mem of size 5
    var vector_stack = InlineArray[Float32, 5](1.0, 2.0, 3.0, 4.0, 5.0)
    var vector = NDBuffer[DType.float32, 1, _, 5](vector_stack)

    @__copy_capture(vector)
    @always_inline
    @parameter
    fn add_two[simd_width: Int](idx: Int):
        vector.store[width=simd_width](
            idx, vector.load[width=simd_width](idx) + 2
        )

    vectorize[add_two, 2](len(vector))

    assert_equal(vector[0], 3.0)
    assert_equal(vector[1], 4.0)
    assert_equal(vector[2], 5.0)
    assert_equal(vector[3], 6.0)
    assert_equal(vector[4], 7.0)

    @always_inline
    @__copy_capture(vector)
    @parameter
    fn add[simd_width: Int](idx: Int):
        vector.store[width=simd_width](
            idx,
            vector.load[width=simd_width](idx)
            + vector.load[width=simd_width](idx),
        )

    vectorize[add, 2](len(vector))

    assert_equal(vector[0], 6.0)
    assert_equal(vector[1], 8.0)
    assert_equal(vector[2], 10.0)
    assert_equal(vector[3], 12.0)
    assert_equal(vector[4], 14.0)


def test_vectorize_unroll():
    alias buf_len = 23

    var vec_stack = InlineArray[Float32, buf_len](uninitialized=True)
    var vec = NDBuffer[DType.float32, 1, _, buf_len](vec_stack)
    var buf_stack = InlineArray[Float32, buf_len](uninitialized=True)
    var buf = NDBuffer[DType.float32, 1, _, buf_len](buf_stack)

    for i in range(buf_len):
        vec[i] = i
        buf[i] = i

    @always_inline
    @__copy_capture(buf)
    @parameter
    fn double_buf[simd_width: Int](idx: Int):
        buf.store[width=simd_width](
            idx,
            buf.load[width=simd_width](idx) + buf.load[width=simd_width](idx),
        )

    @parameter
    @__copy_capture(vec)
    @always_inline
    fn double_vec[simd_width: Int](idx: Int):
        vec.store[width=simd_width](
            idx,
            vec.load[width=simd_width](idx) + vec.load[width=simd_width](idx),
        )

    alias simd_width = 4
    alias unroll_factor = 2

    vectorize[double_vec, simd_width, unroll_factor=unroll_factor](len(vec))
    vectorize[double_buf, simd_width](len(buf))

    var err = memcmp(vec.data, buf.data, len(buf))
    assert_equal(err, 0)


def test_vectorize_size_param():
    var output = String()

    # remainder elements are correctly printed
    @parameter
    fn printer[els: Int](n: Int):
        output.write(els, " ", n, "\n")

    vectorize[printer, 16, size=40]()
    assert_equal(output, "16 0\n16 16\n8 32\n")

    vectorize[printer, 16, size=8]()
    assert_equal(output, "16 0\n16 16\n8 32\n8 0\n")


def main():
    test_vectorize()
    test_vectorize_unroll()
    test_vectorize_size_param()
