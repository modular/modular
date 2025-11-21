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
from testing import TestSuite


def test_vectorize():
    # Create a mem of size 5
    var vector_stack = InlineArray[Float32, 5](1.0, 2.0, 3.0, 4.0, 5.0)
    var vector = NDBuffer[DType.float32, 1, _, 5](vector_stack.unsafe_ptr())

    @always_inline
    fn add_two[width: Int](idx: Int) unified {var vector}:
        vector.store[width=width](idx, vector.load[width=width](idx) + 2)

    vectorize[2](len(vector), add_two)

    assert_equal(vector[0], 3.0)
    assert_equal(vector[1], 4.0)
    assert_equal(vector[2], 5.0)
    assert_equal(vector[3], 6.0)
    assert_equal(vector[4], 7.0)

    @always_inline
    fn add[width: Int](idx: Int) unified {var vector}:
        vector.store[width=width](
            idx,
            vector.load[width=width](idx) + vector.load[width=width](idx),
        )

    vectorize[2](len(vector), add)

    assert_equal(vector[0], 6.0)
    assert_equal(vector[1], 8.0)
    assert_equal(vector[2], 10.0)
    assert_equal(vector[3], 12.0)
    assert_equal(vector[4], 14.0)


def test_vectorize_unroll():
    comptime buf_len = 23

    var vec_stack = InlineArray[Float32, buf_len](uninitialized=True)
    var vec = NDBuffer[DType.float32, 1, _, buf_len](vec_stack.unsafe_ptr())
    var buf_stack = InlineArray[Float32, buf_len](uninitialized=True)
    var buf = NDBuffer[DType.float32, 1, _, buf_len](buf_stack.unsafe_ptr())

    for i in range(buf_len):
        vec[i] = i
        buf[i] = i

    @always_inline
    fn double_buf[simd_width: Int](idx: Int) unified {var buf}:
        buf.store[width=simd_width](
            idx,
            buf.load[width=simd_width](idx) + buf.load[width=simd_width](idx),
        )

    @always_inline
    fn double_vec[simd_width: Int](idx: Int) unified {var vec}:
        vec.store[width=simd_width](
            idx,
            vec.load[width=simd_width](idx) + vec.load[width=simd_width](idx),
        )

    comptime simd_width = 4
    comptime unroll_factor = 2

    vectorize[simd_width, unroll_factor=unroll_factor](len(vec), double_vec)
    vectorize[simd_width](len(buf), double_buf)

    var err = memcmp(vec.data, buf.data, len(buf))
    assert_equal(err, 0)


def test_vectorize_size_param():
    var output = String()

    # remainder elements are correctly printed
    @parameter
    fn printer[els: Int](n: Int) unified {mut output}:
        output.write(els, " ", n, "\n")

    vectorize[16, size=40](printer)
    assert_equal(output, "16 0\n16 16\n8 32\n")

    vectorize[16, size=8](printer)
    assert_equal(output, "16 0\n16 16\n8 32\n8 0\n")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
