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
# RUN: %mojo-no-debug %s | FileCheck %s

from algorithm import vectorize
from buffer import NDBuffer
from memory import memcmp


# CHECK-LABEL: test_vectorize
fn test_vectorize():
    print("== test_vectorize")

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

    # CHECK: 3.0
    print(vector[0])
    # CHECK: 4.0
    print(vector[1])
    # CHECK: 5.0
    print(vector[2])
    # CHECK: 6.0
    print(vector[3])
    # CHECK: 7.0
    print(vector[4])

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

    # CHECK: 6.0
    print(vector[0])
    # CHECK: 8.0
    print(vector[1])
    # CHECK: 10.0
    print(vector[2])
    # CHECK: 12.0
    print(vector[3])
    # CHECK: 14.0
    print(vector[4])


# CHECK-LABEL: test_vectorize_unroll
fn test_vectorize_unroll():
    print("== test_vectorize_unroll")

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
    # CHECK: 0
    print(err)


# CHECK-LABEL: test_vectorize_size_param
fn test_vectorize_size_param():
    print("== test_vectorize_size_param")

    # remainder elements are corectly printed
    @parameter
    fn printer[els: Int](n: Int):
        print(els, n)

    # CHECK: 16 0
    # CHECK: 16 16
    # CHECK: 8 32
    vectorize[printer, 16, size=40]()

    # CHECK: 8 0
    vectorize[printer, 16, size=8]()


fn main():
    test_vectorize()
    test_vectorize_unroll()
    test_vectorize_size_param()
