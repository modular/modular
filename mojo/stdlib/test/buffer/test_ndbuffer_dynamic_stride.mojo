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

from buffer import NDBuffer
from buffer.dimlist import DimList
from testing import TestSuite

from utils.index import Index


# CHECK-LABEL: test_sub_matrix
def test_sub_matrix():
    print("== test_sub_matrix")
    alias num_row = 4
    alias num_col = 4

    var matrix_stack = InlineArray[Float32, num_row * num_col](
        uninitialized=True
    )
    # Create a 4x4 matrix.
    var matrix = NDBuffer[DType.float32, 2, _, DimList(num_row, num_col)](
        matrix_stack
    )
    for i in range(num_row):
        for j in range(num_col):
            matrix[Index(i, j)] = Float32(i * num_col + j)

    # Extract a sub-matrix 2x2 at (1,1).
    var sub_matrix0 = NDBuffer[DType.float32, 2, _, DimList(2, 2)](
        matrix.data.offset(5),
        DimList(2, 2),
        Index(4, 1),
    )

    # CHECK: 4
    print(sub_matrix0.stride(0))
    # CHECK: 1
    print(sub_matrix0.stride(1))
    # CHECK: True
    print(sub_matrix0.is_contiguous())
    # CHECK: 6.0
    print(sub_matrix0[Index(0, 1)])
    # CHECK: 10.0
    print(sub_matrix0[Index(1, 1)])

    # Extract a sub-matrix 2x2 at (1,1) with discontiguous last dim.
    # It includes (1,1) (1,3) (3,1) (3,3) of the original matrix.
    var sub_matrix1 = NDBuffer[DType.float32, 2, _, DimList(2, 2)](
        matrix.data.offset(1),
        DimList(2, 2),
        Index(8, 2),
    )

    # CHECK: 3.0
    print(sub_matrix1[Index(0, 1)])
    # CHECK: 9.0
    print(sub_matrix1[Index(1, 0)])

    # Extract a contiguous 2x2 buffer starting at (1,1).
    # It includes (1,1) (1,2) (1,3) (2,1) of the original matrix.
    var sub_matrix2 = NDBuffer[DType.float32, 2, _, DimList(2, 2)](
        matrix.data.offset(5),
        DimList(2, 2),
        Index(2, 1),
    )

    # CHECK: True
    print(sub_matrix2.is_contiguous())
    # CHECK: 8.0
    print(sub_matrix2[Index(1, 1)])


# CHECK-LABEL: test_broadcast
def test_broadcast():
    print("== test_broadcast")

    # Create a buffer holding a single value with zero stride.
    var arr = InlineArray[Float32, 1](uninitialized=True)
    var stride_buf = NDBuffer[DType.float32, 1, _, DimList(100)](
        arr, DimList(100), Index(0)
    )

    # CHECK: 2.0
    stride_buf[0] = 2.0
    print(stride_buf[13])
    # CHECK: 2.0
    stride_buf[41] = 2.0
    print(stride_buf[99])


def main():
    var suite = TestSuite()

    suite.test[test_sub_matrix]()
    suite.test[test_broadcast]()

    suite^.run()
