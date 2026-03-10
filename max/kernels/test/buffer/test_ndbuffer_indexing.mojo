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

from buffer import NDBuffer
from buffer.dimlist import DimList
from std.testing import TestSuite


# CHECK-LABEL: test_ndbuffer_indexing
def test_ndbuffer_indexing() raises:
    print("== test_ndbuffer_indexing")

    # The total amount of data to allocate
    comptime total_buffer_size: Int = 2 * 3 * 4 * 5 * 6

    # Create a buffer for indexing test:
    var _data = InlineArray[Scalar[DType.int], total_buffer_size](
        uninitialized=True
    )

    # Fill data with increasing order, so that the value of each element in
    #  the test buffer is equal to it's linear index.:
    var fillBufferView = NDBuffer[
        rank=1, DType.int, _, DimList[total_buffer_size]()
    ](_data.unsafe_ptr())

    for fillIdx in range(total_buffer_size):
        fillBufferView[fillIdx] = Scalar[DType.int](fillIdx)

    # ===------------------------------------------------------------------=== #
    # Test 1DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView1D = NDBuffer[rank=1, DType.int, _, DimList[6]()](
        _data.unsafe_ptr()
    )

    # Try to access element[5]
    # CHECK: 5
    print(bufferView1D[5])

    # ===------------------------------------------------------------------=== #
    # Test 2DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView2D = NDBuffer[rank=2, DType.int, _, DimList[5, 6]()](
        _data.unsafe_ptr()
    )

    # Try to access element[4,5]
    # Result should be 4*6+5 = 29
    # CHECK: 29
    print(bufferView2D[4, 5])

    # ===------------------------------------------------------------------=== #
    # Test 3DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView3D = NDBuffer[rank=3, DType.int, _, DimList[4, 5, 6]()](
        _data.unsafe_ptr()
    )

    # Try to access element[3,4,5]
    # Result should be 3*(5*6)+4*6+5 = 119
    # CHECK: 119
    print(bufferView3D[3, 4, 5])

    # ===------------------------------------------------------------------=== #
    # Test 4DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView4D = NDBuffer[rank=4, DType.int, _, DimList[3, 4, 5, 6]()](
        _data.unsafe_ptr()
    )

    # Try to access element[2,3,4,5]
    # Result should be 2*4*5*6+3*5*6+4*6+5 = 359
    # CHECK: 359
    print(bufferView4D[2, 3, 4, 5])

    # ===------------------------------------------------------------------=== #
    # Test 5DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView5D = NDBuffer[rank=5, DType.int, _, DimList[2, 3, 4, 5, 6]()](
        _data.unsafe_ptr()
    )

    # Try to access element[1,2,3,4,5]
    # Result should be 1*3*4*5*6+2*4*5*6+3*5*6+4*6+5 = 719
    # CHECK: 719
    print(bufferView5D[1, 2, 3, 4, 5])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
