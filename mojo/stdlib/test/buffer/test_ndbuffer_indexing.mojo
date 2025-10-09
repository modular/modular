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


# CHECK-LABEL: test_ndbuffer_indexing
def test_ndbuffer_indexing():
    print("== test_ndbuffer_indexing")

    # The total amount of data to allocate
    alias total_buffer_size: Int = 2 * 3 * 4 * 5 * 6

    # Create a buffer for indexing test:
    var _data = InlineArray[Scalar[DType.int], total_buffer_size](
        uninitialized=True
    )

    # Fill data with increasing order, so that the value of each element in
    #  the test buffer is equal to it's linear index.:
    var fillBufferView = NDBuffer[DType.int, 1, _, total_buffer_size](_data)

    for fillIdx in range(total_buffer_size):
        fillBufferView[fillIdx] = fillIdx

    # ===------------------------------------------------------------------=== #
    # Test 1DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView1D = NDBuffer[DType.int, 1, _, DimList(6)](_data)

    # Try to access element[5]
    # CHECK: 5
    print(bufferView1D[5])

    # ===------------------------------------------------------------------=== #
    # Test 2DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView2D = NDBuffer[DType.int, 2, _, DimList(5, 6)](_data)

    # Try to access element[4,5]
    # Result should be 4*6+5 = 29
    # CHECK: 29
    print(bufferView2D[4, 5])

    # ===------------------------------------------------------------------=== #
    # Test 3DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView3D = NDBuffer[DType.int, 3, _, DimList(4, 5, 6)](_data)

    # Try to access element[3,4,5]
    # Result should be 3*(5*6)+4*6+5 = 119
    # CHECK: 119
    print(bufferView3D[3, 4, 5])

    # ===------------------------------------------------------------------=== #
    # Test 4DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView4D = NDBuffer[DType.int, 4, _, DimList(3, 4, 5, 6)](_data)

    # Try to access element[2,3,4,5]
    # Result should be 2*4*5*6+3*5*6+4*6+5 = 359
    # CHECK: 359
    print(bufferView4D[2, 3, 4, 5])

    # ===------------------------------------------------------------------=== #
    # Test 5DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView5D = NDBuffer[DType.int, 5, _, DimList(2, 3, 4, 5, 6)](_data)

    # Try to access element[1,2,3,4,5]
    # Result should be 1*3*4*5*6+2*4*5*6+3*5*6+4*6+5 = 719
    # CHECK: 719
    print(bufferView5D[1, 2, 3, 4, 5])


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
