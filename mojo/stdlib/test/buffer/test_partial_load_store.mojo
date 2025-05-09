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

from math import exp

from buffer.buffer import NDBuffer, partial_simd_load, partial_simd_store
from buffer.dimlist import DimList

from utils.index import IndexList


# CHECK-LABEL: test_partial_load_store
fn test_partial_load_store():
    print("== test_partial_load_store")
    # The total amount of data to allocate
    alias total_buffer_size: Int = 32

    var read_data = InlineArray[Scalar[DType.index], total_buffer_size](
        uninitialized=True
    )
    var write_data = InlineArray[Scalar[DType.index], total_buffer_size](
        uninitialized=True
    )

    var read_buffer = NDBuffer[DType.index, 1, _, total_buffer_size](read_data)

    var write_buffer = NDBuffer[DType.index, 1, _, total_buffer_size](
        write_data
    )

    for idx in range(total_buffer_size):
        # Fill read_bufer with 0->15
        read_buffer[idx] = idx
        # Fill write_buffer with 0
        write_buffer[idx] = 0

    # Test partial load:
    var partial_load_data = partial_simd_load[4](
        read_buffer.data.offset(1),
        1,
        3,
        99,  # idx  # lbound  # rbound  # pad value
    )
    # CHECK: [99, 2, 3, 99]
    print(partial_load_data)

    # Test partial store:
    partial_simd_store[4](
        write_buffer.data.offset(1),
        2,
        4,
        partial_load_data,  # idx  # lbound  # rbound
    )
    var partial_store_data = write_buffer.load[width=4](2)
    # CHECK: [0, 3, 99, 0]
    print(partial_store_data)

    # Test NDBuffer partial load store
    var read_nd_buffer = NDBuffer[DType.index, 2, _, DimList(8, 4)](read_data)

    var write_nd_buffer = NDBuffer[DType.index, 2, _, DimList(8, 4)](write_data)

    # Test partial load:
    var nd_partial_load_data = partial_simd_load[4](
        read_nd_buffer._offset(IndexList[2](3, 2)),
        0,
        2,
        123,  # lbound  # rbound  # pad value
    )
    # CHECK: [14, 15, 123, 123]
    print(nd_partial_load_data)

    # Test partial store
    partial_simd_store[4](
        write_nd_buffer._offset(IndexList[2](3, 1)),
        0,  # lbound
        3,  # rbound
        nd_partial_load_data,  # value
    )
    var nd_partial_store_data = write_nd_buffer.load[width=4](
        IndexList[2](3, 0)
    )

    # CHECK: [0, 14, 15, 123]
    print(nd_partial_store_data)


fn main():
    test_partial_load_store()
