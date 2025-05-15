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

from math import iota

from buffer import NDBuffer
from buffer.dimlist import DimList
from memory import UnsafePointer

from utils.index import Index, IndexList


fn test(m: NDBuffer[mut=True, DType.int32, 2, _, DimList(4, 4)]):
    # CHECK: [0, 1, 2, 3]
    print(m.load[width=4](0, 0))
    # CHECK: [4, 5, 6, 7]
    print(m.load[width=4](1, 0))
    # CHECK: [8, 9, 10, 11]
    print(m.load[width=4](2, 0))
    # CHECK: [12, 13, 14, 15]
    print(m.load[width=4](3, 0))

    var v = iota[DType.int32, 4]()
    m.store[width=4](IndexList[2](3, 0), v)
    # CHECK: [0, 1, 2, 3]
    print(m.load[width=4](3, 0))


fn test_dynamic_shape(
    m: NDBuffer[mut=True, DType.int32, 2, _, DimList.create_unknown[2]()]
):
    # CHECK: [0, 1, 2, 3]
    print(m.load[width=4](0, 0))
    # CHECK: [4, 5, 6, 7]
    print(m.load[width=4](1, 0))
    # CHECK: [8, 9, 10, 11]
    print(m.load[width=4](2, 0))
    # CHECK: [12, 13, 14, 15]
    print(m.load[width=4](3, 0))

    var v = iota[DType.int32, 4]()
    m.store[width=4](IndexList[2](3, 0), v)
    # CHECK: [0, 1, 2, 3]
    print(m.load[width=4](3, 0))


fn test_matrix_static():
    print("== test_matrix_static")
    var a = NDBuffer[DType.int32, 1, MutableAnyOrigin, 16].stack_allocation()
    var m = NDBuffer[DType.int32, 2, _, DimList(4, 4)](a.data)
    for i in range(16):
        a[i] = i
    test(m)


fn test_matrix_dynamic():
    print("== test_matrix_dynamic")
    var a = NDBuffer[DType.int32, 1, MutableAnyOrigin, 16].stack_allocation()
    var m = NDBuffer[DType.int32, 2, _, DimList(4, 4)](a.data)
    for i in range(16):
        a[i] = i
    test(m)


fn test_matrix_dynamic_shape():
    print("== test_matrix_dynamic_shape")
    var a = NDBuffer[DType.int32, 1, MutableAnyOrigin, 16].stack_allocation()
    # var m = Matrix[DimList(4, 4), DType.int32, False](a.data, Index(4,4), DType.int32)
    var m = NDBuffer[DType.int32, 2, _, DimList.create_unknown[2]()](
        a.data, Index(4, 4)
    )
    for i in range(16):
        a[i] = i
    test_dynamic_shape(m)


fn main():
    test_matrix_static()
    test_matrix_dynamic()
    test_matrix_dynamic_shape()
