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

from buffer.buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from testing import TestSuite


# CHECK-LABEL: test_ndbuffer_dynamic_shape
def test_ndbuffer_dynamic_shape():
    print("== test_ndbuffer_dynamic_shape")

    # Create a buffer of size 16
    var buffer = InlineArray[Scalar[DType.int], 16](uninitialized=True)

    var matrix = NDBuffer[DType.int, 2](buffer, DimList(4, 4))

    matrix.dynamic_shape[0] = 42
    matrix.dynamic_shape[1] = 43

    # CHECK: 42
    print(matrix.dim[0]())
    # CHECK: 43
    print(matrix.dim[1]())

    # Mix static and dynamic shape.
    var matrix2 = NDBuffer[
        DType.int,
        2,
        _,
        DimList(42, Dim()),
    ](buffer.unsafe_ptr(), DimList(42, 1))

    matrix2.dynamic_shape[1] = 43

    # CHECK: 42
    print(matrix2.dim[0]())
    # CHECK: 43
    print(matrix2.dim[1]())


def main():
    var suite = TestSuite()

    suite.test[test_ndbuffer_dynamic_shape]()

    suite^.run()
