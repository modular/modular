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
from internal_utils import TestTensor, assert_equal
from nn.gather_scatter import gather_elements


fn test_case[
    type: DType
](
    axis: Int,
    data: TestTensor[type, 2],
    indices: TestTensor[DType.int32, 2],
    output: TestTensor[type, 2],
) raises:
    var output_ref = output

    gather_elements(
        data.ndbuffer,
        indices.ndbuffer,
        axis,
        output.ndbuffer,
    )

    assert_equal(output, output_ref)


fn main() raises:
    fn test_gather_ax1() raises:
        print("== test_gather_ax1")

        alias shape = DimList(2, 2)

        var data = TestTensor[DType.float32, 2](
            shape, List[Float32](1, 2, 3, 4)
        )
        var indices = TestTensor[DType.int32, 2](shape, List[Int32](0, 0, 1, 0))
        var output_ref = TestTensor[DType.float32, 2](
            shape, List[Float32](1, 1, 4, 3)
        )

        test_case[DType.float32](1, data, indices, output_ref)

    # CHECK-LABEL: test_gather_ax1
    # CHECK-NOT: FAIL
    test_gather_ax1()

    fn test_gather_ax0() raises:
        print("== test_gather_ax0")

        var data = TestTensor[DType.float32, 2](
            DimList(3, 3), List[Float32](1, 2, 3, 4, 5, 6, 7, 8, 9)
        )
        var indices = TestTensor[DType.int32, 2](
            DimList(2, 3), List[Int32](1, 2, 0, 2, 0, 0)
        )
        var output_ref = TestTensor[DType.float32, 2](
            DimList(2, 3), List[Float32](4, 8, 3, 7, 2, 3)
        )

        test_case[DType.float32](0, data, indices, output_ref)

    # CHECK-LABEL: test_gather_ax0
    # CHECK-NOT: FAIL
    test_gather_ax0()

    fn test_gather_neg_indices() raises:
        print("== test_gather_neg_indices")

        var data = TestTensor[DType.float32, 2](
            DimList(3, 3), List[Float32](1, 2, 3, 4, 5, 6, 7, 8, 9)
        )
        var indices = TestTensor[DType.int32, 2](
            DimList(2, 3), List[Int32](-1, -2, 0, -2, 0, 0)
        )
        var output_ref = TestTensor[DType.float32, 2](
            DimList(2, 3), List[Float32](7, 5, 3, 4, 2, 3)
        )

        test_case[DType.float32](0, data, indices, output_ref)

    # CHECK-LABEL: test_gather_neg_indices
    # CHECK-NOT: FAIL
    test_gather_neg_indices()
