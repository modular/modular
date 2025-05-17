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

from tensor_internal import (
    foreach,
    DynamicTensor,
    ManagedTensorSlice,
    Tensor,
    TensorShape,
    TensorSpec,
)
from tensor_internal.managed_tensor_slice import StaticTensorSpec
from testing import assert_equal

from utils import Index, IndexList


fn test_foreach() raises:
    print("== test_foreach")
    var shape = (10, 2)

    var tensor1 = Tensor[DType.float32](TensorShape(shape))
    var unsafe_slice1 = DynamicTensor[DType.float32, 2].Type(
        tensor1.unsafe_ptr(), shape
    )

    var tensor2 = Tensor[DType.float32](TensorShape(shape))
    var unsafe_slice2 = DynamicTensor[DType.float32, 2].Type(
        tensor2.unsafe_ptr(), shape
    )

    # Testing in place modifications.
    @always_inline
    @parameter
    fn set_to_one[
        simd_width: Int
    ](idx: IndexList[2]) -> SIMD[DType.float32, simd_width]:
        return SIMD[DType.float32, simd_width](1)

    foreach[set_to_one](unsafe_slice1)

    for i in range(10):
        for j in range(2):
            assert_equal(unsafe_slice1[i, j], 1)
            assert_equal(tensor1[i, j], 1)

    # Testing the capture of unsafe_slice1.
    @always_inline
    @parameter
    fn double_it[
        simd_width: Int
    ](idx: IndexList[2]) -> SIMD[DType.float32, simd_width]:
        return 2 * unsafe_slice1.load[simd_width](idx)

    foreach[double_it](unsafe_slice2)

    for i in range(10):
        for j in range(2):
            assert_equal(unsafe_slice2[i, j], 2)
            assert_equal(tensor2[i, j], 2)

    _ = tensor1^
    _ = tensor2^


fn main() raises:
    test_foreach()
