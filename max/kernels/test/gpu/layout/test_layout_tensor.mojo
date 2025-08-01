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

from buffer.dimlist import DimList
from layout import Layout, LayoutTensor
from internal_utils._utils import ValOrDim, dynamic, static
from utils.index import IndexList
from layout import RuntimeLayout
from testing import assert_equal


fn test_runtime_and_compile_time_dim_and_stride(
    m: ValOrDim, k: ValOrDim
) raises:
    alias static_shape = DimList(k.dim, m.dim)
    var dynamic_shape = IndexList[2](k.value, m.value)
    alias layout = Layout.row_major[2](static_shape)

    var tensor = LayoutTensor[DType.float32, layout,](
        UnsafePointer[Scalar[DType.float32]](),
        RuntimeLayout[layout].row_major(dynamic_shape),
    )

    assert_equal(tensor.dim(0), dynamic_shape[0])
    assert_equal(tensor.dim(1), dynamic_shape[1])
    assert_equal(tensor.stride(0), dynamic_shape[1])
    assert_equal(tensor.stride(1), 1)

    assert_equal(tensor.dim[0](), dynamic_shape[0])
    assert_equal(tensor.dim[1](), dynamic_shape[1])
    assert_equal(tensor.stride[0](), -1)
    assert_equal(tensor.stride[1](), 1)


def main():
    test_runtime_and_compile_time_dim_and_stride(dynamic(120), static[512]())
