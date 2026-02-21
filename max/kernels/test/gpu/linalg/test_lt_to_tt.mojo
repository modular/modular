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
"""Unit tests for lt_to_tt (LayoutTensor to TileTensor conversion)."""

from buffer import Dim, DimList
from layout import Layout, LayoutTensor, RuntimeLayout
from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_types import (
    lt_to_tt,
)
from testing import assert_equal
from utils.index import Index


fn test_static_layout() raises:
    """Static 2D layout: both dims known at compile time."""
    print("--- test_static_layout ---")
    comptime static_layout = Layout.row_major(4, 8)

    var array = InlineArray[Float32, 32](fill=1.0)
    var lt = LayoutTensor[DType.float32, static_layout](array.unsafe_ptr())

    var tt = lt_to_tt(lt)
    _ = tt

    assert_equal(tt.rank, 2)
    print("  PASSED")


fn test_dynamic_layout() raises:
    """Dynamic 2D layout: first dim unknown at compile time."""
    print("--- test_dynamic_layout ---")
    comptime dynamic_layout = Layout.row_major[2](DimList(Dim(), Dim(8)))

    var array = InlineArray[Float32, 32](fill=2.0)
    var lt = LayoutTensor[DType.float32, dynamic_layout](
        array.unsafe_ptr(),
        RuntimeLayout[dynamic_layout].row_major(Index(4, 8)),
    )

    var tt = lt_to_tt(lt)
    _ = tt

    assert_equal(tt.rank, 2)
    print("  PASSED")


fn test_fully_dynamic_layout() raises:
    """Fully dynamic 2D layout: both dims unknown at compile time."""
    print("--- test_fully_dynamic_layout ---")
    comptime dynamic_layout = Layout.row_major[2](DimList(Dim(), Dim()))

    var array = InlineArray[Float32, 24](fill=3.0)
    var lt = LayoutTensor[DType.float32, dynamic_layout](
        array.unsafe_ptr(),
        RuntimeLayout[dynamic_layout].row_major(Index(3, 8)),
    )

    var tt = lt_to_tt(lt)
    _ = tt

    assert_equal(tt.rank, 2)
    print("  PASSED")


def main():
    test_static_layout()
    test_dynamic_layout()
    test_fully_dynamic_layout()
    print("=== ALL TESTS PASSED ===")
