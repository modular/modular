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

from std.sys.info import simd_width_of

from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation as tt_stack_allocation
from linalg.packing import PackMatrixCols
from std.testing import assert_equal

from std.utils.index import Index

comptime type = DType.float32
comptime simd_size: Int = simd_width_of[DType.float32]()
comptime simd_cols: Int = 4
comptime kernel_cols: Int = simd_cols * simd_size
comptime width = 2 * kernel_cols

comptime N: Int = 128
comptime K: Int = 128
comptime kc = 128


def test_pack_b() raises:
    var packed_b = tt_stack_allocation[dtype=type,](
        row_major[width // kernel_cols, K, kernel_cols]()
    )
    for i in range(packed_b.num_elements()):
        packed_b.ptr[i] = 1.0

    var b = tt_stack_allocation[dtype=type,](row_major[K, N]())
    for i in range(b.num_elements()):
        b.ptr[i] = 1.0

    PackMatrixCols[
        type,
        simd_size,
        kernel_cols,
        False,  # use_vnni
        False,  # use_i8mm
        packed_b.origin,
        b.origin,
    ].run(
        packed_b,
        b,
        Index(0, 0),
        Index(kc, width),
        Index(K, N),
    )

    assert_equal(packed_b[0, 0, 0], 1.0)


def main() raises:
    test_pack_b()
