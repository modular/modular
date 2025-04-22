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
# RUN: %mojo-no-debug %s | FileCheck %s

from sys.info import simdwidthof

from buffer import NDBuffer
from buffer.dimlist import DimList
from linalg.packing import PackMatrixCols

from utils.index import Index

alias type = DType.float32
alias simd_size: Int = simdwidthof[DType.float32]()
alias simd_cols: Int = 4
alias kernel_cols: Int = simd_cols * simd_size
alias width = 2 * kernel_cols

alias N: Int = 128
alias K: Int = 128
alias kc = 128


@export(ABI="C")
fn pack_b(
    packed_b: NDBuffer[
        type, 3, MutableAnyOrigin, DimList(width // kernel_cols, K, kernel_cols)
    ],
    b: NDBuffer[type, 2, MutableAnyOrigin, DimList(K, N)],
):
    PackMatrixCols[
        DimList(K, N),
        DimList(width // kernel_cols, K, kernel_cols),
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


fn test_pack_b():
    var packed_b = NDBuffer[
        type, 3, MutableAnyOrigin, DimList(width // kernel_cols, K, kernel_cols)
    ].stack_allocation[alignment=64]()
    packed_b.fill(1)
    var b = NDBuffer[type, 2, MutableAnyOrigin, DimList(K, N)].stack_allocation[
        alignment=64
    ]()
    b.fill(1)
    pack_b(packed_b, b)

    # CHECK: 1.0
    print(packed_b[0, 0, 0])


fn main():
    test_pack_b()
