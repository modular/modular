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

from math import align_up
from sys import alignof, has_neon, has_vnni
from testing import assert_equal

from buffer import NDBuffer
from buffer.dimlist import DimList
from linalg.matmul import GemmShape, KernelConfig
from linalg.matmul_default import Inner_matmul_default
from linalg.matmul_i8mm import Inner_matmul_i8mm
from linalg.matmul_neon import Inner_matmul_neon
from linalg.matmul_vnni import Inner_matmul_vnni
from linalg.utils import (
    InnerKernelID,
    get_kernel_config,
    get_matmul_arch_factor,
    select_inner_kernel,
    use_i8mm_fn,
    use_vnni_fn,
)
from memory import UnsafePointer

from utils import IndexList
from utils.index import Index

alias M: Int = 64
alias N: Int = 64
alias K: Int = 256


fn _matmul_inner_loop[
    kernel_rows: Int,
    kernel_cols: Int,
    simd_size: Int,
    saturated_vnni: Bool,
](
    c: NDBuffer,
    a: NDBuffer,
    b_packed: NDBuffer[_, 3, _, _],
    global_offset: GemmShape,
    global_bound: GemmShape,
    tile_n_k: IndexList[2],
    skip_boundary_check: Bool,
):
    alias kernel_id = select_inner_kernel[a.type, b_packed.type, c.type]()

    @parameter
    if kernel_id == InnerKernelID.DEFAULT:
        Inner_matmul_default().__inner_matmul__[
            kernel_rows, kernel_cols, simd_size
        ](
            c,
            a,
            b_packed,
            global_offset,
            global_bound,
            tile_n_k,
            skip_boundary_check,
        )
    elif kernel_id == InnerKernelID.VNNI:
        Inner_matmul_vnni[saturated_vnni]().__inner_matmul__[
            kernel_rows, kernel_cols, simd_size
        ](
            c,
            a,
            b_packed,
            global_offset,
            global_bound,
            tile_n_k,
            skip_boundary_check,
        )
    elif kernel_id == InnerKernelID.NEON:
        Inner_matmul_neon().__inner_matmul__[
            kernel_rows, kernel_cols, simd_size
        ](
            c,
            a,
            b_packed,
            global_offset,
            global_bound,
            tile_n_k,
            skip_boundary_check,
        )
    elif kernel_id == InnerKernelID.I8MM:
        Inner_matmul_i8mm().__inner_matmul__[
            kernel_rows, kernel_cols, simd_size
        ](
            c,
            a,
            b_packed,
            global_offset,
            global_bound,
            tile_n_k,
            skip_boundary_check,
        )
    else:
        constrained[False, "no _run_inner_loop implementation"]()


fn matmul_inner_loop[
    config: KernelConfig,
](
    c: NDBuffer,
    a: NDBuffer,
    b_packed: NDBuffer[_, 3, _, _],
    m: Int,
    n: Int,
    k: Int,
):
    _matmul_inner_loop[
        config.kernel_rows,
        config.kernel_cols,
        config.simd_size,
        False,  # saturated_vnni
    ](
        c,
        a,
        b_packed,
        # Below are configurations for outer loops, just
        #  use the trivial numbers for now.
        GemmShape(0, 0, 0),  # Tile offset.
        GemmShape(m, n, k),  # Global tile dimension.
        Index(n, k),  # Local tile dimension.
        True,  # skip_boundary_check
    )


fn test_micro_kernel[
    a_type: DType, b_type: DType, c_type: DType, saturated_vnni: Bool = False
](m: Int, n: Int, k: Int) raises:
    print("== test_micro_kernel")
    alias a_shape = DimList.create_unknown[2]()
    alias b_shape = DimList.create_unknown[2]()
    alias c_shape = DimList.create_unknown[2]()
    alias b_packed_shape = DimList.create_unknown[3]()

    alias config = get_kernel_config[a_type, b_type, c_type]()
    alias use_vnni = use_vnni_fn[a_type, b_type, c_type]()
    alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
    alias factor = get_matmul_arch_factor[use_vnni, use_i8mm]()
    var np = align_up(n, config.kernel_cols)
    var kh = align_up(k, factor)

    alias alignment = alignof[SIMD[c_type, config.simd_size]]()

    var a_ptr = UnsafePointer[Scalar[a_type], alignment=alignment].alloc(m * k)
    var b_packed_ptr = UnsafePointer[Scalar[b_type], alignment=alignment].alloc(
        (np // config.kernel_cols)
        * (kh // factor)
        * (factor * config.kernel_cols)
    )
    var c_ptr = UnsafePointer[Scalar[c_type], alignment=alignment].alloc(m * n)
    var a = NDBuffer[a_type, 2, _, a_shape](a_ptr, Index(m, k))

    var b_packed = NDBuffer[b_type, 3, _, config.packed_shape](
        b_packed_ptr,
        Index(
            np // config.kernel_cols,
            kh // factor,
            factor * config.kernel_cols,
        ),
    )

    var c = NDBuffer[c_type, 2, _, c_shape](c_ptr, Index(m, n))

    a.fill(1)
    b_packed.fill(1)
    c.fill(0)

    matmul_inner_loop[config](c, a, b_packed, m, n, k)

    assert_equal(Int(c[0, 0]), 256)
    a_ptr.free()
    b_packed_ptr.free()
    c_ptr.free()


@export(ABI="C")
fn kernel_export_dynamic(m: Int, n: Int, k: Int) raises:
    test_micro_kernel[DType.float32, DType.float32, DType.float32](m, n, k)


fn main() raises:
    test_micro_kernel[DType.float32, DType.float32, DType.float32](M, N, K)
    test_micro_kernel[DType.uint8, DType.int8, DType.int32](M, N, K)
    test_micro_kernel[
        DType.uint8, DType.int8, DType.int32, saturated_vnni=True
    ](M, N, K)

    # TODO(KERN-228): Re-enable after we resolve llvm lowering issues.
    @parameter
    if not has_neon():
        test_micro_kernel[DType.bfloat16, DType.bfloat16, DType.bfloat16](
            M, N, K
        )
        test_micro_kernel[DType.bfloat16, DType.bfloat16, DType.float32](
            M, N, K
        )
