# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo %s | FileCheck %s

from math import align_up
from sys.info import has_neon, has_vnni

from buffer import NDBuffer
from buffer.list import DimList
from LinAlg.Matmul_v2 import GemmShape, MatmulConfig

from LinAlg.matmul_v2_vnni import Inner_matmul_vnni
from LinAlg.matmul_v2_i8mm import Inner_matmul_i8mm
from LinAlg.matmul_v2_neon import Inner_matmul_neon
from LinAlg.matmul_v2_default import Inner_matmul_default
from LinAlg.MatmulUtils import (
    get_matmul_arch_factor,
    get_mm_config,
    InnerKernelID,
    select_inner_kernel,
)

from utils.index import Index

alias M: Int = 64
alias N: Int = 64
alias K: Int = 256


fn _matmul_inner_loop[
    a_row_size: Int,
    pack_inner_size: Int,
    skip_col_bound: Bool,
    saturated_vnni: Bool,
](
    c: NDBuffer,
    a: NDBuffer,
    b_packed: NDBuffer[_, 3, _],
    global_offset: GemmShape,
    global_bound: GemmShape,
    tile_n_k: StaticIntTuple[2],
):
    alias kernel_id = select_inner_kernel[a.type, b_packed.type, c.type]()

    @parameter
    if kernel_id == InnerKernelID.DEFAULT:
        Inner_matmul_default().__inner_matmul__[
            a_row_size, pack_inner_size, skip_col_bound
        ](c, a, b_packed, global_offset, global_bound, tile_n_k)
    elif kernel_id == InnerKernelID.VNNI:
        Inner_matmul_vnni[saturated_vnni]().__inner_matmul__[
            a_row_size, pack_inner_size, skip_col_bound
        ](c, a, b_packed, global_offset, global_bound, tile_n_k)
    elif kernel_id == InnerKernelID.NEON:
        Inner_matmul_neon().__inner_matmul__[
            a_row_size, pack_inner_size, skip_col_bound
        ](c, a, b_packed, global_offset, global_bound, tile_n_k)
    elif kernel_id == InnerKernelID.I8MM:
        Inner_matmul_i8mm().__inner_matmul__[
            a_row_size, pack_inner_size, skip_col_bound
        ](c, a, b_packed, global_offset, global_bound, tile_n_k)
    else:
        constrained[False, "no _run_inner_loop implementation"]()


fn matmul_inner_loop[
    config: MatmulConfig,
](
    c: NDBuffer,
    a: NDBuffer,
    b_packed: NDBuffer[_, 3, _],
    m: Int,
    n: Int,
    k: Int,
):
    _matmul_inner_loop[
        config.a_row_size,
        config.pack_inner_size,
        True,  # skip_col_bound
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
    )


# CHECK-LABEL: test_micro_kernel
fn test_micro_kernel[
    a_type: DType, b_type: DType, c_type: DType, saturated_vnni: Bool = False
](m: Int, n: Int, k: Int):
    print("== test_micro_kernel")
    alias a_shape = DimList.create_unknown[2]()
    alias b_shape = DimList.create_unknown[2]()
    alias c_shape = DimList.create_unknown[2]()
    alias b_packed_shape = DimList.create_unknown[3]()

    alias config = get_mm_config[
        a_type,
        a_shape,
        b_type,
        b_shape,
        c_type,
        c_shape,
        transpose_b=False,
        b_packed=True,
        kernel_type=False,
        saturated_vnni=saturated_vnni,
    ]()
    alias factor = get_matmul_arch_factor[config.use_vnni, config.use_i8mm]()
    var np = align_up(n, config.pack_inner_size)
    var kh = align_up(k, factor)

    alias alignment = alignof[SIMD[c_type, config.simd_size]]()

    var a_ptr = DTypePointer[config.a_type].alloc(m * k, alignment=alignment)
    var b_packed_ptr = DTypePointer[config.b_type].alloc(
        (np // config.pack_inner_size)
        * (kh // factor)
        * (factor * config.pack_inner_size),
        alignment=alignment,
    )
    var c_ptr = DTypePointer[config.c_type].alloc(m * n, alignment=alignment)
    var a = NDBuffer[config.a_type, 2, config.a_shape](a_ptr, Index(m, k))

    var b_packed = NDBuffer[config.b_type, 3, config.packed_shape](
        b_packed_ptr,
        Index(
            np // config.pack_inner_size,
            kh // factor,
            factor * config.pack_inner_size,
        ),
    )

    var c = NDBuffer[config.c_type, 2, config.c_shape](c_ptr, Index(m, n))

    a.fill(1)
    b_packed.fill(1)
    c.fill(0)

    matmul_inner_loop[config](c, a, b_packed, m, n, k)

    # CHECK: 256
    print(int(c[0, 0]))
    a_ptr.free()
    b_packed_ptr.free()
    c_ptr.free()


@export(ABI="C")
fn kernel_export_dynamic(m: Int, n: Int, k: Int):
    test_micro_kernel[DType.float32, DType.float32, DType.float32](m, n, k)


fn main():
    test_micro_kernel[DType.float32, DType.float32, DType.float32](M, N, K)
    test_micro_kernel[DType.uint8, DType.int8, DType.int32](M, N, K)
    test_micro_kernel[
        DType.uint8, DType.int8, DType.int32, saturated_vnni=True
    ](M, N, K)

    # TODO(30525): Re-enable after we resolve llvm lowering issues.
    @parameter
    if not has_neon():
        test_micro_kernel[DType.bfloat16, DType.bfloat16, DType.bfloat16](
            M, N, K
        )
        test_micro_kernel[DType.bfloat16, DType.bfloat16, DType.float32](
            M, N, K
        )
