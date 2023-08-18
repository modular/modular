# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file is only run on targets with Intel AMX and Linux.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: linux, intel_amx
# RUN: %mojo -debug-level full %s | FileCheck %s


from Pointer import Pointer, DTypePointer
from Buffer import Buffer, NDBuffer
from Transpose import transpose, transpose_inplace
from sys.info import os_is_linux, has_intel_amx
from Matmul import Matrix
from List import DimList, Dim
from Memory import memcmp, memset_zero
from Matmul import naive_matmul
from algorithm import unroll
from IntelAMX import _tile_loadconfig, _tile_storeconfig, _tile_release
from IntelAMX import _tile_zero, _tile_dpbssd, _tile_dpbssd_emulated
from IntelAMX import _tile_loadd, _tile_stored
from IntelAMX import init_intel_amx, tileconfig

alias void = DType.invalid.value
alias int32_pop = __mlir_type.`!pop.scalar<si32>`
alias int8_pop = __mlir_type.`!pop.scalar<si8>`


fn print_buffer[n: Int, type: DType](a_ptr: DTypePointer[void]):
    let a = Buffer[Dim(), type](a_ptr.bitcast[type](), n)
    for i in range(n):
        let v = __mlir_op.`pop.cast`[
            _type:int32_pop,
        ](a[i].value)
        print(v)


fn print_matrix[m: Int, n: Int, type: DType](a_ptr: DTypePointer[void]):
    let a = Buffer[Dim(), type](a_ptr.bitcast[type](), m * n)
    for i in range(m):
        print("row")
        for j in range(n):
            let ai = __mlir_op.`pop.cast`[
                _type:int32_pop,
            ](a[n * i + j].value)
            print(ai)


@always_inline
fn identity_epilogue_rowise_func[
    accum_type: DType
](row_idx: Int, row: Buffer[Dim(), accum_type]):
    pass


@always_inline
fn identity_epilogue_elemwise_func[
    accum_type: DType
](row: Int, col: Int, val: SIMD[accum_type, 1]) -> SIMD[accum_type, 1]:
    return val


fn init_matrices(
    a_ptr: DTypePointer[DType.int8],
    b_ptr: DTypePointer[DType.int8],
    c_ptr: DTypePointer[DType.int32],
    c2_ptr: DTypePointer[DType.int32],
):

    let a = Buffer[Dim(), DType.int8](a_ptr.address, 1024)
    let b = Buffer[Dim(), DType.int8](b_ptr.address, 1024)
    let c = Buffer[Dim(), DType.int32](c_ptr.address, 256)
    let c2 = Buffer[Dim(), DType.int32](c2_ptr.address, 256)
    let b2 = Buffer[1024, DType.int8].stack_allocation()

    for i in range(1024):
        a[i] = SIMD[DType.int8, 1](i & 127)
        b2[i] = SIMD[DType.int8, 1](i & 127)

    memset_zero[DType.int32](c.data, 1024)
    memset_zero[DType.int32](c2.data, 1024)

    let b2m = NDBuffer[2, DimList(64, 16), DType.int8](b2.data.address)
    let bm = NDBuffer[2, DimList(16, 64), DType.int8](b_ptr.address)
    # transpose from 64x16 to 16x64
    transpose[2, DimList(16, 64), DimList(64, 16), DType.int8](bm, b2m)

    let b32_ptr = b.data.bitcast[DType.int32]()
    let b32m = NDBuffer[2, DimList(16, 16), DType.int32](b32_ptr.address)
    transpose_inplace[16, 16, DType.int32](b32m)
    let am = NDBuffer[2, DimList(16, 64), DType.int8](a.data.address)
    let c2m = NDBuffer[2, DimList(16, 16), DType.int32](c2.data.address)
    naive_matmul[
        DimList(16, 64),
        DimList(64, 16),
        DimList(16, 16),
        DType.int32,
        DType.int8,
        False,
        False,
        identity_epilogue_elemwise_func,
        identity_epilogue_rowise_func,
    ](c2m, am, b2m)


fn setup_tile_config() -> tileconfig:
    var tc: tileconfig
    let ptr = Pointer.address_of(tc)
    let tc_ptr = DTypePointer[DType.int8](ptr.bitcast[int8_pop]().address)
    memset_zero(tc_ptr, 64)

    let nrows: SIMD[DType.uint8, 1] = 16
    let colb: SIMD[DType.uint16, 1] = 64

    tc.palette_id = 1

    @always_inline
    fn tc_fill[idx: Int]():
        tc.rows.__setitem__[idx](nrows.value)
        tc.colb.__setitem__[idx](colb.value)

    unroll[8, tc_fill]()
    return tc


fn main():
    let a = Buffer[1024, DType.int8].stack_allocation()
    let b = Buffer[1024, DType.int8].stack_allocation()
    let c = Buffer[256, DType.int32].stack_allocation()
    let c2 = Buffer[256, DType.int32].stack_allocation()

    init_matrices(a.data, b.data, c.data, c2.data)
    # print_matrix[16, 64, DType.int8](b.data.bitcast[void]())

    _tile_dpbssd_emulated(c.data, a.data, b.data)
    # print_matrix[16, 16, DType.int32](c.data.bitcast[void]())
    var errors: Int = 0
    errors = memcmp(c.data, c2.data, c.__len__())
    print("Emulated AMX-int8 matmul test.")
    # CHECK: 0
    print(errors)
    if errors != 0:
        print("Matrices don't agree!")
    memset_zero[DType.int32](c.data, 1024)
    if os_is_linux() and has_intel_amx() and init_intel_amx():
        print("Hardware AMX-int8 matmul test.")
        var tc = setup_tile_config()
        let ptr = Pointer[tileconfig].address_of(tc)
        let tc_ptr = DTypePointer[void](
            ptr.bitcast[__mlir_type.`!pop.scalar<invalid>`]().address
        )

        _tile_loadconfig(tc_ptr)
        # _tile_storeconfig(tc_ptr)
        _tile_zero[0]()
        _tile_loadd[1](a.data.bitcast[void](), 64)
        _tile_loadd[2](b.data.bitcast[void](), 64)
        _tile_dpbssd[0, 1, 2]()
        _tile_stored[0](c.data.bitcast[void](), 64)
        _tile_release()

        errors = memcmp(
            c.data.bitcast[void](), c2.data.bitcast[void](), c.__len__()
        )
    # CHECK: 0
    print(errors)
    if errors != 0:
        print("\nMatrices don't agree!")
