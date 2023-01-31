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
# RUN: kgen %s -execute -func='$test_intel_amx::main():index()' -I %stdlibdir | FileCheck %s

from DType import DType
from Bool import Bool
from Int import Int
from SIMD import SIMD
from Pointer import Pointer, DTypePointer
from Buffer import Buffer
from Tuple import StaticTuple
from Buffer import NDBuffer
from Transpose import transpose, transpose_inplace
from IO import put, print
from TargetInfo import os_is_linux, has_intel_amx
from Matmul import Matrix
from List import create_kgen_list
from Memory import memcmp, memset_zero
from Matmul import naive_matmul
from Functional import unroll
from IntelAMX import _tile_loadconfig, _tile_storeconfig, _tile_release
from IntelAMX import _tile_zero, _tile_dpbssd, _tile_dpbssd_emulated
from IntelAMX import _tile_loadd, _tile_stored
from IntelAMX import init_intel_amx, tileconfig

alias void = DType.invalid.value
alias int32_pop = __mlir_type.`!pop.scalar<si32>`
alias int8_pop = __mlir_type.`!pop.scalar<si8>`
alias kunknown = __mlir_attr.`#kgen.unknown : index`


fn print_buffer[
    n: Int, type: __mlir_type.`!kgen.dtype`
](a_ptr: DTypePointer[void]):
    let a = Buffer[kunknown, type](a_ptr.bitcast[type]().address, n)
    var i: Int = 0
    while i < n:
        let v = __mlir_op.`pop.cast`[
            _type:int32_pop,
        ](a.__getitem__(i).value)
        print(v)
        i += 1


fn print_matrix[
    m: Int, n: Int, type: __mlir_type.`!kgen.dtype`
](a_ptr: DTypePointer[void]):
    let a = Buffer[kunknown, type](a_ptr.bitcast[type]().address, m * n)
    var i: Int = 0
    while i < m:
        var j: Int = 0
        print("row\n")
        while j < n:
            let ai = __mlir_op.`pop.cast`[
                _type:int32_pop,
            ](a.__getitem__(n * i + j).value)
            print(ai)
            j += 1
        i += 1


@always_inline
fn identity_epilogue_rowise_func[
    accum_type: __mlir_type.`!kgen.dtype`
](row_idx: Int, row: Buffer[__mlir_attr.`#kgen.unknown : index`, accum_type],):
    pass


@always_inline
fn identity_epilogue_elemwise_func[
    accum_type: __mlir_type.`!kgen.dtype`
](row: Int, col: Int, val: SIMD[1, accum_type]) -> SIMD[1, accum_type]:
    return val


fn init_matrices(
    a_ptr: DTypePointer[DType.si8.value],
    b_ptr: DTypePointer[DType.si8.value],
    c_ptr: DTypePointer[DType.si32.value],
    c2_ptr: DTypePointer[DType.si32.value],
):

    let a = Buffer[kunknown, DType.si8.value](a_ptr.address, 1024)
    let b = Buffer[kunknown, DType.si8.value](b_ptr.address, 1024)
    let c = Buffer[kunknown, DType.si32.value](c_ptr.address, 256)
    let c2 = Buffer[kunknown, DType.si32.value](c2_ptr.address, 256)
    let b2 = Buffer[1024, DType.si8.value].stack_allocation()

    var i: Int = 0
    while i < 1024:
        a.__setitem__(i, SIMD[1, DType.si8.value](i & 127))
        b2.__setitem__(i, SIMD[1, DType.si8.value](i & 127))
        i += 1

    memset_zero[DType.si32.value](c.data, 1024)
    memset_zero[DType.si32.value](c2.data, 1024)

    let b2m = NDBuffer[
        2, create_kgen_list[__mlir_type.index](64, 16), DType.si8.value
    ](b2.data.address)
    let bm = NDBuffer[
        2, create_kgen_list[__mlir_type.index](16, 64), DType.si8.value
    ](b_ptr.address)
    # transpose from 64x16 to 16x64
    transpose[2, 16, 64, DType.si8.value](bm, b2m)

    let b32_ptr = b.data.bitcast[DType.si32.value]()
    let b32m = NDBuffer[
        2, create_kgen_list[__mlir_type.index](16, 16), DType.si32.value
    ](b32_ptr.address)
    transpose_inplace[2, 16, 16, DType.si32.value](b32m)
    let am = NDBuffer[
        2, create_kgen_list[__mlir_type.index](16, 64), DType.si8.value
    ](a.data.address)
    let c2m = NDBuffer[
        2, create_kgen_list[__mlir_type.index](16, 16), DType.si32.value
    ](c2.data.address)
    naive_matmul[
        create_kgen_list[__mlir_type.index](16, 64),
        create_kgen_list[__mlir_type.index](64, 16),
        create_kgen_list[__mlir_type.index](16, 16),
        DType.si32.value,
        DType.si8.value,
        False,
        False,
        identity_epilogue_elemwise_func,
        identity_epilogue_rowise_func,
    ](c2m, am, b2m)


fn setup_tile_config() -> tileconfig:
    var tc: tileconfig
    let ptr = Pointer.address_of(tc)
    let tc_ptr = DTypePointer[DType.si8.value](ptr.bitcast[int8_pop]().address)
    memset_zero(tc_ptr, 64)

    let nrows: SIMD[1, DType.ui8.value] = 16
    let colb: SIMD[1, DType.ui16.value] = 64

    tc.palette_id = 1

    @always_inline
    fn tc_fill[idx: __mlir_type.index]():
        tc.rows.__setitem__[idx](nrows.value)
        tc.colb.__setitem__[idx](colb.value)

    unroll[8, tc_fill]()
    return tc


@export
fn main() -> __mlir_type.index:
    let a = Buffer[1024, DType.si8.value].stack_allocation()
    let b = Buffer[1024, DType.si8.value].stack_allocation()
    let c = Buffer[256, DType.si32.value].stack_allocation()
    let c2 = Buffer[256, DType.si32.value].stack_allocation()

    init_matrices(a.data, b.data, c.data, c2.data)
    # print_matrix[16, 64, DType.si8.value](b.data.bitcast[void]())

    _tile_dpbssd_emulated(c.data, a.data, b.data)
    # print_matrix[16, 16, DType.si32.value](c.data.bitcast[void]())
    var errors: Int = 0
    errors = memcmp(c.data.bitcast[void](), c2.data.bitcast[void](), 1024)
    print("Emulated AMX-int8 matmul test.\n")
    # CHECK: 0
    print(errors)
    if errors != 0:
        print("Matrices don't agree!\n\n")
    memset_zero[DType.si32.value](c.data, 1024)
    let is_linux: Bool = os_is_linux()
    if is_linux & has_intel_amx() & init_intel_amx():
        print("Hardware AMX-int8 matmul test.\n")
        var tc = setup_tile_config()
        let ptr = Pointer[tileconfig].address_of(tc)
        let tc_ptr = DTypePointer[void](
            ptr.bitcast[__mlir_type.`!pop.scalar<invalid>`]().address
        )

        _tile_loadconfig(tc_ptr)
        # _tile_storeconfig(tc_ptr)
        _tile_zero[0]
        _tile_loadd[1](a.data.bitcast[void](), 64)
        _tile_loadd[2](b.data.bitcast[void](), 64)
        _tile_dpbssd[0, 1, 2]()
        _tile_stored[0](c.data.bitcast[void](), 64)
        _tile_release()

        errors = memcmp(c.data.bitcast[void](), c2.data.bitcast[void](), 1024)
    # CHECK: 0
    print(errors)
    if errors != 0:
        print("\nMatrices don't agree!\n\n")

    return 0
