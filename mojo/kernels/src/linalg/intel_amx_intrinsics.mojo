# ===----------------------------------------------------------------------===#
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------===#
#
# This file contains wrappers around Intel AMX intrinsics. See
# https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&techs=AMX
#
# ===----------------------------------------------------------------------===#

from DType import DType
from Int import Int
from Pointer import Pointer, DTypePointer
from Tuple import StaticTuple
from Matmul import Matrix
from List import create_kgen_list
from Bool import Bool
from Range import range
from SIMD import SIMD

alias void = DType.invalid.value


struct __tile:
    """An AMX tile representation"""

    var buf: __mlir_type[`!pop.array<1024, si32>`]
    var rows: SIMD[1, DType.si32.value]
    var cols: SIMD[1, DType.si32.value]


fn to_si8(
    val: Int,
) -> SIMD[1, DType.si8.value]:
    """Converts an input integer to an si8 value"""
    return __mlir_op.`pop.cast`[_type : __mlir_type.`!pop.scalar<si8>`](
        val.value
    )


fn _tile_dpbssd[dst: Int, a: Int, b: Int]():
    """
    Compute dot-product of bytes in tiles with a source/destination accumulator.
    Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with
    corresponding signed 8-bit integers in b, producing 4 intermediate 32-bit
    results. Sum these 4 results with the corresponding 32-bit integer in dst,
    and store the 32-bit result back to tile dst.

    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&ig_expand=7471,7472,7472&text=_tile_dpbssd
    """
    __mlir_op.`pop.call_llvm_intrinsic`[intrin:"llvm.x86.tdpbssd", _type:[],](
        __mlir_op.`pop.cast`[_type : __mlir_type.`!pop.scalar<si8>`](dst.value),
        __mlir_op.`pop.cast`[_type : __mlir_type.`!pop.scalar<si8>`](a.value),
        __mlir_op.`pop.cast`[_type : __mlir_type.`!pop.scalar<si8>`](b.value),
    )


fn _tile_release():
    """
    Compute dot-product of bytes in tiles with a source/destination accumulator.
    Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with
    corresponding signed 8-bit integers in b, producing 4 intermediate 32-bit
    results. Sum these 4 results with the corresponding 32-bit integer in dst,
    and store the 32-bit result back to tile dst.

    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&ig_expand=7471,7472,7472&text=_tile_release
    """

    __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.tilerelease",
        _type:[],
    ]()


fn _tile_zero[tdest: Int]():
    """
    Zero the tile specified by tdest

    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&ig_expand=7471,7472,7472&text=_tile_zero
    """

    __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.tilezero",
        _type:[],
    ](__mlir_op.`pop.cast`[_type : __mlir_type.`!pop.scalar<si8>`](tdest.value))


fn _tile_loadd[dst: Int](base: DTypePointer[void], stride: Int):
    """
    Load tile rows from memory specifieid by base address and stride into destination tile dst using the tile configuration previously configured via _tile_loadconfig.

    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&ig_expand=7471,7472,7472&text=_tile_loadd
    """
    __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.tileloadd64",
        _type:[],
    ](
        __mlir_op.`pop.cast`[_type : __mlir_type.`!pop.scalar<si8>`](dst.value),
        base,
        stride.value,
    )


fn _tile_stored[src: Int](base: DTypePointer[void], stride: Int):
    """
    Store the tile specified by src to memory specifieid by base address and stride using the tile configuration previously configured via _tile_loadconfig.

    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&ig_expand=7471,7472,7472&text=_tile_stored
    """
    __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.tilestored64",
        _type:[],
    ](
        __mlir_op.`pop.cast`[_type : __mlir_type.`!pop.scalar<si8>`](src.value),
        base,
        stride,
    )


fn _tile_loadconfig(mem_addr: DTypePointer[void]):
    """
    Load tile configuration from a 64-byte memory location specified by mem_addr. The tile configuration format is specified below, and includes the tile type pallette, the number of bytes per row, and the number of rows. If the specified pallette_id is zero, that signifies the init state for both the tile config and the tile data, and the tiles are zeroed. Any invalid configurations will result in #GP fault.
    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&ig_expand=7471,7472,7472&text=_tile_loadconfig
    """
    __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.ldtilecfg",
        _type:[],
    ](mem_addr)


fn _tile_storeconfig(mem_addr: DTypePointer[void]):
    """
    Stores the current tile configuration to a 64-byte memory location specified by mem_addr. The tile configuration format is specified below, and includes the tile type pallette, the number of bytes per row, and the number of rows. If tiles are not configured, all zeroes will be stored to memory.
    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&ig_expand=7471,7472,7472&text=_tile_storeconfig
    """
    __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.sttilecfg",
        _type:[],
    ](mem_addr)


fn init_intel_amx() -> Bool:
    return __mlir_op.`pop.external_call`[
        func : __mlir_attr.`@KGEN_CompilerRT_Init_Intel_AMX`,
        _type : __mlir_type[`!pop.scalar<bool>`],
    ]()


# typedef struct tileconfig_t {
#  uint8_t palette_id;
#  uint8_t startRow;
#  uint8_t reserved[14];
#  uint16_t colb[16];
#  uint8_t rows[16];
# } tileconfig_t;
struct tileconfig:
    var palette_id: SIMD[1, DType.ui8.value]
    var start_row: SIMD[1, DType.ui8.value]
    var reserved: StaticTuple[14, __mlir_type.`!pop.scalar<ui8>`]
    var colb: StaticTuple[16, __mlir_type.`!pop.scalar<ui16>`]
    var rows: StaticTuple[16, __mlir_type.`!pop.scalar<ui8>`]


fn _tile_dpbssd_emulated(
    cptr: DTypePointer[DType.si32.value],
    aptr: DTypePointer[DType.si8.value],
    bptr: DTypePointer[DType.si8.value],
):
    let a = Matrix[
        create_kgen_list[__mlir_type.index](16, 64), DType.si8.value, False
    ](aptr.address)
    let b = Matrix[
        create_kgen_list[__mlir_type.index](16, 64), DType.si8.value, False
    ](bptr.address)
    let c = Matrix[
        create_kgen_list[__mlir_type.index](16, 16), DType.si32.value, False
    ](cptr.address)

    for i in range(16):
        for j in range(16):
            for l in range(16):
                let ai0 = a.__getitem__(i, 4 * l + 0).cast[DType.si32.value]()
                let ai1 = a.__getitem__(i, 4 * l + 1).cast[DType.si32.value]()
                let ai2 = a.__getitem__(i, 4 * l + 2).cast[DType.si32.value]()
                let ai3 = a.__getitem__(i, 4 * l + 3).cast[DType.si32.value]()
                let bi0 = b.__getitem__(l, 4 * j + 0).cast[DType.si32.value]()
                let bi1 = b.__getitem__(l, 4 * j + 1).cast[DType.si32.value]()
                let bi2 = b.__getitem__(l, 4 * j + 2).cast[DType.si32.value]()
                let bi3 = b.__getitem__(l, 4 * j + 3).cast[DType.si32.value]()
                var cv = c.__getitem__(i, j)
                cv += ai0 * bi0
                cv += ai1 * bi1
                cv += ai2 * bi2
                cv += ai3 * bi3
                c.__setitem__(i, j, cv)
