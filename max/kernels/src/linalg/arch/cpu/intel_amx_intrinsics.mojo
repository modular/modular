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
#
# This file contains wrappers around Intel AMX intrinsics. See
# https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&techs=AMX
#
# ===-----------------------------------------------------------------------===#

from sys import llvm_intrinsic

from buffer import NDBuffer
from buffer.dimlist import DimList

from utils import Index, StaticTuple

alias void = DType.invalid._mlir_value


struct __tile:
    """An AMX tile representation"""

    var buf: __mlir_type[`!pop.array<1024, si32>`]
    var rows: Int32
    var cols: Int32


fn _tile_dpbssd[dst: Int, a: Int, b: Int]():
    """
    Compute dot-product of bytes in tiles with a source/destination accumulator.
    Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with
    corresponding signed 8-bit integers in b, producing 4 intermediate 32-bit
    results. Sum these 4 results with the corresponding 32-bit integer in dst,
    and store the 32-bit result back to tile dst.

    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&ig_expand=7471,7472,7472&text=_tile_dpbssd
    """
    llvm_intrinsic["llvm.x86.tdpbssd", NoneType](Int8(dst), Int8(a), Int8(b))


fn _tile_release():
    """
    Compute dot-product of bytes in tiles with a source/destination accumulator.
    Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with
    corresponding signed 8-bit integers in b, producing 4 intermediate 32-bit
    results. Sum these 4 results with the corresponding 32-bit integer in dst,
    and store the 32-bit result back to tile dst.

    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&ig_expand=7471,7472,7472&text=_tile_release
    """

    llvm_intrinsic["llvm.x86.tilerelease", NoneType]()


fn _tile_zero[tdest: Int]():
    """
    Zero the tile specified by tdest

    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&ig_expand=7471,7472,7472&text=_tile_zero
    """

    llvm_intrinsic["llvm.x86.tilezero", NoneType](Int8(tdest))


fn _tile_loadd[dst: Int](base: OpaquePointer, stride: Int):
    """
    Load tile rows from memory specifieid by base address and stride into destination tile dst using the tile configuration previously configured via _tile_loadconfig.

    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&ig_expand=7471,7472,7472&text=_tile_loadd
    """
    llvm_intrinsic["llvm.x86.tileloadd64", NoneType](
        Int8(dst), base, stride._mlir_value
    )


fn _tile_stored[src: Int](base: OpaquePointer, stride: Int):
    """
    Store the tile specified by src to memory specifieid by base address and stride using the tile configuration previously configured via _tile_loadconfig.

    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&ig_expand=7471,7472,7472&text=_tile_stored
    """
    llvm_intrinsic["llvm.x86.tilestored64", NoneType](Int8(src), base, stride)


fn _tile_loadconfig(mem_addr: OpaquePointer):
    """
    Load tile configuration from a 64-byte memory location specified by mem_addr. The tile configuration format is specified below, and includes the tile type palvarte, the number of bytes per row, and the number of rows. If the specified palvarte_id is zero, that signifies the init state for both the tile config and the tile data, and the tiles are zeroed. Any invalid configurations will result in #GP fault.
    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&ig_expand=7471,7472,7472&text=_tile_loadconfig
    """
    llvm_intrinsic["llvm.x86.ldtilecfg", NoneType](mem_addr)


fn _tile_storeconfig(mem_addr: OpaquePointer):
    """
    Stores the current tile configuration to a 64-byte memory location specified by mem_addr. The tile configuration format is specified below, and includes the tile type palvarte, the number of bytes per row, and the number of rows. If tiles are not configured, all zeroes will be stored to memory.
    See https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&ig_expand=7471,7472,7472&text=_tile_storeconfig
    """
    llvm_intrinsic["llvm.x86.sttilecfg", NoneType](mem_addr)


fn init_intel_amx() -> Bool:
    return __mlir_op.`pop.external_call`[
        func = "KGEN_CompilerRT_Init_Intel_AMX".value,
        _type=Bool,
    ]()


# typedef struct tileconfig_t {
#  uint8_t pavarte_id;
#  uint8_t startRow;
#  uint8_t reserved[14];
#  uint16_t colb[16];
#  uint8_t rows[16];
# } tileconfig_t;
struct tileconfig:
    var pavarte_id: UInt8
    var start_row: UInt8
    var reserved: StaticTuple[__mlir_type.`!pop.scalar<ui8>`, 14]
    var colb: StaticTuple[__mlir_type.`!pop.scalar<ui16>`, 16]
    var rows: StaticTuple[__mlir_type.`!pop.scalar<ui8>`, 16]


fn _tile_dpbssd_emulated(
    cptr: UnsafePointer[Int32],
    aptr: UnsafePointer[Int8],
    bptr: UnsafePointer[Int8],
):
    var a = NDBuffer[DType.int8, 2, _, DimList(16, 64)](aptr)
    var b = NDBuffer[DType.int8, 2, _, DimList(16, 64)](bptr)
    var c = NDBuffer[DType.int32, 2, _, DimList(16, 16)](cptr)

    for i in range(16):
        for j in range(16):
            for l in range(16):
                var ai0 = a[i, 4 * l + 0].cast[DType.int32]()
                var ai1 = a[i, 4 * l + 1].cast[DType.int32]()
                var ai2 = a[i, 4 * l + 2].cast[DType.int32]()
                var ai3 = a[i, 4 * l + 3].cast[DType.int32]()
                var bi0 = b[l, 4 * j + 0].cast[DType.int32]()
                var bi1 = b[l, 4 * j + 1].cast[DType.int32]()
                var bi2 = b[l, 4 * j + 2].cast[DType.int32]()
                var bi3 = b[l, 4 * j + 3].cast[DType.int32]()
                var cv = c[Index(i, j)]
                cv += ai0 * bi0
                cv += ai1 * bi1
                cv += ai2 * bi2
                cv += ai3 * bi3
                c[Index(i, j)] = cv
