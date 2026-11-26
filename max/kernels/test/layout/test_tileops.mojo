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
"""Host tests for the load and store TileOps: per-thread over pointer atoms,
and warp-collective over the TMEM atoms, compiled for sm_100a."""

from std.sys.info import _is_sm_100x_or_newer
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from layout import TileTensor, col_major, coord, row_major
from layout.coord import ComptimeInt, Coord
from layout.atoms.pointer import PointerLoad, PointerStore
from layout.atoms.tmem import TMemLoad, TMemStore
from layout.bit_layout import is_bit_linear, placement
from layout.building_blocks import wait_issues
from layout.owned_tensor import OwnedGPUTensor
from layout.bit_layout import BitLayout
from layout.two_level_layout import (
    BitTensorLayout,
    SwizzledTwoLevelLayout,
    bit_tensor_layout,
    two_level_layout_from_swizzle,
)
from layout.tensor_engine import DefaultEngine
from layout.layout import Layout as TileLayout
from layout.tileops import (
    _base_clear,
    _plan,
    can_tiled_load,
    can_tiled_store,
    tiled_load,
    tiled_store,
)
from layout.tmem_engine import TMEM_NUM_LANES, TMemEngine, TMemStorage

from max.gpu import thread_idx
from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code

comptime RowMajor = type_of(row_major[4, 8]())
comptime ColMajor = type_of(col_major[4, 8]())
comptime Mem = DefaultEngine[element_width=1]

comptime AccumLayout = TileLayout(
    shape=coord[128, 32], stride=coord[1, TMEM_NUM_LANES]
)
comptime Accum = type_of(AccumLayout)
comptime Row32 = type_of(row_major[1, 32]())


def test_placement() raises:
    # Row-major 4 x 8: row bits reach offsets 8 and 16, column bits 1, 2, 4.
    comptime m = placement[RowMajor]()
    assert_equal(m(UInt64(0b00001)), UInt64(8))
    assert_equal(m(UInt64(0b00010)), UInt64(16))
    assert_equal(m(UInt64(0b00100)), UInt64(1))
    assert_equal(m(UInt64(0b10000)), UInt64(4))
    assert_true(is_bit_linear[RowMajor]())
    assert_false(is_bit_linear[type_of(row_major[3, 8]())]())


def test_can_issue() raises:
    comptime Wide = PointerLoad[.float32, 4]
    comptime Narrow = PointerLoad[.float32, 1]
    assert_true(can_tiled_load[Wide, RowMajor, RowMajor, Mem]())
    # Four adjacent registers are four adjacent elements of a row, but in a
    # column-major tile those are 4 elements apart.
    assert_false(can_tiled_load[Wide, RowMajor, ColMajor, Mem]())
    assert_true(can_tiled_load[Narrow, RowMajor, ColMajor, Mem]())
    assert_true(
        can_tiled_store[PointerStore[.float32, 4], RowMajor, Mem, RowMajor]()
    )


def test_spread_across_threads() raises:
    # Eight threads share a 4 x 32 tile with 4-wide loads. Column bits 0
    # and 1 pick the element within one load, and the thread index takes
    # column bits 2 to 4, so thread t holds columns 4t to 4t + 3 of every
    # row. Each thread issues once per row.
    comptime Tile = type_of(row_major[4, 32]())
    comptime Mine = type_of(row_major[4, 4]())
    comptime Wide = PointerLoad[.float32, 4]
    assert_true(can_tiled_load[Wide, Mine, Tile, Mem, threads=8]())
    assert_false(can_tiled_load[Wide, Mine, Tile, Mem, threads=4]())
    var mem = OwnedGPUTensor[.float32, .GENERIC, alignment=16](
        row_major[4, 32]()
    )
    var out = OwnedGPUTensor[.float32, .GENERIC, alignment=16](
        row_major[4, 32]()
    )
    for r in range(4):
        for c in range(32):
            TileTensor(mem)[r, c] = Float32(r * 32 + c)
            TileTensor(out)[r, c] = -1.0
    for t in range(8):
        var regs = OwnedGPUTensor[.float32, .GENERIC, alignment=16](
            row_major[4, 4]()
        )
        tiled_load[threads=8](Wide(), TileTensor(regs), TileTensor(mem), t)
        for r in range(4):
            for v in range(4):
                assert_equal(
                    TileTensor(regs)[r, v][0], Float32(r * 32 + 4 * t + v)
                )
        tiled_store[threads=8](
            PointerStore[.float32, 4](), TileTensor(out), TileTensor(regs), t
        )
    for r in range(4):
        for c in range(32):
            assert_equal(TileTensor(out)[r, c][0], Float32(r * 32 + c))


comptime Plain8x32 = type_of(row_major[8, 32]())
comptime Swizzled8x32 = SwizzledTwoLevelLayout[Plain8x32, 3, 2, 3]
"""8 rows of 32 fp32 (128 bytes): the 16-byte chunks of row r are XORed
with r % 8, the 128-byte shared memory swizzle."""


def _swizzled_offset(r: Int, c: Int) -> Int:
    return r * 32 + ((c // 4) ^ (r % 8)) * 4 + c % 4


def test_swizzled_placement() raises:
    assert_true(is_bit_linear[Swizzled8x32]())
    comptime m = placement[Swizzled8x32]()
    for r in range(8):
        for c in range(32):
            # Mode 0, the row, holds the low logical bits.
            var x = UInt64(r + 8 * c)
            assert_equal(m(x), UInt64(_swizzled_offset(r, c)))


def test_load_swizzled_tile() raises:
    var storage = OwnedGPUTensor[.float32, .GENERIC, alignment=16](
        row_major[8, 32]()
    )
    var swz = TileTensor(storage).reshape(
        two_level_layout_from_swizzle[Plain8x32, 3, 2, 3](row_major[8, 32]()),
    )
    for r in range(8):
        for c in range(32):
            swz[r, c] = Float32(r * 32 + c)
    # The swizzle only moves whole 16-byte chunks, so 4-wide accesses fit.
    comptime Wide = PointerLoad[.float32, 4]
    assert_true(can_tiled_load[Wide, Plain8x32, Swizzled8x32, Mem]())
    var regs = OwnedGPUTensor[.float32, .GENERIC, alignment=16](
        row_major[8, 32]()
    )
    tiled_load(Wide(), TileTensor(regs), swz)
    for r in range(8):
        for c in range(32):
            assert_equal(TileTensor(regs)[r, c][0], Float32(r * 32 + c))
    # Spread over 8 threads, each thread loads one 4-element chunk per row.
    var out = OwnedGPUTensor[.float32, .GENERIC, alignment=16](
        row_major[8, 32]()
    )
    for t in range(8):
        var mine = OwnedGPUTensor[.float32, .GENERIC, alignment=16](
            row_major[8, 4]()
        )
        tiled_load[threads=8](Wide(), TileTensor(mine), swz, t)
        tiled_store[threads=8](
            PointerStore[.float32, 4](), TileTensor(out), TileTensor(mine), t
        )
    for r in range(8):
        for c in range(32):
            assert_equal(TileTensor(out)[r, c][0], Float32(r * 32 + c))


comptime Row1 = Swizzled8x32._NestedTileLayout[
    Coord[ComptimeInt[1], ComptimeInt[32]].element_types
]
"""Row 1 of `Swizzled8x32`: its XOR base is the row's swizzle, 32 ^ 4 = 36."""


def test_load_swizzled_sub_tile() raises:
    var storage = OwnedGPUTensor[.float32, .GENERIC, alignment=16](
        row_major[8, 32]()
    )
    var full = two_level_layout_from_swizzle[Plain8x32, 3, 2, 3](
        row_major[8, 32]()
    )
    var swz = TileTensor(storage).reshape(full)
    for r in range(8):
        for c in range(32):
            swz[r, c] = Float32(r * 32 + c)
    var row = TileTensor(storage).reshape(
        Row1(rebind[Row1.Outer](full._outer), 36)
    )
    # The base sets offset bits 2 and 5. A 4-wide access covers bits 0 and
    # 1, so 8 threads split the row.
    comptime Wide = PointerLoad[.float32, 4]
    comptime Mine = type_of(row_major[1, 4]())
    assert_true(can_tiled_load[Wide, Mine, Row1, Mem, threads=8]())
    comptime wide = _plan[Wide.mem_map, Wide.reg_map, Row1, Mine, Mem, 4, 8]()
    assert_true(_base_clear(wide, row))
    for t in range(8):
        var regs = OwnedGPUTensor[.float32, .GENERIC, alignment=16](
            row_major[1, 4]()
        )
        tiled_load[threads=8](Wide(), TileTensor(regs), row, t)
        for v in range(4):
            assert_equal(TileTensor(regs)[0, v][0], Float32(32 + 4 * t + v))
    # An 8-wide access covers bits 0 to 2. The bit images fit, but the base
    # splits each access, so `tiled_load` refuses it.
    comptime Wider = PointerLoad[.float32, 8]
    comptime Row = type_of(row_major[1, 32]())
    assert_true(can_tiled_load[Wider, Row, Row1, Mem]())
    comptime wider = _plan[Wider.mem_map, Wider.reg_map, Row1, Row, Mem, 4, 1]()
    assert_false(_base_clear(wider, row))


def test_rejected_thread_counts() raises:
    comptime Tile = type_of(row_major[4, 32]())
    comptime Mine = type_of(row_major[4, 4]())
    comptime Wide = PointerLoad[.float32, 4]
    # Not a power of two.
    assert_false(can_tiled_load[Wide, Mine, Tile, Mem, threads=6]())
    # More than 16 thread index bits.
    assert_false(can_tiled_load[Wide, Mine, Tile, Mem, threads=1 << 17]())
    # 64 threads need 6 free logical bits; the tile has 5 after the vector.
    assert_false(
        can_tiled_load[
            Wide, type_of(row_major[1, 2]()), Tile, Mem, threads=64
        ]()
    )
    # 128 TMEM lanes need 2 warp bits above the 5 lane bits.
    assert_false(
        can_tiled_load[
            TMemLoad[.float32, 16], Row32, Accum, TMemEngine, threads=64
        ]()
    )


comptime _GapImages = BitLayout().run(0, 2, 0).put(2, 16)
"""Points 0 to 3 at offsets 0 to 3, points 4 to 7 at 16 to 19."""
comptime Gap = BitTensorLayout[_GapImages, type_of(row_major[8]())]


def _gap_offset(p: Int) -> Int:
    return p % 4 + 16 * (p // 4)


def test_bit_tensor_layout() raises:
    comptime same = placement[Gap]() == _GapImages
    assert_true(same)
    assert_true(is_bit_linear[Gap]())
    assert_equal(Gap.static_cosize, 20)
    var storage = OwnedGPUTensor[.float32, .GENERIC, alignment=8](
        row_major[32]()
    )
    for i in range(32):
        TileTensor(storage)[i] = Float32(i)
    var gap = TileTensor(storage).reshape(
        bit_tensor_layout[_GapImages, type_of(row_major[8]())](),
    )
    for i in range(8):
        assert_equal(gap[i][0], Float32(_gap_offset(i)))
    var regs = OwnedGPUTensor[.float32, .GENERIC, alignment=8](row_major[8]())
    tiled_load(PointerLoad[.float32, 2](), TileTensor(regs), gap)
    for i in range(8):
        assert_equal(TileTensor(regs)[i][0], Float32(_gap_offset(i)))


def test_bit_tensor_layout_xors() raises:
    # Images 1 and 3 overlap, so point 3 lands at 1 ^ 3 = 2, not 1 + 3.
    comptime images = BitLayout().put(0, 1).put(1, 3)
    var storage = OwnedGPUTensor[.float32, .GENERIC](row_major[4]())
    for i in range(4):
        TileTensor(storage)[i] = Float32(i)
    var view = TileTensor(storage).reshape(
        bit_tensor_layout[images, type_of(row_major[4]())](),
    )
    var expected: List[Float32] = [0, 1, 3, 2]
    for i in range(4):
        assert_equal(view[i][0], expected[i])


def test_bit_tensor_layout_rank_2() raises:
    # Mode 0 (4 rows) holds coordinate bits 0 and 1, mode 1 (8 columns)
    # bits 2 to 4. Column bit 2 also flips offset bit 3, a row bit.
    comptime images = (
        BitLayout().put(0, 8).put(1, 16).put(2, 1).put(3, 2).put(4, 4 | 8)
    )
    comptime Shape = type_of(row_major[4, 8]())
    comptime L = BitTensorLayout[images, Shape]
    comptime same = placement[L]() == images
    assert_true(same)
    var storage = OwnedGPUTensor[.float32, .GENERIC](row_major[32]())
    for i in range(32):
        TileTensor(storage)[i] = Float32(i)
    var view = TileTensor(storage).reshape(bit_tensor_layout[images, Shape]())
    for r in range(4):
        for c in range(8):
            var offset = images(UInt64(r + 4 * c))
            assert_equal(view[r, c][0], Float32(Int(offset)))


def test_bit_tensor_layout_across_threads() raises:
    # A 2-wide access covers coordinate bit 0. The thread takes the next
    # free bit, 1, and the issues walk bit 2, so thread t holds points
    # 2t, 2t + 1, 2t + 4 and 2t + 5.
    comptime Wide = PointerLoad[.float32, 2]
    comptime Mine = type_of(row_major[4]())
    assert_true(can_tiled_load[Wide, Mine, Gap, Mem, threads=2]())
    var storage = OwnedGPUTensor[.float32, .GENERIC, alignment=8](
        row_major[32]()
    )
    for i in range(32):
        TileTensor(storage)[i] = Float32(i)
    var gap = TileTensor(storage).reshape(
        bit_tensor_layout[_GapImages, type_of(row_major[8]())](),
    )
    for t in range(2):
        var regs = OwnedGPUTensor[.float32, .GENERIC, alignment=8](
            row_major[4]()
        )
        tiled_load[threads=2](Wide(), TileTensor(regs), gap, t)
        for j in range(4):
            var p = (j & 1) | (t << 1) | ((j >> 1) << 2)
            assert_equal(TileTensor(regs)[j][0], Float32(_gap_offset(p)))


def test_tmem_splits_across_threads() raises:
    # Lanes and warps play the tile's 128 rows, so each thread keeps one
    # row: 32 registers, which two x16 issues fill.
    comptime Load = TMemLoad[.float32, 16]
    assert_true(can_tiled_load[Load, Row32, Accum, TMemEngine, threads=128]())
    assert_true(
        can_tiled_store[
            TMemStore[.float32, 16], Accum, TMemEngine, Row32, threads=128
        ]()
    )
    assert_false(
        can_tiled_load[
            Load, type_of(row_major[1, 16]()), Accum, TMemEngine, threads=128
        ]()
    )
    # A warp-collective atom needs at least the warps it reaches.
    assert_false(can_tiled_load[Load, Row32, Accum, TMemEngine, threads=32]())
    # TMEM addresses are not byte addresses.
    assert_false(can_tiled_load[Load, Row32, Accum, Mem, threads=128]())


def tmem_tileop_kernel(addr: UInt32):
    # The test binary also elaborates kernels for the host, where tcgen05
    # is unavailable.
    comptime if _is_sm_100x_or_newer():
        var tile = TileTensor[Engine=TMemEngine](
            TMemStorage[.float32, MutUntrackedOrigin](addr), AccumLayout
        )
        var regs = OwnedGPUTensor[.float32, .LOCAL](row_major[1, 32]())
        # The TileOps only issue; this kernel, standing in for a building
        # block, waits before it uses the registers.
        tiled_load[threads=128](
            TMemLoad[.float32, 16](), TileTensor(regs), tile, Int(thread_idx.x)
        )
        wait_issues[TMemLoad[.float32, 16].Completion]()
        tiled_store[threads=128](
            TMemStore[.float32, 16](), tile, TileTensor(regs), Int(thread_idx.x)
        )
        wait_issues[TMemStore[.float32, 16].Completion]()


def tmem_tileop_issue_kernel(addr: UInt32):
    comptime if _is_sm_100x_or_newer():
        var tile = TileTensor[Engine=TMemEngine](
            TMemStorage[.float32, MutUntrackedOrigin](addr), AccumLayout
        )
        var regs = OwnedGPUTensor[.float32, .LOCAL](row_major[1, 32]())
        tiled_load[threads=128](
            TMemLoad[.float32, 16](), TileTensor(regs), tile, Int(thread_idx.x)
        )
        tiled_store[threads=128](
            TMemStore[.float32, 16](), tile, TileTensor(regs), Int(thread_idx.x)
        )


def test_tmem_tileops_do_not_wait() raises:
    var asm = _compile_code[
        tmem_tileop_issue_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    assert_true("tcgen05.ld.sync.aligned.32x32b.x16.b32" in asm)
    assert_false("tcgen05.wait" in asm)


def test_tmem_tileops_ptx() raises:
    var asm = _compile_code[
        tmem_tileop_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    assert_true("tcgen05.ld.sync.aligned.32x32b.x16.b32" in asm)
    assert_true("tcgen05.st.sync.aligned.32x32b.x16.b32" in asm)
    assert_false("32x32b.x32" in asm)
    assert_true("tcgen05.wait::ld.sync.aligned;" in asm)
    assert_true("tcgen05.wait::st.sync.aligned;" in asm)


def test_load_and_store_row_major() raises:
    # The width-4 atoms claim 16-byte alignment, which x86 enforces.
    var mem = OwnedGPUTensor[.float32, .GENERIC, alignment=16](
        row_major[4, 8]()
    )
    var regs = OwnedGPUTensor[.float32, .GENERIC, alignment=16](
        row_major[4, 8]()
    )
    var out = OwnedGPUTensor[.float32, .GENERIC, alignment=16](
        row_major[4, 8]()
    )
    for r in range(4):
        for c in range(8):
            TileTensor(mem)[r, c] = Float32(r * 8 + c)
            TileTensor(out)[r, c] = -1.0
    tiled_load(PointerLoad[.float32, 4](), TileTensor(regs), TileTensor(mem))
    tiled_store(PointerStore[.float32, 4](), TileTensor(out), TileTensor(regs))
    for r in range(4):
        for c in range(8):
            assert_equal(TileTensor(out)[r, c][0], Float32(r * 8 + c))


def test_load_transposes_with_scalar_atoms() raises:
    var mem = OwnedGPUTensor[.float32, .GENERIC](col_major[4, 8]())
    var regs = OwnedGPUTensor[.float32, .GENERIC](row_major[4, 8]())
    for r in range(4):
        for c in range(8):
            TileTensor(mem)[r, c] = Float32(r * 8 + c)
    tiled_load(PointerLoad[.float32, 1](), TileTensor(regs), TileTensor(mem))
    for r in range(4):
        for c in range(8):
            assert_equal(TileTensor(regs)[r, c][0], Float32(r * 8 + c))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
