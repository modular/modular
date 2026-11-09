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
from layout.atoms.pointer import PointerLoad, PointerStore
from layout.atoms.tmem import TMemLoad, TMemStore
from layout.bit_layout import is_bit_linear, placement
from layout.owned_tensor import OwnedGPUTensor
from layout.tensor_engine import DefaultEngine
from layout.layout import Layout as TileLayout
from layout.tileops import can_load, can_store, load, store
from layout.tmem_engine import TMEM_NUM_LANES, TMemEngine, TMemStorage

from max.gpu import warp_id
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
    assert_true(can_load[Wide, RowMajor, RowMajor, Mem]())
    # Four adjacent registers are four adjacent elements of a row, but in a
    # column-major tile those are 4 elements apart.
    assert_false(can_load[Wide, RowMajor, ColMajor, Mem]())
    assert_true(can_load[Narrow, RowMajor, ColMajor, Mem]())
    assert_true(can_store[PointerStore[.float32, 4], RowMajor, Mem, RowMajor]())


def test_tmem_splits_across_threads() raises:
    # Lanes and warps play the tile's 128 rows, so each thread keeps one
    # row: 32 registers, which two x16 issues fill.
    assert_true(can_load[TMemLoad[.float32, 16], Row32, Accum, TMemEngine]())
    assert_true(can_store[TMemStore[.float32, 16], Accum, TMemEngine, Row32]())
    assert_false(
        can_load[
            TMemLoad[.float32, 16],
            type_of(row_major[1, 16]()),
            Accum,
            TMemEngine,
        ]()
    )
    # TMEM addresses are not byte addresses.
    assert_false(can_load[TMemLoad[.float32, 16], Row32, Accum, Mem]())


def tmem_tileop_kernel(addr: UInt32):
    # The test binary also elaborates kernels for the host, where tcgen05
    # is unavailable.
    comptime if _is_sm_100x_or_newer():
        var tile = TileTensor[
            .float32, Accum, MutUntrackedOrigin, Engine=TMemEngine
        ](TMemStorage[.float32, MutUntrackedOrigin](addr), AccumLayout)
        var regs = OwnedGPUTensor[.float32, .LOCAL](row_major[1, 32]())
        # The TileOps only issue; this kernel, standing in for a building
        # block, waits before it uses the registers.
        load(TMemLoad[.float32, 16](), TileTensor(regs), tile, Int(warp_id()))
        TMemLoad[.float32, 16].Completion.wait()
        store(TMemStore[.float32, 16](), tile, TileTensor(regs), Int(warp_id()))
        TMemStore[.float32, 16].Completion.wait()


def tmem_tileop_issue_kernel(addr: UInt32):
    comptime if _is_sm_100x_or_newer():
        var tile = TileTensor[
            .float32, Accum, MutUntrackedOrigin, Engine=TMemEngine
        ](TMemStorage[.float32, MutUntrackedOrigin](addr), AccumLayout)
        var regs = OwnedGPUTensor[.float32, .LOCAL](row_major[1, 32]())
        load(TMemLoad[.float32, 16](), TileTensor(regs), tile, Int(warp_id()))
        store(TMemStore[.float32, 16](), tile, TileTensor(regs), Int(warp_id()))


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
    load(PointerLoad[.float32, 4](), TileTensor(regs), TileTensor(mem))
    store(PointerStore[.float32, 4](), TileTensor(out), TileTensor(regs))
    for r in range(4):
        for c in range(8):
            assert_equal(TileTensor(out)[r, c][0], Float32(r * 8 + c))


def test_load_transposes_with_scalar_atoms() raises:
    var mem = OwnedGPUTensor[.float32, .GENERIC](col_major[4, 8]())
    var regs = OwnedGPUTensor[.float32, .GENERIC](row_major[4, 8]())
    for r in range(4):
        for c in range(8):
            TileTensor(mem)[r, c] = Float32(r * 8 + c)
    load(PointerLoad[.float32, 1](), TileTensor(regs), TileTensor(mem))
    for r in range(4):
        for c in range(8):
            assert_equal(TileTensor(regs)[r, c][0], Float32(r * 8 + c))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
