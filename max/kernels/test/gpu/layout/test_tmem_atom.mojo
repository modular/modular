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
"""Blackwell tests for the TMEM atoms.

Four warps share a `(128, 32)` float32 TMEM allocation. Each warp stores a
lane-and-column tagged register row into its quadrant with `TMemStore` and
loads it back with `TMemLoad`, issuing each 16-column atom twice and waiting
once per direction. Every thread passes its quadrant's row 0 as the TMEM
operand and its own register row as the other.
"""

from std.memory import unsafe_stack_allocation
from std.testing import assert_equal, assert_true

from layout import Idx, TileTensor, coord, row_major
from layout.owned_tensor import OwnedGPUTensor
from layout.layout import Layout as TileLayout
from layout.atoms.tmem import TMemLoad, TMemStore
from layout.tmem_engine import TMEM_NUM_LANES, TMemEngine, TMemStorage

from max.gpu import WARP_SIZE, barrier, lane_id, warp_id
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_release_allocation_lock,
)
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.compile import _compile_code

comptime NUM_LANES = 128
comptime NUM_COLS = 32
comptime ATOM_COLS = 16
comptime NUM_ELEMENTS = NUM_LANES * NUM_COLS

comptime AccumLayout = TileLayout(
    shape=coord[NUM_LANES, NUM_COLS],
    stride=coord[1, TMEM_NUM_LANES],
)
comptime AccumTile = TileTensor[
    .float32, type_of(AccumLayout), MutUntrackedOrigin, Engine=TMemEngine
]


@always_inline
def _expected(lane: Int, col: Int) -> Float32:
    return Float32(lane * 1000 + col)


def tmem_atom_kernel[
    output_origin: MutOrigin
](output: MutPointer[Float32, output_origin]):
    var smem_addr = unsafe_stack_allocation[
        1, DType.uint32, address_space=.SHARED
    ]()
    if warp_id() == 0:
        tcgen05_alloc[1](smem_addr, UInt32(NUM_COLS))
    barrier()

    var tile = AccumTile(
        TMemStorage[.float32, MutUntrackedOrigin](smem_addr[]), AccumLayout
    )
    var row = tile[Int(warp_id()) * WARP_SIZE :, :][:1, :]
    var lane = Int(warp_id()) * WARP_SIZE + Int(lane_id())

    var regs_owned = OwnedGPUTensor[.float32, .LOCAL](row_major[1, NUM_COLS]())
    var regs = TileTensor(regs_owned)
    comptime for c in range(NUM_COLS):
        regs[coord[0, c]] = _expected(lane, c)
    comptime for h in range(NUM_COLS // ATOM_COLS):
        TMemStore[.float32, ATOM_COLS]()(
            row[:, Idx[h * ATOM_COLS] : Idx[(h + 1) * ATOM_COLS]],
            regs[:, Idx[h * ATOM_COLS] : Idx[(h + 1) * ATOM_COLS]],
        )
    TMemStore[.float32, ATOM_COLS].Completion.wait()

    var back_owned = OwnedGPUTensor[.float32, .LOCAL](row_major[1, NUM_COLS]())
    var back = TileTensor(back_owned)
    comptime for h in range(NUM_COLS // ATOM_COLS):
        TMemLoad[.float32, ATOM_COLS]()(
            back[:, Idx[h * ATOM_COLS] : Idx[(h + 1) * ATOM_COLS]],
            row[:, Idx[h * ATOM_COLS] : Idx[(h + 1) * ATOM_COLS]],
        )
    TMemLoad[.float32, ATOM_COLS].Completion.wait()
    comptime for c in range(NUM_COLS):
        output[unsafe_offset=lane * NUM_COLS + c] = back[Idx[0], Idx[c]]

    barrier()
    if warp_id() == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](smem_addr[], UInt32(NUM_COLS))


def test_codegen_issues_the_atom_shape() raises:
    var asm = _compile_code[
        tmem_atom_kernel[output_origin=MutUntrackedOrigin],
        target=get_gpu_target["sm_100a"](),
    ]().asm
    # Each direction is two issues of the atom's `x16` instruction, never
    # one `x32`.
    assert_true("tcgen05.st.sync.aligned.32x32b.x16.b32" in asm)
    assert_true("tcgen05.ld.sync.aligned.32x32b.x16.b32" in asm)
    assert_true("tcgen05.st.sync.aligned.32x32b.x32" not in asm)
    assert_true("tcgen05.ld.sync.aligned.32x32b.x32" not in asm)
    assert_true("tcgen05.wait::st.sync.aligned;" in asm)
    assert_true("tcgen05.wait::ld.sync.aligned;" in asm)


def test_round_trip_all_lanes() raises:
    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[.float32](NUM_ELEMENTS)
        buf.enqueue_fill(Float32(-1))
        ctx.enqueue_function[tmem_atom_kernel[origin_of(buf)]](
            buf, grid_dim=1, block_dim=NUM_LANES
        )
        var host = ctx.enqueue_create_host_buffer[.float32](NUM_ELEMENTS)
        ctx.enqueue_copy(host, buf)
        ctx.synchronize()
        for lane in range(NUM_LANES):
            for col in range(NUM_COLS):
                assert_equal(
                    host[lane * NUM_COLS + col],
                    _expected(lane, col),
                    String("lane ", lane, " col ", col),
                )


def main() raises:
    test_codegen_issues_the_atom_shape()
    test_round_trip_all_lanes()
