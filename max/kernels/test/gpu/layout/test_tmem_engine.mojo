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
"""Blackwell tests for `TMemEngine`.

One kernel drives both checks. Four warps share a `(128, 32)` float32 TMEM
allocation; each thread tiles it down to its warp's quadrant and then to its
own lane row, copies a lane-and-column tagged register tile into that row,
copies the row back into a second register tile and writes it to global
memory. The codegen test asserts the kernel lowers each direction to a single
`tcgen05.ld`/`st` in the `32x32b` shape covering all 32 columns, followed by
one wait; the execution test checks every cell round-trips, which validates
the engine's address encoding of the grid layout and the quadrant addressing
on hardware.

A second kernel uses the 1-CTA M=64 placement, where each warp owns only the
first 16 lanes of its quadrant. Its layout is nested, `((16, 4), 32)` with
strides `((1, 32), 128)`, and threads locate their row with a hierarchical
coordinate instead of `tile()`.

A third kernel copies each half of the row separately with `tmem_copy_async`
and waits once per direction; its codegen test asserts two `x16`
instructions share a single wait.

A fourth kernel round-trips a 128-column row. Its codegen test asserts the
engine never names more than 64 registers in one instruction and waits per
64-column slice: two `x64` instructions and two waits per direction, no
`x128`.

A fifth kernel runs under launch bounds that hand `ptxas` a 128-register
budget, the width the ISA would let a single instruction name, and copies a
128-column row from global memory through TMEM and back. Its test compiles
the kernel on the device and asserts the function reports no local memory,
so the 64-register cap is what keeps a wide copy inside a budget that an
`x128` and its staging registers could not fit.
"""

from std.memory import unsafe_stack_allocation
from std.testing import assert_equal, assert_true
from std.utils.static_tuple import StaticTuple

from layout import Coord, Idx, TileTensor, row_major, stack_allocation
from layout.tile_layout import Layout as TileLayout
from layout.tmem_engine import (
    TMemEngine,
    TMemStorage,
    TMEM_NUM_LANES,
    tmem_copy_async,
)

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    lane_id,
    warp_id,
)
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_release_allocation_lock,
)
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.host.func_attribute import Attribute

comptime NUM_LANES = 128
comptime NUM_COLS = 32
comptime WIDE_COLS = 128
comptime NUM_WARPS = NUM_LANES // WARP_SIZE
comptime NUM_ELEMENTS = NUM_LANES * NUM_COLS
comptime WIDE_ELEMENTS = NUM_LANES * WIDE_COLS

comptime BUDGET_MIN_BLOCKS = 4
comptime BUDGET_REGS = 65536 // (NUM_LANES * BUDGET_MIN_BLOCKS)
"""The per-thread register budget the launch bounds below hand `ptxas`: the
SM's register file over four 128-thread blocks, which is 128, the most
registers the ISA lets one `tcgen05` instruction name."""


comptime AccumLayout[num_cols: Int] = TileLayout(
    shape=Coord(Idx[NUM_LANES], Idx[num_cols]),
    stride=Coord(Idx[1], Idx[TMEM_NUM_LANES]),
)


comptime AccumTile[num_cols: Int] = TileTensor[
    .float32,
    type_of(AccumLayout[num_cols]),
    MutAnyOrigin,
    Engine=TMemEngine,
]
comptime AccumStorage = TMemStorage[.float32, MutAnyOrigin]

# This thread's view of its warp's quadrant: one lane, every column.
comptime ROW_LAYOUT = TileLayout(
    shape=Coord(Idx[1], Idx[NUM_COLS]),
    stride=Coord(Idx[1], Idx[TMEM_NUM_LANES]),
)
comptime RowTile = TileTensor[
    .float32,
    type_of(ROW_LAYOUT),
    MutAnyOrigin,
    Engine=TMemEngine,
]

comptime M64_ROWS = 64
comptime M64_LANES_PER_WARP = M64_ROWS // NUM_WARPS
comptime M64_LAYOUT = TileLayout(
    shape=Coord(Coord(Idx[M64_LANES_PER_WARP], Idx[NUM_WARPS]), Idx[NUM_COLS]),
    stride=Coord(Coord(Idx[1], Idx[WARP_SIZE]), Idx[TMEM_NUM_LANES]),
)
comptime M64Tile = TileTensor[
    .float32,
    type_of(M64_LAYOUT),
    MutAnyOrigin,
    Engine=TMemEngine,
]


@inline(.always)
def _expected(lane: Int, col: Int) -> Float32:
    # Distinct per cell and exact in float32, so a swapped lane/column stride
    # or a wrong quadrant base shows up as a mismatch rather than a coincidence.
    return Float32(lane * 1000 + col)


@inline(.always)
def _round_trip(
    row: TileTensor[
        .float32,
        _,
        MutAnyOrigin,
        Engine=TMemEngine,
        ...,
    ],
    tag: Int,
    output: MutPointer[Float32, MutAnyOrigin],
    out_row: Int,
):
    """Copies a `tag`-stamped register row into `row`, copies it back out and
    writes the result to `output[out_row * num_cols:]`."""
    comptime num_cols = type_of(row).LayoutType.static_product
    var regs = stack_allocation[dtype=.float32](row_major[1, num_cols]())
    comptime for c in range(num_cols):
        regs[Coord(Idx[0], Idx[c])] = _expected(tag, c)
    row.copy_from(regs)

    var back = stack_allocation[dtype=.float32](row_major[1, num_cols]())
    back.copy_from(row)
    comptime for c in range(num_cols):
        output[unsafe_offset=out_row * num_cols + c] = back[
            Coord(Idx[0], Idx[c])
        ]


def _tmem_round_trip_kernel[
    num_cols: Int
](output: MutPointer[Float32, MutAnyOrigin]):
    var smem_addr = unsafe_stack_allocation[
        1, DType.uint32, address_space=.SHARED
    ]()
    if warp_id() == 0:
        tcgen05_alloc[1](smem_addr, UInt32(num_cols))
    barrier()

    var tile = AccumTile[num_cols](
        AccumStorage(smem_addr[]), AccumLayout[num_cols]
    )
    var warp_tile = tile.tile[WARP_SIZE, num_cols](
        Coord(Int(warp_id()), Idx[0])
    )
    # Row 0 of the warp tile is this thread's view: the hardware adds the
    # thread's lane to the warp base, so the storage must stay at the base.
    var row = warp_tile.tile[1, num_cols](Coord(Idx[0], Idx[0]))
    var lane = Int(warp_id()) * WARP_SIZE + Int(lane_id())
    _round_trip(row, lane, output, lane)

    barrier()
    if warp_id() == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](smem_addr[], UInt32(num_cols))


comptime tmem_round_trip_kernel = _tmem_round_trip_kernel[NUM_COLS]
comptime tmem_wide_round_trip_kernel = _tmem_round_trip_kernel[WIDE_COLS]


def tmem_async_round_trip_kernel(output: MutPointer[Float32, MutAnyOrigin]):
    var smem_addr = unsafe_stack_allocation[
        1, DType.uint32, address_space=.SHARED
    ]()
    if warp_id() == 0:
        tcgen05_alloc[1](smem_addr, UInt32(NUM_COLS))
    barrier()

    var tile = AccumTile[NUM_COLS](
        AccumStorage(smem_addr[]), AccumLayout[NUM_COLS]
    )
    var row = tile.tile[WARP_SIZE, NUM_COLS](
        Coord(Int(warp_id()), Idx[0])
    ).tile[1, NUM_COLS](Coord(Idx[0], Idx[0]))
    var lane = Int(warp_id()) * WARP_SIZE + Int(lane_id())
    comptime HALF = NUM_COLS // 2

    var regs = stack_allocation[dtype=.float32](row_major[1, NUM_COLS]())
    comptime for c in range(NUM_COLS):
        regs[Coord(Idx[0], Idx[c])] = _expected(lane, c)
    # Two half-row stores, one wait.
    comptime for h in range(2):
        tmem_copy_async(
            row.tile[1, HALF](Coord(Idx[0], Idx[h])),
            regs.tile[1, HALF](Coord(Idx[0], Idx[h])),
        )
    TMemEngine.wait_store()

    var back = stack_allocation[dtype=.float32](row_major[1, NUM_COLS]())
    comptime for h in range(2):
        tmem_copy_async(
            back.tile[1, HALF](Coord(Idx[0], Idx[h])),
            row.tile[1, HALF](Coord(Idx[0], Idx[h])),
        )
    TMemEngine.wait_load()
    comptime for c in range(NUM_COLS):
        output[unsafe_offset=lane * NUM_COLS + c] = back[Coord(Idx[0], Idx[c])]

    barrier()
    if warp_id() == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](smem_addr[], UInt32(NUM_COLS))


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(NUM_LANES))
)
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(BUDGET_MIN_BLOCKS))
def tmem_budget_round_trip_kernel(
    input: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
):
    var smem_addr = unsafe_stack_allocation[
        1, DType.uint32, address_space=.SHARED
    ]()
    if warp_id() == 0:
        tcgen05_alloc[1](smem_addr, UInt32(WIDE_COLS))
    barrier()

    var tile = AccumTile[WIDE_COLS](
        AccumStorage(smem_addr[]), AccumLayout[WIDE_COLS]
    )
    var row = tile.tile[WARP_SIZE, WIDE_COLS](
        Coord(Int(warp_id()), Idx[0])
    ).tile[1, WIDE_COLS](Coord(Idx[0], Idx[0]))
    var lane = Int(warp_id()) * WARP_SIZE + Int(lane_id())
    # Both ends of the round trip live in global memory, so the staging
    # registers of the two copies are the kernel's only wide register use.
    var in_row = TileTensor(
        input.unsafe_offset(lane * WIDE_COLS), row_major[1, WIDE_COLS]()
    )
    var out_row = TileTensor(
        output.unsafe_offset(lane * WIDE_COLS), row_major[1, WIDE_COLS]()
    )
    row.copy_from(in_row)
    out_row.copy_from(row)

    barrier()
    if warp_id() == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](smem_addr[], UInt32(WIDE_COLS))


def tmem_m64_round_trip_kernel(output: MutPointer[Float32, MutAnyOrigin]):
    var smem_addr = unsafe_stack_allocation[
        1, DType.uint32, address_space=.SHARED
    ]()
    if warp_id() == 0:
        tcgen05_alloc[1](smem_addr, UInt32(NUM_COLS))
    barrier()

    var tile = M64Tile(AccumStorage(smem_addr[]), M64_LAYOUT)
    # Row coordinate (0, w) is the warp's quadrant base; the hardware adds
    # the lane. Only lanes below 16 hold logical rows, but the `.aligned`
    # instructions need every lane to issue them convergently, so all lanes
    # copy, and the host ignores the upper half of each quadrant.
    var row_coord = Coord(Idx[0], Int(warp_id()))
    var row = RowTile(
        tile._offset_storage(Int(tile.layout(Coord(row_coord, Idx[0])))),
        ROW_LAYOUT,
    )
    var tmem_lane = Int(warp_id()) * WARP_SIZE + Int(lane_id())
    var logical_row = Int(warp_id()) * M64_LANES_PER_WARP + Int(lane_id())
    _round_trip(row, logical_row, output, tmem_lane)

    barrier()
    if warp_id() == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](smem_addr[], UInt32(NUM_COLS))


def test_codegen_batches_one_instruction_per_direction() raises:
    print("== test_codegen_batches_one_instruction_per_direction")
    var asm = _compile_code[
        tmem_round_trip_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    # 32 columns fit one `x32` instruction, and each copy waits exactly once.
    assert_equal(asm.count("tcgen05.st.sync.aligned.32x32b.x32.b32"), 1)
    assert_equal(asm.count("tcgen05.wait::st.sync.aligned;"), 1)
    assert_equal(asm.count("tcgen05.ld.sync.aligned.32x32b.x32.b32"), 1)
    assert_equal(asm.count("tcgen05.wait::ld.sync.aligned;"), 1)
    assert_true("tcgen05.ld.sync.aligned.32x32b.x8" not in asm)


def test_codegen_async_copies_share_one_wait() raises:
    print("== test_codegen_async_copies_share_one_wait")
    var asm = _compile_code[
        tmem_async_round_trip_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    assert_equal(asm.count("tcgen05.st.sync.aligned.32x32b.x16.b32"), 2)
    assert_equal(asm.count("tcgen05.wait::st.sync.aligned;"), 1)
    assert_equal(asm.count("tcgen05.ld.sync.aligned.32x32b.x16.b32"), 2)
    assert_equal(asm.count("tcgen05.wait::ld.sync.aligned;"), 1)


def test_codegen_wide_row_caps_registers_per_instruction() raises:
    print("== test_codegen_wide_row_caps_registers_per_instruction")
    var asm = _compile_code[
        tmem_wide_round_trip_kernel, target=get_gpu_target["sm_100a"]()
    ]().asm
    # A 128-column row is two 64-column slices, each with its own wait; a
    # single `x128` would need 128 consecutive registers (ptxas C7602).
    assert_true("32x32b.x128" not in asm)
    assert_equal(asm.count("tcgen05.st.sync.aligned.32x32b.x64.b32"), 2)
    assert_equal(asm.count("tcgen05.wait::st.sync.aligned;"), 2)
    assert_equal(asm.count("tcgen05.ld.sync.aligned.32x32b.x64.b32"), 2)
    assert_equal(asm.count("tcgen05.wait::ld.sync.aligned;"), 2)


def _check_round_trip[
    kernel: def(MutPointer[Float32, MutAnyOrigin]) thin, num_cols: Int
](ctx: DeviceContext) raises:
    comptime num_elements = NUM_LANES * num_cols
    var buf = ctx.enqueue_create_buffer[.float32](num_elements)
    buf.enqueue_fill(Float32(-1))
    ctx.enqueue_function[kernel](buf, grid_dim=1, block_dim=NUM_LANES)
    var host = ctx.enqueue_create_host_buffer[.float32](num_elements)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    for lane in range(NUM_LANES):
        for col in range(num_cols):
            assert_equal(
                host[lane * num_cols + col],
                _expected(lane, col),
                String("lane ", lane, " col ", col),
            )


def test_round_trip_all_lanes(ctx: DeviceContext) raises:
    print("== test_round_trip_all_lanes")
    _check_round_trip[tmem_round_trip_kernel, NUM_COLS](ctx)


def test_async_round_trip_all_lanes(ctx: DeviceContext) raises:
    print("== test_async_round_trip_all_lanes")
    _check_round_trip[tmem_async_round_trip_kernel, NUM_COLS](ctx)


def test_wide_round_trip_all_lanes(ctx: DeviceContext) raises:
    print("== test_wide_round_trip_all_lanes")
    _check_round_trip[tmem_wide_round_trip_kernel, WIDE_COLS](ctx)


def test_budget_round_trip_does_not_spill(ctx: DeviceContext) raises:
    print("== test_budget_round_trip_does_not_spill")
    var func = ctx.compile_function[tmem_budget_round_trip_kernel]()
    # The budget only means something if `ptxas` honored the launch bounds.
    assert_true(func.get_attribute(Attribute.NUM_REGS) <= BUDGET_REGS)
    assert_equal(func.get_attribute(Attribute.LOCAL_SIZE_BYTES), 0)

    var host_in = ctx.enqueue_create_host_buffer[.float32](WIDE_ELEMENTS)
    ctx.synchronize()
    for lane in range(NUM_LANES):
        for col in range(WIDE_COLS):
            host_in[lane * WIDE_COLS + col] = _expected(lane, col)
    var input = ctx.enqueue_create_buffer[.float32](WIDE_ELEMENTS)
    ctx.enqueue_copy(input, host_in)
    var output = ctx.enqueue_create_buffer[.float32](WIDE_ELEMENTS)
    output.enqueue_fill(Float32(-1))
    ctx.enqueue_function(func, input, output, grid_dim=1, block_dim=NUM_LANES)
    var host_out = ctx.enqueue_create_host_buffer[.float32](WIDE_ELEMENTS)
    ctx.enqueue_copy(host_out, output)
    ctx.synchronize()
    for lane in range(NUM_LANES):
        for col in range(WIDE_COLS):
            assert_equal(
                host_out[lane * WIDE_COLS + col],
                _expected(lane, col),
                String("lane ", lane, " col ", col),
            )


def test_m64_round_trip(ctx: DeviceContext) raises:
    print("== test_m64_round_trip")
    var buf = ctx.enqueue_create_buffer[.float32](NUM_ELEMENTS)
    buf.enqueue_fill(Float32(-1))
    ctx.enqueue_function[tmem_m64_round_trip_kernel](
        buf, grid_dim=1, block_dim=NUM_LANES
    )
    var host = ctx.enqueue_create_host_buffer[.float32](NUM_ELEMENTS)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    # Logical row r of the M=64 tile lives in lane 32 * (r // 16) + r % 16.
    for row in range(M64_ROWS):
        var tmem_lane = (row // M64_LANES_PER_WARP) * WARP_SIZE + (
            row % M64_LANES_PER_WARP
        )
        for col in range(NUM_COLS):
            assert_equal(
                host[tmem_lane * NUM_COLS + col],
                _expected(row, col),
                String("row ", row, " col ", col),
            )


def main() raises:
    test_codegen_batches_one_instruction_per_direction()
    test_codegen_async_copies_share_one_wait()
    test_codegen_wide_row_caps_registers_per_instruction()
    with DeviceContext() as ctx:
        test_round_trip_all_lanes(ctx)
        test_async_round_trip_all_lanes(ctx)
        test_wide_round_trip_all_lanes(ctx)
        test_budget_round_trip_does_not_spill(ctx)
        test_m64_round_trip(ctx)
