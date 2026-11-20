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
"""The MoE index build of packed routing rows, in one CTA.

`routing` is `[tokens, 2 * top_k]` float32: each token's expert ids, then
their weights, as a router writes them and an allgather replicates them.
Slot `i = t * top_k + j` goes to group `id - expert_offset` when that is
below `local_experts`, else to the trailing group `local_experts`. The
outputs are `moe_create_indices`' over `local_experts + 1` groups, plus the
groups' scale offsets and the weights copied dense.

A stable counting sort, so the order is deterministic: each warp counts its
span of slots per group into its own row of a `[WARPS, groups]` table;
after a meet, each group's column becomes the warps' exclusive bases; after
another, one warp scans the group totals into the starts; after the last,
each warp places its slots at start + base + its rank among the lanes with
the same group. The schedule and the arithmetic are the Moka component's
(`Kernels/test/gpu/Moka/kernels/moe/moe_indices_packed_plan.mojo`), whose
ops call the helpers below.
"""

from std.bit import pop_count
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import align_up, ceildiv

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace, external_memory
from max.gpu.primitives import warp
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from max.gpu.sync import barrier, syncwarp
from std.utils import StaticTuple

comptime WARPS = 32
comptime THREADS = WARPS * WARP_SIZE
comptime MAX_GROUPS = THREADS
"""A thread per group sums its column of the counts."""
comptime ALIGN = 128
"""Scale rows of a group are padded to this many."""

comptime POISONS = 3
"""Planted faults, `poison` 1 through 3, each of which a test must catch:
1 files other devices' slots under group 0, 2 ranks a lane after every lane
of its group, 3 skips the last slot's weight."""

comptime In[T: DType] = Pointer[Scalar[T], ImmUntrackedOrigin]
comptime Out[T: DType] = Pointer[Scalar[T], MutUntrackedOrigin]
comptime Smem = Pointer[
    UInt32, MutUntrackedOrigin, address_space=AddressSpace.SHARED
]


@fieldwise_init
struct PackedIndicesArgs(DevicePassable, ImplicitlyCopyable):
    """The index build's operands, one kernel parameter."""

    var routing: In[DType.float32]
    """`[tokens, 2 * top_k]`: ids as float32, then weights."""
    var order: Out[DType.uint32]
    """`[slots]` slot of each sorted position."""
    var start: Out[DType.uint32]
    """`[groups + 1]` first sorted position of each group, then the total."""
    var restore: Out[DType.uint32]
    """`[slots]` sorted position of each slot."""
    var expert_ids: Out[DType.int32]
    """`[groups]` each group's id, `g`."""
    var usage: Out[DType.uint32]
    """`[2]` the largest group's count, then the group count."""
    var scales_offset: Out[DType.uint32]
    """`[groups]` scale-row tiles each group's start is shifted by."""
    var weights: Out[DType.float32]
    """`[slots]` each slot's weight, dense."""
    var tokens: Int64
    var expert_offset: Int64

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "PackedIndicesArgs"


@always_inline
def warp_span(slots: Int, w: Int) -> Tuple[Int, Int]:
    """Warp `w`'s slots `[lo, hi)`: equal spans of whole 32-slot chunks."""
    var span = ceildiv(slots, THREADS) * WARP_SIZE
    var lo = min(w * span, slots)
    return (lo, min(lo + span, slots))


@always_inline
def slot_group[
    top_k: Int, local_experts: Int, poison: Int = 0
](a: PackedIndicesArgs, i: Int) -> Int:
    """Slot `i`'s group; other devices' experts share the last one."""
    var e = Int(a.routing[unsafe_offset=(i // top_k) * 2 * top_k + i % top_k])
    var g = e - Int(a.expert_offset)
    if g < 0 or g >= local_experts:
        return 0 if poison == 1 else local_experts
    return g


@always_inline
def _lanes_below() -> UInt32:
    return (UInt32(1) << UInt32(lane_id())) - 1


@always_inline
def _same_group(g: Int) -> UInt32:
    """The lanes of the warp whose slot is in group `g`."""
    return UInt32(warp.match_any[mask_type=DType.uint32](Int32(g)))


@always_inline
def count_span[
    top_k: Int, local_experts: Int, poison: Int = 0
](a: PackedIndicesArgs, row: Smem, w: Int):
    """Warp `w` counts its span per group into its `row`, and copies its
    slots' weights dense."""
    comptime groups = local_experts + 1
    var lane = Int(lane_id())
    for g in range(lane, groups, WARP_SIZE):
        row[unsafe_offset=g] = 0
    syncwarp()
    var slots = Int(a.tokens) * top_k
    var lo, hi = warp_span(slots, w)
    for base in range(lo, hi, WARP_SIZE):
        var i = base + lane
        var on = i < hi
        var g = slot_group[top_k, local_experts, poison](a, i) if on else groups
        if on and not (poison == 3 and i == slots - 1):
            a.weights[unsafe_offset=i] = a.routing[
                unsafe_offset=(i // top_k) * 2 * top_k + top_k + i % top_k
            ]
        var same = _same_group(g)
        # the lowest lane of each group adds the group's lanes
        if on and (same & _lanes_below()) == 0:
            row[unsafe_offset=g] += UInt32(pop_count(same))
        syncwarp()


@always_inline
def column_bases[local_experts: Int](table: Smem, totals: Smem, t: Int):
    """Thread `t`'s group column: each warp's count becomes the counts of the
    warps before it, and the column's sum goes to `totals`."""
    comptime groups = local_experts + 1
    if t < groups:
        var run = UInt32(0)
        comptime for w in range(WARPS):
            var c = table[unsafe_offset=w * groups + t]
            table[unsafe_offset=w * groups + t] = run
            run += c
        totals[unsafe_offset=t] = run


@always_inline
def scan_groups[
    local_experts: Int
](a: PackedIndicesArgs, totals: Smem, starts: Smem):
    """Warp 0 scans the group totals into `starts` and writes the CSR, the
    ids, the scale offsets and the usage."""
    comptime groups = local_experts + 1
    comptime per = ceildiv(groups, WARP_SIZE)
    var lane = Int(lane_id())
    var first = lane * per
    var mine = UInt32(0)
    var mine_aligned = UInt32(0)
    var most = UInt32(0)
    comptime for j in range(per):
        if first + j < groups:
            var c = totals[unsafe_offset=first + j]
            mine += c
            mine_aligned += align_up(c, UInt32(ALIGN))
            most = max(most, c)
    var at = warp.prefix_sum[exclusive=True](mine)
    var at_aligned = warp.prefix_sum[exclusive=True](mine_aligned)
    most = warp.max(most)
    comptime for j in range(per):
        var g = first + j
        if g < groups:
            var c = totals[unsafe_offset=g]
            starts[unsafe_offset=g] = at
            a.start[unsafe_offset=g] = at
            a.expert_ids[unsafe_offset=g] = Int32(g)
            a.scales_offset[unsafe_offset=g] = at_aligned // UInt32(
                ALIGN
            ) - at // UInt32(ALIGN)
            at += c
            at_aligned += align_up(c, UInt32(ALIGN))
    if lane == WARP_SIZE - 1:
        a.start[unsafe_offset=groups] = at
    if lane == 0:
        a.usage[unsafe_offset=0] = most
        a.usage[unsafe_offset=1] = UInt32(groups)


@always_inline
def scatter_span[
    top_k: Int, local_experts: Int, poison: Int = 0
](a: PackedIndicesArgs, row: Smem, starts: Smem, w: Int):
    """Warp `w` places its span's slots, in slot order within each group."""
    comptime groups = local_experts + 1
    var lane = Int(lane_id())
    var lo, hi = warp_span(Int(a.tokens) * top_k, w)
    for base in range(lo, hi, WARP_SIZE):
        var i = base + lane
        var on = i < hi
        var g = slot_group[top_k, local_experts, poison](a, i) if on else groups
        var same = _same_group(g)
        var below = same if poison == 2 else same & _lanes_below()
        if on:
            var pos = (
                starts[unsafe_offset=g]
                + row[unsafe_offset=g]
                + UInt32(pop_count(below))
            )
            a.order[unsafe_offset=Int(pos)] = UInt32(i)
            a.restore[unsafe_offset=i] = pos
        syncwarp()
        if on and (same & _lanes_below()) == 0:
            row[unsafe_offset=g] += UInt32(pop_count(same))
        syncwarp()


def smem_words(local_experts: Int) -> Int:
    """The count table, then the group totals, then the starts."""
    return (WARPS + 2) * (local_experts + 1)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(THREADS))
)
def _moe_indices_packed[
    top_k: Int, local_experts: Int, poison: Int
](a: PackedIndicesArgs):
    comptime groups = local_experts + 1
    var table = (
        external_memory[UInt8, address_space=.SHARED, alignment=16]()
        .unsafe_bitcast[UInt32]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )
    var totals = table.unsafe_offset(WARPS * groups)
    var starts = totals.unsafe_offset(groups)
    var w = Int(warp_id())
    # The routing is the grid before this one's. Release right after the
    # wait, as the other routing kernels do: dependents wait for this kernel
    # to finish before they read what it writes.
    wait_on_dependent_grids()
    launch_dependent_grids()
    count_span[top_k, local_experts, poison](
        a, table.unsafe_offset(w * groups), w
    )
    barrier()
    column_bases[local_experts](table, totals, Int(thread_idx.x))
    barrier()
    if w == 0:
        scan_groups[local_experts](a, totals, starts)
    barrier()
    scatter_span[top_k, local_experts, poison](
        a, table.unsafe_offset(w * groups), starts, w
    )


def moe_indices_packed[
    top_k: Int, local_experts: Int, poison: Int = 0
](a: PackedIndicesArgs, ctx: DeviceContext) raises:
    """Builds the indices of `a.tokens` packed routing rows.

    Every id must be a whole number below 2^24, as a router writes it.
    `poison` plants a fault for the tests.
    """
    comptime assert local_experts + 1 <= MAX_GROUPS, "too many groups"
    ctx.enqueue_function[_moe_indices_packed[top_k, local_experts, poison]](
        a,
        grid_dim=1,
        block_dim=THREADS,
        shared_mem_bytes=smem_words(local_experts) * 4,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )
