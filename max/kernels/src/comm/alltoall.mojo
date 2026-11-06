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
"""Multi-GPU all-to-all over peer-to-peer reads.

Every device of a group splits its input into `group_size` equal chunks along
a split axis, sends chunk `j` to the group's rank `j`, and concatenates the
chunks it receives, in source rank order, along a concat axis. It is
`torch.distributed.all_to_all_single` generalized to any pair of axes, and
the transpose of a sharding from one tensor axis to another.

Each device pulls its chunks straight out of its peers' inputs, the way the
P2P allgather does, between a start and an end barrier on the shared
`Signal` buffers. A chunk is never contiguous in general: viewed against the
input it is `outer_s` runs of `in_inner` elements, one every
`group_size * in_inner`, and viewed against the output it is `outer_c` runs
of `out_inner` elements, again one every `group_size * out_inner`. Both views
number the chunk's elements in the same row-major order, so one flat chunk
index addresses both sides of every copy.
"""

from std.collections import Array
from std.collections.optional import Optional
from std.math.uutils import ualign_down
from std.math import ceildiv
from std.sys import simd_width_of, align_of, size_of

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    global_idx,
    grid_dim,
)
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from max.gpu.host import DeviceContext, get_gpu_target
from std.utils import StaticTuple

from .allgather import allgather_tuning_table
from .device_query import dispatch_select_comm_config
from .reducescatter import _target_address_space
from .sync import (
    Signal,
    _multi_gpu_barrier,
    circular_add,
    is_p2p_enabled,
)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"all_to_all_p2p_{dtype}_{simd_width}")
def _all_to_all_p2p_kernel[
    dtype: DType,
    ngpus: Int,
    simd_width: Int,
    *,
    BLOCK_SIZE: Int,
    domain_id: Int = 0,
](
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
    srcs: Array[ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    chunk_len: Int64,
    in_inner: Int64,
    out_inner: Int64,
    my_rank: Int32,
):
    """Pulls this rank's chunk out of every peer's input.

    `srcs` and `rank_sigs` are this device's group, indexed by group rank,
    and `my_rank` is its rank in the group. `in_inner` and `out_inner` are
    multiples of `simd_width`, so no vector straddles two runs.
    """
    comptime alignment = align_of[SIMD[dtype, simd_width]]()

    var global_tid = global_idx.x
    var stride = grid_dim.x * BLOCK_SIZE
    var _my_rank = Int(my_rank)
    var my_sig = rank_sigs[_my_rank]

    # Peers in round-robin order from this rank, to spread the link traffic.
    comptime SrcPtrType = ImmPointer[Scalar[dtype], ImmutAnyOrigin]
    var srcs_rr = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> SrcPtrType: srcs[
            circular_add[ngpus](_my_rank, i)
        ]
    )

    var _in_inner = Int(in_inner)
    var _out_inner = Int(out_inner)
    var num_vectors = Int(chunk_len) // simd_width
    var in_row = ngpus * _in_inner
    var out_row = ngpus * _out_inner

    with PDL():
        # The peers' inputs are ready to read.
        _multi_gpu_barrier[ngpus, is_start=True, domain_id=domain_id](
            rank_sigs, my_sig, _my_rank
        )

        for idx in range(global_tid, num_vectors, stride):
            var f = idx * simd_width
            var src_off = (
                (f // _in_inner) * in_row + _my_rank * _in_inner + f % _in_inner
            )
            var dst_base = (f // _out_inner) * out_row + f % _out_inner
            # Issue every peer's read before any store, for `ngpus`-way
            # memory-level parallelism.
            var data = Array[SIMD[dtype, simd_width], ngpus](uninitialized=True)
            comptime for i in range(ngpus):
                data[i] = (
                    srcs_rr[i]
                    .unsafe_address_space_cast[_target_address_space]()
                    .unsafe_load[width=simd_width, alignment=alignment](src_off)
                )
            comptime for i in range(ngpus):
                var peer = circular_add[ngpus](_my_rank, i)
                output.unsafe_address_space_cast[
                    _target_address_space
                ]().unsafe_store[width=simd_width, alignment=alignment](
                    dst_base + peer * _out_inner, data[i]
                )

        # The peers are done reading this rank's input.
        _multi_gpu_barrier[
            ngpus, is_start=False, sync_after_wait=True, domain_id=domain_id
        ](rank_sigs, my_sig, _my_rank)


@inline(.always)
def _launch_all_to_all[
    dtype: DType,
    group_size: Int,
    simd_width: Int,
    domain_id: Int,
](
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
    group_srcs: Array[ImmPointer[Scalar[dtype], ImmutAnyOrigin], group_size],
    group_sigs: Array[MutPointer[Signal, MutAnyOrigin], group_size],
    chunk_len: Int,
    in_inner: Int,
    out_inner: Int,
    max_num_blocks: Int,
    ctx: DeviceContext,
    loc_rank: Int,
) raises:
    comptime BLOCK_SIZE = 256
    var grid_size = max(
        1,
        min(
            max_num_blocks,
            ceildiv(ceildiv(chunk_len, simd_width), BLOCK_SIZE),
        ),
    )
    comptime kernel = _all_to_all_p2p_kernel[
        dtype,
        group_size,
        simd_width,
        BLOCK_SIZE=BLOCK_SIZE,
        domain_id=domain_id,
    ]
    ctx.enqueue_function[kernel](
        output,
        group_srcs,
        group_sigs,
        Int64(chunk_len),
        Int64(in_inner),
        Int64(out_inner),
        Int32(loc_rank),
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


@inline(.always)
def all_to_all[
    dtype: DType,
    ngpus: Int,
    *,
    group_size: Int = ngpus,
](
    input_ptrs: Array[ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus],
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
    num_elements: Int,
    in_inner: Int,
    out_inner: Int,
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    ctx: DeviceContext,
    my_rank: Int,
    _max_num_blocks: Optional[Int] = None,
) raises:
    """Per-device all-to-all: one instance per GPU fills its own output.

    Rank `r` of a group splits its input into `group_size` chunks and sends
    chunk `j` to the group's rank `j`; each rank's output is the chunks it
    receives, from source rank `0` to `group_size - 1`. In tensor terms,
    with the input split along axis `a` and the output concatenated along
    axis `b`:

    - `in_inner` is the product of the input's dims from `a` on, divided by
      `group_size`: one chunk's contiguous run in the input.
    - `out_inner` is the product of the output's dims from `b` on, divided
      by `group_size`: one chunk's contiguous run in the output.

    `in_inner == out_inner == num_elements / group_size` is the plain
    `all_to_all_single` over a leading axis. The caller launches one
    instance per device in parallel (e.g. via `_launch_device_collective`).

    World view vs. group: `ngpus` is the total number of devices;
    `input_ptrs` and `rank_sigs` carry every device's, indexed by global
    rank, and `my_rank` is this device's global rank. Devices
    `[g * group_size, (g + 1) * group_size)` form group `g`. Every input of
    a group holds `num_elements` elements.

    Parameters:
        dtype: Data type of the tensor elements.
        ngpus: Total number of devices in the world.
        group_size: Number of devices per independent all-to-all group. Must
            evenly divide `ngpus`. Defaults to `ngpus`.

    Args:
        input_ptrs: Every device's input, indexed by global rank.
        output: This device's output, `num_elements` elements.
        num_elements: Elements in every input and output of this group.
        in_inner: One chunk's contiguous run in the input, in elements.
        out_inner: One chunk's contiguous run in the output, in elements.
        rank_sigs: Every device's Signal pointer, indexed by global rank.
        ctx: Device context for this GPU.
        my_rank: Global rank of this GPU in `[0, ngpus)`.
        _max_num_blocks: Maximum number of blocks for the kernel launch.

    Raises:
        Error: P2P access is not available, or the chunk geometry does not
            divide `num_elements`.
    """
    comptime assert (
        group_size >= 2
    ), "all_to_all requires at least 2 GPUs per group"
    comptime assert (
        ngpus % group_size == 0
    ), "group_size must evenly divide ngpus"
    comptime domain_id = 0 if group_size == ngpus else group_size

    var group_start = ualign_down(my_rank, group_size)
    var loc_rank = my_rank - group_start

    if num_elements % group_size != 0:
        raise Error(
            String(
                "all_to_all: ",
                num_elements,
                " elements do not split into ",
                group_size,
                " chunks",
            )
        )
    var chunk_len = num_elements // group_size
    # Shapes are uniform within a group, so every rank returns here together.
    if chunk_len == 0:
        return
    if (
        in_inner <= 0
        or out_inner <= 0
        or chunk_len % in_inner != 0
        or chunk_len % out_inner != 0
    ):
        raise Error(
            String(
                "all_to_all: runs of ",
                in_inner,
                " and ",
                out_inner,
                " elements do not tile a chunk of ",
                chunk_len,
            )
        )

    if not is_p2p_enabled():
        raise Error("all_to_all requires P2P access between GPUs")

    comptime SrcPtrType = ImmPointer[Scalar[dtype], ImmutAnyOrigin]
    var group_srcs = Array[_, group_size](
        fill_with_unrolled=lambda [i: Int]() -> SrcPtrType: input_ptrs[
            group_start + i
        ]
    )
    var group_sigs = Array[_, group_size](
        fill_with_unrolled=lambda [i: Int]() -> MutPointer[
            Signal, MutAnyOrigin
        ]: rank_sigs[group_start + i]
    )

    comptime sm_version = ctx.default_device_info.version
    var max_num_blocks = _max_num_blocks.or_else(
        dispatch_select_comm_config[
            group_size, sm_version, allgather_tuning_table
        ](num_elements * size_of[dtype]()).get_num_blocks()
    )

    # Vectorize when no vector straddles a run and every run starts
    # aligned; graph buffers are allocated aligned.
    comptime simd_width = simd_width_of[dtype, target=ctx.target]()
    comptime vector_bytes = align_of[SIMD[dtype, simd_width]]()
    var aligned = in_inner % simd_width == 0 and out_inner % simd_width == 0
    aligned = aligned and Int(output) % vector_bytes == 0
    comptime for i in range(group_size):
        aligned = aligned and Int(group_srcs[i]) % vector_bytes == 0

    if aligned:
        _launch_all_to_all[dtype, group_size, simd_width, domain_id](
            output,
            group_srcs,
            group_sigs,
            chunk_len,
            in_inner,
            out_inner,
            max_num_blocks,
            ctx,
            loc_rank,
        )
    else:
        _launch_all_to_all[dtype, group_size, 1, domain_id](
            output,
            group_srcs,
            group_sigs,
            chunk_len,
            in_inner,
            out_inner,
            max_num_blocks,
            ctx,
            loc_rank,
        )
