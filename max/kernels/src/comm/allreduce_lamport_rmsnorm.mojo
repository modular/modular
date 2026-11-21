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
"""Fused one-shot Lamport allreduce + RMSNorm (high-perf protocol).

Grafts the barrier-free Lamport comm protocol from `comm/lamport.mojo` +
`comm/allreduce.mojo` (in-register local pack, poll-all-peers arrival, a sum in
global rank order, 3-way generation rotation, fused clear, the device-resident
`Signal.lamport_region` / `lamport_state`) onto a row-blocked RMSNorm epilogue
so the whole TP allreduce+RMSNorm producer is a single kernel.

Layout: with the 128-bit Lamport pack (`atomic_width` = 16B / sizeof(dtype) =
8 for bf16) one thread owns exactly one pack = one column group of a row
(token). By default one block owns whole rows. For decode-sized messages on 8
GPUs, where pushing every pack of a row from one SM is the bottleneck, a row is
instead split across a thread-block cluster of smaller blocks. After the
allreduce reduces each pack in-register, the block (or cluster) reduces the
row's sum-of-squares and normalizes.

An optional residual is added to the reduced sum before the norm, and the
pre-norm sum is written out as the new residual stream. That is the form the
graph compiler's fused allreduce + add + RMSNorm op dispatches to for
latency-bound payloads (`comm.allreduce_residual_rmsnorm`).

PDL is on by default: wait for the producer, then trigger the consumer
immediately (early trigger) so a following GEMM overlaps this kernel; consumer
correctness is its own `wait_on_dependent_grids`. The signal buffers must be
sentinel-initialized once (`lamport_init`) before the first call.
"""

from std.math import align_down, rsqrt
from std.sys import align_of, size_of

from std.atomic import Atomic
from std.collections import Array
from std.collections.optional import OptionalReg

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, Dim
from max.gpu.host.info import H100
from max.gpu.primitives import block
from max.gpu.primitives.cluster import block_rank_in_cluster, cluster_allreduce
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from layout import (
    Idx,
    ImmTileTensor,
    MutTileTensor,
    TensorLayout,
    row_major,
)
from layout.tile_tensor import _ComptimeConditionalTileTensor
from std.memory import unsafe_stack_allocation
from std.utils import StaticTuple
from std.utils.numerics import get_accum_type

from .sync import (
    MAX_GPUS,
    MAX_NUM_BLOCKS_UPPER_BOUND,
    Signal,
    circular_add,
    is_p2p_enabled,
)
from .lamport import (
    Lamport,
    LamportGeneration,
    has_neg_zero,
    remove_neg_zero,
    set_neg_zero,
)

comptime _ZeroSizedLayout = type_of(row_major(Idx[0], Idx[0]))


@inline(.always)
def _sum(x: SIMD, y: type_of(x)) -> type_of(x):
    return x + y


# Without the bound ptxas can allocate more registers than a 1024-thread block
# may hold (the 8-GPU instantiation keeps eight peer packs live and fails to
# launch).
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(
    t"allreduce_lamport_rmsnorm_{dtype}_{ngpus}_{BLOCK_SIZE}x{CLUSTER_SIZE}_{has_residual}_{multiply_before_cast}_{post_norm_residual}"
)
@__allow_legacy_any_origin_fields
def _allreduce_lamport_rmsnorm_kernel[
    ResidualLayoutType: TensorLayout,
    residual_origin: Origin,
    ResidualOutLayoutType: TensorLayout,
    residual_out_origin: MutOrigin,
    Gamma2LayoutType: TensorLayout,
    gamma2_origin: ImmOrigin,
    MixLayoutType: TensorLayout,
    mix_origin: ImmOrigin,
    //,
    dtype: DType,
    ngpus: Int,
    BLOCK_SIZE: Int,
    *,
    CLUSTER_SIZE: Int = 1,
    has_residual: Bool = False,
    multiply_before_cast: Bool = True,
    post_norm_residual: Bool = False,
](
    result: MutPointer[Scalar[dtype], MutAnyOrigin],
    src: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    gamma: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    rows_dev: Int32,
    cols_dev: Int32,
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    my_rank_dev: Int32,
    residual: _ComptimeConditionalTileTensor[
        dtype, ResidualLayoutType, residual_origin, engaged=has_residual
    ],
    residual_output: _ComptimeConditionalTileTensor[
        dtype, ResidualOutLayoutType, residual_out_origin, engaged=has_residual
    ],
    gamma2: _ComptimeConditionalTileTensor[
        dtype, Gamma2LayoutType, gamma2_origin, engaged=post_norm_residual
    ],
    mix: _ComptimeConditionalTileTensor[
        DType.float32, MixLayoutType, mix_origin, engaged=post_norm_residual
    ],
):
    """Row-blocked fused Lamport allreduce + RMSNorm; one 128-bit pack per thread.

    A row is split across the `CLUSTER_SIZE` blocks of a thread-block cluster,
    each owning `BLOCK_SIZE` consecutive packs; the blocks combine their partial
    sums of squares through distributed shared memory. Spreading a row over
    several SMs is what keeps the peer pushes and polls of a decode-sized
    message from serializing on one SM.

    With `post_norm_residual`, the residual joins after the norm instead of
    before it: `h = mix[0] * normed + mix[1] * residual`, each product and
    the sum rounded to `dtype` as separate bf16 ops would, goes to
    `residual_output`, and `result` gets `h` normed again with `gamma2`.
    That is a transformer block's post-norm, scaled residual join and the
    next pre-norm in one launch. Single-block rows only.
    """
    comptime assert not post_norm_residual or (
        has_residual and CLUSTER_SIZE == 1 and multiply_before_cast
    ), "post_norm_residual needs a residual, 1-block rows, gamma before cast"
    var rows = Int(rows_dev)
    var cols = Int(cols_dev)
    var my_rank = Int(my_rank_dev)
    comptime accum_type = get_accum_type[dtype]()
    # Pin to the 128-bit single-copy-atomic pack the sentinel protocol needs.
    comptime atomic_width = Lamport.ATOMIC_BYTES // size_of[dtype]()
    comptime assert (
        atomic_width * size_of[dtype]() == 16
    ), "Lamport pack must be exactly 16 bytes (the 128-bit atomic width)."
    comptime alignment = align_of[SIMD[dtype, atomic_width]]()
    # Evidence for the (row, col) residual loads/stores below.
    comptime assert residual.T.flat_rank >= 2
    comptime assert residual_output.T.flat_rank >= 2

    var tid = thread_idx.x
    var cta_rank = 0
    comptime if CLUSTER_SIZE > 1:
        cta_rank = Int(block_rank_in_cluster())
    # This thread's pack within a row.
    var pack = cta_rank * BLOCK_SIZE + tid
    var col = pack * atomic_width
    var is_valid = col < cols
    var packs_per_row = cols // atomic_width
    var cluster_slot = unsafe_stack_allocation[
        2, Scalar[accum_type], address_space=.SHARED
    ]()

    # PDL: wait for the producer, then release the consumer immediately so it
    # overlaps the whole AR+RMSNorm. Called by all threads in the block before
    # any divergence.
    wait_on_dependent_grids()
    launch_dependent_grids()

    # This rank's own region (polled) + peer regions in round-robin order.
    var my_region = rank_sigs[my_rank][].lamport_region_ptr[dtype]()
    comptime PtrType = MutPointer[Scalar[dtype], MutAnyOrigin]
    var peer_regions = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> PtrType: rank_sigs[
            circular_add[ngpus](my_rank, i)
        ][].lamport_region_ptr[dtype]()
    )
    var sentinel = set_neg_zero[dtype, atomic_width]()

    # Generation geometry (read the device-resident counter once per call).
    var state = rank_sigs[my_rank][].lamport_state_ptr()
    var flag = Int(
        state.unsafe_load[width=1, volatile=True](Lamport.STATE_FLAG)
    )
    # Previous call's written extent in dtype-independent 16-byte packs.
    var clear_packs = Int(
        state.unsafe_load[width=1, volatile=True](Lamport.STATE_PREV_PACKS)
    )
    comptime gen_stride_packs = ngpus * Lamport.MAX_PACKS
    var data_gen_off = LamportGeneration.data_index(flag) * gen_stride_packs
    var clear_gen_off = LamportGeneration.clear_index(flag) * gen_stride_packs

    # Preload this thread's gamma columns once (reused across all rows).
    var gamma_vec = SIMD[accum_type, atomic_width](0)
    var gamma2_vec = SIMD[accum_type, atomic_width](0)
    var mix_normed = Scalar[accum_type](0)
    var mix_residual = Scalar[accum_type](0)
    if is_valid:
        gamma_vec = (
            gamma.unsafe_load[width=atomic_width, alignment=alignment](
                col
            ).cast[accum_type]()
            + weight_offset.cast[accum_type]()
        )
        comptime if post_norm_residual:
            gamma2_vec = (
                gamma2[].load[width=atomic_width]((col,)).cast[accum_type]()
                + weight_offset.cast[accum_type]()
            )
    comptime if post_norm_residual:
        mix_normed = mix[].load[width=1]((0,)).cast[accum_type]()
        mix_residual = mix[].load[width=1]((1,)).cast[accum_type]()

    for row in range(
        block_idx.x // CLUSTER_SIZE, rows, grid_dim.x // CLUSTER_SIZE
    ):
        var base = row * cols + col
        var global_pack = row * packs_per_row + pack

        var accum = SIMD[accum_type, atomic_width](0)
        var residual_pack = SIMD[dtype, atomic_width](0)
        if is_valid:
            # (a) Local input pack -> producer-sanitize; kept in-register, so
            # this rank only touches the ngpus-1 remote slots.
            var local = remove_neg_zero[dtype, atomic_width](
                src.unsafe_load[width=atomic_width, alignment=alignment](base)
            )
            # Issued before the exchange so its latency hides behind the poll.
            comptime if has_residual:
                residual_pack = residual[].load[width=atomic_width]((row, col))

            # (b) Push the sanitized pack into each PEER's slot (skip self).
            var push_off = (
                data_gen_off + my_rank * Lamport.MAX_PACKS + global_pack
            )
            comptime for i in range(1, ngpus):
                peer_regions[i].unsafe_store[
                    width=atomic_width, alignment=alignment, volatile=True
                ](push_off * atomic_width, local)

            # (d) Poll all ngpus-1 remote slots until none hold the sentinel
            # (observes parallel arrival rather than per-peer spinning), then
            # (e) reduce in global rank order 0 .. ngpus-1 in accum_type: a
            # per-rank order rounds the sum differently on each rank. Branching
            # once on the rank makes every slot index a compile-time constant,
            # so the polls keep the round-robin order and `packs` stays in
            # registers.
            var packs = Array[SIMD[dtype, atomic_width], ngpus](
                uninitialized=True
            )
            comptime if ngpus == 2:
                # Two terms sum to the same bits in either order, so the
                # local-first sum is already rank-consistent.
                var done = False
                while not done:
                    var slot_off = (
                        data_gen_off
                        + (1 - my_rank) * Lamport.MAX_PACKS
                        + global_pack
                    )
                    packs[1] = my_region.unsafe_load[
                        width=atomic_width, alignment=alignment, volatile=True
                    ](slot_off * atomic_width)
                    done = not has_neg_zero[dtype, atomic_width](packs[1])
                accum = local.cast[accum_type]() + packs[1].cast[accum_type]()
            else:
                comptime for r in range(ngpus):
                    if my_rank == r:
                        packs[r] = local
                        var done = False
                        while not done:
                            done = True
                            comptime for i in range(1, ngpus):
                                comptime g = (r + i) % ngpus
                                var slot_off = (
                                    data_gen_off
                                    + g * Lamport.MAX_PACKS
                                    + global_pack
                                )
                                packs[g] = my_region.unsafe_load[
                                    width=atomic_width,
                                    alignment=alignment,
                                    volatile=True,
                                ](slot_off * atomic_width)
                                if has_neg_zero[dtype, atomic_width](packs[g]):
                                    done = False
                        accum = packs[0].cast[accum_type]()
                        comptime for g in range(1, ngpus):
                            accum += packs[g].cast[accum_type]()

            # The residual is replicated, so each rank adds it locally rather
            # than through the reduction.
            comptime if has_residual and not post_norm_residual:
                accum += residual_pack.cast[accum_type]()
                residual_output[].store((row, col), accum.cast[dtype]())

        # RMSNorm over the row (block reduction; invalid threads add 0).
        var thread_sumsq = (accum * accum).reduce_add()
        var row_sumsq: Scalar[accum_type]
        comptime if CLUSTER_SIZE == 1:
            row_sumsq = block.sum[block_size=BLOCK_SIZE, broadcast=True](
                thread_sumsq
            )
        else:
            row_sumsq = cluster_allreduce[_sum, CLUSTER_SIZE](
                cluster_slot,
                block.sum[block_size=BLOCK_SIZE, broadcast=False](thread_sumsq),
            )[0]
        var norm = rsqrt(
            row_sumsq / Scalar[accum_type](cols) + epsilon.cast[accum_type]()
        )
        comptime if post_norm_residual:
            var joined = SIMD[accum_type, atomic_width](0)
            if is_valid:
                var normed = ((accum * norm) * gamma_vec).cast[dtype]()
                var scaled_normed = (
                    normed.cast[accum_type]() * mix_normed
                ).cast[dtype]()
                var scaled_residual = (
                    residual_pack.cast[accum_type]() * mix_residual
                ).cast[dtype]()
                var h = (
                    scaled_normed.cast[accum_type]()
                    + scaled_residual.cast[accum_type]()
                ).cast[dtype]()
                comptime if has_residual:
                    residual_output[].store((row, col), h)
                joined = h.cast[accum_type]()
            var row_sumsq2 = block.sum[block_size=BLOCK_SIZE, broadcast=True](
                (joined * joined).reduce_add()
            )
            var norm2 = rsqrt(
                row_sumsq2 / Scalar[accum_type](cols)
                + epsilon.cast[accum_type]()
            )
            if is_valid:
                result.unsafe_store[width=atomic_width, alignment=alignment](
                    base, ((joined * norm2) * gamma2_vec).cast[dtype]()
                )
        else:
            if is_valid:
                var normalized: SIMD[dtype, atomic_width]
                comptime if multiply_before_cast:
                    normalized = ((accum * norm) * gamma_vec).cast[dtype]()
                else:
                    normalized = (accum * norm).cast[dtype]() * gamma_vec.cast[
                        dtype
                    ]()
                result.unsafe_store[width=atomic_width, alignment=alignment](
                    base, normalized
                )

        # (f) Fused clear: reset this pack's remote slots in the generation
        # reused two calls from now, within the previous writer's extent.
        if global_pack < clear_packs:
            comptime for i in range(1, ngpus):
                var peer = circular_add[ngpus](my_rank, i)
                var clr_off = (
                    clear_gen_off + peer * Lamport.MAX_PACKS + global_pack
                )
                my_region.unsafe_store[
                    width=atomic_width, alignment=alignment, volatile=True
                ](clr_off * atomic_width, sentinel)

    # (f-tail) If the previous call was LARGER, clear the leftover packs of the
    # reused generation here (empty in the common same-size case).
    var cur_packs = rows * packs_per_row
    var gtid = block_idx.x * BLOCK_SIZE + tid
    var nthreads = grid_dim.x * BLOCK_SIZE
    for cp in range(cur_packs + gtid, clear_packs, nthreads):
        comptime for i in range(1, ngpus):
            var peer = circular_add[ngpus](my_rank, i)
            var clr_off = clear_gen_off + peer * Lamport.MAX_PACKS + cp
            my_region.unsafe_store[
                width=atomic_width, alignment=alignment, volatile=True
            ](clr_off * atomic_width, sentinel)

    # (g) Advance this rank's generation flag exactly once per call via an
    # intra-GPU block-arrival counter (device-side state -> CUDA-graph-safe).
    barrier()
    if tid == 0:
        var arrived = Atomic.fetch_add(
            state.unsafe_offset(Lamport.STATE_ARRIVAL), UInt32(1)
        )
        if Int(arrived) == grid_dim.x - 1:
            state.unsafe_offset(Lamport.STATE_FLAG).unsafe_store[volatile=True](
                UInt32(LamportGeneration.advance(flag))
            )
            (state.unsafe_offset(Lamport.STATE_PREV_PACKS)).unsafe_store[
                volatile=True
            ](UInt32(cur_packs))
            (state.unsafe_offset(Lamport.STATE_ARRIVAL)).unsafe_store[
                volatile=True
            ](UInt32(0))


comptime _SINGLE_BLOCK_SIZE = align_down(
    DeviceContext[].default_device_info.max_thread_block_size,
    DeviceContext[].default_device_info.warp_size,
)
comptime _CLUSTER_BLOCK_SIZE = 256
comptime _MAX_CLUSTER_SIZE = 8
# 8xB200, bf16: splitting rows across clusters wins through 32x7168 (448 KiB)
# and 64x4096 (512 KiB) and loses from 48x7168 (672 KiB) on. On 2 and 4 GPUs
# one block per row is as fast or within a few percent at every decode size.
comptime _CLUSTER_MAX_BYTES = 512 * 1024


@inline(.always)
def _clusters_supported[ngpus: Int]() -> Bool:
    comptime info = DeviceContext[].default_device_info
    return ngpus >= 8 and info.api == "cuda" and info.compute >= H100.compute


@inline(.always)
def _max_packs_per_row[
    ngpus: Int, *, post_norm_residual: Bool = False
](num_bytes: Int) -> Int:
    """Returns the widest row the layout chosen for `num_bytes` can hold."""
    # The post-norm residual join's second norm reduces within one block,
    # so it never spreads a row over a cluster.
    comptime if _clusters_supported[ngpus]() and not post_norm_residual:
        if num_bytes <= _CLUSTER_MAX_BYTES:
            return _CLUSTER_BLOCK_SIZE * _MAX_CLUSTER_SIZE
    return _SINGLE_BLOCK_SIZE


def lamport_allreduce_rmsnorm_fits[
    dtype: DType, ngpus: Int
](rows: Int, cols: Int) -> Bool:
    """Returns whether a `[rows, cols]` message fits the Lamport fused kernel.

    The kernel needs whole 128-bit packs, at most one pack per thread of a
    row's blocks, and a message within the reserved Lamport workspace.

    Parameters:
        dtype: Element type of the message.
        ngpus: Number of participating GPUs.

    Args:
        rows: Number of rows (tokens).
        cols: Number of columns (hidden size); RMSNorm runs over it.

    Returns:
        True when the kernel accepts the shape.
    """
    comptime atomic_width = Lamport.ATOMIC_BYTES // size_of[dtype]()
    var num_bytes = rows * cols * size_of[dtype]()
    return (
        cols % atomic_width == 0
        and cols // atomic_width <= _max_packs_per_row[ngpus](num_bytes)
        and num_bytes <= Lamport.MAX_SMALL_MESSAGE_BYTES
    )


def _lamport_allreduce_rmsnorm_launch[
    dtype: DType,
    ngpus: Int,
    *,
    has_residual: Bool = False,
    multiply_before_cast: Bool = True,
    post_norm_residual: Bool = False,
](
    my_rank: Int,
    src: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    gamma: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    rows: Int,
    cols: Int,
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    ctx: DeviceContext,
    residual: _ComptimeConditionalTileTensor[
        dtype, engaged=has_residual, ...
    ] = _ComptimeConditionalTileTensor[
        dtype, _ZeroSizedLayout, ImmutAnyOrigin, engaged=False
    ](),
    residual_output: _ComptimeConditionalTileTensor[
        mut=True, dtype, engaged=has_residual, ...
    ] = _ComptimeConditionalTileTensor[
        dtype, _ZeroSizedLayout, MutAnyOrigin, engaged=False
    ](),
    gamma2: _ComptimeConditionalTileTensor[
        mut=False, dtype, engaged=post_norm_residual, ...
    ] = _ComptimeConditionalTileTensor[
        dtype, _ZeroSizedLayout, ImmUntrackedOrigin, engaged=False
    ](),
    mix: _ComptimeConditionalTileTensor[
        mut=False, DType.float32, engaged=post_norm_residual, ...
    ] = _ComptimeConditionalTileTensor[
        DType.float32, _ZeroSizedLayout, ImmUntrackedOrigin, engaged=False
    ](),
) raises:
    """Validates the shape and launches the fused Lamport kernel on this GPU.

    See `lamport_allreduce_rmsnorm` for the protocol. With `has_residual`,
    `residual` (`[rows, cols]`, replicated on every rank) is added to the
    reduced sum before the norm and the sum is stored to `residual_output`.
    With `post_norm_residual` as well, the residual joins after the norm
    instead, by the weights in `mix`, and `gamma2` norms the join; see
    `lamport_allreduce_post_norm_residual_rmsnorm`.
    `multiply_before_cast` matches `ops.rms_norm`: True applies gamma in
    float32 before the single cast, False casts the normalized value first and
    applies gamma in `dtype`.
    """
    comptime assert ngpus >= 2 and ngpus <= MAX_GPUS, "ngpus must be in [2, 8]"
    comptime assert (
        dtype.is_floating_point()
    ), "lamport_allreduce_rmsnorm requires a floating-point dtype"
    comptime atomic_width = Lamport.ATOMIC_BYTES // size_of[dtype]()

    if cols % atomic_width != 0:
        raise Error(
            "lamport_allreduce_rmsnorm requires cols % atomic_width == 0 (whole"
            " 128-bit packs)"
        )
    var packs_per_row = cols // atomic_width
    var num_bytes = rows * cols * size_of[dtype]()
    if packs_per_row > _max_packs_per_row[
        ngpus, post_norm_residual=post_norm_residual
    ](num_bytes):
        raise Error(
            "lamport_allreduce_rmsnorm: cols too large for one pack/thread"
        )
    if num_bytes > Lamport.MAX_SMALL_MESSAGE_BYTES:
        raise Error(
            "lamport_allreduce_rmsnorm message exceeds the reserved Lamport"
            " workspace"
        )
    # Every rank sees the same shape, so all of them skip together and the
    # generation counters stay in step.
    if rows == 0:
        return

    @__allow_legacy_any_origin_fields
    def launch[block_size: Int, cluster_size: Int]() raises {imm}:
        comptime kernel = _allreduce_lamport_rmsnorm_kernel[
            ResidualLayoutType=residual.T.LayoutType,
            residual_origin=residual.T.origin,
            ResidualOutLayoutType=residual_output.T.LayoutType,
            residual_out_origin=residual_output.T.origin,
            Gamma2LayoutType=gamma2.T.LayoutType,
            gamma2_origin=gamma2.T.origin,
            MixLayoutType=mix.T.LayoutType,
            mix_origin=mix.T.origin,
            dtype,
            ngpus,
            block_size,
            CLUSTER_SIZE=cluster_size,
            has_residual=has_residual,
            multiply_before_cast=multiply_before_cast,
            post_norm_residual=post_norm_residual,
        ]
        var num_clusters = min(MAX_NUM_BLOCKS_UPPER_BOUND // cluster_size, rows)
        ctx.enqueue_function[kernel](
            dst,
            src,
            gamma,
            rank_sigs,
            Int32(rows),
            Int32(cols),
            epsilon,
            weight_offset,
            Int32(my_rank),
            residual,
            residual_output,
            gamma2,
            mix,
            grid_dim=num_clusters * cluster_size,
            block_dim=block_size,
            cluster_dim=OptionalReg(Dim(cluster_size, 1, 1)) if cluster_size
            > 1 else None,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )

    # The post-norm residual join's second norm reduces within one block.
    comptime if _clusters_supported[ngpus]() and not post_norm_residual:
        if num_bytes <= _CLUSTER_MAX_BYTES:
            # The narrowest power-of-two cluster whose blocks cover the row.
            comptime for cluster_size in [1, 2, 4, 8]:
                if packs_per_row <= _CLUSTER_BLOCK_SIZE * cluster_size:
                    return launch[_CLUSTER_BLOCK_SIZE, cluster_size]()
    launch[_SINGLE_BLOCK_SIZE, 1]()


def lamport_allreduce_rmsnorm[
    dtype: DType,
    ngpus: Int,
](
    my_rank: Int,
    src: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    gamma: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    rows: Int,
    cols: Int,
    epsilon: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    """Per-device fused Lamport allreduce (sum) + RMSNorm (high-perf protocol).

    `src`/`dst` are `[rows, cols]` (tokens x hidden); RMSNorm runs over `cols`.
    Uses the device-resident `Signal.lamport_region` / `lamport_state` (so the
    signal buffers must be `lamport_init`-ed once before the first call, then one
    host sync). The generation flag advances in-kernel, so repeated calls --
    including CUDA-graph replay -- rotate the buffers correctly with no per-call
    fill or phase argument.

    The launch carries the PDL attribute, and the kernel waits for its
    predecessor before any read and releases its dependents at entry so a
    following GEMM overlaps the whole AR+RMSNorm.

    Parameters:
        dtype: Element type (bf16/fp16/fp32 transport).
        ngpus: Number of participating GPUs.
    """
    _lamport_allreduce_rmsnorm_launch[dtype, ngpus](
        my_rank,
        src,
        dst,
        gamma,
        rank_sigs,
        rows,
        cols,
        epsilon,
        Scalar[dtype](0),
        ctx,
    )


def lamport_allreduce_post_norm_residual_rmsnorm[
    dtype: DType,
    ngpus: Int,
    //,
](
    my_rank: Int,
    src: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    residual: ImmTileTensor[dtype, ...],
    joined: MutTileTensor[dtype, ...],
    normed: MutPointer[Scalar[dtype], MutAnyOrigin],
    gamma: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    gamma2: ImmTileTensor[dtype, ...],
    mix: ImmTileTensor[DType.float32, ...],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    rows: Int,
    cols: Int,
    epsilon: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    """Per-device Lamport allreduce, post-norm, residual join and pre-norm.

    For each `[rows, cols]` row, with `s` the allreduce sum of `src`:

        joined = bf16(bf16(mix[0] * rmsnorm(s, gamma))
                      + bf16(mix[1] * residual))
        normed = rmsnorm(joined, gamma2)

    Each product and the sum round to `dtype` on their own, as the unfused
    ops do, and both norms multiply by gamma in float32 before the cast. The
    protocol is `lamport_allreduce_rmsnorm`'s, with one block per row, so a
    row holds at most one 128-bit pack per thread of a block. A message past
    the Lamport workspace runs as consecutive row chunks that each fit it.
    The kernel stores straight into the peers' signal buffers, so it needs
    P2P access between the GPUs and raises without it.

    Parameters:
        dtype: Element type (bf16/fp16/fp32 transport), inferred.
        ngpus: Number of participating GPUs, inferred from `rank_sigs`.

    Args:
        my_rank: This GPU's rank.
        src: This rank's `[rows, cols]` partial sum.
        residual: The `[rows, cols]` residual stream, replicated.
        joined: Receives the joined residual stream.
        normed: Receives `joined` normed with `gamma2`.
        gamma: The first norm's weight, `[cols]`.
        gamma2: The second norm's weight, `[cols]`.
        mix: On the device, `[2]` float32: the weight of the normed sum, then
            of the residual.
        rank_sigs: Every rank's signal buffer.
        rows: Rows of the message.
        cols: Columns; both norms run over them.
        epsilon: Both norms' epsilon.
        ctx: This GPU's context.
    """
    # `is_p2p_enabled` is process-wide, so every rank raises before any of
    # them enqueues.
    if not is_p2p_enabled():
        raise Error(
            "lamport_allreduce_post_norm_residual_rmsnorm requires P2P access"
            " between GPUs"
        )
    # Every rank takes the same chunks, so the Lamport generations, which
    # rotate once per launch, stay in step across the ranks.
    var chunk_rows = max(
        1, Lamport.MAX_SMALL_MESSAGE_BYTES // (cols * size_of[dtype]())
    )
    for start in range(0, rows, chunk_rows):
        var end = min(start + chunk_rows, rows)
        var offset = start * cols
        _lamport_allreduce_rmsnorm_launch[
            dtype, ngpus, has_residual=True, post_norm_residual=True
        ](
            my_rank,
            src.unsafe_offset(offset),
            normed.unsafe_offset(offset),
            gamma,
            rank_sigs,
            end - start,
            cols,
            epsilon,
            Scalar[dtype](0),
            ctx,
            residual[start:end, :],
            joined[start:end, :],
            gamma2,
            mix,
        )
