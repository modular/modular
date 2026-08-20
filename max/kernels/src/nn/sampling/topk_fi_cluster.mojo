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
"""Cluster-launched top-k/top-p kernels.

One block per row uses only as many SMs as there are rows, which leaves most
of the GPU idle at decode batch sizes. The kernels here spread each row over
the CTAs of one thread-block cluster and combine the per-row reductions over
distributed shared memory. The launchers fall back to the single-block kernels
in `topk_fi` when a CTA's slice does not fit in shared memory or when the
target has no clusters.
"""

from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx
from std.gpu.primitives import warp
from std.gpu.primitives.id import cluster_dim
from max.gpu.primitives import block
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_allreduce,
    cluster_sync,
)
from max.gpu.primitives.grid_controls import (
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
    PDLLevel,
)
from max.gpu.host import DeviceContext, Dim, FuncAttribute
from max.gpu.memory import external_memory
from std.sys.info import is_apple_gpu
from layout import (
    ComptimeInt,
    Coord,
    Idx,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from layout.tile_layout import Layout
from std.math import ceildiv, exp, gcd
from std.memory import bitcast, unsafe_stack_allocation
from max.runtime.tracing import Trace, TraceLevel, trace_arg
from std.sys import size_of
from std.utils.static_tuple import StaticTuple
from .topk_fi import (
    TopKTopPMaskedProbsKernel,
    ValueCount,
    _block_reduce_value_count,
)

# The largest cluster that these kernels use. All CTAs of a cluster must run
# at the same time on one GPC. Above four CTAs the scheduler gives back more
# than the added parallelism supplies.
comptime _MAX_CUTOFF_CLUSTER = 4

# The largest shared-memory request that a CTA makes to stage its part of a
# row. B200 supplies 228 KiB per SM. Keep some for the reduction scratch and
# the publish slots.
comptime _MAX_STAGE_SMEM_BYTES = 216 * 1024


def _stage_smem_bytes(d: Int, vec_size: Int, cluster_size: Int) -> Int:
    """Bytes one CTA needs to stage its contiguous slice of the row."""
    return (
        ceildiv(ceildiv(d, vec_size), cluster_size)
        * vec_size
        * size_of[DType.float32]()
    )


@always_inline
@__parameter
def _max(x: SIMD, y: type_of(x)) -> type_of(x):
    return max(x, y)


@always_inline
@__parameter
def _sum(x: SIMD, y: type_of(x)) -> type_of(x):
    return x + y


comptime _CUTOFF_SEARCH_MAX_ITERS = 64

# A publish slot holds the widest vector that the CTAs of a cluster combine,
# twice over: once for the peers to read, and once for the combined result of
# this CTA.
comptime _CLUSTER_SLOT_FLOATS = 16


@always_inline
def _block_reduce_cutoff_stats[
    block_size: Int, n: Int, broadcast: Bool = True
](vals: StaticTuple[Scalar[DType.float32], n]) -> StaticTuple[
    Scalar[DType.float32], n
]:
    """Reduces one round of cutoff statistics across the block in one pass.

    Lane 0 holds a minimum, lane 1 a maximum, and the remaining lanes hold
    sums. Counts are carried as floats, which is exact below 2^24.
    """

    @always_inline
    @__parameter
    def _reduce_fn[
        dtype: DType, width: SIMDLength, reduction_idx: Int
    ](v: SIMD[dtype, width]) -> Scalar[dtype]:
        comptime if reduction_idx == 0:
            return warp.min(v)
        elif reduction_idx == 1:
            return warp.max(v)
        else:
            return warp.sum(v)

    var initial = StaticTuple[Scalar[DType.float32], n](0)
    initial[0] = Float32.MAX_FINITE
    initial[1] = Float32.MIN_FINITE

    return block._block_reduce[
        block_size, warp_reduce_fn=_reduce_fn, broadcast=broadcast
    ](vals, initial_vals=initial)


@always_inline
def _cluster_cutoff_search[
    vec_size: Int,
    block_size: Int,
    cluster_size: Int,
](
    d: Int,
    k: Int32,
    p_eff: Float32,
    low_init: Float32,
    high_init: Float32,
    mass_above_low_init: Float32,
    staged: UnsafePointer[
        mut=True, Float32, _, address_space=AddressSpace.SHARED
    ],
    vec_begin: Int,
    vec_end: Int,
) -> Tuple[Float32, Float32]:
    """`_topk_topp_cutoff_search` with the row spread over a cluster.

    Same constraint, contract and value-snapping termination as the
    single-block search in `topk_fi`: a token survives iff `count(> t) < k`
    and `mass(> t) <= p_eff`, the returned cutoff is exact, and callers must
    guarantee the predicate fails at `low_init`, holds at `high_init`, and
    `mass_above_low_init == mass(> low_init)`.

    Three things differ from the single-block form, all tied to the cluster
    launch:

    - Each CTA scans only its half-open vector range `[vec_begin, vec_end)`,
      reading the slice it staged in shared memory directly -- there is no
      load callback, because a closure that captures a dynamic shared-memory
      pointer returns the wrong data.
    - The per-round statistics combine across the cluster in rank order, so
      every CTA holds bit-identical values, takes the same branch, and stops
      on the same round -- a disagreeing CTA would run a different number of
      cluster barriers. The block reduction does not broadcast: thread 0
      publishes to the peers, and the cluster combine supplies the result to
      the rest of the block. Consecutive rounds alternate between two publish
      slots, so neither needs a sync to retire it.
    - Pivots bisect in the float bit domain rather than the value domain,
      which roughly halves the rounds to converge over a distribution that
      spans many orders of magnitude.

    Returns:
        ``(cutoff, kept_mass)`` where ``kept_mass == mass(> cutoff)``. On the
        (defensive) iteration cap, returns the current bracket state, which
        keeps a superset of the constraint set but stays self-consistent.
    """
    var low = low_init
    var high = high_init
    var mass_above_low = mass_above_low_init

    var g_begin = vec_begin + Int(thread_idx.x)

    var cluster_slot = unsafe_stack_allocation[
        2 * _CLUSTER_SLOT_FLOATS,
        Float32,
        alignment=16,
        address_space=AddressSpace.SHARED,
    ]()
    var phase = 0

    @always_inline
    @__parameter
    def _cutoff_stats_combine(x: SIMD, y: type_of(x)) -> type_of(x):
        # Same lane layout as `_block_reduce_cutoff_stats`, padded to a
        # power of two.
        var r = x + y
        r[0] = min(x[0], y[0])
        r[1] = max(x[1], y[1])
        return r

    for _ in range(_CUTOFF_SEARCH_MAX_ITERS):
        # Pivots split the bracket in the float bit domain: non-negative
        # floats order the same in value and bit space, and the working
        # distribution spans many orders of magnitude, so thirds of the bit
        # range cross an exponent per step where thirds of the value range
        # walk `high` down one exponent at a time. `high > low` here, so the
        # span cannot underflow.
        var lo_bits = max(low, Float32(0)).to_bits[DType.uint32]()
        var hi_bits = max(high, Float32(0)).to_bits[DType.uint32]()
        var span = hi_bits - lo_bits
        var pivot_0 = bitcast[DType.float32](lo_bits + span // 3)
        var pivot_1 = bitcast[DType.float32](lo_bits + 2 * (span // 3))

        # Accumulate thread-local counts/masses across the slice. The
        # accumulators stay scalar on purpose: at block_size 1024 only 64
        # registers per thread are available, and SIMD-wide accumulators
        # spill.
        var thread_count_0 = Float32(0)
        var thread_count_1 = Float32(0)
        var thread_mass_0 = Float32(0)
        var thread_mass_1 = Float32(0)
        var min_gt_low = high
        var max_le_high = low

        for g in range(g_begin, vec_end, block_size):
            var v = staged.load[width=vec_size]((g - vec_begin) * vec_size)
            comptime for j in range(vec_size):
                var e = v[j]
                if e > pivot_0:
                    thread_count_0 += 1
                    thread_mass_0 += e
                if e > pivot_1:
                    thread_count_1 += 1
                    thread_mass_1 += e
                if e > low:
                    min_gt_low = min(min_gt_low, e)
                if e <= high:
                    max_le_high = max(max_le_high, e)

        var stats = _block_reduce_cutoff_stats[block_size, 6, broadcast=False](
            StaticTuple[Scalar[DType.float32], 6](
                min_gt_low,
                max_le_high,
                thread_mass_0,
                thread_mass_1,
                thread_count_0,
                thread_count_1,
            )
        )
        var packed = SIMD[DType.float32, 8](0)
        comptime for i in range(6):
            packed[i] = stats[i]
        var combined = cluster_allreduce[
            _cutoff_stats_combine, cluster_size, need_tail_sync=False
        ](cluster_slot + phase * _CLUSTER_SLOT_FLOATS, packed)
        phase ^= 1

        min_gt_low = combined[0]
        max_le_high = combined[1]
        var mass_0 = combined[2]
        var mass_1 = combined[3]
        var count_0 = Int32(combined[4])
        var count_1 = Int32(combined[5])

        # pivot_1 > pivot_0: if the constraint still fails above the higher
        # pivot it also fails above the lower one, so test high-to-low.
        if count_1 >= k or mass_1 > p_eff:
            low = pivot_1
            mass_above_low = mass_1
        elif count_0 >= k or mass_0 > p_eff:
            low = pivot_0
            mass_above_low = mass_0
            high = min(pivot_1, max_le_high)
        else:
            high = min(pivot_0, max_le_high)

        # Exactly one distinct data value remains in (low, high]: every token
        # above `low` passes the predicate, every token at or below fails.
        if min_gt_low == max_le_high or high <= low:
            break

    return Tuple[Float32, Float32](low, mass_above_low)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
@__name(t"topk_topp_masked_probs_cluster_{dtype}_{cluster_size}")
def TopKTopPMaskedProbsClusterKernel[
    block_size: Int,
    vec_size: Int,
    dtype: DType,
    LogitsLayoutType: TensorLayout,
    logits_origin: ImmOrigin,
    cluster_size: Int,
](
    logits: TileTensor[dtype, LogitsLayoutType, logits_origin],
    probs_ptr: UnsafePointer[Float32, MutAnyOrigin],
    top_k_arr: Optional[UnsafePointer[Int64, ImmutAnyOrigin]],
    top_k_val: Int32,
    top_p_arr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    top_p_val: Float32,
    temperature: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    d: Int32,
):
    """`TopKTopPMaskedProbsKernel` with one row spread over a cluster.

    Works in the unnormalized domain `e_i = exp((logit_i - row_max) / temp)`:
    a token survives the joint constraint iff `e > cutoff` (recovered by the
    same dual-pivot search the sampler uses) and its masked probability is
    `e / kept_mass`. The output row is that masked renormalized distribution
    -- the same tensor `TopKTopPSamplingFromProbKernel` emits under
    `emit_dist`, so a verifier's target-side probabilities and a draft's
    proposal distribution are described identically.

    One block per row uses only as many SMs as there are rows, which leaves
    most of the GPU idle at decode batch sizes. Here the CTAs of one cluster
    share the row. Each CTA reduces its own contiguous slice, and the cluster
    combines the whole-row reductions that the cutoff needs.

    The launch must set `cluster_dim` to `cluster_size`.
    """
    comptime assert (
        not is_apple_gpu()
    ), "TopKTopPMaskedProbsClusterKernel is not supported on Apple GPUs"
    var _d = Int(d)
    var tx = Int(thread_idx.x)

    debug_assert(
        Int(cluster_dim.x) == cluster_size,
        "launch cluster_dim must match the kernel's cluster_size",
    )

    # A cluster covers one row, and each CTA takes a contiguous range of
    # vectors: staging in shared memory needs each CTA's elements to be a
    # compact range it can address as `g - vec_begin`, and contiguity keeps
    # the allocation at exactly `ceil(d / cluster_size)` elements.
    var rank = Int(block_rank_in_cluster())
    var bx = Int(block_idx.x) // cluster_size
    var n_vec = _d // vec_size
    var slice_vec = ceildiv(n_vec, cluster_size)
    var vec_begin = min(rank * slice_vec, n_vec)
    var vec_end = min(vec_begin + slice_vec, n_vec)

    # The softmax weights are staged in dynamic shared memory, so the search
    # re-reads them from the SM rather than from L2 on every iteration. The
    # host guarantees the slice fits before choosing this kernel; a row too
    # wide for the budget goes to the single-block kernel instead.
    var smem = external_memory[
        Float32, address_space=AddressSpace.SHARED, alignment=16
    ]()

    # Publish slots for this kernel's own cross-CTA combines, used in turn so
    # neither needs a sync to retire it. The cutoff search brings its own
    # pair.
    var cluster_slot = unsafe_stack_allocation[
        2 * _CLUSTER_SLOT_FLOATS,
        Float32,
        alignment=16,
        address_space=AddressSpace.SHARED,
    ]()

    wait_on_dependent_grids()
    launch_dependent_grids()

    var k = Int(top_k_val)
    if top_k_arr:
        k = Int(top_k_arr.unsafe_value().load(bx))
    if k <= 0 or k > _d:
        k = _d

    var p = top_p_val
    if top_p_arr:
        p = top_p_arr.unsafe_value()[bx]
    p = p.clamp(Float32(0.0), Float32(1.0))

    var temp_val = Float32(1.0)
    if temperature:
        temp_val = temperature.unsafe_value()[bx]
    # Clamp so a greedy (T=0) row cannot divide by zero.
    var inv_temp = 1.0 / max(temp_val, Float32(1e-6))

    var logits_row = TileTensor(logits.ptr + bx * _d, row_major(Idx[1], _d))

    # Row max, combined across the cluster in rank order so every CTA holds
    # the same bits. Every value below derives from it, so one disagreeing
    # CTA would branch differently through the search. The block reduction
    # does not broadcast: thread 0 publishes to the peers, and the cluster
    # combine supplies the result to the rest of the block.
    var thread_max = Scalar[DType.float32].MIN
    for i in range(vec_begin + tx, vec_end, block_size):
        var v = logits_row.load[width=vec_size]((Idx[0], i * vec_size)).cast[
            DType.float32
        ]()
        thread_max = max(thread_max, v.reduce_max())

    var m = cluster_allreduce[_max, cluster_size, need_tail_sync=False](
        cluster_slot,
        SIMD[DType.float32, 1](
            block.max[block_size=block_size, broadcast=False](thread_max)
        ),
    )[0]

    @__parameter
    @always_inline
    def load_e(offset: Int) -> SIMD[DType.float32, vec_size]:
        var v = logits_row.load[width=vec_size]((Idx[0], offset)).cast[
            DType.float32
        ]()
        return exp((v - m) * inv_temp)

    var probs_row = TileTensor(probs_ptr + bx * _d, row_major(Idx[1], _d))

    # Total mass, plus how many tokens carry any: a row whose every positive
    # token already satisfies the constraint has no boundary to find, and the
    # search's precondition (the predicate fails at 0) would not hold.
    #
    # The same pass stages `e` into this CTA's slice of shared memory. The
    # cutoff search reads the working distribution once per iteration and
    # would otherwise rebuild it each time -- a load, an FMA and a
    # transcendental per element per pass -- so staging pays the exp once.
    var thread_sum = Float32(0)
    var thread_pos: Int32 = 0
    for i in range(vec_begin + tx, vec_end, block_size):
        var e = load_e(i * vec_size)
        smem.store[width=vec_size]((i - vec_begin) * vec_size, e)
        thread_sum += e.reduce_add()
        comptime for j in range(vec_size):
            if e[j] > 0:
                thread_pos += 1
    var block_total = _block_reduce_value_count[DType.float32, broadcast=False](
        ValueCount[DType.float32](thread_sum, thread_pos)
    )

    var totals = cluster_allreduce[_sum, cluster_size, need_tail_sync=False](
        cluster_slot + _CLUSTER_SLOT_FLOATS,
        SIMD[DType.float32, 2](block_total.value, Float32(block_total.count)),
    )
    var z = totals[0]
    var total_count = Int32(totals[1])
    var p_eff = p * z

    var cut = Float32(0)
    var mass_s = z
    if total_count >= Int32(k) or z > p_eff:
        # The search reads elements other lanes staged, so order the writes
        # against it -- cluster-wide, since the search's own combines assume
        # every CTA has arrived.
        cluster_sync()
        var refined = _cluster_cutoff_search[
            vec_size, block_size, cluster_size
        ](
            _d,
            Int32(k),
            p_eff,
            0.0,
            1.0,
            z,
            smem,
            vec_begin,
            vec_end,
        )
        cut = refined[0]
        mass_s = refined[1]

    # Masks the staged weights; each lane owns the elements it wrote (the
    # same partition as the staging pass), so this needs no barrier against
    # the search.
    for i in range(vec_begin + tx, vec_end, block_size):
        var e = smem.load[width=vec_size]((i - vec_begin) * vec_size)
        var masked = (e.gt(cut)).select(
            e / mass_s, SIMD[DType.float32, vec_size](0)
        )
        probs_row.store[width=vec_size]((Idx[0], i * vec_size), masked)

    # Terminal keep-alive. Peers read this CTA's shared memory through `mapa`,
    # so it may not retire until every CTA of the cluster is done reading --
    # letting one exit early is an illegal access, not just a stale read.
    cluster_sync()


def topk_topp_masked_probs_cluster[
    dtype: DType,
    block_size: Int = 1024,
    TopKArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TopPArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TemperatureLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    ProbsLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64, Int64].element_types,
        stride_types=Coord[Int64, ComptimeInt[1]].element_types,
    ],
](
    ctx: DeviceContext,
    logits: TileTensor[mut=False, dtype, ...],
    probs: TileTensor[DType.float32, ProbsLayoutType, MutAnyOrigin],
    top_k_val: Int,
    top_p_val: Float32 = 1.0,
    top_k_arr: Optional[
        TileTensor[DType.int64, TopKArrLayoutType, ImmutAnyOrigin]
    ] = None,
    top_p_arr: Optional[
        TileTensor[DType.float32, TopPArrLayoutType, ImmutAnyOrigin]
    ] = None,
    temperature: Optional[
        TileTensor[DType.float32, TemperatureLayoutType, ImmutAnyOrigin]
    ] = None,
) raises:
    """Computes per-row top-k/top-p masked softmax on a cluster device.

    See `TopKTopPMaskedProbsKernel` for what the output means. The package's
    `topk_topp_masked_probs` dispatcher routes here on NVIDIA SM90+ devices;
    a batch that fills the machine or a slice too wide for shared memory
    still falls back to the single-block kernel at runtime.

    Parameters:
        dtype: Element type of `logits`.
        block_size: Threads per block.
        TopKArrLayoutType: Memory layout of `top_k_arr`.
        TopPArrLayoutType: Memory layout of `top_p_arr`.
        TemperatureLayoutType: Memory layout of `temperature`.
        ProbsLayoutType: Memory layout of `probs`.

    Args:
        ctx: Device context.
        logits: Input logits [batch_size, d].
        probs: Output masked renormalized distribution [batch_size, d].
        top_k_val: Default top-k; `<= 0` or `> d` keeps every token.
        top_p_val: Default top-p threshold.
        top_k_arr: Optional per-row top-k [batch_size].
        top_p_arr: Optional per-row top-p [batch_size].
        temperature: Optional per-row temperature [batch_size]; 0 is clamped.

    Raises:
        Error: If the tensor shapes disagree.
    """
    comptime assert logits.rank == 2, "logits rank must be 2"

    var shape = coord_to_index_list(logits.layout.shape_coord())
    var batch_size = shape[0]
    var d = shape[1]

    @__parameter
    def trace_information() -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("logits", shape, dtype),
                    "top_k=" + String(top_k_val),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "topk_topp_masked_probs",
        Trace[TraceLevel.OP]._get_detail_str[trace_information](),
        task_id=Int(ctx.id()),
    ):
        var probs_shape = coord_to_index_list(probs.layout.shape_coord())
        if probs_shape[0] != batch_size or probs_shape[1] != d:
            raise Error("probs shape must match the logits shape")

        # Speculative decoding runs this with zero rows on every step that has
        # no drafts to verify, and a grid of 0 is not a legal launch.
        if batch_size == 0:
            return

        var vec_size = gcd(8, d)

        var top_k_ptr: Optional[UnsafePointer[Int64, ImmutAnyOrigin]] = None
        if top_k_arr:
            top_k_ptr = top_k_arr.unsafe_value().ptr

        var top_p_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]] = None
        if top_p_arr:
            top_p_ptr = top_p_arr.unsafe_value().ptr

        var temperature_ptr: Optional[
            UnsafePointer[Float32, ImmutAnyOrigin]
        ] = None
        if temperature:
            temperature_ptr = temperature.unsafe_value().ptr

        # A row takes the narrowest cluster whose slice fits the staging
        # budget: one CTA for small vocabularies (staged, no cluster
        # traffic), two when the row must split, four for the widest. A row
        # too wide for four slices takes the single-block kernel with no
        # staging.
        var cluster_size = 0
        comptime for c in [1, 2, _MAX_CUTOFF_CLUSTER]:
            if (
                cluster_size == 0
                and _stage_smem_bytes(d, vec_size, c) <= _MAX_STAGE_SMEM_BYTES
            ):
                cluster_size = c

        @__parameter
        def launch_cluster[vec_size: Int, cluster_size: Int]() raises:
            comptime kernel = TopKTopPMaskedProbsClusterKernel[
                block_size,
                vec_size,
                dtype,
                logits.LayoutType,
                ImmOrigin(logits.origin),
                cluster_size,
            ]
            var smem_bytes = _stage_smem_bytes(d, vec_size, cluster_size)
            ctx.enqueue_function[kernel](
                logits.as_immut(),
                probs.ptr,
                top_k_ptr,
                Int32(top_k_val),
                top_p_ptr,
                top_p_val,
                temperature_ptr,
                Int32(d),
                grid_dim=batch_size * cluster_size,
                block_dim=block_size,
                cluster_dim=Dim(cluster_size, 1, 1),
                shared_mem_bytes=smem_bytes,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(smem_bytes)
                ),
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )

        @__parameter
        def launch_single[vec_size: Int]() raises:
            comptime kernel = TopKTopPMaskedProbsKernel[
                block_size,
                vec_size,
                dtype,
                logits.LayoutType,
                ImmOrigin(logits.origin),
            ]
            ctx.enqueue_function[kernel](
                logits.as_immut(),
                probs.ptr,
                top_k_ptr,
                Int32(top_k_val),
                top_p_ptr,
                top_p_val,
                temperature_ptr,
                Int32(d),
                grid_dim=batch_size,
                block_dim=block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )

        # `vec_size = gcd(8, d)`, so 8 is the widest case.
        comptime for param_vec_size in [8, 4, 2, 1]:
            if vec_size == param_vec_size:
                comptime for param_cluster in [1, 2, _MAX_CUTOFF_CLUSTER]:
                    if cluster_size == param_cluster:
                        return launch_cluster[param_vec_size, param_cluster]()
                return launch_single[param_vec_size]()
