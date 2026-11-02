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
"""The DSpark draft sampler: one fused pass per draft position.

Draws each row's draft token by Gumbel-max over the tempered softmax of
``(logit + bias) / T``, and writes that softmax back as ``q``. The verifier's
rejection test is exact for any distribution the draft really samples from,
so one pass keeps the draw and the ``q`` the verifier reads from ever
disagreeing.

One thread-block cluster per row: each CTA stages its contiguous slice in
shared memory, and the row max, the exponential sum and the noised argmax
combine over the cluster's distributed shared memory. The row is read once
from global memory and ``q`` is written once. The Gumbel noise is keyed by
the row's seed and the element's index only, so a row draws the same token
under any split of the batch or of the row.
"""

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.primitives import block, warp
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_allreduce,
)
from max.gpu.primitives.grid_controls import (
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
    PDLLevel,
)
from max.gpu.host import DeviceContext, Dim, FuncAttribute
from max.gpu.memory import external_memory
from max.gpu.sync import barrier
from std.bit import log2_floor
from std.math import ceildiv, exp, log2
from std.memory import unsafe_stack_allocation
from std.random import Random
from std.sys import size_of
from std.sys.info import is_apple_gpu
from std.utils.static_tuple import StaticTuple
from max.runtime.tracing import Trace, TraceLevel, trace_arg
from layout import (
    Idx,
    DefaultEngine,
    TensorLayout,
    TensorEngine,
    TileTensor,
    coord_to_index_list,
)

comptime _DRAFT_CLUSTER_MAX = 4
"""The widest cluster this kernel uses; every CTA of a cluster runs at once
on one GPC."""

comptime _DRAFT_STAGE_BUDGET_BYTES = 216 * 1024
"""The most shared memory one CTA stages. B200 carries 228 KiB per SM."""

comptime _DRAFT_STAGE_HEADROOM_BYTES = 12 * 1024
"""Room below the staging budget for the cluster slots, the reduction
scratch and the driver's reserve."""

comptime _DRAFT_SLOT_FLOATS = 16
"""A cluster publish slot: the widest vector combined, twice over."""

comptime _DRAFT_VEC = 4
"""Elements per load. The Gumbel RNG steps four values at a time, so this
must stay a multiple of four."""

comptime _DRAFT_BLOCK = 1024

comptime _DRAFT_EPS = Float32(1e-20)
comptime _DRAFT_LOG2 = Float32(0.6931471806)
comptime _DRAFT_MIN_TEMP = Float32(1e-6)


def _draft_stage_budget(ctx: DeviceContext) -> Int:
    """The staging budget for one CTA on the compiled target."""
    return min(
        _DRAFT_STAGE_BUDGET_BYTES,
        Int(ctx.default_device_info.shared_memory_per_multiprocessor)
        - _DRAFT_STAGE_HEADROOM_BYTES,
    )


def _draft_stage_bytes(d: Int, cluster_size: Int) -> Int:
    """Bytes one CTA stages for its contiguous slice of a row."""
    return (
        ceildiv(ceildiv(d, _DRAFT_VEC), cluster_size)
        * _DRAFT_VEC
        * size_of[DType.float32]()
    )


@inline(.always)
def _draft_max(x: SIMD, y: type_of(x)) -> type_of(x):
    return max(x, y)


@inline(.always)
def _draft_sum(x: SIMD, y: type_of(x)) -> type_of(x):
    return x + y


@inline(.always)
def _draft_best_combine(x: SIMD, y: type_of(x)) -> type_of(x):
    """Picks the larger value, the lower index on a tie.

    Packs ``(value, index)``; an index below 2^24 is exact in float32, and
    the vocabularies here are far below that.
    """
    var out = x
    if y[0] > x[0] or (y[0] == x[0] and y[1] < x[1]):
        out = y
    return out


@fieldwise_init
struct _DraftBest(Defaultable, TrivialRegisterPassable):
    """One thread's best noised value and its column index."""

    var value: Float32
    var index: Int32

    def __init__(out self):
        self.value = -Float32.MAX
        self.index = Int32(0)

    @inline(.always)
    def insert(mut self, value: Float32, index: Int):
        if value > self.value or (
            value == self.value and Int32(index) < self.index
        ):
            self.value = value
            self.index = Int32(index)


@inline(.always)
def _draft_warp_reduce(val: _DraftBest) -> _DraftBest:
    """Picks each warp's best, broadcasting it to every lane."""
    var res = val
    comptime for i in reversed(range(log2_floor(WARP_SIZE))):
        comptime mask = 1 << i
        var u = warp.shuffle_xor(res.value, UInt32(mask))
        var p = warp.shuffle_xor(res.index, UInt32(mask))
        if u > res.value or (u == res.value and p < res.index):
            res.value = u
            res.index = p
    return res


@inline(.always)
def _draft_block_reduce[block_size: Int](val: _DraftBest) -> _DraftBest:
    """Picks the block's best; valid in thread 0."""
    comptime n_warps = block_size // WARP_SIZE
    var w_value = unsafe_stack_allocation[
        n_warps, Float32, address_space=.SHARED
    ]()
    var w_index = unsafe_stack_allocation[
        n_warps, Int32, address_space=.SHARED
    ]()

    var res = _draft_warp_reduce(val)
    if lane_id() == 0:
        w_value[warp_id()] = res.value
        w_index[warp_id()] = res.index
    barrier()

    var out = _DraftBest()
    if warp_id() == 0 and lane_id() < n_warps:
        out.value = w_value[lane_id()]
        out.index = w_index[lane_id()]
    return _draft_warp_reduce(out)


@inline(.always)
def _draft_gumbel(seed_val: UInt64, n: Int, chunk: Int) -> SIMD[.float32, 4]:
    """The chunk's Gumbel(0, 1) noise, keyed by the row seed and the chunk.

    The sequence matches the fused Gumbel argmax kernel, so an element draws
    the same noise under any thread or split assignment.
    """
    var rng = Random(seed=seed_val * UInt64(n) + UInt64(chunk))
    var noise = SIMD[.float32, 4](0)
    var u = rng.step_uniform()
    comptime for lane in range(4):
        noise[lane] = -_DRAFT_LOG2 * log2(
            -log2(u[lane] + _DRAFT_EPS) + _DRAFT_EPS
        )
    return noise


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_DRAFT_BLOCK))
)
@__name(t"dspark_draft_sample_cluster_{dtype}_{cluster_size}")
def DSparkDraftSampleClusterKernel[
    dtype: DType,
    cluster_size: Int,
    LogitsLayoutType: TensorLayout,
    logits_origin: ImmOrigin,
    BiasLayoutType: TensorLayout,
    bias_origin: ImmOrigin,
    DistLayoutType: TensorLayout,
    TokensLayoutType: TensorLayout,
    LogitsEngine: TensorEngine,
    BiasEngine: TensorEngine,
    DistEngine: TensorEngine,
    TokensEngine: TensorEngine,
](
    logits: TileTensor[
        dtype, LogitsLayoutType, logits_origin, Engine=LogitsEngine
    ],
    bias: TileTensor[.float32, BiasLayoutType, bias_origin, Engine=BiasEngine],
    temperature: ImmPointer[Float32, ImmutAnyOrigin],
    seed: ImmPointer[UInt64, ImmutAnyOrigin],
    out_tokens: TileTensor[
        mut=True, .int64, TokensLayoutType, MutAnyOrigin, Engine=TokensEngine
    ],
    dist: TileTensor[.float32, DistLayoutType, MutAnyOrigin, Engine=DistEngine],
    d: Int32,
):
    """One draft position's draw and its proposal distribution.

    Each CTA owns a contiguous slice of the row and stages
    ``x = (logit + bias) / T`` in shared memory. Three cluster reductions
    follow: the row max ``m``, the row sum ``z = sum(exp(x - m))``, and the
    noised argmax that picks the token. ``q = exp(x - m) / z`` is written
    from the staged slice, so the row is read from global memory once.

    The launch must set ``cluster_dim`` to ``cluster_size``.
    """
    comptime assert (
        not is_apple_gpu()
    ), "DSparkDraftSampleClusterKernel is not supported on Apple GPUs"

    var _d = Int(d)
    var tx = thread_idx.x

    var rank = Int(block_rank_in_cluster())
    var bx = block_idx.x // cluster_size
    var n_vec = _d // _DRAFT_VEC
    var slice_vec = ceildiv(n_vec, cluster_size)
    var vec_begin = min(rank * slice_vec, n_vec)
    var vec_end = min(vec_begin + slice_vec, n_vec)

    var smem = external_memory[Float32, address_space=.SHARED, alignment=16]()
    var cluster_slot = unsafe_stack_allocation[
        2 * _DRAFT_SLOT_FLOATS,
        Float32,
        alignment=16,
        address_space=.SHARED,
    ]()

    wait_on_dependent_grids()
    launch_dependent_grids()

    var inv_temp = Float32(1.0) / max(temperature[bx], _DRAFT_MIN_TEMP)
    var seed_val = seed[bx]

    var logits_row = logits[bx : bx + 1, :]
    var bias_row = bias[bx : bx + 1, :]

    # Stage x and the running max. Each lane writes the slots it reads back.
    var thread_max = -Float32.MAX
    for i in range(vec_begin + tx, vec_end, _DRAFT_BLOCK):
        var v = (
            logits_row.load[width=_DRAFT_VEC]((Idx[0], i * _DRAFT_VEC)).cast[
                .float32
            ]()
            + bias_row.load[width=_DRAFT_VEC]((Idx[0], i * _DRAFT_VEC))
        ) * inv_temp
        smem.unsafe_store((i - vec_begin) * _DRAFT_VEC, v)
        thread_max = max(thread_max, v.reduce_max())

    var m = cluster_allreduce[_draft_max, cluster_size](
        cluster_slot,
        Float32(
            block.max[block_size=_DRAFT_BLOCK, broadcast=False](thread_max)
        ),
    )[0]

    # The row's exponential total, from the staged slice.
    var thread_sum = Float32(0)
    for i in range(vec_begin + tx, vec_end, _DRAFT_BLOCK):
        var x = smem.unsafe_load[width=_DRAFT_VEC]((i - vec_begin) * _DRAFT_VEC)
        thread_sum += exp(x - m).reduce_add()

    var z = cluster_allreduce[_draft_sum, cluster_size](
        cluster_slot,
        Float32(
            block.sum[block_size=_DRAFT_BLOCK, broadcast=False](thread_sum)
        ),
    )[0]

    # Gumbel-argmax over x and the write of q = exp(x - m) / z.
    var thread_best = _DraftBest()
    var dist_row = dist[bx : bx + 1, :]
    for i in range(vec_begin + tx, vec_end, _DRAFT_BLOCK):
        var x = smem.unsafe_load[width=_DRAFT_VEC]((i - vec_begin) * _DRAFT_VEC)
        var noised = x + _draft_gumbel(seed_val, _d, i)
        comptime for lane in range(_DRAFT_VEC):
            thread_best.insert(noised[lane], i * _DRAFT_VEC + lane)
        dist_row.store((Idx[0], i * _DRAFT_VEC), exp(x - m) / z)

    var local_best = _draft_block_reduce[_DRAFT_BLOCK](thread_best)
    var winner = cluster_allreduce[_draft_best_combine, cluster_size](
        cluster_slot,
        SIMD[.float32, 2](local_best.value, Float32(local_best.index)),
    )
    if tx == 0 and rank == 0:
        out_tokens.store((bx,), Int(winner[1]).cast[DType.int64]())


def dspark_draft_sample_gpu[
    dtype: DType,
    LogitsLayoutType: TensorLayout,
    logits_origin: ImmOrigin,
    BiasLayoutType: TensorLayout,
    bias_origin: ImmOrigin,
    TemperatureLayoutType: TensorLayout,
    SeedLayoutType: TensorLayout,
    DistLayoutType: TensorLayout,
    TokensLayoutType: TensorLayout,
    LogitsEngine: TensorEngine = DefaultEngine[element_width=1],
    BiasEngine: TensorEngine = DefaultEngine[element_width=1],
    TemperatureEngine: TensorEngine = DefaultEngine[element_width=1],
    SeedEngine: TensorEngine = DefaultEngine[element_width=1],
    DistEngine: TensorEngine = DefaultEngine[element_width=1],
    TokensEngine: TensorEngine = DefaultEngine[element_width=1],
](
    ctx: DeviceContext,
    logits: TileTensor[
        dtype, LogitsLayoutType, logits_origin, Engine=LogitsEngine
    ],
    bias: TileTensor[.float32, BiasLayoutType, bias_origin, Engine=BiasEngine],
    temperature: TileTensor[
        .float32,
        TemperatureLayoutType,
        ImmutAnyOrigin,
        Engine=TemperatureEngine,
    ],
    seed: TileTensor[
        .uint64, SeedLayoutType, ImmutAnyOrigin, Engine=SeedEngine
    ],
    out_tokens: TileTensor[
        mut=True, .int64, TokensLayoutType, MutAnyOrigin, Engine=TokensEngine
    ],
    dist: TileTensor[.float32, DistLayoutType, MutAnyOrigin, Engine=DistEngine],
) raises:
    """Dispatches the fused DSpark draft sample.

    Args:
        ctx: Device context.
        logits: Raw head logits ``[rows, vocab]``, bfloat16 or float32.
        bias: The Markov bias ``[rows, vocab]``, float32.
        temperature: Per-row temperature ``[rows]``; non-positive clamps.
        seed: Per-row RNG seed ``[rows]``, uint64.
        out_tokens: Sampled column per row, ``int64 [rows]``.
        dist: The sampled distribution ``[rows, vocab]``, float32.

    Raises:
        Error: If the shapes disagree or a row does not fit the staging
            budget even at the widest cluster.
    """
    var shape = coord_to_index_list(logits.layout.shape_coord())
    var bias_shape = coord_to_index_list(bias.layout.shape_coord())
    var rows = shape[0]
    var d = shape[1]

    def trace_information() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("logits", shape, dtype),
                    trace_arg("bias", bias_shape, DType.float32),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "dspark_draft_sample",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
        task_id=Int(ctx.device().id),
    ):
        if bias_shape[0] != rows or bias_shape[1] != d:
            raise Error("dspark_draft_sample: bias must be [rows, vocab]")
        if Int(coord_to_index_list(dist.layout.shape_coord())[0]) != rows:
            raise Error("dspark_draft_sample: dist must be [rows, vocab]")
        if d % _DRAFT_VEC != 0:
            raise Error(
                "dspark_draft_sample: vocab must be a multiple of ",
                _DRAFT_VEC,
                ", got ",
                d,
            )

        if rows == 0 or d == 0:
            return

        var budget = _draft_stage_budget(ctx)
        var cluster_size = 0
        comptime for c in [1, 2, _DRAFT_CLUSTER_MAX]:
            if cluster_size == 0 and _draft_stage_bytes(d, c) <= budget:
                cluster_size = c
        if cluster_size == 0:
            raise Error(
                "dspark_draft_sample: vocab ",
                d,
                " is too wide to stage even at cluster size ",
                _DRAFT_CLUSTER_MAX,
            )

        def launch_cluster[cluster_size: Int]() raises {imm}:
            comptime kernel = DSparkDraftSampleClusterKernel[
                dtype,
                cluster_size,
                logits.LayoutType,
                ImmOrigin(logits.origin),
                bias.LayoutType,
                ImmOrigin(bias.origin),
                dist.LayoutType,
                out_tokens.LayoutType,
                LogitsEngine=logits.Engine,
                BiasEngine=bias.Engine,
                DistEngine=dist.Engine,
                TokensEngine=out_tokens.Engine,
            ]
            var smem_bytes = _draft_stage_bytes(d, cluster_size)
            ctx.enqueue_function[kernel](
                logits.as_imm(),
                bias.as_imm(),
                temperature.ptr,
                seed.ptr,
                out_tokens,
                dist,
                Int32(d),
                grid_dim=rows * cluster_size,
                block_dim=_DRAFT_BLOCK,
                cluster_dim=Dim(cluster_size, 1, 1),
                shared_mem_bytes=smem_bytes,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(budget)
                ),
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )

        comptime for c in [1, 2, _DRAFT_CLUSTER_MAX]:
            if cluster_size == c:
                return launch_cluster[c]()
