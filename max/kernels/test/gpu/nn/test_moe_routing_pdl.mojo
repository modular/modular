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
"""PDL ordering test for the MoE routing chain.

The routers, `eplb_remap` and `moe_create_indices` release their dependents as
soon as their own wait returns, so a successor becomes resident while the
kernel before it is still scanning. That is only sound if every kernel in the
chain waits before it reads what its predecessor writes.

Each chain here starts with a producer kernel that releases its dependents on
entry, sleeps, and only then publishes the real operands over zeroed ones. A
routing kernel that read its input before its wait sees the zeros, and a
`moe_create_indices` that read the router's output before its wait sees the
-1 sentinel the output buffer started with, which it files under no expert.
Every chain must match the same kernels run with a host sync between launches.

`moe_create_indices` releases its own dependents just as early, so the last
tests chain the kernels that read its outputs in production behind it: the
grouped NVFP4 quantize (with and without the slot-layout row indices) and the
grouped bf16 GEMM. Each must match a rerun on the final indices.

`test_indices_release_on_entry` checks the release itself for
`moe_create_indices`: a probe launched behind a long scan must start while the
scan is still running. The group-limited router, probed the same way at 65536
tokens, ends too quickly to separate the two placements, and the other three
kernels are not probed.
"""

from std.math import ceildiv
from std.sys import size_of
from std.testing import assert_equal, assert_true
from std.time import global_perf_counter_ns, sleep

from layout import Coord, Idx, TileTensor, row_major
from layout._fillers import random
from linalg.block_scaled_quantization import (
    grouped_quantize_dynamic_scaled_fp4_async,
)
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
)
from linalg.grouped_matmul import grouped_matmul
from max.gpu import thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from nn.moe import (
    eplb_remap,
    moe_create_indices,
    moe_create_indices_with_slots,
    router_group_limited,
    single_group_router_eplb,
    sink_gate_router,
)

comptime _PRODUCER_THREADS = 256
comptime _SENTINEL = Int32(-1)


def publish_after_delay[
    dtype: DType
](
    dst: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    src: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    n: Int32,
):
    """Releases dependents at once, then publishes `src` into `dst` late."""
    launch_dependent_grids()
    sleep(0.05)
    for i in range(Int(thread_idx.x), Int(n), _PRODUCER_THREADS):
        dst[i] = src[i]


def publish[
    dtype: DType
](
    ctx: DeviceContext,
    dst: DeviceBuffer[dtype],
    src: DeviceBuffer[dtype],
    n: Int,
) raises:
    ctx.enqueue_function[publish_after_delay[dtype]](
        dst,
        src,
        Int32(n),
        grid_dim=1,
        block_dim=_PRODUCER_THREADS,
    )


struct Indices[num_experts: Int]:
    """The device outputs of one `moe_create_indices` launch."""

    var order: DeviceBuffer[DType.uint32]
    var start: DeviceBuffer[DType.uint32]
    var restore: DeviceBuffer[DType.uint32]
    var ids: DeviceBuffer[DType.int32]
    var stats: DeviceBuffer[DType.uint32]
    var n_topk: Int

    def __init__(out self, ctx: DeviceContext, n_topk: Int) raises:
        self.n_topk = n_topk
        self.order = ctx.enqueue_create_buffer[.uint32](n_topk)
        self.start = ctx.enqueue_create_buffer[.uint32](Self.num_experts + 1)
        self.restore = ctx.enqueue_create_buffer[.uint32](n_topk)
        self.ids = ctx.enqueue_create_buffer[.int32](Self.num_experts)
        self.stats = ctx.enqueue_create_buffer[.uint32](2)

    def enqueue(
        mut self, ctx: DeviceContext, topk_ids: DeviceBuffer[DType.int32]
    ) raises:
        moe_create_indices["gpu"](
            TileTensor(self.order, row_major(self.n_topk)),
            TileTensor(self.start, row_major(Self.num_experts + 1)),
            TileTensor(self.restore, row_major(self.n_topk)),
            TileTensor(self.ids, row_major(Self.num_experts)),
            TileTensor(self.stats, row_major(Idx[2])),
            TileTensor(topk_ids, row_major(self.n_topk)),
            ctx,
        )


def assert_buffers_equal[
    dtype: DType
](
    ctx: DeviceContext,
    got: DeviceBuffer[dtype],
    want: DeviceBuffer[dtype],
    n: Int,
    what: String,
) raises:
    var got_host = ctx.enqueue_create_host_buffer[dtype](n)
    var want_host = ctx.enqueue_create_host_buffer[dtype](n)
    ctx.enqueue_copy(got_host, got)
    ctx.enqueue_copy(want_host, want)
    ctx.synchronize()
    # Compare bytes: the host float8 compare does not lower.
    var nbytes = n * size_of[dtype]()
    var got_bytes = got_host.unsafe_ptr().unsafe_bitcast[UInt8]()
    var want_bytes = want_host.unsafe_ptr().unsafe_bitcast[UInt8]()
    var bad = 0
    for i in range(nbytes):
        if got_bytes[i] != want_bytes[i]:
            bad += 1
    assert_equal(
        bad, 0, msg=String(what, ": ", bad, " of ", nbytes, " bytes differ")
    )


def assert_indices_equal[
    num_experts: Int
](
    ctx: DeviceContext,
    got: Indices[num_experts],
    want: Indices[num_experts],
    topk_ids: DeviceBuffer[DType.int32],
) raises:
    """Checks `got` against `want`, the same kernel without overlap.

    The CSR offsets, expert ids and stats are deterministic. The order within
    one expert's slots follows an atomic cursor, so `order` and `restore` are
    checked as a valid grouping of `topk_ids` instead of bit for bit.
    """
    assert_buffers_equal(
        ctx, got.start, want.start, num_experts + 1, "expert_start_indices"
    )
    assert_buffers_equal(ctx, got.ids, want.ids, num_experts, "expert_ids")
    assert_buffers_equal(ctx, got.stats, want.stats, 2, "usage_stats")

    var n = got.n_topk
    var ids_host = ctx.enqueue_create_host_buffer[.int32](n)
    var start_host = ctx.enqueue_create_host_buffer[.uint32](num_experts + 1)
    var order_host = ctx.enqueue_create_host_buffer[.uint32](n)
    var restore_host = ctx.enqueue_create_host_buffer[.uint32](n)
    ctx.enqueue_copy(ids_host, topk_ids)
    ctx.enqueue_copy(start_host, got.start)
    ctx.enqueue_copy(order_host, got.order)
    ctx.enqueue_copy(restore_host, got.restore)
    ctx.synchronize()
    var bad = 0
    for i in range(n):
        var e = Int(ids_host[i])
        var pos = Int(restore_host[i])
        if (
            e < 0
            or e >= num_experts
            or pos < Int(start_host[e])
            or pos >= Int(start_host[e + 1])
            or Int(order_host[pos]) != i
        ):
            bad += 1
    assert_equal(
        bad, 0, msg=String("order/restore: ", bad, " of ", n, " slots misfiled")
    )


def test_group_limited_chain(num_tokens: Int, ctx: DeviceContext) raises:
    """DeepSeek-V3 geometry: scores -> group-limited router -> indices."""
    comptime dtype = DType.float32
    comptime n_routed = 256
    comptime topk = 8
    comptime n_groups = 8
    comptime topk_group = 4
    var n_topk = num_tokens * topk

    var scores_host = ctx.enqueue_create_host_buffer[dtype](
        num_tokens * n_routed
    )
    var bias_host = ctx.enqueue_create_host_buffer[dtype](n_routed)
    random(
        TileTensor(scores_host, row_major(num_tokens, Idx[n_routed])),
        min=0.0,
        max=1.0,
    )
    random(TileTensor(bias_host, row_major(Idx[n_routed])), min=-0.1, max=0.1)

    var scores_src = ctx.enqueue_create_buffer[dtype](num_tokens * n_routed)
    var scores = ctx.enqueue_create_buffer[dtype](num_tokens * n_routed)
    var bias = ctx.enqueue_create_buffer[dtype](n_routed)
    var idx_ref = ctx.enqueue_create_buffer[.int32](n_topk)
    var w_ref = ctx.enqueue_create_buffer[dtype](n_topk)
    var idx = ctx.enqueue_create_buffer[.int32](n_topk)
    var w = ctx.enqueue_create_buffer[dtype](n_topk)
    ctx.enqueue_copy(scores_src, scores_host)
    ctx.enqueue_copy(bias, bias_host)
    ctx.enqueue_memset(scores, Scalar[dtype](0))

    def route(
        mut out_idx: DeviceBuffer[DType.int32],
        mut out_w: DeviceBuffer[dtype],
        in_scores: DeviceBuffer[dtype],
    ) raises {imm}:
        router_group_limited[
            n_routed_experts=n_routed,
            n_experts_per_tok=topk,
            n_groups=n_groups,
            topk_group=topk_group,
            norm_weights=True,
            target="gpu",
        ](
            TileTensor(out_idx, row_major(num_tokens, Idx[topk])),
            TileTensor(out_w, row_major(num_tokens, Idx[topk])),
            TileTensor(
                in_scores, row_major(num_tokens, Idx[n_routed])
            ).as_imm(),
            TileTensor(bias, row_major(Idx[n_routed])).as_imm(),
            Float32(2.5),
            ctx,
        )

    # Reference: every launch is host-synchronized, so nothing overlaps.
    var want = Indices[n_routed](ctx, n_topk)
    route(idx_ref, w_ref, scores_src)
    ctx.synchronize()
    want.enqueue(ctx, idx_ref)
    ctx.synchronize()

    # Chain: producer -> router -> indices, with no sync in between.
    var got = Indices[n_routed](ctx, n_topk)
    ctx.enqueue_memset(idx, _SENTINEL)
    ctx.enqueue_memset(w, Scalar[dtype](0))
    ctx.synchronize()
    publish(ctx, scores, scores_src, num_tokens * n_routed)
    route(idx, w, scores)
    got.enqueue(ctx, idx)
    ctx.synchronize()

    assert_buffers_equal(ctx, idx, idx_ref, n_topk, "router indices")
    assert_buffers_equal(ctx, w, w_ref, n_topk, "router weights")
    assert_indices_equal(ctx, got, want, idx)


def test_sink_gate_chain(num_tokens: Int, ctx: DeviceContext) raises:
    """Inkling geometry: logits -> sink-gate router -> indices."""
    comptime dtype = DType.float32
    comptime n_routed = 256
    comptime topk = 6
    comptime n_shared = 2
    comptime n_total = n_routed + n_shared
    var n_topk = num_tokens * topk

    var logits_host = ctx.enqueue_create_host_buffer[dtype](
        num_tokens * n_total
    )
    var bias_host = ctx.enqueue_create_host_buffer[.float32](n_routed)
    var gscale_host = ctx.enqueue_create_host_buffer[.float32](1)
    random(
        TileTensor(logits_host, row_major(num_tokens, Idx[n_total])),
        min=-4.0,
        max=4.0,
    )
    random(TileTensor(bias_host, row_major(Idx[n_routed])), min=-0.1, max=0.1)
    gscale_host[0] = 1.3

    var logits_src = ctx.enqueue_create_buffer[dtype](num_tokens * n_total)
    var logits = ctx.enqueue_create_buffer[dtype](num_tokens * n_total)
    var bias = ctx.enqueue_create_buffer[.float32](n_routed)
    var gscale = ctx.enqueue_create_buffer[.float32](1)
    var idx_ref = ctx.enqueue_create_buffer[.int32](n_topk)
    var w_ref = ctx.enqueue_create_buffer[dtype](n_topk)
    var sink_ref = ctx.enqueue_create_buffer[dtype](num_tokens * n_shared)
    var idx = ctx.enqueue_create_buffer[.int32](n_topk)
    var w = ctx.enqueue_create_buffer[dtype](n_topk)
    var sink = ctx.enqueue_create_buffer[dtype](num_tokens * n_shared)
    ctx.enqueue_copy(logits_src, logits_host)
    ctx.enqueue_copy(bias, bias_host)
    ctx.enqueue_copy(gscale, gscale_host)
    ctx.enqueue_memset(logits, Scalar[dtype](0))

    def route(
        mut out_idx: DeviceBuffer[DType.int32],
        mut out_w: DeviceBuffer[dtype],
        mut out_sink: DeviceBuffer[dtype],
        in_logits: DeviceBuffer[dtype],
    ) raises {imm}:
        sink_gate_router[n_routed, topk, n_shared, "gpu"](
            TileTensor(out_idx, row_major(num_tokens, Idx[topk])),
            TileTensor(out_w, row_major(num_tokens, Idx[topk])),
            TileTensor(out_sink, row_major(num_tokens, Idx[n_shared])),
            TileTensor(in_logits, row_major(num_tokens, Idx[n_total])).as_imm(),
            TileTensor(bias, row_major(Idx[n_routed])).as_imm(),
            TileTensor(gscale, row_major(Idx[1])).as_imm(),
            Float32(8.0),
            ctx,
        )

    var want = Indices[n_routed](ctx, n_topk)
    route(idx_ref, w_ref, sink_ref, logits_src)
    ctx.synchronize()
    want.enqueue(ctx, idx_ref)
    ctx.synchronize()

    var got = Indices[n_routed](ctx, n_topk)
    ctx.enqueue_memset(idx, _SENTINEL)
    ctx.enqueue_memset(w, Scalar[dtype](0))
    ctx.enqueue_memset(sink, Scalar[dtype](0))
    ctx.synchronize()
    publish(ctx, logits, logits_src, num_tokens * n_total)
    route(idx, w, sink, logits)
    got.enqueue(ctx, idx)
    ctx.synchronize()

    assert_buffers_equal(ctx, idx, idx_ref, n_topk, "router indices")
    assert_buffers_equal(ctx, w, w_ref, n_topk, "router weights")
    assert_buffers_equal(ctx, sink, sink_ref, num_tokens * n_shared, "sinks")
    assert_indices_equal(ctx, got, want, idx)


comptime _num_log = 256
comptime _max_replicas = 2
comptime _num_layers = 2
comptime _num_phys = _num_log * _max_replicas
comptime _eplb_topk = 8


def eplb_tables(
    ctx: DeviceContext,
    logcnt: DeviceBuffer[DType.int32],
    log2phy: DeviceBuffer[DType.int32],
    layer_idx: DeviceBuffer[DType.int32],
) raises:
    """Fills the per-layer replica tables and selects the last layer."""
    var logcnt_host = ctx.enqueue_create_host_buffer[.int32](
        _num_layers * _num_log
    )
    var log2phy_host = ctx.enqueue_create_host_buffer[.int32](
        _num_layers * _num_log * _max_replicas
    )
    var layer_host = ctx.enqueue_create_host_buffer[.int32](1)
    for layer in range(_num_layers):
        for log in range(_num_log):
            logcnt_host[layer * _num_log + log] = Int32(
                _max_replicas if (log & 1) == 0 else 1
            )
            for r in range(_max_replicas):
                log2phy_host[
                    (layer * _num_log + log) * _max_replicas + r
                ] = Int32((layer * 7 + log * 3 + r * 131) % _num_phys)
    layer_host[0] = Int32(_num_layers - 1)
    ctx.enqueue_copy(logcnt, logcnt_host)
    ctx.enqueue_copy(log2phy, log2phy_host)
    ctx.enqueue_copy(layer_idx, layer_host)
    ctx.synchronize()


def test_eplb_router_chain(num_tokens: Int, ctx: DeviceContext) raises:
    """Scores -> EPLB router (remap fused in) -> indices."""
    comptime dtype = DType.float32
    var n_topk = num_tokens * _eplb_topk

    var scores_host = ctx.enqueue_create_host_buffer[dtype](
        num_tokens * _num_log
    )
    var bias_host = ctx.enqueue_create_host_buffer[dtype](_num_log)
    random(
        TileTensor(scores_host, row_major(num_tokens, Idx[_num_log])),
        min=0.0,
        max=1.0,
    )
    random(TileTensor(bias_host, row_major(Idx[_num_log])), min=-0.1, max=0.1)

    var scores_src = ctx.enqueue_create_buffer[dtype](num_tokens * _num_log)
    var scores = ctx.enqueue_create_buffer[dtype](num_tokens * _num_log)
    var bias = ctx.enqueue_create_buffer[dtype](_num_log)
    var logcnt = ctx.enqueue_create_buffer[.int32](_num_layers * _num_log)
    var log2phy = ctx.enqueue_create_buffer[.int32](
        _num_layers * _num_log * _max_replicas
    )
    var layer_idx = ctx.enqueue_create_buffer[.int32](1)
    var phy_ref = ctx.enqueue_create_buffer[.int32](n_topk)
    var log_ref = ctx.enqueue_create_buffer[.int32](n_topk)
    var w_ref = ctx.enqueue_create_buffer[dtype](n_topk)
    var phy = ctx.enqueue_create_buffer[.int32](n_topk)
    var log = ctx.enqueue_create_buffer[.int32](n_topk)
    var w = ctx.enqueue_create_buffer[dtype](n_topk)
    ctx.enqueue_copy(scores_src, scores_host)
    ctx.enqueue_copy(bias, bias_host)
    ctx.enqueue_memset(scores, Scalar[dtype](0))
    eplb_tables(ctx, logcnt, log2phy, layer_idx)

    def route(
        mut out_phy: DeviceBuffer[DType.int32],
        mut out_log: DeviceBuffer[DType.int32],
        mut out_w: DeviceBuffer[dtype],
        in_scores: DeviceBuffer[dtype],
    ) raises {imm}:
        single_group_router_eplb[
            n_routed_experts=_num_log,
            n_experts_per_tok=_eplb_topk,
            norm_weights=True,
            num_log=_num_log,
            max_replicas=_max_replicas,
            hash_decorrelate=True,
            target="gpu",
        ](
            TileTensor(out_phy, row_major(num_tokens, Idx[_eplb_topk])),
            TileTensor(out_log, row_major(num_tokens, Idx[_eplb_topk])),
            TileTensor(out_w, row_major(num_tokens, Idx[_eplb_topk])),
            TileTensor(
                in_scores, row_major(num_tokens, Idx[_num_log])
            ).as_imm(),
            TileTensor(bias, row_major(Idx[_num_log])).as_imm(),
            TileTensor(
                logcnt, row_major(Idx[_num_layers], Idx[_num_log])
            ).as_imm(),
            TileTensor(
                log2phy,
                row_major(Idx[_num_layers], Idx[_num_log], Idx[_max_replicas]),
            ).as_imm(),
            TileTensor(layer_idx, row_major(Idx[1])).as_imm(),
            Float32(2.5),
            ctx,
        )

    var want = Indices[_num_phys](ctx, n_topk)
    route(phy_ref, log_ref, w_ref, scores_src)
    ctx.synchronize()
    want.enqueue(ctx, phy_ref)
    ctx.synchronize()

    var got = Indices[_num_phys](ctx, n_topk)
    ctx.enqueue_memset(phy, _SENTINEL)
    ctx.enqueue_memset(log, _SENTINEL)
    ctx.enqueue_memset(w, Scalar[dtype](0))
    ctx.synchronize()
    publish(ctx, scores, scores_src, num_tokens * _num_log)
    route(phy, log, w, scores)
    got.enqueue(ctx, phy)
    ctx.synchronize()

    assert_buffers_equal(ctx, phy, phy_ref, n_topk, "physical indices")
    assert_buffers_equal(ctx, log, log_ref, n_topk, "logical indices")
    assert_buffers_equal(ctx, w, w_ref, n_topk, "router weights")
    assert_indices_equal(ctx, got, want, phy)


def test_eplb_remap_chain(num_tokens: Int, ctx: DeviceContext) raises:
    """Logical ids -> EPLB remap -> indices."""
    var n_topk = num_tokens * _eplb_topk

    var router_host = ctx.enqueue_create_host_buffer[.int32](n_topk)
    for i in range(n_topk):
        router_host[i] = Int32((i * 37 + i // 5) % _num_log)

    var router_src = ctx.enqueue_create_buffer[.int32](n_topk)
    var router_idx = ctx.enqueue_create_buffer[.int32](n_topk)
    var logcnt = ctx.enqueue_create_buffer[.int32](_num_layers * _num_log)
    var log2phy = ctx.enqueue_create_buffer[.int32](
        _num_layers * _num_log * _max_replicas
    )
    var layer_idx = ctx.enqueue_create_buffer[.int32](1)
    var phy_ref = ctx.enqueue_create_buffer[.int32](n_topk)
    var phy = ctx.enqueue_create_buffer[.int32](n_topk)
    ctx.enqueue_copy(router_src, router_host)
    ctx.enqueue_memset(router_idx, Int32(0))
    eplb_tables(ctx, logcnt, log2phy, layer_idx)

    def remap(
        mut out_phy: DeviceBuffer[DType.int32],
        in_idx: DeviceBuffer[DType.int32],
    ) raises {imm}:
        eplb_remap[
            num_log=_num_log,
            max_replicas=_max_replicas,
            K=_eplb_topk,
            hash_decorrelate=True,
            target="gpu",
        ](
            TileTensor(out_phy, row_major(num_tokens, Idx[_eplb_topk])),
            TileTensor(in_idx, row_major(num_tokens, Idx[_eplb_topk])).as_imm(),
            TileTensor(
                logcnt, row_major(Idx[_num_layers], Idx[_num_log])
            ).as_imm(),
            TileTensor(
                log2phy,
                row_major(Idx[_num_layers], Idx[_num_log], Idx[_max_replicas]),
            ).as_imm(),
            TileTensor(layer_idx, row_major(Idx[1])).as_imm(),
            ctx,
        )

    var want = Indices[_num_phys](ctx, n_topk)
    remap(phy_ref, router_src)
    ctx.synchronize()
    want.enqueue(ctx, phy_ref)
    ctx.synchronize()

    var got = Indices[_num_phys](ctx, n_topk)
    ctx.enqueue_memset(phy, _SENTINEL)
    ctx.synchronize()
    publish(ctx, router_idx, router_src, n_topk)
    remap(phy, router_idx)
    got.enqueue(ctx, phy)
    ctx.synchronize()

    assert_buffers_equal(ctx, phy, phy_ref, n_topk, "remapped indices")
    assert_indices_equal(ctx, got, want, phy)


comptime _E = 256
comptime _TOPK = 6
comptime _COLS = 256


def late_topk_ids(
    ctx: DeviceContext, ids_src: DeviceBuffer[DType.int32], slots: Int
) raises:
    """Fills `ids_src` with a spread of expert ids, a few experts left empty."""
    var host = ctx.enqueue_create_host_buffer[.int32](slots)
    for i in range(slots):
        host[i] = Int32((i * 7919 + (i // 3) * 104729) % (_E - 5))
    ctx.enqueue_copy(ids_src, host)
    ctx.synchronize()


def enqueue_quantize[
    with_slots: Bool
](
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.uint8],
    mut scales: DeviceBuffer[NVFP4_SF_DTYPE],
    inp: DeviceBuffer[DType.bfloat16],
    start: DeviceBuffer[DType.uint32],
    scales_offset: DeviceBuffer[DType.uint32],
    expert_ids: DeviceBuffer[DType.int32],
    sf: DeviceBuffer[DType.float32],
    slot_token: DeviceBuffer[DType.int32],
    in_rows: Int,
    slots: Int,
    m_tiles: Int,
) raises:
    comptime K_tiles = ceildiv(_COLS, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
    var scales_shape = Coord(
        m_tiles,
        Idx[K_tiles],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    comptime if with_slots:
        grouped_quantize_dynamic_scaled_fp4_async[
            row_indices_dtype=DType.int32, row_index_divisor=_TOPK
        ](
            TileTensor(out, row_major(slots, Idx[_COLS // 2])),
            TileTensor(scales, row_major(scales_shape)),
            TileTensor(inp, row_major(in_rows, Idx[_COLS])),
            TileTensor(start, row_major(Idx[_E + 1])),
            TileTensor(scales_offset, row_major(Idx[_E])),
            TileTensor(expert_ids, row_major(Idx[_E])),
            TileTensor(sf, row_major(Idx[_E])),
            ctx,
            row_indices=TileTensor(slot_token, row_major(slots))
            .as_unsafe_any_origin()
            .as_imm(),
        )
    else:
        grouped_quantize_dynamic_scaled_fp4_async(
            TileTensor(out, row_major(slots, Idx[_COLS // 2])),
            TileTensor(scales, row_major(scales_shape)),
            TileTensor(inp, row_major(in_rows, Idx[_COLS])),
            TileTensor(start, row_major(Idx[_E + 1])),
            TileTensor(scales_offset, row_major(Idx[_E])),
            TileTensor(expert_ids, row_major(Idx[_E])),
            TileTensor(sf, row_major(Idx[_E])),
            ctx,
        )


def test_quantize_successor[
    with_slots: Bool
](num_tokens: Int, ctx: DeviceContext) raises:
    """Ids published late -> indices (plain or slot layout) -> NVFP4 quantize.

    The quantize reads the CSR, scale offsets, expert ids and, with slots, the
    slot -> token map that the indices kernel writes. Its output must match a
    quantize rerun on the final indices.
    """
    comptime K_tiles = ceildiv(_COLS, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
    comptime SENTINEL = UInt8(0xEE)
    var slots = num_tokens * _TOPK
    var in_rows = num_tokens if with_slots else slots
    var m_tiles = ceildiv(slots, SF_MN_GROUP_SIZE) + min(_E, slots)
    var scale_rows = SF_MN_GROUP_SIZE * m_tiles
    var total_scales = m_tiles * K_tiles * SF_MN_GROUP_SIZE * SF_ATOM_K

    var inp_host = ctx.enqueue_create_host_buffer[.bfloat16](in_rows * _COLS)
    random(
        TileTensor(inp_host, row_major(in_rows, Idx[_COLS])),
        min=-1.0,
        max=1.0,
    )
    var sf_host = ctx.enqueue_create_host_buffer[.float32](_E)
    for e in range(_E):
        sf_host[e] = Float32(1.5 + 0.01 * Float64(e))
    var inp = ctx.enqueue_create_buffer[.bfloat16](in_rows * _COLS)
    var sf = ctx.enqueue_create_buffer[.float32](_E)
    ctx.enqueue_copy(inp, inp_host)
    ctx.enqueue_copy(sf, sf_host)

    var ids_src = ctx.enqueue_create_buffer[.int32](slots)
    late_topk_ids(ctx, ids_src, slots)
    var ids = ctx.enqueue_create_buffer[.int32](slots)
    var order = ctx.enqueue_create_buffer[.uint32](slots)
    var restore_u = ctx.enqueue_create_buffer[.uint32](slots)
    var restore_i = ctx.enqueue_create_buffer[.int32](slots)
    var start = ctx.enqueue_create_buffer[.uint32](_E + 1)
    var expert_ids = ctx.enqueue_create_buffer[.int32](_E)
    var stats = ctx.enqueue_create_buffer[.uint32](2)
    var scales_offset = ctx.enqueue_create_buffer[.uint32](_E)
    var slot_token = ctx.enqueue_create_buffer[.int32](slots)
    var slot_weight = ctx.enqueue_create_buffer[.float32](slots)
    var scale_slot = ctx.enqueue_create_buffer[.int32](scale_rows)
    var slot_weights_in = ctx.enqueue_create_buffer[.float32](slots)
    slot_weights_in.enqueue_fill(Float32(1))

    # Everything the quantize reads from the indices kernel starts zeroed, so
    # a read ahead of its wait sees empty experts and writes nothing.
    ids.enqueue_fill(_SENTINEL)
    start.enqueue_fill(UInt32(0))
    expert_ids.enqueue_fill(Int32(0))
    stats.enqueue_fill(UInt32(0))
    scales_offset.enqueue_fill(UInt32(0))
    slot_token.enqueue_fill(Int32(0))
    var out = ctx.enqueue_create_buffer[.uint8](slots * (_COLS // 2))
    var scales = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](total_scales)
    out.enqueue_fill(SENTINEL)
    scales.enqueue_fill(Scalar[NVFP4_SF_DTYPE](0))
    ctx.synchronize()

    publish(ctx, ids, ids_src, slots)
    comptime if with_slots:
        moe_create_indices_with_slots[_TOPK, "gpu"](
            TileTensor(order, row_major(slots)),
            TileTensor(start, row_major(_E + 1)),
            TileTensor(restore_i, row_major(slots)),
            TileTensor(expert_ids, row_major(_E)),
            TileTensor(stats, row_major(Idx[2])),
            scales_offset.unsafe_ptr().as_unsafe_any_origin(),
            slot_token.unsafe_ptr().as_unsafe_any_origin(),
            slot_weight.unsafe_ptr().as_unsafe_any_origin(),
            scale_slot.unsafe_ptr().as_unsafe_any_origin(),
            scale_rows,
            TileTensor(ids, row_major(slots)).as_imm(),
            slot_weights_in.unsafe_ptr().as_unsafe_any_origin(),
            0,
            _E,
            ctx,
        )
    else:
        moe_create_indices["gpu"](
            TileTensor(order, row_major(slots)),
            TileTensor(start, row_major(_E + 1)),
            TileTensor(restore_u, row_major(slots)),
            TileTensor(expert_ids, row_major(_E)),
            TileTensor(stats, row_major(Idx[2])),
            TileTensor(ids, row_major(slots)),
            ctx,
            scales_offset_p=scales_offset.unsafe_ptr().as_unsafe_any_origin(),
        )
    enqueue_quantize[with_slots](
        ctx,
        out,
        scales,
        inp,
        start,
        scales_offset,
        expert_ids,
        sf,
        slot_token,
        in_rows,
        slots,
        m_tiles,
    )
    ctx.synchronize()

    # Reference: the same quantize on the final indices, nothing overlapping.
    var want_out = ctx.enqueue_create_buffer[.uint8](slots * (_COLS // 2))
    var want_scales = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](total_scales)
    want_out.enqueue_fill(SENTINEL)
    want_scales.enqueue_fill(Scalar[NVFP4_SF_DTYPE](0))
    ctx.synchronize()
    enqueue_quantize[with_slots](
        ctx,
        want_out,
        want_scales,
        inp,
        start,
        scales_offset,
        expert_ids,
        sf,
        slot_token,
        in_rows,
        slots,
        m_tiles,
    )
    ctx.synchronize()
    assert_buffers_equal(
        ctx, out, want_out, slots * (_COLS // 2), "quantized values"
    )
    assert_buffers_equal(
        ctx, scales, want_scales, total_scales, "quantized scales"
    )


def test_grouped_gemm_successor(num_tokens: Int, ctx: DeviceContext) raises:
    """Ids published late -> indices -> bf16 grouped GEMM under PDL.

    The persistent SM100 GEMM reads the usage stats, CSR and expert ids that
    the indices kernel writes; it must match a PDL-off rerun on the final
    indices.
    """
    comptime N = 512
    comptime K = 256
    var slots = num_tokens * _TOPK

    var a_host = ctx.enqueue_create_host_buffer[.bfloat16](slots * K)
    var b_host = ctx.enqueue_create_host_buffer[.bfloat16](_E * N * K)
    random(TileTensor(a_host, row_major(slots, Idx[K])))
    random(TileTensor(b_host, row_major[_E, N, K]()))
    var a = ctx.enqueue_create_buffer[.bfloat16](slots * K)
    var b = ctx.enqueue_create_buffer[.bfloat16](_E * N * K)
    ctx.enqueue_copy(a, a_host)
    ctx.enqueue_copy(b, b_host)

    var ids_src = ctx.enqueue_create_buffer[.int32](slots)
    late_topk_ids(ctx, ids_src, slots)
    var ids = ctx.enqueue_create_buffer[.int32](slots)
    var order = ctx.enqueue_create_buffer[.uint32](slots)
    var restore = ctx.enqueue_create_buffer[.uint32](slots)
    var start = ctx.enqueue_create_buffer[.uint32](_E + 1)
    var expert_ids = ctx.enqueue_create_buffer[.int32](_E)
    var stats = ctx.enqueue_create_buffer[.uint32](2)
    ids.enqueue_fill(_SENTINEL)
    start.enqueue_fill(UInt32(0))
    expert_ids.enqueue_fill(Int32(0))
    stats.enqueue_fill(UInt32(0))
    var c = ctx.enqueue_create_buffer[.bfloat16](slots * N)
    var want = ctx.enqueue_create_buffer[.bfloat16](slots * N)
    c.enqueue_fill(Scalar[DType.bfloat16](-1))
    want.enqueue_fill(Scalar[DType.bfloat16](-1))
    ctx.synchronize()

    publish(ctx, ids, ids_src, slots)
    moe_create_indices["gpu"](
        TileTensor(order, row_major(slots)),
        TileTensor(start, row_major(_E + 1)),
        TileTensor(restore, row_major(slots)),
        TileTensor(expert_ids, row_major(_E)),
        TileTensor(stats, row_major(Idx[2])),
        TileTensor(ids, row_major(slots)),
        ctx,
    )
    grouped_matmul[pdl_level=PDLLevel.OVERLAP_AT_BEGINNING](
        TileTensor(c, row_major(slots, Idx[N])),
        TileTensor(a, row_major(slots, Idx[K])),
        TileTensor(b, row_major[_E, N, K]()),
        TileTensor(start, row_major(_E + 1)),
        TileTensor(expert_ids, row_major(Idx[_E])),
        TileTensor(stats, row_major(Idx[2])),
        ctx,
    )
    ctx.synchronize()

    grouped_matmul(
        TileTensor(want, row_major(slots, Idx[N])),
        TileTensor(a, row_major(slots, Idx[K])),
        TileTensor(b, row_major[_E, N, K]()),
        TileTensor(start, row_major(_E + 1)),
        TileTensor(expert_ids, row_major(Idx[_E])),
        TileTensor(stats, row_major(Idx[2])),
        ctx,
    )
    ctx.synchronize()
    assert_buffers_equal(ctx, c, want, slots * N, "grouped GEMM output")


def stamp_around_wait(times: MutPointer[UInt64, MutUntrackedOrigin]):
    """Records the device clock before and after this grid's own wait."""
    var before = global_perf_counter_ns()
    wait_on_dependent_grids()
    var after = global_perf_counter_ns()
    times[0] = before
    times[1] = after


def test_indices_release_on_entry(ctx: DeviceContext) raises:
    """A grid behind `moe_create_indices` is resident while its scan runs.

    The scan is a single block and takes about as long as its token count, so
    a probe launched behind it stamps the device clock before and after its
    own wait. With the release right after the kernel's wait the probe starts
    early and the two stamps are far apart; released at exit it would start
    only as the scan ends and they would nearly coincide.
    """
    var slots = 1 << 20
    var ids = ctx.enqueue_create_buffer[.int32](slots)
    late_topk_ids(ctx, ids, slots)
    var indices = Indices[_E](ctx, slots)
    var times = ctx.enqueue_create_buffer[DType.uint64](2)
    var times_host = ctx.enqueue_create_host_buffer[DType.uint64](2)
    ctx.synchronize()

    # The first iteration also loads both modules, so a first-launch load
    # cannot serialize the launches that follow.
    var gaps = List[Int]()
    for i in range(6):
        indices.enqueue(ctx, ids)
        ctx.enqueue_function[stamp_around_wait](
            times,
            grid_dim=1,
            block_dim=1,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
        ctx.enqueue_copy(times_host, times)
        ctx.synchronize()
        if i > 0:
            gaps.append(Int(times_host[1] - times_host[0]))
    sort(gaps)
    var gap_ns = gaps[len(gaps) // 2]
    print("  indices start-to-release gap (ns):", gap_ns)
    assert_true(
        gap_ns > 50_000,
        String("probe did not start early: gap ", gap_ns, " ns"),
    )


def main() raises:
    with DeviceContext() as ctx:
        test_indices_release_on_entry(ctx)
        # A decode batch, and a prefill chunk whose router grid is far larger
        # than one wave so its last CTAs are still starting when the indices
        # kernel is released.
        for num_tokens in [8, 4099]:
            test_group_limited_chain(num_tokens, ctx)
            test_sink_gate_chain(num_tokens, ctx)
            test_eplb_router_chain(num_tokens, ctx)
            test_eplb_remap_chain(num_tokens, ctx)
            test_quantize_successor[False](num_tokens, ctx)
            test_quantize_successor[True](num_tokens, ctx)
            test_grouped_gemm_successor(num_tokens, ctx)
