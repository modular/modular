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
"""SM100 parallel split-K whose first launch in the process is recorded into
a device graph, as in serving with graph capture: nothing the first call
enqueues runs until the graph replays, so the tile counters must already be
zero without it. Runs alone in its process so that no earlier split-K call
has set anything up."""

import linalg.matmul.vendor.blas as vendor_blas
from layout import Idx, TileTensor, row_major
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
    _get_splitk_scratch,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
)
from max.gpu.primitives.grid_controls import PDLLevel
from std.utils.index import Index

comptime dtype = DType.bfloat16
comptime N = 2112
comptime K = 14336
comptime M = 8
comptime config = MatmulConfig[dtype, dtype, dtype, True](
    cluster_shape=Index(1, 1, 1),
    mma_shape=Index(128, 16, 16),
    cta_group=1,
    AB_swapped=True,
    k_group_size=2,
    num_split_k=8,
    num_pipeline_stages=8,
    num_accum_pipeline_stages=1,
    num_clc_pipeline_stages=0,
)


def _run(
    ctx: DeviceContext,
    a: DeviceBuffer[dtype],
    b: DeviceBuffer[dtype],
    c: DeviceBuffer[dtype],
) raises:
    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=True, config=config, pdl_level=PDLLevel.ON
    ](
        TileTensor(c, row_major(M, Idx[N])),
        TileTensor(a, row_major(M, Idx[K])),
        TileTensor(b, row_major(Idx[N], Idx[K])),
        ctx,
    )


def _untracked(
    buf: DeviceBuffer[dtype],
) -> MutPointer[Scalar[dtype], MutUntrackedOrigin]:
    # vendor BLAS takes a mutable output tensor; the caller keeps every
    # buffer alive past the call.
    return (
        buf.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


def _check(
    ctx: DeviceContext,
    c: DeviceBuffer[dtype],
    r: DeviceBuffer[dtype],
    what: String,
) raises:
    var c_host = ctx.enqueue_create_host_buffer[dtype](M * N)
    var r_host = ctx.enqueue_create_host_buffer[dtype](M * N)
    ctx.enqueue_copy(c_host, c)
    ctx.enqueue_copy(r_host, r)
    ctx.synchronize()
    for i in range(M * N):
        if c_host[i] != r_host[i]:
            raise Error(
                t"split-K {what}: c[{i // N}, {i % N}] = {c_host[i]},"
                t" expected {r_host[i]}"
            )
    print(t"{what} OK")


def main() raises:
    with DeviceContext() as ctx:
        var a_host = ctx.enqueue_create_host_buffer[dtype](M * K)
        var b_host = ctx.enqueue_create_host_buffer[dtype](N * K)
        ctx.synchronize()
        # Small integers keep every partial and sum exact in fp32, so the
        # output must equal vendor BLAS bit for bit.
        for i in range(M * K):
            a_host[i] = Scalar[dtype](Float32((i * 7 + 3) % 5 - 2))
        for i in range(N * K):
            b_host[i] = Scalar[dtype](Float32((i * 11 + 1) % 5 - 2))
        var a = ctx.enqueue_create_buffer[dtype](M * K)
        var b = ctx.enqueue_create_buffer[dtype](N * K)
        var c_graph = ctx.enqueue_create_buffer[dtype](M * N)
        var c_eager = ctx.enqueue_create_buffer[dtype](M * N)
        var r = ctx.enqueue_create_buffer[dtype](M * N)
        ctx.enqueue_copy(a, a_host)
        ctx.enqueue_copy(b, b_host)

        def build(mut builder: DeviceGraphBuilder) raises {imm}:
            with builder.recording_context() as rec:
                _run(rec, a, b, c_graph)

        var graph = DeviceGraph.create(ctx, build)
        # The pool may hand the first call memory holding anything, and
        # nothing it enqueued has run yet.
        _get_splitk_scratch(ctx).enqueue_fill(UInt8(0xFF))

        vendor_blas.matmul(
            ctx,
            TileTensor(_untracked(r), row_major(M, Idx[N])),
            TileTensor(_untracked(a), row_major(M, Idx[K])),
            TileTensor(_untracked(b), row_major(Idx[N], Idx[K])),
            c_row_major=True,
            transpose_b=True,
        )

        # An eager launch before the graph has ever replayed.
        _run(ctx, a, b, c_eager)
        _check(ctx, c_eager, r, "eager launch before the first replay")

        graph.replay()
        graph.replay()
        _check(ctx, c_graph, r, "graph replays")

        c_eager.enqueue_fill(Scalar[dtype](-7))
        _run(ctx, a, b, c_eager)
        _check(ctx, c_eager, r, "eager launch after replays")
