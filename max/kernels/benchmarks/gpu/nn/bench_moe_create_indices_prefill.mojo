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
"""`moe_create_indices` at prefill sizes: the single-CTA kernel versus the
production entry point (multi-CTA above a token threshold), with a check that
both build the same CSR.
"""

from std.random import random_ui64, seed
from std.sys import get_defined_int

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.moe import moe_create_indices, moe_create_indices_kernel

comptime E = get_defined_int["EXPERTS", 128]()
comptime ITERS = get_defined_int["ITERS", 100]()
comptime TOPK = 4


def run(ctx: DeviceContext, tokens: Int) raises:
    var n = tokens * TOPK
    var ids_h = ctx.enqueue_create_host_buffer[DType.int32](n)
    seed(11)
    for i in range(n):
        # Skewed: half of the slots go to the first eighth of the experts.
        var r = Int(random_ui64(0, UInt64(E - 1)))
        if i % 2 == 0:
            r = r % (E // 8)
        ids_h[i] = Int32(r)
    var ids_d = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(ids_d, ids_h)

    var order_a = ctx.enqueue_create_buffer[DType.uint32](n)
    var order_b = ctx.enqueue_create_buffer[DType.uint32](n)
    var start_a = ctx.enqueue_create_buffer[DType.uint32](E + 1)
    var start_b = ctx.enqueue_create_buffer[DType.uint32](E + 1)
    var restore_a = ctx.enqueue_create_buffer[DType.uint32](n)
    var restore_b = ctx.enqueue_create_buffer[DType.uint32](n)
    var eid_a = ctx.enqueue_create_buffer[DType.int32](E)
    var eid_b = ctx.enqueue_create_buffer[DType.int32](E)
    var stats_a = ctx.enqueue_create_buffer[DType.uint32](2)
    var stats_b = ctx.enqueue_create_buffer[DType.uint32](2)
    var so_a = ctx.enqueue_create_buffer[DType.uint32](E)
    var so_b = ctx.enqueue_create_buffer[DType.uint32](E)

    var so_a_p = so_a.unsafe_ptr().as_unsafe_any_origin()
    var so_b_p = so_b.unsafe_ptr().as_unsafe_any_origin()
    var ids = TileTensor(ids_d, row_major(n)).as_imm()
    var oa = TileTensor(order_a, row_major(n))
    var ob = TileTensor(order_b, row_major(n))
    var sa = TileTensor(start_a, row_major[E + 1]())
    var sb = TileTensor(start_b, row_major[E + 1]())
    var ra = TileTensor(restore_a, row_major(n))
    var rb = TileTensor(restore_b, row_major(n))
    var ea = TileTensor(eid_a, row_major[E]())
    var eb = TileTensor(eid_b, row_major[E]())
    var ua = TileTensor(stats_a, row_major[2]())
    var ub = TileTensor(stats_b, row_major[2]())

    def single_cta() raises {imm}:
        comptime kernel = moe_create_indices_kernel[
            DType.int32,
            DType.uint32,
            oa.LayoutType,
            sa.LayoutType,
            ra.LayoutType,
            ea.LayoutType,
            ua.LayoutType,
            ids.LayoutType,
            512,
            TokenExpertOrderEngine=oa.Engine,
            ExpertStartIndicesEngine=sa.Engine,
            RestoreTokenOrderEngine=ra.Engine,
            ExpertIdsEngine=ea.Engine,
            ExpertUsageStatsEngine=ua.Engine,
            TopkIdsEngine=ids.Engine,
        ]
        ctx.enqueue_function[kernel](
            oa,
            sa,
            ra,
            ea,
            ua,
            ids,
            Optional[MutPointer[UInt32, MutAnyOrigin]](so_a_p),
            Int32(0),
            Int32(0),
            Optional[MutPointer[Float32, MutAnyOrigin]](),
            Optional[MutPointer[Int32, MutAnyOrigin]](),
            Optional[MutPointer[Float32, MutAnyOrigin]](),
            Optional[MutPointer[Int32, MutAnyOrigin]](),
            Int32(0),
            grid_dim=1,
            block_dim=512,
        )

    def production() raises {imm}:
        moe_create_indices(
            ob,
            sb,
            rb,
            eb,
            ub,
            ids,
            ctx,
            Optional[MutPointer[UInt32, MutAnyOrigin]](so_b_p),
        )

    single_cta()
    production()
    ctx.synchronize()
    var t_single = (
        Float64(ctx.execution_time(single_cta, ITERS)) / 1e3 / Float64(ITERS)
    )
    var t_prod = (
        Float64(ctx.execution_time(production, ITERS)) / 1e3 / Float64(ITERS)
    )

    var start_ha = ctx.enqueue_create_host_buffer[DType.uint32](E + 1)
    var start_hb = ctx.enqueue_create_host_buffer[DType.uint32](E + 1)
    var order_hb = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var restore_hb = ctx.enqueue_create_host_buffer[DType.uint32](n)
    var eid_ha = ctx.enqueue_create_host_buffer[DType.int32](E)
    var eid_hb = ctx.enqueue_create_host_buffer[DType.int32](E)
    var stats_ha = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var stats_hb = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var so_ha = ctx.enqueue_create_host_buffer[DType.uint32](E)
    var so_hb = ctx.enqueue_create_host_buffer[DType.uint32](E)
    ctx.enqueue_copy(start_ha, start_a)
    ctx.enqueue_copy(start_hb, start_b)
    ctx.enqueue_copy(order_hb, order_b)
    ctx.enqueue_copy(restore_hb, restore_b)
    ctx.enqueue_copy(eid_ha, eid_a)
    ctx.enqueue_copy(eid_hb, eid_b)
    ctx.enqueue_copy(stats_ha, stats_a)
    ctx.enqueue_copy(stats_hb, stats_b)
    ctx.enqueue_copy(so_ha, so_a)
    ctx.enqueue_copy(so_hb, so_b)
    ctx.synchronize()

    var bad = 0
    for e in range(E + 1):
        if start_ha[e] != start_hb[e]:
            bad += 1
    for e in range(E):
        if eid_ha[e] != eid_hb[e] or so_ha[e] != so_hb[e]:
            bad += 1
    if stats_ha[0] != stats_hb[0] or stats_ha[1] != stats_hb[1]:
        bad += 1
    # The new order is a valid placement: restore inverts order, and every
    # slot lands inside its expert's segment.
    for i in range(n):
        var pos = Int(restore_hb[i])
        var e = Int(ids_h[i])
        if pos >= n or Int(order_hb[pos]) != i:
            bad += 1
        elif pos < Int(start_hb[e]) or pos >= Int(start_hb[e + 1]):
            bad += 1
    print(
        "tokens=" + String(tokens),
        "slots=" + String(n),
        "single_cta_us=" + String(t_single),
        "production_us=" + String(t_prod),
        "mismatches=" + String(bad),
    )


def main() raises:
    with DeviceContext() as ctx:
        for tokens in [16384, 15000, 8192, 4096, 512]:
            run(ctx, tokens)
