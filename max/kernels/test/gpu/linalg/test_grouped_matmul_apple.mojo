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
"""Tests the Apple M5 grouped (MoE) matmul against the naive kernel.

Every case runs the production dispatch `grouped_matmul(...)` and both Apple
kernels directly (the grouped GEMV at several `tokens` settings and the
grouped MMA), whatever the routing would pick, so each kernel is checked at
every shape. The reference `naive_grouped_matmul` reads the same bf16 or fp16
bytes and accumulates in fp32; the only slack is fp32 reduction order plus the
output rounding.

Shapes include MoE expert dims gate_up `[E, 6144, 3584]` and down
`[E, 3584, 3072]` at batch-1 decode (16 experts x 1 token), small-batch decode,
a ragged prefill, experts at the end of an 896-expert stack, and the LoRA SGMV
shrink and expand shapes (rank 16 on a 4096-wide layer).
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from layout import Coord, Idx, TileTensor, row_major

from linalg.grouped_matmul import grouped_matmul, naive_grouped_matmul
from linalg.matmul.gpu.apple.grouped_matmul import (
    enqueue_apple_grouped_gemv,
    enqueue_apple_grouped_mma,
)
from std.testing import assert_almost_equal

from std.utils import IndexList
from std.utils.index import Index


def _fill(seed: Int, i: Int) -> Float32:
    """Deterministic pseudo-random value in [-1, 1) (cheap enough for GBs)."""
    var h = UInt32(i * 2654435761 + seed * 40503) ^ UInt32(seed)
    h ^= h >> 15
    h *= 2246822519
    h ^= h >> 13
    return Float32(Int(h % 2048) - 1024) * Float32(1.0 / 1024.0)


def _check[
    dtype: DType
](
    ctx: DeviceContext,
    out_buf: DeviceBuffer[dtype],
    out_host: HostBuffer[dtype],
    ref_host: HostBuffer[dtype],
    n: Int,
    atol: Float64,
    rtol: Float64,
    name: String,
) raises:
    ctx.enqueue_copy(out_host, out_buf)
    ctx.synchronize()
    for i in range(len(out_host)):
        assert_almost_equal(
            out_host[i],
            ref_host[i],
            msg=String(t"{name} m={i // n} n={i % n}"),
            atol=atol,
            rtol=rtol,
        )
    # Poison so a kernel that writes nothing cannot pass the next check.
    out_buf.enqueue_fill(Scalar[dtype](1234.0))


def _run_case[
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
    in_type: DType = DType.bfloat16,
](
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
    atol: Float64 = 5e-2,
    rtol: Float64 = 1e-2,
    expert_base: Int = 0,
) raises:
    """Checks one routing against `naive_grouped_matmul`.

    With `expert_base > 0` only experts `[expert_base, num_experts)` are
    allocated, and the `[num_experts, N, K]` view starts `expert_base`
    experts before the allocation. That exercises expert offsets past 2^31
    elements without allocating the whole stack; `expert_ids` must then all
    be `>= expert_base`.
    """
    comptime N = expert_shape[0]
    comptime K = expert_shape[1]
    var num_active = len(num_tokens_by_expert)

    var total_m = 0
    var max_m = 0
    for i in range(num_active):
        total_m += num_tokens_by_expert[i]
        max_m = max(max_m, num_tokens_by_expert[i])
    print(
        t"E={num_experts} N={N} K={K} in={in_type} out={out_type}"
        t" active={num_active}"
        t" total_m={total_m} max_m={max_m}"
    )

    var a_size = total_m * K
    var c_size = total_m * N
    var b_size = (num_experts - expert_base) * N * K

    var a_host = ctx.enqueue_create_host_buffer[in_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[in_type](b_size)
    var off_host = ctx.enqueue_create_host_buffer[.uint32](num_experts + 1)
    var eid_host = ctx.enqueue_create_host_buffer[.int32](num_experts)
    ctx.synchronize()
    for i in range(a_size):
        a_host[i] = _fill(1, i).cast[in_type]()
    for i in range(b_size):
        b_host[i] = _fill(2, i).cast[in_type]()
    for i in range(num_experts + 1):
        off_host[i] = 0
    for i in range(num_experts):
        eid_host[i] = 0
    for i in range(num_active):
        off_host[i + 1] = off_host[i] + UInt32(num_tokens_by_expert[i])
        eid_host[i] = Int32(expert_ids[i])

    var a_buf = ctx.enqueue_create_buffer[in_type](a_size)
    var b_buf = ctx.enqueue_create_buffer[in_type](b_size)
    var off_buf = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var eid_buf = ctx.enqueue_create_buffer[.int32](num_experts)
    ctx.enqueue_copy(a_buf, a_host)
    ctx.enqueue_copy(b_buf, b_host)
    ctx.enqueue_copy(off_buf, off_host)
    ctx.enqueue_copy(eid_buf, eid_host)

    var a = TileTensor(a_buf, row_major(Coord(total_m, Idx[K])))
    var b = TileTensor(
        b_buf.unsafe_ptr() - expert_base * N * K,
        row_major[num_experts, N, K](),
    )
    var off = TileTensor(off_buf, row_major(Coord(num_experts + 1)))
    var eid = TileTensor(eid_buf, row_major(Coord(Idx[num_experts])))

    var ref_buf = ctx.enqueue_create_buffer[out_type](c_size)
    var ref_c = TileTensor(ref_buf, row_major(Coord(total_m, Idx[N])))
    naive_grouped_matmul(ref_c, a, b, off, eid, max_m, num_active, ctx)
    var ref_host = ctx.enqueue_create_host_buffer[out_type](c_size)
    ctx.enqueue_copy(ref_host, ref_buf)
    ctx.synchronize()

    var out_buf = ctx.enqueue_create_buffer[out_type](c_size)
    var out_host = ctx.enqueue_create_host_buffer[out_type](c_size)
    var out_c = TileTensor(out_buf, row_major(Coord(total_m, Idx[N])))

    out_buf.enqueue_fill(Scalar[out_type](1234.0))
    grouped_matmul(out_c, a, b, off, eid, max_m, num_active, ctx)
    _check(ctx, out_buf, out_host, ref_host, N, atol, rtol, "dispatch")

    # The dispatch with a fused epilogue, here `2 * x + 1`, which writes its
    # own buffer.
    var epi_buf = ctx.enqueue_create_buffer[out_type](c_size)
    var epi_ptr = epi_buf.unsafe_ptr()

    @inline(.always)
    def epilogue_fn[
        dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: IndexList[2], val: SIMD[dtype, width]) {var epi_ptr}:
        var out = val.cast[.float32]() * 2 + 1
        epi_ptr.unsafe_store[width=width](
            idx[0] * N + idx[1], out.cast[out_type]()
        )

    grouped_matmul[has_epilogue_fn=True](
        out_c, a, b, off, eid, max_m, num_active, ctx, epilogue_fn
    )
    ctx.enqueue_copy(out_host, epi_buf)
    ctx.synchronize()
    _ = epi_buf^
    for i in range(c_size):
        var expected = ref_host[i].cast[.float32]() * 2 + 1
        assert_almost_equal(
            out_host[i],
            expected.cast[out_type](),
            msg=String(t"dispatch_epilogue m={i // N} n={i % N}"),
            atol=2 * atol,
            rtol=rtol,
        )
    out_buf.enqueue_fill(Scalar[out_type](1234.0))

    enqueue_apple_grouped_mma(out_c, a, b, off, eid, max_m, num_active, ctx)
    _check(ctx, out_buf, out_host, ref_host, N, atol, rtol, "mma_g4")

    enqueue_apple_grouped_mma[groups_per_launch=1](
        out_c, a, b, off, eid, max_m, num_active, ctx
    )
    _check(ctx, out_buf, out_host, ref_host, N, atol, rtol, "mma_g1")

    enqueue_apple_grouped_mma[groups_per_launch=64](
        out_c, a, b, off, eid, max_m, num_active, ctx
    )
    _check(ctx, out_buf, out_host, ref_host, N, atol, rtol, "mma_g64")

    enqueue_apple_grouped_gemv(out_c, a, b, off, eid, num_active, ctx)
    _check(ctx, out_buf, out_host, ref_host, N, atol, rtol, "gemv_r2_t1")

    enqueue_apple_grouped_gemv[rows_per_sg=1, tokens_per_pass=4](
        out_c, a, b, off, eid, num_active, ctx
    )
    _check(ctx, out_buf, out_host, ref_host, N, atol, rtol, "gemv_r1_t4")

    enqueue_apple_grouped_gemv[rows_per_sg=2, tokens_per_pass=8, acc_width=1](
        out_c, a, b, off, eid, num_active, ctx
    )
    _check(ctx, out_buf, out_host, ref_host, N, atol, rtol, "gemv_r2_t8_a1")

    enqueue_apple_grouped_gemv[
        rows_per_sg=4, tokens_per_pass=2, num_sg=4, acc_width=2
    ](out_c, a, b, off, eid, num_active, ctx)
    _check(ctx, out_buf, out_host, ref_host, N, atol, rtol, "gemv_r4_t2_a2")

    _ = a_buf^
    _ = b_buf^
    _ = off_buf^
    _ = eid_buf^
    _ = ref_buf^
    _ = out_buf^
    print("  PASS")


def main() raises:
    with DeviceContext() as ctx:
        if ctx.compute_capability() != 5:
            print("skip: Apple grouped matmul requires Apple M5 (cc==5)")
            return

        comptime BF16 = DType.bfloat16
        comptime F16 = DType.float16
        comptime F32 = DType.float32

        # Single group, aligned.
        _run_case[BF16, 1, Index(64, 256)]([3], [0], ctx)
        # Several groups incl. an empty one; fp32 output.
        _run_case[F32, 4, Index(128, 512)](
            [3, 0, 5, 2], [2, 0, 3, 1], ctx, atol=1e-3, rtol=1e-4
        )
        # N not a multiple of rows or the tile; K < 256 (scalar K loop only).
        _run_case[BF16, 4, Index(70, 200)]([7, 0, 20], [1, 0, 3], ctx)
        # K % 8 != 0 (scalar tail, 2-byte alignment path).
        _run_case[F32, 3, Index(45, 203)](
            [1, 4, 9], [2, 0, 1], ctx, atol=1e-3, rtol=1e-4
        )
        # Inactive group (expert_ids == -1) must produce zeros.
        _run_case[BF16, 2, Index(128, 64)]([16, 24], [0, -1], ctx)
        # Ragged prefill-sized groups (MMA route through the dispatch).
        _run_case[BF16, 4, Index(96, 320)]([70, 130, 1, 64], [3, 1, 0, 2], ctx)
        # fp16 operands, decode and prefill routes, fp16 and fp32 outputs.
        _run_case[F16, 4, Index(128, 512), in_type=F16](
            [3, 0, 5, 2], [2, 0, 3, 1], ctx
        )
        _run_case[F32, 4, Index(96, 320), in_type=F16](
            [70, 130, 1, 64], [3, 1, 0, 2], ctx, atol=1e-3, rtol=1e-4
        )

        # LoRA SGMV (`mo.lora_sgmv.ragged`): 4 adapters of rank 16 on a
        # 4096-wide layer, shrink `[4, 16, 4096]` and expand `[4, 4096, 16]`.
        # Id -1 is a request without an adapter. Decode, then prefill.
        var lora_ids: List[Int] = [1, -1, 3, 0]
        var lora_decode: List[Int] = [2, 1, 3, 2]
        var lora_prefill: List[Int] = [300, 64, 0, 129]
        _run_case[BF16, 4, Index(16, 4096)](
            lora_decode, lora_ids, ctx, atol=0.25
        )
        _run_case[BF16, 4, Index(4096, 16)](lora_decode, lora_ids, ctx)
        _run_case[BF16, 4, Index(16, 4096)](
            lora_prefill, lora_ids, ctx, atol=0.25
        )
        _run_case[BF16, 4, Index(4096, 16)](lora_prefill, lora_ids, ctx)

        # Batch-1 decode: top-16 of 32 experts, 1 token each.
        comptime E = 32
        var ids16: List[Int] = [
            3,
            17,
            0,
            29,
            8,
            12,
            31,
            5,
            22,
            14,
            9,
            26,
            1,
            19,
            11,
            30,
        ]
        var ones16: List[Int] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # gate_up
        _run_case[BF16, E, Index(6144, 3584)](ones16, ids16, ctx, atol=0.25)
        # down
        _run_case[BF16, E, Index(3584, 3072)](ones16, ids16, ctx, atol=0.25)

        # Small-batch decode (batch 4): 1-3 tokens per expert.
        var ids_b4: List[Int] = [
            0,
            2,
            4,
            5,
            7,
            9,
            10,
            13,
            15,
            16,
            18,
            20,
            21,
            23,
            25,
            27,
            28,
            31,
        ]
        var tok_b4: List[Int] = [
            3,
            1,
            2,
            1,
            4,
            1,
            1,
            2,
            3,
            1,
            2,
            1,
            1,
            4,
            3,
            2,
            1,
            1,
        ]
        _run_case[BF16, E, Index(6144, 3584)](tok_b4, ids_b4, ctx, atol=0.25)

        # Batch-16 decode: 5-16 tokens per expert (GEMV `tokens_per_pass=8` route).
        var ids_b16: List[Int] = [1, 4, 8, 13, 16, 22, 27]
        var tok_b16: List[Int] = [9, 16, 5, 12, 7, 11, 14]
        _run_case[BF16, E, Index(3584, 3072)](tok_b16, ids_b16, ctx, atol=0.25)

        # The last experts of an 896-expert stack: offsets of ~19.7e9
        # elements, past both int32 and uint32. Decode and prefill routes.
        _run_case[BF16, 896, Index(6144, 3584)](
            [1, 3], [895, 894], ctx, atol=0.25, expert_base=894
        )
        _run_case[BF16, 896, Index(6144, 3584)](
            [40, 0, 70], [894, 895, 895], ctx, atol=0.25, expert_base=894
        )

        # Prefill: ragged groups up to a few hundred tokens, one empty.
        var ids_pf: List[Int] = [6, 1, 30, 12, 20]
        var tok_pf: List[Int] = [300, 0, 517, 64, 33]
        _run_case[BF16, E, Index(3584, 3072)](tok_pf, ids_pf, ctx, atol=0.25)

        print("all Apple grouped matmul tests passed")
