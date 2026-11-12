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
"""Exact test of the grouped MXFP8 dispatch at the decode down-projection
shape: 32 experts, N 6144, K 2048, as one EP8 rank of a 256-expert MoE with
hidden 6144 and moe_dim 2048 runs it.

Inputs are small integers with power-of-two scales, so every fp32 sum is
exact and the bf16 output must equal the host sum rounded to bf16, bit for
bit. Each case runs at its decode estimate (the swapped 32-token tile) and
at a prefill estimate (the 128-token tile). Rows past the grouped output
hold a sentinel that must survive. Edge cases: one row, ragged experts, a
partial second token tile, padded (-1) slots and no rows at all.
"""

from std.math import ceildiv
from std.sys.defines import get_defined_int
from std.testing import assert_equal

from layout import Idx, TileTensor, row_major
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    set_scale_factor,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_block_scaled_sm100_dispatch,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory.unsafe import bitcast

comptime E = 32
comptime N = 6144
comptime K = 2048
comptime VEC = MXFP8_SF_VECTOR_SIZE
comptime KGR = ceildiv(K, VEC * SF_ATOM_K)
comptime NGR = ceildiv(N, SF_MN_GROUP_SIZE)
comptime WSF1 = NGR * KGR * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
comptime GUARD = 8
comptime SENTINEL = UInt16(0x7FC1)
comptime POISON = get_defined_int["POISON", 0]()
"""Planted faults, each must fail: 1 hands the kernel the expert ids
rotated by one; 2 drops the last row from the kernel's offsets."""


def _hash(x: UInt64) -> UInt64:
    var h = x * UInt64(0x9E3779B97F4A7C15)
    h ^= h >> 29
    h *= UInt64(0xBF58476D1CE4E5B9)
    h ^= h >> 32
    return h


def _small(i: Int, salt: Int) -> Int:
    return Int(_hash(UInt64(i) * 4 + UInt64(salt)) >> 40) % 5 - 2


def _pow2(i: Int, salt: Int) -> Int:
    return 1 << Int(_hash(UInt64(i) * 4 + UInt64(salt)) >> 62 & 1)


struct _Weights(Movable):
    """All 32 experts' weights and scales, on device and as host integers."""

    var w: DeviceBuffer[DType.float8_e4m3fn]
    var wsf: DeviceBuffer[MXFP8_SF_DTYPE]
    var w_int: List[Int8]
    var wsf_int: List[Int8]
    """Scale values (1 or 2) per (expert, n, k block)."""

    def __init__(out self, ctx: DeviceContext) raises:
        self.w_int = List[Int8](length=E * N * K, fill=0)
        self.wsf_int = List[Int8](length=E * N * (K // VEC), fill=0)
        self.w = ctx.enqueue_create_buffer[DType.float8_e4m3fn](E * N * K)
        self.wsf = ctx.enqueue_create_buffer[MXFP8_SF_DTYPE](E * WSF1)
        # One expert at a time: all 32 exceed one pinned host allocation.
        var w_h = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](N * K)
        var wsf_h = ctx.enqueue_create_host_buffer[MXFP8_SF_DTYPE](WSF1)
        ctx.synchronize()
        for e in range(E):
            for i in range(N * K):
                var v = _small(e * N * K + i, 2)
                self.w_int[e * N * K + i] = Int8(v)
                w_h[i] = Float32(v).cast[DType.float8_e4m3fn]()
            var view = TileTensor(
                wsf_h,
                row_major(
                    Idx[NGR],
                    Idx[KGR],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                ),
            )
            for n in range(N):
                for kb in range(K // VEC):
                    var idx = (e * N + n) * (K // VEC) + kb
                    var p = _pow2(idx, 3)
                    self.wsf_int[idx] = Int8(p)
                    set_scale_factor[SF_VECTOR_SIZE=VEC](
                        view, n, kb * VEC, Float32(p).cast[MXFP8_SF_DTYPE]()
                    )
            ctx.enqueue_copy(
                self.w.create_sub_buffer[DType.float8_e4m3fn](e * N * K, N * K),
                w_h,
            )
            ctx.enqueue_copy(
                self.wsf.create_sub_buffer[MXFP8_SF_DTYPE](e * WSF1, WSF1),
                wsf_h,
            )
            ctx.synchronize()


def _run_case(
    ctx: DeviceContext,
    wts: _Weights,
    name: String,
    rows: List[Int],
    ids: List[Int],
    decode_est: Int,
) raises:
    """Checks one routing at its decode estimate and at a prefill one."""
    var active = len(rows)
    var L = 0
    for r in rows:
        L += r
    var cap = L + GUARD

    var off_h = ctx.enqueue_create_host_buffer[DType.uint32](E + 1)
    var soff_h = ctx.enqueue_create_host_buffer[DType.uint32](E)
    var ids_h = ctx.enqueue_create_host_buffer[DType.int32](E)
    ctx.synchronize()
    var sdim0 = 0
    off_h[0] = 0
    for i in range(E):
        var r = rows[i] if i < active else 0
        soff_h[i] = (
            UInt32(sdim0 - Int(off_h[i]) // SF_MN_GROUP_SIZE) if i
            < active else 0
        )
        off_h[i + 1] = off_h[i] + UInt32(r)
        sdim0 += ceildiv(r, SF_MN_GROUP_SIZE)
        ids_h[i] = Int32(ids[i]) if i < active else Int32(-1)
    sdim0 = max(sdim0, 1)

    var asf_n = sdim0 * KGR * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    var a_h = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](max(L, 1) * K)
    var asf_h = ctx.enqueue_create_host_buffer[MXFP8_SF_DTYPE](asf_n)
    ctx.enqueue_memset(asf_h, Scalar[MXFP8_SF_DTYPE](0.0))
    ctx.synchronize()
    var a_int = List[Int8](length=max(L, 1) * K, fill=0)
    var asf_int = List[Int8](length=max(L, 1) * (K // VEC), fill=0)
    for i in range(L * K):
        var v = _small(i, 1)
        a_int[i] = Int8(v)
        a_h[i] = Float32(v).cast[DType.float8_e4m3fn]()
    var asf_t = TileTensor(
        asf_h,
        row_major(
            sdim0,
            Idx[KGR],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        ),
    )
    for e in range(active):
        var base = (
            Int(off_h[e]) // SF_MN_GROUP_SIZE + Int(soff_h[e])
        ) * SF_MN_GROUP_SIZE
        for r in range(rows[e]):
            var row = Int(off_h[e]) + r
            for kb in range(K // VEC):
                var p = _pow2(row * (K // VEC) + kb, 5)
                asf_int[row * (K // VEC) + kb] = Int8(p)
                set_scale_factor[SF_VECTOR_SIZE=VEC](
                    asf_t, base + r, kb * VEC, Float32(p).cast[MXFP8_SF_DTYPE]()
                )

    # The exact output, rounded to bf16 as the kernel's epilogue rounds.
    var want = List[UInt16](length=max(L, 1) * N, fill=0)
    for e in range(active):
        var x = ids[e]
        for row in range(Int(off_h[e]), Int(off_h[e + 1])):
            for n in range(N):
                var acc = 0
                var wbase = (x * N + n) * K
                for kb in range(K // VEC):
                    var part = 0
                    for k in range(kb * VEC, kb * VEC + VEC):
                        part += Int(a_int[row * K + k]) * Int(
                            wts.w_int[wbase + k]
                        )
                    acc += (
                        part
                        * Int(asf_int[row * (K // VEC) + kb])
                        * Int(wts.wsf_int[(x * N + n) * (K // VEC) + kb])
                    )
                want[row * N + n] = bitcast[DType.uint16, 1](
                    Float32(acc).cast[DType.bfloat16]()
                )

    comptime if POISON == 1:
        for i in range(active):
            ids_h[i] = Int32((ids[i] + 1) % E)
    comptime if POISON == 2:
        for i in range(active, E + 1):
            off_h[i] = off_h[i] - 1 if L > 0 else off_h[i]

    var a_d = ctx.enqueue_create_buffer[DType.float8_e4m3fn](max(L, 1) * K)
    var asf_d = ctx.enqueue_create_buffer[MXFP8_SF_DTYPE](asf_n)
    var off_d = ctx.enqueue_create_buffer[DType.uint32](E + 1)
    var soff_d = ctx.enqueue_create_buffer[DType.uint32](E)
    var ids_d = ctx.enqueue_create_buffer[DType.int32](E)
    var esc_d = ctx.enqueue_create_buffer[DType.float32](E)
    var c_d = ctx.enqueue_create_buffer[DType.bfloat16](cap * N)
    var c_h = ctx.enqueue_create_host_buffer[DType.bfloat16](cap * N)
    ctx.enqueue_memset(esc_d, Float32(1.0))
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(asf_d, asf_h)
    ctx.enqueue_copy(off_d, off_h)
    ctx.enqueue_copy(soff_d, soff_h)
    ctx.enqueue_copy(ids_d, ids_h)

    var a_t = TileTensor(a_d, row_major(L, Idx[K]))
    var w_t = TileTensor(wts.w, row_major(Idx[E], Idx[N], Idx[K]))
    var c_t = TileTensor(c_d, row_major(L, Idx[N]))
    var asf_tt = TileTensor(
        asf_d,
        row_major(
            sdim0,
            Idx[KGR],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        ),
    )
    var wsf_tt = TileTensor(
        wts.wsf,
        row_major(
            Idx[E],
            Idx[NGR],
            Idx[KGR],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        ),
    )
    for est in [decode_est, 4096]:
        ctx.enqueue_memset(c_d, bitcast[DType.bfloat16, 1](SENTINEL))
        grouped_matmul_block_scaled_sm100_dispatch[transpose_b=True](
            c_t,
            a_t,
            w_t,
            asf_tt,
            wsf_tt,
            TileTensor(off_d, row_major(Idx[E + 1])),
            TileTensor(soff_d, row_major(Idx[E])),
            TileTensor(ids_d, row_major(Idx[E])),
            TileTensor(esc_d, row_major(Idx[E])),
            E,
            est,
            ctx,
        )
        ctx.enqueue_copy(c_h, c_d)
        ctx.synchronize()
        var bad = 0
        for i in range(cap * N):
            var got = bitcast[DType.uint16, 1](c_h[i])
            var exp = want[i] if i < L * N else SENTINEL
            if got != exp:
                if bad < 4:
                    print(
                        t"  {name} est={est} row={i // N} col={i % N}"
                        t" got={got} want={exp}"
                    )
                bad += 1
        print(t"{name} rows={L} est={est} bad={bad}")
        assert_equal(bad, 0, String(t"{name} est={est}"))
    _ = a_d^
    _ = asf_d^
    _ = off_d^
    _ = soff_d^
    _ = ids_d^
    _ = esc_d^
    _ = c_d^


def main() raises:
    with DeviceContext() as ctx:
        var wts = _Weights(ctx)
        _run_case(ctx, wts, "one_row", [1], [5], 8)
        _run_case(ctx, wts, "four_experts", [1, 1, 1, 1], [0, 9, 17, 31], 32)
        _run_case(
            ctx,
            wts,
            "ragged_partial_tile",
            [1, 2, 3, 5, 7, 9, 8],
            [0, 3, 7, 12, 20, 31, 16],
            256,
        )
        # 32 tokens top-8 over 256 experts at most, every local expert hit.
        var full = List[Int]()
        var all_ids = List[Int]()
        for e in range(E):
            full.append(1 + (e % 3))
            all_ids.append(E - 1 - e)
        _run_case(ctx, wts, "all_experts", full, all_ids, 256)
        # One expert past one 32-token tile, at a decode estimate.
        _run_case(ctx, wts, "two_token_tiles", [33, 2], [4, 30], 256)
        _run_case(ctx, wts, "no_rows", List[Int](), List[Int](), 0)
