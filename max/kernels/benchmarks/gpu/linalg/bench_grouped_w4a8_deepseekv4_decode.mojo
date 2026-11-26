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
"""Grouped W4A8 matmul at DeepSeek-V4-Flash TP2 decode, called as the model
calls it: 128 local expert groups (ids 0..127, every group present), six
routed slots of which `L` are local (one row each), the rest past the last
group, `estimated_total_m` = 6 (L rows when L > 6).

usage: <bin>                   hash mode: gate_up and down at L in 1..6 and 8;
                               an FNV-1a hash of the bf16 output per case, so
                               two builds compare bit for bit
       <bin> --bench <s> <L> <n>
                               s = 0 gate_up (N 4096, K 4096), 1 down
                               (N 4096, K 2048); n calls, each on the next of
                               43 expert sets so weights stream from DRAM.
                               Time the calls from outside (nsys).
"""

from std.math import ceildiv
from std.memory import bitcast
from std.random import random_ui64, seed
from std.sys import argv

from max.gpu.host import DeviceContext, HostBuffer
from std.simd import _convert_f32_to_float8_ue8m0
from layout import Idx, TileTensor, row_major

from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    W4A8_A_DTYPE,
    W4A8_B_DTYPE,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_block_scaled_sm100_dispatch,
)

comptime E = 128
comptime SLOTS = 6
comptime SETS = 43
comptime bf16 = DType.bfloat16


def _fnv1a(buf: HostBuffer[bf16], n: Int) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for i in range(n):
        h = (h ^ UInt64(bitcast[DType.uint16, 1](buf[i]))) * UInt64(
            0x100000001B3
        )
    return h


def run_case[
    N: Int, K: Int
](ctx: DeviceContext, name: StaticString, L: Int, iters: Int) raises:
    comptime K_PACKED = K // 2
    comptime k_groups = ceildiv(K, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K)
    comptime n_groups = ceildiv(N, SF_MN_GROUP_SIZE)
    comptime b_elems = E * N * K_PACKED
    comptime sfb_stride = n_groups * k_groups * SF_ATOM_M[0] * SF_ATOM_M[
        1
    ] * SF_ATOM_K
    comptime sfb_elems = E * sfb_stride
    var M = max(SLOTS, L)
    var sfa_rows = SF_MN_GROUP_SIZE * M
    comptime sfa_row_elems = k_groups * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    var sets = SETS if iters > 0 else 1

    # Per set: L experts with one row each, ascending ids; the other slots sit
    # past a_offsets[E] (another device's experts).
    var off_h = ctx.enqueue_create_host_buffer[.uint32](sets * (E + 1))
    var soff_h = ctx.enqueue_create_host_buffer[.uint32](sets * E)
    var ids_h = ctx.enqueue_create_host_buffer[.int32](sets * E)
    var esc_h = ctx.enqueue_create_host_buffer[.float32](E)
    ctx.synchronize()
    for e in range(E):
        esc_h[e] = 1.0
    for s in range(sets):
        var first = (s * L) % E
        var rows = 0
        var aligned = 0
        for g in range(E):
            var d = (g - first + E) % E
            var cnt = 1 if d < L else 0
            ids_h[s * E + g] = Int32(g)
            off_h[s * (E + 1) + g] = UInt32(rows)
            soff_h[s * E + g] = UInt32(aligned - rows // SF_MN_GROUP_SIZE)
            rows += cnt
            aligned += ceildiv(cnt, SF_MN_GROUP_SIZE)
        off_h[s * (E + 1) + E] = UInt32(rows)

    var a_h = ctx.enqueue_create_host_buffer[W4A8_A_DTYPE](M * K)
    var sfa_h = ctx.enqueue_create_host_buffer[MXFP8_SF_DTYPE](
        sfa_rows * sfa_row_elems
    )
    ctx.synchronize()
    seed(N * 31 + K + L)
    comptime vals = SIMD[.float32, 8](0.0, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0, 1.5)
    for i in range(M * K):
        a_h[i] = vals[Int(random_ui64(0, 7))].cast[W4A8_A_DTYPE]()
    for i in range(sfa_rows * sfa_row_elems):
        sfa_h[i] = _convert_f32_to_float8_ue8m0[MXFP8_SF_DTYPE](
            Float32(1 << Int(random_ui64(0, 2)))
        )

    var a_d = ctx.enqueue_create_buffer[W4A8_A_DTYPE](M * K)
    var b_d = ctx.enqueue_create_buffer[W4A8_B_DTYPE](b_elems)
    var c_d = ctx.enqueue_create_buffer[bf16](M * N)
    var sfa_d = ctx.enqueue_create_buffer[MXFP8_SF_DTYPE](
        sfa_rows * sfa_row_elems
    )
    var sfb_d = ctx.enqueue_create_buffer[MXFP8_SF_DTYPE](sfb_elems)
    var off_d = ctx.enqueue_create_buffer[.uint32](sets * (E + 1))
    var soff_d = ctx.enqueue_create_buffer[.uint32](sets * E)
    var ids_d = ctx.enqueue_create_buffer[.int32](sets * E)
    var esc_d = ctx.enqueue_create_buffer[.float32](E)
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(sfa_d, sfa_h)
    ctx.enqueue_copy(off_d, off_h)
    ctx.enqueue_copy(soff_d, soff_h)
    ctx.enqueue_copy(ids_d, ids_h)
    ctx.enqueue_copy(esc_d, esc_h)
    ctx.enqueue_memset(c_d, Scalar[bf16](0))

    if iters > 0:
        # Timing only: constant weights and unit scales stream like real ones.
        ctx.enqueue_memset(b_d, UInt8(0x23))
        ctx.enqueue_memset(
            sfb_d, _convert_f32_to_float8_ue8m0[MXFP8_SF_DTYPE](Float32(1.0))
        )
    else:
        # Hash mode: random weights and scales for set 0's experts.
        var bh = ctx.enqueue_create_host_buffer[W4A8_B_DTYPE](N * K_PACKED)
        var sh = ctx.enqueue_create_host_buffer[MXFP8_SF_DTYPE](sfb_stride)
        ctx.enqueue_memset(b_d, UInt8(0))
        ctx.enqueue_memset(sfb_d, Scalar[MXFP8_SF_DTYPE](0.0))
        ctx.synchronize()
        for j in range(L):
            for i in range(N * K_PACKED):
                bh[i] = UInt8(random_ui64(0, 255))
            for i in range(sfb_stride):
                sh[i] = _convert_f32_to_float8_ue8m0[MXFP8_SF_DTYPE](
                    Float32(1 << Int(random_ui64(0, 2)))
                )
            ctx.enqueue_copy(
                b_d.create_sub_buffer[W4A8_B_DTYPE](
                    j * N * K_PACKED, N * K_PACKED
                ),
                bh,
            )
            ctx.enqueue_copy(
                sfb_d.create_sub_buffer[MXFP8_SF_DTYPE](
                    j * sfb_stride, sfb_stride
                ),
                sh,
            )
            ctx.synchronize()

    var a = TileTensor(a_d, row_major(M, Idx[K])).as_unsafe_any_origin()
    var b = TileTensor(
        b_d, row_major(Idx[E], Idx[N], Idx[K_PACKED])
    ).as_unsafe_any_origin()
    var c = TileTensor(c_d, row_major(M, Idx[N])).as_unsafe_any_origin()
    var sfa = TileTensor(
        sfa_d,
        row_major(
            sfa_rows,
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        ),
    ).as_unsafe_any_origin()
    var sfb = TileTensor(
        sfb_d,
        row_major(
            Idx[E],
            Idx[n_groups],
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        ),
    ).as_unsafe_any_origin()
    var esc = TileTensor(esc_d, row_major(Idx[E])).as_unsafe_any_origin()

    var n_calls = iters if iters > 0 else 1
    for it in range(n_calls):
        var s = it % sets
        var off = TileTensor(
            off_d.create_sub_buffer[.uint32](s * (E + 1), E + 1),
            row_major(E + 1),
        ).as_unsafe_any_origin()
        var soff = TileTensor(
            soff_d.create_sub_buffer[.uint32](s * E, E), row_major(E)
        ).as_unsafe_any_origin()
        var ids = TileTensor(
            ids_d.create_sub_buffer[.int32](s * E, E), row_major(E)
        ).as_unsafe_any_origin()
        grouped_matmul_block_scaled_sm100_dispatch[transpose_b=True](
            c, a, b, sfa, sfb, off, soff, ids, esc, E, M, ctx
        )
    ctx.synchronize()

    if iters == 0:
        var c_h = ctx.enqueue_create_host_buffer[bf16](M * N)
        ctx.enqueue_copy(c_h, c_d)
        ctx.synchronize()
        var nz = 0
        for i in range(L * N):
            if c_h[i].cast[.float32]() != 0.0:
                nz += 1
        print(
            name,
            "L=" + String(L),
            "N=" + String(N),
            "K=" + String(K),
            "hash=" + hex(_fnv1a(c_h, M * N)),
            "nonzero_rows_elems=" + String(nz) + "/" + String(L * N),
        )
    _ = b_d^
    _ = sfb_d^


def run(ctx: DeviceContext, shape: Int, L: Int, iters: Int) raises:
    if shape == 0:
        run_case[4096, 4096](ctx, "gate_up", L, iters)
    else:
        run_case[4096, 2048](ctx, "down", L, iters)


def main() raises:
    var args = argv()
    with DeviceContext() as ctx:
        if len(args) > 4 and args[1] == "--bench":
            run(ctx, atol(args[2]), atol(args[3]), atol(args[4]))
            return
        for shape in range(2):
            for L in [1, 2, 3, 4, 5, 6, 8]:
                run(ctx, shape, L, 0)
