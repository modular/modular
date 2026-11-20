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
"""PDL ordering test for the SM100 grouped NVFP4 matmul (target: sm_100).

The grouped matmul launches with programmatic dependent launch (its default is
`PDLLevel.ON`), so it can become resident while the kernel before it is still
running. In an expert-parallel MoE that kernel is the dispatch callback, which
writes the matmul's routing operands (`a_offsets`, `a_scale_offsets` and
`expert_ids`). The matmul must not read any of them before
`wait_on_dependent_grids()`.

Here a producer kernel releases its dependents on entry, sleeps, and only then
publishes the real routing over zeroed tables. A matmul that reads the tables
before its wait sees every expert empty and leaves the output at its sentinel.
The PDL output must match a run whose routing was published first, byte for
byte. A matmul that loads weights for the expert it read early would also hang
here, since its load warp would see no work while the other warps see the
published routing. Programmatic launch only overlaps inside a device graph, so
the producer and matmul are recorded and replayed together.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from std.sys.intrinsics import llvm_intrinsic

from internal_utils._utils import (
    InitializationType,
    _init_block_scaled_scales_launch,
    init_vector_launch,
)
from layout import Idx, TileTensor, row_major
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_nvfp4_dispatch,
)
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
)
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
)
from std.testing import assert_equal

comptime fp4 = DType.uint8
comptime sdt = NVFP4_SF_DTYPE
comptime bf16 = DType.bfloat16


def publish_after_delay(
    offsets: MutPointer[UInt32, MutUntrackedOrigin],
    scale_offsets: MutPointer[UInt32, MutUntrackedOrigin],
    expert_ids: MutPointer[Int32, MutUntrackedOrigin],
    src_offsets: MutPointer[UInt32, MutUntrackedOrigin],
    src_scale_offsets: MutPointer[UInt32, MutUntrackedOrigin],
    src_expert_ids: MutPointer[Int32, MutUntrackedOrigin],
    num_active_experts: Int32,
):
    """Releases dependents at once, then publishes the routing late."""
    launch_dependent_grids()
    # nanosleep takes nanoseconds and caps near 1 ms per call.
    for _ in range(5):
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(1_000_000))
    for i in range(Int(num_active_experts) + 1):
        offsets[i] = src_offsets[i]
    for i in range(Int(num_active_experts)):
        scale_offsets[i] = src_scale_offsets[i]
        expert_ids[i] = src_expert_ids[i]


def _bits(
    ctx: DeviceContext, buf: DeviceBuffer[bf16], n: Int
) raises -> List[UInt16]:
    var host = unsafe_alloc[Scalar[bf16]](n)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    var raw = host.unsafe_bitcast[UInt16]()
    var out = List[UInt16]()
    for i in range(n):
        out.append(raw[i])
    host.unsafe_free()
    return out^


def _launch[
    num_experts: Int, N: Int, K: Int, level: PDLLevel
](
    ctx: DeviceContext,
    c: DeviceBuffer[bf16],
    a: DeviceBuffer[fp4],
    a_sc: DeviceBuffer[sdt],
    b: DeviceBuffer[fp4],
    b_sc: DeviceBuffer[sdt],
    expert_scales: DeviceBuffer[DType.float32],
    off: DeviceBuffer[DType.uint32],
    soff: DeviceBuffer[DType.uint32],
    eids: DeviceBuffer[DType.int32],
    nae: Int,
    M: Int,
    sf_tiles: Int,
) raises:
    comptime packed_K = K // 2
    comptime k_groups = ceildiv(K, 64)
    comptime n_groups = ceildiv(N, SF_MN_GROUP_SIZE)
    grouped_matmul_nvfp4_dispatch[transpose_b=True, pdl_level=level](
        TileTensor(c, row_major(Int64(M), Idx[N])),
        TileTensor(a, row_major(Int64(M), Idx[packed_K])),
        TileTensor(b, row_major(Idx[num_experts], Idx[N], Idx[packed_K])),
        TileTensor(
            a_sc,
            row_major(
                Int64(sf_tiles),
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            ),
        ),
        TileTensor(
            b_sc,
            row_major(
                Idx[num_experts],
                Idx[n_groups],
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            ),
        ),
        TileTensor(off, row_major(Int64(nae + 1))),
        TileTensor(soff, row_major(Int64(nae))),
        TileTensor(eids, row_major(Int64(nae))),
        TileTensor(expert_scales, row_major(Idx[num_experts])),
        nae,
        M,
        ctx,
    )


def _test_pdl_order[
    num_experts: Int, N: Int, K: Int, level: PDLLevel = PDLLevel.ON
](ctx: DeviceContext, tokens_by_expert: List[Int], ids: List[Int]) raises:
    comptime packed_K = K // 2
    comptime k_groups = ceildiv(K, 64)
    comptime n_groups = ceildiv(N, SF_MN_GROUP_SIZE)
    var nae = len(ids)

    var offsets_host = unsafe_alloc[UInt32](nae + 1)
    var scale_offsets_host = unsafe_alloc[UInt32](nae)
    var ids_host = unsafe_alloc[Int32](nae)
    var zeros_u32 = unsafe_alloc[UInt32](nae + 1)
    var zeros_i32 = unsafe_alloc[Int32](nae)
    var expert_scales_host = unsafe_alloc[Float32](num_experts)
    offsets_host[0] = 0
    zeros_u32[nae] = 0
    var sf_tiles = 0
    for i in range(nae):
        offsets_host[i + 1] = offsets_host[i] + UInt32(tokens_by_expert[i])
        scale_offsets_host[i] = UInt32(
            sf_tiles - Int(offsets_host[i]) // SF_MN_GROUP_SIZE
        )
        sf_tiles += ceildiv(tokens_by_expert[i], SF_MN_GROUP_SIZE)
        ids_host[i] = Int32(ids[i])
        zeros_u32[i] = 0
        zeros_i32[i] = 0
    for e in range(num_experts):
        expert_scales_host[e] = 0.5 + Float32(e) * 0.25
    var M = Int(offsets_host[nae])
    var out_n = M * N

    var a = ctx.enqueue_create_buffer[fp4](M * packed_K)
    var a_sc = ctx.enqueue_create_buffer[sdt](sf_tiles * k_groups * 512)
    var b = ctx.enqueue_create_buffer[fp4](num_experts * N * packed_K)
    var b_sc = ctx.enqueue_create_buffer[sdt](
        num_experts * n_groups * k_groups * 512
    )
    init_vector_launch[fp4](
        a, M * packed_K, InitializationType.uniform_distribution, ctx
    )
    init_vector_launch[fp4](
        b,
        num_experts * N * packed_K,
        InitializationType.uniform_distribution,
        ctx,
    )
    _init_block_scaled_scales_launch[sdt](a_sc, sf_tiles * k_groups * 512, ctx)
    _init_block_scaled_scales_launch[sdt](
        b_sc, num_experts * n_groups * k_groups * 512, ctx
    )

    # The routing the matmul should see, and the tables it reads.
    var src_offsets = ctx.enqueue_create_buffer[.uint32](nae + 1)
    var src_scale_offsets = ctx.enqueue_create_buffer[.uint32](nae)
    var src_ids = ctx.enqueue_create_buffer[.int32](nae)
    var offsets = ctx.enqueue_create_buffer[.uint32](nae + 1)
    var scale_offsets = ctx.enqueue_create_buffer[.uint32](nae)
    var expert_ids = ctx.enqueue_create_buffer[.int32](nae)
    var expert_scales = ctx.enqueue_create_buffer[.float32](num_experts)
    ctx.enqueue_copy(src_offsets, offsets_host)
    ctx.enqueue_copy(src_scale_offsets, scale_offsets_host)
    ctx.enqueue_copy(src_ids, ids_host)
    ctx.enqueue_copy(expert_scales, expert_scales_host)

    var c_ref = ctx.enqueue_create_buffer[bf16](out_n)
    var c_pdl = ctx.enqueue_create_buffer[bf16](out_n)
    c_ref.enqueue_fill(Scalar[bf16](12345.0))
    c_pdl.enqueue_fill(Scalar[bf16](12345.0))
    ctx.synchronize()

    # Reference: the routing is already in place when the matmul runs.
    _launch[num_experts, N, K, PDLLevel.ON](
        ctx,
        c_ref,
        a,
        a_sc,
        b,
        b_sc,
        expert_scales,
        src_offsets,
        src_scale_offsets,
        src_ids,
        nae,
        M,
        sf_tiles,
    )
    ctx.synchronize()

    # Programmatic launch overlaps only inside a device graph, so record the
    # producer and the matmul together and replay them.
    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        with builder.recording_context() as rec:
            rec.enqueue_function[publish_after_delay](
                offsets,
                scale_offsets,
                expert_ids,
                src_offsets,
                src_scale_offsets,
                src_ids,
                Int32(nae),
                grid_dim=1,
                block_dim=1,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
            _launch[num_experts, N, K, level](
                rec,
                c_pdl,
                a,
                a_sc,
                b,
                b_sc,
                expert_scales,
                offsets,
                scale_offsets,
                expert_ids,
                nae,
                M,
                sf_tiles,
            )

    var graph = DeviceGraph.create(ctx, build)

    # The reference must have written every element, or it is no oracle.
    var sentinel_bits = bitcast[DType.uint16, 1](Scalar[bf16](12345.0))[0]
    var ref_bits = _bits(ctx, c_ref, out_n)
    var ref_unwritten = 0
    for i in range(out_n):
        if ref_bits[i] == sentinel_bits:
            ref_unwritten += 1
    assert_equal(ref_unwritten, 0, msg="reference run left output unwritten")

    # Check every replay: an intermittent early read can hit any of them.
    for rep in range(5):
        ctx.enqueue_copy(offsets, zeros_u32)
        ctx.enqueue_copy(scale_offsets, zeros_u32)
        ctx.enqueue_copy(expert_ids, zeros_i32)
        c_pdl.enqueue_fill(Scalar[bf16](12345.0))
        ctx.synchronize()
        graph.replay()
        ctx.synchronize()
        var pdl_bits = _bits(ctx, c_pdl, out_n)
        var mismatches = 0
        for i in range(out_n):
            if pdl_bits[i] != ref_bits[i]:
                mismatches += 1
        print("    replay", rep, "mismatches=", mismatches, "/", out_n)
        assert_equal(
            mismatches,
            0,
            msg=(
                "grouped matmul read its routing operands before the producer"
                " kernel published them"
            ),
        )

    offsets_host.unsafe_free()
    scale_offsets_host.unsafe_free()
    ids_host.unsafe_free()
    zeros_u32.unsafe_free()
    zeros_i32.unsafe_free()
    expert_scales_host.unsafe_free()


def main() raises:
    with DeviceContext() as ctx:
        comptime levels = [PDLLevel.ON, PDLLevel.OVERLAP_AT_BEGINNING]
        comptime for lvl in levels:
            print("decode: ragged, prime counts")
            _test_pdl_order[8, 1024, 1024, lvl](
                ctx, [3, 1, 5, 2, 7], [0, 3, 5, 7, 2]
            )
            print("small prefill: 2-CTA tiles")
            _test_pdl_order[6, 1024, 1024, lvl](ctx, [97, 13, 31], [3, 0, 4])
            print("large prefill")
            _test_pdl_order[4, 1024, 2048, lvl](ctx, [257, 131, 61], [2, 1, 3])
