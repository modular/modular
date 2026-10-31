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
"""Numerical-equivalence test for the rowwise / per-token grouped FP8 matmul.

Target: NVIDIA SM100 (B200). Validates
``grouped_matmul_rowwise_dynamic_scaled_fp8`` (the kernel under test, KUT)
against the in-tree naive grouped FP8 reference
``naive_blockwise_scaled_fp8_grouped_matmul`` configured with
``scales_granularity_mnk = (1, 1, K)`` - i.e. one activation scale per token
and one weight scale per (expert, output-channel). The two paths are distinct
code: the reference reloads + multiplies the (constant-over-K) scales inside
the K-loop, while the KUT applies them once after the full-K fp32 reduction.

Note the scale-tensor *layout* convention differs between the two, which is
the point of the cross-check:
  * reference ``a_scales``: ``[K // K = 1, total_tokens]`` (K-block x token)
  * KUT ``a_scales``:       ``[total_tokens, 1]``           (token x 1)
  * both ``b_scales``:      ``[num_experts, N, 1]``          ((expert,n) x 1)

Covered cases: single expert, partial last M-tile, a zero-token expert in a
multi-expert ragged batch, sparse expert-id routing, and a production-ish
Llama-4-Scout gate/up shape.
"""

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from layout import Idx, TileTensor, row_major, Coord
from layout._fillers import random
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_grouped_matmul
from linalg.grouped_matmul import grouped_matmul_rowwise_dynamic_scaled_fp8
from std.math import ceildiv
from std.testing import assert_almost_equal, assert_equal
from std.time import sleep

from std.utils.index import Index, IndexList


def test_grouped_matmul_rowwise_scaled_fp8[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],  # (N, K)
    pdl_level: PDLLevel = PDLLevel(),
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
) raises:
    comptime transpose_b = True
    comptime scales_type = DType.float32

    comptime a_type = in_type
    comptime b_type = in_type
    comptime c_type = out_type

    comptime N = expert_shape[0]
    comptime K = expert_shape[1]

    var total_num_tokens = 0
    var max_num_tokens_by_expert = 0
    for i in range(len(num_tokens_by_expert)):
        var M = num_tokens_by_expert[i]
        total_num_tokens += M
        max_num_tokens_by_expert = max(max_num_tokens_by_expert, M)

    print(
        t"== test_grouped_matmul_rowwise_scaled_fp8 {a_type} ->"
        t" {c_type} problem shape: ( {total_num_tokens} x {N} x {K} )"
        t" num_active_experts: {num_active_experts}"
    )

    # Sizes.
    var a_size = total_num_tokens * K
    var b_size = num_experts * N * K
    var c_size = total_num_tokens * N
    # KUT layout: per-token, [T, 1].
    var a_scales_kut_size = total_num_tokens
    # Reference layout: [1, T] (K-block count is 1).
    var a_scales_ref_size = total_num_tokens
    # Both b_scales layouts are [E, N, 1].
    var b_scales_size = num_experts * N

    # KUT TileTensor shapes.
    var a_tt_shape = row_major(total_num_tokens, Idx[K])
    var b_tt_shape = row_major(Idx[num_experts], Idx[N], Idx[K])
    var c_tt_shape = row_major(total_num_tokens, Idx[N])
    var a_scales_kut_shape = row_major(total_num_tokens, Idx[1])
    var a_scales_ref_shape = row_major(Idx[1], total_num_tokens)
    var b_scales_shape = row_major(Idx[num_experts], Idx[N], Idx[1])

    # Host allocations.
    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_active_experts
    )
    var a_scales_kut_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        a_scales_kut_size
    )
    var a_scales_ref_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        a_scales_ref_size
    )
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        b_scales_size
    )

    var a_host = TileTensor(a_host_ptr, a_tt_shape)
    var b_host = TileTensor(b_host_ptr, b_tt_shape)
    var c_host = TileTensor(c_host_ptr, c_tt_shape)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_tt_shape)

    # Offsets and expert ids.
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host_ptr[i] = Int32(expert_ids_list[i])

    random(a_host)
    random(b_host)
    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    # Fill per-token activation scales once, then mirror into both layouts.
    # KUT a_scales is [T, 1]; reference a_scales is [1, T]. Same values.
    var a_scales_kut_host = TileTensor(
        a_scales_kut_host_ptr, a_scales_kut_shape
    )
    random(a_scales_kut_host)
    for t in range(total_num_tokens):
        a_scales_ref_host_ptr[t] = a_scales_kut_host_ptr[t]

    # b_scales is [E, N, 1] for both paths.
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)
    random(b_scales_host)

    # Device allocations.
    var a_device_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var a_offsets_device_buffer = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_device_buffer = ctx.enqueue_create_buffer[.int32](
        num_active_experts
    )
    var a_scales_kut_device_buffer = ctx.enqueue_create_buffer[scales_type](
        a_scales_kut_size
    )
    var a_scales_ref_device_buffer = ctx.enqueue_create_buffer[scales_type](
        a_scales_ref_size
    )
    var b_scales_device_buffer = ctx.enqueue_create_buffer[scales_type](
        b_scales_size
    )

    var a_device_tt = TileTensor(a_device_buffer, a_tt_shape)
    var b_device_tt = TileTensor(b_device_buffer, b_tt_shape)
    var c_device_tt = TileTensor(c_device_buffer, c_tt_shape)
    var c_device_ref_tt = TileTensor(c_device_ref_buffer, c_tt_shape)
    var a_offsets_device_tt = TileTensor(
        a_offsets_device_buffer,
        row_major(num_active_experts + 1),
    )
    var expert_ids_device_tt = TileTensor(
        expert_ids_device_buffer,
        row_major(num_active_experts),
    )
    var a_scales_kut_device_tt = TileTensor(
        a_scales_kut_device_buffer, a_scales_kut_shape
    )
    var a_scales_ref_device_tt = TileTensor(
        a_scales_ref_device_buffer, a_scales_ref_shape
    )
    var b_scales_device_tt = TileTensor(b_scales_device_buffer, b_scales_shape)

    ctx.enqueue_copy(a_device_buffer, a_host_ptr)
    ctx.enqueue_copy(b_device_buffer, b_host_ptr)
    ctx.enqueue_copy(c_device_buffer, c_host_ptr)
    ctx.enqueue_copy(c_device_ref_buffer, c_host_ref_ptr)
    ctx.enqueue_copy(a_offsets_device_buffer, a_offsets_host_ptr)
    ctx.enqueue_copy(expert_ids_device_buffer, expert_ids_host_ptr)
    ctx.enqueue_copy(a_scales_kut_device_buffer, a_scales_kut_host_ptr)
    ctx.enqueue_copy(a_scales_ref_device_buffer, a_scales_ref_host_ptr)
    ctx.enqueue_copy(b_scales_device_buffer, b_scales_host_ptr)

    # Reference path - distinct code from the KUT.
    naive_blockwise_scaled_fp8_grouped_matmul[
        BLOCK_DIM_M=16,
        BLOCK_DIM_N=16,
        transpose_b=transpose_b,
        scales_granularity_mnk=Coord[Int, Int, Int](1, 1, K),
    ](
        c_device_ref_tt,
        a_device_tt,
        b_device_tt,
        a_scales_ref_device_tt,
        b_scales_device_tt,
        a_offsets_device_tt,
        expert_ids_device_tt,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    ctx.synchronize()

    # Kernel under test (TileTensor).
    grouped_matmul_rowwise_dynamic_scaled_fp8[
        transpose_b=transpose_b,
        target="gpu",
        pdl_level=pdl_level,
    ](
        c_device_tt,
        a_device_tt,
        b_device_tt,
        a_scales_kut_device_tt,
        b_scales_device_tt,
        a_offsets_device_tt,
        expert_ids_device_tt,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device_buffer)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref_buffer)
    ctx.synchronize()

    var rtol = 1e-2
    var atol = 1e-2
    for mi in range(total_num_tokens):
        for ni in range(N):
            assert_almost_equal(
                c_host_ptr[mi * N + ni],
                c_host_ref_ptr[mi * N + ni],
                msg=String(t"m: {mi} n: {ni}"),
                rtol=rtol,
                atol=atol,
            )


def late_producer(
    a_dst: MutPointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin],
    a_src: MutPointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin],
    s_dst: MutPointer[Scalar[DType.float32], MutAnyOrigin],
    s_src: MutPointer[Scalar[DType.float32], MutAnyOrigin],
    o_dst: MutPointer[Scalar[DType.uint32], MutAnyOrigin],
    o_src: MutPointer[Scalar[DType.uint32], MutAnyOrigin],
    e_dst: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    e_src: MutPointer[Scalar[DType.int32], MutAnyOrigin],
    a_len: Int32,
    s_len: Int32,
    groups: Int32,
):
    """Releases its dependent at entry, sleeps, then publishes the operands.

    The activations, their scales, the ragged offsets and the expert ids are
    all produced upstream in the MoE path. A dependent that reads any of them
    before its own PDL wait sees the stale fill: zero offsets (every group
    empty) and expert id -1 (every group inactive) give an all-zero output.
    """
    launch_dependent_grids()
    # The host builds the dependent's launch (TMA descriptors, cluster setup)
    # while this grid runs, so the delay has to outlast that enqueue time.
    sleep(0.005)
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, Int(a_len), stride):
        a_dst[i] = a_src[i]
    for i in range(tid, Int(s_len), stride):
        s_dst[i] = s_src[i]
    for i in range(tid, Int(groups) + 1, stride):
        o_dst[i] = o_src[i]
    for i in range(tid, Int(groups), stride):
        e_dst[i] = e_src[i]


def copy_consumer(
    src: MutPointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dst: MutPointer[Scalar[DType.bfloat16], MutAnyOrigin],
    length: Int32,
):
    """Waits for the matmul grid, then copies its output."""
    wait_on_dependent_grids()
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, Int(length), stride):
        dst[i] = src[i]


def test_pdl_chain[
    num_experts: Int,
    expert_shape: IndexList[2],  # (N, K)
    pdl_level: PDLLevel,
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
) raises:
    """Late-publishing producer -> PDL matmul -> PDL consumer, no host sync.

    The matmul output must equal the same matmul run without PDL on the
    published operands, bit for bit.
    """
    comptime N = expert_shape[0]
    comptime K = expert_shape[1]

    var total = 0
    var max_tokens = 0
    for i in range(len(num_tokens_by_expert)):
        total += num_tokens_by_expert[i]
        max_tokens = max(max_tokens, num_tokens_by_expert[i])

    print(
        "== test_pdl_chain N:",
        N,
        "K:",
        K,
        "tokens:",
        total,
        "pdl_level:",
        pdl_level._level,
    )

    var a_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](total * K)
    var b_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        num_experts * N * K
    )
    var as_host = ctx.enqueue_create_host_buffer[.float32](total)
    var bs_host = ctx.enqueue_create_host_buffer[.float32](num_experts * N)
    var offs_host = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts + 1
    )
    var ids_host = ctx.enqueue_create_host_buffer[.int32](num_active_experts)
    var c_host = ctx.enqueue_create_host_buffer[.bfloat16](total * N)
    var ref_host = ctx.enqueue_create_host_buffer[.bfloat16](total * N)

    random(TileTensor(a_host, row_major(total, Idx[K])))
    random(TileTensor(b_host, row_major(Idx[num_experts], Idx[N], Idx[K])))
    random(TileTensor(as_host, row_major(total, Idx[1])))
    random(TileTensor(bs_host, row_major(Idx[num_experts], Idx[N], Idx[1])))
    offs_host[0] = 0
    for i in range(num_active_experts):
        offs_host[i + 1] = offs_host[i] + UInt32(num_tokens_by_expert[i])
        ids_host[i] = Int32(expert_ids_list[i])

    var a_src = ctx.enqueue_create_buffer[.float8_e4m3fn](total * K)
    var a_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](total * K)
    var b_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](num_experts * N * K)
    var as_src = ctx.enqueue_create_buffer[.float32](total)
    var as_dev = ctx.enqueue_create_buffer[.float32](total)
    var bs_dev = ctx.enqueue_create_buffer[.float32](num_experts * N)
    var offs_dev = ctx.enqueue_create_buffer[.uint32](num_active_experts + 1)
    var ids_dev = ctx.enqueue_create_buffer[.int32](num_active_experts)
    var offs_chain = ctx.enqueue_create_buffer[.uint32](num_active_experts + 1)
    var ids_chain = ctx.enqueue_create_buffer[.int32](num_active_experts)
    var c_dev = ctx.enqueue_create_buffer[.bfloat16](total * N)
    var out_dev = ctx.enqueue_create_buffer[.bfloat16](total * N)
    var ref_dev = ctx.enqueue_create_buffer[.bfloat16](total * N)

    ctx.enqueue_copy(a_src, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(as_src, as_host)
    ctx.enqueue_copy(bs_dev, bs_host)
    ctx.enqueue_copy(offs_dev, offs_host)
    ctx.enqueue_copy(ids_dev, ids_host)

    var a_src_tt = TileTensor(a_src, row_major(total, Idx[K]))
    var a_tt = TileTensor(a_dev, row_major(total, Idx[K]))
    var b_tt = TileTensor(b_dev, row_major(Idx[num_experts], Idx[N], Idx[K]))
    var as_src_tt = TileTensor(as_src, row_major(total, Idx[1]))
    var as_tt = TileTensor(as_dev, row_major(total, Idx[1]))
    var bs_tt = TileTensor(bs_dev, row_major(Idx[num_experts], Idx[N], Idx[1]))
    var offs_tt = TileTensor(offs_dev, row_major(num_active_experts + 1))
    var ids_tt = TileTensor(ids_dev, row_major(num_active_experts))
    var offs_chain_tt = TileTensor(
        offs_chain, row_major(num_active_experts + 1)
    )
    var ids_chain_tt = TileTensor(ids_chain, row_major(num_active_experts))
    var c_tt = TileTensor(c_dev, row_major(total, Idx[N]))
    var ref_tt = TileTensor(ref_dev, row_major(total, Idx[N]))

    # Reference: the same op without PDL on the published operands.
    grouped_matmul_rowwise_dynamic_scaled_fp8[transpose_b=True, target="gpu"](
        ref_tt,
        a_src_tt,
        b_tt,
        as_src_tt,
        bs_tt,
        offs_tt,
        ids_tt,
        max_tokens,
        num_active_experts,
        ctx,
    )
    ctx.synchronize()

    # The chain is recorded once and replayed, so the host does no work between
    # the launches (as in a captured production graph) and the producer's delay
    # only has to cover the device-side launch of its dependents.
    def build(
        mut builder: DeviceGraphBuilder,
    ) raises {
        imm a_dev,
        imm a_src,
        imm as_dev,
        imm as_src,
        imm c_dev,
        imm out_dev,
        imm a_tt,
        imm b_tt,
        imm as_tt,
        imm bs_tt,
        imm offs_dev,
        imm ids_dev,
        imm offs_chain,
        imm ids_chain,
        imm offs_chain_tt,
        imm ids_chain_tt,
        imm c_tt,
        imm total,
        imm max_tokens,
        imm num_active_experts,
    }:
        with builder.recording_context() as rec:
            rec.enqueue_function[late_producer](
                a_dev,
                a_src,
                as_dev,
                as_src,
                offs_chain,
                offs_dev,
                ids_chain,
                ids_dev,
                Int32(total * K),
                Int32(total),
                Int32(num_active_experts),
                grid_dim=ceildiv(total * K, 256),
                block_dim=256,
            )
            grouped_matmul_rowwise_dynamic_scaled_fp8[
                transpose_b=True, target="gpu", pdl_level=pdl_level
            ](
                c_tt,
                a_tt,
                b_tt,
                as_tt,
                bs_tt,
                offs_chain_tt,
                ids_chain_tt,
                max_tokens,
                num_active_experts,
                rec,
            )
            rec.enqueue_function[copy_consumer](
                c_dev,
                out_dev,
                Int32(total * N),
                grid_dim=ceildiv(total * N, 256),
                block_dim=256,
                attributes=pdl_launch_attributes(pdl_level),
            )

    var graph = DeviceGraph.create(ctx, build)

    # Stale operands, routing tensors and outputs are staged before the replay.
    ctx.enqueue_memset[.float8_e4m3fn](a_dev, 0)
    ctx.enqueue_memset[.float32](as_dev, 0)
    ctx.enqueue_memset[.uint32](offs_chain, 0)
    ctx.enqueue_memset[.int32](ids_chain, Int32(-1))
    ctx.enqueue_memset[.bfloat16](c_dev, 0)
    ctx.enqueue_memset[.bfloat16](out_dev, 0)
    graph.replay()
    ctx.synchronize()

    ctx.enqueue_copy(c_host, out_dev)
    ctx.enqueue_copy(ref_host, ref_dev)
    ctx.synchronize()
    for i in range(total * N):
        assert_equal(c_host[i], ref_host[i], msg=String(t"element {i}"))


def main() raises:
    with DeviceContext() as ctx:
        # Single expert, M aligned.
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=1,
            expert_shape=(256, 256),
        ](1, [128], [0], ctx)

        # Single expert, partial last M-tile (100 mod 16 != 0).
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=1,
            expert_shape=(256, 256),
        ](1, [100], [0], ctx)

        # Sparse routing: a single active expert that is NOT id 0.
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=4,
            expert_shape=(512, 1024),
        ](1, [256], [2], ctx)

        # Multi-expert ragged, sparse ids.
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=4,
            expert_shape=(512, 1024),
        ](2, [20, 40], [0, 2], ctx)

        # Zero-token expert in the middle of a ragged batch (M == 0 group).
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=6,
            expert_shape=(512, 1024),
        ](4, [20, 0, 300, 28], [0, 3, 2, 4], ctx)

        # Several small + medium experts, sparse ids.
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=6,
            expert_shape=(7168, 2048),
        ](4, [20, 1500, 300, 28], [0, 3, 2, 4], ctx)

        # fp32 output.
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .float32,
            num_experts=4,
            expert_shape=(512, 1024),
        ](2, [20, 40], [0, 2], ctx)

        # Llama-4-Scout gate/up shape (N x K = 8192 x 5120) with unaligned
        # per-expert token counts. ``num_experts`` is kept at 2 here so the B
        # weight allocation fits the bazel test sandbox's per-chunk GPU memory
        # cap; the full 16-expert Scout shape with sparse ids [4, 9] is
        # validated by a direct ``mojo run`` on an unconstrained B200, and
        # sparse-id-against-many-experts routing is covered by the 7168x2048
        # ``num_experts=6`` case above.
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=2,
            expert_shape=(8192, 5120),
        ](2, [13, 51], [0, 1], ctx)

        # PDL on: the wait at kernel entry and both trigger placements, on a
        # ragged multi-expert batch with a zero-token group and a partial tile.
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=6,
            expert_shape=(512, 1024),
            pdl_level=PDLLevel.ON,
        ](4, [20, 0, 300, 28], [0, 3, 2, 4], ctx)
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=6,
            expert_shape=(512, 1024),
            pdl_level=PDLLevel.OVERLAP_AT_BEGINNING,
        ](4, [20, 0, 300, 28], [0, 3, 2, 4], ctx)
        # N = 1000 is a ragged last N tile on the tensor-core path.
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=4,
            expert_shape=(1000, 1024),
            pdl_level=PDLLevel.ON,
        ](2, [17, 41], [1, 3], ctx)
        # N = 998 breaks the 16-byte TMA row stride, so this takes the naive
        # CUDA-core fallback with PDL on.
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=4,
            expert_shape=(998, 1024),
            pdl_level=PDLLevel.ON,
        ](2, [17, 41], [1, 3], ctx)
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=4,
            expert_shape=(998, 1024),
            pdl_level=PDLLevel.OVERLAP_AT_BEGINNING,
        ](2, [17, 41], [1, 3], ctx)

        # Late-publishing producer -> op -> consumer, on the tensor-core path
        # (N=512) and on the naive fallback (N=998, off the TMA stride).
        test_pdl_chain[4, (512, 1024), PDLLevel.ON](2, [20, 300], [0, 3], ctx)
        test_pdl_chain[4, (512, 1024), PDLLevel.OVERLAP_AT_BEGINNING](
            4, [20, 0, 300, 28], [0, 3, 2, 1], ctx
        )
        test_pdl_chain[4, (998, 1024), PDLLevel.ON](2, [17, 41], [1, 3], ctx)
        test_pdl_chain[4, (998, 1024), PDLLevel.OVERLAP_AT_BEGINNING](
            2, [17, 41], [1, 3], ctx
        )

        print("All rowwise grouped FP8 tests passed.")
