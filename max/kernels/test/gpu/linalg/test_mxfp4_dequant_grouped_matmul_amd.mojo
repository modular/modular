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
"""Tests for MXFP4 dequant-then-FP8 grouped matmul on AMD CDNA GPUs.

Validates mxfp4_dequant_grouped_matmul_amd against a reference path that:
  1. Dequants MXFP4 weights to FP8 per-expert
  2. Casts BF16 activations to FP8 per-expert
  3. Runs vendor BLAS per-expert

This layered approach isolates errors to the grouped dispatch logic since
the underlying dequant and FP8 GEMM are validated by test_mxfp4_dequant_matmul_amd.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.random import random_float64, random_ui64
from std.gpu.host import DeviceContext
import linalg.matmul.vendor.blas as vendor_blas
from layout import Coord, Idx, Layout, LayoutTensor, TileTensor, row_major
from linalg.mxfp4_dequant import dequant_mxfp4
from linalg.matmul.gpu.amd.mxfp4_dequant_matmul_amd import _cast_bf16_to_fp8
from linalg.matmul.gpu.amd.mxfp4_dequant_grouped_matmul_amd import (
    mxfp4_dequant_grouped_matmul_amd,
)


def test_mxfp4_grouped_matmul[
    num_experts: Int, N: Int, K: Int
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
) raises:
    """Test mxfp4_dequant_grouped_matmul_amd against per-expert vendor BLAS reference.
    """
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)
    comptime fp8_type = DType.float8_e4m3fn

    # Compute total tokens and max tokens per expert
    var total_tokens = 0
    var max_tokens = 0
    for i in range(len(num_tokens_by_expert)):
        total_tokens += num_tokens_by_expert[i]
        max_tokens = max(max_tokens, num_tokens_by_expert[i])

    print(
        "  grouped matmul: num_experts=",
        num_experts,
        " N=",
        N,
        " K=",
        K,
        " num_active=",
        num_active_experts,
        " total_tokens=",
        total_tokens,
    )

    # Allocate host buffers
    var a_host = ctx.enqueue_create_host_buffer[DType.bfloat16](
        total_tokens * K
    )
    var b_packed_host = ctx.enqueue_create_host_buffer[DType.uint8](
        num_experts * N * packed_K
    )
    var b_scales_host = ctx.enqueue_create_host_buffer[DType.uint8](
        num_experts * N * scale_K
    )
    var a_offsets_host = ctx.enqueue_create_host_buffer[DType.uint32](
        num_active_experts + 1
    )
    var expert_ids_host = ctx.enqueue_create_host_buffer[DType.int32](
        num_active_experts
    )

    # Initialize with random data
    for i in range(total_tokens * K):
        a_host[i] = random_float64(-0.5, 0.5).cast[DType.bfloat16]()
    for i in range(num_experts * N * packed_K):
        b_packed_host[i] = UInt8(random_ui64(0, 255))
    for i in range(num_experts * N * scale_K):
        b_scales_host[i] = UInt8(random_ui64(125, 129))

    # Setup offsets and expert ids
    a_offsets_host[0] = UInt32(0)
    for i in range(num_active_experts):
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host[i] = Int32(expert_ids_list[i])

    # Compute reference: per-expert dequant + vendor BLAS
    var c_ref_host = ctx.enqueue_create_host_buffer[DType.bfloat16](
        total_tokens * N
    )

    for i in range(num_active_experts):
        var token_start = Int(a_offsets_host[i])
        var token_end = Int(a_offsets_host[i + 1])
        var num_tokens = token_end - token_start
        var expert_id = Int(expert_ids_host[i])

        if num_tokens <= 0 or expert_id < 0:
            for j in range(num_tokens * N):
                c_ref_host[token_start * N + j] = Scalar[DType.bfloat16](0.0)
            continue

        # Copy this expert's packed weights and scales to device
        var bp_dev = ctx.enqueue_create_buffer[DType.uint8](N * packed_K)
        var bs_dev = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](
            N * scale_K
        )

        var bp_hbuf = ctx.enqueue_create_host_buffer[DType.uint8](N * packed_K)
        var bs_hbuf = ctx.enqueue_create_host_buffer[DType.float8_e8m0fnu](
            N * scale_K
        )
        for j in range(N * packed_K):
            bp_hbuf[j] = b_packed_host[expert_id * N * packed_K + j]
        for j in range(N * scale_K):
            bs_hbuf[j] = rebind[Scalar[DType.float8_e8m0fnu]](
                b_scales_host[expert_id * N * scale_K + j]
            )
        ctx.enqueue_copy(bp_dev, bp_hbuf)
        ctx.enqueue_copy(bs_dev, bs_hbuf)

        # Dequant this expert's weights to FP8
        var b_fp8_dev = ctx.enqueue_create_buffer[fp8_type](N * K)
        var bp_tt = TileTensor(bp_dev, row_major[N, packed_K]())
        var bs_tt = TileTensor(bs_dev, row_major[N, scale_K]())
        var b_fp8_tt = TileTensor(b_fp8_dev, row_major((Idx[N](), Idx[K]())))

        dequant_mxfp4(ctx, b_fp8_tt, bp_tt, bs_tt, num_rows=N, num_cols=K)

        # Cast this expert's activations to FP8 and copy to device
        var a_dev = ctx.enqueue_create_buffer[DType.bfloat16](num_tokens * K)
        var a_hbuf = ctx.enqueue_create_host_buffer[DType.bfloat16](
            num_tokens * K
        )
        for j in range(num_tokens * K):
            a_hbuf[j] = a_host[token_start * K + j]
        ctx.enqueue_copy(a_dev, a_hbuf)

        var a_fp8_dev = ctx.enqueue_create_buffer[fp8_type](num_tokens * K)
        var a_tt = TileTensor(a_dev, row_major((Idx(num_tokens), Idx[K]())))
        var a_fp8_tt = TileTensor(
            a_fp8_dev, row_major((Idx(num_tokens), Idx[K]()))
        )
        _cast_bf16_to_fp8(ctx, a_fp8_tt, a_tt, num_tokens, K)
        ctx.synchronize()

        # Run vendor BLAS on FP8 data
        var c_dev = ctx.enqueue_create_buffer[DType.bfloat16](num_tokens * N)
        var c_tt = TileTensor(c_dev, row_major((Idx(num_tokens), Idx[N]())))
        vendor_blas.matmul(
            ctx,
            c_tt,
            a_fp8_tt,
            b_fp8_tt,
            c_row_major=True,
            transpose_b=True,
        )
        ctx.synchronize()

        # Copy result back to host
        var c_hbuf = ctx.enqueue_create_host_buffer[DType.bfloat16](
            num_tokens * N
        )
        ctx.enqueue_copy(c_hbuf, c_dev)
        ctx.synchronize()

        for j in range(num_tokens * N):
            c_ref_host[token_start * N + j] = c_hbuf[j]

        _ = bp_dev^
        _ = bs_dev^
        _ = b_fp8_dev^
        _ = a_dev^
        _ = a_fp8_dev^
        _ = c_dev^

    # Run kernel under test
    var a_dev = ctx.enqueue_create_buffer[DType.bfloat16](total_tokens * K)
    var b_packed_dev = ctx.enqueue_create_buffer[DType.uint8](
        num_experts * N * packed_K
    )
    var b_scales_dev = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    var a_offsets_dev = ctx.enqueue_create_buffer[DType.uint32](
        num_active_experts + 1
    )
    var expert_ids_dev = ctx.enqueue_create_buffer[DType.int32](
        num_active_experts
    )
    var c_dev = ctx.enqueue_create_buffer[DType.bfloat16](total_tokens * N)

    ctx.enqueue_copy(a_dev, a_host)

    var bp_hbuf = ctx.enqueue_create_host_buffer[DType.uint8](
        num_experts * N * packed_K
    )
    for j in range(num_experts * N * packed_K):
        bp_hbuf[j] = b_packed_host[j]
    ctx.enqueue_copy(b_packed_dev, bp_hbuf)

    var bs_hbuf = ctx.enqueue_create_host_buffer[DType.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    for j in range(num_experts * N * scale_K):
        bs_hbuf[j] = rebind[Scalar[DType.float8_e8m0fnu]](b_scales_host[j])
    ctx.enqueue_copy(b_scales_dev, bs_hbuf)

    ctx.enqueue_copy(a_offsets_dev, a_offsets_host)
    ctx.enqueue_copy(expert_ids_dev, expert_ids_host)
    ctx.synchronize()

    # Build TileTensors
    var a_tt = TileTensor(a_dev, row_major((Idx(total_tokens), Idx[K]())))
    var b_packed_tt = TileTensor(
        b_packed_dev, row_major[num_experts, N, packed_K]()
    )
    var b_scales_tt = TileTensor(
        b_scales_dev, row_major[num_experts, N, scale_K]()
    )
    var a_offsets_tt = TileTensor(
        a_offsets_dev,
        row_major(Coord(Idx(num_active_experts + 1))),
    )
    var expert_ids_tt = TileTensor(
        expert_ids_dev,
        row_major(Coord(Idx(num_active_experts))),
    )
    var c_tt = TileTensor(c_dev, row_major((Idx(total_tokens), Idx[N]())))

    mxfp4_dequant_grouped_matmul_amd(
        c_tt,
        a_tt,
        b_packed_tt,
        b_scales_tt,
        a_offsets_tt,
        expert_ids_tt,
        max_tokens,
        num_active_experts,
        total_tokens,
        ctx,
    )
    ctx.synchronize()

    # Compare results
    var c_host = ctx.enqueue_create_host_buffer[DType.bfloat16](
        total_tokens * N
    )
    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    var max_rel_err = Float32(0.0)
    var num_mismatches = 0

    for i in range(total_tokens * N):
        var got = c_host[i].cast[DType.float32]()
        var expected = c_ref_host[i].cast[DType.float32]()
        var magnitude = max(abs(got), abs(expected))
        if magnitude < Float32(1.0):
            continue
        var rel_err = abs(got - expected) / magnitude
        max_rel_err = max(max_rel_err, rel_err)
        if rel_err > Float32(0.02):
            if num_mismatches < 5:
                var row, col = divmod(i, N)
                print(
                    "    MISMATCH [",
                    row,
                    ",",
                    col,
                    "]: got=",
                    got,
                    " expected=",
                    expected,
                    " rel_err=",
                    rel_err,
                )
            num_mismatches += 1

    if num_mismatches > 0:
        print(
            "    FAIL:",
            num_mismatches,
            "mismatches, max_rel_err=",
            max_rel_err,
        )
        raise Error("MXFP4 grouped matmul test failed")

    print("    PASS max_rel_err=", max_rel_err)

    # Cleanup
    _ = a_dev^
    _ = b_packed_dev^
    _ = b_scales_dev^
    _ = a_offsets_dev^
    _ = expert_ids_dev^
    _ = c_dev^


def test_dequant_all_experts[
    num_experts: Int, N: Int, K: Int
](ctx: DeviceContext) raises:
    """Verify that dequant across all experts is consistent with per-expert."""
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)
    comptime fp8_type = DType.float8_e4m3fn

    print(
        "  dequant all experts: num_experts=", num_experts, " N=", N, " K=", K
    )

    # Allocate and fill random packed data
    var bp_host = ctx.enqueue_create_host_buffer[DType.uint8](
        num_experts * N * packed_K
    )
    var bs_host = ctx.enqueue_create_host_buffer[DType.uint8](
        num_experts * N * scale_K
    )
    for i in range(num_experts * N * packed_K):
        bp_host[i] = UInt8(random_ui64(0, 255))
    for i in range(num_experts * N * scale_K):
        bs_host[i] = UInt8(random_ui64(125, 129))

    # Copy to device
    var bp_dev = ctx.enqueue_create_buffer[DType.uint8](
        num_experts * N * packed_K
    )
    var bs_dev = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](
        num_experts * N * scale_K
    )

    var bp_hbuf = ctx.enqueue_create_host_buffer[DType.uint8](
        num_experts * N * packed_K
    )
    var bs_hbuf = ctx.enqueue_create_host_buffer[DType.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    for i in range(num_experts * N * packed_K):
        bp_hbuf[i] = bp_host[i]
    for i in range(num_experts * N * scale_K):
        bs_hbuf[i] = rebind[Scalar[DType.float8_e8m0fnu]](bs_host[i])
    ctx.enqueue_copy(bp_dev, bp_hbuf)
    ctx.enqueue_copy(bs_dev, bs_hbuf)
    ctx.synchronize()

    # Dequant all experts via loop (as mxfp4_dequant_grouped_matmul_amd does)
    var all_fp8_dev = ctx.enqueue_create_buffer[fp8_type](num_experts * N * K)
    for e in range(num_experts):
        var bp_expert = TileTensor(
            bp_dev.unsafe_ptr() + e * N * packed_K,
            row_major[N, packed_K](),
        )
        var bs_expert = TileTensor(
            bs_dev.unsafe_ptr() + e * N * scale_K,
            row_major[N, scale_K](),
        )
        var fp8_expert = TileTensor(
            all_fp8_dev.unsafe_ptr() + e * N * K,
            row_major((Idx[N](), Idx[K]())),
        )
        dequant_mxfp4(
            ctx, fp8_expert, bp_expert, bs_expert, num_rows=N, num_cols=K
        )

    # Dequant single expert independently for comparison
    var single_fp8_dev = ctx.enqueue_create_buffer[fp8_type](N * K)
    var bp_expert0 = TileTensor(bp_dev, row_major[N, packed_K]())
    var bs_expert0 = TileTensor(bs_dev, row_major[N, scale_K]())
    var fp8_expert0 = TileTensor(
        single_fp8_dev, row_major((Idx[N](), Idx[K]()))
    )
    dequant_mxfp4(
        ctx, fp8_expert0, bp_expert0, bs_expert0, num_rows=N, num_cols=K
    )
    ctx.synchronize()

    # Compare expert 0 from batched vs single
    var all_host = ctx.enqueue_create_host_buffer[fp8_type](num_experts * N * K)
    var single_host = ctx.enqueue_create_host_buffer[fp8_type](N * K)
    ctx.enqueue_copy(all_host, all_fp8_dev)
    ctx.enqueue_copy(single_host, single_fp8_dev)
    ctx.synchronize()

    var mismatches = 0
    for i in range(N * K):
        var batched = all_host[i]
        var single = single_host[i]
        if rebind[UInt8](batched) != rebind[UInt8](single):
            if mismatches < 5:
                print("    DEQUANT MISMATCH at", i)
            mismatches += 1

    if mismatches > 0:
        print("    FAIL:", mismatches, "byte mismatches")
        raise Error("Multi-expert dequant test failed")
    print("    PASS (byte-identical)")

    _ = bp_dev^
    _ = bs_dev^
    _ = all_fp8_dev^
    _ = single_fp8_dev^


def main() raises:
    with DeviceContext() as ctx:
        print("MXFP4 Grouped Matmul Tests")
        print("==========================")

        # Test 1: Multi-expert dequant consistency
        print("-- Multi-expert dequant consistency --")
        test_dequant_all_experts[4, 256, 256](ctx)
        test_dequant_all_experts[2, 2048, 2048](ctx)

        # Test 2: Single expert (degenerates to regular matmul)
        print("-- Single expert (degenerate case) --")
        test_mxfp4_grouped_matmul[1, 256, 256](1, [64], [0], ctx)
        test_mxfp4_grouped_matmul[1, 2048, 2048](1, [128], [0], ctx)

        # Test 3: Multiple experts, simple routing
        print("-- Multiple experts, simple routing --")
        test_mxfp4_grouped_matmul[4, 256, 256](2, [32, 64], [0, 2], ctx)
        test_mxfp4_grouped_matmul[4, 256, 256](3, [32, 32, 32], [0, 1, 3], ctx)

        # Test 4: Larger dimensions
        print("-- Larger dimensions --")
        test_mxfp4_grouped_matmul[4, 2048, 2048](2, [64, 128], [1, 3], ctx)

        # Test 5: Unequal token counts
        print("-- Unequal token counts --")
        test_mxfp4_grouped_matmul[4, 256, 256](2, [16, 96], [0, 2], ctx)

        # Test 6: Many experts (Kimi K2.5 like)
        print("-- Many experts (Kimi K2.5 like) --")
        test_mxfp4_grouped_matmul[8, 4096, 7168](
            8, [96, 96, 96, 96, 96, 96, 96, 96], [0, 1, 2, 3, 4, 5, 6, 7], ctx
        )

        print("==========================")
        print("ALL TESTS PASSED")
