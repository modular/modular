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

"""Test for native FP8 MLA decode kernel where Q, K, V are ALL FP8.

This test verifies the SM100 native FP8 MLA decode kernel (MLA_SM100_Decode_QKV_FP8)
which uses native FP8 WGMMA for both QK and PV matmuls. Unlike the KV-only FP8
kernel, here Q is also stored and loaded as FP8 e4m3fn.

The test:
1. Creates random BF16 Q, quantizes to FP8 -> Q_fp8 (for the kernel)
2. Dequantizes Q_fp8 -> Q_bf16_dequant (for the GPU naive reference)
3. Creates random FP8 K, dequantizes to BF16 K_bf16 (for the reference)
4. Runs the native FP8 kernel via mla_decode_sm100_dispatch
5. Runs GPU naive reference with BF16 dequantized inputs
6. Compares results with tolerances accounting for FP8 quantization
"""

from std.random import randn
from std.sys import (
    argv,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)

from std.gpu import *
from std.gpu.host import DeviceContext
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    lt_to_tt,
    row_major,
)
from nn.attention.gpu.mha import mha_gpu_naive
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.mha_mask import (
    CausalMask,
    NullMask,
    SlidingWindowCausalMask,
)
from nn.attention.mha_operand import LayoutTensorMHAOperand
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from nn.attention.mha_utils import MHAConfig
from std.testing import assert_almost_equal
from std.gpu.host.info import B200, _is_sm10x_gpu
from std.utils.index import Index


# ===-----------------------------------------------------------------------===#
# MLAMaskType
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct MLAMaskType(TrivialRegisterPassable):
    """Enum-like structure for MLA mask types."""

    var value: UInt8

    comptime NO_MASK = Self(0)
    comptime CAUSAL = Self(1)

    def __eq__(self, rhs: Self) -> Bool:
        return self.value == rhs.value

    def __ne__(self, rhs: Self) -> Bool:
        return self.value != rhs.value


@always_inline
def host_cast_fp8_to_bf16[
    fp8_t: DType,
    bf16_t: DType,
](
    src: UnsafePointer[Scalar[fp8_t], _],
    dst: UnsafePointer[mut=True, Scalar[bf16_t], _],
    size: Int,
):
    """Cast FP8 data to BF16 element-by-element on the host."""
    for i in range(size):
        dst[i] = src[i].cast[bf16_t]()


@always_inline
def host_quantize_bf16_to_fp8[
    bf16_t: DType,
    fp8_t: DType,
](
    src: UnsafePointer[Scalar[bf16_t], _],
    dst: UnsafePointer[mut=True, Scalar[fp8_t], _],
    size: Int,
):
    """Quantize BF16 data to FP8 element-by-element on the host."""
    for i in range(size):
        dst[i] = src[i].cast[fp8_t]()


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


def test[
    mla_mask_type: MLAMaskType,
    q_type: DType,  # float8_e4m3fn
    kv_type: DType,  # float8_e4m3fn
    output_type: DType,  # bfloat16
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    batch_size: Int = 1,
](seq_len: Int, num_keys: Int, ctx: DeviceContext,) raises:
    print(
        "test_mla_decode_qkv_fp8",
        "batch_size:",
        batch_size,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
        "q_type:",
        q_type,
        "kv_type:",
        kv_type,
        "output_type:",
        output_type,
        "num_heads:",
        num_heads,
        "mla_mask_type:",
        mla_mask_type.value,
    )

    comptime assert (
        mla_mask_type == MLAMaskType.NO_MASK
        or mla_mask_type == MLAMaskType.CAUSAL
    ), "mha only supports NO_MASK or CAUSAL."

    # Query, key, value dimensions.
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group
    # v_depth is depth - 64 (the rope portion is not in the output)
    comptime v_depth = depth - 64

    # Q, K shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    # Output has v_depth per head (MLA decode only outputs 512, not 576)
    var o_size = batch_size * num_heads * seq_len * v_depth

    # Allocate memory: BF16 reference Q and K, then quantize to FP8.
    var q_bf16_ptr = ctx.enqueue_create_host_buffer[output_type](q_size)
    var q_fp8_ptr = ctx.enqueue_create_host_buffer[q_type](q_size)
    var q_bf16_dequant_ptr = ctx.enqueue_create_host_buffer[output_type](q_size)
    var k_fp8_ptr = ctx.enqueue_create_host_buffer[kv_type](k_size)
    var k_bf16_ptr = ctx.enqueue_create_host_buffer[output_type](k_size)
    var output_ptr = ctx.enqueue_create_host_buffer[output_type](o_size)
    var flash_output_ptr = ctx.enqueue_create_host_buffer[output_type](o_size)

    # Q: create as BF16, quantize to FP8, dequant back for reference
    randn(q_bf16_ptr.as_span())
    host_quantize_bf16_to_fp8[bf16_t=output_type, fp8_t=q_type](
        q_bf16_ptr.unsafe_ptr(), q_fp8_ptr.unsafe_ptr(), q_size
    )
    host_cast_fp8_to_bf16[fp8_t=q_type, bf16_t=output_type](
        q_fp8_ptr.unsafe_ptr(), q_bf16_dequant_ptr.unsafe_ptr(), q_size
    )

    # K: create as FP8, dequant to BF16 for reference
    randn(k_fp8_ptr.as_span())
    host_cast_fp8_to_bf16[fp8_t=kv_type, bf16_t=output_type](
        k_fp8_ptr.unsafe_ptr(), k_bf16_ptr.unsafe_ptr(), k_size
    )

    # ---- Device buffers ----
    # FP8 Q for the kernel
    var q_fp8_device_ptr = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_fp8_device_ptr, q_fp8_ptr)

    # FP8 K for the kernel
    var k_fp8_device_ptr = ctx.enqueue_create_buffer[kv_type](k_size)
    ctx.enqueue_copy(k_fp8_device_ptr, k_fp8_ptr)

    # BF16 dequantized Q for reference
    var q_bf16_dequant_device_ptr = ctx.enqueue_create_buffer[output_type](
        q_size
    )
    ctx.enqueue_copy(q_bf16_dequant_device_ptr, q_bf16_dequant_ptr)

    # BF16 K for reference
    var k_bf16_device_ptr = ctx.enqueue_create_buffer[output_type](k_size)
    ctx.enqueue_copy(k_bf16_device_ptr, k_bf16_ptr)

    # Output for the kernel
    var output_device_ptr = ctx.enqueue_create_buffer[output_type](o_size)

    # Output for reference
    var output_ref_device_ptr = ctx.enqueue_create_buffer[output_type](o_size)

    # ---- Construct TileTensors for kernel inputs ----
    var q_fp8_tt = TileTensor(
        q_fp8_device_ptr,
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
    )
    var k_fp8_tt = TileTensor(
        k_fp8_device_ptr,
        row_major(
            (
                Idx(batch_size),
                Idx(num_keys),
                Idx[kv_num_heads](),
                Idx[depth](),
            )
        ),
    )
    var out_tt = TileTensor(
        output_device_ptr,
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[v_depth]())
        ),
    )

    # LayoutTensors for FP8 K (needed by LayoutTensorMHAOperand)
    comptime k_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, kv_num_heads, depth)
    )

    # BF16 K reference device tensor (for mha_gpu_naive)
    var k_bf16_device = LayoutTensor[output_type, k_layout](
        k_bf16_device_ptr.unsafe_ptr(),
        RuntimeLayout[k_layout].row_major(
            Index(batch_size, num_keys, kv_num_heads, depth)
        ),
    )

    # BF16 dequantized Q device tensor for reference (for mha_gpu_naive)
    comptime q_fp8_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var q_bf16_dequant_device = LayoutTensor[output_type, q_fp8_layout](
        q_bf16_dequant_device_ptr.unsafe_ptr(),
        RuntimeLayout[q_fp8_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    # Output ref layout (for mha_gpu_naive)
    comptime output_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, v_depth)
    )
    var output_ref_device = LayoutTensor[output_type, output_layout](
        output_ref_device_ptr.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            Index(batch_size, seq_len, num_heads, v_depth)
        ),
    )

    # Valid length (empty -- not using ragged) for mha_gpu_naive
    var null_valid_length = LayoutTensor[
        DType.uint32,
        Layout.row_major(UNKNOWN_VALUE),
        MutAnyOrigin,
    ](
        None,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(Index(0)),
    )

    # ---- Launch the native FP8 kernel via mla_decode_sm100_dispatch ----
    print("  Launching native FP8 kernel...")

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        _is_cache_length_accurate=True,
        is_fp8_kv=True,
    ](
        batch_size,
        num_keys,
        seq_len,
        ctx,
    )
    var scalar_args_buf_lt = mla_args.gpu_layout_tensor()

    @parameter
    @always_inline
    @__copy_capture(
        q_fp8_tt,
        k_fp8_tt,
        out_tt,
        scalar_args_buf_lt,
    )
    def kernel_launch(ctx: DeviceContext) raises:
        comptime config = MHAConfig[q_type](num_heads, depth)
        comptime if mla_mask_type == MLAMaskType.CAUSAL:
            flare_mla_decoding[config=config](
                out_tt.as_any_origin(),
                q_fp8_tt,
                k_fp8_tt,
                CausalMask(),
                scale,
                ctx,
                lt_to_tt(scalar_args_buf_lt),
            )
        elif mla_mask_type == MLAMaskType.NO_MASK:
            flare_mla_decoding[config=config](
                out_tt.as_any_origin(),
                q_fp8_tt,
                k_fp8_tt,
                NullMask(),
                scale,
                ctx,
                lt_to_tt(scalar_args_buf_lt),
            )

    kernel_launch(ctx)
    ctx.synchronize()
    print("  Kernel completed.")

    ctx.enqueue_copy(flash_output_ptr, output_device_ptr)

    # ---- GPU naive reference (BF16 dequantized inputs) ----
    # Use BF16 dequantized Q and K for the reference computation.
    # The reference uses the full depth (576), but we only compare the first
    # v_depth (512) elements per head since MLA only outputs V's portion.
    print("  Computing GPU naive reference...")

    # Reference output needs full-depth layout for the naive kernel
    comptime ref_output_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var ref_full_o_size = batch_size * num_heads * seq_len * depth
    var output_ref_full_device_ptr = ctx.enqueue_create_buffer[output_type](
        ref_full_o_size
    )
    var output_ref_full_device = LayoutTensor[output_type, ref_output_layout](
        output_ref_full_device_ptr.unsafe_ptr(),
        RuntimeLayout[ref_output_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    # Create BF16 K operand for reference
    var k_bf16_operand = LayoutTensorMHAOperand(
        LayoutTensor[output_type, k_layout, MutAnyOrigin](
            k_bf16_device.ptr,
            RuntimeLayout[k_layout].row_major(
                k_bf16_device.runtime_layout.shape.value.canonicalize()
            ),
        )
    )

    comptime if mla_mask_type == MLAMaskType.CAUSAL:
        mha_gpu_naive[_is_cache_length_accurate=True](
            q_bf16_dequant_device,
            k_bf16_operand,
            k_bf16_operand,
            CausalMask(),
            output_ref_full_device,
            null_valid_length,
            scale,
            batch_size,
            seq_len,
            num_keys,
            num_heads,
            depth,
            group,
            ctx,
        )
    elif mla_mask_type == MLAMaskType.NO_MASK:
        mha_gpu_naive[_is_cache_length_accurate=True](
            q_bf16_dequant_device,
            k_bf16_operand,
            k_bf16_operand,
            NullMask(),
            output_ref_full_device,
            null_valid_length,
            scale,
            batch_size,
            seq_len,
            num_keys,
            num_heads,
            depth,
            group,
            ctx,
        )

    ctx.synchronize()
    print("  Reference completed.")

    # Copy reference output to host
    var ref_full_output_ptr = ctx.enqueue_create_host_buffer[output_type](
        ref_full_o_size
    )
    ctx.enqueue_copy(ref_full_output_ptr, output_ref_full_device_ptr)
    ctx.synchronize()

    if o_size == 0:
        return

    # ---- Compare results ----
    # The kernel output has shape [B, S, H, v_depth] (512 per head).
    # The reference output has shape [B, S, H, depth] (576 per head).
    # We compare the first v_depth elements of each head.
    # The last 64 elements of each head in the reference are the rope portion,
    # which the MLA decode kernel doesn't output.
    var rtol = 5e-2
    var atol = 3e-1
    var num_mismatches = 0
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(v_depth):
                    # Reference: [b, s, h, d] with stride depth
                    var expect = ref_full_output_ptr[
                        d
                        + depth * (h + s * num_heads)
                        + b * depth * num_heads * seq_len
                    ].cast[DType.float64]()
                    # Kernel output: [b, s, h, d] with stride v_depth
                    var actual = flash_output_ptr[
                        d
                        + v_depth * (h + s * num_heads)
                        + b * v_depth * num_heads * seq_len
                    ].cast[DType.float64]()
                    if abs((actual - expect)) > 1e-1:
                        if num_mismatches < 10:
                            print(b, h, s, d, actual, expect)
                        num_mismatches += 1
                    assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

    if num_mismatches > 0:
        print(
            "  WARNING:",
            num_mismatches,
            "mismatches > 1e-1 (but all passed atol/rtol)",
        )
    print("  PASSED")

    _ = q_fp8_device_ptr
    _ = k_fp8_device_ptr
    _ = q_bf16_dequant_device_ptr
    _ = k_bf16_device_ptr
    _ = output_device_ptr
    _ = output_ref_device_ptr
    _ = output_ref_full_device_ptr


def bench[
    q_type: DType,
    kv_type: DType,
    output_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int,
    batch_size: Int,
](num_keys: Int, ctx: DeviceContext) raises:
    """Benchmark native FP8 MLA decode kernel (no correctness check)."""
    comptime scale = Float32(0.125)
    comptime kv_num_heads = num_heads // group
    comptime v_depth = depth - 64
    comptime seq_len = 1

    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var o_size = batch_size * num_heads * seq_len * v_depth

    # Allocate and fill FP8 Q and K
    var q_bf16_ptr = ctx.enqueue_create_host_buffer[output_type](q_size)
    var q_fp8_ptr = ctx.enqueue_create_host_buffer[q_type](q_size)
    var k_fp8_ptr = ctx.enqueue_create_host_buffer[kv_type](k_size)

    randn(q_bf16_ptr.as_span())
    host_quantize_bf16_to_fp8[bf16_t=output_type, fp8_t=q_type](
        q_bf16_ptr.unsafe_ptr(), q_fp8_ptr.unsafe_ptr(), q_size
    )
    randn(k_fp8_ptr.as_span())

    # Device buffers
    var q_fp8_device_ptr = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_fp8_device_ptr, q_fp8_ptr)

    var k_fp8_device_ptr = ctx.enqueue_create_buffer[kv_type](k_size)
    ctx.enqueue_copy(k_fp8_device_ptr, k_fp8_ptr)

    var output_device_ptr = ctx.enqueue_create_buffer[output_type](o_size)

    ctx.synchronize()

    # TileTensors for kernel inputs
    var q_fp8_tt = TileTensor(
        q_fp8_device_ptr,
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
    )
    var k_fp8_tt = TileTensor(
        k_fp8_device_ptr,
        row_major(
            (
                Idx(batch_size),
                Idx(num_keys),
                Idx[kv_num_heads](),
                Idx[depth](),
            )
        ),
    )
    var out_tt = TileTensor(
        output_device_ptr,
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[v_depth]())
        ),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        _is_cache_length_accurate=True,
        is_fp8_kv=True,
    ](
        batch_size,
        num_keys,
        seq_len,
        ctx,
    )
    var scalar_args_buf_lt = mla_args.gpu_layout_tensor()

    @parameter
    @always_inline
    @__copy_capture(
        q_fp8_tt,
        k_fp8_tt,
        out_tt,
        scalar_args_buf_lt,
    )
    def kernel_launch(ctx: DeviceContext) raises:
        comptime config = MHAConfig[q_type](num_heads, depth)
        flare_mla_decoding[config=config](
            out_tt.as_any_origin(),
            q_fp8_tt,
            k_fp8_tt,
            NullMask(),
            scale,
            ctx,
            lt_to_tt(scalar_args_buf_lt),
        )

    comptime nrun = 200

    # Warmup
    for _ in range(10):
        kernel_launch(ctx)
    ctx.synchronize()

    var nstime = Float64(ctx.execution_time[kernel_launch](nrun)) / Float64(
        nrun
    )
    var ustime = nstime / 1000.0
    print(
        "bench: bs=",
        batch_size,
        " num_keys=",
        num_keys,
        " heads=",
        num_heads,
        " depth=",
        depth,
        " =>",
        ustime,
        "us",
    )

    _ = q_fp8_device_ptr
    _ = k_fp8_device_ptr
    _ = output_device_ptr


def test_decoding[
    batch_size: Int,
    mla_mask_type: MLAMaskType,
](ctx: DeviceContext, seq_len: Int, num_keys: Int) raises:
    test[
        mla_mask_type,
        DType.float8_e4m3fn,  # q_type (FP8)
        DType.float8_e4m3fn,  # kv_type (FP8)
        DType.bfloat16,  # output_type
        576,
        16,
        group=16,
        batch_size=batch_size,
    ](seq_len, num_keys, ctx)

    test[
        mla_mask_type,
        DType.float8_e4m3fn,  # q_type (FP8)
        DType.float8_e4m3fn,  # kv_type (FP8)
        DType.bfloat16,  # output_type
        576,
        64,
        group=64,
        batch_size=batch_size,
    ](seq_len, num_keys, ctx)

    test[
        mla_mask_type,
        DType.float8_e4m3fn,  # q_type (FP8)
        DType.float8_e4m3fn,  # kv_type (FP8)
        DType.bfloat16,  # output_type
        576,
        128,
        group=128,
        batch_size=batch_size,
    ](seq_len, num_keys, ctx)


# fold-triggering configurations.
# The dispatcher enables fold_q when num_q_heads * q_len <= BM(64) and q_len > 1.
# Small head groups (num_heads in {8, 16}) at seq_len in {2, 4, 8} exercise the
# M-dimension fold: BM=64 packs q_len_fold * num_q_heads rows of the same
# batch element, collapsing grid.y to 1.
def test_decoding_fold[
    batch_size: Int,
    num_heads: Int,
    mla_mask_type: MLAMaskType,
](ctx: DeviceContext, seq_len: Int, num_keys: Int) raises:
    test[
        mla_mask_type,
        DType.float8_e4m3fn,  # q_type (FP8)
        DType.float8_e4m3fn,  # kv_type (FP8)
        DType.bfloat16,  # output_type
        576,
        num_heads,
        group=num_heads,
        batch_size=batch_size,
    ](seq_len, num_keys, ctx)


# SlidingWindowCausalMask helper for Layout G coverage.
# Mirrors `test` but launches the kernel and the GPU naive reference with
# a SlidingWindowCausalMask instead of NullMask/CausalMask, since the
# MLAMaskType enum only encodes NO_MASK / CAUSAL.
def test_sw[
    window_size: Int,
    q_type: DType,
    kv_type: DType,
    output_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    batch_size: Int = 1,
](seq_len: Int, num_keys: Int, ctx: DeviceContext,) raises:
    print(
        "test_mla_decode_qkv_fp8 [SlidingWindow]",
        "W:",
        window_size,
        "batch_size:",
        batch_size,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
        "num_heads:",
        num_heads,
    )

    comptime scale = Float32(0.125)
    comptime kv_num_heads = num_heads // group
    comptime v_depth = depth - 64

    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var o_size = batch_size * num_heads * seq_len * v_depth

    var q_bf16_ptr = alloc[Scalar[output_type]](q_size)
    var q_fp8_ptr = alloc[Scalar[q_type]](q_size)
    var q_bf16_dequant_ptr = alloc[Scalar[output_type]](q_size)
    var k_fp8_ptr = alloc[Scalar[kv_type]](k_size)
    var k_bf16_ptr = alloc[Scalar[output_type]](k_size)
    var output_ptr = alloc[Scalar[output_type]](o_size)
    var flash_output_ptr = alloc[Scalar[output_type]](o_size)

    randn[output_type](q_bf16_ptr, q_size)
    host_quantize_bf16_to_fp8[bf16_t=output_type, fp8_t=q_type](
        q_bf16_ptr, q_fp8_ptr, q_size
    )
    host_cast_fp8_to_bf16[fp8_t=q_type, bf16_t=output_type](
        q_fp8_ptr, q_bf16_dequant_ptr, q_size
    )

    randn[kv_type](k_fp8_ptr, k_size)
    host_cast_fp8_to_bf16[fp8_t=kv_type, bf16_t=output_type](
        k_fp8_ptr, k_bf16_ptr, k_size
    )

    var q_fp8_device_ptr = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_fp8_device_ptr, q_fp8_ptr)

    var k_fp8_device_ptr = ctx.enqueue_create_buffer[kv_type](k_size)
    ctx.enqueue_copy(k_fp8_device_ptr, k_fp8_ptr)

    var q_bf16_dequant_device_ptr = ctx.enqueue_create_buffer[output_type](
        q_size
    )
    ctx.enqueue_copy(q_bf16_dequant_device_ptr, q_bf16_dequant_ptr)

    var k_bf16_device_ptr = ctx.enqueue_create_buffer[output_type](k_size)
    ctx.enqueue_copy(k_bf16_device_ptr, k_bf16_ptr)

    var output_device_ptr = ctx.enqueue_create_buffer[output_type](o_size)
    var output_ref_device_ptr = ctx.enqueue_create_buffer[output_type](o_size)

    var q_fp8_tt = TileTensor(
        q_fp8_device_ptr,
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
    )
    var k_fp8_tt = TileTensor(
        k_fp8_device_ptr,
        row_major(
            (
                Idx(batch_size),
                Idx(num_keys),
                Idx[kv_num_heads](),
                Idx[depth](),
            )
        ),
    )
    var out_tt = TileTensor(
        output_device_ptr,
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[v_depth]())
        ),
    )

    comptime k_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, kv_num_heads, depth)
    )
    var k_bf16_device = LayoutTensor[output_type, k_layout](
        k_bf16_device_ptr.unsafe_ptr(),
        RuntimeLayout[k_layout].row_major(
            Index(batch_size, num_keys, kv_num_heads, depth)
        ),
    )

    comptime q_fp8_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var q_bf16_dequant_device = LayoutTensor[output_type, q_fp8_layout](
        q_bf16_dequant_device_ptr.unsafe_ptr(),
        RuntimeLayout[q_fp8_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    comptime output_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, v_depth)
    )
    var output_ref_device = LayoutTensor[output_type, output_layout](
        output_ref_device_ptr.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            Index(batch_size, seq_len, num_heads, v_depth)
        ),
    )

    var null_valid_length = LayoutTensor[
        DType.uint32,
        Layout.row_major(UNKNOWN_VALUE),
        MutAnyOrigin,
    ](
        None,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(Index(0)),
    )

    print("  Launching native FP8 kernel (SlidingWindow)...")

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        _is_cache_length_accurate=True,
        is_fp8_kv=True,
    ](
        batch_size,
        num_keys,
        seq_len,
        ctx,
    )
    var scalar_args_buf_lt = mla_args.gpu_layout_tensor()

    @parameter
    @always_inline
    @__copy_capture(
        q_fp8_tt,
        k_fp8_tt,
        out_tt,
        scalar_args_buf_lt,
    )
    def kernel_launch(ctx: DeviceContext) raises:
        comptime config = MHAConfig[q_type](num_heads, depth)
        flare_mla_decoding[config=config](
            out_tt.as_any_origin(),
            q_fp8_tt,
            k_fp8_tt,
            SlidingWindowCausalMask[window_size](),
            scale,
            ctx,
            lt_to_tt(scalar_args_buf_lt),
        )

    kernel_launch(ctx)
    ctx.synchronize()
    print("  Kernel completed.")

    ctx.enqueue_copy(flash_output_ptr, output_device_ptr)

    print("  Computing GPU naive reference (SlidingWindow)...")

    comptime ref_output_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var ref_full_o_size = batch_size * num_heads * seq_len * depth
    var output_ref_full_device_ptr = ctx.enqueue_create_buffer[output_type](
        ref_full_o_size
    )
    var output_ref_full_device = LayoutTensor[output_type, ref_output_layout](
        output_ref_full_device_ptr.unsafe_ptr(),
        RuntimeLayout[ref_output_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    var k_bf16_operand = LayoutTensorMHAOperand(
        LayoutTensor[output_type, k_layout, MutAnyOrigin](
            k_bf16_device.ptr,
            RuntimeLayout[k_layout].row_major(
                k_bf16_device.runtime_layout.shape.value.canonicalize()
            ),
        )
    )

    mha_gpu_naive[_is_cache_length_accurate=True](
        q_bf16_dequant_device,
        k_bf16_operand,
        k_bf16_operand,
        SlidingWindowCausalMask[window_size](),
        output_ref_full_device,
        null_valid_length,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        group,
        ctx,
    )

    ctx.synchronize()
    print("  Reference completed.")

    var ref_full_output_ptr = alloc[Scalar[output_type]](ref_full_o_size)
    ctx.enqueue_copy(ref_full_output_ptr, output_ref_full_device_ptr)
    ctx.synchronize()

    if o_size == 0:
        return

    var rtol = 5e-2
    var atol = 3e-1
    var num_mismatches = 0
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(v_depth):
                    var expect = ref_full_output_ptr.load(
                        d
                        + depth * (h + s * num_heads)
                        + b * depth * num_heads * seq_len
                    ).cast[DType.float64]()
                    var actual = flash_output_ptr.load(
                        d
                        + v_depth * (h + s * num_heads)
                        + b * v_depth * num_heads * seq_len
                    ).cast[DType.float64]()
                    if abs((actual - expect)) > 1e-1:
                        if num_mismatches < 10:
                            print(b, h, s, d, actual, expect)
                        num_mismatches += 1
                    assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

    if num_mismatches > 0:
        print(
            "  WARNING:",
            num_mismatches,
            "mismatches > 1e-1 (but all passed atol/rtol)",
        )
    print("  PASSED")

    _ = q_fp8_device_ptr
    _ = k_fp8_device_ptr
    _ = q_bf16_dequant_device_ptr
    _ = k_bf16_device_ptr
    _ = output_device_ptr
    _ = output_ref_device_ptr
    _ = output_ref_full_device_ptr

    q_bf16_ptr.free()
    q_fp8_ptr.free()
    q_bf16_dequant_ptr.free()
    k_fp8_ptr.free()
    k_bf16_ptr.free()
    output_ptr.free()
    flash_output_ptr.free()
    ref_full_output_ptr.free()


# SlidingWindowCausalMask helper that wraps `test_sw` with the standard
# FP8 dtypes / depth=576 used elsewhere in this file.
def test_decoding_sw[
    batch_size: Int,
    num_heads: Int,
    window_size: Int,
](ctx: DeviceContext, seq_len: Int, num_keys: Int) raises:
    test_sw[
        window_size,
        DType.float8_e4m3fn,  # q_type (FP8)
        DType.float8_e4m3fn,  # kv_type (FP8)
        DType.bfloat16,  # output_type
        576,
        num_heads,
        group=num_heads,
        batch_size=batch_size,
    ](seq_len, num_keys, ctx)


def main() raises:
    print("Starting test_mla_decode_qkv_fp8...")
    with DeviceContext() as ctx:
        comptime if has_amd_gpu_accelerator() or (
            has_nvidia_gpu_accelerator()
            and _is_sm10x_gpu(ctx.default_device_info)
        ):
            # Basic functionality tests
            print("=== Basic tests ===")
            test_decoding[1, MLAMaskType.NO_MASK](ctx, 1, 256)
            test_decoding[1, MLAMaskType.CAUSAL](ctx, 1, 256)

            # Various cache lengths
            print("=== Cache length tests ===")
            test_decoding[1, MLAMaskType.NO_MASK](ctx, 1, 30)
            test_decoding[1, MLAMaskType.NO_MASK](ctx, 1, 2048)
            test_decoding[1, MLAMaskType.CAUSAL](ctx, 1, 2048)

            # Batch size tests
            print("=== Batch size tests ===")
            test_decoding[2, MLAMaskType.NO_MASK](ctx, 1, 256)
            test_decoding[2, MLAMaskType.CAUSAL](ctx, 1, 1024)

            # Large cache tests (exercises split-K with many tiles per split)
            print("=== Large cache tests ===")
            test_decoding[1, MLAMaskType.NO_MASK](ctx, 1, 9600)
            test_decoding[1, MLAMaskType.NO_MASK](ctx, 1, 32768)
            test_decoding[1, MLAMaskType.NO_MASK](ctx, 1, 65536)

            # fold-path configs.  num_q_heads * q_len <= BM(64) and
            # q_len > 1 triggers fold_q=True in the dispatcher.  BM=64 packs
            # q_len_fold * num_q_heads M-rows into a single tile, so grid.y=1.
            #
            # Fold path is implemented for NVIDIA SM10x only (mla_decode_qkv_fp8
            # struct fold_q comptime branch).  AMD backend does not yet support
            # the fold path — gate this block to NVIDIA SM10x at comptime so AMD
            # skips it cleanly.
            comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
                ctx.default_device_info
            ):
                print(
                    "=== Fold-path tests (num_heads*q_len <= BM, q_len > 1) ==="
                )

                # num_heads=16, q_len=2: 16*2=32 <= 64  (half-pack)
                test_decoding_fold[1, 16, MLAMaskType.NO_MASK](ctx, 2, 256)
                test_decoding_fold[1, 16, MLAMaskType.CAUSAL](ctx, 2, 256)
                test_decoding_fold[2, 16, MLAMaskType.NO_MASK](ctx, 2, 1024)
                test_decoding_fold[4, 16, MLAMaskType.CAUSAL](ctx, 2, 1024)
                # Narrowing probes for A4 bug: vary bs and cl independently.
                print("  probe: bs=1 cl=4096")
                test_decoding_fold[1, 16, MLAMaskType.NO_MASK](ctx, 2, 4096)
                print("  probe: bs=8 cl=256")
                test_decoding_fold[8, 16, MLAMaskType.NO_MASK](ctx, 2, 256)
                print("  probe: bs=8 cl=1024")
                test_decoding_fold[8, 16, MLAMaskType.NO_MASK](ctx, 2, 1024)
                print("  probe: bs=8 cl=2048")
                test_decoding_fold[8, 16, MLAMaskType.NO_MASK](ctx, 2, 2048)
                print("  probe: bs=8 cl=3072")
                test_decoding_fold[8, 16, MLAMaskType.NO_MASK](ctx, 2, 3072)
                print("  probe: bs=8 cl=3328")
                test_decoding_fold[8, 16, MLAMaskType.NO_MASK](ctx, 2, 3328)
                print("  probe: bs=4 cl=4096")
                test_decoding_fold[4, 16, MLAMaskType.NO_MASK](ctx, 2, 4096)
                print("  probe: bs=16 cl=4096")
                test_decoding_fold[16, 16, MLAMaskType.NO_MASK](ctx, 2, 4096)
                test_decoding_fold[8, 16, MLAMaskType.NO_MASK](ctx, 2, 4096)

                # num_heads=16, q_len=4: 16*4=64  (exactly fills BM)
                test_decoding_fold[1, 16, MLAMaskType.NO_MASK](ctx, 4, 256)
                test_decoding_fold[1, 16, MLAMaskType.CAUSAL](ctx, 4, 256)
                test_decoding_fold[2, 16, MLAMaskType.NO_MASK](ctx, 4, 1024)
                test_decoding_fold[4, 16, MLAMaskType.CAUSAL](ctx, 4, 1024)
                test_decoding_fold[8, 16, MLAMaskType.NO_MASK](ctx, 4, 4096)
                test_decoding_fold[16, 16, MLAMaskType.CAUSAL](ctx, 4, 4096)

                # num_heads=8, q_len=4: 8*4=32 <= 64  (half-pack)
                test_decoding_fold[1, 8, MLAMaskType.NO_MASK](ctx, 4, 256)
                test_decoding_fold[2, 8, MLAMaskType.CAUSAL](ctx, 4, 1024)
                test_decoding_fold[8, 8, MLAMaskType.NO_MASK](ctx, 4, 4096)
                test_decoding_fold[8, 8, MLAMaskType.NO_MASK](ctx, 4, 65536)

                # num_heads=8, q_len=8: 8*8=64  (exactly fills BM)
                test_decoding_fold[1, 8, MLAMaskType.NO_MASK](ctx, 8, 256)
                test_decoding_fold[2, 8, MLAMaskType.CAUSAL](ctx, 8, 1024)

                # Odd/corner q_len values (A4 corner-case coverage):
                # num_heads=8, q_len=3: M=24 (40 padding rows)
                test_decoding_fold[8, 8, MLAMaskType.NO_MASK](ctx, 3, 4096)
                # num_heads=8, q_len=5: M=40 (24 padding rows)
                test_decoding_fold[8, 8, MLAMaskType.NO_MASK](ctx, 5, 4096)
                # num_heads=8, q_len=7: M=56 (8 padding rows — near BM edge)
                test_decoding_fold[8, 8, MLAMaskType.CAUSAL](ctx, 7, 4096)
                # num_heads=16, q_len=3: M=48 (16 padding rows)
                test_decoding_fold[8, 16, MLAMaskType.NO_MASK](ctx, 3, 4096)
                test_decoding_fold[4, 16, MLAMaskType.CAUSAL](ctx, 3, 1024)

                # ===-------------------------------------------------=== #
                # A9.2.8 Layout G coverage:
                #   Trigger condition (mla_decode_qkv_fp8_layout_g):
                #     fold_active && num_heads * q_max_seq_len <= 32
                #     && cl >= 1024 && sm10x.
                # ===-------------------------------------------------=== #
                # Block 1 — Layout G triggering configs (fork kernel).
                print("=== Layout G triggers (M<=32, cl>=1024) ===")

                # Production target: Kimi K2.5 TP=8 spec decode.
                #   batch=8 num_heads=8 q_len=4 cl=65536 NO_MASK.
                test_decoding_fold[8, 8, MLAMaskType.NO_MASK](ctx, 4, 65536)

                # num_heads=8, q_len=2 (M=16, eligible, half-fill).
                test_decoding_fold[1, 8, MLAMaskType.NO_MASK](ctx, 2, 1024)
                test_decoding_fold[2, 8, MLAMaskType.NO_MASK](ctx, 2, 4096)
                test_decoding_fold[4, 8, MLAMaskType.NO_MASK](ctx, 2, 16384)
                test_decoding_fold[8, 8, MLAMaskType.NO_MASK](ctx, 2, 65536)

                # num_heads=8, q_len=4 (M=32, exact-fill of M=32 tile).
                test_decoding_fold[1, 8, MLAMaskType.NO_MASK](ctx, 4, 1024)
                test_decoding_fold[2, 8, MLAMaskType.NO_MASK](ctx, 4, 4096)
                test_decoding_fold[4, 8, MLAMaskType.NO_MASK](ctx, 4, 16384)

                # num_heads=16, q_len=2 (M=32, exact-fill of M=32 tile).
                test_decoding_fold[1, 16, MLAMaskType.NO_MASK](ctx, 2, 1024)
                test_decoding_fold[2, 16, MLAMaskType.NO_MASK](ctx, 2, 4096)
                test_decoding_fold[4, 16, MLAMaskType.NO_MASK](ctx, 2, 16384)
                test_decoding_fold[8, 16, MLAMaskType.NO_MASK](ctx, 2, 65536)

                # Layout G + CAUSAL mask coverage.
                test_decoding_fold[2, 8, MLAMaskType.CAUSAL](ctx, 4, 4096)
                test_decoding_fold[4, 16, MLAMaskType.CAUSAL](ctx, 2, 16384)

                # Edge M values reachable with num_heads in {8,16}, q_len <= 4:
                #   num_heads=8 q_len=3 -> M=24 (Layout G eligible).
                #   num_heads=8 q_len=4 -> M=32 (Layout G eligible, exact fit).
                # Other M values (17, 25, 31, 33) are unreachable with these
                # head counts; covered indirectly by the M=24 / M=32 probes.
                test_decoding_fold[4, 8, MLAMaskType.NO_MASK](ctx, 3, 4096)
                test_decoding_fold[8, 8, MLAMaskType.NO_MASK](ctx, 3, 16384)

                # Above-threshold q_len values that should fall back to
                # Layout E (M > 32 routes to non-Layout-G fold kernel).
                # num_heads=8 q_len=5 -> M=40 (Layout E fold path).
                test_decoding_fold[4, 8, MLAMaskType.NO_MASK](ctx, 5, 4096)
                # num_heads=8 q_len=7 -> M=56 (Layout E fold path).
                test_decoding_fold[4, 8, MLAMaskType.CAUSAL](ctx, 7, 4096)

                # Block 2 — q_len=1 regression (NON-fold; Layout E path).
                #   These verify the Layout E non-fold path still works
                #   after the Layout G dispatcher edits.
                print("=== Layout E q_len=1 regression (non-fold) ===")

                # num_heads=8 (group=8), q_len=1, varying batch and cl.
                test_decoding_fold[1, 8, MLAMaskType.NO_MASK](ctx, 1, 1024)
                test_decoding_fold[2, 8, MLAMaskType.NO_MASK](ctx, 1, 4096)
                test_decoding_fold[4, 8, MLAMaskType.NO_MASK](ctx, 1, 16384)
                test_decoding_fold[8, 8, MLAMaskType.NO_MASK](ctx, 1, 65536)
                test_decoding_fold[2, 8, MLAMaskType.CAUSAL](ctx, 1, 4096)

                # num_heads=16 (group=16), q_len=1, varying batch and cl.
                test_decoding_fold[1, 16, MLAMaskType.NO_MASK](ctx, 1, 1024)
                test_decoding_fold[2, 16, MLAMaskType.NO_MASK](ctx, 1, 4096)
                test_decoding_fold[4, 16, MLAMaskType.NO_MASK](ctx, 1, 16384)
                test_decoding_fold[8, 16, MLAMaskType.NO_MASK](ctx, 1, 65536)
                test_decoding_fold[2, 16, MLAMaskType.CAUSAL](ctx, 1, 4096)

                # Block 3 — Below-threshold cl (eligible M, but cl < 1024
                #   -> routes to Layout E even though M <= 32).  Verifies
                #   the cl >= 1024 gate works.
                print("=== Layout G cl<1024 below-threshold (Layout E) ===")

                test_decoding_fold[4, 8, MLAMaskType.NO_MASK](ctx, 4, 64)
                test_decoding_fold[4, 8, MLAMaskType.NO_MASK](ctx, 4, 256)
                test_decoding_fold[4, 16, MLAMaskType.NO_MASK](ctx, 2, 256)

                # ===-------------------------------------------------=== #
                # Relaxed Layout G coverage (post-relaxation):
                #   - q_max_seq_len ∈ {1, 2, 3, 4, 5, 6, 7, 8} eligible
                #     when num_heads * q_max_seq_len <= 32.
                #   - num_heads ≤ 32 eligible (was ≤ 16).
                #   - cl gate removed (any cl admissible).
                #   - SlidingWindowCausalMask admissible.
                #   - fold_q=False with q=1 admissible whenever
                #     num_heads ≤ 32.
                # The blocks below exercise configs that are NEWLY routed
                # to Layout G after the dispatch relaxation.
                # ===-------------------------------------------------=== #

                # Block A — newly-added q values (q ∈ {3, 5, 6, 7}) and
                # newly-relaxed num_heads (up to 32).  num_heads ∈ {4, 5, 6,
                # 10, 24} are not standard MLA dispatch num_heads (the
                # supported set is {8, 16, 32, 64, 128}); for q ∈ {5, 6,
                # 7} we use num_heads=4 (the only value satisfying
                # num_heads * q ≤ 32 outside that set), which is reachable
                # via the parameterized comptime dispatch path.  For
                # num_heads=24 we fall back to num_heads=32 (the nearest
                # supported value at q=1).
                print("=== Block A: relaxed q ∈ {1,3,5,6,7} ===")

                # q=1, num_heads=32 (was capped at 16 pre-relaxation,
                # M=32 exact-fill of BM_G=32 tile, non-fold path).
                test_decoding_fold[4, 32, MLAMaskType.NO_MASK](ctx, 1, 4096)
                # q=3, num_heads=8 → M=24 (Layout G; cl=4096).
                test_decoding_fold[4, 8, MLAMaskType.NO_MASK](ctx, 3, 4096)
                # q=5, num_heads=4 → M=20 (Layout G; only num_heads=4
                # admits q=5 with num_heads*q ≤ 32).
                test_decoding_fold[4, 4, MLAMaskType.NO_MASK](ctx, 5, 4096)
                # q=6, num_heads=4 → M=24 (Layout G).
                test_decoding_fold[4, 4, MLAMaskType.NO_MASK](ctx, 6, 4096)
                # q=7, num_heads=4 → M=28 (Layout G; near BM_G=32 edge).
                test_decoding_fold[4, 4, MLAMaskType.NO_MASK](ctx, 7, 4096)

                # Block B — Short cache_len (cl < 1024).  Pre-relaxation
                # the cl ≥ 1024 gate forced these into Layout E; with the
                # gate removed they now route to Layout G.
                print("=== Block B: relaxed cl < 1024 ===")

                # q=4, num_heads=8 → M=32 with various small cl values.
                test_decoding_fold[4, 8, MLAMaskType.NO_MASK](ctx, 4, 64)
                test_decoding_fold[4, 8, MLAMaskType.NO_MASK](ctx, 4, 256)
                test_decoding_fold[4, 8, MLAMaskType.NO_MASK](ctx, 4, 512)
                # q=2, num_heads=16 → M=32 with very-short cl values.
                test_decoding_fold[4, 16, MLAMaskType.NO_MASK](ctx, 2, 64)
                test_decoding_fold[4, 16, MLAMaskType.NO_MASK](ctx, 2, 128)
                # q=8, num_heads=4 → M=32 with cl=128.
                test_decoding_fold[4, 4, MLAMaskType.NO_MASK](ctx, 8, 128)

                # Block C — SlidingWindowCausalMask (newly admitted to
                # Layout G).  Pre-relaxation Layout G dispatch only
                # admitted NullMask / CausalMask; with mask relaxation
                # SlidingWindowCausalMask now routes to Layout G when
                # M ≤ 32.  window_size=256 is a representative value.
                print("=== Block C: SlidingWindowCausalMask ===")

                # q=4, num_heads=8 → M=32, W=256.
                test_decoding_sw[4, 8, 256](ctx, 4, 4096)
                # q=2, num_heads=16 → M=32, W=256.
                test_decoding_sw[4, 16, 256](ctx, 2, 4096)
                # q=8, num_heads=4 → M=32, W=256 (uses num_heads=4 path).
                test_decoding_sw[4, 4, 256](ctx, 8, 4096)
                # q=1, num_heads=16 → M=16, W=256 (non-fold Layout G).
                test_decoding_sw[4, 16, 256](ctx, 1, 4096)

                # Block D — fold_q=False (q=1) with relaxed num_heads.
                # Pre-relaxation Layout G capped num_heads at 16 in the
                # non-fold path; now num_heads up to 32 is admissible.
                print("=== Block D: q=1 non-fold with high num_heads ===")

                # q=1, num_heads=32 (max relaxed num_heads), batch=8.
                test_decoding_fold[8, 32, MLAMaskType.NO_MASK](ctx, 1, 8192)
                # q=1, num_heads=32 with CAUSAL mask, batch=2.
                test_decoding_fold[2, 32, MLAMaskType.CAUSAL](ctx, 1, 16384)

            print("All tests passed!")

            if is_benchmark():
                print()
                print("=" * 72)
                print(
                    "Native FP8 MLA Decode BENCHMARK (B200) — 128"
                    " heads, depth=576"
                )
                print("=" * 72)
                print()

                # batch_size=1, various cache lengths
                bench[
                    DType.float8_e4m3fn,
                    DType.float8_e4m3fn,
                    DType.bfloat16,
                    576,
                    128,
                    128,
                    1,
                ](4096, ctx)

                bench[
                    DType.float8_e4m3fn,
                    DType.float8_e4m3fn,
                    DType.bfloat16,
                    576,
                    128,
                    128,
                    1,
                ](32768, ctx)

                bench[
                    DType.float8_e4m3fn,
                    DType.float8_e4m3fn,
                    DType.bfloat16,
                    576,
                    128,
                    128,
                    1,
                ](65536, ctx)

                bench[
                    DType.float8_e4m3fn,
                    DType.float8_e4m3fn,
                    DType.bfloat16,
                    576,
                    128,
                    128,
                    1,
                ](131072, ctx)

                bench[
                    DType.float8_e4m3fn,
                    DType.float8_e4m3fn,
                    DType.bfloat16,
                    576,
                    128,
                    128,
                    1,
                ](163840, ctx)

                # batch_size=8
                bench[
                    DType.float8_e4m3fn,
                    DType.float8_e4m3fn,
                    DType.bfloat16,
                    576,
                    128,
                    128,
                    8,
                ](32768, ctx)

                bench[
                    DType.float8_e4m3fn,
                    DType.float8_e4m3fn,
                    DType.bfloat16,
                    576,
                    128,
                    128,
                    8,
                ](65536, ctx)

                print()
                print("=" * 72)
                print("BENCHMARK COMPLETE")
                print("=" * 72)
        else:
            print("Skipping: requires B200 or AMD GPU")
