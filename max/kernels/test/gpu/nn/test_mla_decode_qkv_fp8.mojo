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
from std.sys import argv, has_nvidia_gpu_accelerator

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
from nn.attention.mha_mask import CausalMask, NullMask
from nn.attention.mha_operand import LayoutTensorMHAOperand
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
    mla_decode_sm100_dispatch,
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
    var q_bf16_ptr = alloc[Scalar[output_type]](q_size)
    var q_fp8_ptr = alloc[Scalar[q_type]](q_size)
    var q_bf16_dequant_ptr = alloc[Scalar[output_type]](q_size)
    var k_fp8_ptr = alloc[Scalar[kv_type]](k_size)
    var k_bf16_ptr = alloc[Scalar[output_type]](k_size)
    var output_ptr = alloc[Scalar[output_type]](o_size)
    var flash_output_ptr = alloc[Scalar[output_type]](o_size)

    # Q: create as BF16, quantize to FP8, dequant back for reference
    randn[output_type](q_bf16_ptr, q_size)
    host_quantize_bf16_to_fp8[bf16_t=output_type, fp8_t=q_type](
        q_bf16_ptr, q_fp8_ptr, q_size
    )
    host_cast_fp8_to_bf16[fp8_t=q_type, bf16_t=output_type](
        q_fp8_ptr, q_bf16_dequant_ptr, q_size
    )

    # K: create as FP8, dequant to BF16 for reference
    randn[kv_type](k_fp8_ptr, k_size)
    host_cast_fp8_to_bf16[fp8_t=kv_type, bf16_t=output_type](
        k_fp8_ptr, k_bf16_ptr, k_size
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
        q_fp8_device_ptr.unsafe_ptr(),
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
    )
    var out_tt = TileTensor(
        output_device_ptr.unsafe_ptr(),
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[v_depth]())
        ),
    )
    var null_valid_length_tt = TileTensor(
        UnsafePointer[UInt32, MutAnyOrigin](),
        row_major(Idx(0)),
    )

    # LayoutTensors for FP8 K (needed by LayoutTensorMHAOperand)
    comptime k_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, kv_num_heads, depth)
    )
    var k_fp8_device = LayoutTensor[kv_type, k_layout](
        k_fp8_device_ptr.unsafe_ptr(),
        RuntimeLayout[k_layout].row_major(
            Index(batch_size, num_keys, kv_num_heads, depth)
        ),
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
        DType.uint32, Layout.row_major(UNKNOWN_VALUE)
    ](
        UnsafePointer[UInt32, MutAnyOrigin](),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(Index(0)),
    )

    # Create the KV operand (LayoutTensorMHAOperand)
    var k_operand = LayoutTensorMHAOperand(
        LayoutTensor[kv_type, k_layout, MutAnyOrigin](
            k_fp8_device.ptr,
            RuntimeLayout[k_layout].row_major(
                k_fp8_device.runtime_layout.shape.value.canonicalize()
            ),
        )
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
        k_operand,
        out_tt,
        null_valid_length_tt,
        scalar_args_buf_lt,
    )
    def kernel_launch(ctx: DeviceContext) raises:
        comptime config = MHAConfig[q_type](UInt(num_heads), UInt(depth))
        comptime if mla_mask_type == MLAMaskType.CAUSAL:
            mla_decode_sm100_dispatch[
                q_type,
                type_of(k_operand),
                output_type,
                CausalMask,
                config,
                depth,
                num_heads,
                group,
                _is_cache_length_accurate=True,
                decoding_warp_split_k=False,
            ](
                q_fp8_tt,
                k_operand,
                out_tt,
                scale,
                null_valid_length_tt,
                CausalMask(),
                lt_to_tt(scalar_args_buf_lt),
                batch_size,
                seq_len,
                num_keys,
                ctx,
            )
        elif mla_mask_type == MLAMaskType.NO_MASK:
            mla_decode_sm100_dispatch[
                q_type,
                type_of(k_operand),
                output_type,
                NullMask,
                config,
                depth,
                num_heads,
                group,
                _is_cache_length_accurate=True,
                decoding_warp_split_k=False,
            ](
                q_fp8_tt,
                k_operand,
                out_tt,
                scale,
                null_valid_length_tt,
                NullMask(),
                lt_to_tt(scalar_args_buf_lt),
                batch_size,
                seq_len,
                num_keys,
                ctx,
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
    var ref_full_output_ptr = alloc[Scalar[output_type]](ref_full_o_size)
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
                    var expect = ref_full_output_ptr.load(
                        d
                        + depth * (h + s * num_heads)
                        + b * depth * num_heads * seq_len
                    ).cast[DType.float64]()
                    # Kernel output: [b, s, h, d] with stride v_depth
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
    var q_bf16_ptr = alloc[Scalar[output_type]](q_size)
    var q_fp8_ptr = alloc[Scalar[q_type]](q_size)
    var k_fp8_ptr = alloc[Scalar[kv_type]](k_size)

    randn[output_type](q_bf16_ptr, q_size)
    host_quantize_bf16_to_fp8[bf16_t=output_type, fp8_t=q_type](
        q_bf16_ptr, q_fp8_ptr, q_size
    )
    randn[kv_type](k_fp8_ptr, k_size)

    # Device buffers
    var q_fp8_device_ptr = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_fp8_device_ptr, q_fp8_ptr)

    var k_fp8_device_ptr = ctx.enqueue_create_buffer[kv_type](k_size)
    ctx.enqueue_copy(k_fp8_device_ptr, k_fp8_ptr)

    var output_device_ptr = ctx.enqueue_create_buffer[output_type](o_size)

    ctx.synchronize()

    # TileTensors for kernel inputs
    var q_fp8_tt = TileTensor(
        q_fp8_device_ptr.unsafe_ptr(),
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
    )
    var out_tt = TileTensor(
        output_device_ptr.unsafe_ptr(),
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[v_depth]())
        ),
    )
    var null_valid_length_tt = TileTensor(
        UnsafePointer[UInt32, MutAnyOrigin](),
        row_major(Idx(0)),
    )

    # LayoutTensor for FP8 K (needed by LayoutTensorMHAOperand)
    comptime k_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, kv_num_heads, depth)
    )
    var k_fp8_device = LayoutTensor[kv_type, k_layout](
        k_fp8_device_ptr.unsafe_ptr(),
        RuntimeLayout[k_layout].row_major(
            Index(batch_size, num_keys, kv_num_heads, depth)
        ),
    )

    var k_operand = LayoutTensorMHAOperand(
        LayoutTensor[kv_type, k_layout, MutAnyOrigin](
            k_fp8_device.ptr,
            RuntimeLayout[k_layout].row_major(
                k_fp8_device.runtime_layout.shape.value.canonicalize()
            ),
        )
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
        k_operand,
        out_tt,
        null_valid_length_tt,
        scalar_args_buf_lt,
    )
    def kernel_launch(ctx: DeviceContext) raises:
        comptime config = MHAConfig[q_type](UInt(num_heads), UInt(depth))
        mla_decode_sm100_dispatch[
            q_type,
            type_of(k_operand),
            output_type,
            NullMask,
            config=config,
            depth=depth,
            num_heads=num_heads,
            group=group,
            _is_cache_length_accurate=True,
            decoding_warp_split_k=False,
        ](
            q_fp8_tt,
            k_operand,
            out_tt,
            scale,
            null_valid_length_tt,
            NullMask(),
            lt_to_tt(scalar_args_buf_lt),
            batch_size,
            seq_len,
            num_keys,
            ctx,
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

    # Cleanup
    q_bf16_ptr.free()
    q_fp8_ptr.free()
    k_fp8_ptr.free()

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


def main() raises:
    print("Starting test_mla_decode_qkv_fp8...")
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
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
            print("Skipping: requires B200 GPU")
