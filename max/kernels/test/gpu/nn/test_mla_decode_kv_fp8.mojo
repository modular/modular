# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from collections import Optional
from math import ceildiv, isclose
from random import randn
from sys import argv, has_nvidia_gpu_accelerator

from buffer import Dim, DimList, NDBuffer
from gpu import *
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from nn.mha import _naive_attention_with_transpose, mha_gpu_naive
from nn.mha_mask import CausalMask, MaterializedMask, NullMask
from nn.mha_operand import LayoutTensorMHAOperand
from nn.mha_score_mod import IdentityScoreMod
from nn.mla import flare_mla_decoding, flare_mla_prefill
from tensor import IOUnknown, ManagedTensorSlice
from tensor.managed_tensor_slice import StaticTensorSpec
from testing import assert_almost_equal
from gpu.host.info import B200
from utils.index import Index
from utils.numerics import get_accum_type


# ===-----------------------------------------------------------------------===#
# MLAMaskType
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct MLAMaskType(TrivialRegisterPassable):
    """Enum-like structure for MLA mask types."""

    var value: UInt8

    comptime NO_MASK = Self(0)
    comptime CAUSAL = Self(1)
    comptime MASK_3D = Self(2)
    comptime MASK_4D = Self(3)

    fn __eq__(self, rhs: Self) -> Bool:
        return self.value == rhs.value

    fn __ne__(self, rhs: Self) -> Bool:
        return self.value != rhs.value


@always_inline
fn host_cast_k_fp8_to_bf16[
    kv_fp8_t: DType,
    k_bf16_t: DType,
](
    k_fp8: UnsafePointer[Scalar[kv_fp8_t]],
    k_bf16: UnsafePointer[Scalar[k_bf16_t]],
    depth: Int,
    num_keys: Int,
    kv_num_heads: Int,
    batch_size: Int,
):
    # Layout of k in your test:
    # k_ptr[(i * kv_num_heads + h) * depth + j]
    for b in range(batch_size):
        var b_off = b * num_keys * kv_num_heads * depth
        for i in range(num_keys):
            for h in range(kv_num_heads):
                var base = b_off + (i * kv_num_heads + h) * depth
                for j in range(depth):
                    k_bf16[base + j] = k_fp8[base + j].cast[k_bf16_t]()


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


fn test[
    mla_mask_type: MLAMaskType,
    q_type: DType,
    kv_type: DType,
    mask_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    against_gpu_naive: Bool = False,
    batch_size: Int = 1,
    num_partitions: Optional[Int] = None,
    decoding_warp_split_k: Bool = False,
](
    seq_len: Int,
    num_keys: Int,
    ctx: DeviceContext,
    use_index_input: Bool = False,
) raises:
    print(
        "test_mla_decoding",
        "batch_size:",
        batch_size,
        "num_partitions:",
        num_partitions.value() if num_partitions else -1,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
        "q_type:",
        q_type,
        "kv_type:",
        kv_type,
        "mask_type:",
        mask_type,
        "mla_mask_type:",
        mla_mask_type.value,
    )

    comptime assert (
        mla_mask_type == MLAMaskType.NO_MASK
        or mla_mask_type == MLAMaskType.CAUSAL
        or mla_mask_type == MLAMaskType.MASK_3D
        or mla_mask_type == MLAMaskType.MASK_4D
    ), "mha only supports NO_MASK, CAUSAL, MASK_3D, or MASK_4D."
    comptime assert (
        against_gpu_naive or mla_mask_type == MLAMaskType.MASK_3D
    ), "Testing against cpu requires mask of MASK_3D."

    # Query, key, value dimensions.
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    # var v_size = k_size
    var o_size = q_size
    var mask_size = (
        (num_heads if mla_mask_type == MLAMaskType.MASK_4D else 1)
        * seq_len
        * num_keys
        * batch_size
    )

    # Allocate memory for all variables.
    var q_ptr = UnsafePointer[Scalar[q_type]].alloc(q_size)
    var k_ptr = UnsafePointer[Scalar[kv_type]].alloc(k_size)  # fp8 host
    var k_bf16_ptr = UnsafePointer[Scalar[q_type]].alloc(k_size)
    var mask_ptr = UnsafePointer[Scalar[mask_type]].alloc(mask_size)
    var output_ptr = UnsafePointer[Scalar[q_type]].alloc(o_size)
    var flash_output_ptr = UnsafePointer[Scalar[q_type]].alloc(o_size)

    # Q, K, V are randomly initialized.
    randn[q_type](q_ptr, q_size)
    randn[kv_type](k_ptr, k_size)
    randn[mask_type](mask_ptr, mask_size)

    host_cast_k_fp8_to_bf16[kv_fp8_t=kv_type, k_bf16_t=q_type](
        k_ptr, k_bf16_ptr, depth, num_keys, kv_num_heads, batch_size
    )

    # Construct buffers.
    comptime layout_4d = Layout.row_major[4]()
    var q = LayoutTensor[q_type, layout_4d](
        q_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )
    var k = LayoutTensor[kv_type, layout_4d](
        k_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_keys, kv_num_heads, depth)
        ),
    )
    var k_bf16 = LayoutTensor[q_type, layout_4d](
        k_bf16_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_keys, kv_num_heads, depth)
        ),
    )
    var mask = LayoutTensor[mask_type, Layout.row_major[2]()](
        mask_ptr,
        RuntimeLayout[Layout.row_major[2]()].row_major(
            Index(seq_len, num_keys)
        ),
    )
    var output = LayoutTensor[q_type, layout_4d](
        output_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    var flash_output = LayoutTensor[q_type, layout_4d](
        flash_output_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    @parameter
    if not against_gpu_naive:
        comptime assert (
            q_type == mask_type
        ), "expect qkv and mask have same type for CPU."
        _naive_attention_with_transpose[q_type](
            output, q, k_bf16, k_bf16, mask.bitcast[q_type](), scale
        )

    # Device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[q_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[kv_type](k_size)
    var mask_device_ptr = ctx.enqueue_create_buffer[mask_type](mask_size)
    var output_device_ptr = ctx.enqueue_create_buffer[q_type](o_size)

    # Copy from host to device
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(mask_device_ptr, mask_ptr)

    # Construct layout tensor buffers.
    comptime q_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var q_device = LayoutTensor[q_type, q_layout](
        q_device_ptr.unsafe_ptr(),
        RuntimeLayout[q_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )
    comptime k_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, kv_num_heads, depth)
    )
    var k_device = LayoutTensor[kv_type, k_layout](
        k_device_ptr.unsafe_ptr(),
        RuntimeLayout[k_layout].row_major(
            Index(batch_size, num_keys, kv_num_heads, depth)
        ),
    )
    var mask3d = LayoutTensor[mask_type, Layout.row_major[3]()](
        mask_device_ptr.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[3]()].row_major(
            Index(batch_size, seq_len, num_keys)
        ),
    )
    var mask4d = LayoutTensor[mask_type, Layout.row_major[4]()](
        mask_device_ptr.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[4]()].row_major(
            Index(batch_size, num_heads, seq_len, num_keys)
        ),
    )
    comptime output_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var output_device = LayoutTensor[q_type, output_layout](
        output_device_ptr.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    comptime q_tile_num_rows = 32
    comptime k_tile_num_rows = 128

    @parameter
    @always_inline
    @__copy_capture(q_device, k_device, mask3d, mask4d, output_device)
    fn kernel_launch(ctx: DeviceContext) raises:
        @parameter
        if mla_mask_type == MLAMaskType.CAUSAL:
            flare_mla_decoding[decoding_warp_split_k=decoding_warp_split_k](
                output_device.as_any_origin(),
                q_device,
                k_device,
                CausalMask(),
                IdentityScoreMod(),
                scale,
                ctx,
                num_partitions,
            )
        elif mla_mask_type == MLAMaskType.MASK_3D:
            flare_mla_decoding[decoding_warp_split_k=decoding_warp_split_k](
                output_device.as_any_origin(),
                q_device,
                k_device,
                MaterializedMask(mask3d),
                IdentityScoreMod(),
                scale,
                ctx,
                num_partitions,
            )
        elif mla_mask_type == MLAMaskType.MASK_4D:
            flare_mla_decoding[decoding_warp_split_k=decoding_warp_split_k](
                output_device.as_any_origin(),
                q_device,
                k_device,
                MaterializedMask(mask4d),
                IdentityScoreMod(),
                scale,
                ctx,
                num_partitions,
            )
        elif mla_mask_type == MLAMaskType.NO_MASK:
            flare_mla_decoding[decoding_warp_split_k=decoding_warp_split_k](
                output_device.as_any_origin(),
                q_device,
                k_device,
                NullMask(),
                IdentityScoreMod(),
                scale,
                ctx,
                num_partitions,
            )

    if is_benchmark():
        comptime nrun = 200

        # Warmup
        kernel_launch(ctx)

        var nstime = Float64(ctx.execution_time[kernel_launch](nrun)) / Float64(
            nrun
        )
        var sectime = nstime / 1000000
        print(nrun, "runs avg", sectime, "ms")

    else:
        kernel_launch(ctx)

    ctx.synchronize()

    ctx.enqueue_copy(flash_output_ptr, output_device_ptr)

    @parameter
    if against_gpu_naive:
        var output_ref_device_ptr = ctx.enqueue_create_buffer[q_type](o_size)
        comptime output_ref_layout = Layout.row_major(
            Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
        )
        var output_ref_device = LayoutTensor[q_type, output_ref_layout](
            output_ref_device_ptr.unsafe_ptr(),
            RuntimeLayout[output_ref_layout].row_major(
                Index(batch_size, seq_len, num_heads, depth)
            ),
        )
        ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

        var k_ref_device_ptr = ctx.enqueue_create_buffer[q_type](k_size)

        var k_ref_device = LayoutTensor[q_type, k_layout](
            k_ref_device_ptr.unsafe_ptr(),
            RuntimeLayout[k_layout].row_major(
                Index(batch_size, num_keys, kv_num_heads, depth)
            ),
        )
        ctx.enqueue_copy(k_ref_device_ptr, k_bf16_ptr)

        @parameter
        if mla_mask_type == MLAMaskType.CAUSAL:
            var k_operand = LayoutTensorMHAOperand(k_ref_device)
            var null_valid_length = LayoutTensor[
                DType.uint32, Layout.row_major(UNKNOWN_VALUE)
            ](
                UnsafePointer[UInt32](),
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    Index(0)
                ),
            )
            mha_gpu_naive[_is_cache_length_accurate=True,](
                q_device,
                k_operand,
                k_operand,
                CausalMask(),
                output_ref_device,
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
        elif mla_mask_type == MLAMaskType.MASK_3D:
            mha_gpu_naive(
                q_device,
                k_device,
                k_device,
                mask3d,
                output_ref_device,
                scale,
                batch_size,
                seq_len,
                num_keys,
                num_heads,
                depth,
                group,
                ctx,
            )
        elif mla_mask_type == MLAMaskType.MASK_4D:
            mha_gpu_naive(
                q_device,
                k_device,
                k_device,
                mask4d,
                output_ref_device,
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
            mha_gpu_naive(
                q_device,
                k_device,
                k_device,
                NullMask(),
                output_ref_device,
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
        ctx.enqueue_copy(output_ptr, output_ref_device_ptr)
        _ = output_ref_device_ptr

    if o_size == 0:
        return

    # since we pass the whole K tensor as the V tensor to our naive mha kernel,
    # the last 64 elements of each head in the reference result are invalid.
    # b , s, h, d
    var rtol = 5e-2  # 0.05
    var atol = 3e-1  # 0.3
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(depth - 64):
                    var expect = output_ptr.load(
                        d
                        + depth * (h + s * num_heads)
                        + b * depth * num_heads * seq_len
                    ).cast[DType.float64]()
                    var actual = flash_output_ptr.load(
                        d
                        + (depth - 64) * (h + s * num_heads)
                        + b * (depth - 64) * num_heads * seq_len
                    ).cast[DType.float64]()
                    # if not isclose(actual, expect, atol=1e-3, rtol=rtol):
                    #     var rerr = abs((actual - expect) / expect)
                    #     print(h, s, d, actual, expect, rerr)
                    if abs((actual - expect)) > 1e-1:
                        print(b, h, s, d, actual, expect)
                    assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

    _ = q_device_ptr
    _ = k_device_ptr
    _ = mask_device_ptr
    _ = output_device_ptr

    q_ptr.free()
    k_ptr.free()
    mask_ptr.free()
    output_ptr.free()
    flash_output_ptr.free()


fn test_decoding[
    batch_size: Int,
    mla_mask_type: MLAMaskType,
    split_k: Bool = False,
    num_partitions: Optional[Int] = 1,
    against_gpu_naive: Bool = True,
](
    ctx: DeviceContext, use_index_input: Bool, seq_len: Int, num_keys: Int
) raises:
    test[
        mla_mask_type,
        DType.bfloat16,  # q_type
        DType.float8_e4m3fn,  # kv_type  (fp8 KV)
        DType.float32,  # mask_type
        576,
        16,
        group=16,
        against_gpu_naive=against_gpu_naive,
        batch_size=batch_size,
        num_partitions=num_partitions,
        decoding_warp_split_k=split_k,
    ](seq_len, num_keys, ctx, use_index_input=use_index_input)

    test[
        mla_mask_type,
        DType.bfloat16,  # q_type
        DType.float8_e4m3fn,  # kv_type  (fp8 KV)
        DType.float32,  # mask_type
        576,
        64,
        group=64,
        against_gpu_naive=against_gpu_naive,
        batch_size=batch_size,
        num_partitions=num_partitions,
        decoding_warp_split_k=split_k,
    ](seq_len, num_keys, ctx, use_index_input=use_index_input)

    test[
        mla_mask_type,
        DType.bfloat16,  # q_type
        DType.float8_e4m3fn,  # kv_type  (fp8 KV)
        DType.float32,  # mask_type
        576,
        128,
        group=128,
        against_gpu_naive=against_gpu_naive,
        batch_size=batch_size,
        num_partitions=num_partitions,
        decoding_warp_split_k=split_k,
    ](seq_len, num_keys, ctx, use_index_input=use_index_input)


def main():
    with DeviceContext() as ctx:

        @parameter
        if has_nvidia_gpu_accelerator() and ctx.default_device_info == B200:
            # tests with mask tensor
            # Test with benchmark parameters: batch_size=1, cache_len=32768, num_heads=128
            test_decoding[1, MLAMaskType.NO_MASK](ctx, False, 1, 32768)
            test_decoding[1, MLAMaskType.CAUSAL](ctx, False, 1, 32768)
            test_decoding[128, MLAMaskType.NO_MASK](ctx, False, 1, 1024)
            test_decoding[128, MLAMaskType.CAUSAL](ctx, False, 2, 1024)
            test_decoding[64, MLAMaskType.NO_MASK](ctx, False, 1, 2048)
            test_decoding[64, MLAMaskType.CAUSAL](ctx, False, 2, 2048)
            test_decoding[64, MLAMaskType.CAUSAL](ctx, False, 3, 50)
            test_decoding[64, MLAMaskType.NO_MASK](ctx, False, 4, 193)
            test_decoding[27, MLAMaskType.MASK_3D](ctx, False, 5, 50)
            test_decoding[1, MLAMaskType.NO_MASK](ctx, False, 1, 32768 * 2)
            test_decoding[64, MLAMaskType.MASK_4D](ctx, False, 6, 517)
        else:
            pass
