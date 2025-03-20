# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# UNSUPPORTED: AMD-GPU
# RUN: %mojo-no-debug %s

from math import ceildiv, isclose, isqrt
from random import randn, rand
from sys import argv

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import *
from gpu.host import DeviceContext
from gpu.host.info import DEFAULT_GPU_ARCH
from internal_utils import assert_with_measure
from internal_utils._measure import cosine
from memory import UnsafePointer
from nn.mha import (
    _naive_attention_with_transpose,
    mha_gpu_naive,
)
from nn.mla import flare_mla_decoding
from nn.mha_mask import NullMask, CausalMask
from nn.mha_operand import NDBufferMHAOperand
from nn.mha_score_mod import IdentityScoreMod

from utils.index import Index
from testing import assert_almost_equal


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


fn test[
    mask_rank: Int,
    qkv_type: DType,
    mask_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    against_gpu_naive: Bool = False,
    batch_size: Int = 1,
    num_partitions: OptionalReg[Int] = None,
    decoding_warp_split_k: Bool = False,
    use_causal_mask: Bool = True,
](
    seq_len: Int,
    num_keys: Int,
    ctx: DeviceContext,
    use_index_input: Bool = False,
) raises:
    print(
        "test_flash_attention",
        "batch_size:",
        batch_size,
        "num_partitions:",
        num_partitions.value() if num_partitions else -1,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
        "qkv_type:",
        qkv_type,
        "mask_type:",
        mask_type,
        "mask_rank:",
        mask_rank,
    )

    constrained[mask_rank in (3, 4), "mha only support rank 3 or 4."]()
    constrained[
        against_gpu_naive or mask_rank == 3,
        "Testing against cpu requires mask of rank 3.",
    ]()

    # Query, key, value dimensions.
    alias scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    alias kv_num_heads = num_heads // group

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    # var v_size = k_size
    var o_size = q_size
    var mask_size = (
        num_heads if mask_rank == 4 else 1
    ) * seq_len * num_keys * batch_size

    # Allocate memory for all variables.
    var q_ptr = UnsafePointer[Scalar[qkv_type]].alloc(q_size)
    var k_ptr = UnsafePointer[Scalar[qkv_type]].alloc(k_size)
    var mask_ptr = UnsafePointer[Scalar[mask_type]].alloc(mask_size)
    var output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)
    var flash_output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)

    # Q, K, V are randomly initalized.
    if use_index_input:
        debug_assert(batch_size == 1)
        for i in range(seq_len):
            for h in range(num_heads):
                for j in range(depth):
                    q_ptr[(i * num_heads + h) * depth + j] = i * depth + j
        for i in range(num_keys):
            for h in range(kv_num_heads):
                for j in range(depth):
                    k_ptr[(i * kv_num_heads + h) * depth + j] = i * depth + j

        @parameter
        if mask_rank == 3:
            for i in range(seq_len):
                for j in range(num_keys):
                    mask_ptr[i * num_keys + j] = (
                        (seq_len - i) * num_keys + num_keys - j
                    )
        else:
            for h in range(num_heads):
                var mask_head_ptr = mask_ptr + h * seq_len * num_keys
                for i in range(seq_len):
                    for j in range(num_keys):
                        mask_head_ptr[i * num_keys + j] = (
                            (seq_len - i) * num_keys + num_keys - j
                        )

    else:
        randn[qkv_type](q_ptr, q_size)
        randn[qkv_type](k_ptr, k_size)
        randn[mask_type](mask_ptr, mask_size)

    # Contruct buffers.
    var q = NDBuffer[qkv_type, 4](
        q_ptr, Index(batch_size, seq_len, num_heads, depth)
    )
    var k = NDBuffer[qkv_type, 4](
        k_ptr, Index(batch_size, num_keys, kv_num_heads, depth)
    )
    var mask = NDBuffer[mask_type, 2](mask_ptr, Index(seq_len, num_keys))
    var output = NDBuffer[qkv_type, 4](
        output_ptr, Index(batch_size, seq_len, num_heads, depth)
    )

    var flash_output = NDBuffer[qkv_type, 4](
        flash_output_ptr, Index(batch_size, seq_len, num_heads, depth)
    )

    @parameter
    if not against_gpu_naive:
        constrained[
            qkv_type == mask_type, "expect qkv and mask have same type for CPU."
        ]()
        _naive_attention_with_transpose[qkv_type](
            rebind[NDBuffer[qkv_type, 4]](output),
            rebind[NDBuffer[qkv_type, 4]](q),
            rebind[NDBuffer[qkv_type, 4]](k),
            rebind[NDBuffer[qkv_type, 4]](k),
            rebind[NDBuffer[qkv_type, 2]](mask),
            scale,
        )

    # Device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var mask_device_ptr = ctx.enqueue_create_buffer[mask_type](mask_size)
    var output_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)

    # Copy from host to device
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(mask_device_ptr, mask_ptr)

    # Contruct device buffers.
    var q_device = NDBuffer[
        qkv_type, 4, DimList(Dim(), Dim(), num_heads, depth)
    ](
        q_device_ptr.unsafe_ptr(),
        Index(batch_size, seq_len, num_heads, depth),
    )
    var k_device = NDBuffer[
        qkv_type, 4, DimList(Dim(), Dim(), kv_num_heads, depth)
    ](
        k_device_ptr.unsafe_ptr(),
        Index(batch_size, num_keys, kv_num_heads, depth),
    )
    var mask3d = NDBuffer[mask_type, 3, DimList.create_unknown[3]()](
        mask_device_ptr.unsafe_ptr(), Index(batch_size, seq_len, num_keys)
    )
    var mask4d = NDBuffer[mask_type, 4, DimList.create_unknown[4]()](
        mask_device_ptr.unsafe_ptr(),
        Index(batch_size, num_heads, seq_len, num_keys),
    )
    var output_device = NDBuffer[
        qkv_type, 4, DimList(Dim(), Dim(), num_heads, depth)
    ](
        output_device_ptr.unsafe_ptr(),
        Index(batch_size, seq_len, num_heads, depth),
    )

    alias q_tile_num_rows = 32
    alias k_tile_num_rows = 128

    @parameter
    @always_inline
    @__copy_capture(q_device, k_device, mask3d, mask4d, output_device)
    fn kernel_launch(ctx: DeviceContext) raises:
        @parameter
        if use_causal_mask:
            flare_mla_decoding[
                decoding_warp_split_k=decoding_warp_split_k,
                add_attn_mask=False,
            ](
                output_device,
                q_device,
                k_device,
                mask3d,
                CausalMask(),
                IdentityScoreMod(),
                scale,
                ctx,
                num_partitions,
            )
        elif mask_rank == 3:
            flare_mla_decoding[decoding_warp_split_k=decoding_warp_split_k](
                output_device,
                q_device,
                k_device,
                mask3d,
                NullMask(),
                IdentityScoreMod(),
                scale,
                ctx,
                num_partitions,
            )
        else:
            flare_mla_decoding[decoding_warp_split_k=decoding_warp_split_k](
                output_device,
                q_device,
                k_device,
                mask4d,
                NullMask(),
                IdentityScoreMod(),
                scale,
                ctx,
                num_partitions,
            )

    if is_benchmark():
        alias nrun = 200

        # Warmup
        kernel_launch(ctx)

        var nstime = ctx.execution_time[kernel_launch](nrun) / nrun
        var sectime = nstime / 1000000
        print(nrun, "runs avg", sectime, "ms")

    else:
        kernel_launch(ctx)

    ctx.synchronize()

    ctx.enqueue_copy(flash_output_ptr, output_device_ptr)

    @parameter
    if against_gpu_naive:
        var output_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
        var output_ref_device = NDBuffer[
            qkv_type, 4, DimList(Dim(), Dim(), num_heads, depth)
        ](
            output_ref_device_ptr.unsafe_ptr(),
            Index(batch_size, seq_len, num_heads, depth),
        )
        ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

        @parameter
        if use_causal_mask:
            var k_operand = NDBufferMHAOperand[ragged=False](k_device)
            var output_operand = NDBufferMHAOperand[ragged=False](
                output_ref_device
            )
            var q_operand = NDBufferMHAOperand[ragged=False](q_device)
            var null_valid_length = NDBuffer[DType.uint32, 1](
                UnsafePointer[UInt32](), Index(0)
            )
            mha_gpu_naive[use_mask_tensor=False,](
                q_operand,
                k_operand,
                k_operand,
                mask3d,
                CausalMask(),
                output_operand,
                scale,
                num_heads,
                depth,
                group,
                ctx,
            )
        elif mask_rank == 3:
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
        elif mask_rank == 4:
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

        ctx.synchronize()
        ctx.enqueue_copy(output_ptr, output_ref_device_ptr)
        _ = output_ref_device_ptr

    # since we pass the whole K tensor as the V tensor to our naive mha kernel,
    # the last 64 elements of each head in the reference result are invalid.
    var rtol = 1e-3
    for h in range(num_heads):
        for s in range(seq_len):
            for d in range(depth - 64):
                var expect = output_ptr.load(
                    d + depth * (h + s * num_heads)
                ).cast[DType.float64]()
                var actual = flash_output_ptr.load(
                    d + (depth - 64) * (h + s * num_heads)
                ).cast[DType.float64]()
                # if not isclose(actual, expect, atol=1e-3, rtol=rtol):
                #     var rerr = abs((actual - expect) / expect)
                #     print(h, s, d, actual, expect, rerr)
                assert_almost_equal(actual, expect, atol=1e-1, rtol=rtol)

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
    num_partitions: OptionalReg[Int],
    split_k: Bool,
    use_causal_mask: Bool = True,
    qkv_type: DType = DType.bfloat16,
](ctx: DeviceContext, use_index_input: Bool) raises:
    # BF16 token gen
    test[
        4,
        qkv_type,
        DType.float32,
        576,
        128,
        group=128,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
        decoding_warp_split_k=split_k,
        use_causal_mask=use_causal_mask,
    ](1, 50, ctx, use_index_input=use_index_input)

    test[
        3,
        qkv_type,
        DType.float32,
        576,
        128,
        group=128,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
        decoding_warp_split_k=split_k,
        use_causal_mask=use_causal_mask,
    ](1, 1024, ctx, use_index_input=use_index_input)

    # BF16 token gen, with num_heads=16 (deepseek-v2 lite)
    test[
        4,
        qkv_type,
        DType.float32,
        576,
        16,
        group=16,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
        decoding_warp_split_k=split_k,
        use_causal_mask=use_causal_mask,
    ](1, 50, ctx, use_index_input=use_index_input)

    test[
        3,
        qkv_type,
        DType.float32,
        576,
        16,
        group=16,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
        decoding_warp_split_k=split_k,
        use_causal_mask=use_causal_mask,
    ](1, 1024, ctx, use_index_input=use_index_input)


def main():
    with DeviceContext() as ctx:
        # tests with mask tensor
        test_decoding[27, 1, False, False](ctx, False)

        # tests with casual mask
        test_decoding[27, 1, False, True](ctx, False)
