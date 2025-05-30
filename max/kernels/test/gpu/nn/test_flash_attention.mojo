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

from collections import OptionalReg
from math import ceildiv, isclose, isqrt
from random import rand
from sys import argv, has_amd_gpu_accelerator

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import *
from gpu.host import DeviceContext
from gpu.host.info import A100, DEFAULT_GPU_ARCH, H100, Info, Vendor
from internal_utils import assert_with_measure
from memory import UnsafePointer
from nn.mha import (
    _naive_attention_with_transpose,
    flash_attention,
    mha_gpu_naive,
)
from nn.mha_mask import MaterializedMask
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal

from utils.index import Index


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


fn is_sm8(info: Info) -> Bool:
    return (
        info.vendor == Vendor.NVIDIA_GPU
        and info.compute >= 8
        and info.compute < 9
    )


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
](
    seq_len: Int,
    num_keys: Int,
    ctx: DeviceContext,
    is_benchmark: Bool = False,
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
    var v_size = k_size
    var o_size = q_size
    var mask_size = (
        (num_heads if mask_rank == 4 else 1) * seq_len * num_keys * batch_size
    )

    # Allocate memory for all variables.
    var q_ptr = UnsafePointer[Scalar[qkv_type]].alloc(q_size)
    var k_ptr = UnsafePointer[Scalar[qkv_type]].alloc(k_size)
    var v_ptr = UnsafePointer[Scalar[qkv_type]].alloc(v_size)
    var mask_ptr = UnsafePointer[Scalar[mask_type]].alloc(mask_size)
    var output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)
    var flash_output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)

    # Q, K, V are randomly initialized.
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
        for i in range(num_keys):
            for h in range(kv_num_heads):
                for j in range(depth):
                    v_ptr[(i * kv_num_heads + h) * depth + j] = i * depth + j

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
        rand[qkv_type](q_ptr, q_size)
        rand[qkv_type](k_ptr, k_size)
        rand[qkv_type](v_ptr, v_size)
        rand[mask_type](mask_ptr, mask_size)

    # Contruct buffers.
    var q = NDBuffer[qkv_type, 4](
        q_ptr, Index(batch_size, seq_len, num_heads, depth)
    )
    var k = NDBuffer[qkv_type, 4](
        k_ptr, Index(batch_size, num_keys, kv_num_heads, depth)
    )
    var v = NDBuffer[qkv_type, 4](
        v_ptr, Index(batch_size, num_keys, kv_num_heads, depth)
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
            rebind[NDBuffer[qkv_type, 4, output.origin]](output),
            rebind[NDBuffer[qkv_type, 4, q.origin]](q),
            rebind[NDBuffer[qkv_type, 4, k.origin]](k),
            rebind[NDBuffer[qkv_type, 4, v.origin]](v),
            rebind[NDBuffer[qkv_type, 2, mask.origin]](mask),
            scale,
        )

    # Device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_device_ptr = ctx.enqueue_create_buffer[qkv_type](v_size)
    var mask_device_ptr = ctx.enqueue_create_buffer[mask_type](mask_size)
    var output_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)

    # Copy from host to device
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(v_device_ptr, v_ptr)
    ctx.enqueue_copy(mask_device_ptr, mask_ptr)

    # Contruct device buffers.
    var q_device = NDBuffer[
        qkv_type, 4, _, DimList(Dim(), Dim(), num_heads, depth)
    ](
        q_device_ptr.unsafe_ptr(),
        Index(batch_size, seq_len, num_heads, depth),
    )
    var k_device = NDBuffer[
        qkv_type, 4, _, DimList(Dim(), Dim(), kv_num_heads, depth)
    ](
        k_device_ptr.unsafe_ptr(),
        Index(batch_size, num_keys, kv_num_heads, depth),
    )
    var v_device = NDBuffer[
        qkv_type, 4, _, DimList(Dim(), Dim(), kv_num_heads, depth)
    ](
        v_device_ptr.unsafe_ptr(),
        Index(batch_size, num_keys, kv_num_heads, depth),
    )
    var mask3d = NDBuffer[mask_type, 3, _, DimList.create_unknown[3]()](
        mask_device_ptr.unsafe_ptr(), Index(batch_size, seq_len, num_keys)
    )
    var mask4d = NDBuffer[mask_type, 4, _, DimList.create_unknown[4]()](
        mask_device_ptr.unsafe_ptr(),
        Index(batch_size, num_heads, seq_len, num_keys),
    )
    var output_device = NDBuffer[
        qkv_type, 4, _, DimList(Dim(), Dim(), num_heads, depth)
    ](
        output_device_ptr.unsafe_ptr(),
        Index(batch_size, seq_len, num_heads, depth),
    )

    alias q_tile_num_rows = 32
    alias k_tile_num_rows = 128

    @parameter
    @always_inline
    @__copy_capture(q_device, k_device, v_device, mask3d, mask4d, output_device)
    fn kernel_launch(ctx: DeviceContext) raises:
        @parameter
        if mask_rank == 3:
            flash_attention[decoding_warp_split_k=decoding_warp_split_k](
                output_device,
                q_device,
                k_device,
                v_device,
                MaterializedMask(mask3d),
                IdentityScoreMod(),
                scale,
                ctx,
                num_partitions,
            )
        else:
            flash_attention[decoding_warp_split_k=decoding_warp_split_k](
                output_device,
                q_device,
                k_device,
                v_device,
                MaterializedMask(mask4d),
                IdentityScoreMod(),
                scale,
                ctx,
                num_partitions,
            )

    if is_benchmark:
        alias nrun = 50

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
            qkv_type, 4, _, DimList(Dim(), Dim(), num_heads, depth)
        ](
            output_ref_device_ptr.unsafe_ptr(),
            Index(batch_size, seq_len, num_heads, depth),
        )
        ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

        @parameter
        if mask_rank == 3:
            mha_gpu_naive(
                q_device,
                k_device,
                v_device,
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
                v_device,
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

    @parameter
    fn get_rtol() -> Float64:
        @parameter
        if has_amd_gpu_accelerator():
            if num_partitions.value() >= 16:
                return 4e-2
            elif num_partitions.value() >= 4:
                return 3e-2
            elif num_partitions.value() >= 2:
                return 2.5e-2
            else:
                return 1e-2
        else:
            return 2e-2 if num_partitions.value() >= 4 else 1e-2

    var rtol = get_rtol()
    for h in range(num_heads):
        for s in range(seq_len):
            for d in range(depth):
                var expect = output_ptr.load(
                    d + depth * (h + s * num_heads)
                ).cast[DType.float64]()
                var actual = flash_output_ptr.load(
                    d + depth * (h + s * num_heads)
                ).cast[DType.float64]()
                var rerr = abs((actual - expect) / expect)
                assert_almost_equal(
                    actual,
                    expect,
                    atol=1e-5,
                    rtol=rtol,
                    msg=String(h, s, d, actual, expect, rerr, sep=" "),
                )

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = mask_device_ptr
    _ = output_device_ptr

    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    mask_ptr.free()
    output_ptr.free()
    flash_output_ptr.free()


fn test_context_encoding(ctx: DeviceContext) raises:
    # fp32 arbitrary depth and num_heads, baseline impl.
    test[3, DType.float32, DType.float32, 127, 2](111, 121, ctx)
    # fp32 depth == 128, tf32-fp32 mma, llama2 shape.
    test[
        4,
        DType.float32,
        DType.float32,
        128,
        32,
        against_gpu_naive=True,
    ](1024, 1024, ctx, is_benchmark())
    test[
        3,
        DType.float32,
        DType.float32,
        128,
        3,
        against_gpu_naive=True,
    ](14, 14, ctx, is_benchmark())
    test[
        3,
        DType.float32,
        DType.float32,
        128,
        1,
        against_gpu_naive=True,
    ](178, 178, ctx, is_benchmark())
    # bf16 depth == 128, bf16-fp32 mma
    test[
        4,
        DType.bfloat16,
        DType.bfloat16,
        depth=128,
        num_heads=1,
        against_gpu_naive=True,
    ](128, 128, ctx)
    test[
        4,
        DType.bfloat16,
        DType.float32,
        depth=128,
        num_heads=1,
        against_gpu_naive=True,
    ](384, 384, ctx)
    test[
        3,
        DType.bfloat16,
        DType.float32,
        128,
        3,
        against_gpu_naive=True,
    ](256, 256, ctx)
    test[
        4,
        DType.bfloat16,
        DType.float32,
        128,
        32,
        against_gpu_naive=True,
    ](1024, 1024, ctx, is_benchmark())
    test[
        4,
        DType.bfloat16,
        DType.float32,
        128,
        24,
        group=3,
        against_gpu_naive=True,
    ](1024, 1024, ctx)
    # BF16 with sequence length not multiple of 128
    test[
        4,
        DType.bfloat16,
        DType.float32,
        128,
        3,
        group=3,
        against_gpu_naive=True,
    ](64, 64, ctx)
    test[
        4,
        DType.bfloat16,
        DType.bfloat16,
        128,
        3,
        group=3,
        against_gpu_naive=True,
    ](102, 102, ctx)
    test[
        3,
        DType.bfloat16,
        DType.float32,
        128,
        1,
        against_gpu_naive=True,
    ](14, 14, ctx)
    test[
        3,
        DType.bfloat16,
        DType.bfloat16,
        128,
        1,
        against_gpu_naive=True,
    ](528, 528, ctx)


fn test_decoding[
    batch_size: Int,
    num_partitions: OptionalReg[Int],
    split_k: Bool,
    qkv_type: DType = DType.bfloat16,
](ctx: DeviceContext, use_index_input: Bool) raises:
    test[
        3,
        qkv_type,
        DType.float32,
        128,
        1,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
        decoding_warp_split_k=split_k,
    ](1, 11, ctx, use_index_input=use_index_input)
    if (
        not is_sm8(ctx.device_info)
        or num_partitions
        and num_partitions.value() == 1
    ):
        test[
            4,
            qkv_type,
            DType.bfloat16,
            128,
            2,
            against_gpu_naive=True,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
        ](1, 523, ctx, use_index_input=use_index_input)
    test[
        4,
        qkv_type,
        DType.float32,
        128,
        24,
        group=3,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
        decoding_warp_split_k=split_k,
    ](1, 29, ctx, use_index_input=use_index_input)

    # TODO(KERN-1674): enable these tests after fixing the bug
    # test[
    #     4,
    #     qkv_type,
    #     DType.bfloat16,
    #     128,
    #     3,
    #     group=3,
    #     against_gpu_naive=True,
    #     batch_size=batch_size,
    #     num_partitions=num_partitions,
    #     decoding_warp_split_k=split_k,
    # ](1, 156, ctx, use_index_input=use_index_input)
    # test[
    #     4,
    #     qkv_type,
    #     DType.bfloat16,
    #     128,
    #     3,
    #     group=3,
    #     against_gpu_naive=True,
    #     batch_size=batch_size,
    #     num_partitions=num_partitions,
    #     decoding_warp_split_k=split_k,
    # ](1, 208, ctx, use_index_input=use_index_input)


fn test_decoding_large_group[
    batch_size: Int,
    num_partitions: OptionalReg[Int] = None,
    split_k: Bool = False,
    qkv_type: DType = DType.bfloat16,
](ctx: DeviceContext, use_index_input: Bool = False) raises:
    test[
        4,
        qkv_type,
        DType.float32,
        128,
        32,
        group=16,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
        decoding_warp_split_k=split_k,
    ](1, 2000, ctx, use_index_input=use_index_input)


fn test_cross_attention[batch_size: Int](ctx: DeviceContext) raises:
    test[
        4,
        DType.bfloat16,
        DType.bfloat16,
        depth=128,
        num_heads=1,
        against_gpu_naive=True,
    ](128, 64, ctx, use_index_input=True)

    test[
        4,
        DType.bfloat16,
        DType.float32,
        128,
        3,
        against_gpu_naive=True,
    ](256, 128, ctx)

    test[
        3,
        DType.bfloat16,
        DType.float32,
        128,
        24,
        group=3,
        against_gpu_naive=True,
    ](1024, 100, ctx)

    test[
        4,
        DType.float32,
        DType.float32,
        128,
        24,
        group=3,
        against_gpu_naive=True,
    ](214, 300, ctx)

    test[
        3,
        DType.bfloat16,
        DType.float32,
        128,
        24,
        group=1,
        against_gpu_naive=True,
    ](512, 1024, ctx)

    test[
        3,
        DType.float32,
        DType.float32,
        128,
        32,
        group=3,
        against_gpu_naive=True,
    ](12, 8, ctx)

    test[
        4,
        DType.bfloat16,
        DType.float32,
        128,
        3,
        against_gpu_naive=True,
    ](14, 18, ctx)

    # odd seq_len
    test[
        4,
        DType.bfloat16,
        DType.float32,
        128,
        3,
        against_gpu_naive=True,
    ](15, 18, ctx)
    test[
        3,
        DType.bfloat16,
        DType.float32,
        128,
        3,
        against_gpu_naive=True,
    ](119, 200, ctx)


def main():
    with DeviceContext() as ctx:
        test_context_encoding(ctx)
        test_cross_attention[1](ctx)

        # KERN-1726: Disable warp split-k because it fails with mha_decoding_single_batch
        # specifically for num_keys = 523.
        @parameter
        for split_k in range(1):

            @parameter
            for batch_size in range(1, 5, 3):
                test_decoding[batch_size, 1, split_k](ctx, False)

                @parameter
                if not split_k:
                    test_decoding[batch_size, 1, split_k, DType.float32](
                        ctx, False
                    )

                @parameter
                if ctx.device_info is A100 or ctx.device_info is H100:
                    test_decoding_large_group[batch_size, 1](ctx)

                test_decoding[batch_size, 2, split_k](ctx, False)
                test_decoding[batch_size, 4, split_k](ctx, False)

                @parameter
                if not split_k:
                    test_decoding[batch_size, 4, split_k, DType.float32](
                        ctx, False
                    )
                test_decoding[batch_size, None, split_k](ctx, False)
                test_decoding[batch_size, 32, split_k](ctx, False)
