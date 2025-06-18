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
from math import ceildiv, isclose
from random import randn
from sys import argv

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import *
from gpu.host import DeviceContext
from nn.mha import _naive_attention_with_transpose, mha_gpu_naive
from nn.mha_mask import CausalMask, MaterializedMask
from nn.mha_operand import NDBufferMHAOperand
from nn.mha_score_mod import IdentityScoreMod
from nn.mla import flare_mla_decoding, flare_mla_prefill
from tensor_internal import IOUnknown, ManagedTensorSlice
from tensor_internal.managed_tensor_slice import StaticTensorSpec
from testing import assert_almost_equal

from utils.index import Index
from utils.numerics import get_accum_type


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
        "test_mla_decoding",
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
        (num_heads if mask_rank == 4 else 1) * seq_len * num_keys * batch_size
    )

    # Allocate memory for all variables.
    var q_ptr = UnsafePointer[Scalar[qkv_type]].alloc(q_size)
    var k_ptr = UnsafePointer[Scalar[qkv_type]].alloc(k_size)
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

    # Construct buffers.
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
            rebind[NDBuffer[qkv_type, 4, output.origin]](output),
            rebind[NDBuffer[qkv_type, 4, q.origin]](q),
            rebind[NDBuffer[qkv_type, 4, k.origin]](k),
            rebind[NDBuffer[qkv_type, 4, k.origin]](k),
            rebind[NDBuffer[qkv_type, 2, mask.origin]](mask),
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

    # Construct device buffers.
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
    @__copy_capture(q_device, k_device, mask3d, mask4d, output_device)
    fn kernel_launch(ctx: DeviceContext) raises:
        @parameter
        if use_causal_mask:
            flare_mla_decoding[decoding_warp_split_k=decoding_warp_split_k,](
                output_device,
                q_device,
                k_device,
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
                MaterializedMask(mask3d),
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
                MaterializedMask(mask4d),
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
            qkv_type, 4, _, DimList(Dim(), Dim(), num_heads, depth)
        ](
            output_ref_device_ptr.unsafe_ptr(),
            Index(batch_size, seq_len, num_heads, depth),
        )
        ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

        @parameter
        if use_causal_mask:
            var k_operand = NDBufferMHAOperand(k_device)
            var null_valid_length = NDBuffer[DType.uint32, 1](
                UnsafePointer[UInt32](), Index(0)
            )
            mha_gpu_naive[_is_cache_length_accurate=True,](
                q_device,
                k_operand,
                k_operand,
                CausalMask(),
                output_ref_device,
                ManagedTensorSlice[
                    io_spec=IOUnknown,
                    static_spec = StaticTensorSpec[
                        DType.uint32, 1
                    ].create_unknown(),
                ](null_valid_length),
                scale,
                batch_size,
                seq_len,
                num_keys,
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


fn test_prefill[
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    kv_depth: Int,
    cache_depth: Int,
    cache_num_heads: Int,
    batch_size: Int = 1,
    use_causal_mask: Bool = True,
](seq_len: Int, num_keys: Int, ctx: DeviceContext,) raises:
    print(
        "test_mla_prefill",
        "batch_size:",
        batch_size,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
        "qkv_type:",
        qkv_type,
        "depth:",
        depth,
        "kv_depth:",
        kv_depth,
        "cache_depth:",
        cache_depth,
        "cache_num_heads:",
        cache_num_heads,
    )

    alias scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))

    var q_size = batch_size * seq_len * num_heads * depth
    var k_size = batch_size * num_keys * num_heads * kv_depth
    var v_size = k_size
    var o_size = batch_size * seq_len * num_heads * kv_depth
    var cache_size = batch_size * num_keys * cache_num_heads * cache_depth

    var q_ptr = UnsafePointer[Scalar[qkv_type]].alloc(q_size)
    var k_ptr = UnsafePointer[Scalar[qkv_type]].alloc(k_size)
    var v_ptr = UnsafePointer[Scalar[qkv_type]].alloc(v_size)
    var cache_ptr = UnsafePointer[Scalar[qkv_type]].alloc(cache_size)
    var output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)

    # Q, K, V, cache are randomly initialized.
    randn[qkv_type](q_ptr, q_size)
    randn[qkv_type](k_ptr, k_size)
    randn[qkv_type](v_ptr, v_size)
    randn[qkv_type](cache_ptr, cache_size)

    # input row offsets and cache row offsets
    var input_row_offsets = UnsafePointer[UInt32].alloc(batch_size + 1)
    var cache_row_offsets = UnsafePointer[UInt32].alloc(batch_size + 1)
    for i in range(batch_size):
        input_row_offsets[i] = i * seq_len
        cache_row_offsets[i] = i * num_keys
    input_row_offsets[batch_size] = batch_size * seq_len
    cache_row_offsets[batch_size] = batch_size * num_keys

    # ragged inputs
    var q = NDBuffer[qkv_type, 3](
        q_ptr, Index(batch_size * seq_len, num_heads, depth)
    )
    var k = NDBuffer[qkv_type, 3](
        k_ptr, Index(batch_size * num_keys, num_heads, kv_depth)
    )
    var v = NDBuffer[qkv_type, 3](
        v_ptr, Index(batch_size * num_keys, num_heads, kv_depth)
    )
    var cache = NDBuffer[qkv_type, 4](
        cache_ptr, Index(batch_size, num_keys, cache_num_heads, cache_depth)
    )
    var output = NDBuffer[qkv_type, 3](
        output_ptr, Index(batch_size * seq_len, num_heads, kv_depth)
    )

    # device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_device_ptr = ctx.enqueue_create_buffer[qkv_type](v_size)
    var cache_device_ptr = ctx.enqueue_create_buffer[qkv_type](cache_size)
    var output_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
    var input_row_offsets_device_ptr = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var cache_row_offsets_device_ptr = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )

    # copy from host to device
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(v_device_ptr, v_ptr)
    ctx.enqueue_copy(cache_device_ptr, cache_ptr)
    ctx.enqueue_copy(input_row_offsets_device_ptr, input_row_offsets)
    ctx.enqueue_copy(cache_row_offsets_device_ptr, cache_row_offsets)

    # construct device buffers
    var q_device = NDBuffer[qkv_type, 3, _, DimList(Dim(), num_heads, depth)](
        q_device_ptr.unsafe_ptr(),
        Index(batch_size * seq_len, num_heads, depth),
    )
    var k_device = NDBuffer[
        qkv_type, 3, _, DimList(Dim(), num_heads, kv_depth)
    ](
        k_device_ptr.unsafe_ptr(),
        Index(batch_size * num_keys, num_heads, kv_depth),
    )
    var v_device = NDBuffer[
        qkv_type, 3, _, DimList(Dim(), num_heads, kv_depth)
    ](
        v_device_ptr.unsafe_ptr(),
        Index(batch_size * num_keys, num_heads, kv_depth),
    )
    var cache_device = NDBuffer[
        qkv_type, 4, _, DimList(Dim(), Dim(), cache_num_heads, cache_depth)
    ](
        cache_device_ptr.unsafe_ptr(),
        Index(batch_size, num_keys, cache_num_heads, cache_depth),
    )
    var output_device = NDBuffer[
        qkv_type, 3, _, DimList(Dim(), num_heads, kv_depth)
    ](
        output_device_ptr.unsafe_ptr(),
        Index(batch_size * seq_len, num_heads, kv_depth),
    )
    var input_row_offsets_device = NDBuffer[DType.uint32, 1, _, DimList(Dim())](
        input_row_offsets_device_ptr.unsafe_ptr(),
        Index(batch_size + 1),
    )
    var cache_row_offsets_device = NDBuffer[DType.uint32, 1, _, DimList(Dim())](
        cache_row_offsets_device_ptr.unsafe_ptr(),
        Index(batch_size + 1),
    )

    @parameter
    @always_inline
    @__copy_capture(
        q_device,
        k_device,
        v_device,
        cache_device,
        input_row_offsets_device,
        cache_row_offsets_device,
        output_device,
    )
    fn kernel_launch(ctx: DeviceContext) raises:
        flare_mla_prefill[softmax_type = DType.float32](
            output_device,
            q_device,
            k_device,
            v_device,
            cache_device,
            CausalMask(),
            IdentityScoreMod(),
            input_row_offsets_device,
            cache_row_offsets_device,
            scale,
            ctx,
            q_max_seq_len=seq_len,
        )

    if is_benchmark():
        alias nrun = 200

        # Warmup
        for i in range(20):
            kernel_launch(ctx)

        var nstime = ctx.execution_time[kernel_launch](nrun) / nrun
        var sectime = nstime / 1000000

        var tflops = (
            2
            * batch_size
            * num_heads
            * ((-seq_len * seq_len + 2 * seq_len * num_keys))
            * (depth + kv_depth)
            / sectime
            / 1e9
        )
        print(nrun, "runs avg: ", sectime, " ms   ", tflops, " TFLOPs")

    else:
        kernel_launch(ctx)

    ctx.synchronize()
    ctx.enqueue_copy(output_ptr, output_device_ptr)

    # create reference K and V
    # unlike flare_mla_prefill, K_ref and V_ref each head is of size depth (not kv_depth)
    var k_ref_ptr = UnsafePointer[Scalar[qkv_type]].alloc(
        batch_size * num_keys * num_heads * depth
    )
    var v_ref_ptr = UnsafePointer[Scalar[qkv_type]].alloc(
        batch_size * num_keys * num_heads * depth
    )
    var output_ref_ptr = UnsafePointer[Scalar[qkv_type]].alloc(
        batch_size * seq_len * num_heads * depth
    )

    # create reference K and V
    var k_ref = NDBuffer[qkv_type, 4](
        k_ref_ptr, Index(batch_size, num_keys, num_heads, depth)
    )
    var v_ref = NDBuffer[qkv_type, 4](
        v_ref_ptr, Index(batch_size, num_keys, num_heads, depth)
    )
    var output_ref = NDBuffer[qkv_type, 4](
        output_ref_ptr, Index(batch_size, seq_len, num_heads, depth)
    )

    # the first kv_depth elements of each head in K_ref and V_ref are the same as K and V
    for b in range(batch_size):
        for s in range(num_keys):
            for h in range(num_heads):
                for d in range(kv_depth):
                    k_ref[b, s, h, d] = k[b * num_keys + s, h, d]
                    v_ref[b, s, h, d] = v[b * num_keys + s, h, d]

    # the rest of the elements in K_ref are broadcasted from the last (depth - kv_depth) elements of the head in cache
    # the rest of the elements in V_ref are zeros
    for b in range(batch_size):
        for s in range(num_keys):
            for h in range(num_heads):
                for d in range(depth - kv_depth):
                    k_ref[b, s, h, d + kv_depth] = cache[
                        b, s, 0, cache_depth - (depth - kv_depth) + d
                    ]
                    v_ref[b, s, h, d + kv_depth] = 0

    # view q_device as a rank 4 buffer
    var q_device_rank4 = NDBuffer[
        qkv_type, 4, _, DimList(Dim(), Dim(), num_heads, depth)
    ](
        q_device_ptr.unsafe_ptr(),
        Index(batch_size, seq_len, num_heads, depth),
    )

    # create device pointers for K_ref and V_ref
    var k_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](
        batch_size * num_keys * num_heads * depth
    )
    var v_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](
        batch_size * num_keys * num_heads * depth
    )
    var output_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](
        batch_size * seq_len * num_heads * depth
    )
    # create device buffers for K_ref and V_ref
    var k_ref_device = NDBuffer[
        qkv_type, 4, _, DimList(Dim(), Dim(), num_heads, depth)
    ](
        k_ref_device_ptr.unsafe_ptr(),
        Index(batch_size, num_keys, num_heads, depth),
    )
    var v_ref_device = NDBuffer[
        qkv_type, 4, _, DimList(Dim(), Dim(), num_heads, depth)
    ](
        v_ref_device_ptr.unsafe_ptr(),
        Index(batch_size, num_keys, num_heads, depth),
    )
    var output_ref_device = NDBuffer[
        qkv_type, 4, _, DimList(Dim(), Dim(), num_heads, depth)
    ](
        output_ref_device_ptr.unsafe_ptr(),
        Index(batch_size, seq_len, num_heads, depth),
    )

    # copy from host to device
    ctx.enqueue_copy(k_ref_device_ptr, k_ref_ptr)
    ctx.enqueue_copy(v_ref_device_ptr, v_ref_ptr)

    var null_valid_length = NDBuffer[DType.uint32, 1](
        UnsafePointer[UInt32](), Index(0)
    )

    var k_ref_operand = NDBufferMHAOperand(k_ref_device)
    var v_ref_operand = NDBufferMHAOperand(v_ref_device)

    # create reference output
    mha_gpu_naive[_is_cache_length_accurate=True](
        q_device_rank4,
        k_ref_operand,
        v_ref_operand,
        CausalMask(),
        output_ref_device,
        ManagedTensorSlice[
            io_spec=IOUnknown,
            static_spec = StaticTensorSpec[DType.uint32, 1].create_unknown(),
        ](null_valid_length),
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        1,
        ctx,
    )

    ctx.enqueue_copy(output_ref_ptr, output_ref_device_ptr)
    ctx.synchronize()

    # view output as a rank 4 buffer
    var output_rank4 = NDBuffer[qkv_type, 4](
        output_ptr, Index(batch_size, seq_len, num_heads, kv_depth)
    )

    # compare output with reference
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(kv_depth):
                    lhs = output_rank4[b, s, h, d]
                    rhs = output_ref[b, s, h, d]
                    assert_almost_equal(lhs, rhs, atol=2e-2, rtol=2e-2)

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = cache_device_ptr
    _ = output_device_ptr
    _ = k_ref_device_ptr
    _ = v_ref_device_ptr
    _ = output_ref_device_ptr

    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    cache_ptr.free()
    output_ptr.free()
    k_ref_ptr.free()
    v_ref_ptr.free()
    output_ref_ptr.free()


fn test_cascade_prefill[
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    kv_depth: Int,
    cache_depth: Int,
    cache_num_heads: Int,
    batch_size: Int = 1,
    chunk_size: Int = 128,
    use_causal_mask: Bool = True,
](seq_len: Int, num_keys: Int, ctx: DeviceContext,) raises:
    print(
        "test_mla_cascade_prefill",
        "batch_size:",
        batch_size,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
        "qkv_type:",
        qkv_type,
        "depth:",
        depth,
        "kv_depth:",
        kv_depth,
        "cache_depth:",
        cache_depth,
        "cache_num_heads:",
        cache_num_heads,
        "chunk_size:",
        chunk_size,
    )

    alias scale = Float32(0.125)
    alias accum_type = get_accum_type[qkv_type]()

    var total_iters = ceildiv(num_keys, chunk_size)
    var q_size = batch_size * seq_len * num_heads * depth
    var k_size = batch_size * num_keys * num_heads * kv_depth
    var v_size = k_size
    var k_chunk_size = batch_size * chunk_size * num_heads * kv_depth
    var v_chunk_size = k_chunk_size
    var o_size = batch_size * seq_len * num_heads * kv_depth
    var cache_size = batch_size * num_keys * cache_num_heads * cache_depth
    var softmax_info_size = num_heads * batch_size * seq_len * 2

    var q_ptr = UnsafePointer[Scalar[qkv_type]].alloc(q_size)
    var k_ptr = UnsafePointer[Scalar[qkv_type]].alloc(k_size)
    var v_ptr = UnsafePointer[Scalar[qkv_type]].alloc(v_size)
    var k_chunk_ptr = UnsafePointer[Scalar[qkv_type]].alloc(k_chunk_size)
    var v_chunk_ptr = UnsafePointer[Scalar[qkv_type]].alloc(v_chunk_size)
    var output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)
    var cache_ptr = UnsafePointer[Scalar[qkv_type]].alloc(cache_size)

    # Q, K, V, cache are randomly initialized.
    randn[qkv_type](q_ptr, q_size)
    randn[qkv_type](k_ptr, k_size)
    randn[qkv_type](v_ptr, v_size)
    randn[qkv_type](cache_ptr, cache_size)

    # input row offsets and cache row offsets
    var input_row_offsets = UnsafePointer[UInt32].alloc(batch_size + 1)
    var cache_row_offsets = UnsafePointer[UInt32].alloc(batch_size + 1)
    var cache_offsets = UnsafePointer[UInt32].alloc(batch_size)

    for i in range(batch_size):
        input_row_offsets[i] = i * seq_len
    input_row_offsets[batch_size] = batch_size * seq_len

    # ragged inputs
    var k = NDBuffer[qkv_type, 3](
        k_ptr, Index(batch_size * num_keys, num_heads, kv_depth)
    )
    var v = NDBuffer[qkv_type, 3](
        v_ptr, Index(batch_size * num_keys, num_heads, kv_depth)
    )
    var k_chunk = NDBuffer[qkv_type, 3](
        k_chunk_ptr, Index(batch_size * chunk_size, num_heads, kv_depth)
    )
    var v_chunk = NDBuffer[qkv_type, 3](
        v_chunk_ptr, Index(batch_size * chunk_size, num_heads, kv_depth)
    )
    var output = NDBuffer[qkv_type, 3](
        output_ptr, Index(batch_size * seq_len, num_heads, kv_depth)
    )

    # device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_device_ptr = ctx.enqueue_create_buffer[qkv_type](v_size)
    var k_chunk_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_chunk_size)
    var v_chunk_device_ptr = ctx.enqueue_create_buffer[qkv_type](v_chunk_size)
    var cache_device_ptr = ctx.enqueue_create_buffer[qkv_type](cache_size)
    var output_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
    var input_row_offsets_device_ptr = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var cache_row_offsets_device_ptr = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var cache_offsets_device_ptr = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    var softmax_info_device_ptr = ctx.enqueue_create_buffer[accum_type](
        softmax_info_size
    )

    # copy from host to device
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(v_device_ptr, v_ptr)
    ctx.enqueue_copy(cache_device_ptr, cache_ptr)
    ctx.enqueue_copy(input_row_offsets_device_ptr, input_row_offsets)

    # construct device buffers
    var q_device = NDBuffer[qkv_type, 3, _, DimList(Dim(), num_heads, depth)](
        q_device_ptr.unsafe_ptr(),
        Index(batch_size * seq_len, num_heads, depth),
    )
    var k_device = NDBuffer[
        qkv_type, 3, _, DimList(Dim(), num_heads, kv_depth)
    ](
        k_device_ptr.unsafe_ptr(),
        Index(batch_size * num_keys, num_heads, kv_depth),
    )
    var v_device = NDBuffer[
        qkv_type, 3, _, DimList(Dim(), num_heads, kv_depth)
    ](
        v_device_ptr.unsafe_ptr(),
        Index(batch_size * num_keys, num_heads, kv_depth),
    )
    var k_chunk_device = NDBuffer[
        qkv_type, 3, _, DimList(Dim(), num_heads, kv_depth)
    ](
        k_chunk_device_ptr.unsafe_ptr(),
        Index(batch_size * chunk_size, num_heads, kv_depth),
    )
    var v_chunk_device = NDBuffer[
        qkv_type, 3, _, DimList(Dim(), num_heads, kv_depth)
    ](
        v_chunk_device_ptr.unsafe_ptr(),
        Index(batch_size * chunk_size, num_heads, kv_depth),
    )
    var cache_device = NDBuffer[
        qkv_type, 4, _, DimList(Dim(), Dim(), cache_num_heads, cache_depth)
    ](
        cache_device_ptr.unsafe_ptr(),
        Index(batch_size, num_keys, cache_num_heads, cache_depth),
    )
    var output_device = NDBuffer[
        qkv_type, 3, _, DimList(Dim(), num_heads, kv_depth)
    ](
        output_device_ptr.unsafe_ptr(),
        Index(batch_size * seq_len, num_heads, kv_depth),
    )
    var input_row_offsets_device = NDBuffer[DType.uint32, 1, _, DimList(Dim())](
        input_row_offsets_device_ptr.unsafe_ptr(),
        Index(batch_size + 1),
    )
    var cache_row_offsets_device = NDBuffer[DType.uint32, 1, _, DimList(Dim())](
        cache_row_offsets_device_ptr.unsafe_ptr(),
        Index(batch_size + 1),
    )
    var cache_offsets_device = NDBuffer[DType.uint32, 1, _, DimList(Dim())](
        cache_offsets_device_ptr.unsafe_ptr(),
        Index(batch_size),
    )
    var softmax_info_device = NDBuffer[accum_type, 3, _, DimList(Dim(), Dim())](
        softmax_info_device_ptr.unsafe_ptr(),
        Index(batch_size * seq_len, num_heads, 2),
    )

    for i_iter in range(total_iters):
        curr_chunk_size = min(chunk_size, num_keys - i_iter * chunk_size)

        # update cache_row_offsets and cache_offsets
        for i in range(batch_size):
            cache_offsets[i] = i_iter * chunk_size
            cache_row_offsets[i] = i * curr_chunk_size
        cache_row_offsets[batch_size] = batch_size * curr_chunk_size

        # copy from k and v to k_chunk and v_chunk
        for b in range(batch_size):
            for s in range(curr_chunk_size):
                for h in range(num_heads):
                    for d in range(kv_depth):
                        k_chunk[b * curr_chunk_size + s, h, d] = k[
                            b * num_keys + i_iter * chunk_size + s, h, d
                        ]
                        v_chunk[b * curr_chunk_size + s, h, d] = v[
                            b * num_keys + i_iter * chunk_size + s, h, d
                        ]

        # copy to device
        ctx.enqueue_copy(cache_row_offsets_device_ptr, cache_row_offsets)
        ctx.enqueue_copy(cache_offsets_device_ptr, cache_offsets)
        ctx.enqueue_copy(k_chunk_device_ptr, k_chunk_ptr)
        ctx.enqueue_copy(v_chunk_device_ptr, v_chunk_ptr)

        if i_iter == 0:
            flare_mla_prefill[write_softmax_info=True](
                output_device,
                q_device,
                k_chunk_device,
                v_chunk_device,
                cache_device,
                CausalMask(),
                IdentityScoreMod(),
                input_row_offsets_device,
                cache_row_offsets_device,
                scale,
                ctx,
                q_max_seq_len=seq_len,
                softmax_info=OptionalReg[
                    NDBuffer[accum_type, 3, MutableAnyOrigin]
                ](softmax_info_device),
                cache_offsets=OptionalReg[
                    NDBuffer[DType.uint32, 1, MutableAnyOrigin]
                ](cache_offsets_device),
            )
        else:
            flare_mla_prefill[
                write_softmax_info=True, use_cascade_attention=True
            ](
                output_device,
                q_device,
                k_chunk_device,
                v_chunk_device,
                cache_device,
                CausalMask(),
                IdentityScoreMod(),
                input_row_offsets_device,
                cache_row_offsets_device,
                scale,
                ctx,
                q_max_seq_len=seq_len,
                softmax_info=OptionalReg[
                    NDBuffer[accum_type, 3, MutableAnyOrigin]
                ](softmax_info_device),
                cache_offsets=OptionalReg[
                    NDBuffer[DType.uint32, 1, MutableAnyOrigin]
                ](cache_offsets_device),
            )

    ctx.enqueue_copy(output_ptr, output_device_ptr)

    # create reference output
    var output_ref_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)
    var output_ref = NDBuffer[qkv_type, 3](
        output_ref_ptr, Index(batch_size * seq_len, num_heads, kv_depth)
    )

    # create device pointers for K_ref and V_ref
    var output_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
    var output_ref_device = NDBuffer[
        qkv_type, 3, _, DimList(Dim(), num_heads, kv_depth)
    ](
        output_ref_device_ptr.unsafe_ptr(),
        Index(batch_size * seq_len, num_heads, kv_depth),
    )

    # create cache_row_offsets for reference
    var cache_row_offsets_ref = UnsafePointer[UInt32].alloc(batch_size + 1)
    for i in range(batch_size + 1):
        cache_row_offsets_ref[i] = i * num_keys

    # create device pointers for cache_row_offsets_ref and input_row_offsets_device
    var cache_row_offsets_ref_device_ptr = ctx.enqueue_create_buffer[
        DType.uint32
    ](batch_size + 1)

    # copy from host to device
    ctx.enqueue_copy(cache_row_offsets_ref_device_ptr, cache_row_offsets_ref)

    var cache_row_offsets_ref_device = NDBuffer[
        DType.uint32, 1, _, DimList(Dim())
    ](
        cache_row_offsets_ref_device_ptr.unsafe_ptr(),
        Index(batch_size + 1),
    )

    # create reference output
    flare_mla_prefill[softmax_type = DType.float32](
        output_ref_device,
        q_device,
        k_device,
        v_device,
        cache_device,
        CausalMask(),
        IdentityScoreMod(),
        input_row_offsets_device,
        cache_row_offsets_ref_device,
        scale,
        ctx,
        q_max_seq_len=seq_len,
    )

    ctx.enqueue_copy(output_ref_ptr, output_ref_device_ptr)
    ctx.synchronize()

    # compare output with reference
    for s in range(batch_size * seq_len):
        for h in range(num_heads):
            for d in range(kv_depth):
                lhs = output[s, h, d]
                rhs = output_ref[s, h, d]
                assert_almost_equal(lhs, rhs, atol=2e-2, rtol=1e-3)

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = k_chunk_device_ptr
    _ = v_chunk_device_ptr
    _ = cache_device_ptr
    _ = output_device_ptr
    _ = output_ref_device_ptr
    _ = input_row_offsets_device_ptr
    _ = cache_row_offsets_device_ptr
    _ = cache_offsets_device_ptr
    _ = softmax_info_device_ptr

    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    k_chunk_ptr.free()
    v_chunk_ptr.free()
    cache_ptr.free()
    output_ptr.free()
    output_ref_ptr.free()
    input_row_offsets.free()
    cache_row_offsets.free()
    cache_offsets.free()


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


fn test_mla_prefill[
    batch_size: Int,
](ctx: DeviceContext) raises:
    test_prefill[
        DType.bfloat16,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](140, 140, ctx)

    test_prefill[
        DType.bfloat16,
        depth=192,
        num_heads=16,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](140, 140, ctx)


fn test_mla_cascade_prefill[
    batch_size: Int,
](ctx: DeviceContext) raises:
    test_cascade_prefill[
        DType.bfloat16,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
        chunk_size=128,
    ](287, 287, ctx)

    test_cascade_prefill[
        DType.bfloat16,
        depth=192,
        num_heads=16,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
        chunk_size=128,
    ](287, 555, ctx)


def main():
    with DeviceContext() as ctx:
        # tests with mask tensor
        test_decoding[27, 1, False, False](ctx, False)

        # tests with casual mask
        test_decoding[27, 1, False, True](ctx, False)

        # test mla prefill
        test_mla_prefill[2](ctx)

        # test mla cascade prefill
        test_mla_cascade_prefill[2](ctx)
