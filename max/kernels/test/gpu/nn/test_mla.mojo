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

from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from math import ceildiv, isclose
from random import randn
from sys import argv, has_nvidia_gpu_accelerator

from gpu import *
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from nn.mha import _naive_attention_with_transpose, mha_gpu_naive
from nn.mha_mask import CausalMask, MaterializedMask
from nn.mha_operand import LayoutTensorMHAOperand
from nn.mha_score_mod import IdentityScoreMod
from nn.mla import flare_mla_decoding, flare_mla_prefill
from tensor import IOUnknown, ManagedTensorSlice
from tensor.managed_tensor_slice import StaticTensorSpec
from testing import assert_almost_equal
from gpu.host.info import B200, GPUInfo


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
    num_partitions: Optional[Int] = None,
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

    comptime assert mask_rank in (3, 4), "mha only support rank 3 or 4."
    comptime assert (
        against_gpu_naive or mask_rank == 3
    ), "Testing against cpu requires mask of rank 3."

    # Query, key, value dimensions.
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group

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
                    q_ptr[(i * num_heads + h) * depth + j] = Scalar[qkv_type](
                        i * depth + j
                    )
        for i in range(num_keys):
            for h in range(kv_num_heads):
                for j in range(depth):
                    k_ptr[(i * kv_num_heads + h) * depth + j] = Scalar[
                        qkv_type
                    ](i * depth + j)

        comptime if mask_rank == 3:
            for i in range(seq_len):
                for j in range(num_keys):
                    mask_ptr[i * num_keys + j] = Scalar[mask_type](
                        (seq_len - i) * num_keys + num_keys - j
                    )
        else:
            for h in range(num_heads):
                var mask_head_ptr = mask_ptr + h * seq_len * num_keys
                for i in range(seq_len):
                    for j in range(num_keys):
                        mask_head_ptr[i * num_keys + j] = Scalar[mask_type](
                            (seq_len - i) * num_keys + num_keys - j
                        )

    else:
        randn[qkv_type](q_ptr, q_size)
        randn[qkv_type](k_ptr, k_size)
        randn[mask_type](mask_ptr, mask_size)

    # Construct buffers.
    comptime layout_4d = Layout.row_major[4]()
    var q = LayoutTensor[qkv_type, layout_4d](
        q_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )
    var k = LayoutTensor[qkv_type, layout_4d](
        k_ptr,
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
    var output = LayoutTensor[qkv_type, layout_4d](
        output_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    var flash_output = LayoutTensor[qkv_type, layout_4d](
        flash_output_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    comptime if not against_gpu_naive:
        comptime assert (
            qkv_type == mask_type
        ), "expect qkv and mask have same type for CPU."
        _naive_attention_with_transpose[qkv_type](
            output, q, k, k, mask.bitcast[qkv_type](), scale
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

    # Construct layout tensor buffers.
    comptime q_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var q_device = LayoutTensor[qkv_type, q_layout](
        q_device_ptr.unsafe_ptr(),
        RuntimeLayout[q_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )
    comptime k_layout = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, kv_num_heads, depth)
    )
    var k_device = LayoutTensor[qkv_type, k_layout](
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
    var output_device = LayoutTensor[qkv_type, output_layout](
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
        comptime if use_causal_mask:
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
        elif mask_rank == 3:
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
        else:
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

    comptime if against_gpu_naive:
        var output_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
        comptime output_ref_layout = Layout.row_major(
            Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
        )
        var output_ref_device = LayoutTensor[qkv_type, output_ref_layout](
            output_ref_device_ptr.unsafe_ptr(),
            RuntimeLayout[output_ref_layout].row_major(
                Index(batch_size, seq_len, num_heads, depth)
            ),
        )
        ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

        comptime if use_causal_mask:
            var k_operand = LayoutTensorMHAOperand(k_device)
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

    if o_size == 0:
        return

    # since we pass the whole K tensor as the V tensor to our naive mha kernel,
    # the last 64 elements of each head in the reference result are invalid.
    # b , s, h, d
    var rtol = 1e-3
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
                    if abs((actual - expect)) > 9e-2:
                        print(b, h, s, d, actual, expect)
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
    k_rope_type: DType,
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
        "k_rope_type:",
        k_rope_type,
        "depth:",
        depth,
        "kv_depth:",
        kv_depth,
        "cache_depth:",
        cache_depth,
        "cache_num_heads:",
        cache_num_heads,
    )

    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))

    var q_size = batch_size * seq_len * num_heads * depth
    var k_size = batch_size * num_keys * num_heads * kv_depth
    var v_size = k_size
    var o_size = batch_size * seq_len * num_heads * kv_depth
    var cache_size = batch_size * num_keys * cache_num_heads * cache_depth

    var q_ptr = UnsafePointer[Scalar[qkv_type]].alloc(q_size)
    var k_ptr = UnsafePointer[Scalar[qkv_type]].alloc(k_size)
    var v_ptr = UnsafePointer[Scalar[qkv_type]].alloc(v_size)
    var cache_ptr = UnsafePointer[Scalar[k_rope_type]].alloc(cache_size)
    var output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)

    # Q, K, V, cache are randomly initialized.
    randn[qkv_type](q_ptr, q_size)
    randn[qkv_type](k_ptr, k_size)
    randn[qkv_type](v_ptr, v_size)
    randn[k_rope_type](cache_ptr, cache_size)

    # input row offsets and cache row offsets
    var input_row_offsets = UnsafePointer[UInt32].alloc(batch_size + 1)
    var cache_row_offsets = UnsafePointer[UInt32].alloc(batch_size + 1)
    for i in range(batch_size):
        input_row_offsets[i] = UInt32(i * seq_len)
        cache_row_offsets[i] = UInt32(i * num_keys)
    input_row_offsets[batch_size] = UInt32(batch_size * seq_len)
    cache_row_offsets[batch_size] = UInt32(batch_size * num_keys)

    # ragged inputs
    var q = LayoutTensor[qkv_type, Layout.row_major[3]()](
        q_ptr,
        RuntimeLayout[Layout.row_major[3]()].row_major(
            Index(batch_size * seq_len, num_heads, depth)
        ),
    )
    var k = LayoutTensor[qkv_type, Layout.row_major[3]()](
        k_ptr,
        RuntimeLayout[Layout.row_major[3]()].row_major(
            Index(batch_size * num_keys, num_heads, kv_depth)
        ),
    )
    var v = LayoutTensor[qkv_type, Layout.row_major[3]()](
        v_ptr,
        RuntimeLayout[Layout.row_major[3]()].row_major(
            Index(batch_size * num_keys, num_heads, kv_depth)
        ),
    )
    var cache = LayoutTensor[k_rope_type, Layout.row_major[4]()](
        cache_ptr,
        RuntimeLayout[Layout.row_major[4]()].row_major(
            Index(batch_size, num_keys, cache_num_heads, cache_depth)
        ),
    )
    var output = LayoutTensor[qkv_type, Layout.row_major[3]()](
        output_ptr,
        RuntimeLayout[Layout.row_major[3]()].row_major(
            Index(batch_size * seq_len, num_heads, kv_depth)
        ),
    )

    # device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_device_ptr = ctx.enqueue_create_buffer[qkv_type](v_size)
    var cache_device_ptr = ctx.enqueue_create_buffer[k_rope_type](cache_size)
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
    comptime q_layout = Layout.row_major(UNKNOWN_VALUE, num_heads, depth)
    var q_device = LayoutTensor[qkv_type, q_layout](
        q_device_ptr.unsafe_ptr(),
        RuntimeLayout[q_layout].row_major(
            Index(batch_size * seq_len, num_heads, depth)
        ),
    )
    comptime k_layout = Layout.row_major(UNKNOWN_VALUE, num_heads, kv_depth)
    var k_device = LayoutTensor[qkv_type, k_layout](
        k_device_ptr.unsafe_ptr(),
        RuntimeLayout[k_layout].row_major(
            Index(batch_size * num_keys, num_heads, kv_depth)
        ),
    )
    comptime v_layout = Layout.row_major(UNKNOWN_VALUE, num_heads, kv_depth)
    var v_device = LayoutTensor[qkv_type, v_layout](
        v_device_ptr.unsafe_ptr(),
        RuntimeLayout[v_layout].row_major(
            Index(batch_size * num_keys, num_heads, kv_depth)
        ),
    )
    comptime cache_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, cache_num_heads, cache_depth
    )
    var cache_device = LayoutTensor[k_rope_type, cache_layout](
        cache_device_ptr.unsafe_ptr(),
        RuntimeLayout[cache_layout].row_major(
            Index(batch_size, num_keys, cache_num_heads, cache_depth)
        ),
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_heads, kv_depth
    )
    var output_device = LayoutTensor[qkv_type, output_layout](
        output_device_ptr.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            Index(batch_size * seq_len, num_heads, kv_depth)
        ),
    )
    var input_row_offsets_device = LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE)
    ](
        input_row_offsets_device_ptr.unsafe_ptr(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            Index(batch_size + 1),
        ),
    )
    var cache_row_offsets_device = LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE)
    ](
        cache_row_offsets_device_ptr.unsafe_ptr(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            Index(batch_size + 1),
        ),
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
        flare_mla_prefill[rank = q.rank](
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
        comptime nrun = 200

        # Warmup
        for i in range(20):
            kernel_launch(ctx)

        var nstime = Float64(ctx.execution_time[kernel_launch](nrun)) / Float64(
            nrun
        )
        var sectime = nstime / 1000000

        var tflops = (
            Float64(2)
            * Float64(batch_size)
            * Float64(num_heads)
            * Float64(-seq_len * seq_len + 2 * seq_len * num_keys)
            * Float64(depth + kv_depth)
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
    var k_ref = LayoutTensor[qkv_type, Layout.row_major[4]()](
        k_ref_ptr,
        RuntimeLayout[Layout.row_major[4]()].row_major(
            Index(batch_size, num_keys, num_heads, depth)
        ),
    )
    var v_ref = LayoutTensor[qkv_type, Layout.row_major[4]()](
        v_ref_ptr,
        RuntimeLayout[Layout.row_major[4]()].row_major(
            Index(batch_size, num_keys, num_heads, depth)
        ),
    )
    var output_ref = LayoutTensor[qkv_type, Layout.row_major[4]()](
        output_ref_ptr,
        RuntimeLayout[Layout.row_major[4]()].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
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
                    k_ref[b, s, h, d + kv_depth] = Scalar[qkv_type](
                        cache[b, s, 0, cache_depth - (depth - kv_depth) + d][0]
                    )
                    v_ref[b, s, h, d + kv_depth] = 0

    # view q_device as a rank 4 buffer
    comptime q_layout_4d = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var q_device_rank4 = LayoutTensor[qkv_type, q_layout_4d](
        q_device_ptr.unsafe_ptr(),
        RuntimeLayout[q_layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
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
    comptime k_layout_4d = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var k_ref_device = LayoutTensor[qkv_type, k_layout_4d](
        k_ref_device_ptr.unsafe_ptr(),
        RuntimeLayout[k_layout_4d].row_major(
            Index(batch_size, num_keys, num_heads, depth)
        ),
    )
    comptime v_layout_4d = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var v_ref_device = LayoutTensor[qkv_type, v_layout_4d](
        v_ref_device_ptr.unsafe_ptr(),
        RuntimeLayout[v_layout_4d].row_major(
            Index(batch_size, num_keys, num_heads, depth)
        ),
    )
    comptime output_layout_4d = Layout.row_major(
        Index(UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth)
    )
    var output_ref_device = LayoutTensor[qkv_type, output_layout_4d](
        output_ref_device_ptr.unsafe_ptr(),
        RuntimeLayout[output_layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    # copy from host to device
    ctx.enqueue_copy(k_ref_device_ptr, k_ref_ptr)
    ctx.enqueue_copy(v_ref_device_ptr, v_ref_ptr)

    var null_valid_length = LayoutTensor[
        DType.uint32, Layout.row_major(UNKNOWN_VALUE)
    ](
        UnsafePointer[UInt32](),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(Index(0)),
    )

    var k_ref_operand = LayoutTensorMHAOperand(k_ref_device)
    var v_ref_operand = LayoutTensorMHAOperand(v_ref_device)

    # create reference output
    mha_gpu_naive[_is_cache_length_accurate=True](
        q_device_rank4,
        k_ref_operand,
        v_ref_operand,
        CausalMask(),
        output_ref_device,
        null_valid_length,
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
    var output_rank4 = LayoutTensor[qkv_type, Layout.row_major[4]()](
        output_ptr,
        RuntimeLayout[Layout.row_major[4]()].row_major(
            Index(batch_size, seq_len, num_heads, kv_depth)
        ),
    )

    # compare output with reference
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(kv_depth):
                    lhs = output_rank4[b, s, h, d]
                    rhs = output_ref[b, s, h, d]
                    if abs((lhs - rhs)) > 2e-2:
                        print(b, s, h, d, lhs, rhs)
                    # print(b, s, h, d, lhs, rhs)
                    assert_almost_equal(
                        lhs,
                        rhs,
                        atol=2e-2,
                        rtol=2e-2 if has_nvidia_gpu_accelerator() else 3e-2,
                    )

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


fn test_decoding[
    batch_size: Int,
    num_partitions: Optional[Int],
    split_k: Bool,
    use_causal_mask: Bool = True,
    qkv_type: DType = DType.bfloat16,
](ctx: DeviceContext, use_index_input: Bool) raises:
    comptime if ctx.default_device_info == B200:
        if batch_size <= 2:
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
            ](1, 32768, ctx, use_index_input=use_index_input)
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
            ](1, 32768 * 2, ctx, use_index_input=use_index_input)
        else:
            for seq_len in range(1, 9):
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
                ](seq_len, 50, ctx, use_index_input=use_index_input)

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
                ](seq_len, 50, ctx, use_index_input=use_index_input)

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
            ](1, 4096, ctx, use_index_input=use_index_input)

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
            ](2, 4096, ctx, use_index_input=use_index_input)

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
            ](3, 1024, ctx, use_index_input=use_index_input)

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
            ](4, 1024, ctx, use_index_input=use_index_input)

    else:  # H100 AND AMD
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
        ](1, 4096, ctx, use_index_input=use_index_input)
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
        ](1, 2048, ctx, use_index_input=use_index_input)


fn test_mla_prefill[
    batch_size: Int,
    qkv_type: DType,
    k_rope_type: DType,
](ctx: DeviceContext) raises:
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](120, 120, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=16,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](1179, 1179, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](700, 700, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](701, 701, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](12, 12, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](350, 700, ctx)
    test_prefill[
        qkv_type,
        k_rope_type,
        depth=192,
        num_heads=128,
        kv_depth=128,
        cache_depth=576,
        cache_num_heads=1,
        batch_size=batch_size,
    ](120, 240, ctx)


def main():
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator():
            # tests with mask tensor
            test_decoding[27, 1, False, False](ctx, False)
            test_decoding[128, 1, False, False](ctx, False)
            test_decoding[1, 1, False, True](ctx, False)
            test_decoding[0, 1, False, False](ctx, False)

        # tests with causal mask
        test_decoding[27, 1, False, True](ctx, False)
        test_decoding[128, 1, False, True](ctx, False)
        test_decoding[0, 1, False, True](ctx, False)

        # test mla prefill
        test_mla_prefill[2, DType.bfloat16, DType.bfloat16](ctx)
        test_mla_prefill[4, DType.bfloat16, DType.bfloat16](ctx)
        test_mla_prefill[0, DType.bfloat16, DType.bfloat16](ctx)

        comptime if ctx.default_device_info == B200:
            test_mla_prefill[2, DType.bfloat16, DType.float8_e4m3fn](ctx)
            test_mla_prefill[4, DType.bfloat16, DType.float8_e4m3fn](ctx)
            test_mla_prefill[0, DType.bfloat16, DType.float8_e4m3fn](ctx)
