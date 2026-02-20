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
from math import exp
from random import rand, random_float64, seed
from sys import argv, has_amd_gpu_accelerator

from gpu import *
from gpu.host import DeviceContext
from gpu.host.info import A100, B200, H100, GPUInfo, Vendor
from layout import LayoutTensor, Layout, RuntimeLayout, UNKNOWN_VALUE
from nn.mha import (
    _naive_attention_with_transpose,
    flash_attention,
    mha_gpu_naive,
)
from nn.mha_mask import MaterializedMask, NullMask
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal

from utils.index import Index


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


fn is_sm8(info: GPUInfo) -> Bool:
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
    num_partitions: Optional[Int] = None,
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
        "num_heads:",
        num_heads,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
        "group:",
        group,
        "qkv_type:",
        qkv_type,
        "mask_type:",
        mask_type,
        "mask_rank:",
        mask_rank,
        "depth:",
        depth,
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
                    q_ptr[(i * num_heads + h) * depth + j] = Scalar[qkv_type](
                        i * depth + j
                    )
        for i in range(num_keys):
            for h in range(kv_num_heads):
                for j in range(depth):
                    k_ptr[(i * kv_num_heads + h) * depth + j] = Scalar[
                        qkv_type
                    ](i * depth + j)
        for i in range(num_keys):
            for h in range(kv_num_heads):
                for j in range(depth):
                    v_ptr[(i * kv_num_heads + h) * depth + j] = Scalar[
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
        seed(1234567890)
        rand[qkv_type](q_ptr, q_size)
        rand[qkv_type](k_ptr, k_size)
        rand[qkv_type](v_ptr, v_size)
        rand[mask_type](mask_ptr, mask_size)

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
    var v = LayoutTensor[qkv_type, layout_4d](
        v_ptr,
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
            output, q, k, v, mask.bitcast[qkv_type](), scale
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

    # Construct device buffers.
    comptime q_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth
    )
    var q_device = LayoutTensor[qkv_type, q_layout](
        q_device_ptr.unsafe_ptr(),
        RuntimeLayout[q_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )
    comptime k_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, kv_num_heads, depth
    )
    var k_device = LayoutTensor[qkv_type, k_layout](
        k_device_ptr.unsafe_ptr(),
        RuntimeLayout[k_layout].row_major(
            Index(batch_size, num_keys, kv_num_heads, depth)
        ),
    )
    comptime v_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, kv_num_heads, depth
    )
    var v_device = LayoutTensor[qkv_type, v_layout](
        v_device_ptr.unsafe_ptr(),
        RuntimeLayout[v_layout].row_major(
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
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth
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
    @__copy_capture(q_device, k_device, v_device, mask3d, mask4d, output_device)
    fn kernel_launch(ctx: DeviceContext) raises:
        comptime if mask_rank == 3:
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
        comptime nrun = 50

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
            UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth
        )
        var output_ref_device = LayoutTensor[qkv_type, output_ref_layout](
            output_ref_device_ptr.unsafe_ptr(),
            RuntimeLayout[output_ref_layout].row_major(
                Index(batch_size, seq_len, num_heads, depth)
            ),
        )
        ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

        comptime if mask_rank == 3:
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
        return 2e-2 if num_partitions and num_partitions.value() >= 4 else 1e-2

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


fn test_depth_supported_by_gpu(info: GPUInfo) -> List[Int]:
    var depths = [64, 128]

    if info == materialize[H100]() or info == materialize[B200]():
        depths.append(80)
    return depths^


fn test_context_encoding(ctx: DeviceContext) raises:
    # fp32 arbitrary depth and num_heads, baseline impl.
    test[3, DType.float32, DType.float32, depth=127, num_heads=2](111, 121, ctx)

    comptime depths = test_depth_supported_by_gpu(ctx.default_device_info)

    comptime for d in range(len(depths)):
        comptime depth = depths[d]
        # fp32 depth == 128, tf32-fp32 mma, llama2 shape.
        test[
            4,
            DType.float32,
            DType.float32,
            depth=depth,
            num_heads=32,
            against_gpu_naive=True,
        ](1024, 1024, ctx, is_benchmark())
        test[
            3,
            DType.float32,
            DType.float32,
            depth=depth,
            num_heads=3,
            against_gpu_naive=True,
        ](14, 14, ctx, is_benchmark())
        test[
            3,
            DType.float32,
            DType.float32,
            depth=depth,
            num_heads=1,
            against_gpu_naive=True,
        ](178, 178, ctx, is_benchmark())
        # bf16 depth == 128, bf16-fp32 mma
        test[
            4,
            DType.bfloat16,
            DType.bfloat16,
            depth=depth,
            num_heads=1,
            against_gpu_naive=True,
        ](128, 128, ctx)
        test[
            4,
            DType.bfloat16,
            DType.float32,
            depth=depth,
            num_heads=1,
            against_gpu_naive=True,
        ](384, 384, ctx)
        test[
            3,
            DType.bfloat16,
            DType.float32,
            depth=depth,
            num_heads=3,
            against_gpu_naive=True,
        ](256, 256, ctx)
        test[
            4,
            DType.bfloat16,
            DType.float32,
            depth=depth,
            num_heads=32,
            against_gpu_naive=True,
        ](1024, 1024, ctx, is_benchmark())
        test[
            4,
            DType.bfloat16,
            DType.float32,
            depth=depth,
            num_heads=24,
            group=3,
            against_gpu_naive=True,
        ](1024, 1024, ctx)
        # BF16 with sequence length not multiple of 128
        test[
            4,
            DType.bfloat16,
            DType.float32,
            depth=depth,
            num_heads=3,
            group=3,
            against_gpu_naive=True,
        ](64, 64, ctx)
        test[
            4,
            DType.bfloat16,
            DType.bfloat16,
            depth=depth,
            num_heads=3,
            group=3,
            against_gpu_naive=True,
        ](102, 102, ctx)
        test[
            3,
            DType.bfloat16,
            DType.float32,
            depth=depth,
            num_heads=1,
            against_gpu_naive=True,
        ](14, 14, ctx)
        test[
            3,
            DType.bfloat16,
            DType.bfloat16,
            depth=depth,
            num_heads=1,
            against_gpu_naive=True,
        ](528, 528, ctx)

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
            depth=128,
            num_heads=3,
            against_gpu_naive=True,
        ](256, 128, ctx)

        test[
            3,
            DType.bfloat16,
            DType.float32,
            depth=128,
            num_heads=24,
            group=3,
            against_gpu_naive=True,
        ](1024, 100, ctx)

        test[
            4,
            DType.float32,
            DType.float32,
            depth=128,
            num_heads=24,
            group=3,
            against_gpu_naive=True,
        ](214, 300, ctx)

        test[
            3,
            DType.bfloat16,
            DType.float32,
            depth=128,
            num_heads=24,
            group=1,
            against_gpu_naive=True,
        ](512, 1024, ctx)

        test[
            3,
            DType.float32,
            DType.float32,
            depth=128,
            num_heads=32,
            group=3,
            against_gpu_naive=True,
        ](12, 8, ctx)

        test[
            4,
            DType.bfloat16,
            DType.float32,
            depth=128,
            num_heads=3,
            against_gpu_naive=True,
        ](14, 18, ctx)

        # odd seq_len
        test[
            4,
            DType.bfloat16,
            DType.float32,
            depth=128,
            num_heads=3,
            against_gpu_naive=True,
        ](15, 18, ctx)
        test[
            3,
            DType.bfloat16,
            DType.float32,
            depth=128,
            num_heads=3,
            against_gpu_naive=True,
        ](119, 200, ctx)


fn test_decoding[
    batch_size: Int,
    num_partitions: Optional[Int],
    split_k: Bool,
    qkv_type: DType = DType.bfloat16,
](ctx: DeviceContext, use_index_input: Bool) raises:
    comptime depths = test_depth_supported_by_gpu(ctx.default_device_info)

    comptime for d in range(len(depths)):
        comptime depth = depths[d]
        test[
            3,
            qkv_type,
            DType.float32,
            depth=depth,
            num_heads=1,
            against_gpu_naive=True,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
        ](1, 11, ctx, use_index_input=use_index_input)

        comptime if (
            not is_sm8(ctx.default_device_info)
            or num_partitions
            and num_partitions.value() == 1
        ):
            test[
                4,
                qkv_type,
                DType.bfloat16,
                depth=depth,
                num_heads=2,
                against_gpu_naive=True,
                batch_size=batch_size,
                num_partitions=num_partitions,
                decoding_warp_split_k=split_k,
            ](1, 523, ctx, use_index_input=use_index_input)
        test[
            4,
            qkv_type,
            DType.float32,
            depth=depth,
            num_heads=24,
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
    #     depth=128,
    #     num_heads=3,
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
    #     depth=128,
    #     num_heads=3,
    #     group=3,
    #     against_gpu_naive=True,
    #     batch_size=batch_size,
    #     num_partitions=num_partitions,
    #     decoding_warp_split_k=split_k,
    # ](1, 208, ctx, use_index_input=use_index_input)


fn test_decoding_large_group[
    batch_size: Int,
    num_partitions: Optional[Int] = None,
    split_k: Bool = False,
    qkv_type: DType = DType.bfloat16,
](ctx: DeviceContext, use_index_input: Bool = False) raises:
    comptime depths = test_depth_supported_by_gpu(ctx.default_device_info)

    comptime for d in range(len(depths)):
        comptime depth = depths[d]
        test[
            4,
            qkv_type,
            DType.float32,
            depth=depth,
            num_heads=32,
            group=16,
            against_gpu_naive=True,
            batch_size=batch_size,
            num_partitions=num_partitions,
            decoding_warp_split_k=split_k,
        ](1, 2000, ctx, use_index_input=use_index_input)


fn test_flash_attention_sink_kernel(ctx: DeviceContext, seq_len: Int) raises:
    print("test_flash_attention_sink_kernel")
    comptime batch_size = 1
    comptime num_heads = 2
    comptime kv_heads = num_heads
    comptime num_keys = 64
    comptime depth = 128
    comptime qkv_type = DType.bfloat16  # fast path on A100/H100
    comptime mask_type = DType.float32
    comptime scale = Float32(0.0)  # force QK logits to exactly 0

    var q_ptr = UnsafePointer[Scalar[qkv_type]].alloc(
        batch_size * seq_len * num_heads * depth
    )
    var k_ptr = UnsafePointer[Scalar[qkv_type]].alloc(
        batch_size * num_keys * kv_heads * depth
    )
    var v_ptr = UnsafePointer[Scalar[qkv_type]].alloc(
        batch_size * num_keys * kv_heads * depth
    )
    var mask_ptr = UnsafePointer[Scalar[mask_type]].alloc(
        batch_size * seq_len * num_keys
    )
    var out_ptr = UnsafePointer[Scalar[qkv_type]].alloc(
        batch_size * seq_len * num_heads * depth
    )
    var sinks_ptr = UnsafePointer[Scalar[qkv_type]].alloc(num_heads)

    # Q,K don't matter when scale=0, but set deterministically
    for i in range(batch_size * seq_len * num_heads * depth):
        q_ptr[i] = Float32(0.123).cast[qkv_type]()
    for i in range(batch_size * num_keys * kv_heads * depth):
        k_ptr[i] = Float32(-0.456).cast[qkv_type]()

    # V = 1 so the attention output equals total probability mass assigned to
    # the real keys
    for i in range(batch_size * num_keys * kv_heads * depth):
        v_ptr[i] = Float32(1.0).cast[qkv_type]()

    # No masking
    for i in range(batch_size * seq_len * num_keys):
        mask_ptr[i] = 0.0

    # Two different sinks for the two heads
    var sink_h0 = Float32(5.0)  # large positive
    var sink_h1 = Float32(3.0)  # moderately positive
    sinks_ptr[0] = sink_h0.cast[qkv_type]()
    sinks_ptr[1] = sink_h1.cast[qkv_type]()

    var q_host = LayoutTensor[qkv_type, Layout.row_major[4]()](
        q_ptr,
        RuntimeLayout[Layout.row_major[4]()].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )
    var k_host = LayoutTensor[qkv_type, Layout.row_major[4]()](
        k_ptr,
        RuntimeLayout[Layout.row_major[4]()].row_major(
            Index(batch_size, num_keys, kv_heads, depth)
        ),
    )
    var v_host = LayoutTensor[qkv_type, Layout.row_major[4]()](
        v_ptr,
        RuntimeLayout[Layout.row_major[4]()].row_major(
            Index(batch_size, num_keys, kv_heads, depth)
        ),
    )
    var m3_host = LayoutTensor[mask_type, Layout.row_major[3]()](
        mask_ptr,
        RuntimeLayout[Layout.row_major[3]()].row_major(
            Index(batch_size, seq_len, num_keys)
        ),
    )
    var out_host = LayoutTensor[qkv_type, Layout.row_major[4]()](
        out_ptr,
        RuntimeLayout[Layout.row_major[4]()].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )
    var sinks_host = LayoutTensor[qkv_type, Layout.row_major(UNKNOWN_VALUE)](
        sinks_ptr,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            Index(num_heads)
        ),
    )

    var q_dev = ctx.enqueue_create_buffer[qkv_type](q_host.size())
    var k_dev = ctx.enqueue_create_buffer[qkv_type](k_host.size())
    var v_dev = ctx.enqueue_create_buffer[qkv_type](v_host.size())
    var m_dev = ctx.enqueue_create_buffer[mask_type](m3_host.size())
    var out_dev = ctx.enqueue_create_buffer[qkv_type](out_host.size())
    var sinks_dev = ctx.enqueue_create_buffer[qkv_type](sinks_host.size())

    ctx.enqueue_copy(q_dev, q_ptr)
    ctx.enqueue_copy(k_dev, k_ptr)
    ctx.enqueue_copy(v_dev, v_ptr)
    ctx.enqueue_copy(m_dev, mask_ptr)
    ctx.enqueue_copy(sinks_dev, sinks_ptr)

    comptime q_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth
    )
    var q_device = LayoutTensor[qkv_type, q_layout](
        q_dev.unsafe_ptr(),
        RuntimeLayout[q_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )
    comptime k_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, kv_heads, depth
    )
    var k_device = LayoutTensor[qkv_type, k_layout](
        k_dev.unsafe_ptr(),
        RuntimeLayout[k_layout].row_major(
            Index(batch_size, num_keys, kv_heads, depth)
        ),
    )
    comptime v_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, kv_heads, depth
    )
    var v_device = LayoutTensor[qkv_type, v_layout](
        v_dev.unsafe_ptr(),
        RuntimeLayout[v_layout].row_major(
            Index(batch_size, num_keys, kv_heads, depth)
        ),
    )
    var mask3d = LayoutTensor[mask_type, Layout.row_major[3]()](
        m_dev.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[3]()].row_major(
            Index(batch_size, seq_len, num_keys)
        ),
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth
    )
    var out_device = LayoutTensor[qkv_type, output_layout](
        out_dev.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )
    comptime sinks_layout = Layout.row_major(UNKNOWN_VALUE)
    var sinks_device = LayoutTensor[qkv_type, sinks_layout](
        sinks_dev.unsafe_ptr(),
        RuntimeLayout[sinks_layout].row_major(Index(num_heads)),
    )

    @always_inline
    fn launch(ctx: DeviceContext) raises:
        flash_attention[sink=True](
            out_device,
            q_device,
            k_device,
            v_device,
            NullMask(),
            IdentityScoreMod(),
            scale,  # 0.0 -> all QK logits are exactly zero
            ctx,
            None,
            sink_weights=sinks_device.get_immutable(),
        )

    launch(ctx)
    ctx.synchronize()
    ctx.enqueue_copy(out_ptr, out_dev)

    fn expected_mass(sink: Float32) -> Float32:
        return Float32(num_keys) / (Float32(num_keys) + exp(sink))

    var want0 = expected_mass(sink_h0)
    var want1 = expected_mass(sink_h1)

    # Every element of the output vector for a given head should equal that mass
    # (since V=1)
    for s in range(seq_len):
        for d in range(depth):
            var got0 = out_host[0, s, 0, d].cast[DType.float32]()
            var got1 = out_host[0, s, 1, d].cast[DType.float32]()
            assert_almost_equal(got0, want0, atol=2e-2, rtol=2e-2)
            assert_almost_equal(got1, want1, atol=2e-2, rtol=2e-2)

    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    mask_ptr.free()
    out_ptr.free()
    sinks_ptr.free()


def main():
    with DeviceContext() as ctx:
        test_context_encoding(ctx)
        # Test flash attention with sink kernel during encoding
        test_flash_attention_sink_kernel(ctx, 8)
        # Test flash attention with sink kernel during decoding
        test_flash_attention_sink_kernel(ctx, 1)

        comptime for split_k in range(1):
            comptime for batch_size in range(1, 5, 3):
                test_decoding[batch_size, 1, Bool(split_k)](ctx, False)

                comptime if not split_k:
                    test_decoding[batch_size, 1, Bool(split_k), DType.float32](
                        ctx, False
                    )

                comptime if (
                    ctx.default_device_info == A100
                    or ctx.default_device_info == H100
                ):
                    test_decoding_large_group[batch_size, 1](ctx)

                test_decoding[batch_size, 2, Bool(split_k)](ctx, False)
                test_decoding[batch_size, 4, Bool(split_k)](ctx, False)

                comptime if not split_k:
                    test_decoding[batch_size, 4, Bool(split_k), DType.float32](
                        ctx, False
                    )
                test_decoding[batch_size, None, Bool(split_k)](ctx, False)
                test_decoding[batch_size, 32, Bool(split_k)](ctx, False)
