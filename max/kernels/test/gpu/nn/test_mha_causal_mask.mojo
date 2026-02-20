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
from math import isclose
from random import rand
from sys import argv, size_of

from bit import count_trailing_zeros
from gpu import *
from gpu.host import DeviceContext
from gpu.host.info import A100, B200, H100
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from nn.mha import flash_attention, mha_gpu_naive
from nn.mha_mask import CausalMask, MHAMask, SlidingWindowCausalMask
from nn.mha_score_mod import IdentityScoreMod
from nn.mha_utils import FlashAttentionAlgorithm, MHAConfig
from testing import assert_almost_equal, assert_equal

from utils.index import Index
from utils.numerics import min_or_neg_inf


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


fn test[
    MaskType: MHAMask,
    //,
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    *,
    group: Int = 1,
](
    seq_len: Int,
    num_keys: Int,
    mask: MaskType,
    ctx: DeviceContext,
    *,
    is_benchmark: Bool = False,
    num_partitions: Optional[Int] = None,
) raises:
    print("test_mha_causal_mask")
    print(
        "qkv_type:",
        qkv_type,
        "depth:",
        depth,
        "num_heads:",
        num_heads,
        "group:",
        group,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
        "mask:",
        MaskType.name(),
    )
    # Query, key, value dimensions.
    comptime batch_size = 1
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size

    # Allocate memory for all variables.
    var q_ptr = UnsafePointer[Scalar[qkv_type]].alloc(q_size)
    var k_ptr = UnsafePointer[Scalar[qkv_type]].alloc(k_size)
    var v_ptr = UnsafePointer[Scalar[qkv_type]].alloc(v_size)
    var output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)
    var flash_output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)

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
    var output = LayoutTensor[qkv_type, layout_4d](
        output_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    # Q, K, V are randomly initialized.
    rand[qkv_type](q_ptr, q_size)
    rand[qkv_type](k_ptr, k_size)
    rand[qkv_type](v_ptr, v_size)

    # Device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_device_ptr = ctx.enqueue_create_buffer[qkv_type](v_size)
    var output_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)

    # Copy from host to device
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(v_device_ptr, v_ptr)

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
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth
    )
    var output_device = LayoutTensor[qkv_type, output_layout](
        output_device_ptr.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    comptime config = MHAConfig[qkv_type](
        UInt(num_heads),
        UInt(depth),
        BK=Optional[UInt](UInt(128 // size_of[qkv_type]())),
        num_pipeline_stages=UInt(4) if (
            ctx.default_device_info == H100 or ctx.default_device_info == B200
        ) else 2,
    )

    @parameter
    @always_inline
    @__copy_capture(q_device, k_device, v_device, output_device)
    fn kernel_launch(ctx: DeviceContext) raises:
        flash_attention[config=config](
            output_device,
            q_device,
            k_device,
            v_device,
            mask,
            IdentityScoreMod(),
            scale,
            ctx,
            num_partitions=num_partitions,
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

    var output_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
    ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

    var output_device_ref = LayoutTensor[qkv_type, output_layout](
        output_ref_device_ptr.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    comptime config_baseline = MHAConfig[qkv_type](
        UInt(num_heads),
        UInt(depth),
        BK=Optional[UInt](UInt(128 // size_of[qkv_type]())),
        num_pipeline_stages=2,
        algorithm=FlashAttentionAlgorithm(2),
    )
    mha_gpu_naive(
        q_device,
        k_device,
        v_device,
        mask,
        output_device_ref,
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

    var rtol = 1e-2
    for s in range(seq_len):
        for h in range(num_heads):
            for d in range(depth):
                var expect = output_ptr.load(
                    d + depth * (h + s * num_heads)
                ).cast[DType.float64]()
                var actual = flash_output_ptr.load(
                    d + depth * (h + s * num_heads)
                ).cast[DType.float64]()
                if not isclose(actual, expect, atol=1e-5, rtol=rtol):
                    var next_expect = 0 * expect
                    var next_actual = 0 * actual
                    if h < num_heads and s < seq_len and d < depth - 1:
                        next_expect = output_ptr.load(
                            d + depth * (h + s * num_heads) + 1
                        ).cast[DType.float64]()
                        next_actual = flash_output_ptr.load(
                            d + depth * (h + s * num_heads) + 1
                        ).cast[DType.float64]()
                    var rerr = abs((actual - expect) / expect)
                    print(
                        "s, h, d = ",
                        "(" + String(s),
                        h,
                        String(d) + ")",
                        "actual =",
                        actual,
                        "expect =",
                        expect,
                        "rerr =",
                        rerr,
                        "next_expect =",
                        next_expect,
                        "next_actual =",
                        next_actual,
                    )

                assert_almost_equal(actual, expect, atol=1e-5, rtol=rtol)

    for repeat in range(16):
        # test reproducibility
        flash_attention[config=config](
            output_device_ref,
            q_device,
            k_device,
            v_device,
            mask,
            IdentityScoreMod(),
            scale,
            ctx,
            num_partitions=num_partitions,
        )
        ctx.enqueue_copy(output_ptr, output_ref_device_ptr)
        ctx.synchronize()
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(depth):
                    orig = flash_output_ptr.load(
                        d + depth * (h + s * num_heads)
                    )
                    rep = output_ptr.load(d + depth * (h + s * num_heads))
                    if rep != orig:
                        print("repeat s h d =", repeat, s, h, d)
                    assert_equal(rep, orig)
                    output_ptr.store(d + depth * (h + s * num_heads), 123.4567)

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = output_device_ptr
    _ = output_ref_device_ptr

    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    output_ptr.free()
    flash_output_ptr.free()


fn construct_depths(is_sm90orsm100: Bool) -> List[Int]:
    var depths = [64, 128]
    if is_sm90orsm100:
        depths.append(72)
        depths.append(80)
        depths.append(96)
        depths.append(256)
    return depths^


def main():
    with DeviceContext() as ctx:
        comptime is_sm90orsm100 = ctx.default_device_info == H100 or ctx.default_device_info == B200
        comptime depths = construct_depths(is_sm90orsm100)

        comptime for d in range(len(depths)):
            comptime depth = depths[d]

            comptime if depth <= 128:
                # fp32 tf32-fp32 mma
                test[DType.float32, depth, 1](
                    128, 128, CausalMask(), ctx, is_benchmark=is_benchmark()
                )

                test[
                    DType.float32,
                    depth,
                    3,
                ](14, 14, CausalMask(), ctx, is_benchmark=is_benchmark())

                test[
                    DType.float32,
                    depth,
                    1,
                ](178, 178, CausalMask(), ctx, is_benchmark=is_benchmark())

            # bf16 depth == 128, bf16-fp32 mma
            test[
                DType.bfloat16,
                depth=depth,
                num_heads=1,
            ](128, 128, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth=depth,
                num_heads=1,
            ](384, 384, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                24,
                group=3,
            ](1024, 1024, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                24,
                group=3,
            ](1024, 1024, SlidingWindowCausalMask[128](), ctx)

            # BF16 with sequence length not multiple of 128
            test[
                DType.bfloat16,
                depth,
                3,
                group=3,
            ](64, 64, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                3,
                group=3,
            ](102, 102, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                1,
            ](14, 14, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                1,
            ](528, 528, CausalMask(), ctx)

            # BF16 with different length for prompt and cache.
            test[
                DType.bfloat16,
                depth,
                1,
            ](128, 256, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                3,
                group=3,
            ](32, 77, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                16,
                group=8,
            ](201, 400, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                12,
                group=4,
            ](1000, 2000, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                12,
                group=4,
            ](1000, 2000, SlidingWindowCausalMask[100](), ctx)

            test[
                DType.bfloat16,
                depth,
                32,
                group=4,
            ](201, 600, CausalMask(), ctx)

            # BF16 token gen

            test[
                DType.bfloat16,
                depth,
                32,
            ](1, 512, CausalMask(), ctx, is_benchmark=is_benchmark())

            test[
                DType.bfloat16,
                depth,
                11,
            ](1, 256, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                1,
            ](1, 11, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                1,
            ](1, 11, CausalMask(), ctx, num_partitions=Optional[Int](2))

            test[
                DType.bfloat16,
                depth,
                2,
            ](1, 523, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                24,
                group=3,
            ](1, 29, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                3,
                group=3,
            ](1, 156, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                3,
                group=3,
            ](1, 208, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                32,
                group=4,
            ](1, 1208, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                32,
                group=4,
            ](1, 2008, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                32,
                group=4,
            ](1, 2008, SlidingWindowCausalMask[77](), ctx)

            test[
                DType.bfloat16,
                depth,
                32,
                group=4,
            ](1, 5000, CausalMask(), ctx)

            test[
                DType.bfloat16,
                depth,
                32,
                group=4,
            ](1, 5000, SlidingWindowCausalMask[89](), ctx)

            test[
                DType.bfloat16,
                depth,
                32,
                group=4,
            ](1, 600, CausalMask(), ctx)

            comptime if ctx.default_device_info == A100 or is_sm90orsm100:
                test[
                    DType.bfloat16,
                    depth,
                    32,
                    group=16,
                ](1, 2008, CausalMask(), ctx)
