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
from math import isclose
from random import rand
from sys import argv, size_of

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import *
from gpu.host import DeviceContext
from gpu.host.info import A100, H100, B200
from nn.mha import flash_attention
from nn.mha_mask import CausalMask, MaterializedMask
from nn.mha_score_mod import IdentityScoreMod
from nn.mha_utils import MHAConfig, FlashAttentionAlgorithm
from testing import assert_almost_equal

from bit import count_trailing_zeros

from utils.index import Index
from utils.numerics import min_or_neg_inf


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


fn test[
    qkv_type: DType,
    mask_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
](
    seq_len: Int,
    num_keys: Int,
    ctx: DeviceContext,
    is_benchmark: Bool = False,
    num_partitions: OptionalReg[Int] = None,
) raises:
    print("test_mha_causal_mask")
    print(
        "qkv_type:",
        qkv_type,
        "mask_type:",
        mask_type,
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
    )
    # Query, key, value dimensions.
    alias batch_size = 1
    alias scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    alias kv_num_heads = num_heads // group

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size
    var mask_size = num_heads * seq_len * num_keys

    # Allocate memory for all variables.
    var q_ptr = UnsafePointer[Scalar[qkv_type]].alloc(q_size)
    var k_ptr = UnsafePointer[Scalar[qkv_type]].alloc(k_size)
    var v_ptr = UnsafePointer[Scalar[qkv_type]].alloc(v_size)
    var mask_ptr = UnsafePointer[Scalar[mask_type]].alloc(mask_size)
    var output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)
    var flash_output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)

    # Construct buffers.
    var q = NDBuffer[qkv_type, 4](
        q_ptr, Index(batch_size, seq_len, num_heads, depth)
    )
    var k = NDBuffer[qkv_type, 4](
        k_ptr, Index(batch_size, num_keys, kv_num_heads, depth)
    )
    var v = NDBuffer[qkv_type, 4](
        v_ptr, Index(batch_size, num_keys, kv_num_heads, depth)
    )
    var mask = NDBuffer[mask_type, 4](
        mask_ptr, Index(batch_size, num_heads, seq_len, num_keys)
    )
    var output = NDBuffer[qkv_type, 4](
        output_ptr, Index(batch_size, seq_len, num_heads, depth)
    )

    # Q, K, V are randomly initialized.
    rand[qkv_type](q_ptr, q_size)
    rand[qkv_type](k_ptr, k_size)
    rand[qkv_type](v_ptr, v_size)

    # Initialize causal mask.
    for b in range(batch_size):
        for h in range(num_heads):
            for q_idx in range(seq_len):
                for k_idx in range(num_keys):
                    mask.store(
                        Index(b, h, q_idx, k_idx),
                        0 if q_idx + num_keys - seq_len
                        >= k_idx else min_or_neg_inf[mask_type](),
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
    var q_device = NDBuffer[
        qkv_type, 4, _, DimList(Dim(), Dim(), num_heads, depth)
    ](
        q_device_ptr._unsafe_ptr(),
        Index(batch_size, seq_len, num_heads, depth),
    )
    var k_device = NDBuffer[
        qkv_type, 4, _, DimList(Dim(), Dim(), kv_num_heads, depth)
    ](
        k_device_ptr._unsafe_ptr(),
        Index(batch_size, num_keys, kv_num_heads, depth),
    )
    var v_device = NDBuffer[
        qkv_type, 4, _, DimList(Dim(), Dim(), kv_num_heads, depth)
    ](
        v_device_ptr._unsafe_ptr(),
        Index(batch_size, num_keys, kv_num_heads, depth),
    )
    var mask4d = NDBuffer[mask_type, 4, _, DimList.create_unknown[4]()](
        mask_device_ptr._unsafe_ptr(),
        Index(batch_size, num_heads, seq_len, num_keys),
    )
    var output_device = NDBuffer[
        qkv_type, 4, _, DimList(Dim(), Dim(), num_heads, depth)
    ](
        output_device_ptr._unsafe_ptr(),
        Index(batch_size, seq_len, num_heads, depth),
    )

    alias config = MHAConfig(
        qkv_type,
        UInt(num_heads),
        UInt(depth),
        BK=OptionalReg[UInt](UInt(128 // size_of[qkv_type]())),
        num_pipeline_stages=UInt(4) if (
            ctx.default_device_info is H100 or ctx.default_device_info is B200
        ) else 2,
    )

    @parameter
    @always_inline
    @__copy_capture(q_device, k_device, v_device, mask4d, output_device)
    fn kernel_launch(ctx: DeviceContext) raises:
        flash_attention[config=config](
            output_device,
            q_device,
            k_device,
            v_device,
            CausalMask(),
            IdentityScoreMod(),
            scale,
            ctx,
            num_partitions=num_partitions,
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

    var output_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
    ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

    var output_device_ref = NDBuffer[
        qkv_type, 4, _, DimList(Dim(), Dim(), num_heads, depth)
    ](
        output_ref_device_ptr._unsafe_ptr(),
        Index(batch_size, seq_len, num_heads, depth),
    )

    alias config_baseline = MHAConfig(
        qkv_type,
        UInt(num_heads),
        UInt(depth),
        BK=OptionalReg[UInt](UInt(128 // size_of[qkv_type]())),
        num_pipeline_stages=2,
        algorithm=FlashAttentionAlgorithm(2),
    )
    flash_attention[config=config_baseline](
        output_device_ref,
        q_device,
        k_device,
        v_device,
        MaterializedMask(mask4d),
        IdentityScoreMod(),
        scale,
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(output_ptr, output_ref_device_ptr)
    _ = output_ref_device_ptr

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


fn construct_depths(is_sm90orsm100: Bool) -> List[Int]:
    var depths = [64, 128]
    if is_sm90orsm100:
        depths.append(80)
        depths.append(256)
    return depths^


def main():
    with DeviceContext() as ctx:
        alias is_sm90orsm100 = ctx.default_device_info is H100 or ctx.default_device_info is B200
        alias depths = construct_depths(is_sm90orsm100)

        @parameter
        for d in range(len(depths)):
            alias depth = depths[d]

            @parameter
            if depth <= 128:
                # fp32 tf32-fp32 mma
                test[
                    DType.float32,
                    DType.float32,
                    depth,
                    1,
                ](128, 128, ctx, is_benchmark())

                test[
                    DType.float32,
                    DType.float32,
                    depth,
                    3,
                ](14, 14, ctx, is_benchmark())

                test[
                    DType.float32,
                    DType.float32,
                    depth,
                    1,
                ](178, 178, ctx, is_benchmark())

            # bf16 depth == 128, bf16-fp32 mma
            test[
                DType.bfloat16,
                DType.bfloat16,
                depth=depth,
                num_heads=1,
            ](128, 128, ctx)

            test[
                DType.bfloat16,
                DType.float32,
                depth=depth,
                num_heads=1,
            ](384, 384, ctx)

            test[
                DType.bfloat16,
                DType.float32,
                depth,
                24,
                group=3,
            ](1024, 1024, ctx)

            # BF16 with sequence length not multiple of 128
            test[
                DType.bfloat16,
                DType.float32,
                depth,
                3,
                group=3,
            ](64, 64, ctx)

            test[
                DType.bfloat16,
                DType.bfloat16,
                depth,
                3,
                group=3,
            ](102, 102, ctx)

            test[
                DType.bfloat16,
                DType.float32,
                depth,
                1,
            ](14, 14, ctx)

            test[
                DType.bfloat16,
                DType.bfloat16,
                depth,
                1,
            ](528, 528, ctx)

            # BF16 with different length for prompt and cache.
            test[
                DType.bfloat16,
                DType.float32,
                depth,
                1,
            ](128, 256, ctx)

            test[
                DType.bfloat16,
                DType.float32,
                depth,
                3,
                group=3,
            ](32, 77, ctx)

            test[
                DType.bfloat16,
                DType.float32,
                depth,
                16,
                group=8,
            ](201, 400, ctx)

            test[
                DType.bfloat16,
                DType.float32,
                depth,
                12,
                group=4,
            ](1000, 2000, ctx)

            test[
                DType.bfloat16,
                DType.bfloat16,
                depth,
                32,
                group=4,
            ](201, 600, ctx)

            # BF16 token gen

            test[
                DType.bfloat16,
                DType.bfloat16,
                depth,
                32,
            ](1, 512, ctx, is_benchmark())

            test[
                DType.bfloat16,
                DType.bfloat16,
                depth,
                11,
            ](1, 256, ctx)

            test[
                DType.bfloat16,
                DType.float32,
                depth,
                1,
            ](1, 11, ctx)

            test[
                DType.bfloat16,
                DType.float32,
                depth,
                1,
            ](1, 11, ctx, num_partitions=OptionalReg[Int](2))

            test[
                DType.bfloat16,
                DType.bfloat16,
                depth,
                2,
            ](1, 523, ctx)

            test[
                DType.bfloat16,
                DType.float32,
                depth,
                24,
                group=3,
            ](1, 29, ctx)

            test[
                DType.bfloat16,
                DType.bfloat16,
                depth,
                3,
                group=3,
            ](1, 156, ctx)

            test[
                DType.bfloat16,
                DType.bfloat16,
                depth,
                3,
                group=3,
            ](1, 208, ctx)

            test[
                DType.bfloat16,
                DType.bfloat16,
                depth,
                32,
                group=4,
            ](1, 1208, ctx)

            test[
                DType.bfloat16,
                DType.bfloat16,
                depth,
                32,
                group=4,
            ](1, 2008, ctx)

            test[
                DType.bfloat16,
                DType.bfloat16,
                depth,
                32,
                group=4,
            ](1, 5000, ctx)

            test[
                DType.bfloat16,
                DType.bfloat16,
                depth,
                32,
                group=4,
            ](1, 600, ctx)

            @parameter
            if ctx.default_device_info is A100 or is_sm90orsm100:
                test[
                    DType.bfloat16,
                    DType.bfloat16,
                    depth,
                    32,
                    group=16,
                ](1, 2008, ctx)
