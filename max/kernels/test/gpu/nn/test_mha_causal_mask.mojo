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

from std.math import isclose
from std.random import rand
from std.sys import argv, size_of
from std.sys.defines import get_defined_int

from std.gpu import *
from std.gpu.host import DeviceContext
from std.gpu.host.info import A100, H100, _is_sm10x_gpu
from layout import (
    Idx,
    TileTensor,
    row_major,
)
from layout import Layout, LayoutTensor
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import (
    CausalMask,
    CausalPaddingMask,
    MHAMask,
    SlidingWindowCausalMask,
)
from std.testing import assert_almost_equal, assert_equal


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


def test[
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
    var q_ptr = alloc[Scalar[qkv_type]](q_size)
    var k_ptr = alloc[Scalar[qkv_type]](k_size)
    var v_ptr = alloc[Scalar[qkv_type]](v_size)
    var output_ptr = alloc[Scalar[qkv_type]](o_size)
    var flash_output_ptr = alloc[Scalar[qkv_type]](o_size)

    # Q, K, V are randomly initialized.
    rand[qkv_type](q_ptr, q_size)
    # rand[qkv_type](k_ptr, k_size)
    # rand[qkv_type](v_ptr, v_size)
    for i in range(v_size):
        v_ptr[i] = 0.5
    # for i in range(q_size):
    #     q_ptr[i] = 1.0
    for i in range(k_size):
        k_ptr[i] = 0.25

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
    var q_device = TileTensor(
        q_device_ptr,
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
    )
    var k_device = TileTensor(
        k_device_ptr,
        row_major(
            (Idx(batch_size), Idx(num_keys), Idx[kv_num_heads](), Idx[depth]())
        ),
    )
    var v_device = TileTensor(
        v_device_ptr,
        row_major(
            (Idx(batch_size), Idx(num_keys), Idx[kv_num_heads](), Idx[depth]())
        ),
    )
    var output_device = TileTensor(
        output_device_ptr,
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
    )

    @parameter
    @always_inline
    @__copy_capture(q_device, k_device, v_device, output_device)
    def kernel_launch(ctx: DeviceContext) raises:
        flash_attention(
            output_device,
            q_device,
            k_device,
            v_device,
            mask,
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

    var output_device_ref = TileTensor(
        output_ref_device_ptr,
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
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
        flash_attention(
            output_device_ref,
            q_device,
            k_device,
            v_device,
            mask,
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


def main() raises:
    with DeviceContext() as ctx:
        comptime depth = get_defined_int["depth", 128]()

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

        # bf16 bf16-fp32 mma
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
            group=8,
        ](1024, 1024, CausalMask(), ctx)

        test[
            DType.bfloat16,
            depth,
            24,
            group=8,
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

        comptime if (
            ctx.default_device_info == A100
            or ctx.default_device_info == H100
            or _is_sm10x_gpu(ctx.default_device_info)
        ):
            test[
                DType.bfloat16,
                depth,
                32,
                group=16,
            ](1, 2008, CausalMask(), ctx)

        # CausalPaddingMask tests: allocate valid_lengths on device.
        @parameter
        def make_vl(
            val: UInt32, ctx: DeviceContext
        ) raises -> LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ]:
            var host_ptr = alloc[Scalar[DType.uint32]](1)
            host_ptr[] = val
            var dev_buf = ctx.enqueue_create_buffer[DType.uint32](1)
            ctx.enqueue_copy(dev_buf, host_ptr)
            host_ptr.free()
            return LayoutTensor[
                DType.uint32, Layout.row_major(1), MutAnyOrigin
            ](dev_buf.unsafe_ptr())

        # valid_length == num_keys (equivalent to CausalMask).
        var vl_128_t = make_vl(128, ctx)
        test[
            DType.bfloat16,
            depth,
            1,
        ](128, 128, CausalPaddingMask(vl_128_t), ctx)

        # valid_length < num_keys (padding active).
        var vl_100_t = make_vl(100, ctx)
        test[
            DType.bfloat16,
            depth,
            1,
        ](128, 128, CausalPaddingMask(vl_100_t), ctx)

        # CausalPaddingMask with GQA.
        var vl_384_t = make_vl(384, ctx)
        test[
            DType.bfloat16,
            depth,
            24,
            group=3,
        ](384, 384, CausalPaddingMask(vl_384_t), ctx)

        # CausalPaddingMask with padding and GQA.
        var vl_300_t = make_vl(300, ctx)
        test[
            DType.bfloat16,
            depth,
            24,
            group=3,
        ](384, 384, CausalPaddingMask(vl_300_t), ctx)

        # CausalPaddingMask: token gen with padding.
        var vl_400_t = make_vl(400, ctx)
        test[
            DType.bfloat16,
            depth,
            32,
        ](1, 512, CausalPaddingMask(vl_400_t), ctx)
