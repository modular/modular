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
from std.sys import argv


from std.gpu import *
from std.gpu.host import DeviceContext
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import CausalMask
from std.testing import assert_almost_equal

from std.utils.index import Index
from std.utils.numerics import min_or_neg_inf


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


def test[
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
    comptime batch_size = 1
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size
    var mask_size = num_heads * seq_len * num_keys

    # Allocate memory for all variables.
    var q_ptr = alloc[Scalar[qkv_type]](q_size)
    var k_ptr = alloc[Scalar[qkv_type]](k_size)
    var v_ptr = alloc[Scalar[qkv_type]](v_size)
    var mask_ptr = alloc[Scalar[mask_type]](mask_size)
    var output_ptr = alloc[Scalar[qkv_type]](o_size)
    var flash_output_ptr = alloc[Scalar[qkv_type]](o_size)

    # Construct mask buffer for causal mask initialization.
    comptime layout_4d = Layout.row_major[4]()
    var mask = LayoutTensor[mask_type, layout_4d](
        mask_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_heads, seq_len, num_keys)
        ),
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
    var q_device = TileTensor(
        q_device_ptr.unsafe_ptr(),
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
    )
    var k_device = TileTensor(
        k_device_ptr.unsafe_ptr(),
        row_major(
            (Idx(batch_size), Idx(num_keys), Idx[kv_num_heads](), Idx[depth]())
        ),
    )
    var v_device = TileTensor(
        v_device_ptr.unsafe_ptr(),
        row_major(
            (Idx(batch_size), Idx(num_keys), Idx[kv_num_heads](), Idx[depth]())
        ),
    )
    var mask4d = TileTensor(
        mask_device_ptr.unsafe_ptr(),
        row_major(
            (Idx(batch_size), Idx(num_heads), Idx(seq_len), Idx(num_keys))
        ),
    )
    var output_device = TileTensor(
        output_device_ptr.unsafe_ptr(),
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
    )

    @parameter
    @always_inline
    @__copy_capture(q_device, k_device, v_device, mask4d, output_device)
    def kernel_launch(ctx: DeviceContext) raises:
        flash_attention(
            output_device,
            q_device,
            k_device,
            v_device,
            CausalMask(),
            scale,
            ctx,
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
        output_ref_device_ptr.unsafe_ptr(),
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
    )

    mha_gpu_naive(
        q_device,
        k_device,
        v_device,
        mask4d,
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
    _ = output_ref_device_ptr

    var rtol = 2e-2
    for h in range(num_heads):
        for s in range(seq_len):
            for d in range(depth):
                var expect = output_ptr.load(
                    d + depth * (h + s * num_heads)
                ).cast[DType.float64]()
                var actual = flash_output_ptr.load(
                    d + depth * (h + s * num_heads)
                ).cast[DType.float64]()
                if not isclose(actual, expect, atol=1e-5, rtol=rtol):
                    var rerr = abs((actual - expect) / expect)
                    print(h, s, d, actual, expect, rerr)
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


def test_helper[depth: Int](ctx: DeviceContext) raises:
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
        depth=depth,
        num_heads=24,
        group=3,
    ](1024, 1024, ctx)
    # BF16 with sequence length not multiple of 128
    test[
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=3,
        group=3,
    ](128, 128, ctx)
    test[
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=3,
        group=3,
    ](102, 102, ctx)
    test[
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=1,
    ](14, 14, ctx)
    test[
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=1,
    ](528, 528, ctx)
    # BF16 token gen
    test[
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=32,
    ](1, 512, ctx, is_benchmark())
    test[
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=11,
    ](1, 256, ctx)
    test[
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=1,
    ](1, 11, ctx)
    test[
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=2,
    ](1, 523, ctx)
    test[
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=24,
        group=3,
    ](1, 29, ctx)
    test[
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=3,
        group=3,
    ](1, 156, ctx)
    test[
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=3,
        group=3,
    ](1, 208, ctx)
    test[
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=32,
        group=4,
    ](1, 1208, ctx)
    test[
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=32,
        group=4,
    ](1, 2008, ctx)
    test[
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=32,
        group=4,
    ](1, 5000, ctx)


def main() raises:
    with DeviceContext() as ctx:
        # experimental kernel only supports depth == 128
        comptime depths = [64, 128, 256]

        comptime for depth in depths:
            test_helper[depth](ctx)
