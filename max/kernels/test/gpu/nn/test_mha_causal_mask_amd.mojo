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
from std.sys import argv, get_defined_bool


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
    comptime mask_type = DType.bfloat16
    comptime output_type = DType.bfloat16

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size
    var mask_size = num_heads * seq_len * num_keys

    # Allocate memory for all variables.
    var q_ptr = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_ptr = ctx.enqueue_create_host_buffer[qkv_type](k_size)
    var v_ptr = ctx.enqueue_create_host_buffer[qkv_type](v_size)
    var mask_ptr = ctx.enqueue_create_host_buffer[mask_type](mask_size)
    var output_ptr = ctx.enqueue_create_host_buffer[output_type](o_size)
    var flash_output_ptr = ctx.enqueue_create_host_buffer[output_type](o_size)

    for i in range(o_size):
        output_ptr[i] = Scalar[output_type](0)

    # Construct mask buffer for causal mask initialization.
    comptime layout_4d = Layout.row_major[4]()
    var mask = LayoutTensor[mask_type, layout_4d](
        mask_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_heads, seq_len, num_keys)
        ),
    )

    # Initialize Q, K, V in bf16, then roundtrip through qkv_type so the
    # naive bf16 reference sees identical values (matters for fp8).
    var q_bf16_ptr = ctx.enqueue_create_host_buffer[DType.bfloat16](q_size)
    var k_bf16_ptr = ctx.enqueue_create_host_buffer[DType.bfloat16](k_size)
    var v_bf16_ptr = ctx.enqueue_create_host_buffer[DType.bfloat16](v_size)
    rand(q_bf16_ptr.as_span())
    rand(k_bf16_ptr.as_span())
    rand(v_bf16_ptr.as_span())
    for i in range(q_size):
        var val = q_bf16_ptr[i].cast[qkv_type]()
        q_ptr[i] = val
        q_bf16_ptr[i] = val.cast[DType.bfloat16]()
    for i in range(k_size):
        var val = k_bf16_ptr[i].cast[qkv_type]()
        k_ptr[i] = val
        k_bf16_ptr[i] = val.cast[DType.bfloat16]()
    for i in range(v_size):
        var val = v_bf16_ptr[i].cast[qkv_type]()
        v_ptr[i] = val
        v_bf16_ptr[i] = val.cast[DType.bfloat16]()

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
    var output_device_ptr = ctx.enqueue_create_buffer[output_type](o_size)

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

    # Naive reference: use roundtripped bf16 values so both flash and
    # naive see identical input data.
    var q_ref_device_ptr = ctx.enqueue_create_buffer[DType.bfloat16](q_size)
    var k_ref_device_ptr = ctx.enqueue_create_buffer[DType.bfloat16](k_size)
    var v_ref_device_ptr = ctx.enqueue_create_buffer[DType.bfloat16](v_size)
    ctx.enqueue_copy(q_ref_device_ptr, q_bf16_ptr)
    ctx.enqueue_copy(k_ref_device_ptr, k_bf16_ptr)
    ctx.enqueue_copy(v_ref_device_ptr, v_bf16_ptr)

    var q_ref_device = TileTensor(
        q_ref_device_ptr.unsafe_ptr(),
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
    )
    var k_ref_device = TileTensor(
        k_ref_device_ptr.unsafe_ptr(),
        row_major(
            (Idx(batch_size), Idx(num_keys), Idx[kv_num_heads](), Idx[depth]())
        ),
    )
    var v_ref_device = TileTensor(
        v_ref_device_ptr.unsafe_ptr(),
        row_major(
            (Idx(batch_size), Idx(num_keys), Idx[kv_num_heads](), Idx[depth]())
        ),
    )
    var output_ref_device_ptr = ctx.enqueue_create_buffer[output_type](o_size)
    ctx.enqueue_copy(output_ref_device_ptr, output_ptr)
    var output_device_ref = TileTensor(
        output_ref_device_ptr.unsafe_ptr(),
        row_major(
            (Idx(batch_size), Idx(seq_len), Idx[num_heads](), Idx[depth]())
        ),
    )

    mha_gpu_naive(
        q_ref_device,
        k_ref_device,
        v_ref_device,
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
    ctx.synchronize()
    _ = output_ref_device_ptr
    _ = q_ref_device_ptr
    _ = k_ref_device_ptr
    _ = v_ref_device_ptr

    var rtol = 6e-2 if qkv_type.is_float8() else 3e-2
    var atol = 1e-3 if qkv_type.is_float8() else 1e-5
    for h in range(num_heads):
        for s in range(seq_len):
            for d in range(depth):
                var expect = output_ptr[d + depth * (h + s * num_heads)].cast[
                    DType.float64
                ]()
                var actual = flash_output_ptr[
                    d + depth * (h + s * num_heads)
                ].cast[DType.float64]()
                if not isclose(actual, expect, atol=atol, rtol=rtol):
                    var rerr = abs((actual - expect) / expect)
                    print(h, s, d, actual, expect, rerr)
                assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = mask_device_ptr
    _ = output_device_ptr


comptime USE_FP8 = get_defined_bool["USE_FP8", False]()


def test_helper[depth: Int](ctx: DeviceContext) raises:
    comptime dtype = DType.float8_e4m3fn if USE_FP8 else DType.bfloat16
    test[dtype, depth=depth, num_heads=1](128, 128, ctx)
    test[dtype, depth=depth, num_heads=1](384, 384, ctx)
    test[dtype, depth=depth, num_heads=24, group=3](1024, 1024, ctx)
    test[dtype, depth=depth, num_heads=16, group=16](128, 128, ctx)
    test[dtype, depth=depth, num_heads=16, group=16](1024, 1024, ctx)
    # Sequence length not multiple of 128
    test[dtype, depth=depth, num_heads=3, group=3](128, 128, ctx)
    test[dtype, depth=depth, num_heads=3, group=3](102, 102, ctx)
    test[dtype, depth=depth, num_heads=1](14, 14, ctx)
    test[dtype, depth=depth, num_heads=1](528, 528, ctx)
    # Token gen
    test[dtype, depth=depth, num_heads=32](1, 512, ctx, is_benchmark())
    test[dtype, depth=depth, num_heads=11](1, 256, ctx)
    test[dtype, depth=depth, num_heads=1](1, 11, ctx)
    test[dtype, depth=depth, num_heads=2](1, 523, ctx)
    test[dtype, depth=depth, num_heads=24, group=3](1, 29, ctx)
    test[dtype, depth=depth, num_heads=3, group=3](1, 156, ctx)
    test[dtype, depth=depth, num_heads=3, group=3](1, 208, ctx)
    test[dtype, depth=depth, num_heads=32, group=4](1, 1208, ctx)
    test[dtype, depth=depth, num_heads=32, group=4](1, 2008, ctx)
    test[dtype, depth=depth, num_heads=32, group=4](1, 5000, ctx)
    test[dtype, depth=depth, num_heads=16, group=16](1, 128, ctx)
    test[dtype, depth=depth, num_heads=16, group=16](1, 1024, ctx)
    test[dtype, depth=depth, num_heads=16, group=16](1, 5000, ctx)


def main() raises:
    with DeviceContext() as ctx:
        comptime if USE_FP8:
            comptime for depth in [128, 256]:
                test_helper[depth](ctx)
        else:
            comptime for depth in [64, 128, 256]:
                test_helper[depth](ctx)
