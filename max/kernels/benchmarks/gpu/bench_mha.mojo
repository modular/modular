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
from math import isclose, isqrt
from random import rand
from sys import env_get_dtype, env_get_int

from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import *
from gpu.host import DeviceContext
from internal_utils import arg_parse
from nn.mha import flash_attention, mha_gpu_naive
from nn.mha_mask import CausalMask
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal

from utils.index import Index
from utils.numerics import min_or_neg_inf


fn run_mha[
    qkv_type: DType,
    mask_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
](
    mut m: Bench,
    seq_len: Int,
    num_keys: Int,
    batch_size: Int,
    num_partitions: Int,
    bench_and_verify: Bool,
    mode: String,
    ctx: DeviceContext,
) raises:
    # Query, key, value dimensions.
    alias scale = Float32(0.125)  # isqrt[type, 1](Float32(depth))
    alias kv_num_heads = num_heads // group

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size
    var mask_size = batch_size * num_heads * seq_len * num_keys

    # Allocate memory for all variables.
    var q_ptr = UnsafePointer[Scalar[qkv_type]].alloc(q_size)
    var k_ptr = UnsafePointer[Scalar[qkv_type]].alloc(k_size)
    var v_ptr = UnsafePointer[Scalar[qkv_type]].alloc(v_size)
    var mask_ptr = UnsafePointer[Scalar[mask_type]].alloc(mask_size)
    var output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)
    var flash_output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)

    # Q, K, V are randomly initialized.
    rand[qkv_type](q_ptr, q_size)
    rand[qkv_type](k_ptr, k_size)
    rand[qkv_type](v_ptr, v_size)

    # Initialize causal mask
    var mask = NDBuffer[mask_type, 4](
        mask_ptr, Index(batch_size, num_heads, seq_len, num_keys)
    )
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
        qkv_type, 4, MutableAnyOrigin, DimList(Dim(), Dim(), num_heads, depth)
    ](
        q_device_ptr.unsafe_ptr(),
        Index(batch_size, seq_len, num_heads, depth),
    )
    var k_device = NDBuffer[
        qkv_type,
        4,
        MutableAnyOrigin,
        DimList(Dim(), Dim(), kv_num_heads, depth),
    ](
        k_device_ptr.unsafe_ptr(),
        Index(batch_size, num_keys, kv_num_heads, depth),
    )
    var v_device = NDBuffer[
        qkv_type,
        4,
        MutableAnyOrigin,
        DimList(Dim(), Dim(), kv_num_heads, depth),
    ](
        v_device_ptr.unsafe_ptr(),
        Index(batch_size, num_keys, kv_num_heads, depth),
    )
    var mask4d = NDBuffer[
        mask_type, 4, MutableAnyOrigin, DimList.create_unknown[4]()
    ](
        mask_device_ptr.unsafe_ptr(),
        Index(batch_size, num_heads, seq_len, num_keys),
    )
    var output_device = NDBuffer[
        qkv_type, 4, MutableAnyOrigin, DimList(Dim(), Dim(), num_heads, depth)
    ](
        output_device_ptr.unsafe_ptr(),
        Index(batch_size, seq_len, num_heads, depth),
    )

    alias q_tile_num_rows = 32
    alias k_tile_num_rows = 128

    @parameter
    @always_inline
    @__copy_capture(q_device, k_device, v_device, mask4d, output_device)
    fn kernel_launch(ctx: DeviceContext) raises:
        flash_attention(
            output_device,
            q_device,
            k_device,
            v_device,
            CausalMask(),
            IdentityScoreMod(),
            scale,
            ctx,
            num_partitions if num_partitions > 0 else OptionalReg[Int](None),
        )

    if bench_and_verify:

        @parameter
        @always_inline
        fn bench_func(mut b: Bencher):
            @parameter
            @always_inline
            fn _kernel_launch(ctx: DeviceContext) raises:
                kernel_launch(ctx)

            b.iter_custom[_kernel_launch](ctx)

        fn compute_flops() -> Int:
            return 4 * batch_size * num_heads * seq_len * num_keys * depth

        m.bench_function[bench_func](
            BenchId(
                "mha",
                # fmt: off
            input_id=String(
                "qkv_type=", qkv_type,
                "/num_heads=", num_heads,
                "/seq_len=", seq_len,
                "/num_keys=", num_keys,
                "/batch_size=", batch_size,
                "/mode=", mode,
            ),
                # fmt: on
            ),
            ThroughputMeasure(BenchMetric.flops, compute_flops()),
        )
    else:
        kernel_launch(ctx)

    ctx.synchronize()
    ctx.enqueue_copy(flash_output_ptr, output_device_ptr)

    if bench_and_verify:
        var output_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
        var output_ref_device = NDBuffer[
            qkv_type,
            4,
            MutableAnyOrigin,
            DimList(Dim(), Dim(), num_heads, depth),
        ](
            output_ref_device_ptr.unsafe_ptr(),
            Index(batch_size, seq_len, num_heads, depth),
        )
        ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

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

        ctx.enqueue_copy(output_ptr, output_ref_device_ptr)
        _ = output_ref_device_ptr

        var rtol = 0.02

        for h in range(num_heads):
            for s in range(seq_len):
                for d in range(depth):
                    var expect = output_ptr.load(
                        d + depth * (h + s * num_heads)
                    )
                    var actual = flash_output_ptr.load(
                        d + depth * (h + s * num_heads)
                    )
                    if not isclose(expect, actual, atol=1e-5, rtol=rtol):
                        print(h, s, d, actual, expect)
                    assert_almost_equal(expect, actual, atol=1e-5, rtol=rtol)

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


@fieldwise_init
struct MHA_cfg(Copyable, Movable):
    # params
    var qkv_type: DType
    var mask_type: DType
    var depth: Int
    var num_heads: Int
    var group: Int

    @no_inline
    fn __str__(self) -> String:
        # fmt: off
        return String(
            "qkv_type=", self.qkv_type,
            "/mask_type=", self.mask_type,
            "/depth=", self.depth,
            "/num_heads=", self.num_heads,
            "/group=", self.group,
        )
        # fmt: on


fn main() raises:
    alias qkv_type = env_get_dtype["qkv_type", DType.bfloat16]()
    alias mask_type = env_get_dtype["mask_type", DType.float32]()
    alias depth = env_get_int["depth", 128]()
    alias num_heads = env_get_int["num_heads", 32]()
    alias group = env_get_int["group", 1]()

    var seq_len = Int(arg_parse("seq_len", 64))
    var num_keys = Int(arg_parse("num_keys", 64))
    var batch_size = Int(arg_parse("batch_size", 1))
    var num_partitions = Int(arg_parse("num_partitions", 1))
    var mode = String(arg_parse("mode", "none"))
    var bench_and_verify = arg_parse("benchmark_and_verify", True)

    alias cfg = MHA_cfg(
        qkv_type=qkv_type,
        mask_type=mask_type,
        depth=depth,
        num_heads=num_heads,
        group=group,
    )

    var m = Bench()
    with DeviceContext() as ctx:
        run_mha[
            cfg.qkv_type,
            cfg.mask_type,
            cfg.depth,
            cfg.num_heads,
            cfg.group,
        ](
            m,
            seq_len,
            num_keys,
            batch_size,
            num_partitions,
            bench_and_verify,
            mode,
            ctx,
        )
    m.dump_report()
