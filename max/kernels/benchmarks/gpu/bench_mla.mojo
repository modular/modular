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

from sys import env_get_dtype, env_get_int, env_get_bool

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from gpu import *
from gpu.host import DeviceContext
from internal_utils import arg_parse, CacheBustingBuffer
from internal_utils._utils import InitializationType
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from nn.mha import flash_attention
from nn.mla import flare_mla_decoding, flare_mla_prefill
from nn.mha_mask import CausalMask, MaterializedMask
from nn.mha_score_mod import IdentityScoreMod

from utils.index import Index


fn bench_decode[
    mask_rank: Int,
    qkv_type: DType,
    mask_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    decoding_warp_split_k: Bool = False,
    use_causal_mask: Bool = True,
    cache_busting: Bool = True,
](
    mut m: Bench,
    seq_len: Int,
    num_keys: Int,
    batch_size: Int,
    num_partitions: Int,
    mode: String,
    ctx: DeviceContext,
) raises:
    comptime assert mask_rank in (3, 4), "MLA only supports rank 3 or 4."

    # Query, key, value dimensions.
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var o_size = q_size
    var mask_size = (
        (num_heads if mask_rank == 4 else 1) * seq_len * num_keys * batch_size
    )

    # For cache busting: calculate strides and larger buffer sizes.
    comptime simd_size = 4
    var cb_q = CacheBustingBuffer[qkv_type](
        q_size, simd_size, ctx, cache_busting
    )
    var cb_k = CacheBustingBuffer[qkv_type](
        k_size, simd_size, ctx, cache_busting
    )
    var cb_mask = CacheBustingBuffer[mask_type](
        mask_size, simd_size, ctx, cache_busting
    )
    var cb_o = CacheBustingBuffer[qkv_type](
        o_size, simd_size, ctx, cache_busting
    )

    # Initialize data on the device.
    comptime random_distribution = InitializationType.uniform_distribution

    cb_q.init_on_device(random_distribution, ctx)
    cb_k.init_on_device(random_distribution, ctx)
    cb_mask.init_on_device(random_distribution, ctx)

    # Layout definitions.
    comptime q_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth
    )
    comptime k_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, kv_num_heads, depth
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth
    )

    @parameter
    @always_inline
    @__copy_capture(cb_q, cb_k, cb_mask, cb_o)
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn _kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            var q_device = LayoutTensor[qkv_type, q_layout](
                cb_q.offset_ptr(iteration),
                RuntimeLayout[q_layout].row_major(
                    Index(batch_size, seq_len, num_heads, depth)
                ),
            )
            var k_device = LayoutTensor[qkv_type, k_layout](
                cb_k.offset_ptr(iteration),
                RuntimeLayout[k_layout].row_major(
                    Index(batch_size, num_keys, kv_num_heads, depth)
                ),
            )
            var mask3d = LayoutTensor[mask_type, Layout.row_major[3]()](
                cb_mask.offset_ptr(iteration),
                RuntimeLayout[Layout.row_major[3]()].row_major(
                    Index(batch_size, seq_len, num_keys)
                ),
            )
            var mask4d = LayoutTensor[mask_type, Layout.row_major[4]()](
                cb_mask.offset_ptr(iteration),
                RuntimeLayout[Layout.row_major[4]()].row_major(
                    Index(batch_size, num_heads, seq_len, num_keys)
                ),
            )
            var output_device = LayoutTensor[qkv_type, output_layout](
                cb_o.offset_ptr(iteration),
                RuntimeLayout[output_layout].row_major(
                    Index(batch_size, seq_len, num_heads, depth)
                ),
            )

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

        b.iter_custom[_kernel_launch](ctx)

    fn compute_flops() -> Int:
        return 4 * batch_size * num_heads * seq_len * num_keys * depth

    m.bench_function[bench_func](
        BenchId(
            "mla_decode",
            # fmt: off
        input_id=String(
            "qkv_type=", qkv_type,
            "/num_heads=", num_heads,
            "/seq_len=", seq_len,
            "/num_keys=", num_keys,
            "/batch_size=", batch_size,
            "/mode=", mode,
            "/cache_busting=", cache_busting,
        ),
            # fmt: on
        ),
        [ThroughputMeasure(BenchMetric.flops, compute_flops())],
    )

    ctx.synchronize()

    _ = cb_q
    _ = cb_k
    _ = cb_mask
    _ = cb_o


fn bench_prefill[
    qkv_type: DType,
    mask_type: DType,
    depth: Int,
    num_heads: Int,
    kv_depth: Int,
    cache_depth: Int,
    cache_num_heads: Int,
    use_causal_mask: Bool = True,
    cache_busting: Bool = True,
](
    mut m: Bench,
    seq_len: Int,
    num_keys: Int,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    # Query, key, value dimensions.
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))

    # Q, K, V shapes.
    var q_size = batch_size * seq_len * num_heads * depth
    var k_size = batch_size * num_keys * num_heads * kv_depth
    var v_size = k_size
    var o_size = batch_size * seq_len * num_heads * kv_depth
    var cache_size = batch_size * num_keys * cache_num_heads * cache_depth

    # For cache busting: calculate strides and larger buffer sizes.
    comptime simd_size = 4
    var cb_q = CacheBustingBuffer[qkv_type](
        q_size, simd_size, ctx, cache_busting
    )
    var cb_k = CacheBustingBuffer[qkv_type](
        k_size, simd_size, ctx, cache_busting
    )
    var cb_v = CacheBustingBuffer[qkv_type](
        v_size, simd_size, ctx, cache_busting
    )
    var cb_cache = CacheBustingBuffer[qkv_type](
        cache_size, simd_size, ctx, cache_busting
    )
    var cb_o = CacheBustingBuffer[qkv_type](
        o_size, simd_size, ctx, cache_busting
    )

    # input row offsets and cache row offsets
    var input_row_offsets = UnsafePointer[UInt32].alloc(batch_size + 1)
    var cache_row_offsets = UnsafePointer[UInt32].alloc(batch_size + 1)
    for i in range(batch_size):
        input_row_offsets[i] = UInt32(i * seq_len)
        cache_row_offsets[i] = UInt32(i * num_keys)
    input_row_offsets[batch_size] = UInt32(batch_size * seq_len)
    cache_row_offsets[batch_size] = UInt32(batch_size * num_keys)
    var input_row_offsets_device_ptr = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var cache_row_offsets_device_ptr = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )

    # Initialize data on the device.
    comptime random_distribution = InitializationType.uniform_distribution

    cb_q.init_on_device(random_distribution, ctx)
    cb_k.init_on_device(random_distribution, ctx)
    cb_v.init_on_device(random_distribution, ctx)
    cb_cache.init_on_device(random_distribution, ctx)

    # Copy from host to device
    ctx.enqueue_copy(input_row_offsets_device_ptr, input_row_offsets)
    ctx.enqueue_copy(cache_row_offsets_device_ptr, cache_row_offsets)

    # Layout definitions.
    comptime q_layout = Layout.row_major(UNKNOWN_VALUE, num_heads, depth)
    comptime k_layout = Layout.row_major(UNKNOWN_VALUE, num_heads, kv_depth)
    comptime v_layout = Layout.row_major(UNKNOWN_VALUE, num_heads, kv_depth)
    comptime cache_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, cache_num_heads, cache_depth
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_heads, kv_depth
    )

    # Row offsets tensors (these don't need cache busting offsets).
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
        cb_q,
        cb_k,
        cb_v,
        cb_cache,
        cb_o,
        input_row_offsets_device,
        cache_row_offsets_device,
    )
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn _kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            var q_device = LayoutTensor[qkv_type, q_layout](
                cb_q.offset_ptr(iteration),
                RuntimeLayout[q_layout].row_major(
                    Index(batch_size * seq_len, num_heads, depth)
                ),
            )
            var k_device = LayoutTensor[qkv_type, k_layout](
                cb_k.offset_ptr(iteration),
                RuntimeLayout[k_layout].row_major(
                    Index(batch_size * num_keys, num_heads, kv_depth)
                ),
            )
            var v_device = LayoutTensor[qkv_type, v_layout](
                cb_v.offset_ptr(iteration),
                RuntimeLayout[v_layout].row_major(
                    Index(batch_size * num_keys, num_heads, kv_depth)
                ),
            )
            var cache_device = LayoutTensor[qkv_type, cache_layout](
                cb_cache.offset_ptr(iteration),
                RuntimeLayout[cache_layout].row_major(
                    Index(batch_size, num_keys, cache_num_heads, cache_depth)
                ),
            )
            var output_device = LayoutTensor[qkv_type, output_layout](
                cb_o.offset_ptr(iteration),
                RuntimeLayout[output_layout].row_major(
                    Index(batch_size * seq_len, num_heads, kv_depth)
                ),
            )

            flare_mla_prefill[rank = q_device.rank](
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

        b.iter_custom[_kernel_launch](ctx)

    fn compute_flops() -> Int:
        return 4 * batch_size * num_heads * seq_len * num_keys * depth

    m.bench_function[bench_func](
        BenchId(
            "mla_prefill",
            # fmt: off
        input_id=String(
            "qkv_type=", qkv_type,
            "/num_heads=", num_heads,
            "/seq_len=", seq_len,
            "/num_keys=", num_keys,
            "/batch_size=", batch_size,
            "/kv_depth=", kv_depth,
            "/cache_depth=", cache_depth,
            "/cache_num_heads=", cache_num_heads,
            "/cache_busting=", cache_busting,
        ),
            # fmt: on
        ),
        [ThroughputMeasure(BenchMetric.flops, compute_flops())],
    )

    ctx.synchronize()

    _ = cb_q
    _ = cb_k
    _ = cb_v
    _ = cb_cache
    _ = cb_o


@fieldwise_init
struct MLA_cfg(ImplicitlyCopyable):
    # params
    var mask_rank: Int
    var qkv_type: DType
    var mask_type: DType
    var depth: Int
    var prefill_depth: Int
    var num_heads: Int
    var group: Int
    var decoding_warp_split_k: Bool
    var use_causal_mask: Bool
    var cache_busting: Bool
    var kv_depth: Int
    var cache_depth: Int
    var cache_num_heads: Int

    @no_inline
    fn __str__(self) -> String:
        # fmt: off
        return String(
            "mask_rank", self.mask_rank,
            "qkv_type=", self.qkv_type,
            "/mask_type=", self.mask_type,
            "/depth=", self.depth,
            "/prefill_depth=", self.prefill_depth,
            "/num_heads=", self.num_heads,
            "/group=", self.group,
            "/kv_depth=", self.kv_depth,
            "/cache_depth=", self.cache_depth,
            "/cache_num_heads=", self.cache_num_heads,
            "/cache_busting=", self.cache_busting,
        )
        # fmt: on


def main():
    comptime mask_rank = env_get_int["mask_rank", 3]()
    comptime qkv_type = env_get_dtype["qkv_type", DType.bfloat16]()
    comptime mask_type = env_get_dtype["mask_type", DType.float32]()
    comptime depth = env_get_int["depth", 576]()
    comptime prefill_depth = env_get_int["prefill_depth", 192]()
    comptime num_heads = env_get_int["num_heads", 128]()
    comptime group = env_get_int["group", 128]()
    comptime use_causal_mask = env_get_bool["use_causal_mask", True]()
    comptime decoding_warp_split_k = env_get_bool[
        "decoding_warp_split_k", False
    ]()
    comptime cache_busting = env_get_bool["cache_busting", True]()
    comptime kv_depth = env_get_int["kv_depth", 128]()
    comptime cache_depth = env_get_int["cache_depth", 576]()
    comptime cache_num_heads = env_get_int["cache_num_heads", 1]()

    var seq_len = Int(arg_parse("seq_len", 64))
    var num_keys = Int(arg_parse("num_keys", 64))
    var batch_size = Int(arg_parse("batch_size", 1))
    var num_partitions = Int(arg_parse("num_partitions", 1))
    var mode = String(arg_parse("mode", "decode"))

    comptime cfg = MLA_cfg(
        mask_rank=mask_rank,
        qkv_type=qkv_type,
        mask_type=mask_type,
        depth=depth,
        prefill_depth=prefill_depth,
        num_heads=num_heads,
        group=group,
        decoding_warp_split_k=decoding_warp_split_k,
        use_causal_mask=use_causal_mask,
        cache_busting=cache_busting,
        kv_depth=kv_depth,
        cache_depth=cache_depth,
        cache_num_heads=cache_num_heads,
    )

    var m = Bench()
    with DeviceContext() as ctx:
        bench_decode[
            cfg.mask_rank,
            cfg.qkv_type,
            cfg.mask_type,
            cfg.depth,
            cfg.num_heads,
            cfg.group,
            cfg.decoding_warp_split_k,
            cfg.use_causal_mask,
            cfg.cache_busting,
        ](
            m,
            seq_len,
            num_keys,
            batch_size,
            num_partitions,
            mode,
            ctx,
        )

        bench_prefill[
            cfg.qkv_type,
            cfg.mask_type,
            cfg.prefill_depth,
            cfg.num_heads,
            cfg.kv_depth,
            cfg.cache_depth,
            cfg.cache_num_heads,
            use_causal_mask=True,
            cache_busting = cfg.cache_busting,
        ](m, seq_len, num_keys, batch_size, ctx)

    m.dump_report()
