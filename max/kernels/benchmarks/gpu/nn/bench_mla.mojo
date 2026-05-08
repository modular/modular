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

from std.sys import get_defined_dtype, get_defined_int, get_defined_bool

from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu import *
from std.gpu.host import DeviceContext
from internal_utils import arg_parse, CacheBustingBuffer
from internal_utils._utils import InitializationType
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from nn.attention.gpu.mla import flare_mla_decoding, flare_mla_prefill
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from nn.attention.mha_mask import CausalMask

from std.utils.index import Index


def bench_decode[
    qkv_type: DType,
    output_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    decoding_warp_split_k: Bool = False,
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
    # Query, key, value dimensions.
    comptime scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    comptime kv_num_heads = num_heads // group
    # MLA decode output has the V (no-rope) portion only: depth - 64.
    comptime v_depth = depth - 64

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var o_size = batch_size * num_heads * seq_len * v_depth

    # For cache busting: calculate strides and larger buffer sizes.
    comptime simd_size = 4
    var cb_q = CacheBustingBuffer[qkv_type](
        q_size, simd_size, ctx, cache_busting
    )
    var cb_k = CacheBustingBuffer[qkv_type](
        k_size, simd_size, ctx, cache_busting
    )
    var cb_o = CacheBustingBuffer[output_type](
        o_size, simd_size, ctx, cache_busting
    )

    # Initialize data on the device.
    comptime random_distribution = InitializationType.uniform_distribution

    cb_q.init_on_device(random_distribution, ctx)
    cb_k.init_on_device(random_distribution, ctx)

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        _is_cache_length_accurate=True,
    ](batch_size, num_keys, 1, ctx)
    var scalar_args_buf_lt = mla_args.gpu_layout_tensor()

    @parameter
    @always_inline
    @__copy_capture(cb_q, cb_k, cb_o, scalar_args_buf_lt)
    def bench_func(mut b: Bencher):
        @parameter
        @always_inline
        def _kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            var q_device = TileTensor(
                cb_q.offset_ptr(iteration),
                row_major(
                    Coord(
                        Idx(batch_size),
                        Idx(seq_len),
                        Idx[num_heads](),
                        Idx[depth](),
                    )
                ),
            )
            var k_device = TileTensor(
                cb_k.offset_ptr(iteration),
                row_major(
                    Coord(
                        Idx(batch_size),
                        Idx(num_keys),
                        Idx[kv_num_heads](),
                        Idx[depth](),
                    )
                ),
            )
            var output_device = TileTensor(
                cb_o.offset_ptr(iteration),
                row_major(
                    Coord(
                        Idx(batch_size),
                        Idx(seq_len),
                        Idx[num_heads](),
                        Idx[v_depth](),
                    )
                ),
            )
            var scalar_args_tt = TileTensor(
                scalar_args_buf_lt.ptr, row_major[3]()
            )

            flare_mla_decoding[
                config=MHAConfig[qkv_type](num_heads, depth),
                decoding_warp_split_k=decoding_warp_split_k,
            ](
                output_device.as_any_origin(),
                q_device,
                k_device,
                CausalMask(),
                scale,
                ctx,
                scalar_args_tt,
                num_partitions=num_partitions,
            )

        b.iter_custom[_kernel_launch](ctx)

    def compute_flops() {read} -> Int:
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
    _ = cb_o
    _ = mla_args


def bench_prefill[
    qkv_type: DType,
    output_type: DType,
    depth: Int,
    num_heads: Int,
    kv_depth: Int,
    cache_depth: Int,
    cache_num_heads: Int,
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
    var cb_o = CacheBustingBuffer[output_type](
        o_size, simd_size, ctx, cache_busting
    )

    # input row offsets and cache row offsets
    var input_row_offsets = alloc[UInt32](batch_size + 1)
    var cache_row_offsets = alloc[UInt32](batch_size + 1)
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

    # Row offsets tensors (these don't need cache busting offsets).
    var input_row_offsets_device = TileTensor(
        input_row_offsets_device_ptr,
        row_major(Coord(Idx(batch_size + 1))),
    )
    var cache_row_offsets_device = TileTensor(
        cache_row_offsets_device_ptr,
        row_major(Coord(Idx(batch_size + 1))),
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
    def bench_func(mut b: Bencher):
        @parameter
        @always_inline
        def _kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            var q_device = TileTensor(
                cb_q.offset_ptr(iteration),
                row_major(
                    Coord(
                        Idx(batch_size * seq_len),
                        Idx[num_heads](),
                        Idx[depth](),
                    )
                ),
            )
            var k_device = TileTensor(
                cb_k.offset_ptr(iteration),
                row_major(
                    Coord(
                        Idx(batch_size * num_keys),
                        Idx[num_heads](),
                        Idx[kv_depth](),
                    )
                ),
            )
            var v_device = TileTensor(
                cb_v.offset_ptr(iteration),
                row_major(
                    Coord(
                        Idx(batch_size * num_keys),
                        Idx[num_heads](),
                        Idx[kv_depth](),
                    )
                ),
            )
            var cache_device = TileTensor(
                cb_cache.offset_ptr(iteration),
                row_major(
                    Coord(
                        Idx(batch_size),
                        Idx(num_keys),
                        Idx[cache_num_heads](),
                        Idx[cache_depth](),
                    )
                ),
            )
            var output_device = TileTensor(
                cb_o.offset_ptr(iteration),
                row_major(
                    Coord(
                        Idx(batch_size * seq_len),
                        Idx[num_heads](),
                        Idx[kv_depth](),
                    )
                ),
            )

            flare_mla_prefill[rank=q_device.rank](
                output_device,
                q_device,
                k_device,
                v_device,
                cache_device,
                CausalMask(),
                input_row_offsets_device,
                cache_row_offsets_device,
                scale,
                ctx,
                q_max_seq_len=seq_len,
            )

        b.iter_custom[_kernel_launch](ctx)

    def compute_flops() {read} -> Int:
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
struct MLA_cfg(ImplicitlyCopyable, Writable):
    # params
    var qkv_type: DType
    var output_type: DType
    var depth: Int
    var prefill_depth: Int
    var num_heads: Int
    var group: Int
    var decoding_warp_split_k: Bool
    var cache_busting: Bool
    var kv_depth: Int
    var cache_depth: Int
    var cache_num_heads: Int


def main() raises:
    comptime qkv_type = get_defined_dtype["qkv_type", DType.bfloat16]()
    comptime output_type = get_defined_dtype["output_type", DType.bfloat16]()
    comptime depth = get_defined_int["depth", 576]()
    comptime prefill_depth = get_defined_int["prefill_depth", 192]()
    comptime num_heads = get_defined_int["num_heads", 128]()
    comptime group = get_defined_int["group", 128]()
    comptime decoding_warp_split_k = get_defined_bool[
        "decoding_warp_split_k", False
    ]()
    comptime cache_busting = get_defined_bool["cache_busting", True]()
    comptime kv_depth = get_defined_int["kv_depth", 128]()
    comptime cache_depth = get_defined_int["cache_depth", 576]()
    comptime cache_num_heads = get_defined_int["cache_num_heads", 1]()

    var seq_len = Int(arg_parse("seq_len", 64))
    var num_keys = Int(arg_parse("num_keys", 64))
    var batch_size = Int(arg_parse("batch_size", 1))
    var num_partitions = Int(arg_parse("num_partitions", 1))
    var mode = String(arg_parse("mode", "decode"))

    comptime cfg = MLA_cfg(
        qkv_type=qkv_type,
        output_type=output_type,
        depth=depth,
        prefill_depth=prefill_depth,
        num_heads=num_heads,
        group=group,
        decoding_warp_split_k=decoding_warp_split_k,
        cache_busting=cache_busting,
        kv_depth=kv_depth,
        cache_depth=cache_depth,
        cache_num_heads=cache_num_heads,
    )

    var m = Bench()
    with DeviceContext() as ctx:
        bench_decode[
            cfg.qkv_type,
            cfg.output_type,
            cfg.depth,
            cfg.num_heads,
            cfg.group,
            cfg.decoding_warp_split_k,
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
            cfg.output_type,
            cfg.prefill_depth,
            cfg.num_heads,
            cfg.kv_depth,
            cfg.cache_depth,
            cfg.cache_num_heads,
            cache_busting=cfg.cache_busting,
        ](m, seq_len, num_keys, batch_size, ctx)

    m.dump_report()
