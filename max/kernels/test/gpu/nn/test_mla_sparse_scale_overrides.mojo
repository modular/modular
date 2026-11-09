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

"""Checks sparse MLA metadata, unit, and external scale selection exactly."""

from std.math import ceildiv
from std.testing import assert_equal
from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder
from max.gpu.host.info import _is_sm10x_gpu
from kv_cache.types import KVCacheStaticParams, KVCacheT, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from nn.attention.mha_utils import NullPointer
from nn.attention.gpu.nvidia.sm100.mla_sparse_attention import (
    mla_sparse_attention,
)

comptime HEADS = 8
comptime DEPTH = 576
comptime V_DEPTH = 512
comptime PAGE = 128
comptime TOKENS = 4
comptime TOPK = 4
comptime SCALE_BLOCK = 32
comptime SCALES = ceildiv(DEPTH, SCALE_BLOCK)
comptime Collection = PagedKVCacheCollection[
    DType.float8_e4m3fn,
    KVCacheStaticParams(num_heads=1, head_size=DEPTH, is_mla=True),
    PAGE,
    MutAnyOrigin,
    ImmutAnyOrigin,
    ImmutAnyOrigin,
    MutAnyOrigin,
    scale_dtype_=DType.float32,
    quantization_granularity_=SCALE_BLOCK,
]


def run_case[
    cache_t: KVCacheT,
    //,
    scale_override: Int,
](
    ctx: DeviceContext,
    cache: cache_t,
    q: TileTensor[address_space=.GENERIC, ...],
    indices: MutPointer[Int32, MutAnyOrigin],
    lengths: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    external_scales: MutPointer[Float32, MutAnyOrigin],
) raises:
    var output = ctx.enqueue_create_buffer[.bfloat16](TOKENS * HEADS * V_DEPTH)
    var out = TileTensor(output, row_major(TOKENS, Idx[HEADS], Idx[V_DEPTH]))

    def launch(launch_ctx: DeviceContext) raises {imm}:
        mla_sparse_attention[
            num_heads=HEADS,
            depth=DEPTH,
            ragged=False,
            fp8_scale_bf16=True,
            scale_block_size_override=scale_override,
        ](
            out,
            q,
            cache,
            indices,
            TOPK,
            lengths,
            lengths,
            NullPointer[DType.float32](),
            Float32(1),
            TOKENS,
            TOPK,
            launch_ctx,
            num_partitions_in=Optional[Int](1),
            scales_ptr_override=Optional[MutPointer[Float32, MutAnyOrigin]](
                external_scales
            ) if scale_override
            > 0 else None,
        )

    var expected = (
        Float32(2) if scale_override
        < 0 else (Float32(1) if scale_override == 0 else Float32(1.25))
    ).cast[.bfloat16]()
    var host = ctx.enqueue_create_host_buffer[.bfloat16](len(output))
    launch(ctx)
    ctx.enqueue_copy(host, output)
    ctx.synchronize()
    for i in range(len(host)):
        assert_equal(host[i], expected)

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        with builder.recording_context() as rec:
            launch(rec)

    var graph = DeviceGraph.create(ctx, build)
    for _ in range(3):
        output.enqueue_fill(-1)
        graph.replay()
        ctx.enqueue_copy(host, output)
        ctx.synchronize()
        for i in range(len(host)):
            assert_equal(host[i], expected)


def main() raises:
    with DeviceContext() as ctx:
        comptime if ctx.target.is_nvidia_gpu() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            var values = ctx.enqueue_create_buffer[.float8_e4m3fn](
                2 * PAGE * DEPTH
            )
            values.enqueue_fill(1)
            var scales = ctx.enqueue_create_buffer[.float32](2 * PAGE * SCALES)
            scales.enqueue_fill(2)
            var external_scales = ctx.enqueue_create_buffer[.float32](
                2 * PAGE * SCALES
            )
            external_scales.enqueue_fill(1.25)
            var lengths = ctx.enqueue_create_buffer[.uint32](1)
            lengths.enqueue_fill(0)
            var lut = ctx.enqueue_create_buffer[.uint32](1)
            lut.enqueue_fill(0)
            var collection = Collection(
                rebind[Collection.blocks_tt_type](
                    TileTensor(
                        values,
                        row_major(
                            Int64(2),
                            Idx[1],
                            Int64(1),
                            Idx[PAGE],
                            Idx[1],
                            Idx[DEPTH],
                        ),
                    ).as_unsafe_any_origin()
                ),
                TileTensor(lengths, row_major(Int64(1))).as_unsafe_any_origin(),
                TileTensor(lut, row_major(Int64(1), Int64(1)))
                .as_imm()
                .as_unsafe_any_origin(),
                UInt32(TOKENS),
                UInt32(TOKENS),
                rebind[Collection.scales_tt_type](
                    TileTensor(
                        scales,
                        row_major(
                            Int64(2),
                            Idx[1],
                            Int64(1),
                            Idx[PAGE],
                            Idx[1],
                            Idx[SCALES],
                        ),
                    ).as_unsafe_any_origin()
                ),
            )
            var q = ctx.enqueue_create_buffer[.bfloat16](TOKENS * HEADS * DEPTH)
            q.enqueue_fill(0)
            var indices_host = ctx.enqueue_create_host_buffer[.int32](
                TOKENS * TOPK
            )
            for t in range(TOKENS):
                for j in range(TOPK):
                    indices_host[t * TOPK + j] = Int32(j)
            var indices = ctx.enqueue_create_buffer[.int32](len(indices_host))
            ctx.enqueue_copy(indices, indices_host)
            var topk_lengths = ctx.enqueue_create_buffer[.uint32](TOKENS)
            topk_lengths.enqueue_fill(UInt32(TOPK))
            var q_tt = TileTensor(q, row_major(TOKENS, Idx[HEADS], Idx[DEPTH]))
            var selected = indices.unsafe_ptr().as_unsafe_any_origin()
            var count = TileTensor(topk_lengths, row_major(TOKENS)).as_imm()
            var external = external_scales.unsafe_ptr().as_unsafe_any_origin()
            var cache = collection.get_key_cache(0)
            run_case[scale_override=-1](
                ctx, cache, q_tt, selected, count, external
            )
            run_case[scale_override=0](
                ctx, cache, q_tt, selected, count, external
            )
            run_case[scale_override=SCALE_BLOCK](
                ctx, cache, q_tt, selected, count, external
            )
            _ = values^
            _ = scales^
            _ = external_scales^
            _ = lengths^
            _ = lut^
            _ = q^
            _ = indices^
            _ = topk_lengths^
