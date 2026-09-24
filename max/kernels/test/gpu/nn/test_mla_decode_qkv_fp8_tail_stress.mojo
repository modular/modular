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
"""Checks that the SM100 native-FP8 MLA decode kernels keep small softmax weights.

Q and the first key have 8 in the first rope dimension, so that key scores
64 * scale and the other attended keys score 0; each of them has 2^-12 of the
first key's softmax weight. The latent (V) is 0 for the first key and 1 for the
others, so every output element equals the share of the softmax mass held by
the other keys. The kernels convert P to e4m3 before the P@V MMA, and e4m3
rounds values below 2^-10 to zero, so these weights survive only if P is scaled
up before the conversion. Covers sparse decode over 2048 selected keys and
dense decode over 2049 keys, with 16 query heads (Layout G for dense) and 128,
at split-K counts 1, 2, 4, 8 and the dispatch default.
"""

from std.math import ceildiv, exp, log
from std.testing import assert_true
from std.collections import Optional

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from nn.attention.mha_mask import NullMask
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from std.utils.index import IndexList

comptime Q_DEPTH = 576
comptime V_DEPTH = 512
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime FP8 = DType.float8_e4m3fn


def share[
    num_heads: Int, sparse: Bool
](ctx: DeviceContext, topk: Int, weight_log2: Int, np: Int) raises -> Float64:
    """Returns the kernel's output mean divided by the exact small-weight share.
    """
    var batch_size = 1
    var cache_len = topk
    var num_keys = cache_len + 1
    var scale = Float32(Float64(weight_log2) * log(Float64(2.0)) / 64.0)
    var w = exp(-64.0 * Float64(scale))

    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=Q_DEPTH, is_mla=True
    )
    var pages = ceildiv(num_keys, PAGE_SIZE)
    var block_shape = IndexList[6](pages, 1, NUM_LAYERS, PAGE_SIZE, 1, Q_DEPTH)
    var block_elems = pages * NUM_LAYERS * PAGE_SIZE * Q_DEPTH
    var blocks_host = ctx.enqueue_create_host_buffer[FP8](block_elems)
    var lut_host = ctx.enqueue_create_host_buffer[.uint32](pages)
    var cl_host = ctx.enqueue_create_host_buffer[.uint32](1)
    ctx.synchronize()
    for p in range(pages):
        lut_host[p] = UInt32(p)
    cl_host[0] = UInt32(cache_len)
    for t in range(pages * PAGE_SIZE):
        for d in range(Q_DEPTH):
            blocks_host[t * Q_DEPTH + d] = Scalar[FP8](0)
        if t > 0:
            for d in range(V_DEPTH):
                blocks_host[t * Q_DEPTH + d] = Scalar[FP8](1)
    blocks_host[V_DEPTH] = Scalar[FP8](8)

    var q_host = ctx.enqueue_create_host_buffer[FP8](num_heads * Q_DEPTH)
    ctx.synchronize()
    for i in range(num_heads * Q_DEPTH):
        q_host[i] = Scalar[FP8](0)
    for h in range(num_heads):
        q_host[h * Q_DEPTH + V_DEPTH] = Scalar[FP8](8)

    var idx_host = ctx.enqueue_create_host_buffer[.int32](topk)
    var ro_host = ctx.enqueue_create_host_buffer[.uint32](2)
    ctx.synchronize()
    for i in range(topk):
        idx_host[i] = Int32(i)
    ro_host[0] = 0
    ro_host[1] = 1

    var blocks_dev = ctx.enqueue_create_buffer[FP8](block_elems)
    var lut_dev = ctx.enqueue_create_buffer[.uint32](pages)
    var cl_dev = ctx.enqueue_create_buffer[.uint32](1)
    var q_dev = ctx.enqueue_create_buffer[FP8](num_heads * Q_DEPTH)
    var idx_dev = ctx.enqueue_create_buffer[.int32](topk)
    var ro_dev = ctx.enqueue_create_buffer[.uint32](2)
    var out_dev = ctx.enqueue_create_buffer[.bfloat16](num_heads * V_DEPTH)
    ctx.enqueue_copy(blocks_dev, blocks_host)
    ctx.enqueue_copy(lut_dev, lut_host)
    ctx.enqueue_copy(cl_dev, cl_host)
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.enqueue_copy(ro_dev, ro_host)
    ctx.synchronize()

    comptime cl_layout = Layout(UNKNOWN_VALUE)
    comptime lt_layout_2d = Layout.row_major[2]()
    var kv_collection = PagedKVCacheCollection[FP8, kv_params, PAGE_SIZE](
        LayoutTensor[FP8, Layout.row_major[6]()](
            blocks_dev.unsafe_ptr(),
            RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
        ),
        LayoutTensor[mut=False, .uint32, cl_layout](
            cl_dev.unsafe_ptr(),
            RuntimeLayout[cl_layout].row_major(IndexList[1](1)),
        ),
        LayoutTensor[mut=False, .uint32, lt_layout_2d](
            lut_dev.unsafe_ptr(),
            RuntimeLayout[lt_layout_2d].row_major(IndexList[2](1, pages)),
        ),
        UInt32(1),
        UInt32(cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)
    var q_tt = TileTensor(
        q_dev.unsafe_ptr(), row_major((1, Idx[num_heads], Idx[Q_DEPTH]))
    )
    var out_tt = TileTensor(
        out_dev.unsafe_ptr(), row_major((1, Idx[num_heads], Idx[V_DEPTH]))
    )
    var ro_tt = TileTensor(ro_dev.unsafe_ptr(), row_major(2))
    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads, is_fp8_kv=True, fold_shared_index=False
    ](batch_size, cache_len, 1, ctx)
    var np_ovr = Optional[Int](np) if np > 0 else Optional[Int](None)
    comptime if sparse:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[FP8](num_heads, Q_DEPTH),
            ragged=True,
            sparse=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            NullMask(),
            ro_tt,
            scale,
            ctx,
            mla_args.gpu_tile_tensor(),
            d_indices=rebind[MutPointer[Int32, MutAnyOrigin]](
                idx_dev.unsafe_ptr()
            ),
            indices_stride=topk,
            num_partitions_in=np_ovr,
        )
    else:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[FP8](num_heads, Q_DEPTH),
            ragged=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            NullMask(),
            ro_tt,
            scale,
            ctx,
            mla_args.gpu_tile_tensor(),
            num_partitions_in=np_ovr,
        )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](
        num_heads * V_DEPTH
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    var attended = topk if sparse else num_keys
    var small_mass = Float64(attended - 1) * w
    var exact = small_mass / (1.0 + small_mass)
    var acc: Float64 = 0
    for i in range(num_heads * V_DEPTH):
        acc += out_host[i].cast[.float64]()
    _ = mla_args
    _ = blocks_dev
    _ = lut_dev
    _ = cl_dev
    _ = q_dev
    _ = idx_dev
    _ = ro_dev
    _ = out_dev
    return acc / Float64(num_heads * V_DEPTH) / exact


def check[num_heads: Int, sparse: Bool](ctx: DeviceContext) raises:
    # 0 selects the dispatch's own split-K count.
    var split_k_counts: List[Int] = [1, 2, 4, 8, 0]
    for np in split_k_counts:
        var got = share[num_heads, sparse](ctx, 2048, 12, np)
        var name = String(
            "sparse" if sparse else "dense",
            " num_heads=",
            num_heads,
            " num_partitions=",
            np,
            " share=",
            got,
        )
        print(name)
        assert_true(abs(got - 1.0) < 0.02, name)


def main() raises:
    with DeviceContext() as ctx:
        check[16, True](ctx)
        check[128, True](ctx)
        check[16, False](ctx)
        check[128, False](ctx)
