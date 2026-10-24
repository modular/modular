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
"""Short fp8 queries against a long paged cache in a pool of more than 2^30
block rows.

The short-query FA4 routes zero-fill the page slots of the last KV tile
that lie past the sequence by issuing TMAs at an out-of-bounds row. A
positive row such as 2^30 is out of bounds only while the pool has fewer
block rows: the Nemotron-H TP=2 pool (725,328 pages of 2 x 6 layers x 128
rows) has 1.1e9, so that "zero fill" read a live page and NaN reached the
output.

The cache declares the production pool shape over an allocation that holds
only the pages the request uses, so a TMA aimed at a row the request does
not own faults (or reads foreign bytes, which the host reference catches
when they are non-finite). A correct kernel touches only real pages.
"""

from std.math import ceildiv, exp, sqrt
from std.random import randn, seed

from layout import Idx, TileTensor, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from max.gpu.host import DeviceContext

from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from kv_cache_test_utils import padded_lut_cols
from nn.attention.gpu.mha import MHADecodeDispatchMetadata
from nn.kv_cache_ragged import generic_flash_attention_kv_cache_ragged

comptime num_q_heads = 16
comptime head_dim = 128
comptime page_size = 128
comptime num_layers = 6
comptime fp8 = DType.float8_e4m3fn
comptime bf16 = DType.bfloat16
comptime kv_params = KVCacheStaticParams(num_heads=1, head_size=head_dim)
# Production TP=2 pool: 725,328 pages x 2 x 6 x 128 rows.
comptime declared_blocks = 725_328


def run_case(q_len: Int, cache_len: Int, extra: Int, ctx: DeviceContext) raises:
    var seq_len = cache_len + q_len
    var num_pages = ceildiv(seq_len, page_size)
    var real_blocks = num_pages + 1
    var block_elems = 2 * num_layers * page_size * head_dim
    var layer = 0
    var scale = Float32(1.0) / sqrt(Float32(head_dim))

    var row_offsets = HostDeviceTileTensor[.uint32](row_major(Int64(2)), ctx)
    var cache_lengths = HostDeviceTileTensor[.uint32](row_major(Int64(1)), ctx)
    row_offsets.host_tensor()[0] = 0
    row_offsets.host_tensor()[1] = UInt32(q_len)
    cache_lengths.host_tensor()[0] = UInt32(cache_len)
    row_offsets.to_device()
    cache_lengths.to_device()

    var q_layout = row_major(q_len, Idx[num_q_heads], Idx[head_dim])
    var q = HostDeviceTileTensor[fp8](q_layout, ctx)
    var qh = q.host_tensor().unsafe_ptr()
    randn(qh, q_len * num_q_heads * head_dim)
    q.to_device()

    # Only the request's pages are allocated: logical page p is block p + 1,
    # and block 0 is the null page the LUT padding points at.
    var kv_host = List[Scalar[fp8]](
        length=real_blocks * block_elems, fill=Scalar[fp8](0)
    )
    var kvh = kv_host.unsafe_ptr()
    randn(kvh.unsafe_offset(block_elems), num_pages * block_elems)
    var kv_dev = ctx.enqueue_create_buffer[fp8](real_blocks * block_elems)
    ctx.enqueue_copy(kv_dev, kvh)
    var blocks = TileTensor(
        kv_dev.unsafe_ptr(),
        row_major(
            Int64(declared_blocks),
            Idx[2],
            Idx[num_layers],
            Idx[page_size],
            Idx[1],
            Idx[head_dim],
        ),
    )

    var lut_cols = padded_lut_cols(num_pages)
    var lut = HostDeviceTileTensor[.uint32](
        row_major(Int64(1), Int64(lut_cols)), ctx
    )
    for c in range(lut_cols):
        lut.host_tensor()[0, c] = 0
    for p in range(num_pages):
        lut.host_tensor()[0, p] = UInt32(p + 1)
    lut.to_device()

    comptime Collection = PagedKVCacheCollection[
        fp8,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var collection = Collection(
        blocks.as_unsafe_any_origin(),
        cache_lengths.device_tensor().as_imm().as_unsafe_any_origin(),
        lut.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(q_len),
        UInt32(seq_len + extra),
    )
    var meta = MHADecodeDispatchMetadata.from_runtime_values[
        num_q_heads, num_q_heads
    ](1, q_len, seq_len + extra, ctx)
    var out = HostDeviceTileTensor[bf16](q_layout, ctx)
    generic_flash_attention_kv_cache_ragged[
        target="gpu", mask_str="causal", output_dtype=bf16
    ](
        q.device_tensor().as_imm(),
        row_offsets.device_tensor().as_imm(),
        collection,
        UInt32(layer),
        scale,
        out.device_tensor(),
        ctx,
        meta,
    )
    ctx.synchronize()
    out.to_host()

    var blk = List[Int](length=num_pages, fill=0)
    for p in range(num_pages):
        blk[p] = p + 1
    var max_err = check_against_host[num_layers](
        qh,
        kvh,
        blk,
        layer,
        q_len,
        cache_len,
        scale,
        out.host_tensor().unsafe_ptr(),
    )
    print(
        t"OK q_len={q_len} cache={cache_len} max_ctx=seq+{extra}"
        t" max_abs_err={max_err}"
    )


def check_against_host[
    layers: Int
](
    qh: Pointer[Scalar[fp8], _],
    kvh: Pointer[Scalar[fp8], _],
    blk: List[Int],
    layer: Int,
    q_len: Int,
    cache_len: Int,
    scale: Float32,
    oh: Pointer[Scalar[bf16], _],
) raises -> Float32:
    """Checks the output against host attention; raises on any NaN or any
    element off by more than 2e-2. Returns the largest error."""
    var seq_len = cache_len + q_len
    # The layer's K and V rows in sequence order, converted to f32 once.
    var kf = List[Float32](length=seq_len * head_dim, fill=0)
    var vf = List[Float32](length=seq_len * head_dim, fill=0)
    for k in range(seq_len):
        var b = blk[k // page_size]
        var off = k % page_size
        var kb = (((b * 2 + 0) * layers + layer) * page_size + off) * head_dim
        var vb = (((b * 2 + 1) * layers + layer) * page_size + off) * head_dim
        for d in range(head_dim):
            kf[k * head_dim + d] = kvh[kb + d].cast[DType.float32]()
            vf[k * head_dim + d] = kvh[vb + d].cast[DType.float32]()
    var qf = List[Float32](length=head_dim, fill=0)
    var s = List[Float32](length=seq_len, fill=0)
    var o = List[Float32](length=head_dim, fill=0)
    var max_err = Float32(0)
    for r in range(q_len):
        var visible = cache_len + r + 1
        for h in range(num_q_heads):
            var qbase = (r * num_q_heads + h) * head_dim
            for d in range(head_dim):
                qf[d] = qh[qbase + d].cast[DType.float32]()
            var m = Float32(-1e30)
            for k in range(visible):
                var acc = SIMD[DType.float32, 8](0)
                for d in range(0, head_dim, 8):
                    acc += qf.unsafe_ptr().unsafe_load[width=8](
                        d
                    ) * kf.unsafe_ptr().unsafe_load[width=8](k * head_dim + d)
                s[k] = acc.reduce_add() * scale
                m = max(m, s[k])
            var l = Float32(0)
            for d in range(head_dim):
                o[d] = 0
            for k in range(visible):
                var p = exp(s[k] - m)
                l += p
                for d in range(0, head_dim, 8):
                    var cur = o.unsafe_ptr().unsafe_load[width=8](d)
                    o.unsafe_ptr().unsafe_store(
                        d,
                        cur
                        + p
                        * vf.unsafe_ptr().unsafe_load[width=8](
                            k * head_dim + d
                        ),
                    )
            for d in range(head_dim):
                var want = o[d] / l
                var got = oh[qbase + d].cast[DType.float32]()
                # Negated so a NaN counts as a mismatch.
                if not (abs(got - want) <= 2e-2):
                    raise Error(
                        t"q_len={q_len} cache={cache_len} row={r} head={h}"
                        t" dim={d}: got {got}, expected {want}"
                    )
                max_err = max(max_err, abs(got - want))
    return max_err


def main() raises:
    comptime assert (
        declared_blocks * 2 * num_layers * page_size > 1 << 30
    ), "the pool must have more block rows than the old OOB row (2^30)"
    seed(7)
    with DeviceContext() as ctx:
        for cache in [57344, 56832]:
            for q_len in [1, 2, 3, 4]:
                run_case(q_len, cache, 7, ctx)
