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
"""Stores and in-cache RoPE into a split paged KV cache.

MiMo-V2.6-Flash's shapes: a 192-wide key and a 128-wide value packed into a
320-wide row, 128-token pages, and a partial NeoX RoPE over the first 64 key
and query columns. Each case writes a ragged batch on top of an existing
cache (a chunked prefill, then decode and verify widths) twice: into the
split cache, and into the padded 256-wide cache it replaces, with the inputs
zero-padded as the padded production path pads them.

- The split cache's key columns and value columns, and the rotated query,
  must equal the padded path's first 192, 128 and 192 columns bitwise.
- Both must match an fp32 reference of the rotation to within one bfloat16
  rounding, and the stored values and the existing cache exactly.
"""

from std.math import cos, sin

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.fused_qk_rope import fused_qk_rope_ragged
from nn.kv_cache_ragged import kv_cache_store_ragged
from std.testing import assert_equal

from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable

comptime dtype = DType.bfloat16
comptime page_size = 128
comptime k_dim = 192
comptime v_dim = 128
comptime row_dim = k_dim + v_dim
comptime padded_dim = 256
comptime rope_dim = 64
comptime num_layers = 2
comptime layer = 1
comptime num_blocks = 64
comptime max_pos = 4224


def _val(seed: Int, i: Int, j: Int, d: Int) -> Scalar[dtype]:
    # Multiples of 1/8 below 16 in magnitude: exact in bfloat16.
    return Scalar[dtype](
        Float32((seed * 101 + i * 37 + j * 19 + d * 7) % 253 - 126) * 0.125
    )


def _new_k(h: Int, t: Int, d: Int) -> Scalar[dtype]:
    return _val(1, t, h, d)


def _new_v(h: Int, t: Int, d: Int) -> Scalar[dtype]:
    return _val(2, t, h, d)


def _old_k(b: Int, h: Int, t: Int, d: Int) -> Scalar[dtype]:
    return _val(3 + b, t, h, d)


def _old_v(b: Int, h: Int, t: Int, d: Int) -> Scalar[dtype]:
    return _val(7 + b, t, h, d)


def _q(h: Int, t: Int, d: Int) -> Scalar[dtype]:
    return _val(11, t, h, d)


def _rotated(
    x: Float32, partner: Float32, c: Float32, s: Float32, first: Bool
) -> Float32:
    """NeoX RoPE of one element: `x` pairs with `partner` 32 columns away."""
    if first:
        return x * c - partner * s
    return partner * s + x * c


def _assert_rounding_of(got: Scalar[dtype], want: Float32, what: String) raises:
    """`got` must be within one bfloat16 rounding of the fp32 `want`."""
    var err = abs(got.cast[DType.float32]() - want)
    if err > max(abs(want) * 0.0079, 1e-3):
        raise Error(t"{what}: got {got}, fp32 reference {want}")


def _bits_equal(a: Scalar[dtype], b: Scalar[dtype], what: String) raises:
    if a.to_bits() != b.to_bits():
        raise Error(t"{what}: split {a} != padded {b}")


def run_case[
    num_q_heads: Int, num_kv_heads: Int
](prompt_lens: List[Int], cache_lens: List[Int], ctx: DeviceContext,) raises:
    comptime split_params = KVCacheStaticParams(
        num_kv_heads, row_dim, v_head_size=v_dim
    )
    comptime padded_params = KVCacheStaticParams(num_kv_heads, padded_dim)
    comptime SplitCollection = PagedKVCacheCollection[
        dtype,
        split_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    comptime PaddedCollection = PagedKVCacheCollection[
        dtype,
        padded_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    comptime assert split_params.v_col_base() == k_dim

    var batch_size = len(prompt_lens)
    var table = CacheLengthsTable.build(prompt_lens, cache_lens, ctx)
    var total = table.total_length
    var max_context = table.max_full_context_length
    var lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_lens, max_context, num_blocks, ctx
    )
    var lut_host = lut.host_tensor()
    comptime assert lut_host.flat_rank == 2
    var offsets_host = table.input_row_offsets.host_tensor()
    comptime assert offsets_host.flat_rank == 1

    var split_blocks = HostDeviceTileTensor[dtype](
        row_major(
            Int64(num_blocks),
            Idx[1],
            Int64(num_layers),
            Idx[page_size],
            Idx[num_kv_heads],
            Idx[row_dim],
        ),
        ctx,
    )
    var padded_blocks = HostDeviceTileTensor[dtype](
        row_major(
            Int64(num_blocks),
            Idx[2],
            Int64(num_layers),
            Idx[page_size],
            Idx[num_kv_heads],
            Idx[padded_dim],
        ),
        ctx,
    )
    var split_host = split_blocks.host_tensor().fill(0)
    var padded_host = padded_blocks.host_tensor().fill(0)
    comptime assert split_host.flat_rank == 6
    comptime assert padded_host.flat_rank == 6
    # The cache each request already holds.
    for b in range(batch_size):
        for t in range(cache_lens[b]):
            var page, slot = divmod(t, page_size)
            var block = Int(lut_host[b, page])
            for h in range(num_kv_heads):
                for d in range(k_dim):
                    split_host[block, 0, layer, slot, h, d] = _old_k(b, h, t, d)
                    padded_host[block, 0, layer, slot, h, d] = _old_k(
                        b, h, t, d
                    )
                for d in range(v_dim):
                    split_host[block, 0, layer, slot, h, k_dim + d] = _old_v(
                        b, h, t, d
                    )
                    padded_host[block, 1, layer, slot, h, d] = _old_v(
                        b, h, t, d
                    )
    split_blocks.to_device()
    padded_blocks.to_device()

    # New tokens: the split inputs, and the padded path's zero-padded ones.
    var k_new = HostDeviceTileTensor[dtype](
        row_major(Int64(total), Idx[num_kv_heads], Idx[k_dim]), ctx
    )
    var v_new = HostDeviceTileTensor[dtype](
        row_major(Int64(total), Idx[num_kv_heads], Idx[v_dim]), ctx
    )
    var q = HostDeviceTileTensor[dtype](
        row_major(Int64(total), Idx[num_q_heads], Idx[k_dim]), ctx
    )
    var k_pad = HostDeviceTileTensor[dtype](
        row_major(Int64(total), Idx[num_kv_heads], Idx[padded_dim]), ctx
    )
    var v_pad = HostDeviceTileTensor[dtype](
        row_major(Int64(total), Idx[num_kv_heads], Idx[padded_dim]), ctx
    )
    var q_pad = HostDeviceTileTensor[dtype](
        row_major(Int64(total), Idx[num_q_heads], Idx[padded_dim]), ctx
    )
    var k_host = k_new.host_tensor()
    var v_host = v_new.host_tensor()
    var q_host = q.host_tensor()
    var k_pad_host = k_pad.host_tensor().fill(0)
    var v_pad_host = v_pad.host_tensor().fill(0)
    var q_pad_host = q_pad.host_tensor().fill(0)
    comptime assert k_host.flat_rank == 3
    comptime assert v_host.flat_rank == 3
    comptime assert q_host.flat_rank == 3
    comptime assert k_pad_host.flat_rank == 3
    comptime assert v_pad_host.flat_rank == 3
    comptime assert q_pad_host.flat_rank == 3
    for t in range(total):
        for h in range(num_kv_heads):
            for d in range(k_dim):
                k_host[t, h, d] = _new_k(h, t, d)
                k_pad_host[t, h, d] = _new_k(h, t, d)
            for d in range(v_dim):
                v_host[t, h, d] = _new_v(h, t, d)
                v_pad_host[t, h, d] = _new_v(h, t, d)
        for h in range(num_q_heads):
            for d in range(k_dim):
                q_host[t, h, d] = _q(h, t, d)
                q_pad_host[t, h, d] = _q(h, t, d)
    k_new.to_device()
    v_new.to_device()
    q.to_device()
    k_pad.to_device()
    v_pad.to_device()
    q_pad.to_device()

    var freqs = HostDeviceTileTensor[DType.float32](
        row_major(Int64(max_pos), Idx[rope_dim]), ctx
    )
    var freqs_host = freqs.host_tensor()
    comptime assert freqs_host.flat_rank == 2
    for p in range(max_pos):
        for j in range(rope_dim // 2):
            var theta = Float32(p) * (
                Float32(10000.0) ** (Float32(-2 * j) / Float32(rope_dim))
            )
            freqs_host[p, 2 * j] = cos(theta)
            freqs_host[p, 2 * j + 1] = sin(theta)
    freqs.to_device()

    var q_out = HostDeviceTileTensor[dtype](
        row_major(Int64(total), Idx[num_q_heads], Idx[k_dim]), ctx
    )
    var q_pad_out = HostDeviceTileTensor[dtype](
        row_major(Int64(total), Idx[num_q_heads], Idx[padded_dim]), ctx
    )

    var split = SplitCollection(
        split_blocks.device_tensor().as_unsafe_any_origin(),
        table.cache_lengths.device_tile_tensor(),
        lut.device_tile_tensor(),
        UInt32(table.max_seq_length_batch),
        UInt32(max_context),
    )
    var padded = PaddedCollection(
        padded_blocks.device_tensor().as_unsafe_any_origin(),
        table.cache_lengths.device_tile_tensor(),
        lut.device_tile_tensor(),
        UInt32(table.max_seq_length_batch),
        UInt32(max_context),
    )
    var offsets = table.input_row_offsets.device_tile_tensor()

    var k_dev = k_new.device_tensor()
    var v_dev = v_new.device_tensor()
    var k_pad_dev = k_pad.device_tensor()
    var v_pad_dev = v_pad.device_tensor()

    @inline(.always)
    def k_fn[
        width: Int, alignment: Int
    ](idx: Coord) {var k_dev} -> SIMD[dtype, width]:
        return k_dev.load[width=width](idx)

    @inline(.always)
    def v_fn[
        width: Int, alignment: Int
    ](idx: Coord) {var v_dev} -> SIMD[dtype, width]:
        return v_dev.load[width=width](idx)

    @inline(.always)
    def k_pad_fn[
        width: Int, alignment: Int
    ](idx: Coord) {var k_pad_dev} -> SIMD[dtype, width]:
        return k_pad_dev.load[width=width](idx)

    @inline(.always)
    def v_pad_fn[
        width: Int, alignment: Int
    ](idx: Coord) {var v_pad_dev} -> SIMD[dtype, width]:
        return v_pad_dev.load[width=width](idx)

    # The split path: the key at column 0, the value after it.
    kv_cache_store_ragged[target="gpu"](
        split.get_key_cache(layer),
        (total, num_kv_heads, k_dim),
        offsets,
        k_fn,
        ctx,
    )
    kv_cache_store_ragged[
        target="gpu", head_dim_offset=split_params.v_col_base()
    ](
        split.get_value_cache(layer),
        (total, num_kv_heads, v_dim),
        offsets,
        v_fn,
        ctx,
    )
    fused_qk_rope_ragged[
        SplitCollection.CacheType, interleaved=False, target="gpu"
    ](
        q_proj=q.device_tensor(),
        input_row_offsets=offsets,
        kv_collection=split,
        freqs_cis=freqs.device_tensor(),
        position_ids=None,
        layer_idx=UInt32(layer),
        output=q_out.device_tensor(),
        context=ctx,
    )

    # The padded production path.
    kv_cache_store_ragged[target="gpu"](
        padded.get_key_cache(layer),
        (total, num_kv_heads, padded_dim),
        offsets,
        k_pad_fn,
        ctx,
    )
    kv_cache_store_ragged[target="gpu"](
        padded.get_value_cache(layer),
        (total, num_kv_heads, padded_dim),
        offsets,
        v_pad_fn,
        ctx,
    )
    fused_qk_rope_ragged[
        PaddedCollection.CacheType, interleaved=False, target="gpu"
    ](
        q_proj=q_pad.device_tensor(),
        input_row_offsets=offsets,
        kv_collection=padded,
        freqs_cis=freqs.device_tensor(),
        position_ids=None,
        layer_idx=UInt32(layer),
        output=q_pad_out.device_tensor(),
        context=ctx,
    )
    ctx.synchronize()
    split_blocks.to_host()
    padded_blocks.to_host()
    q_out.to_host()
    q_pad_out.to_host()

    var split_out = split_blocks.host_tensor()
    var padded_out = padded_blocks.host_tensor()
    var q_split = q_out.host_tensor()
    var q_padded = q_pad_out.host_tensor()
    comptime assert split_out.flat_rank == 6
    comptime assert padded_out.flat_rank == 6
    comptime assert q_split.flat_rank == 3
    comptime assert q_padded.flat_rank == 3
    comptime half = rope_dim // 2

    for b in range(batch_size):
        var start = Int(offsets_host[b])
        var cache_len = cache_lens[b]
        for t in range(cache_len + prompt_lens[b]):
            var page, slot = divmod(t, page_size)
            var block = Int(lut_host[b, page])
            var is_new = t >= cache_len
            var row = start + t - cache_len
            for h in range(num_kv_heads):
                for d in range(k_dim):
                    var got = split_out[block, 0, layer, slot, h, d]
                    _bits_equal(
                        got, padded_out[block, 0, layer, slot, h, d], "key"
                    )
                    if not is_new:
                        assert_equal(
                            got.to_bits(), _old_k(b, h, t, d).to_bits()
                        )
                    elif d >= rope_dim:
                        assert_equal(got.to_bits(), _new_k(h, row, d).to_bits())
                    else:
                        var j = d % half
                        var c = freqs_host[t, 2 * j]
                        var s = freqs_host[t, 2 * j + 1]
                        var x = _new_k(h, row, d).cast[DType.float32]()
                        var partner = _new_k(
                            h, row, d + half if d < half else d - half
                        ).cast[DType.float32]()
                        _assert_rounding_of(
                            got,
                            _rotated(x, partner, c, s, d < half),
                            "roped key",
                        )
                for d in range(v_dim):
                    var got = split_out[block, 0, layer, slot, h, k_dim + d]
                    _bits_equal(
                        got, padded_out[block, 1, layer, slot, h, d], "value"
                    )
                    var want = _new_v(h, row, d) if is_new else _old_v(
                        b, h, t, d
                    )
                    assert_equal(got.to_bits(), want.to_bits())
        for r in range(prompt_lens[b]):
            var row = start + r
            var pos = cache_len + r
            for h in range(num_q_heads):
                for d in range(k_dim):
                    var got = q_split[row, h, d]
                    _bits_equal(got, q_padded[row, h, d], "query")
                    if d >= rope_dim:
                        assert_equal(got.to_bits(), _q(h, row, d).to_bits())
                    else:
                        var j = d % half
                        var partner = _q(
                            h, row, d + half if d < half else d - half
                        ).cast[DType.float32]()
                        _assert_rounding_of(
                            got,
                            _rotated(
                                _q(h, row, d).cast[DType.float32](),
                                partner,
                                freqs_host[pos, 2 * j],
                                freqs_host[pos, 2 * j + 1],
                                d < half,
                            ),
                            "roped query",
                        )


def run_cases[num_q_heads: Int, num_kv_heads: Int](ctx: DeviceContext) raises:
    # A chunked prefill: an empty cache, a chunk that crosses a page
    # boundary, and a chunk on top of a long cache.
    run_case[num_q_heads, num_kv_heads]([9, 130, 300], [0, 117, 1000], ctx)
    # Decode and DFlash-verify widths at page edges and deep in the cache.
    run_case[num_q_heads, num_kv_heads]([1, 8, 1], [511, 4000, 127], ctx)


def main() raises:
    with DeviceContext() as ctx:
        # Full-attention group (group 16) at TP4, the sliding group (group 8)
        # at TP4, and the sliding group at DP8.
        run_cases[16, 1](ctx)
        run_cases[16, 2](ctx)
        run_cases[64, 8](ctx)
