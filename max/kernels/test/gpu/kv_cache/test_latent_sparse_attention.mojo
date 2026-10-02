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
"""GPU test for `latent_sparse_attention_ragged_paged`.

Two paged leaves of one shared latent head -- a window leaf paged by token and
a compressed leaf paged by entry -- read by the GPU kernel and by a plain
full-softmax reference on the host built from the same host-side cache views.
Covers a ragged prefill batch, a decode batch with non-zero cache lengths,
`-1` padding in the compressed list, and an empty compressed list.
"""

from std.math import ceildiv, exp
from std.random import randn_float64, seed
from std.testing import assert_almost_equal
from std.utils.numerics import min_or_neg_inf

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, TileTensor, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.attention.latent_sparse_attention import (
    latent_sparse_attention_ragged_paged,
)

from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable

comptime dtype = DType.float32
comptime HEAD_DIM = 512
comptime NUM_HEADS = 64
comptime WINDOW = 128
comptime PAGE_SIZE = 128
comptime COMP_SLOTS = 32
comptime NUM_LAYERS = 2
comptime kv_params = KVCacheStaticParams(
    num_heads=1, head_size=HEAD_DIM, is_mla=True
)
comptime Collection[page_size: Int] = PagedKVCacheCollection[
    dtype,
    kv_params,
    page_size,
    MutAnyOrigin,
    ImmutAnyOrigin,
    ImmutAnyOrigin,
    MutAnyOrigin,
]


def _fill_random(tensor: TileTensor[mut=True, ...]):
    for i in range(tensor.num_elements()):
        tensor.unsafe_ptr().unsafe_store(
            i, Scalar[tensor.dtype](randn_float64())
        )


def run_case(
    ctx: DeviceContext,
    prompt_lens: List[Int],
    cache_lens: List[Int],
    num_comp: Int,
    layer_swa: Int,
    layer_comp: Int,
) raises:
    seed(0x5EED)
    var batch_size = len(prompt_lens)
    comptime scale = Float32(0.044194173824159216)  # 512 ** -0.5

    # --- window leaf, paged by token -------------------------------------
    var swa_lengths = CacheLengthsTable.build(prompt_lens, cache_lens, ctx)
    var total_rows = swa_lengths.total_length
    var swa_pages = (
        ceildiv(swa_lengths.max_full_context_length, PAGE_SIZE) * batch_size
    )
    var swa_blocks = HostDeviceTileTensor[dtype](
        row_major(
            Coord(
                Int64(swa_pages),
                Idx[1],
                Int64(NUM_LAYERS),
                Idx[PAGE_SIZE],
                Idx[1],
                Idx[HEAD_DIM],
            )
        ),
        ctx,
    )
    _fill_random(swa_blocks.host_tensor())
    swa_blocks.to_device()
    var swa_lut = PagedLookupTable[PAGE_SIZE].build(
        prompt_lens,
        cache_lens,
        swa_lengths.max_full_context_length,
        swa_pages,
        ctx,
    )

    # --- compressed leaf, paged by entry ---------------------------------
    # Entry counts stand in for token counts when building its tables.
    var comp_prompt = List[Int]()
    var comp_cache = List[Int]()
    var comp_total = List[Int]()
    for b in range(batch_size):
        comp_prompt.append(max(prompt_lens[b] // 4, 1))
        comp_cache.append(cache_lens[b] // 4)
        comp_total.append(comp_prompt[b] + comp_cache[b])
    var comp_lengths = CacheLengthsTable.build(comp_prompt, comp_cache, ctx)
    var comp_pages = (
        ceildiv(comp_lengths.max_full_context_length, COMP_SLOTS) * batch_size
    )
    var comp_blocks = HostDeviceTileTensor[dtype](
        row_major(
            Coord(
                Int64(comp_pages),
                Idx[1],
                Int64(NUM_LAYERS),
                Idx[COMP_SLOTS],
                Idx[1],
                Idx[HEAD_DIM],
            )
        ),
        ctx,
    )
    _fill_random(comp_blocks.host_tensor())
    comp_blocks.to_device()
    var comp_lut = PagedLookupTable[COMP_SLOTS].build(
        comp_prompt,
        comp_cache,
        comp_lengths.max_full_context_length,
        comp_pages,
        ctx,
    )

    # --- q, sink, compressed indices -------------------------------------
    var q_layout = row_major(
        Coord(Int64(total_rows), Idx[NUM_HEADS], Idx[HEAD_DIM])
    )
    var q = HostDeviceTileTensor[dtype](q_layout, ctx)
    _fill_random(q.host_tensor())
    q.to_device()
    var out = HostDeviceTileTensor[dtype](q_layout, ctx)
    _ = out.host_tensor().fill(0)
    out.to_device()

    var sink = HostDeviceTileTensor[.float32](row_major[NUM_HEADS](), ctx)
    _fill_random(sink.host_tensor())
    sink.to_device()

    var idx_cols = max(num_comp, 1)
    var idx = HostDeviceTileTensor[.int32](
        row_major(Coord(Int64(total_rows), Int64(idx_cols))), ctx
    )
    var idx_host = idx.host_tensor()
    var row_offsets_host = swa_lengths.input_row_offsets.host_tile_tensor()
    for t in range(total_rows):
        var b = 0
        while UInt32(t) >= row_offsets_host[b + 1]:
            b += 1
        for j in range(idx_cols):
            # Cycle through this sequence's entries; every other row drops
            # its last slot to exercise the -1 skip.
            var take = num_comp - (t % 2)
            if j < take and j < comp_total[b]:
                idx_host[t, j] = Int32((t * 7 + j * 3) % comp_total[b])
            else:
                idx_host[t, j] = Int32(-1)
    idx.to_device()

    # --- device run --------------------------------------------------------
    # The collections spell their block strides symbolically in `kv_params`,
    # which the compiler cannot fold against `row_major`'s; the two layouts
    # are structurally identical.
    comptime SwaCollection = Collection[PAGE_SIZE]
    comptime CompCollection = Collection[COMP_SLOTS]
    var swa_dev = SwaCollection(
        rebind[SwaCollection.blocks_tt_type](
            swa_blocks.device_tensor().as_unsafe_any_origin()
        ),
        swa_lengths.cache_lengths.device_tile_tensor(),
        swa_lut.device_tile_tensor(),
        UInt32(swa_lengths.max_seq_length_batch),
        UInt32(swa_lengths.max_full_context_length),
    )
    var comp_dev = CompCollection(
        rebind[CompCollection.blocks_tt_type](
            comp_blocks.device_tensor().as_unsafe_any_origin()
        ),
        comp_lengths.cache_lengths.device_tile_tensor(),
        comp_lut.device_tile_tensor(),
        UInt32(comp_lengths.max_seq_length_batch),
        UInt32(comp_lengths.max_full_context_length),
    )
    # With num_comp == 0 the single column is all -1, so nothing is read
    # from the compressed leaf; the graph-level test covers a true [rows, 0].
    latent_sparse_attention_ragged_paged[target="gpu", window=WINDOW](
        out.device_tensor(),
        q.device_tensor(),
        swa_lengths.input_row_offsets.device_tile_tensor(),
        idx.device_tensor(),
        sink.device_tensor(),
        swa_dev.get_key_cache(layer_swa),
        comp_dev.get_key_cache(layer_comp),
        scale,
        ctx,
    )
    ctx.synchronize()
    out.to_host()

    # --- host reference: plain two-pass softmax ------------------------------
    var swa_host = SwaCollection(
        rebind[SwaCollection.blocks_tt_type](
            swa_blocks.host_tensor().as_unsafe_any_origin()
        ),
        swa_lengths.cache_lengths.host_tile_tensor(),
        swa_lut.host_tile_tensor(),
        UInt32(swa_lengths.max_seq_length_batch),
        UInt32(swa_lengths.max_full_context_length),
    ).get_key_cache(layer_swa)
    var comp_host = CompCollection(
        rebind[CompCollection.blocks_tt_type](
            comp_blocks.host_tensor().as_unsafe_any_origin()
        ),
        comp_lengths.cache_lengths.host_tile_tensor(),
        comp_lut.host_tile_tensor(),
        UInt32(comp_lengths.max_seq_length_batch),
        UInt32(comp_lengths.max_full_context_length),
    ).get_key_cache(layer_comp)
    var q_host = q.host_tensor()
    var out_host = out.host_tensor()
    var sink_host = sink.host_tensor()

    var max_keys = WINDOW + idx_cols
    var scores = List[Float64](capacity=max_keys)
    for t in range(total_rows):
        var b = 0
        while UInt32(t) >= row_offsets_host[b + 1]:
            b += 1
        var pos = cache_lens[b] + t - Int(row_offsets_host[b])
        var start = max(pos - WINDOW + 1, 0)
        for h in range(NUM_HEADS):
            scores.clear()
            var m = Float64(min_or_neg_inf[DType.float64]())
            # scores over window keys then compressed keys
            for p in range(start, pos + 1):
                var s = Float64(0)
                for d in range(HEAD_DIM):
                    s += Float64(q_host[t, h, d]) * Float64(
                        swa_host.load[width=1](b, 0, p, d)[0]
                    )
                s *= Float64(scale)
                scores.append(s)
                m = max(m, s)
            for j in range(num_comp):
                var e = Int(idx_host[t, j])
                if e < 0:
                    continue
                var s = Float64(0)
                for d in range(HEAD_DIM):
                    s += Float64(q_host[t, h, d]) * Float64(
                        comp_host.load[width=1](b, 0, e, d)[0]
                    )
                s *= Float64(scale)
                scores.append(s)
                m = max(m, s)
            var den = exp(Float64(sink_host[h]) - m)
            for i in range(len(scores)):
                den += exp(scores[i] - m)
            for d in range(HEAD_DIM):
                var o = Float64(0)
                var i = 0
                for p in range(start, pos + 1):
                    o += exp(scores[i] - m) * Float64(
                        swa_host.load[width=1](b, 0, p, d)[0]
                    )
                    i += 1
                for j in range(num_comp):
                    var e = Int(idx_host[t, j])
                    if e < 0:
                        continue
                    o += exp(scores[i] - m) * Float64(
                        comp_host.load[width=1](b, 0, e, d)[0]
                    )
                    i += 1
                assert_almost_equal(
                    Float64(out_host[t, h, d]),
                    o / den,
                    rtol=1e-4,
                    atol=1e-5,
                    msg="row "
                    + String(t)
                    + " head "
                    + String(h)
                    + " dim "
                    + String(d),
                )

    # The collections and `row_offsets_host` hold untracked views of these.
    _ = swa_blocks^
    _ = comp_blocks^
    _ = swa_lengths^
    _ = comp_lengths^
    _ = swa_lut^
    _ = comp_lut^


def main() raises:
    with DeviceContext() as ctx:
        # Ragged prefill from empty caches: one row past the window, one under.
        run_case(ctx, [200, 40], [0, 0], 16, 1, 0)
        # Decode: one row per sequence, non-zero cache lengths.
        run_case(ctx, [1, 1, 1], [300, 5, 130], 48, 0, 1)
        # Chunked extend with no compressed entries at all.
        run_case(ctx, [64, 10], [70, 500], 0, 1, 1)
    print("OK")
