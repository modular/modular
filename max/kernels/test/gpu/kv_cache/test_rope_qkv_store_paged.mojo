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
"""Head-major RoPE + paged K/V store kernels against the unfused kernels.

`rope_qkv_store_paged_ragged` and `k_rope_v_store_paged_ragged` replace a K
store, a V store of the FP32-scaled value, and `fused_qk_rope_ragged`'s
in-cache key rotation. Each case writes a ragged batch on top of an existing
cache through both, into two caches that start identical, and requires:

- every byte of the two caches equal, so the rotated keys, the scaled values,
  the existing tokens and every untouched slot all agree;
- the rotated queries equal to the unfused ones, bitwise;
- the projections' tail row, past the final ragged offset, written nowhere.
"""

from std.math import cos, sin

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Coord, Idx, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.chunked_qkv_rope_store import (
    k_rope_v_store_paged_ragged,
    rope_qkv_store_paged_ragged,
)
from nn.fused_qk_rope import fused_qk_rope_ragged
from nn.kv_cache_ragged import kv_cache_store_ragged
from std.testing import assert_equal

from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable

comptime dtype = DType.bfloat16
comptime page_size = 128
comptime head_dim = 128
comptime num_layers = 2
comptime layer = 1
comptime num_blocks = 16
comptime max_pos = 512
# Not representable in BF16, so a reference that rounded the scale first
# would differ.
comptime value_scale = Float32(0.612)
comptime q_sentinel = Scalar[dtype](-7.0)


def _val(seed: Int, i: Int, j: Int, d: Int) -> Scalar[dtype]:
    # Multiples of 1/8 below 16 in magnitude: exact in bfloat16.
    return Scalar[dtype](
        Float32((seed * 101 + i * 37 + j * 19 + d * 7) % 253 - 126) * 0.125
    )


def run_case[
    num_q_heads: Int,
    num_kv_heads: Int,
    rope_dim: Int,
    freq_dtype: DType,
    with_q: Bool,
](prompt_lens: List[Int], cache_lens: List[Int], ctx: DeviceContext) raises:
    comptime params = KVCacheStaticParams(num_kv_heads, head_dim)
    comptime Collection = PagedKVCacheCollection[
        dtype,
        params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]

    var batch_size = len(prompt_lens)
    var table = CacheLengthsTable.build(prompt_lens, cache_lens, ctx)
    var total = table.total_length
    # One row past the final ragged offset, which only the fused path sees.
    var rows = total + 1
    var max_context = table.max_full_context_length
    var lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_lens, max_context, num_blocks, ctx
    )
    var lut_host = lut.host_tensor()
    comptime assert lut_host.flat_rank == 2

    var fused_blocks = HostDeviceTileTensor[dtype](
        row_major(
            Int64(num_blocks),
            Idx[2],
            Int64(num_layers),
            Idx[page_size],
            Idx[num_kv_heads],
            Idx[head_dim],
        ),
        ctx,
    )
    var ref_blocks = HostDeviceTileTensor[dtype](
        row_major(
            Int64(num_blocks),
            Idx[2],
            Int64(num_layers),
            Idx[page_size],
            Idx[num_kv_heads],
            Idx[head_dim],
        ),
        ctx,
    )
    var fused_host = fused_blocks.host_tensor().fill(0)
    var ref_host = ref_blocks.host_tensor().fill(0)
    comptime assert fused_host.flat_rank == 6
    comptime assert ref_host.flat_rank == 6
    # The cache each request already holds, in both layers.
    for b in range(batch_size):
        for t in range(cache_lens[b]):
            var page, slot = divmod(t, page_size)
            var block = Int(lut_host[b, page])
            for kv in range(2):
                for l in range(num_layers):
                    for h in range(num_kv_heads):
                        for d in range(head_dim):
                            var x = _val(3 + kv + 2 * l + 4 * b, t, h, d)
                            fused_host[block, kv, l, slot, h, d] = x
                            ref_host[block, kv, l, slot, h, d] = x
    fused_blocks.to_device()
    ref_blocks.to_device()

    # The fused path's projections carry the tail row; the reference's stop
    # at `total`, which is all the unfused kernels are given.
    var q = HostDeviceTileTensor[dtype](
        row_major(Int64(rows), Idx[num_q_heads], Idx[head_dim]), ctx
    )
    var k = HostDeviceTileTensor[dtype](
        row_major(Int64(rows), Idx[num_kv_heads], Idx[head_dim]), ctx
    )
    var v = HostDeviceTileTensor[dtype](
        row_major(Int64(rows), Idx[num_kv_heads], Idx[head_dim]), ctx
    )
    var q_ref = HostDeviceTileTensor[dtype](
        row_major(Int64(total), Idx[num_q_heads], Idx[head_dim]), ctx
    )
    var k_ref = HostDeviceTileTensor[dtype](
        row_major(Int64(total), Idx[num_kv_heads], Idx[head_dim]), ctx
    )
    var v_ref = HostDeviceTileTensor[dtype](
        row_major(Int64(total), Idx[num_kv_heads], Idx[head_dim]), ctx
    )
    var q_host = q.host_tensor()
    var k_host = k.host_tensor()
    var v_host = v.host_tensor()
    var q_ref_host = q_ref.host_tensor()
    var k_ref_host = k_ref.host_tensor()
    var v_ref_host = v_ref.host_tensor()
    comptime assert q_host.flat_rank == 3
    comptime assert k_host.flat_rank == 3
    comptime assert v_host.flat_rank == 3
    comptime assert q_ref_host.flat_rank == 3
    comptime assert k_ref_host.flat_rank == 3
    comptime assert v_ref_host.flat_rank == 3
    for t in range(rows):
        for h in range(num_q_heads):
            for d in range(head_dim):
                q_host[t, h, d] = _val(11, t, h, d)
                if t < total:
                    q_ref_host[t, h, d] = _val(11, t, h, d)
        for h in range(num_kv_heads):
            for d in range(head_dim):
                k_host[t, h, d] = _val(1, t, h, d)
                v_host[t, h, d] = _val(2, t, h, d)
                if t < total:
                    k_ref_host[t, h, d] = _val(1, t, h, d)
                    v_ref_host[t, h, d] = _val(2, t, h, d)
    q.to_device()
    k.to_device()
    v.to_device()
    q_ref.to_device()
    k_ref.to_device()
    v_ref.to_device()

    var freqs = HostDeviceTileTensor[freq_dtype](
        row_major(Int64(max_pos), Idx[rope_dim]), ctx
    )
    var freqs_host = freqs.host_tensor()
    comptime assert freqs_host.flat_rank == 2
    for p in range(max_pos):
        for j in range(rope_dim // 2):
            var theta = Float32(p) * (
                Float32(10000.0) ** (Float32(-2 * j) / Float32(rope_dim))
            )
            freqs_host[p, 2 * j] = cos(theta).cast[freq_dtype]()
            freqs_host[p, 2 * j + 1] = sin(theta).cast[freq_dtype]()
    freqs.to_device()

    var q_out = HostDeviceTileTensor[dtype](
        row_major(Int64(rows), Idx[num_q_heads], Idx[head_dim]), ctx
    )
    var q_out_host = q_out.host_tensor().fill(q_sentinel)
    comptime assert q_out_host.flat_rank == 3
    q_out.to_device()
    # Without a query, the reference ropes the keys through the query slot,
    # as the graph it replaces does, and discards the result.
    comptime ref_q_heads = num_q_heads if with_q else num_kv_heads
    var q_ref_out = HostDeviceTileTensor[dtype](
        row_major(Int64(total), Idx[ref_q_heads], Idx[head_dim]), ctx
    )

    var fused = Collection(
        fused_blocks.device_tensor().as_unsafe_any_origin(),
        table.cache_lengths.device_tile_tensor(),
        lut.device_tile_tensor(),
        UInt32(table.max_seq_length_batch),
        UInt32(max_context),
    )
    var reference = Collection(
        ref_blocks.device_tensor().as_unsafe_any_origin(),
        table.cache_lengths.device_tile_tensor(),
        lut.device_tile_tensor(),
        UInt32(table.max_seq_length_batch),
        UInt32(max_context),
    )
    var offsets = table.input_row_offsets.device_tile_tensor()

    comptime if with_q:
        rope_qkv_store_paged_ragged(
            q.device_tensor().as_imm(),
            k.device_tensor().as_imm(),
            v.device_tensor().as_imm(),
            offsets,
            freqs.device_tensor().as_imm(),
            fused,
            UInt32(layer),
            q_out.device_tensor(),
            value_scale,
            ctx,
        )
    else:
        k_rope_v_store_paged_ragged(
            k.device_tensor().as_imm(),
            v.device_tensor().as_imm(),
            offsets,
            freqs.device_tensor().as_imm(),
            fused,
            UInt32(layer),
            value_scale,
            ctx,
        )

    var k_dev = k_ref.device_tensor()
    var v_dev = v_ref.device_tensor()

    @inline(.always)
    def k_fn[
        width: Int, alignment: Int
    ](idx: Coord) {var k_dev} -> SIMD[dtype, width]:
        return k_dev.load[width=width](idx)

    @inline(.always)
    def v_fn[
        width: Int, alignment: Int
    ](idx: Coord) {var v_dev} -> SIMD[dtype, width]:
        return (
            v_dev.load[width=width](idx).cast[DType.float32]() * value_scale
        ).cast[dtype]()

    kv_cache_store_ragged(
        reference.get_key_cache(layer),
        (total, num_kv_heads, head_dim),
        offsets,
        k_fn,
        ctx,
    )
    kv_cache_store_ragged(
        reference.get_value_cache(layer),
        (total, num_kv_heads, head_dim),
        offsets,
        v_fn,
        ctx,
    )
    comptime if with_q:
        fused_qk_rope_ragged[Collection.CacheType, interleaved=False](
            q_proj=q_ref.device_tensor(),
            input_row_offsets=offsets,
            kv_collection=reference,
            freqs_cis=freqs.device_tensor(),
            position_ids=None,
            layer_idx=UInt32(layer),
            output=q_ref_out.device_tensor(),
            context=ctx,
        )
    else:
        fused_qk_rope_ragged[Collection.CacheType, interleaved=False](
            q_proj=k_ref.device_tensor(),
            input_row_offsets=offsets,
            kv_collection=reference,
            freqs_cis=freqs.device_tensor(),
            position_ids=None,
            layer_idx=UInt32(layer),
            output=q_ref_out.device_tensor(),
            context=ctx,
        )
    ctx.synchronize()
    fused_blocks.to_host()
    ref_blocks.to_host()
    q_out.to_host()
    q_ref_out.to_host()

    var fused_out = fused_blocks.host_tensor()
    var ref_out = ref_blocks.host_tensor()
    comptime assert fused_out.flat_rank == 6
    comptime assert ref_out.flat_rank == 6
    var differ = 0
    for blk in range(num_blocks):
        for kv in range(2):
            for l in range(num_layers):
                for s in range(page_size):
                    for h in range(num_kv_heads):
                        for d in range(head_dim):
                            if (
                                fused_out[blk, kv, l, s, h, d].to_bits()
                                != ref_out[blk, kv, l, s, h, d].to_bits()
                            ):
                                differ += 1
    assert_equal(differ, 0, "cache bytes differ from the unfused kernels")

    comptime if with_q:
        var q_fused = q_out.host_tensor()
        var q_unfused = q_ref_out.host_tensor()
        comptime assert q_fused.flat_rank == 3
        comptime assert q_unfused.flat_rank == 3
        for t in range(rows):
            for h in range(num_q_heads):
                for d in range(head_dim):
                    var got = q_fused[t, h, d].to_bits()
                    var want = q_sentinel.to_bits()
                    if t < total:
                        want = q_unfused[t, h, d].to_bits()
                    assert_equal(got, want, "rotated query")


def run_cases[
    num_q_heads: Int, num_kv_heads: Int, rope_dim: Int, freq_dtype: DType
](ctx: DeviceContext) raises:
    comptime for with_q in [True, False]:
        # A prefill chunk on an empty cache, one that crosses two page
        # boundaries, and a decode token in a page's last slot.
        run_case[num_q_heads, num_kv_heads, rope_dim, freq_dtype, with_q](
            [9, 140, 1], [0, 117, 127], ctx
        )
        # DFlash-verify widths: a block of 8 straddling a page boundary.
        run_case[num_q_heads, num_kv_heads, rope_dim, freq_dtype, with_q](
            [8, 8], [124, 250], ctx
        )


def main() raises:
    with DeviceContext() as ctx:
        # The MiMo-V2.6 drafter at TP4, through the FP32 table its context
        # writer builds and the BF16 table the target's layers carry.
        run_cases[16, 2, 64, DType.float32](ctx)
        run_cases[16, 2, 64, DType.bfloat16](ctx)
        # Full rotary, where no lane passes through unrotated.
        run_cases[8, 1, 128, DType.float32](ctx)
