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
"""AMD sliding-window decode (S == 1) and speculative-verify fold (S > 1).

Drives `flash_attention[ragged=True]` over a paged cache with a
`SlidingWindowCausalMask` or a `SlidingWindowNonCausalMask` (a windowed draft
block that also sees its later tokens) and compares against a CPU reference
that visits only each token's visible keys. Keys older than every token's
window are poisoned with a large magnitude, so a kernel that streams (and fails
to mask) them, or that starts its split-K range at the wrong key, corrupts the
output instead of averaging out.

Run the binary with `--benchmark` to time the draft-model shape, optionally
with `--partitions=N` to override the split-K count.
"""

from std.math import ceildiv, exp
from std.sys import argv
from std.sys.defines import get_defined_int
from std.memory import unsafe_memset_zero
from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from kv_cache_test_utils import padded_lut_cols, random_distinct
from layout import Idx, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.attention.gpu.mha import _mha_decode_fold_ok, flash_attention
from nn.attention.mha_mask import (
    SlidingWindowCausalMask,
    SlidingWindowNonCausalMask,
)
from std.random import seed
from std.testing import assert_true

comptime dtype = DType.bfloat16
comptime depth = 128
comptime num_heads = 16
comptime kv_params = KVCacheStaticParams(num_heads=1, head_size=depth)
comptime page_size = get_defined_int["page_size", 128]()
comptime scale = Float32(0.5)
comptime poison_gain = Float32(8)


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark":
            return True
    return False


def partitions_override() raises -> Optional[Int]:
    """Split-K count from `--partitions=N`, else the dispatch heuristic's."""
    for arg in argv():
        if arg.startswith("--partitions="):
            return Int(arg.removeprefix("--partitions="))
    return None


def hash_unit(index: Int, stream: Int) -> Float32:
    """Deterministic value in `[-1, 1)` per `(index, stream)`."""
    var x = UInt64(index) * UInt64(0x9E3779B97F4A7C15)
    x += UInt64(stream) * UInt64(0xD1B54A32D192ED03)
    x ^= x >> 30
    x *= UInt64(0xBF58476D1CE4E5B9)
    x ^= x >> 27
    x *= UInt64(0x94D049BB133111EB)
    x ^= x >> 31
    return Float32(Int(x >> 40)) / Float32(1 << 23) - 1.0


def run_case[
    window: Int, causal: Bool = True
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    ctx: DeviceContext[],
    benchmark: Bool = False,
) raises:
    var batch_size = len(valid_lengths)
    var total_q = 0
    var total_keys = 0
    var max_prompt_length = 0
    var max_context = 0
    var total_pages = 0
    var max_pages = 0
    var num_keys = List[Int]()
    for b in range(batch_size):
        var keys = cache_lengths[b] + valid_lengths[b]
        num_keys.append(keys)
        total_q += valid_lengths[b]
        total_keys += keys
        max_prompt_length = max(max_prompt_length, valid_lengths[b])
        max_context = max(max_context, keys)
        total_pages += ceildiv(keys, page_size)
        max_pages = max(max_pages, ceildiv(keys, page_size))

    print(
        "window:",
        window,
        "batch:",
        batch_size,
        "S:",
        max_prompt_length,
        "valid:",
        String(valid_lengths),
        "cache:",
        String(cache_lengths),
    )

    var q_layout = row_major(total_q, Idx[num_heads], Idx[depth])
    var lengths_layout = row_major(Int64(batch_size))
    var input_row_offsets = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size + 1)), ctx
    )
    var cache_lengths_tensor = HostDeviceTileTensor[.uint32](
        lengths_layout, ctx
    )
    var q_ragged = HostDeviceTileTensor[dtype](q_layout, ctx)
    var test_output = HostDeviceTileTensor[dtype](q_layout, ctx)

    var offsets_host = input_row_offsets.host_tensor()
    var cache_host = cache_lengths_tensor.host_tensor()
    var running = 0
    for b in range(batch_size):
        offsets_host[b] = UInt32(running)
        cache_host[b] = UInt32(cache_lengths[b])
        running += valid_lengths[b]
    offsets_host[batch_size] = UInt32(running)
    input_row_offsets.to_device()
    cache_lengths_tensor.to_device()

    var q_ptr = q_ragged.host_tensor().unsafe_ptr()
    for i in range(total_q * num_heads * depth):
        q_ptr[unsafe_offset=i] = Scalar[dtype](hash_unit(i, 1))
    q_ragged.to_device()
    var out_init = test_output.host_tensor().unsafe_ptr()
    for i in range(total_q * num_heads * depth):
        out_init[unsafe_offset=i] = Scalar[dtype](7)
    test_output.to_device()

    # Dense K/V on the host, `[sum(num_keys), depth]` (one KV head), kept for
    # the reference. A key is poisoned when no token of its sequence can see it.
    var k_dense = List[Scalar[dtype]](length=total_keys * depth, fill=0)
    var v_dense = List[Scalar[dtype]](length=total_keys * depth, fill=0)
    var key_base = List[Int]()
    var base = 0
    for b in range(batch_size):
        key_base.append(base)
        var first_visible = max(cache_lengths[b] - window + 1, 0)
        for j in range(num_keys[b]):
            var gain = poison_gain if j < first_visible else Float32(1)
            for d in range(depth):
                var idx = (base + j) * depth + d
                k_dense[idx] = Scalar[dtype](hash_unit(idx, 2) * gain)
                v_dense[idx] = Scalar[dtype](hash_unit(idx, 3) * gain)
        base += num_keys[b]

    var num_blocks = total_pages + 2
    var kv_block = HostDeviceTileTensor[dtype](
        row_major(
            Int64(num_blocks),
            Idx[2],
            Idx[1],
            Idx[page_size],
            Idx[1],
            Idx[depth],
        ),
        ctx,
    )
    var lut_cols = padded_lut_cols(max_pages)
    var paged_lut = HostDeviceTileTensor[.uint32](
        row_major(Int64(batch_size), Int64(lut_cols)), ctx
    )
    var kv_ptr = kv_block.host_tensor().unsafe_ptr()
    unsafe_memset_zero(kv_ptr, num_blocks * 2 * page_size * depth)
    var lut_host = paged_lut.host_tensor()
    var physical = random_distinct(num_blocks, total_pages)
    var page_pos = 0
    for b in range(batch_size):
        for page in range(ceildiv(num_keys[b], page_size)):
            var blk = physical[page_pos]
            page_pos += 1
            lut_host[b, page] = UInt32(blk)
            for tok in range(min(page_size, num_keys[b] - page * page_size)):
                var src = (key_base[b] + page * page_size + tok) * depth
                for kv in range(2):
                    var dst = ((blk * 2 + kv) * page_size + tok) * depth
                    for d in range(depth):
                        kv_ptr[unsafe_offset=dst + d] = (
                            k_dense[src + d] if kv == 0 else v_dense[src + d]
                        )
    kv_block.to_device()
    paged_lut.to_device()

    comptime PagedCollection = PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var collection = PagedCollection(
        kv_block.device_tensor().as_unsafe_any_origin(),
        cache_lengths_tensor.device_tensor().as_imm().as_unsafe_any_origin(),
        paged_lut.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(max_prompt_length),
        UInt32(max_context),
    )

    var q_lt = q_ragged.device_tensor()
    var out_lt = test_output.device_tensor()
    var offsets_lt = input_row_offsets.device_tensor()
    var partitions = partitions_override()

    @inline(.always)
    def launch(
        ctx: DeviceContext[],
    ) raises {
        var q_lt,
        var out_lt,
        var offsets_lt,
        var collection,
        var partitions,
        imm,
    }:
        comptime if causal:
            flash_attention[ragged=True](
                out_lt,
                q_lt,
                collection.get_key_cache(0),
                collection.get_value_cache(0),
                SlidingWindowCausalMask[window](),
                offsets_lt,
                scale,
                ctx,
                num_partitions=partitions,
            )
        else:
            flash_attention[ragged=True](
                out_lt,
                q_lt,
                collection.get_key_cache(0),
                collection.get_value_cache(0),
                SlidingWindowNonCausalMask[window](),
                offsets_lt,
                scale,
                ctx,
                num_partitions=partitions,
            )

    launch(ctx)
    ctx.synchronize()
    test_output.to_host()

    if benchmark:
        comptime nrun = 50
        var nstime = Float64(ctx.execution_time(launch, nrun)) / Float64(nrun)
        print("  kernel time:", nstime / 1000, "us (avg of", nrun, "runs)")

    var out_host = test_output.host_tensor()
    var scores = List[Float32](length=max_context, fill=0)
    var acc = List[Float32](length=depth, fill=0)
    var max_err = Float64(0)
    var bad_total = 0
    for b in range(batch_size):
        var row0 = Int(offsets_host[b])
        var bad_here = 0
        for t in range(valid_lengths[b]):
            var pos = cache_lengths[b] + t
            var lo = max(pos - window + 1, 0)
            var hi = pos if causal else num_keys[b] - 1
            var count = hi - lo + 1
            for h in range(num_heads):
                var best = Float32(-1e30)
                for j in range(count):
                    var s = Float32(0)
                    for d in range(depth):
                        s += (
                            q_ptr[
                                unsafe_offset=((row0 + t) * num_heads + h)
                                * depth
                                + d
                            ].cast[.float32]()
                            * k_dense[(key_base[b] + lo + j) * depth + d].cast[
                                .float32
                            ]()
                        )
                    scores[j] = s * scale
                    best = max(best, scores[j])
                var denom = Float32(0)
                for j in range(count):
                    scores[j] = exp(scores[j] - best)
                    denom += scores[j]
                for d in range(depth):
                    acc[d] = 0
                for j in range(count):
                    var w = scores[j] / denom
                    for d in range(depth):
                        acc[d] += (
                            w
                            * v_dense[(key_base[b] + lo + j) * depth + d].cast[
                                .float32
                            ]()
                        )
                for d in range(depth):
                    var actual = out_host[row0 + t, h, d].cast[.float64]()
                    var expect = Float64(acc[d])
                    var err = abs(actual - expect)
                    if not (err <= 5e-3 + 3e-2 * abs(expect)):
                        if bad_here == 0:
                            print(
                                "  MISMATCH seq",
                                b,
                                "(cache",
                                cache_lengths[b],
                                ") token",
                                t,
                                "head",
                                h,
                                "d",
                                d,
                                "actual",
                                actual,
                                "expect",
                                expect,
                            )
                        bad_here += 1
                    else:
                        max_err = max(max_err, err)
        if bad_here > 0:
            bad_total += bad_here
            print("  seq", b, "bad elements:", bad_here)
    print("  max abs err:", max_err)
    assert_true(bad_total == 0, "output differs from the reference")

    # The collection holds raw device pointers, so keep the buffers alive past
    # the last launch.
    _ = collection^
    _ = kv_block^
    _ = paged_lut^
    _ = cache_lengths_tensor^
    _ = q_ragged^
    _ = input_row_offsets^
    _ = test_output^


def run_suite[window: Int, causal: Bool = True](ctx: DeviceContext[]) raises:
    # Contexts below, at, and just past the window, the last two straddling the
    # first-visible-key tile boundary of the earliest token.
    for S in [1, 2, 4, 8]:
        var valid = List[Int]()
        var cache = List[Int]()
        for c in [
            1,
            window // 2,
            window - S,
            window - S + 1,
            window,
            window + 1,
            window + 127,
            window + 129,
            3 * window + 5,
        ]:
            if c > 0:
                valid.append(S)
                cache.append(c)
        run_case[window, causal](valid, cache, ctx)

    # Mixed ragged batch: sequences holding fewer than S keys (prefill-shaped,
    # cache 0) and a short chunk beside long ones, S = 8 the batch max.
    run_case[window, causal](
        [8, 3, 8, 1, 8, 2], [0, 0, window + 300, 0, 40, 5], ctx
    )
    run_case[window, causal]([8, 8, 5, 8], [window + 17, 1, 0, 2 * window], ctx)


def main() raises:
    comptime assert _mha_decode_fold_ok[
        dtype,
        depth,
        num_heads,
        num_heads,
        8,
        use_valid_length=True,
        ragged=True,
        check_mask_during_decoding=SlidingWindowCausalMask[
            4096
        ].check_mask_during_decoding,
        mask_supports_fold=SlidingWindowCausalMask[
            4096
        ].supports_decode_token_fold,
    ](), "the sliding-window draft shape must take the decode fold"

    seed(7)
    with DeviceContext() as ctx:
        if is_benchmark():
            # Draft-model shape: 20 requests, ~80K context, window 4096, S = 8.
            var valid = List[Int]()
            var cache = List[Int]()
            for i in range(20):
                valid.append(8)
                cache.append(80000 - 4000 + i * 400)
            run_case[4096](valid, cache, ctx, benchmark=True)
            var valid1 = List[Int]()
            for _ in range(20):
                valid1.append(1)
            run_case[4096](valid1, cache, ctx, benchmark=True)
            run_case[4096, False](valid, cache, ctx, benchmark=True)
            return

        run_suite[4096](ctx)
        run_suite[192](ctx)
        run_suite[4096, False](ctx)
        run_suite[192, False](ctx)
        # One long context at the production window.
        run_case[4096]([8, 8, 1], [80000, 20000, 79999], ctx)
