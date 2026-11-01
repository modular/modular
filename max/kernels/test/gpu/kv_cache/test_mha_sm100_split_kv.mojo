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
"""SM100 FA4 attention over a split paged KV cache, at MiMo-V2.6-Flash's
full-attention shapes.

A 192-wide key and a 128-wide value are packed into one 320-wide row per KV
head; the GQA group is 16 and pages hold 128 tokens. Each case runs a ragged,
causal batch three ways on the same bfloat16 values:

- the split cache: through the 2Q K-tail tile when the batch's longest prompt
  times the group exceeds 128 query rows, else (decode, speculative verify of
  up to 8 tokens) through the 1Q tile with split-K;
- the padded 256-wide cache the split cache replaces, with Q, K and V
  zero-padded as the padded production path pads them, its output sliced to
  the value's 128 columns;
- an fp32 reference on the GPU that reads K and V from dense tensors, so it
  shares no paging or layout code with either kernel.

The split path's error against the reference must be at most 1.25x the
padded path's (1.1x through the split-K workspace, whose f32 partials leave
one rounding to bfloat16, as the unsplit padded path has), and both paths must
be within two bfloat16 steps of it. The
q = 129, 2048 and 8192 cases end on a short final tile, which is what would
have the kernel switch to its 1Q body; the K-tail tile has none, so they also
show that no switch happens.

A K/V ring with too few slots for its schedule hangs rather than returning a
wrong answer, so every case arms a deadline that ends the process with
SIGALRM. `-D FA4_RING_SLEEP_PRODUCER_NS=<ns>` or
`-D FA4_RING_SLEEP_CONSUMER_NS=<ns>` delays one side of every ring hand-off,
which holds the rings empty or full for the whole run.

The decode and verify cases run the 1Q tile at the partition count its own
sizing picks (1 on a short cache, up to the SM count at 128k keys) and at
pinned counts that take each split-K mechanism: one partition, the cluster
size, and the workspace with its combine kernel, including more partitions
than the cache has key tiles. A pinned count that a route cannot honor must
raise rather than run unsplit.

`-D SPLIT_KV_K_DIM=128 -D SPLIT_KV_V_DIM=128` runs the same cases on an
ordinary 128-wide cache. With `-D FA4_TMEM_CROSS_P=false` that is the 2Q
shared-ring schedule without cross-stage P, the schedule the K-tail tile
runs, covered here independently of the tail code.

`-D SPLIT_KV_SWA=true` runs the sliding-window group instead: group 8, a
128-key window and a learned sink logit per Q head, on all three paths. Its
cases put chunks longer than the window over prefixes far beyond it, and
decode and verify run at the one partition the window leaves them
(`mha_split_kv_decode_num_partitions`) as well as at pinned counts above the
window's two key tiles.
"""

from std.collections import OptionalReg
from std.math import ceildiv, exp, exp2, isnan, logb, rsqrt, sqrt
from std.sys import get_defined_bool, get_defined_int

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.attention.gpu.mha import MHADecodeDispatchMetadata, flash_attention
from nn.attention.gpu.nvidia.sm100.dispatch import (
    mha_split_kv_decode_num_partitions,
)
from nn.attention.gpu.nvidia.common import ImmutTileTensor1D
from nn.attention.mha_mask import CausalMask, MHAMask, SlidingWindowCausalMask
from std.ffi import external_call
from std.testing import assert_false, assert_raises, assert_true

from kv_cache_test_utils import CacheLengthsTable, PagedLookupTable

comptime dtype = DType.bfloat16
comptime page_size = 128
comptime swa = get_defined_bool["SPLIT_KV_SWA", False]()
comptime group = 8 if swa else 16
# Keys a query attends to, its own included; 0 for none.
comptime window = 128 if swa else 0
comptime k_dim = get_defined_int["SPLIT_KV_K_DIM", 192]()
comptime v_dim = get_defined_int["SPLIT_KV_V_DIM", 128]()
comptime cross_p = get_defined_bool["FA4_TMEM_CROSS_P", True]()
comptime split = k_dim != v_dim
comptime padded_dim = 256
# Workspace partials rounded to bfloat16 put the split path at 1.19-1.27x.
comptime workspace_error_factor = 1.1
comptime chunk = 64
comptime ref_block = 128


comptime watchdog_seconds = 60


def _arm_watchdog(seconds: Int):
    """Ends the process with SIGALRM after `seconds`, unless re-armed first.

    `DeviceContext.synchronize` has no timeout, and a launch that frees a
    buffer can block before it returns, so the deadline cannot be a poll on
    this thread. `0` disarms it.
    """
    _ = external_call["alarm", UInt32](UInt32(seconds))


def _uniform(seed: UInt64, idx: Int) -> Float32:
    """A splitmix64 hash of `(seed, idx)` mapped to [-1, 1)."""
    var z = seed * 0x9E3779B97F4A7C15 + UInt64(idx) + 0x632BE59BD9B4E019
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float32(z >> 40) * Float32(1.0 / 8388608.0) - 1.0


def reference_kernel[
    num_q_heads: Int, num_kv_heads: Int
](
    q: MutPointer[Scalar[dtype], MutAnyOrigin],
    k: MutPointer[Scalar[dtype], MutAnyOrigin],
    v: MutPointer[Scalar[dtype], MutAnyOrigin],
    out_ref: MutPointer[Float32, MutAnyOrigin],
    q_offsets: MutPointer[UInt32, MutAnyOrigin],
    key_offsets: MutPointer[UInt32, MutAnyOrigin],
    cache_lens: MutPointer[UInt32, MutAnyOrigin],
    sinks: MutPointer[Scalar[dtype], MutAnyOrigin],
    num_rows: UInt32,
    scale: Float32,
):
    """Causal attention for one (query token, Q head), in fp32, over the
    `window` keys up to the query's own when `swa`, with the head's sink
    logit as one more score that has no value row."""
    var tid = global_idx.x
    if tid >= Int(num_rows) * num_q_heads:
        return
    var token, h = divmod(tid, num_q_heads)
    var kv_h = h // group
    var b = 0
    while UInt32(token) >= q_offsets[unsafe_offset=b + 1]:
        b += 1
    var pos = (
        Int(cache_lens[unsafe_offset=b])
        + token
        - Int(q_offsets[unsafe_offset=b])
    )
    var q_row = q.unsafe_offset((token * num_q_heads + h) * k_dim)
    var row_max = Float32(-3.0e38)
    var row_sum = Float32(0)
    var first = 0
    comptime if swa:
        row_max = sinks[unsafe_offset=h].cast[DType.float32]()
        row_sum = 1
        first = max(0, pos - window + 1)
    var acc = SIMD[DType.float32, v_dim](0)
    for j in range(first, pos + 1):
        var kv_row = (Int(key_offsets[unsafe_offset=b]) + j) * num_kv_heads + (
            kv_h
        )
        var k_row = k.unsafe_offset(kv_row * k_dim)
        var s = Float32(0)
        comptime for c in range(k_dim // chunk):
            var qc = q_row.unsafe_offset(c * chunk).unsafe_load[width=chunk]()
            var kc = k_row.unsafe_offset(c * chunk).unsafe_load[width=chunk]()
            s += (
                qc.cast[DType.float32]() * kc.cast[DType.float32]()
            ).reduce_add()
        s *= scale
        var new_max = max(row_max, s)
        var correction = exp(row_max - new_max)
        var p = exp(s - new_max)
        row_sum = row_sum * correction + p
        var vr = v.unsafe_offset(kv_row * v_dim).unsafe_load[width=v_dim]()
        acc = acc * correction + vr.cast[DType.float32]() * p
        row_max = new_max
    out_ref.unsafe_store((token * num_q_heads + h) * v_dim, acc / row_sum)


@fieldwise_init
struct Errors(Copyable):
    var max_abs: Float64
    var rms: Float64
    var max_ref: Float64


def _errors(
    got: MutPointer[Scalar[dtype], _],
    got_dim: Int,
    want: MutPointer[Float32, _],
    rows: Int,
) raises -> Errors:
    """Errors of the first `v_dim` columns of each `got_dim`-wide row."""
    var max_abs = Float64(0)
    var sum_sq = Float64(0)
    var max_ref = Float64(0)
    for r in range(rows):
        for d in range(v_dim):
            var g = got[unsafe_offset=r * got_dim + d].cast[DType.float64]()
            assert_false(isnan(g), "NaN in an attention output")
            var w = want[unsafe_offset=r * v_dim + d].cast[DType.float64]()
            var e = abs(g - w)
            max_abs = max(max_abs, e)
            max_ref = max(max_ref, abs(w))
            sum_sq += e * e
    return Errors(max_abs, sqrt(sum_sq / Float64(rows * v_dim)), max_ref)


def _bf16_ulp(x: Float64) -> Float64:
    """The spacing of bfloat16 values at magnitude `x` (8 significant bits)."""
    return exp2(logb(x) - 7)


def run_case[
    num_kv_heads: Int
](
    prompt_lens: List[Int],
    cache_lens: List[Int],
    ctx: DeviceContext,
    *,
    num_partitions: Optional[Int] = None,
    metadata_partitions: Optional[Int] = None,
    stale_cache_len: Optional[Int] = None,
    refused: Optional[String] = None,
    error_factor: Float64 = 1.25,
) raises -> List[Scalar[dtype]]:
    """Runs `_run_case` under the group's mask: causal, or a window when
    `swa`."""
    comptime if swa:
        return _run_case[num_kv_heads](
            SlidingWindowCausalMask[window](),
            prompt_lens,
            cache_lens,
            ctx,
            num_partitions=num_partitions,
            metadata_partitions=metadata_partitions,
            stale_cache_len=stale_cache_len,
            refused=refused,
            error_factor=error_factor,
        )
    else:
        return _run_case[num_kv_heads](
            CausalMask(),
            prompt_lens,
            cache_lens,
            ctx,
            num_partitions=num_partitions,
            metadata_partitions=metadata_partitions,
            stale_cache_len=stale_cache_len,
            refused=refused,
            error_factor=error_factor,
        )


def _run_case[
    num_kv_heads: Int, mask_t: MHAMask
](
    mask: mask_t,
    prompt_lens: List[Int],
    cache_lens: List[Int],
    ctx: DeviceContext,
    *,
    num_partitions: Optional[Int] = None,
    metadata_partitions: Optional[Int] = None,
    stale_cache_len: Optional[Int] = None,
    refused: Optional[String] = None,
    error_factor: Float64 = 1.25,
) raises -> List[Scalar[dtype]]:
    """Runs one batch through the split, padded and reference paths.

    `metadata_partitions` hands the split path dispatch metadata with that
    partition count, as the graph op does. `stale_cache_len` replaces the
    batch's longest cache length wherever the host reads it (the cache's
    `max_cache_length` and the metadata), as a graph replayed after its
    capture would. With `refused`, only the split path runs, and it must
    raise an error that contains `refused`. The split path's error must be at
    most `error_factor` times the padded path's.

    Returns:
        The split path's output.
    """
    # A hung case is named by the last line it printed.
    print(
        (
            t"case: kv_heads={num_kv_heads} prompts={prompt_lens}"
            t" caches={cache_lens}"
        ),
        flush=True,
    )
    _arm_watchdog(watchdog_seconds)
    comptime num_q_heads = group * num_kv_heads
    comptime native_params = KVCacheStaticParams(
        num_kv_heads, k_dim + v_dim, v_head_size=v_dim
    ) if split else KVCacheStaticParams(num_kv_heads, k_dim)
    comptime native_planes = native_params.num_planes()
    comptime native_row = native_params.head_size
    comptime padded_params = KVCacheStaticParams(num_kv_heads, padded_dim)
    comptime NativeCollection = PagedKVCacheCollection[
        dtype,
        native_params,
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

    var batch_size = len(prompt_lens)
    var table = CacheLengthsTable.build(prompt_lens, cache_lens, ctx)
    var rows = table.total_length
    var max_context = table.max_full_context_length
    var num_pages = 0
    var num_keys = 0
    var key_offsets_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size
    )
    for b in range(batch_size):
        key_offsets_host[b] = UInt32(num_keys)
        num_keys += cache_lens[b] + prompt_lens[b]
        num_pages += ceildiv(cache_lens[b] + prompt_lens[b], page_size)
    var num_blocks = num_pages + 1
    var lut = PagedLookupTable[page_size].build(
        prompt_lens, cache_lens, max_context, num_blocks, ctx
    )
    var lut_host = lut.host_tensor()
    comptime assert lut_host.flat_rank == 2

    var k_dense = HostDeviceTileTensor[dtype](
        row_major(Int64(num_keys), Idx[num_kv_heads], Idx[k_dim]), ctx
    )
    var v_dense = HostDeviceTileTensor[dtype](
        row_major(Int64(num_keys), Idx[num_kv_heads], Idx[v_dim]), ctx
    )
    var q = HostDeviceTileTensor[dtype](
        row_major(Int64(rows), Idx[num_q_heads], Idx[k_dim]), ctx
    )
    var q_pad = HostDeviceTileTensor[dtype](
        row_major(Int64(rows), Idx[num_q_heads], Idx[padded_dim]), ctx
    )
    var native_blocks = HostDeviceTileTensor[dtype](
        row_major(
            Int64(num_blocks),
            Idx[native_planes],
            Idx[1],
            Idx[page_size],
            Idx[num_kv_heads],
            Idx[native_row],
        ),
        ctx,
    )
    var padded_blocks = HostDeviceTileTensor[dtype](
        row_major(
            Int64(num_blocks),
            Idx[2],
            Idx[1],
            Idx[page_size],
            Idx[num_kv_heads],
            Idx[padded_dim],
        ),
        ctx,
    )

    var kd = k_dense.host_tensor().ptr
    var vd = v_dense.host_tensor().ptr
    var qh = q.host_tensor().ptr
    var qph = q_pad.host_tensor().fill(0).ptr
    var nb = native_blocks.host_tensor().fill(0).ptr
    var pb = padded_blocks.host_tensor().fill(0).ptr
    for i in range(rows * num_q_heads):
        for d in range(k_dim):
            # Q in [-3, 3) gives the softmax some peaks to resolve.
            var x = (3 * _uniform(1, i * k_dim + d)).cast[dtype]()
            qh[unsafe_offset=i * k_dim + d] = x
            qph[unsafe_offset=i * padded_dim + d] = x
    for b in range(batch_size):
        for j in range(cache_lens[b] + prompt_lens[b]):
            var key = Int(key_offsets_host[b]) + j
            var page, slot = divmod(j, page_size)
            var block = Int(lut_host[b, page])
            for h in range(num_kv_heads):
                var dense = key * num_kv_heads + h
                var native = (
                    (block * native_planes * page_size + slot) * num_kv_heads
                    + h
                ) * native_row
                var v_native = native + (
                    k_dim if split else page_size * num_kv_heads * native_row
                )
                var padded = (
                    (block * 2 * page_size + slot) * num_kv_heads + h
                ) * padded_dim
                var v_padded = padded + page_size * num_kv_heads * padded_dim
                for d in range(k_dim):
                    var x = _uniform(2, dense * k_dim + d).cast[dtype]()
                    kd[unsafe_offset=dense * k_dim + d] = x
                    nb[unsafe_offset=native + d] = x
                    pb[unsafe_offset=padded + d] = x
                for d in range(v_dim):
                    var x = _uniform(3, dense * v_dim + d).cast[dtype]()
                    vd[unsafe_offset=dense * v_dim + d] = x
                    nb[unsafe_offset=v_native + d] = x
                    pb[unsafe_offset=v_padded + d] = x
    k_dense.to_device()
    v_dense.to_device()
    q.to_device()
    q_pad.to_device()
    native_blocks.to_device()
    padded_blocks.to_device()
    var key_offsets = ctx.enqueue_create_buffer[DType.uint32](batch_size)
    ctx.enqueue_copy(key_offsets, key_offsets_host)

    var host_cache_len = stale_cache_len.or_else(max_context)
    var native = NativeCollection(
        native_blocks.device_tensor().as_unsafe_any_origin(),
        table.cache_lengths.device_tile_tensor(),
        lut.device_tile_tensor(),
        UInt32(table.max_seq_length_batch),
        UInt32(host_cache_len),
    )
    var metadata: OptionalReg[MHADecodeDispatchMetadata] = None
    if metadata_partitions:
        metadata = MHADecodeDispatchMetadata(
            batch_size,
            table.max_seq_length_batch,
            metadata_partitions.value(),
            host_cache_len,
        )
    var padded = PaddedCollection(
        padded_blocks.device_tensor().as_unsafe_any_origin(),
        table.cache_lengths.device_tile_tensor(),
        lut.device_tile_tensor(),
        UInt32(table.max_seq_length_batch),
        UInt32(max_context),
    )
    var offsets = table.input_row_offsets.device_tile_tensor()

    var out = HostDeviceTileTensor[dtype](
        row_major(Int64(rows), Idx[num_q_heads], Idx[v_dim]), ctx
    )
    var out_pad = HostDeviceTileTensor[dtype](
        row_major(Int64(rows), Idx[num_q_heads], Idx[padded_dim]), ctx
    )
    var out_ref = HostDeviceTileTensor[DType.float32](
        row_major(Int64(rows), Idx[num_q_heads], Idx[v_dim]), ctx
    )
    # The model's softmax scale is the key's, not the padded width's.
    var scale = rsqrt(Float32(k_dim))
    # Sink logits in [-4, 4): some heads' sinks outweigh every key, some are
    # negligible.
    var sinks = HostDeviceTileTensor[dtype](row_major(Int64(num_q_heads)), ctx)
    var sinks_host = sinks.host_tensor()
    for h in range(num_q_heads):
        sinks_host[h] = (4 * _uniform(4, h)).cast[dtype]()
    sinks.to_device()
    var sink_weights: OptionalReg[ImmutTileTensor1D[dtype]] = None
    comptime if swa:
        sink_weights = sinks.device_tensor().as_imm().as_unsafe_any_origin()

    if refused:
        with assert_raises(contains=refused.value()):
            flash_attention[ragged=True, sink=swa, cross_p=cross_p](
                out.device_tensor(),
                q.device_tensor().as_imm(),
                native.get_key_cache(0),
                native.get_value_cache(0),
                mask,
                offsets.as_imm(),
                scale,
                ctx,
                num_partitions=num_partitions,
                sink_weights=sink_weights,
                decode_dispatch_metadata=metadata,
            )
        print(
            t"kv_heads={num_kv_heads} prompts={prompt_lens} caches={cache_lens}"
            t" num_partitions={num_partitions} metadata"
            t" {metadata_partitions}: refused"
        )
        _ = key_offsets^
        return []
    flash_attention[ragged=True, sink=swa, cross_p=cross_p](
        out.device_tensor(),
        q.device_tensor().as_imm(),
        native.get_key_cache(0),
        native.get_value_cache(0),
        mask,
        offsets.as_imm(),
        scale,
        ctx,
        num_partitions=num_partitions,
        sink_weights=sink_weights,
        decode_dispatch_metadata=metadata,
    )
    flash_attention[ragged=True, sink=swa, cross_p=cross_p](
        out_pad.device_tensor(),
        q_pad.device_tensor().as_imm(),
        padded.get_key_cache(0),
        padded.get_value_cache(0),
        mask,
        offsets.as_imm(),
        scale,
        ctx,
        sink_weights=sink_weights,
    )
    comptime ref_kernel = reference_kernel[num_q_heads, num_kv_heads]
    ctx.enqueue_function[ref_kernel](
        q.device_tensor().ptr,
        k_dense.device_tensor().ptr,
        v_dense.device_tensor().ptr,
        out_ref.device_tensor().ptr,
        table.input_row_offsets.device_tile_tensor().ptr,
        key_offsets,
        table.cache_lengths.device_tile_tensor().ptr,
        sinks.device_tensor().ptr,
        UInt32(rows),
        scale,
        grid_dim=ceildiv(rows * num_q_heads, ref_block),
        block_dim=ref_block,
    )
    ctx.synchronize()
    out.to_host()
    out_pad.to_host()
    out_ref.to_host()

    var want = out_ref.host_tensor().ptr
    var native_err = _errors(
        out.host_tensor().ptr, v_dim, want, rows * num_q_heads
    )
    var padded_err = _errors(
        out_pad.host_tensor().ptr, padded_dim, want, rows * num_q_heads
    )
    print(
        t"kv_heads={num_kv_heads} prompts={prompt_lens} caches={cache_lens}"
        t" num_partitions={num_partitions} metadata {metadata_partitions}"
        t" stale cache {stale_cache_len}: native max {native_err.max_abs}"
        t" rms {native_err.rms}, padded max {padded_err.max_abs} rms"
        t" {padded_err.rms}, output ulp {_bf16_ulp(native_err.max_ref)}"
    )
    assert_true(
        native_err.rms <= error_factor * padded_err.rms,
        "split path rms error above the gate's multiple of the padded path's",
    )
    # A decode or verify case has a few thousand outputs, whose largest error
    # is a fraction of one bfloat16 step and moves by a step with the
    # summation order; split-K changes that order. The max gate allows that
    # one step on top of the 1.25x; the rms gate does not.
    assert_true(
        native_err.max_abs
        <= error_factor * padded_err.max_abs + _bf16_ulp(native_err.max_ref),
        "split path max error above the gate's multiple of the padded path's",
    )
    # Both gates are relative, so a defect in code the two paths share, such
    # as the window mask, moves both errors alike and passes them. Neither
    # path may be more than two bfloat16 steps of the largest output off.
    var step = _bf16_ulp(native_err.max_ref)
    assert_true(
        native_err.max_abs <= 2 * step and padded_err.max_abs <= 2 * step,
        "attention output more than two bfloat16 steps from the reference",
    )
    _ = key_offsets^
    _ = sinks^
    var got = out.host_tensor().ptr
    var result = List[Scalar[dtype]](capacity=rows * num_q_heads * v_dim)
    for i in range(rows * num_q_heads * v_dim):
        result.append(got[unsafe_offset=i])
    return result^


def main() raises:
    with DeviceContext() as ctx:
        comptime if swa:
            sliding_window(ctx)
            _arm_watchdog(0)
            return
        # TP4 (one KV head per device): a ragged batch of chunked prefills
        # with chunk ends on and off page and tile boundaries, then long
        # prompts on an empty cache.
        _ = run_case[1]([129, 9, 96, 128], [0, 5000, 130, 100000], ctx)
        _ = run_case[1]([2048], [0], ctx)
        _ = run_case[1]([8192], [0], ctx)
        # TP2: a chunk on a long cache next to a decode row.
        _ = run_case[2]([2048, 1], [100000, 7], ctx)
        # DP (all four KV heads on one device).
        _ = run_case[4]([8192], [5000], ctx)

        comptime if split:
            decode_and_verify(ctx)
            partitions_from_metadata(ctx)
    _arm_watchdog(0)


def decode_and_verify(ctx: DeviceContext) raises:
    # Decode at the partition count the route picks: 1 below 512 keys, then
    # up the split-K ladder to the SM count at 128k keys on one KV head.
    _ = run_case[1]([1], [100], ctx)
    _ = run_case[1]([1], [4096], ctx)
    _ = run_case[1]([1], [32768], ctx)
    _ = run_case[1]([1], [131072], ctx)
    # Verify of 2 to 8 tokens; 8 tokens x group 16 fill the 128-row tile.
    _ = run_case[1]([8], [32768], ctx)
    _ = run_case[1]([2, 5, 8], [4000, 70000, 131072], ctx)
    _ = run_case[2]([1, 1, 1, 1], [511, 4096, 32768, 131072], ctx)
    _ = run_case[2]([8, 3], [32768, 5000], ctx)
    _ = run_case[4]([1], [131072], ctx)
    _ = run_case[4]([8, 8], [4096, 32768], ctx)

    # Pinned counts: 1 (no split), 2 (the B200 cluster size), the rest through
    # the workspace and its combine, up to the combine's limit.
    for p in [1, 2, 3, 6, 16, 48, 148, 160]:
        _ = run_case[1](
            [1],
            [32768],
            ctx,
            num_partitions=p,
            error_factor=workspace_error_factor if p > 2 else 1.25,
        )
    _ = run_case[2](
        [8, 1],
        [32768, 4000],
        ctx,
        num_partitions=6,
        error_factor=workspace_error_factor,
    )
    # More partitions than the 8 key tiles: the trailing ones are empty.
    _ = run_case[1](
        [1],
        [1000],
        ctx,
        num_partitions=16,
        error_factor=workspace_error_factor,
    )

    # Counts no route can honor.
    _ = run_case[1](
        [1], [4096], ctx, num_partitions=161, refused=String("exceeds")
    )
    _ = run_case[1](
        [9], [4096], ctx, num_partitions=2, refused=String("cannot split")
    )


def partitions_from_metadata(ctx: DeviceContext) raises:
    # The metadata's count is the launch's: three counts sum the keys in
    # three orders, so the outputs differ bitwise. A count dropped on the
    # way would leave them identical.
    var outs = List[List[Scalar[dtype]]]()
    for p in [1, 2, 6]:
        outs.append(
            run_case[1](
                [8],
                [4096],
                ctx,
                metadata_partitions=p,
                error_factor=workspace_error_factor if p > 2 else 1.25,
            )
        )
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            var differs = False
            for e in range(len(outs[i])):
                if outs[i][e] != outs[j][e]:
                    differs = True
                    break
            assert_true(
                differs, "split-K outputs identical across partition counts"
            )
    _ = run_case[2](
        [1, 1],
        [32768, 600],
        ctx,
        metadata_partitions=48,
        error_factor=workspace_error_factor,
    )

    # A prompt too long for the split-K tile runs the 2Q tile, which cannot
    # split, so the metadata's count must not reach it.
    _ = run_case[1]([129, 1], [0, 4096], ctx, metadata_partitions=6)

    # A captured graph replays with the host cache length it was captured
    # at. The key count each sequence attends over comes from the device
    # cache lengths, so a stale host length changes nothing.
    _ = run_case[1](
        [8],
        [32768],
        ctx,
        metadata_partitions=6,
        stale_cache_len=129,
        error_factor=workspace_error_factor,
    )
    _ = run_case[1](
        [1],
        [131072],
        ctx,
        metadata_partitions=48,
        stale_cache_len=4096,
        error_factor=workspace_error_factor,
    )

    _ = run_case[1](
        [1], [4096], ctx, metadata_partitions=161, refused=String("exceeds")
    )


def sliding_window(ctx: DeviceContext) raises:
    # Two, four and eight KV heads per device: TP4, TP2 and DP8.
    #
    # Chunked prefill through the 2Q K-tail tile: chunks longer than the
    # window, chunk starts on and off its edge, and prefixes far beyond it.
    _ = run_case[2]([129, 9, 96, 128], [0, 5000, 130, 100000], ctx)
    _ = run_case[2]([300, 127, 129, 17], [262000, 1, 255, 128], ctx)
    _ = run_case[2]([2048], [0], ctx)
    _ = run_case[4]([8192], [5000], ctx)
    _ = run_case[8]([2048, 1], [100000, 7], ctx)
    # A prompt of up to 16 tokens fills at most the 1Q tile's 128 rows.
    _ = run_case[4]([16, 3], [0, 200], ctx)

    # Decode around the window's edge and far past it.
    _ = run_case[2]([1], [100], ctx)
    _ = run_case[2]([1], [127], ctx)
    _ = run_case[2]([1], [128], ctx)
    _ = run_case[2]([1], [131072], ctx)
    _ = run_case[4]([1, 1, 1, 1], [5, 511, 4096, 262143], ctx)
    _ = run_case[8]([1], [32768], ctx)
    # Verify of 2 to 8 tokens; 16 tokens x group 8 fill the 1Q tile.
    _ = run_case[2]([2, 5, 8], [4000, 70000, 131072], ctx)
    _ = run_case[4]([8, 8], [100, 32768], ctx)
    _ = run_case[8]([8, 3], [131072, 120], ctx)
    _ = run_case[2]([16], [1000], ctx)

    # The metadata's count, as the graph op passes it: one partition, since
    # the window is under the 512 keys a partition is sized for.
    assert_true(
        mha_split_kv_decode_num_partitions[148](1, 262144, 2, window) == 1,
        "a 128-key window split over more than one partition",
    )
    _ = run_case[2]([8], [32768], ctx, metadata_partitions=1)
    _ = run_case[4]([1, 8], [262143, 600], ctx, metadata_partitions=1)
    _ = run_case[2](
        [8], [32768], ctx, metadata_partitions=1, stale_cache_len=129
    )

    # Pinned counts, more partitions than the window has key tiles; the sink
    # must still be counted once. Two partitions combine in fp32 through the
    # cluster, three or more through the workspace.
    _ = run_case[2]([1], [32768], ctx, num_partitions=2)
    for p in [3, 6, 16]:
        _ = run_case[2](
            [1],
            [32768],
            ctx,
            num_partitions=p,
            error_factor=workspace_error_factor,
        )
    _ = run_case[4](
        [8, 1],
        [4000, 300],
        ctx,
        num_partitions=6,
        error_factor=workspace_error_factor,
    )
