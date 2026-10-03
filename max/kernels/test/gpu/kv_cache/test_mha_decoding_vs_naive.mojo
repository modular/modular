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
# Test mha_decoding (the batched decode kernel) via the KV cache
# flash_attention overload, which sets is_token_generation=True and
# dispatches to mha_decoding.
#
# Two independent checks per case: mha_gpu_naive within tolerance owns
# correctness, a hash against a known-good MI355 value pins bitwise
# reproducibility.
#
# This test is separate from test_batch_kv_cache_flash_attention_*
# because that test compares continuous-vs-paged (both go through the
# same mha_decoding kernel, so they can't detect bugs in mha_decoding
# itself).

from std.math import rsqrt
from std.memory import bitcast
from std.random import seed
from std.collections import Set

from max.gpu.host import DeviceContext
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import Coord, Idx, row_major
from layout._fillers import random
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import CausalMask
from std.testing import assert_almost_equal, assert_true


def compute_hash[
    type: DType
](ptr: Pointer[Scalar[type], _], size: Int) -> UInt64:
    var h: UInt64 = 14695981039346656037
    for i in range(size):
        var val = ptr[i].cast[.float32]()
        var bits = bitcast[.uint32, 1](val)
        h ^= bits.cast[.uint64]()
        h *= 1099511628211
    return h


def test_decode_kv_cache[
    num_q_heads: Int,
    kv_params: KVCacheStaticParams,
    dtype: DType,
](
    cache_lengths: List[Int],
    ctx: DeviceContext,
    expected_hash: UInt64 = 0,
    target_q_head: Int = -1,
    channel_idx: Int = 0,
    q_channel_value: Float32 = 0,
    k_channel_value: Float32 = 0,
) raises:
    """Test mha_decoding by calling the KV cache flash_attention overload
    with max_prompt_length=1 (triggering is_token_generation=True).

    Compares against `mha_gpu_naive` within tolerance, and optionally against a
    known-good hash for bitwise reproducibility.

    When `target_q_head >= 0`, one Q/K channel is overwritten so that head's
    score against every key is `q_channel_value * k_channel_value` plus random
    noise, while every other head keeps a zero Q in that channel and stays at
    the noise floor. This drives the targeted head's scores to an extreme
    magnitude that random inputs never reach.
    """
    comptime page_size = 256
    var batch_size = len(cache_lengths)
    var num_layers = 2
    var layer_idx = 1

    # TG: all seq_len=1
    var valid_lengths = List[Int]()
    for _ in range(batch_size):
        valid_lengths.append(1)

    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]

    var q_layout = row_major(
        total_length, Idx[num_q_heads], Idx[kv_params.head_size]
    )

    # Row offsets
    var row_offsets = HostDeviceTileTensor[.uint32](
        row_major(Coord(Int64(batch_size + 1))), ctx
    )
    var row_offsets_host = row_offsets.host_tensor()
    var running_offset: UInt32 = 0
    for i in range(batch_size):
        row_offsets_host[i] = running_offset
        running_offset += UInt32(valid_lengths[i])
    row_offsets_host[batch_size] = running_offset
    row_offsets.to_device()

    # Cache lengths
    var cache_lens = HostDeviceTileTensor[.uint32](
        row_major(Coord(Int64(batch_size))), ctx
    )
    var cache_lens_host = cache_lens.host_tensor()
    for i in range(batch_size):
        cache_lens_host[i] = UInt32(cache_lengths[i])
    cache_lens.to_device()

    # Q (random, ragged)
    var q = HostDeviceTileTensor[dtype](q_layout, ctx)
    var q_host = q.host_tensor()
    random(q_host)
    if target_q_head >= 0:
        for r in range(total_length):
            for h in range(num_q_heads):
                if h == target_q_head:
                    q_host[r, h, channel_idx] = Scalar[dtype](q_channel_value)
                else:
                    q_host[r, h, channel_idx] = Scalar[dtype](0)
    q.to_device()

    # Output
    var output = HostDeviceTileTensor[dtype](q_layout, ctx)

    # Naive reference output
    var ref_output = HostDeviceTileTensor[dtype](q_layout, ctx)

    # Continuous KV blocks
    var num_continuous_blocks = batch_size + 2
    var kv_block_continuous = HostDeviceTileTensor[dtype](
        row_major(
            Coord(
                Int64(num_continuous_blocks),
                Int64(2),
                Int64(num_layers),
                Int64(max_full_context_length),
                Idx[kv_params.num_heads],
                Idx[kv_params.head_size],
            )
        ),
        ctx,
    )
    var kv_block_host = kv_block_continuous.host_tensor()
    random(kv_block_host)
    if target_q_head >= 0:
        var group = num_q_heads // kv_params.num_heads
        var target_kv_head = target_q_head // group
        for blk in range(num_continuous_blocks):
            for pos in range(max_full_context_length):
                kv_block_host[
                    blk, 0, layer_idx, pos, target_kv_head, channel_idx
                ] = Scalar[dtype](k_channel_value)
    kv_block_continuous.to_device()

    # Lookup table for continuous batching
    var lookup_table = HostDeviceTileTensor[.uint32](
        row_major(Coord(Int64(batch_size))), ctx
    )
    var lookup_host = lookup_table.host_tensor()
    for i in range(batch_size):
        lookup_host[i] = UInt32(i)
    lookup_table.to_device()

    # Build continuous collection
    comptime Collection = ContinuousBatchingKVCacheCollection[
        dtype, kv_params, MutAnyOrigin, ImmutAnyOrigin, ImmutAnyOrigin
    ]
    var kv_continuous = Collection(
        kv_block_continuous.device_tensor().as_unsafe_any_origin(),
        cache_lens.device_tensor().as_imm().as_unsafe_any_origin(),
        lookup_table.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    var scale = rsqrt(Float32(kv_params.head_size))

    # Run flash_attention with continuous KV cache
    flash_attention[ragged=True](
        output.device_tensor(),
        q.device_tensor(),
        kv_continuous.get_key_cache(layer_idx),
        kv_continuous.get_value_cache(layer_idx),
        CausalMask(),
        row_offsets.device_tensor(),
        scale,
        ctx,
    )

    mha_gpu_naive[ragged=True](
        q.device_tensor(),
        kv_continuous.get_key_cache(layer_idx),
        kv_continuous.get_value_cache(layer_idx),
        CausalMask(),
        ref_output.device_tensor(),
        row_offsets.device_tensor(),
        scale,
        batch_size,
        max_prompt_length,
        max_full_context_length,
        num_q_heads,
        kv_params.head_size,
        num_q_heads // kv_params.num_heads,
        ctx,
    )

    output.to_host()
    ref_output.to_host()
    var out_host = output.host_tensor()
    var ref_host = ref_output.host_tensor()
    for r in range(total_length):
        for h in range(num_q_heads):
            for d in range(kv_params.head_size):
                assert_almost_equal(
                    out_host[r, h, d],
                    ref_host[r, h, d],
                    atol=1e-5,
                    rtol=8e-3,
                )
    print("REF OK")

    var actual_hash = compute_hash(
        out_host.unsafe_ptr(),
        total_length * num_q_heads * kv_params.head_size,
    )
    print("HASH:", actual_hash)

    # The reference above owns correctness, so a mismatch here alongside
    # `REF OK` means the numerics moved within tolerance -- a re-association
    # from a new partition count, not a wrong answer.
    if expected_hash != 0:
        if actual_hash != expected_hash:
            print("HASH MISMATCH: expected", expected_hash, "got", actual_hash)
            raise Error("Hash mismatch for mha_decoding output")
        else:
            print("HASH OK")


def main() raises:
    seed(42)
    with DeviceContext() as ctx:
        # Hash values are AMD MI355-specific (different GPUs produce
        # different MMA results). Only check hashes on AMD.
        # The split-K case also pins the partition count: partials are summed
        # per partition, so re-targeting `hip_mha_decoding_num_partitions`
        # re-associates that sum and moves the hash.
        comptime amd = ctx.target.is_amd_gpu()
        comptime group4_hash = UInt64(9912283832381023013) if amd else UInt64(0)
        comptime group16_small_hash = UInt64(
            13849357457032651557
        ) if amd else UInt64(0)
        comptime group16_large_hash = UInt64(
            6793244972047180581
        ) if amd else UInt64(0)

        # group=4 (baseline)
        print("TG group=4 depth=128 bs=4")
        test_decode_kv_cache[
            32,
            KVCacheStaticParams(num_heads=8, head_size=128),
            DType.bfloat16,
        ]([500, 700, 200, 300], ctx, expected_hash=group4_hash)

        # group=16 small caches (single partition, no split-k)
        print("TG group=16 bs=4 small caches (no split-k)")
        test_decode_kv_cache[
            16,
            KVCacheStaticParams(num_heads=1, head_size=128),
            DType.bfloat16,
        ]([50, 100, 150, 200], ctx, expected_hash=group16_small_hash)

        # group=16 large caches (split-k)
        print("TG group=16 bs=4 large caches (split-k)")
        test_decode_kv_cache[
            16,
            KVCacheStaticParams(num_heads=1, head_size=128),
            DType.bfloat16,
        ]([500, 700, 200, 300], ctx, expected_hash=group16_large_hash)

        # Mask-sentinel regression. Decode kernels substitute the finite
        # MASK_VALUE for masked out-of-bounds score columns, and online
        # softmax takes the row max over those columns too, so MASK_VALUE
        # must sit below every reachable score or it wins the max and zeroes
        # a head's real columns. This case drives one head's raw score to
        # about -262144 against every key (about -33000 after scale and
        # log2e), a magnitude random Q/K (roughly +-50) never reach, and
        # num_keys=490 is not a multiple of any power-of-2 tile width, so
        # the last KV tile always contains masked columns.
        print("TG group=4 depth=128 bs=1 mask sentinel")
        test_decode_kv_cache[
            32,
            KVCacheStaticParams(num_heads=8, head_size=128),
            DType.bfloat16,
        ](
            [489],
            ctx,
            target_q_head=10,
            channel_idx=5,
            q_channel_value=1024.0,
            k_channel_value=-256.0,
        )
