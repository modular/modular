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
"""Apple (Metal) split-K decode attention SINK correctness check.

Apple silicon GPU (Metal) only; decode-only.

`naive_fa_decode_apple[sink=True]` must apply the attention sink (a virtual
"key -1" of raw weight `sink_weight`), seeded into the running softmax state on
split 0 only (split-K cross-split LSE combine counts the sink once). This is a
regression test for the pre-fix bug where the decode kernel accepted
`sink_weights` but never applied it (the seq_len=1 decode case yielded 1.0
instead of the correct ~0.30).

Independent closed-form reference (mirrors
`test/gpu/nn/test_flash_attention.mojo::test_flash_attention_sink_kernel`):
with `scale = 0` every QK logit is exactly 0 and `V = 1`, so the attention
output for a head equals the probability mass on the real keys:

    out = num_keys / (num_keys + exp(sink_weight))

For sink=5.0, num_keys=64 this is 64/(64+exp(5)) ~= 0.30 -- distinctly NOT 1.0,
so a dropped sink seed fails loudly. Spans multiple SplitSize=32 partitions
(num_keys=64) to exercise the split-0-only seeding across the cross-split
combine.
"""

from std.collections import OptionalReg
from max.gpu.host import DeviceContext
from std.math import exp
from std.random import seed
from std.sys import has_apple_gpu_accelerator

from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import Idx, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor

from nn.attention.gpu.apple.naive_fa_decode import naive_fa_decode_apple
from nn.attention.gpu.nvidia.common import (
    ImmutTileTensor1D,
    immut_tile_tensor_1d,
)
from nn.attention.mha_mask import NullMask
from nn.attention.mha_operand import KVCacheMHAOperand

from std.testing import assert_almost_equal


def _run_sink_closed_form[
    num_q_heads: Int,
    kv_params: KVCacheStaticParams,
    dtype: DType = .bfloat16,
](num_keys: Int, sink_weights: List[Float32], ctx: DeviceContext,) raises:
    """Decode one token with `scale=0`, `V=1`, NullMask, and per-head sink
    weights; assert each head's output equals `num_keys/(num_keys+exp(sink))`.
    """
    comptime group = num_q_heads // kv_params.num_heads
    comptime depth = kv_params.head_size
    comptime num_blocks = 8
    comptime num_layers = 1
    comptime layer_idx = 0
    comptime batch_size = 1
    comptime max_prompt_len = 1  # decode: one query token

    # The new token attends `cache_length + 1` keys (`_is_cache_length_accurate
    # = False`). Want exactly `num_keys` attended -> cache_length = num_keys - 1.
    var cache_len = num_keys - 1
    var max_context_len = num_keys

    print(
        "  decode-sink",
        "dtype:",
        dtype,
        "depth:",
        depth,
        "q_heads:",
        num_q_heads,
        "kv_heads:",
        kv_params.num_heads,
        "num_keys:",
        num_keys,
    )

    # ---- q [batch, 1, num_q_heads, depth]; value irrelevant (scale=0). ---- #
    var q_layout = row_major(
        batch_size, max_prompt_len, Idx[num_q_heads], Idx[depth]
    )
    var q = HostDeviceTileTensor[dtype](q_layout, ctx)
    var q_host = q.host_tensor()
    for h in range(num_q_heads):
        for d in range(depth):
            q_host[0, 0, h, d] = Scalar[dtype](0.123)

    # ---- valid_length (per-sequence query length = 1) -------------------- #
    var valid_lengths = HostDeviceTileTensor[.uint32](
        row_major(batch_size), ctx
    )
    var vl_host = valid_lengths.host_tensor()
    vl_host[0] = UInt32(max_prompt_len)

    # ---- output [batch, 1, num_q_heads, depth] --------------------------- #
    var test_output = HostDeviceTileTensor[dtype](q_layout, ctx)

    # ---- per-sequence cache lengths -------------------------------------- #
    var cache_lengths_managed = HostDeviceTileTensor[.uint32](
        row_major(batch_size), ctx
    )
    var cl_host = cache_lengths_managed.host_tensor()
    cl_host[0] = UInt32(cache_len)

    # ---- continuous KV blocks; V = 1 everywhere so the output is the
    #      probability mass on the real keys. K irrelevant (scale=0). ------ #
    var kv_block = HostDeviceTileTensor[dtype](
        row_major(
            num_blocks,
            2,
            num_layers,
            max_context_len,
            kv_params.num_heads,
            depth,
        ),
        ctx,
    )
    var kv_host = kv_block.host_tensor()
    # kv_idx 0 = K (0.0), kv_idx 1 = V (1.0).
    for blk in range(num_blocks):
        for tok in range(max_context_len):
            for kvh in range(kv_params.num_heads):
                for d in range(depth):
                    kv_host[blk, 0, layer_idx, tok, kvh, d] = Scalar[dtype](0.0)
                    kv_host[blk, 1, layer_idx, tok, kvh, d] = Scalar[dtype](1.0)

    # ---- lookup table: sequence 0 uses block 0 --------------------------- #
    var lookup_table = HostDeviceTileTensor[.uint32](row_major(batch_size), ctx)
    var lut_host = lookup_table.host_tensor()
    lut_host[0] = UInt32(0)

    # ---- sink weights [num_heads] ---------------------------------------- #
    var sink_managed = HostDeviceTileTensor[dtype](row_major(num_q_heads), ctx)
    var sink_host = sink_managed.host_tensor()
    for h in range(num_q_heads):
        sink_host[h] = Scalar[dtype](sink_weights[h])

    # ---- build the collection + operands --------------------------------- #
    q.to_device()
    valid_lengths.to_device()
    cache_lengths_managed.to_device()
    kv_block.to_device()
    lookup_table.to_device()
    sink_managed.to_device()
    var kv_collection = ContinuousBatchingKVCacheCollection[dtype, kv_params](
        kv_block.device_tensor().as_unsafe_any_origin(),
        cache_lengths_managed.device_tensor(),
        lookup_table.device_tensor(),
        UInt32(max_prompt_len),
        UInt32(max_context_len),
    )
    var k_op = KVCacheMHAOperand(kv_collection.get_key_cache(layer_idx))
    var v_op = KVCacheMHAOperand(kv_collection.get_value_cache(layer_idx))

    var sink_dev = sink_managed.device_tensor()
    var sink_opt = OptionalReg[ImmutTileTensor1D[dtype]](
        immut_tile_tensor_1d(sink_dev.ptr, num_q_heads)
    )
    var valid_lengths_dev = valid_lengths.device_tensor()

    naive_fa_decode_apple[
        sink=True,
        _use_valid_length=True,
        _is_cache_length_accurate=False,
    ](
        q.device_tensor(),
        k_op,
        v_op,
        NullMask(),
        test_output.device_tensor(),
        immut_tile_tensor_1d(
            valid_lengths_dev.ptr, valid_lengths_dev.num_elements()
        ),
        Float32(0.0),  # scale = 0 -> all QK logits exactly 0
        batch_size,
        max_prompt_len,
        max_context_len,
        num_q_heads,
        depth,
        group,
        ctx,
        sink_opt,
    )
    ctx.synchronize()

    # ---- assert closed-form mass per head -------------------------------- #
    test_output.to_host()
    var out = test_output.host_tensor()
    for h in range(num_q_heads):
        var want = Float32(num_keys) / (
            Float32(num_keys) + exp(sink_weights[h])
        )
        for d in range(depth):
            var got = out[0, 0, h, d].cast[.float32]()[0]
            assert_almost_equal(got, want, atol=2e-2, rtol=2e-2)
    _ = q^
    _ = test_output^
    _ = valid_lengths^
    _ = cache_lengths_managed^
    _ = kv_block^
    _ = lookup_table^
    _ = sink_managed^
    print("    PASS")


comptime kv_d64_h2 = KVCacheStaticParams(num_heads=2, head_size=64)
comptime kv_d128_h2 = KVCacheStaticParams(num_heads=2, head_size=128)
comptime kv_d64_gqa = KVCacheStaticParams(num_heads=2, head_size=64)


def run_all(ctx: DeviceContext) raises:
    print("== test_naive_fa_decode_apple_sink (closed-form mass reference)")

    # The exact regression: num_keys=64 (spans 2 SplitSize=32 partitions),
    # depth=128, two heads with distinct sinks 5.0 / 3.0. Pre-fix this yielded
    # 1.0; correct is 64/(64+exp(5))~=0.30 and 64/(64+exp(3))~=0.76.
    _run_sink_closed_form[2, kv_d128_h2](64, [Float32(5.0), Float32(3.0)], ctx)

    # depth=64, fewer keys (single partition) -> seq_len=1 with sink seeded.
    _run_sink_closed_form[2, kv_d64_h2](16, [Float32(4.0), Float32(2.0)], ctx)

    # Larger num_keys (4 partitions) so the split-0-only seed is exercised
    # across more cross-split combines.
    _run_sink_closed_form[2, kv_d64_h2](128, [Float32(6.0), Float32(1.0)], ctx)

    # GQA: 8 query heads, 2 KV heads.
    _run_sink_closed_form[8, kv_d64_gqa](
        96,
        [
            Float32(5.0),
            Float32(4.5),
            Float32(4.0),
            Float32(3.5),
            Float32(3.0),
            Float32(2.5),
            Float32(2.0),
            Float32(1.5),
        ],
        ctx,
    )

    print("== all decode-sink cases PASS")


def main() raises:
    comptime if not has_apple_gpu_accelerator():
        print("SKIP: naive_fa_decode_apple targets Apple silicon GPUs only")
        return
    seed(42)
    with DeviceContext() as ctx:
        if ctx.compute_capability() != 5:
            # The sink seed is dtype-agnostic, but keep this Apple-decode test
            # aligned with the M5-gated suite.
            print("SKIP: Apple M5 required")
            return
        run_all(ctx)
