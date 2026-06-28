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
"""Correctness test for the fused QK RMSNorm + RoPE kernel.

Verifies `fused_qk_rms_norm_rope_ragged_paged` against the two-step reference
(`fused_qk_rms_norm_ragged_paged` followed by `fused_qk_rope_ragged`) on
identical inputs. Covers the two MiniMax-M3 RoPE geometries:

  * head_dim=128, rope_dim=128 (full rope, main attention): 64 Q-heads, 4 KV-heads.
  * head_dim=128, rope_dim=64  (partial rope, sparse indexer): 4 Q-heads, 1 KV-head.

Both BF16 and non-interleaved (safetensors) RoPE, matching M3.
"""

from std.collections import Set
from std.math import ceildiv
from std.random import random_ui64

from std.gpu.host import DeviceContext
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
from layout._fillers import random
from std.memory import memcpy

from nn.fused_qk_rope import fused_qk_rope_ragged
from nn.kv_cache import (
    fused_qk_rms_norm_ragged_paged,
    fused_qk_rms_norm_rope_ragged_paged,
)
from std.testing import assert_almost_equal

from std.utils import Index, IndexList


def run_fused_qk_rms_norm_rope[
    dtype: DType,
    head_size: Int,
    rope_dim: Int,
    num_q_heads: Int,
    num_k_heads: Int,
](
    ctx: DeviceContext,
    # The two-step reference rounds the normed Q/K to BF16 *before* RoPE (the
    # rms_norm kernel writes a BF16 intermediate that the rope kernel reads
    # back). The fused kernel keeps the normed value in FP32 through the
    # rotation and rounds once at the end, so it is strictly more accurate but
    # differs from the reference by ~1 BF16 ULP of the normed value, amplified
    # through the complex multiply. Tolerate that intermediate-rounding gap.
    rtol: Float64 = 6e-2,
    atol: Float64 = 1e-2,
) raises:
    print(
        "== run_fused_qk_rms_norm_rope dtype=",
        dtype,
        " head_size=",
        head_size,
        " rope_dim=",
        rope_dim,
        " num_q_heads=",
        num_q_heads,
        " num_k_heads=",
        num_k_heads,
    )

    comptime kv_params = KVCacheStaticParams(
        num_heads=num_k_heads, head_size=head_size
    )
    comptime num_paged_blocks = 32
    comptime page_size = 128
    comptime num_layers = 1
    comptime layer_idx = 0
    comptime max_seq_len = 1024
    comptime weight_offset = 1.0
    var epsilon = Scalar[dtype](1e-6)

    var prompt_lens = [16, 24, 8, 32]
    var cache_lens = [0, 7, 13, 5]
    var batch_size = len(prompt_lens)

    var total_length = 0
    var max_cache_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        total_length += prompt_lens[i]
        max_cache_length = max(max_cache_length, cache_lens[i])
        max_full_context_length = max(
            max_full_context_length, cache_lens[i] + prompt_lens[i]
        )
        max_prompt_length = max(max_prompt_length, prompt_lens[i])

    comptime cache_lengths_layout = Layout.row_major(UNKNOWN_VALUE)
    comptime kv_block_layout = Layout.row_major(
        UNKNOWN_VALUE,
        2,
        UNKNOWN_VALUE,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    comptime paged_lut_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var row_offsets_tile_layout = row_major(batch_size + 1)
    comptime freqs_tile_layout = row_major[max_seq_len, rope_dim]()

    var row_offsets_shape = Index(batch_size + 1)
    var cache_lengths_shape = Index(batch_size)
    var q_ragged_shape = IndexList[3](total_length, num_q_heads, head_size)
    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var paged_lut_shape = IndexList[2](
        batch_size, ceildiv(max_full_context_length, page_size)
    )
    var freqs_shape = IndexList[2](max_seq_len, rope_dim)

    var row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    var q_in_device = ctx.enqueue_create_buffer[dtype](
        q_ragged_shape.flattened_length()
    )
    var q_norm_ref_device = ctx.enqueue_create_buffer[dtype](
        q_ragged_shape.flattened_length()
    )
    var q_rope_ref_device = ctx.enqueue_create_buffer[dtype](
        q_ragged_shape.flattened_length()
    )
    var q_fused_device = ctx.enqueue_create_buffer[dtype](
        q_ragged_shape.flattened_length()
    )
    var gamma_q_device = ctx.enqueue_create_buffer[dtype](head_size)
    var gamma_k_device = ctx.enqueue_create_buffer[dtype](head_size)
    var kv_block_ref_device = ctx.enqueue_create_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    var kv_block_fused_device = ctx.enqueue_create_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    var paged_lut_device = ctx.enqueue_create_buffer[DType.uint32](
        paged_lut_shape.flattened_length()
    )
    var freqs_device = ctx.enqueue_create_buffer[dtype](
        freqs_shape.flattened_length()
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size + 1
    )
    var cache_lengths_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size
    )
    var offset = 0
    for i in range(batch_size):
        row_offsets_host[i] = UInt32(offset)
        cache_lengths_host[i] = UInt32(cache_lens[i])
        offset += prompt_lens[i]
    row_offsets_host[batch_size] = UInt32(offset)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    comptime q_ragged_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, head_size
    )
    var q_ragged_runtime_layout = RuntimeLayout[q_ragged_layout].row_major(
        q_ragged_shape
    )
    with q_in_device.map_to_host() as q_in_host:
        var q_in_tensor = LayoutTensor[dtype, q_ragged_layout](
            q_in_host, q_ragged_runtime_layout
        )
        random(q_in_tensor)

    comptime gamma_layout = Layout.row_major(head_size)
    var gamma_runtime_layout = RuntimeLayout[gamma_layout].row_major(
        Index(head_size)
    )
    with gamma_q_device.map_to_host() as gamma_q_host:
        var gamma_q_tensor = LayoutTensor[dtype, gamma_layout](
            gamma_q_host, gamma_runtime_layout
        )
        random(gamma_q_tensor)
    with gamma_k_device.map_to_host() as gamma_k_host:
        var gamma_k_tensor = LayoutTensor[dtype, gamma_layout](
            gamma_k_host, gamma_runtime_layout
        )
        random(gamma_k_tensor)

    comptime freqs_layout = Layout.row_major(max_seq_len, rope_dim)
    var freqs_runtime_layout = RuntimeLayout[freqs_layout].row_major(
        freqs_shape
    )
    with freqs_device.map_to_host() as freqs_host:
        var freqs_init = LayoutTensor[dtype, freqs_layout](
            freqs_host, freqs_runtime_layout
        )
        random(freqs_init)

    var kv_block_runtime_layout = RuntimeLayout[kv_block_layout].row_major(
        kv_block_shape
    )
    var kv_block_host = ctx.enqueue_create_host_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    var kv_block_host_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_host.unsafe_ptr(), kv_block_runtime_layout
    )
    random(kv_block_host_tensor)
    ctx.enqueue_copy(kv_block_ref_device, kv_block_host)
    ctx.enqueue_copy(kv_block_fused_device, kv_block_host)
    ctx.synchronize()

    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )
    with paged_lut_device.map_to_host() as paged_lut_host:
        var paged_lut_tensor = LayoutTensor[DType.uint32, paged_lut_layout](
            paged_lut_host, paged_lut_runtime_layout
        )
        var paged_lut_set = Set[Int]()
        for bs in range(batch_size):
            var seq_len = cache_lens[bs] + prompt_lens[bs]
            for block_idx in range(0, ceildiv(seq_len, page_size)):
                var randval = Int(random_ui64(0, num_paged_blocks - 1))
                while randval in paged_lut_set:
                    randval = Int(random_ui64(0, num_paged_blocks - 1))
                paged_lut_set.add(randval)
                paged_lut_tensor[bs, block_idx] = UInt32(randval)

    var row_offsets_tensor = TileTensor(
        row_offsets_device, row_offsets_tile_layout
    )
    var q_in_tt = TileTensor(
        q_in_device,
        row_major((total_length, Idx[num_q_heads], Idx[head_size])),
    )
    var q_norm_ref_tt = TileTensor(
        q_norm_ref_device,
        row_major((total_length, Idx[num_q_heads], Idx[head_size])),
    )
    var q_rope_ref_tt = TileTensor(
        q_rope_ref_device,
        row_major((total_length, Idx[num_q_heads], Idx[head_size])),
    )
    var q_fused_tt = TileTensor(
        q_fused_device,
        row_major((total_length, Idx[num_q_heads], Idx[head_size])),
    )
    var gamma_q_tt = TileTensor(gamma_q_device, row_major[head_size]())
    var gamma_k_tt = TileTensor(gamma_k_device, row_major[head_size]())
    var freqs_tt = TileTensor(freqs_device, freqs_tile_layout)

    var cache_lengths_tensor = LayoutTensor[
        mut=False, DType.uint32, Layout(UNKNOWN_VALUE)
    ](
        cache_lengths_device,
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(cache_lengths_shape),
    )
    var paged_lut_tensor = LayoutTensor[
        mut=False, DType.uint32, Layout.row_major[2]()
    ](
        paged_lut_device,
        RuntimeLayout[Layout.row_major[2]()].row_major(paged_lut_shape),
    )

    var ref_kv_block_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_ref_device, kv_block_runtime_layout
    )
    var fused_kv_block_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_fused_device, kv_block_runtime_layout
    )
    var ref_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        LayoutTensor[dtype, Layout.row_major[6]()](
            ref_kv_block_tensor.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                ref_kv_block_tensor.runtime_layout.shape.value.canonicalize(),
                ref_kv_block_tensor.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        cache_lengths_tensor,
        paged_lut_tensor,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )
    var fused_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        LayoutTensor[dtype, Layout.row_major[6]()](
            fused_kv_block_tensor.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                fused_kv_block_tensor.runtime_layout.shape.value.canonicalize(),
                fused_kv_block_tensor.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        cache_lengths_tensor,
        paged_lut_tensor,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )

    fused_qk_rms_norm_ragged_paged[target="gpu", multiply_before_cast=True](
        q_in_tt,
        ref_collection,
        gamma_q_tt,
        gamma_k_tt,
        epsilon,
        Scalar[dtype](weight_offset),
        UInt32(layer_idx),
        row_offsets_tensor,
        q_norm_ref_tt,
        ctx,
    )
    fused_qk_rope_ragged[
        ref_collection.CacheType, interleaved=False, target="gpu"
    ](
        q_proj=q_norm_ref_tt,
        input_row_offsets=row_offsets_tensor,
        kv_collection=ref_collection,
        freqs_cis=freqs_tt,
        position_ids=None,
        layer_idx=UInt32(layer_idx),
        output=q_rope_ref_tt,
        context=ctx,
    )

    fused_qk_rms_norm_rope_ragged_paged[
        target="gpu", multiply_before_cast=True, interleaved=False
    ](
        q_in_tt,
        fused_collection,
        gamma_q_tt,
        gamma_k_tt,
        freqs_tt,
        epsilon,
        Scalar[dtype](weight_offset),
        UInt32(layer_idx),
        row_offsets_tensor,
        q_fused_tt,
        ctx,
    )
    ctx.synchronize()

    print("comparing Q")
    with q_rope_ref_device.map_to_host() as q_rope_ref_host:
        with q_fused_device.map_to_host() as q_fused_host:
            var ref_t = LayoutTensor[dtype, q_ragged_layout](
                q_rope_ref_host, q_ragged_runtime_layout
            )
            var fused_t = LayoutTensor[dtype, q_ragged_layout](
                q_fused_host, q_ragged_runtime_layout
            )
            for tok in range(total_length):
                for h in range(num_q_heads):
                    for d in range(head_size):
                        assert_almost_equal(
                            fused_t[tok, h, d],
                            ref_t[tok, h, d],
                            rtol=rtol,
                            atol=atol,
                        )

    ctx.enqueue_copy(kv_block_host, kv_block_ref_device)
    var kv_block_fused_host = ctx.enqueue_create_host_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    ctx.enqueue_copy(kv_block_fused_host, kv_block_fused_device)
    var paged_lut_host_ptr = ctx.enqueue_create_host_buffer[DType.uint32](
        paged_lut_shape.flattened_length()
    )
    ctx.enqueue_copy(paged_lut_host_ptr, paged_lut_device)
    ctx.synchronize()

    var ref_kv_host_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_host.unsafe_ptr(), kv_block_runtime_layout
    )
    var fused_kv_host_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_fused_host.unsafe_ptr(), kv_block_runtime_layout
    )
    var paged_lut_host_tensor = LayoutTensor[DType.uint32, paged_lut_layout](
        paged_lut_host_ptr.unsafe_ptr(), paged_lut_runtime_layout
    )

    print("comparing K")
    for bs in range(batch_size):
        var cache_len = cache_lens[bs]
        for tok in range(prompt_lens[bs]):
            var ctx_pos = cache_len + tok
            var block = paged_lut_host_tensor[bs, ctx_pos // page_size]
            var in_page = ctx_pos % page_size
            for h in range(kv_params.num_heads):
                for d in range(head_size):
                    # kv_idx 0 == key cache.
                    assert_almost_equal(
                        fused_kv_host_tensor[
                            Int(block), 0, layer_idx, in_page, h, d
                        ],
                        ref_kv_host_tensor[
                            Int(block), 0, layer_idx, in_page, h, d
                        ],
                        rtol=rtol,
                        atol=atol,
                    )

    # FIXME(MSTDL-2742): HostBuffer is origin incorrect.
    _ = UnsafePointer(to=kv_block_host).as_unsafe_any_origin()[]
    _ = UnsafePointer(to=kv_block_fused_host).as_unsafe_any_origin()[]

    _ = row_offsets_device^
    _ = cache_lengths_device^
    _ = q_in_device^
    _ = q_norm_ref_device^
    _ = q_rope_ref_device^
    _ = q_fused_device^
    _ = gamma_q_device^
    _ = gamma_k_device^
    _ = kv_block_ref_device^
    _ = kv_block_fused_device^
    _ = paged_lut_device^
    _ = freqs_device^


def main() raises:
    with DeviceContext() as ctx:
        run_fused_qk_rms_norm_rope[
            DType.bfloat16,
            head_size=128,
            rope_dim=128,
            num_q_heads=64,
            num_k_heads=4,
        ](ctx)
        run_fused_qk_rms_norm_rope[
            DType.bfloat16,
            head_size=128,
            rope_dim=64,
            num_q_heads=4,
            num_k_heads=1,
        ](ctx)
