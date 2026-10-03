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

from std.random import random_ui64, seed
from std.math.uutils import udivmod

from max.gpu.host import DeviceContext
from kv_cache_test_utils import random_distinct
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import Coord, TileTensor, Idx, row_major, coord
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout._fillers import random
from linalg.matmul.gpu import _matmul_gpu
from nn.kv_cache import _fused_qkv_matmul_kv_cache_impl
from std.testing import assert_almost_equal


comptime kv_params_replit = KVCacheStaticParams(num_heads=8, head_size=128)
comptime replit_num_q_heads = 24

comptime kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
comptime llama_num_q_heads = 32


def execute_fused_qkv_matmul[
    num_q_heads: Int, dtype: DType, kv_params: KVCacheStaticParams
](
    batch_size: Int,
    prompt_len: Int,
    max_seq_len: Int,
    cache_sizes: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
) raises:
    comptime hidden_size = num_q_heads * kv_params.head_size
    comptime kv_hidden_size = kv_params.num_heads * kv_params.head_size
    comptime fused_hidden_size = (2 * kv_hidden_size) + hidden_size
    comptime num_blocks = 32
    comptime CollectionType = ContinuousBatchingKVCacheCollection[
        dtype, kv_params, MutAnyOrigin, ImmutAnyOrigin, ImmutAnyOrigin
    ]

    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured max_batch_size (",
        num_blocks,
        ")",
    )

    # Define shapes
    var hidden_state_shape = Coord(batch_size, prompt_len, Idx[hidden_size])
    var weight_shape = coord[fused_hidden_size, hidden_size]
    var ref_output_shape = Coord(batch_size * prompt_len, fused_hidden_size)
    var test_output_shape = Coord(batch_size, prompt_len, hidden_size)
    var lengths_shape = Coord(Int64(batch_size))
    var kv_block_shape = Coord(
        Int64(num_blocks),
        Int64(2),
        Int64(num_layers),
        Int64(max_seq_len),
        Idx[kv_params.num_heads],
        Idx[kv_params.head_size],
    )

    # Initialize hidden state
    var hidden_state = HostDeviceTileTensor[dtype](
        row_major(hidden_state_shape), ctx
    )
    var hidden_state_host = hidden_state.host_tensor()
    random(hidden_state_host)
    hidden_state.to_device()

    var hidden_state_device_2d = TileTensor(
        hidden_state.device_tensor()._storage,
        row_major(batch_size * prompt_len, Idx[hidden_size]),
    )

    # Keep matmul weights on a direct device buffer; _matmul_gpu expects this
    # static layout path and currently does not compose well with managed views.
    var weight_device = ctx.enqueue_create_buffer[dtype](weight_shape.product())
    with weight_device.map_to_host() as weight_host_ptr:
        var weight_host = TileTensor(weight_host_ptr, row_major(weight_shape))
        random(weight_host)

    # Initialize reference output
    var ref_output = HostDeviceTileTensor[dtype](
        row_major(ref_output_shape), ctx
    )

    # Initialize test output
    var test_output = HostDeviceTileTensor[dtype](
        row_major(test_output_shape), ctx
    )

    # Initialize our KVCache
    var is_context_encoding = True
    var cache_lengths = HostDeviceTileTensor[.uint32](
        row_major(lengths_shape), ctx
    )
    var cache_lengths_host = cache_lengths.host_tensor()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_sizes[i])
        if cache_lengths_host[i] != 0:
            is_context_encoding = False
    cache_lengths.to_device()

    var kv_block = HostDeviceTileTensor[dtype](row_major(kv_block_shape), ctx)

    var lookup_table = HostDeviceTileTensor[.uint32](
        row_major(lengths_shape), ctx
    )
    var lookup_table_host = lookup_table.host_tensor()

    # Assign each batch entry a distinct block. `random_ui64` is inclusive, so
    # the original draw range `[0, num_blocks - 1]` is a population of
    # `num_blocks` blocks.
    var lut_blocks = random_distinct(num_blocks, batch_size)
    for idx in range(batch_size):
        lookup_table_host[idx] = UInt32(lut_blocks[idx])
    lookup_table.to_device()

    # The collection spells its block strides symbolically in `kv_params`,
    # which the compiler cannot fold against `row_major`'s; the two layouts
    # are structurally identical.
    var kv_collection_device = CollectionType(
        rebind[CollectionType.blocks_tt_type](
            kv_block.device_tensor().as_unsafe_any_origin()
        ),
        cache_lengths.device_tensor().as_imm().as_unsafe_any_origin(),
        lookup_table.device_tensor().as_imm().as_unsafe_any_origin(),
        UInt32(max_seq_len),
        UInt32(0 if is_context_encoding else max_seq_len),
    )

    # Create device tensors for kernel calls
    var hidden_state_device_tensor = hidden_state.device_tensor()
    var weight_device_tensor = TileTensor(
        weight_device, row_major(weight_shape)
    )
    var test_output_device_tensor = test_output.device_tensor()

    # Create valid_lengths - all sequences have full prompt_len valid
    var valid_lengths = HostDeviceTileTensor[.uint32](
        row_major(lengths_shape), ctx
    )
    var valid_lengths_host = valid_lengths.host_tensor()
    for i in range(batch_size):
        valid_lengths_host[i] = UInt32(prompt_len)
    valid_lengths.to_device()
    var valid_lengths_tensor = valid_lengths.device_tensor()

    _fused_qkv_matmul_kv_cache_impl[target="gpu"](
        hidden_state_device_tensor,
        weight_device_tensor,
        kv_collection_device,
        UInt32(layer_idx),
        valid_lengths_tensor,
        test_output_device_tensor,
        ctx,
    )

    var ref_output_device_ndbuffer = TileTensor(
        ref_output.device_tensor().ptr,
        row_major(ref_output_shape[0], Idx[fused_hidden_size]),
    )
    var weight_device_ndbuffer = TileTensor(
        weight_device,
        row_major(Idx[fused_hidden_size], Idx[hidden_size]),
    )

    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        ref_output_device_ndbuffer,
        hidden_state_device_2d,
        weight_device_ndbuffer,
        ctx,
    )

    kv_block.to_host()
    test_output.to_host()
    ref_output.to_host()
    var kv_block_host_after = kv_block.host_tensor()
    var test_output_host = test_output.host_tensor()
    var ref_output_host = ref_output.host_tensor()
    var kv_collection_host = CollectionType(
        rebind[CollectionType.blocks_tt_type](
            kv_block_host_after.as_unsafe_any_origin()
        ),
        cache_lengths_host.as_imm().as_unsafe_any_origin(),
        lookup_table_host.as_imm().as_unsafe_any_origin(),
        UInt32(max_seq_len),
        UInt32(0 if is_context_encoding else max_seq_len),
    )

    var k_cache_host = kv_collection_host.get_key_cache(layer_idx)
    var v_cache_host = kv_collection_host.get_value_cache(layer_idx)
    for bs in range(batch_size):
        for s in range(prompt_len):
            for q_dim in range(hidden_size):
                assert_almost_equal(
                    ref_output_host[bs * prompt_len + s, q_dim],
                    test_output_host[bs, s, q_dim],
                )

            for k_dim in range(kv_hidden_size):
                var head_idx, head_dim_idx = udivmod(k_dim, kv_params.head_size)
                assert_almost_equal(
                    ref_output_host[bs * prompt_len + s, hidden_size + k_dim],
                    k_cache_host.load[width=1](
                        bs,
                        head_idx,
                        cache_sizes[bs] + s,
                        head_dim_idx,
                    ),
                )

            for v_dim in range(kv_hidden_size):
                var head_idx, head_dim_idx = udivmod(v_dim, kv_params.head_size)
                assert_almost_equal(
                    ref_output_host[
                        bs * prompt_len + s,
                        hidden_size + kv_hidden_size + v_dim,
                    ],
                    v_cache_host.load[width=1](
                        bs,
                        head_idx,
                        cache_sizes[bs] + s,
                        head_dim_idx,
                    ),
                )


def execute_fused_matmul_suite(ctx: DeviceContext) raises:
    comptime dtypes = (DType.float32, DType.bfloat16)

    comptime for dtype_idx in range(2):
        comptime dtype = dtypes[dtype_idx]
        for bs in [1, 16]:
            var ce_cache_sizes = List[Int]()
            var tg_cache_sizes = List[Int]()
            for _ in range(bs):
                tg_cache_sizes.append(Int(random_ui64(0, 100)))
                ce_cache_sizes.append(0)

            # llama3 context encoding
            execute_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3
            ](bs, 128, 1024, ce_cache_sizes, 4, 1, ctx)

            execute_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3
            ](bs, 512, 1024, ce_cache_sizes, 4, 0, ctx)

            # llama3 token gen
            execute_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3
            ](bs, 1, 1024, tg_cache_sizes, 4, 3, ctx)

            execute_fused_qkv_matmul[
                llama_num_q_heads, dtype, kv_params_llama3
            ](bs, 1, 1024, tg_cache_sizes, 4, 0, ctx)


def main() raises:
    seed(42)
    with DeviceContext() as ctx:
        execute_fused_matmul_suite(ctx)
