# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-no-debug %s

from collections import Set
from random import random_ui64, seed
from memory import memcpy
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    fill,
    linspace,
    random,
    zero,
)
from kv_cache.types import (
    ContinuousBatchingKVCache,
    KVCacheStaticParams,
)
from linalg.matmul_gpu import _matmul_gpu
from memory import UnsafePointer
from nn.kv_cache_ragged import _fused_qkv_matmul_kv_cache_ragged_impl
from runtime.asyncrt import MojoCallContextPtr
from testing import assert_almost_equal

from utils import IndexList
from gpu.host.info import DEFAULT_GPU_ARCH

alias kv_params_replit = KVCacheStaticParams(num_heads=8, head_size=128)
alias replit_num_q_heads = 24

alias kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
alias llama_num_q_heads = 32


def execute_fused_qkv_matmul[
    num_q_heads: Int, type: DType, kv_params: KVCacheStaticParams
](
    prompt_lens: List[Int],
    max_seq_length_cache: Int,
    cache_sizes: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
):
    alias hidden_size = num_q_heads * kv_params.head_size
    alias kv_hidden_size = kv_params.num_heads * kv_params.head_size
    alias fused_hidden_size = (2 * kv_hidden_size) + hidden_size
    alias num_blocks = 32

    alias CacheType = ContinuousBatchingKVCache[
        type,
        kv_params,
    ]

    debug_assert(
        len(prompt_lens) == len(cache_sizes),
        (
            "mismatch between cache_sizes and prompt_lens, both should be"
            " batch_size in length"
        ),
    )

    batch_size = len(prompt_lens)

    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test ("
        + str(batch_size)
        + ") is larger than configured max_batch_size ("
        + str(num_blocks)
        + ")",
    )

    # initialize input_row_offset
    input_row_offset_host = HostNDBuffer[DType.uint32, 1]((batch_size + 1,))

    total_length = 0
    max_seq_length_batch = -1
    for i in range(batch_size):
        input_row_offset_host.tensor[i] = total_length

        curr_len = prompt_lens[i]
        total_length += curr_len
        if curr_len > max_seq_length_batch:
            max_seq_length_batch = curr_len

    input_row_offset_host.tensor[batch_size] = total_length

    input_row_offset_device = input_row_offset_host.copy_to_device(ctx)

    # initialize ragged hidden state
    hidden_state_ragged_host = HostNDBuffer[
        type, 2, DimList(Dim(), hidden_size)
    ](IndexList[2](total_length, hidden_size))

    random(hidden_state_ragged_host.tensor)
    hidden_state_ragged_device = hidden_state_ragged_host.copy_to_device(ctx)

    # initialize padded hidden state
    hidden_state_padded_host = HostNDBuffer[
        type, 2, DimList(Dim(), hidden_size)
    ](IndexList[2](batch_size * max_seq_length_batch, hidden_size))

    # copy over the ragged values to the padded tensor.
    # Don't worry about padded values, we won't read them.
    for bs in range(batch_size):
        unpadded_seq_len = prompt_lens[bs]
        ragged_start_idx = int(input_row_offset_host.tensor[bs])
        for s in range(unpadded_seq_len):
            padded_ptr = hidden_state_padded_host.tensor._offset(
                (bs * max_seq_length_batch + s, 0)
            )
            ragged_ptr = hidden_state_ragged_host.tensor._offset(
                (ragged_start_idx + s, 0)
            )
            memcpy(padded_ptr, ragged_ptr, hidden_size)

    hidden_state_padded_device = hidden_state_padded_host.copy_to_device(ctx)

    # initialize the weights
    weight_host = HostNDBuffer[
        type,
        2,
        DimList(fused_hidden_size, hidden_size),
    ](IndexList[2](fused_hidden_size, hidden_size))
    random(weight_host.tensor)
    weight_device = weight_host.copy_to_device(ctx)

    # initialize reference output
    ref_output_host = HostNDBuffer[type, 2, DimList(Dim(), fused_hidden_size),](
        IndexList[2](
            batch_size * max_seq_length_batch,
            fused_hidden_size,
        )
    )
    ref_output_device = ref_output_host.copy_to_device(ctx)

    test_output_host = HostNDBuffer[type, 2, DimList(Dim(), hidden_size),](
        IndexList[2](
            total_length,
            hidden_size,
        )
    )
    test_output_device = test_output_host.copy_to_device(ctx)

    # initialize our KVCache
    var is_context_encoding = True
    var cache_lengths_host = HostNDBuffer[DType.uint32, 1]((batch_size,))
    for i in range(batch_size):
        cache_lengths_host.tensor[i] = cache_sizes[i]
        if cache_lengths_host.tensor[i] != 0:
            is_context_encoding = False

    var cache_lengths_device = cache_lengths_host.copy_to_device(ctx)

    kv_block_host = HostNDBuffer[type, 6,](
        IndexList[6](
            num_blocks,
            2,
            num_layers,
            max_seq_length_cache,
            kv_params.num_heads,
            kv_params.head_size,
        ),
    )
    kv_block_device = kv_block_host.copy_to_device(ctx)
    var lookup_table_host = HostNDBuffer[DType.uint32, 1,](
        IndexList[1](
            batch_size,
        ),
    )

    # hacky way to select random blocks.
    var block_idx_set = Set[Int]()
    var idx = 0
    while idx < batch_size:
        var randval = int(random_ui64(0, num_blocks - 1))
        if randval in block_idx_set:
            continue

        block_idx_set.add(randval)
        lookup_table_host.tensor[idx] = UInt32(randval)
        idx += 1

    var lookup_table_device = lookup_table_host.copy_to_device(ctx)

    var k_cache_device = CacheType(
        kv_block_device.tensor,
        cache_lengths_device.tensor,
        lookup_table_device.tensor,
        is_context_encoding,
        layer_idx,
        CacheType.KeyIdx,
    )
    var k_cache_host = CacheType(
        kv_block_host.tensor,
        cache_lengths_host.tensor,
        lookup_table_host.tensor,
        is_context_encoding,
        layer_idx,
        CacheType.KeyIdx,
    )
    var v_cache_device = CacheType(
        kv_block_device.tensor,
        cache_lengths_device.tensor,
        lookup_table_device.tensor,
        is_context_encoding,
        layer_idx,
        CacheType.ValueIdx,
    )
    var v_cache_host = CacheType(
        kv_block_host.tensor,
        cache_lengths_host.tensor,
        lookup_table_host.tensor,
        is_context_encoding,
        layer_idx,
        CacheType.ValueIdx,
    )

    # execute the matmul
    _fused_qkv_matmul_kv_cache_ragged_impl[target="cuda",](
        hidden_state_ragged_device.tensor,
        input_row_offset_device.tensor,
        weight_device.tensor,
        k_cache_device,
        v_cache_device,
        test_output_device.tensor,
        ctx,
    )

    _matmul_gpu[
        target=DEFAULT_GPU_ARCH, use_tensor_core=True, transpose_b=True
    ](
        ref_output_device.tensor,
        hidden_state_padded_device.tensor,
        weight_device.tensor,
        ctx,
    )

    ctx.enqueue_copy_from_device(
        kv_block_host.tensor.data, kv_block_device.buffer
    )
    ctx.enqueue_copy_from_device(
        test_output_host.tensor.data, test_output_device.buffer
    )
    ctx.enqueue_copy_from_device(
        ref_output_host.tensor.data, ref_output_device.buffer
    )
    ctx.synchronize()

    ref_out = ref_output_host.tensor
    test_out = test_output_host.tensor
    for bs in range(batch_size):
        prompt_len = prompt_lens[bs]
        ragged_offset = int(input_row_offset_host.tensor[bs])
        for s in range(prompt_len):
            for q_dim in range(hidden_size):
                assert_almost_equal(
                    ref_out[
                        bs * max_seq_length_batch + s,
                        q_dim,
                    ],
                    test_out[ragged_offset + s, q_dim],
                )

            for k_dim in range(kv_hidden_size):
                head_idx = k_dim // kv_params.head_size
                head_dim_idx = k_dim % kv_params.head_size
                assert_almost_equal(
                    ref_out[
                        bs * max_seq_length_batch + s,
                        hidden_size + k_dim,
                    ],
                    k_cache_host.load[type, width=1](
                        bs,
                        head_idx,
                        cache_sizes[bs] + s,
                        head_dim_idx,
                    ),
                )

            for v_dim in range(kv_hidden_size):
                head_idx = v_dim // kv_params.head_size
                head_dim_idx = v_dim % kv_params.head_size
                assert_almost_equal(
                    ref_out[
                        bs * max_seq_length_batch + s,
                        hidden_size + kv_hidden_size + v_dim,
                    ],
                    v_cache_host.load[type, width=1](
                        bs,
                        head_idx,
                        cache_sizes[bs] + s,
                        head_dim_idx,
                    ),
                )

    _ = hidden_state_ragged_device^
    _ = hidden_state_ragged_host^
    _ = weight_device^
    _ = weight_host^
    _ = ref_output_device^
    _ = ref_output_host^
    _ = test_output_device^
    _ = test_output_host^
    _ = kv_block_device^
    _ = kv_block_host^
    _ = lookup_table_device^
    _ = lookup_table_host^
    _ = cache_lengths_device^
    _ = cache_lengths_host^
    _ = input_row_offset_device^


def execute_fused_matmul_suite(ctx: DeviceContext):
    alias types = Tuple[DType, DType](DType.float32, DType.bfloat16)

    @parameter
    for type_idx in range(2):
        alias type = types.get[type_idx, DType]()
        for bs_ref in List[Int](1, 16):
            bs = bs_ref[]
            ce_cache_sizes = List[Int]()
            ce_seq_lens = List[Int]()
            tg_cache_sizes = List[Int]()
            tg_seq_lens = List[Int]()
            for _ in range(bs):
                tg_seq_lens.append(1)
                tg_cache_sizes.append(int(random_ui64(0, 100)))
                ce_seq_lens.append(int(random_ui64(0, 100)))
                ce_cache_sizes.append(0)

            # llama3 context encoding
            execute_fused_qkv_matmul[llama_num_q_heads, type, kv_params_llama3](
                ce_seq_lens, 1024, ce_cache_sizes, 4, 1, ctx
            )

            execute_fused_qkv_matmul[llama_num_q_heads, type, kv_params_llama3](
                ce_seq_lens, 1024, ce_cache_sizes, 4, 0, ctx
            )

            # llama3 token gen
            execute_fused_qkv_matmul[llama_num_q_heads, type, kv_params_llama3](
                tg_seq_lens, 1024, tg_cache_sizes, 4, 3, ctx
            )

            execute_fused_qkv_matmul[llama_num_q_heads, type, kv_params_llama3](
                tg_seq_lens, 1024, tg_cache_sizes, 4, 0, ctx
            )


def main():
    seed(42)
    with DeviceContext() as ctx:
        execute_fused_matmul_suite(ctx)
