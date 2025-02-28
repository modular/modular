# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# TODO: MSTDL-1147 understand why this test fails with asserts turned on.
# RUN: %mojo-no-debug-no-assert %s

from collections import Set
from math import ceildiv
from random import random_ui64, seed

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from gpu.host.info import DEFAULT_GPU_ARCH
from internal_utils import DeviceNDBuffer, HostNDBuffer, fill, random, zero
from kv_cache.types import (
    ContinuousBatchingKVCache,
    KVCacheStaticParams,
    KVCacheT,
    KVCollectionT,
    PagedKVCache,
)
from linalg.matmul import matmul
from linalg.matmul_gpu import _matmul_gpu
from memory import UnsafePointer, memcpy
from nn.kv_cache_ragged import (
    _fused_qkv_matmul_kv_cache_ragged_impl,
    _matmul_kv_cache_ragged_impl,
)
from testing import assert_almost_equal

from utils import IndexList

alias kv_params_replit = KVCacheStaticParams(num_heads=8, head_size=128)
alias replit_num_q_heads = 24

alias kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
alias llama_num_q_heads = 32


def _initialize_ragged_inputs[
    type: DType, hidden_size: Int
](
    mut input_row_offsets_host: HostNDBuffer[DType.uint32, 1],
    batch_size: Int,
    prompt_lens: List[Int],
    ctx: DeviceContext,
) -> (
    DeviceNDBuffer[DType.uint32, 1],
    DeviceNDBuffer[type, 2, DimList(Dim(), hidden_size)],
    DeviceNDBuffer[type, 2, DimList(Dim(), hidden_size)],
):
    """Initializes input row offsets and hidden state ragged tensor inputs."""
    total_length = 0
    max_seq_length_batch = -1
    for i in range(batch_size):
        input_row_offsets_host.tensor[i] = total_length

        curr_len = prompt_lens[i]
        total_length += curr_len
        if curr_len > max_seq_length_batch:
            max_seq_length_batch = curr_len

    input_row_offsets_host.tensor[batch_size] = total_length
    input_row_offsets_device = input_row_offsets_host.copy_to_device(ctx)

    # Initialize ragged hidden state.
    hidden_state_ragged_host = HostNDBuffer[
        type, 2, DimList(Dim(), hidden_size)
    ](IndexList[2](total_length, hidden_size))

    random(hidden_state_ragged_host.tensor)

    hidden_state_ragged_device = hidden_state_ragged_host.copy_to_device(ctx)

    # Initialize padded hidden state.
    hidden_state_padded_host = HostNDBuffer[
        type, 2, DimList(Dim(), hidden_size)
    ](IndexList[2](batch_size * max_seq_length_batch, hidden_size))

    # Copy over the ragged values to the padded tensor.
    # Don't worry about padded values, we won't read them.
    for bs in range(batch_size):
        unpadded_seq_len = prompt_lens[bs]
        ragged_start_idx = Int(input_row_offsets_host.tensor[bs])
        for s in range(unpadded_seq_len):
            padded_ptr = hidden_state_padded_host.tensor._offset(
                (bs * max_seq_length_batch + s, 0)
            )
            ragged_ptr = hidden_state_ragged_host.tensor._offset(
                (ragged_start_idx + s, 0)
            )
            memcpy(padded_ptr, ragged_ptr, hidden_size)

    hidden_state_padded_device = hidden_state_padded_host.copy_to_device(ctx)

    # Sync here so that HtoD transfers complete prior to host buffer dtor.
    ctx.synchronize()

    _ = hidden_state_ragged_host^
    _ = hidden_state_padded_host^

    return (
        input_row_offsets_device,
        hidden_state_ragged_device,
        hidden_state_padded_device,
    )


def execute_matmul_kv_cache_ragged[
    num_q_heads: Int,
    type: DType,
    kv_params: KVCacheStaticParams,
    rtol: Float64,
](
    prompt_lens: List[Int],
    max_seq_length_cache: Int,
    cache_sizes: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
):
    """Tests the KV cache matmul.

    Note that here `prompt_lens` indicates the sequence length of the hidden
    states, although in general the sequence may not originate from a prompt.
    For example, in cross attention the sequence would be from a sequence of
    patch embeddings of an image.
    """
    alias hidden_size = num_q_heads * kv_params.head_size
    alias kv_hidden_size = kv_params.num_heads * kv_params.head_size
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
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_blocks (",
        num_blocks,
        ")",
    )

    # Initialize input row offsets and hidden states.
    input_row_offsets_host = HostNDBuffer[DType.uint32, 1]((batch_size + 1,))
    input_row_offsets_device, hidden_state_ragged_device, hidden_state_padded_device = _initialize_ragged_inputs[
        type, hidden_size
    ](
        input_row_offsets_host, batch_size, prompt_lens, ctx
    )

    # Initialize the weights.
    weight_host = HostNDBuffer[
        type, 2, DimList(2 * kv_hidden_size, hidden_size)
    ](IndexList[2](2 * kv_hidden_size, hidden_size))
    random(weight_host.tensor)

    weight_device = weight_host.copy_to_device(ctx)

    # Initialize reference output.
    padded_batch_dim = hidden_state_padded_device.tensor.dim(0)
    max_seq_length_batch = padded_batch_dim // batch_size
    ref_output_host = HostNDBuffer[type, 2, DimList(Dim(), 2 * kv_hidden_size)](
        IndexList[2](padded_batch_dim, 2 * kv_hidden_size)
    )
    ref_output_device = ref_output_host.copy_to_device(ctx)

    # Initialize our KVCache.
    cache_lengths_host = HostNDBuffer[DType.uint32, 1]((batch_size,))
    max_prompt_len = 0
    max_context_len = 0
    for i in range(batch_size):
        cache_lengths_host.tensor[i] = cache_sizes[i]
        max_prompt_len = max(max_prompt_len, prompt_lens[i])
        max_context_len = max(
            max_context_len, Int(cache_sizes[i] + prompt_lens[i])
        )

    cache_lengths_device = cache_lengths_host.copy_to_device(ctx)
    kv_block_host = HostNDBuffer[type, 6](
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
    lookup_table_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size),
    )

    # Hacky way to select random blocks.
    block_idx_set = Set[Int]()
    idx = 0
    while idx < batch_size:
        randval = Int(random_ui64(0, num_blocks - 1))
        if randval in block_idx_set:
            continue

        block_idx_set.add(randval)
        lookup_table_host.tensor[idx] = UInt32(randval)
        idx += 1

    lookup_table_device = lookup_table_host.copy_to_device(ctx)

    k_cache_device = CacheType(
        kv_block_device.tensor,
        cache_lengths_device.tensor,
        lookup_table_device.tensor,
        max_prompt_len,
        max_context_len,
        layer_idx,
        CacheType.KeyIdx,
    )
    k_cache_host = CacheType(
        kv_block_host.tensor,
        cache_lengths_host.tensor,
        lookup_table_host.tensor,
        max_prompt_len,
        max_context_len,
        layer_idx,
        CacheType.KeyIdx,
    )
    v_cache_device = CacheType(
        kv_block_device.tensor,
        cache_lengths_device.tensor,
        lookup_table_device.tensor,
        max_prompt_len,
        max_context_len,
        layer_idx,
        CacheType.ValueIdx,
    )
    v_cache_host = CacheType(
        kv_block_host.tensor,
        cache_lengths_host.tensor,
        lookup_table_host.tensor,
        max_prompt_len,
        max_context_len,
        layer_idx,
        CacheType.ValueIdx,
    )

    # Execute test.
    _matmul_kv_cache_ragged_impl[target="gpu"](
        hidden_state_ragged_device.tensor,
        input_row_offsets_device.tensor,
        weight_device.tensor,
        k_cache_device,
        v_cache_device,
        ctx,
    )

    # Execute reference.
    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        ref_output_device.tensor,
        hidden_state_padded_device.tensor,
        weight_device.tensor,
        ctx,
    )

    ctx.enqueue_copy(kv_block_host.tensor.data, kv_block_device.buffer)
    ctx.enqueue_copy(ref_output_host.tensor.data, ref_output_device.buffer)
    ctx.synchronize()

    ref_out = ref_output_host.tensor
    for bs in range(batch_size):
        prompt_len = prompt_lens[bs]
        for s in range(prompt_len):
            for k_dim in range(kv_hidden_size):
                head_idx = k_dim // kv_params.head_size
                head_dim_idx = k_dim % kv_params.head_size
                assert_almost_equal(
                    ref_out[bs * max_seq_length_batch + s, k_dim],
                    k_cache_host.load[width=1](
                        bs,
                        head_idx,
                        cache_sizes[bs] + s,
                        head_dim_idx,
                    ),
                    rtol=rtol.cast[type](),
                )

            for v_dim in range(kv_hidden_size):
                head_idx = v_dim // kv_params.head_size
                head_dim_idx = v_dim % kv_params.head_size
                assert_almost_equal(
                    ref_out[
                        bs * max_seq_length_batch + s, kv_hidden_size + v_dim
                    ],
                    v_cache_host.load[width=1](
                        bs,
                        head_idx,
                        cache_sizes[bs] + s,
                        head_dim_idx,
                    ),
                    rtol=rtol.cast[type](),
                )

    _ = hidden_state_ragged_device^
    _ = hidden_state_padded_device^
    _ = weight_host^
    _ = weight_device^
    _ = ref_output_device^
    _ = ref_output_host^
    _ = kv_block_device^
    _ = kv_block_host^
    _ = lookup_table_host^
    _ = lookup_table_device^
    _ = cache_lengths_device^
    _ = cache_lengths_host^
    _ = input_row_offsets_device^
    _ = input_row_offsets_host^


def generic_assert_output_equals[
    cache_t: KVCacheT, type: DType, //, num_q_heads: Int, rtol: Float64
](
    k_cache: cache_t,
    v_cache: cache_t,
    ref_output_device: DeviceNDBuffer[type, 2, *_],
    test_output_device: DeviceNDBuffer[type, 2, *_],
    prompt_lens: List[Int],
    max_seq_length_batch: Int,
    ctx: DeviceContext,
):
    constrained[cache_t.type == type, "type mismatch"]()
    alias kv_params = cache_t.kv_params
    alias hidden_size = num_q_heads * kv_params.head_size
    alias kv_hidden_size = kv_params.num_heads * kv_params.head_size

    ref_output_host = HostNDBuffer[
        ref_output_device.type, ref_output_device.rank, ref_output_device.shape
    ](ref_output_device.tensor.dynamic_shape)
    test_output_host = HostNDBuffer[
        test_output_device.type,
        test_output_device.rank,
        test_output_device.shape,
    ](test_output_device.tensor.dynamic_shape)

    ctx.enqueue_copy(test_output_host.tensor.data, test_output_device.buffer)
    ctx.enqueue_copy(ref_output_host.tensor.data, ref_output_device.buffer)
    ctx.synchronize()

    batch_size = len(prompt_lens)

    ref_out = ref_output_host.tensor
    test_out = test_output_host.tensor
    ragged_offset = 0
    for bs in range(batch_size):
        prompt_len = prompt_lens[bs]
        for s in range(prompt_len):
            for q_dim in range(hidden_size):
                try:
                    assert_almost_equal(
                        ref_out[
                            bs * max_seq_length_batch + s,
                            q_dim,
                        ],
                        test_out[ragged_offset + s, q_dim],
                        rtol=rtol.cast[type](),
                    )
                except e:
                    print("Q", bs, s, q_dim)
                    raise e

            for k_dim in range(kv_hidden_size):
                head_idx = k_dim // kv_params.head_size
                head_dim_idx = k_dim % kv_params.head_size
                try:
                    assert_almost_equal(
                        ref_out[
                            bs * max_seq_length_batch + s,
                            hidden_size + k_dim,
                        ],
                        k_cache.load[width=1](
                            bs,
                            head_idx,
                            k_cache.cache_length(bs) + s,
                            head_dim_idx,
                        ).cast[type](),
                        rtol=rtol.cast[type](),
                    )
                except e:
                    print("K", bs, s, k_dim)
                    raise e

            for v_dim in range(kv_hidden_size):
                head_idx = v_dim // kv_params.head_size
                head_dim_idx = v_dim % kv_params.head_size
                try:
                    assert_almost_equal(
                        ref_out[
                            bs * max_seq_length_batch + s,
                            hidden_size + kv_hidden_size + v_dim,
                        ],
                        v_cache.load[width=1](
                            bs,
                            head_idx,
                            v_cache.cache_length(bs) + s,
                            head_dim_idx,
                        ).cast[type](),
                        rtol=rtol.cast[type](),
                    )
                except e:
                    print("V", bs, s, v_dim)
                    raise e

        ragged_offset += prompt_len

    _ = ref_output_host^
    _ = test_output_host^


def generic_execute_fused_qkv_cache_ragged[
    cache_t: KVCacheT, //,
    kv_params: KVCacheStaticParams,
    type: DType,
    num_q_heads: Int,
](
    prompt_lens: List[Int],
    cache_sizes: List[Int],
    k_cache: cache_t,
    v_cache: cache_t,
    ctx: DeviceContext,
    out result: (
        DeviceNDBuffer[
            type,
            2,
            DimList(
                Dim(),
                (kv_params.num_heads * 2 + num_q_heads) * kv_params.head_size,
            ),
        ],
        DeviceNDBuffer[
            type, 2, DimList(Dim(), num_q_heads * kv_params.head_size)
        ],
    ),
):
    """Executes fused QKV matmul, writing results kv_cache objects.

    Returns:
      - Tuple[HostNDBuffer, HostNDBuffer]: (ref_output, test_output).
    """
    alias hidden_size = num_q_heads * kv_params.head_size
    alias kv_hidden_size = kv_params.num_heads * kv_params.head_size
    alias fused_hidden_size = (2 * kv_hidden_size) + hidden_size
    alias num_blocks = 32

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
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured max_batch_size (",
        num_blocks,
        ")",
    )

    # Initialize input row offsets and hidden states.
    input_row_offsets_host = HostNDBuffer[DType.uint32, 1]((batch_size + 1,))
    input_row_offsets_device, hidden_state_ragged_device, hidden_state_padded_device = _initialize_ragged_inputs[
        type, hidden_size
    ](
        input_row_offsets_host, batch_size, prompt_lens, ctx
    )

    # initialize the weights
    weight_host = HostNDBuffer[
        type,
        2,
        DimList(fused_hidden_size, hidden_size),
    ](IndexList[2](fused_hidden_size, hidden_size))
    random(weight_host.tensor)
    weight_device = weight_host.copy_to_device(ctx)

    # initialize reference output
    padded_batch_dim = hidden_state_padded_device.tensor.dim(0)
    max_seq_length_batch = padded_batch_dim // batch_size
    ref_output_host = HostNDBuffer[type, 2, DimList(Dim(), fused_hidden_size)](
        IndexList[2](
            padded_batch_dim,
            fused_hidden_size,
        )
    )
    ref_output_device = ref_output_host.copy_to_device(ctx)

    total_length = hidden_state_ragged_device.tensor.dim(0)
    test_output_host = HostNDBuffer[type, 2, DimList(Dim(), hidden_size)](
        IndexList[2](total_length, hidden_size)
    )
    test_output_device = test_output_host.copy_to_device(ctx)

    # execute the matmul
    _fused_qkv_matmul_kv_cache_ragged_impl[target="gpu"](
        hidden_state_ragged_device.tensor,
        input_row_offsets_device.tensor,
        weight_device.tensor,
        k_cache,
        v_cache,
        test_output_device.tensor,
        ctx,
    )

    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        ref_output_device.tensor,
        hidden_state_padded_device.tensor,
        weight_device.tensor,
        ctx,
    )

    _ = weight_device^
    _ = weight_host^
    _ = ref_output_device^
    _ = test_output_device^
    _ = input_row_offsets_device^
    _ = input_row_offsets_host^

    # Pass around the DeviceNDBuffers, these are reference counted using DeviceBuffers.
    # If we return around HostNDBuffers then we'll double free because HostNDBuffers are
    # @value decorated and can't be moved.
    return rebind[__type_of(result)](
        (
            ref_output_device,
            test_output_device,
        )
    )


def execute_paged_fused_qkv_matmul[
    num_q_heads: Int,
    type: DType,
    kv_params: KVCacheStaticParams,
    rtol: Float64,
](
    prompt_lens: List[Int],
    max_seq_length_cache: Int,
    cache_sizes: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
):
    alias num_paged_blocks = 32
    alias page_size = 512
    alias PagedCacheType = PagedKVCache[type, kv_params, page_size]
    var batch_size = len(prompt_lens)
    debug_assert(
        len(prompt_lens) == len(cache_sizes),
        "expected prompt_lens and cache_sizes size to be equal",
    )

    var cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )

    kv_block_host = HostNDBuffer[type, 6](
        IndexList[6](
            num_layers,
            2,
            num_paged_blocks,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )

    var total_length = 0
    var max_full_context_length = 0
    var max_seq_length_batch = 0
    for i in range(batch_size):
        cache_lengths_host.tensor[i] = cache_sizes[i]
        max_full_context_length = max(
            max_full_context_length, Int(cache_sizes[i] + prompt_lens[i])
        )
        max_seq_length_batch = max(max_seq_length_batch, prompt_lens[i])
        total_length += prompt_lens[i]

    cache_lengths_device = cache_lengths_host.copy_to_device(ctx)

    paged_lut_host = HostNDBuffer[DType.uint32, 2](
        IndexList[2](batch_size, ceildiv(max_full_context_length, page_size))
    )
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = cache_sizes[bs] + prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut_host.tensor[bs, block_idx] = randval

    paged_lut_device = paged_lut_host.copy_to_device(ctx)
    kv_block_device = kv_block_host.copy_to_device(ctx)

    k_cache_device = PagedCacheType(
        kv_block_device.tensor,
        cache_lengths_device.tensor,
        paged_lut_device.tensor,
        max_seq_length_batch,
        max_full_context_length,
        layer_idx,
        PagedCacheType.KeyIdx,
    )
    k_cache_host = PagedCacheType(
        kv_block_host.tensor,
        cache_lengths_host.tensor,
        paged_lut_host.tensor,
        max_seq_length_batch,
        max_full_context_length,
        layer_idx,
        PagedCacheType.KeyIdx,
    )

    v_cache_device = PagedCacheType(
        kv_block_device.tensor,
        cache_lengths_device.tensor,
        paged_lut_device.tensor,
        max_seq_length_batch,
        max_full_context_length,
        layer_idx,
        PagedCacheType.ValueIdx,
    )

    v_cache_host = PagedCacheType(
        kv_block_host.tensor,
        cache_lengths_host.tensor,
        paged_lut_host.tensor,
        max_seq_length_batch,
        max_full_context_length,
        layer_idx,
        PagedCacheType.ValueIdx,
    )
    # execute the matmul
    var results = generic_execute_fused_qkv_cache_ragged[
        kv_params, type, num_q_heads
    ](prompt_lens, cache_sizes, k_cache_device, v_cache_device, ctx)

    var ref_output_device = results[0]
    var test_output_device = results[1]

    ctx.enqueue_copy(kv_block_host.tensor.data, kv_block_device.buffer)

    generic_assert_output_equals[num_q_heads=num_q_heads, rtol=rtol](
        k_cache_host,
        v_cache_host,
        ref_output_device,
        test_output_device,
        prompt_lens,
        max_seq_length_batch,
        ctx,
    )

    _ = kv_block_device^
    _ = kv_block_host^
    _ = paged_lut_device^
    _ = paged_lut_host^
    _ = cache_lengths_device^
    _ = cache_lengths_host^


def execute_cont_batch_fused_qkv_matmul[
    num_q_heads: Int,
    type: DType,
    kv_params: KVCacheStaticParams,
    rtol: Float64,
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

    # initialize our KVCache
    var is_context_encoding = True
    var batch_size = len(cache_sizes)
    var cache_lengths_host = HostNDBuffer[DType.uint32, 1]((batch_size,))
    var max_seq_length_batch = -1
    var max_context_length = 0

    for i in range(batch_size):
        cache_lengths_host.tensor[i] = cache_sizes[i]
        max_context_length = max(
            max_context_length, Int(cache_sizes[i] + prompt_lens[i])
        )
        if cache_lengths_host.tensor[i] != 0:
            is_context_encoding = False

        if prompt_lens[i] > max_seq_length_batch:
            max_seq_length_batch = prompt_lens[i]

    var cache_lengths_device = cache_lengths_host.copy_to_device(ctx)

    kv_block_host = HostNDBuffer[type, 6](
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
    var lookup_table_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](
            batch_size,
        ),
    )

    # hacky way to select random blocks.
    var block_idx_set = Set[Int]()
    var idx = 0
    while idx < batch_size:
        var randval = Int(random_ui64(0, num_blocks - 1))
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
        max_seq_length_batch,
        max_context_length,
        layer_idx,
        CacheType.KeyIdx,
    )
    var k_cache_host = CacheType(
        kv_block_host.tensor,
        cache_lengths_host.tensor,
        lookup_table_host.tensor,
        max_seq_length_batch,
        max_context_length,
        layer_idx,
        CacheType.KeyIdx,
    )
    var v_cache_device = CacheType(
        kv_block_device.tensor,
        cache_lengths_device.tensor,
        lookup_table_device.tensor,
        max_seq_length_batch,
        max_context_length,
        layer_idx,
        CacheType.ValueIdx,
    )
    var v_cache_host = CacheType(
        kv_block_host.tensor,
        cache_lengths_host.tensor,
        lookup_table_host.tensor,
        max_seq_length_batch,
        max_context_length,
        layer_idx,
        CacheType.ValueIdx,
    )

    # execute the matmul
    var results = generic_execute_fused_qkv_cache_ragged[
        kv_params, type, num_q_heads
    ](prompt_lens, cache_sizes, k_cache_device, v_cache_device, ctx)

    var ref_output_device = results[0]
    var test_output_device = results[1]

    ctx.enqueue_copy(kv_block_host.tensor.data, kv_block_device.buffer)

    generic_assert_output_equals[num_q_heads=num_q_heads, rtol=rtol](
        k_cache_host,
        v_cache_host,
        ref_output_device,
        test_output_device,
        prompt_lens,
        max_seq_length_batch,
        ctx,
    )

    _ = kv_block_device^
    _ = kv_block_host^
    _ = lookup_table_device^
    _ = lookup_table_host^
    _ = cache_lengths_device^
    _ = cache_lengths_host^


# TODO implement fused qkv matmul for paged
def execute_fused_matmul_suite(ctx: DeviceContext):
    alias types_tolerances = ((DType.float32, 1e-4), (DType.bfloat16, 1e-2))

    @parameter
    for type_idx in range(2):
        alias type = types_tolerances[type_idx][0]
        alias rtol = types_tolerances[type_idx][1]

        for bs_ref in List[Int](1, 16):
            bs = bs_ref[]
            ce_cache_sizes = List[Int]()
            ce_seq_lens = List[Int]()
            tg_cache_sizes = List[Int]()
            tg_seq_lens = List[Int]()
            for _ in range(bs):
                tg_seq_lens.append(1)
                # TODO increase sizes here to ensure we cross page boundary.
                tg_cache_sizes.append(Int(random_ui64(512, 700)))
                ce_seq_lens.append(Int(random_ui64(512, 700)))
                ce_cache_sizes.append(0)

            # llama3 context encoding
            execute_cont_batch_fused_qkv_matmul[
                llama_num_q_heads, type, kv_params_llama3, rtol
            ](ce_seq_lens, 1024, ce_cache_sizes, 4, 1, ctx)
            execute_paged_fused_qkv_matmul[
                llama_num_q_heads, type, kv_params_llama3, rtol
            ](ce_seq_lens, 1024, ce_cache_sizes, 4, 1, ctx)
            execute_matmul_kv_cache_ragged[
                llama_num_q_heads, type, kv_params_llama3, rtol
            ](
                ce_seq_lens,
                max_seq_length_cache=1024,
                cache_sizes=ce_cache_sizes,
                num_layers=4,
                layer_idx=1,
                ctx=ctx,
            )

            # llama3 token gen
            execute_cont_batch_fused_qkv_matmul[
                llama_num_q_heads, type, kv_params_llama3, rtol
            ](tg_seq_lens, 1024, tg_cache_sizes, 4, 3, ctx)
            execute_paged_fused_qkv_matmul[
                llama_num_q_heads, type, kv_params_llama3, rtol
            ](tg_seq_lens, 1024, tg_cache_sizes, 4, 3, ctx)
            execute_matmul_kv_cache_ragged[
                llama_num_q_heads, type, kv_params_llama3, rtol
            ](
                tg_seq_lens,
                max_seq_length_cache=1024,
                cache_sizes=tg_cache_sizes,
                num_layers=4,
                layer_idx=3,
                ctx=ctx,
            )


def main():
    seed(42)
    with DeviceContext() as ctx:
        execute_fused_matmul_suite(ctx)
