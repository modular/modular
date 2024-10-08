# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-no-debug %s -t | FileCheck %s
# CHECK-NOT: CUDA ERROR

from collections import Set
from random import random_ui64, seed

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
    KVCacheLayout,
    KVCacheStaticParams,
)
from linalg.matmul_gpu import _matmul_gpu
from memory import UnsafePointer
from nn.kv_cache import _fused_qkv_matmul_kv_cache_impl, _matmul_kv_cache_impl
from runtime.asyncrt import MojoCallContextPtr
from testing import assert_almost_equal

from utils import IndexList

alias kv_params_replit = KVCacheStaticParams(
    num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
)
alias replit_num_q_heads = 24

alias kv_params_llama3 = KVCacheStaticParams(
    num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
)
alias llama_num_q_heads = 32


def execute_matmul[
    num_q_heads: Int, type: DType, kv_params: KVCacheStaticParams
](
    batch_size: Int,
    prompt_len: Int,
    max_seq_len: Int,
    cache_sizes: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
):
    alias hidden_size = num_q_heads * kv_params.head_size
    alias kv_hidden_size = kv_params.num_heads * kv_params.head_size
    alias num_blocks = 32

    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test ("
        + str(batch_size)
        + ") is larger than configured num_blocks ("
        + str(num_blocks)
        + ")",
    )
    debug_assert(
        len(cache_sizes) == batch_size,
        "number of cache_size entries do not equal batch_size",
    )
    # initialize hidden state
    hidden_state_host = HostNDBuffer[
        type, 3, DimList(Dim(), Dim(), hidden_size)
    ](IndexList[3](batch_size, prompt_len, hidden_size))

    random(hidden_state_host.tensor)

    hidden_state_device = DeviceNDBuffer[
        type, 3, DimList(Dim(), Dim(), hidden_size)
    ](IndexList[3](batch_size, prompt_len, hidden_size), ctx=ctx)
    ctx.enqueue_copy_to_device(
        hidden_state_device.buffer, hidden_state_host.tensor.data
    )
    hidden_state_device_2d = NDBuffer[type, 2, DimList(Dim(), hidden_size)](
        hidden_state_device.buffer.ptr,
        IndexList[2](batch_size * prompt_len, hidden_size),
    )

    # initialize the weights
    weight_host = HostNDBuffer[type, 2, DimList(kv_hidden_size, hidden_size)](
        IndexList[2](kv_hidden_size, hidden_size)
    )
    random(weight_host.tensor)

    weight_device = DeviceNDBuffer[
        type, 2, DimList(kv_hidden_size, hidden_size)
    ](IndexList[2](kv_hidden_size, hidden_size), ctx=ctx)
    ctx.enqueue_copy_to_device(weight_device.buffer, weight_host.tensor.data)

    # initialize reference output
    ref_output_host = HostNDBuffer[
        type,
        2,
        DimList(Dim(), kv_params.num_heads * kv_params.head_size),
    ](
        IndexList[2](
            batch_size * prompt_len, kv_params.num_heads * kv_params.head_size
        ),
    )
    ref_output_device = DeviceNDBuffer[
        type,
        2,
        DimList(Dim(), kv_params.num_heads * kv_params.head_size),
    ](
        IndexList[2](
            batch_size * prompt_len, kv_params.num_heads * kv_params.head_size
        ),
        ctx=ctx,
    )

    # initialize our KVCache
    var is_context_encoding = True
    var cache_lengths_host = HostNDBuffer[DType.uint32, 1]((batch_size,))
    for i in range(batch_size):
        cache_lengths_host.tensor[i] = cache_sizes[i]
        if cache_lengths_host.tensor[i] != 0:
            is_context_encoding = False

    var cache_lengths_dev = DeviceNDBuffer[DType.uint32, 1](
        (batch_size,), ctx=ctx
    )
    ctx.enqueue_copy_to_device(
        cache_lengths_dev.buffer, cache_lengths_host.tensor.data
    )

    kv_block_host = HostNDBuffer[type, 6,](
        IndexList[6](
            num_blocks,
            2,
            num_layers,
            max_seq_len,
            kv_params.num_heads,
            kv_params.head_size,
        ),
    )
    kv_block_device = DeviceNDBuffer[type, 6,](
        IndexList[6](
            num_blocks,
            2,
            num_layers,
            max_seq_len,
            kv_params.num_heads,
            kv_params.head_size,
        ),
        ctx=ctx,
    )

    var lookup_table_host = HostNDBuffer[DType.uint32, 1,](
        IndexList[1](
            batch_size,
        ),
    )

    var lookup_table_device = DeviceNDBuffer[DType.uint32, 1,](
        IndexList[1](
            batch_size,
        ),
        ctx=ctx,
    )

    # hacky way to get random block indices
    var block_idx_set = Set[Int]()
    var idx = 0
    while len(block_idx_set) < batch_size:
        var randval = int(random_ui64(0, num_blocks - 1))
        if randval in block_idx_set:
            continue
        block_idx_set.add(randval)
        lookup_table_host.tensor[idx] = UInt32(randval)
        idx += 1

    ctx.enqueue_copy_to_device(
        lookup_table_device.buffer, lookup_table_host.tensor.data
    )

    var cache_device = ContinuousBatchingKVCache[type, kv_params,](
        kv_block_device.tensor,
        cache_lengths_dev.tensor,
        lookup_table_device.tensor,
        is_context_encoding,
        layer_idx,
        1,
    )
    var cache_host = ContinuousBatchingKVCache[type, kv_params,](
        kv_block_host.tensor,
        cache_lengths_host.tensor,
        lookup_table_host.tensor,
        is_context_encoding,
        layer_idx,
        1,
    )

    # execute reference
    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        ref_output_device.tensor,
        hidden_state_device_2d,
        weight_device.tensor,
        ctx,
    )

    # excute test
    _ = _matmul_kv_cache_impl[target="cuda"](
        hidden_state_device.tensor, weight_device.tensor, cache_device, ctx
    )
    ctx.enqueue_copy_from_device(
        kv_block_host.tensor.data, kv_block_device.buffer
    )
    ctx.enqueue_copy_from_device(
        ref_output_host.tensor.data, ref_output_device.buffer
    )
    ctx.synchronize()

    ref_out = ref_output_host.tensor
    for bs in range(batch_size):
        for s in range(prompt_len):
            for h in range(kv_params.num_heads):
                for hd in range(kv_params.head_size):
                    assert_almost_equal(
                        ref_out[
                            bs * prompt_len + s,
                            h * kv_params.head_size + hd,
                        ],
                        cache_host.load[type, width=1](
                            bs, h, cache_sizes[bs] + s, hd
                        ),
                    )

    _ = hidden_state_host^
    _ = hidden_state_device^
    _ = weight_host^
    _ = weight_device^
    _ = ref_output_device^
    _ = ref_output_host^
    _ = kv_block_device^
    _ = kv_block_host^
    _ = lookup_table_host^
    _ = lookup_table_device^
    _ = cache_lengths_dev^
    _ = cache_lengths_host^


def execute_fused_qkv_matmul[
    num_q_heads: Int, type: DType, kv_params: KVCacheStaticParams
](
    batch_size: Int,
    prompt_len: Int,
    max_seq_len: Int,
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
        batch_size < num_blocks,
        "batch_size passed to unit test ("
        + str(batch_size)
        + ") is larger than configured max_batch_size ("
        + str(num_blocks)
        + ")",
    )
    # initialize hidden state
    hidden_state_host = HostNDBuffer[
        type, 3, DimList(Dim(), Dim(), hidden_size)
    ](IndexList[3](batch_size, prompt_len, hidden_size))

    random(hidden_state_host.tensor)

    hidden_state_device = DeviceNDBuffer[
        type, 3, DimList(Dim(), Dim(), hidden_size)
    ](IndexList[3](batch_size, prompt_len, hidden_size), ctx=ctx)
    ctx.enqueue_copy_to_device(
        hidden_state_device.buffer, hidden_state_host.tensor.data
    )
    hidden_state_device_2d = NDBuffer[type, 2, DimList(Dim(), hidden_size)](
        hidden_state_device.buffer.ptr,
        IndexList[2](batch_size * prompt_len, hidden_size),
    )

    # initialize the weights
    weight_host = HostNDBuffer[
        type,
        2,
        DimList(fused_hidden_size, hidden_size),
    ](IndexList[2](fused_hidden_size, hidden_size))
    random(weight_host.tensor)

    weight_device = DeviceNDBuffer[
        type,
        2,
        DimList(fused_hidden_size, hidden_size),
    ](
        IndexList[2](fused_hidden_size, hidden_size),
        ctx=ctx,
    )
    ctx.enqueue_copy_to_device(weight_device.buffer, weight_host.tensor.data)

    # initialize reference output
    ref_output_host = HostNDBuffer[type, 2, DimList(Dim(), fused_hidden_size),](
        IndexList[2](
            batch_size * prompt_len,
            fused_hidden_size,
        ),
    )
    ref_output_device = DeviceNDBuffer[
        type,
        2,
        DimList(Dim(), fused_hidden_size),
    ](
        IndexList[2](
            batch_size * prompt_len,
            fused_hidden_size,
        ),
        ctx=ctx,
    )
    test_output_host = HostNDBuffer[
        type,
        3,
        DimList(Dim(), Dim(), hidden_size),
    ](
        IndexList[3](batch_size, prompt_len, hidden_size),
    )
    test_output_device = DeviceNDBuffer[
        type,
        3,
        DimList(Dim(), Dim(), hidden_size),
    ](
        IndexList[3](batch_size, prompt_len, hidden_size),
        ctx=ctx,
    )

    # initialize our KVCache
    var is_context_encoding = True
    var cache_lengths_host = HostNDBuffer[DType.uint32, 1]((batch_size,))
    for i in range(batch_size):
        cache_lengths_host.tensor[i] = cache_sizes[i]
        if cache_lengths_host.tensor[i] != 0:
            is_context_encoding = False

    var cache_lengths_dev = DeviceNDBuffer[DType.uint32, 1](
        (batch_size,), ctx=ctx
    )
    ctx.enqueue_copy_to_device(
        cache_lengths_dev.buffer, cache_lengths_host.tensor.data
    )

    kv_block_host = HostNDBuffer[type, 6,](
        IndexList[6](
            num_blocks,
            2,
            num_layers,
            max_seq_len,
            kv_params.num_heads,
            kv_params.head_size,
        ),
    )
    kv_block_device = DeviceNDBuffer[type, 6,](
        IndexList[6](
            num_blocks,
            2,
            num_layers,
            max_seq_len,
            kv_params.num_heads,
            kv_params.head_size,
        ),
        ctx=ctx,
    )

    var lookup_table_host = HostNDBuffer[DType.uint32, 1,](
        IndexList[1](
            batch_size,
        ),
    )

    var lookup_table_device = DeviceNDBuffer[DType.uint32, 1,](
        IndexList[1](
            batch_size,
        ),
        ctx=ctx,
    )

    # hacky way to get random block indices
    var block_idx_set = Set[Int]()
    var idx = 0
    while len(block_idx_set) < batch_size:
        var randval = int(random_ui64(0, num_blocks - 1))
        if randval in block_idx_set:
            continue
        block_idx_set.add(randval)
        lookup_table_host.tensor[idx] = UInt32(randval)
        idx += 1

    ctx.enqueue_copy_to_device(
        lookup_table_device.buffer, lookup_table_host.tensor.data
    )

    var k_cache_device = CacheType(
        kv_block_device.tensor,
        cache_lengths_dev.tensor,
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
        cache_lengths_dev.tensor,
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
    _fused_qkv_matmul_kv_cache_impl[target="cuda",](
        hidden_state_device.tensor,
        weight_device.tensor,
        k_cache_device,
        v_cache_device,
        test_output_device.tensor,
        ctx,
    )

    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        ref_output_device.tensor,
        hidden_state_device_2d,
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
        for s in range(prompt_len):
            for q_dim in range(hidden_size):
                assert_almost_equal(
                    ref_out[
                        bs * prompt_len + s,
                        q_dim,
                    ],
                    test_out[bs, s, q_dim],
                )

            for k_dim in range(kv_hidden_size):
                head_idx = k_dim // kv_params.head_size
                head_dim_idx = k_dim % kv_params.head_size
                assert_almost_equal(
                    ref_out[
                        bs * prompt_len + s,
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
                        bs * prompt_len + s,
                        hidden_size + kv_hidden_size + v_dim,
                    ],
                    v_cache_host.load[type, width=1](
                        bs,
                        head_idx,
                        cache_sizes[bs] + s,
                        head_dim_idx,
                    ),
                )

    _ = hidden_state_device^
    _ = hidden_state_host^
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
    _ = cache_lengths_dev^
    _ = cache_lengths_host^


def execute_matmul_suite(ctx: DeviceContext):
    alias types = Tuple[DType, DType](DType.float32, DType.bfloat16)

    @parameter
    for type_idx in range(2):
        alias type = types.get[type_idx, DType]()
        for bs_ref in List[Int](1, 16):
            bs = bs_ref[]
            ce_cache_sizes = List[Int]()
            tg_cache_sizes = List[Int]()
            for _ in range(bs):
                tg_cache_sizes.append(int(random_ui64(0, 100)))
                ce_cache_sizes.append(0)

            # llama3 context encoding
            execute_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 128, 1024, ce_cache_sizes, 4, 1, ctx
            )

            execute_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 512, 1024, ce_cache_sizes, 4, 1, ctx
            )

            # llama3 token gen
            execute_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 1, 1024, tg_cache_sizes, 4, 0, ctx
            )

            execute_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 1, 1024, tg_cache_sizes, 4, 1, ctx
            )
            execute_matmul[replit_num_q_heads, type, kv_params_replit](
                bs, 1, 1024, tg_cache_sizes, 4, 1, ctx
            )


def execute_fused_matmul_suite(ctx: DeviceContext):
    alias types = Tuple[DType, DType](DType.float32, DType.bfloat16)

    @parameter
    for type_idx in range(2):
        alias type = types.get[type_idx, DType]()
        for bs_ref in List[Int](1, 16):
            bs = bs_ref[]
            ce_cache_sizes = List[Int]()
            tg_cache_sizes = List[Int]()
            for _ in range(bs):
                tg_cache_sizes.append(int(random_ui64(0, 100)))
                ce_cache_sizes.append(0)

            # llama3 context encoding
            execute_fused_qkv_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 128, 1024, ce_cache_sizes, 4, 1, ctx
            )

            execute_fused_qkv_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 512, 1024, ce_cache_sizes, 4, 0, ctx
            )

            # llama3 token gen
            execute_fused_qkv_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 1, 1024, tg_cache_sizes, 4, 3, ctx
            )

            execute_fused_qkv_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 1, 1024, tg_cache_sizes, 4, 0, ctx
            )


def main():
    seed(42)
    try:
        with DeviceContext() as ctx:
            execute_matmul_suite(ctx)
            execute_fused_matmul_suite(ctx)

        print("Success!")
    except e:
        print("CUDA ERROR:", str(e))
