# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-no-debug-no-assert %s

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from gpu.host.info import DEFAULT_GPU_ARCH
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    fill,
    arange,
    random,
    zero,
)
from kv_cache.types import ContiguousKVCacheCollection, KVCacheStaticParams
from linalg.matmul_gpu import _matmul_gpu
from memory import UnsafePointer
from nn.kv_cache import _fused_qkv_matmul_kv_cache_impl
from testing import assert_almost_equal

from utils import Index, IndexList

alias kv_params_replit = KVCacheStaticParams(num_heads=8, head_size=128)
alias replit_num_q_heads = 24

alias kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
alias llama_num_q_heads = 32


def execute_fused_qkv_matmul[
    num_q_heads: Int, type: DType, kv_params: KVCacheStaticParams
](
    batch_size: Int,
    prompt_len: Int,
    max_seq_len: Int,
    cache_size: Int,
    ctx: DeviceContext,
):
    alias hidden_size = num_q_heads * kv_params.head_size
    alias kv_hidden_size = kv_params.num_heads * kv_params.head_size
    alias fused_hidden_size = (2 * kv_hidden_size) + hidden_size
    alias max_batch_size = 32
    alias num_layers = 1
    alias layer_idx: Int = 0

    debug_assert(
        batch_size < max_batch_size,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured max_batch_size (",
        max_batch_size,
        ")",
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
        hidden_state_device.buffer.unsafe_ptr(),
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
    ref_output_host = HostNDBuffer[type, 2, DimList(Dim(), fused_hidden_size)](
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
    var valid_lengths_host_ptr = UnsafePointer[UInt32].alloc(max_batch_size)
    var max_seq_len_in_batch = 0
    var max_cache_len_in_batch = 0
    for i in range(max_batch_size):
        valid_lengths_host_ptr[i] = -1
    for i in range(batch_size):
        if valid_lengths_host_ptr[i] != 0:
            is_context_encoding = False
        valid_lengths_host_ptr[i] = cache_size
        max_seq_len_in_batch = max(max_seq_len_in_batch, prompt_len)
        max_cache_len_in_batch = max(max_cache_len_in_batch, cache_size)
    var valid_lengths_dev = ctx.enqueue_create_buffer[DType.uint32](
        max_batch_size
    )
    ctx.enqueue_copy_to_device(valid_lengths_dev, valid_lengths_host_ptr)
    var valid_lengths_host = NDBuffer[DType.uint32, 1](
        valid_lengths_host_ptr, Index(batch_size)
    )
    var valid_lengths = NDBuffer[DType.uint32, 1](
        valid_lengths_dev.unsafe_ptr(), Index(batch_size)
    )

    k_block_host = HostNDBuffer[type, 5](
        IndexList[5](
            num_layers,
            batch_size,
            max_seq_len,
            kv_params.num_heads,
            kv_params.head_size,
        ),
    )
    k_block_device = k_block_host.copy_to_device(ctx)
    v_block_host = HostNDBuffer[type, 5](
        IndexList[5](
            num_layers,
            batch_size,
            max_seq_len,
            kv_params.num_heads,
            kv_params.head_size,
        ),
    )
    v_block_device = v_block_host.copy_to_device(ctx)

    var kv_collection_device = ContiguousKVCacheCollection[type, kv_params](
        k_block_device.tensor,
        v_block_device.tensor,
        valid_lengths,
        is_context_encoding,
        num_layers,
        batch_size,
        max_seq_len_in_batch,
        max_cache_len_in_batch,
    )
    var kv_collection_host = ContiguousKVCacheCollection[type, kv_params](
        k_block_host.tensor,
        v_block_host.tensor,
        valid_lengths_host,
        is_context_encoding,
        num_layers,
        batch_size,
        max_seq_len_in_batch,
        max_cache_len_in_batch,
    )

    _fused_qkv_matmul_kv_cache_impl[target="gpu"](
        hidden_state_device.tensor,
        weight_device.tensor,
        kv_collection_device,
        UInt32(layer_idx),
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
        k_block_host.tensor.data, k_block_device.buffer
    )
    ctx.enqueue_copy_from_device(
        v_block_host.tensor.data, v_block_device.buffer
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
    k_cache_host = kv_collection_host.get_key_cache(layer_idx)
    v_cache_host = kv_collection_host.get_value_cache(layer_idx)
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
                    k_cache_host.load[width=1](
                        bs,
                        head_idx,
                        cache_size + s,
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
                    v_cache_host.load[width=1](
                        bs,
                        head_idx,
                        cache_size + s,
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
    _ = v_block_device^
    _ = v_block_host^
    _ = k_block_device^
    _ = k_block_host^
    _ = valid_lengths_dev^
    valid_lengths_host_ptr.free()


def execute_fused_matmul_suite(ctx: DeviceContext):
    alias types = (DType.float32, DType.bfloat16)

    @parameter
    for type_idx in range(2):
        alias type = types[type_idx]
        for bs_ref in List[Int](1, 16):
            bs = bs_ref[]

            # Replit context encoding
            execute_fused_qkv_matmul[
                replit_num_q_heads, type, kv_params_replit
            ](bs, 128, 1024, 0, ctx)

            execute_fused_qkv_matmul[
                replit_num_q_heads, type, kv_params_replit
            ](bs, 512, 1024, 0, ctx)

            # Replit token gen
            execute_fused_qkv_matmul[
                replit_num_q_heads, type, kv_params_replit
            ](bs, 1, 1024, 10, ctx)

            execute_fused_qkv_matmul[
                replit_num_q_heads, type, kv_params_replit
            ](bs, 1, 1024, 128, ctx)

            # llama3 context encoding
            execute_fused_qkv_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 128, 1024, 0, ctx
            )

            execute_fused_qkv_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 512, 1024, 0, ctx
            )

            # llama3 token gen
            execute_fused_qkv_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 1, 1024, 10, ctx
            )

            execute_fused_qkv_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 1, 1024, 128, ctx
            )


def main():
    with DeviceContext() as ctx:
        execute_fused_matmul_suite(ctx)
