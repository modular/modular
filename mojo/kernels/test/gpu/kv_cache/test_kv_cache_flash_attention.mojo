# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from math import isqrt

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    random,
    assert_with_measure,
)
from internal_utils._measure import cosine
from kv_cache.types import (
    ContiguousKVCache,
    ContiguousKVCacheCollection,
    KVCacheStaticParams,
)
from memory import UnsafePointer
from nn.kv_cache import _flash_attention_kv_cache_impl
from nn.mha import mha_gpu_naive
from testing import assert_almost_equal

from utils import Index, IndexList

alias kv_params_replit = KVCacheStaticParams(num_heads=8, head_size=128)
alias replit_num_q_heads = 24

alias kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
alias llama_num_q_heads = 32


def execute_flash_attention[
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

    debug_assert(
        batch_size < max_batch_size,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured max_batch_size (",
        max_batch_size,
        ")",
    )
    # initialize q tensor
    # TODO parameterize to layout
    q_host = HostNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[4](batch_size, prompt_len, num_q_heads, kv_params.head_size))

    random(q_host.tensor)

    q_device = DeviceNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](batch_size, prompt_len, num_q_heads, kv_params.head_size),
        ctx=ctx,
    )
    ctx.enqueue_copy(q_device.buffer, q_host.tensor.data)

    # initialize mask tensor
    # TODO this should ideally create a triangular matrix
    # but the output should be consistent regardless.
    mask_host = HostNDBuffer[
        type, 4, DimList(Dim(), num_q_heads, Dim(), Dim())
    ](
        IndexList[4](
            batch_size, num_q_heads, prompt_len, prompt_len + cache_size
        )
    )

    random(mask_host.tensor)

    mask_device = DeviceNDBuffer[
        type, 4, DimList(Dim(), num_q_heads, Dim(), Dim())
    ](
        IndexList[4](
            batch_size, num_q_heads, prompt_len, prompt_len + cache_size
        ),
        ctx=ctx,
    )
    ctx.enqueue_copy(mask_device.buffer, mask_host.tensor.data)

    # initialize reference output
    ref_output_host = HostNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](batch_size, prompt_len, num_q_heads, kv_params.head_size),
    )
    ref_output_device = DeviceNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](batch_size, prompt_len, num_q_heads, kv_params.head_size),
        ctx=ctx,
    )

    # initialize test output
    test_output_host = HostNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](batch_size, prompt_len, num_q_heads, kv_params.head_size),
    )
    test_output_device = DeviceNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](batch_size, prompt_len, num_q_heads, kv_params.head_size),
        ctx=ctx,
    )

    # initialize our KVCache
    var is_context_encoding = True
    var valid_lengths_host_ptr = UnsafePointer[UInt32].alloc(max_batch_size)
    for i in range(max_batch_size):
        valid_lengths_host_ptr[i] = -1
    for i in range(batch_size):
        if valid_lengths_host_ptr[i] != 0:
            is_context_encoding = False
        valid_lengths_host_ptr[i] = cache_size
    var valid_lengths_dev = ctx.enqueue_create_buffer[DType.uint32](
        max_batch_size
    )
    ctx.enqueue_copy(valid_lengths_dev, valid_lengths_host_ptr)
    var valid_lengths = NDBuffer[DType.uint32, 1](
        valid_lengths_dev.unsafe_ptr(), Index(batch_size)
    )
    k_block_host = HostNDBuffer[type, 5](
        Index(
            num_layers,
            batch_size,
            max_seq_len,
            Int(kv_params.num_heads),
            Int(kv_params.head_size),
        ),
    )
    random(k_block_host.tensor)
    k_block_device = k_block_host.copy_to_device(ctx)

    v_block_host = HostNDBuffer[type, 5](
        Index(
            num_layers,
            batch_size,
            max_seq_len,
            Int(kv_params.num_heads),
            Int(kv_params.head_size),
        ),
    )
    random(v_block_host.tensor)
    v_block_device = v_block_host.copy_to_device(ctx)

    kv_cache_device = ContiguousKVCacheCollection[type, kv_params](
        key_cache=k_block_device.tensor,
        value_cache=v_block_device.tensor,
        cache_lengths=valid_lengths,
        is_context_encoding=is_context_encoding,
        num_layers=num_layers,
        batch_size=batch_size,
        max_seq_len_in_batch=prompt_len,
        max_cache_len_in_batch=cache_size + prompt_len,
    )

    valid_lengths_host = HostNDBuffer[DType.uint32, 1](Index(batch_size))
    valid_lengths_device = DeviceNDBuffer[DType.uint32, 1](
        Index(batch_size),
        ctx=ctx,
    )
    for i in range(batch_size):
        valid_lengths_host.tensor[i] = prompt_len

    ctx.enqueue_copy(
        valid_lengths_device.buffer, valid_lengths_host.tensor.data
    )

    scale = isqrt(Float32(kv_params.head_size))

    _flash_attention_kv_cache_impl[kv_cache_device.CacheType, target="gpu"](
        q_device.tensor,
        kv_cache_device,
        UInt32(0),
        mask_device.tensor,
        valid_lengths_device.tensor,
        scale,
        test_output_device.tensor,
        ctx,
    )

    var kv_4d_shape = Index(
        batch_size,
        cache_size + prompt_len,
        Int(kv_params.num_heads),
        Int(kv_params.head_size),
    )
    var k_4dbuffer = NDBuffer[type, 4](
        k_block_device.buffer.unsafe_ptr(), kv_4d_shape
    )
    var v_4dbuffer = NDBuffer[type, 4](
        v_block_device.buffer.unsafe_ptr(), kv_4d_shape
    )

    mha_gpu_naive(
        q_device.tensor,
        k_4dbuffer,
        v_4dbuffer,
        mask_device.tensor,
        ref_output_device.tensor,
        scale,
        batch_size,
        prompt_len,
        cache_size + prompt_len,
        num_q_heads,  # TODO fix this for GQA
        kv_params.head_size,
        num_q_heads // kv_params.num_heads,
        ctx,
    )

    ctx.enqueue_copy(test_output_host.tensor.data, test_output_device.buffer)
    ctx.enqueue_copy(ref_output_host.tensor.data, ref_output_device.buffer)
    ctx.synchronize()

    ref_out = ref_output_host.tensor
    test_out = test_output_host.tensor
    for bs in range(batch_size):
        for s in range(prompt_len):
            for h in range(num_q_heads):
                var ref_view = NDBuffer[type, 1, DimList(kv_params.head_size)](
                    ref_out._offset((bs, s, h, 0))
                )
                var test_view = NDBuffer[type, 1, DimList(kv_params.head_size)](
                    test_out._offset((bs, s, h, 0))
                )
                assert_with_measure[cosine](ref_view, test_view, threshold=1e-5)

    _ = q_device^
    _ = q_host^
    _ = ref_output_device^
    _ = ref_output_host^
    _ = test_output_device^
    _ = test_output_host^
    _ = v_block_device^
    _ = v_block_host^
    _ = k_block_device^
    _ = k_block_host^
    _ = ref_output_device^
    _ = ref_output_host^
    _ = test_output_device^
    _ = test_output_host^
    _ = valid_lengths_dev^
    valid_lengths_host_ptr.free()


def execute_flash_attention_suite(ctx: DeviceContext):
    alias types = (DType.float32, DType.bfloat16)

    @parameter
    for type_idx in range(2):
        alias type = types[type_idx]

        bs = 1
        # Replit context encoding
        execute_flash_attention[replit_num_q_heads, type, kv_params_replit](
            bs, 128, 1024, 0, ctx
        )
        execute_flash_attention[replit_num_q_heads, type, kv_params_replit](
            bs, 512, 1024, 0, ctx
        )

        # Replit token gen
        execute_flash_attention[replit_num_q_heads, type, kv_params_replit](
            bs, 1, 1024, 200, ctx
        )

        # llama3 context encoding
        execute_flash_attention[llama_num_q_heads, type, kv_params_llama3](
            bs, 128, 1024, 0, ctx
        )
        execute_flash_attention[llama_num_q_heads, type, kv_params_llama3](
            bs, 512, 1024, 0, ctx
        )
        execute_flash_attention[llama_num_q_heads, type, kv_params_llama3](
            bs, 1, 1024, 200, ctx
        )


def main():
    with DeviceContext() as ctx:
        execute_flash_attention_suite(ctx)
