# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-no-debug %s -t | FileCheck %s
# CHECK-NOT: CUDA ERROR

from gpu.host import DeviceContext
from runtime.asyncrt import (
    MojoCallContextPtr,
)
from buffer import NDBuffer, Dim, DimList
from kv_cache.types import KVCacheLayout, ContiguousKVCache, KVCacheStaticParams
from nn.kv_cache import (
    _flash_attention_kv_cache_impl,
)
from math import isqrt
from nn.mha import mha_gpu_naive

from utils import IndexList, Index
from internal_utils import (
    HostNDBuffer,
    DeviceNDBuffer,
    random,
)
from testing import assert_almost_equal
from memory import UnsafePointer

alias kv_params_replit = KVCacheStaticParams(
    num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
)
alias replit_num_q_heads = 24

alias kv_params_llama3 = KVCacheStaticParams(
    num_heads=8, head_size=128, layout=KVCacheLayout.BSHD
)
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

    debug_assert(
        batch_size < max_batch_size,
        "batch_size passed to unit test ("
        + str(batch_size)
        + ") is larger than configured max_batch_size ("
        + str(max_batch_size)
        + ")",
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
    ctx.enqueue_copy_to_device(q_device.buffer, q_host.tensor.data)

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
    ctx.enqueue_copy_to_device(mask_device.buffer, mask_host.tensor.data)

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
    var valid_lengths_dev = ctx.create_buffer[DType.uint32](max_batch_size)
    ctx.enqueue_copy_to_device(valid_lengths_dev, valid_lengths_host_ptr)
    var valid_lengths = NDBuffer[DType.uint32, 1](
        valid_lengths_dev.ptr, Index(batch_size)
    )

    k_block_host = HostNDBuffer[
        type,
        4,
        ContiguousKVCache[
            type,
            kv_params,
        ]._internal_block_shape,
    ](
        Index(
            batch_size, max_seq_len, kv_params.num_heads, kv_params.head_size
        ),
    )
    random(k_block_host.tensor)
    k_block_device = DeviceNDBuffer[
        type,
        4,
        ContiguousKVCache[
            type,
            kv_params,
        ]._internal_block_shape,
    ](
        Index(
            batch_size, max_seq_len, kv_params.num_heads, kv_params.head_size
        ),
        ctx=ctx,
    )
    ctx.enqueue_copy_to_device(k_block_device.buffer, k_block_host.tensor.data)

    k_cache_device = ContiguousKVCache[type, kv_params,](
        k_block_device.tensor,
        valid_lengths,
        is_context_encoding,
        batch_size,
    )

    v_block_host = HostNDBuffer[
        type,
        4,
        ContiguousKVCache[
            type,
            kv_params,
        ]._internal_block_shape,
    ](
        Index(
            batch_size, max_seq_len, kv_params.num_heads, kv_params.head_size
        ),
    )
    random(v_block_host.tensor)
    v_block_device = DeviceNDBuffer[
        type,
        4,
        ContiguousKVCache[
            type,
            kv_params,
        ]._internal_block_shape,
    ](
        Index(
            batch_size, max_seq_len, kv_params.num_heads, kv_params.head_size
        ),
        ctx=ctx,
    )
    ctx.enqueue_copy_to_device(v_block_device.buffer, v_block_host.tensor.data)

    v_cache_device = ContiguousKVCache[type, kv_params,](
        v_block_device.tensor,
        valid_lengths,
        is_context_encoding,
        batch_size,
    )

    valid_lengths_host = HostNDBuffer[DType.uint32, 1](Index(batch_size))
    valid_lengths_device = DeviceNDBuffer[DType.uint32, 1](
        Index(batch_size),
        ctx=ctx,
    )
    for i in range(batch_size):
        valid_lengths_host.tensor[i] = prompt_len

    ctx.enqueue_copy_to_device(
        valid_lengths_device.buffer, valid_lengths_host.tensor.data
    )

    scale = isqrt(Float32(kv_params.head_size))

    _flash_attention_kv_cache_impl[target="cuda"](
        q_device.tensor,
        k_cache_device,
        v_cache_device,
        mask_device.tensor,
        valid_lengths_device.tensor,
        scale,
        test_output_device.tensor,
        ctx,
    )

    mha_gpu_naive[4](
        q_device.buffer.ptr,
        k_block_device.buffer.ptr,
        v_block_device.buffer.ptr,
        mask_device.buffer.ptr,
        ref_output_device.buffer.ptr,
        scale,
        batch_size,
        prompt_len,
        cache_size + prompt_len,
        num_q_heads,  # TODO fix this for GQA
        kv_params.head_size,
        num_q_heads // kv_params.num_heads,
        ctx,
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
            for h in range(kv_params.num_heads):
                for hd in range(kv_params.head_size):
                    assert_almost_equal(
                        ref_out[bs, s, h, hd],
                        test_out[bs, s, h, hd],
                        atol=1e-5,
                        rtol=8e-3,
                    )

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
    alias types = Tuple[DType, DType](DType.float32, DType.bfloat16)

    @parameter
    for type_idx in range(2):
        alias type = types.get[type_idx, DType]()

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
    try:
        with DeviceContext() as ctx:
            execute_flash_attention_suite(ctx)

        print("Success!")
    except e:
        print("CUDA ERROR:", str(e))
