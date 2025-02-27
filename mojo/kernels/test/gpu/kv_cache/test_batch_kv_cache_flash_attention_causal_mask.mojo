# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1437
# UNSUPPORTED: H100-GPU
# RUN: %mojo-no-debug %s -t | FileCheck %s
# CHECK-NOT: CUDA ERROR

from math import isclose, isqrt

from algorithm import max as tensor_max
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, random
from kv_cache.types import ContiguousKVCache, KVCacheStaticParams
from memory import UnsafePointer
from nn.mha import flash_attention
from nn.mha_mask import CausalMask, NullMask
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal

from utils import IndexList
from utils.index import Index
from utils.numerics import min_or_neg_inf

alias kv_params_replit = KVCacheStaticParams(num_heads=8, head_size=128)
alias replit_num_q_heads = 24

alias kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
alias llama_num_q_heads = 32


def execute_flash_attention[
    num_q_heads: Int, type: DType, kv_params: KVCacheStaticParams
](
    batch_size: Int,
    valid_length: NDBuffer[DType.uint32, 1],
    max_prompt_len: Int,
    max_seq_len: Int,
    cache_valid_length: NDBuffer[DType.uint32, 1],
    ctx: DeviceContext,
):
    alias max_batch_size = 32

    debug_assert(
        batch_size < max_batch_size,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured max_batch_size (",
        max_batch_size,
        ")",
    )

    var max_cache_valid_length = Int(
        tensor_max(
            NDBuffer[DType.uint32, 1](cache_valid_length.data, batch_size)
        )
    )

    # initialize q tensor
    # TODO parameterize to layout
    q_host = HostNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        )
    )

    random(q_host.tensor)

    # TODO: Do I need to zero-pad q_host_tensor beyond the valid_size of the
    #       current batch's valid length? What to initialize with?

    valid_length_device = DeviceNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size),
        ctx=ctx,
    )
    ctx.enqueue_copy(valid_length_device.buffer, valid_length.data)

    q_device = DeviceNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        ),
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
            batch_size,
            num_q_heads,
            max_prompt_len,
            max_prompt_len + max_cache_valid_length,
        )
    )

    # random(mask_host.tensor)
    # Initialize causal mask.
    for b in range(batch_size):
        for h in range(num_q_heads):
            for q_idx in range(max_prompt_len):
                for k_idx in range(max_prompt_len + max_cache_valid_length):
                    mask_host.tensor.store(
                        Index(b, h, q_idx, k_idx),
                        0 if q_idx + cache_valid_length[b]
                        >= k_idx else min_or_neg_inf[type](),
                    )

    mask_device = DeviceNDBuffer[
        type, 4, DimList(Dim(), num_q_heads, Dim(), Dim())
    ](
        IndexList[4](
            batch_size,
            num_q_heads,
            max_prompt_len,
            max_prompt_len + max_cache_valid_length,
        ),
        ctx=ctx,
    )
    ctx.enqueue_copy(mask_device.buffer, mask_host.tensor.data)

    # initialize scale tensor
    scale_host = HostNDBuffer[DType.float32, 1, DimList(1)](IndexList[1](1))

    scale_host.tensor[0] = isqrt(Float32(kv_params.head_size))
    scale_device = DeviceNDBuffer[DType.float32, 1, DimList(1)](
        IndexList[1](1),
        ctx=ctx,
    )
    ctx.enqueue_copy(scale_device.buffer, scale_host.tensor.data)

    # initialize reference output
    ref_output_host = HostNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        ),
    )
    ref_output_device = DeviceNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        ),
        ctx=ctx,
    )

    # initialize test output
    test_output_host = HostNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        ),
    )
    test_output_device = DeviceNDBuffer[
        type, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        ),
        ctx=ctx,
    )

    # initialize our KVCache
    var is_context_encoding = True
    var max_cache_len_in_batch = 0
    var max_seq_len_in_batch = 0
    for i in range(batch_size):
        if cache_valid_length[i] != 0:
            is_context_encoding = False
        max_cache_len_in_batch = max(
            max_cache_len_in_batch, Int(cache_valid_length[i] + valid_length[i])
        )
        max_seq_len_in_batch = max(max_seq_len_in_batch, Int(valid_length[i]))

    var cache_lengths_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size)

    ctx.enqueue_copy(cache_lengths_dev, cache_valid_length.data)
    var cache_lengths = NDBuffer[DType.uint32, 1](
        cache_lengths_dev.unsafe_ptr(), Index(batch_size)
    )

    k_block_host = HostNDBuffer[
        type,
        4,
        ContiguousKVCache[
            type,
            kv_params,
        ]._internal_block_shape,
    ](
        IndexList[4](
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
        IndexList[4](
            batch_size, max_seq_len, kv_params.num_heads, kv_params.head_size
        ),
        ctx=ctx,
    )
    ctx.enqueue_copy(k_block_device.buffer, k_block_host.tensor.data)

    k_cache_device = ContiguousKVCache[type, kv_params](
        k_block_device.tensor,
        cache_lengths,
        is_context_encoding,
        batch_size,
        max_seq_len_in_batch,
        max_cache_len_in_batch,
    )

    v_block_host = HostNDBuffer[
        type,
        4,
        ContiguousKVCache[
            type,
            kv_params,
        ]._internal_block_shape,
    ](
        IndexList[4](
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
        IndexList[4](
            batch_size, max_seq_len, kv_params.num_heads, kv_params.head_size
        ),
        ctx=ctx,
    )
    ctx.enqueue_copy(v_block_device.buffer, v_block_host.tensor.data)

    v_cache_device = ContiguousKVCache[type, kv_params](
        v_block_device.tensor,
        cache_lengths,
        is_context_encoding,
        batch_size,
        max_seq_len_in_batch,
        max_cache_len_in_batch,
    )

    flash_attention[add_attn_mask=False](
        test_output_device.tensor,
        q_device.tensor,
        k_cache_device,
        v_cache_device,
        mask_device.tensor,
        CausalMask(),
        IdentityScoreMod(),
        valid_length_device.tensor,
        # TODO take scale from argument GEX-750
        isqrt(Float32(kv_params.head_size)),
        ctx,
    )

    flash_attention[add_attn_mask=True](
        ref_output_device.tensor,
        q_device.tensor,
        k_cache_device,
        v_cache_device,
        mask_device.tensor,
        NullMask(),
        IdentityScoreMod(),
        valid_length_device.tensor,
        # TODO take scale from argument GEX-750
        isqrt(Float32(kv_params.head_size)),
        ctx,
    )

    ctx.enqueue_copy(test_output_host.tensor.data, test_output_device.buffer)
    ctx.enqueue_copy(ref_output_host.tensor.data, ref_output_device.buffer)
    ctx.synchronize()

    var ref_out = ref_output_host.tensor
    var test_out = test_output_host.tensor
    for bs in range(batch_size):
        for s in range(valid_length[bs]):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    var ref_val = ref_out[bs, Int(s), h, hd]
                    var test_val = test_out[bs, Int(s), h, hd]
                    assert_almost_equal(ref_val, test_val, atol=1e-5, rtol=8e-3)

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
    _ = scale_device^
    _ = scale_host^
    _ = mask_host^
    _ = mask_device^
    _ = cache_lengths_dev^
    _ = valid_length_device^
    _ = valid_length


def execute_flash_attention_suite(ctx: DeviceContext):
    # alias types = (DType.float32, DType.bfloat16)
    alias types = (DType.bfloat16,)
    var bs = 2
    var valid_length_ptr = UnsafePointer[UInt32].alloc(bs)
    var valid_length = NDBuffer[DType.uint32, 1](valid_length_ptr, Index(1))

    var cache_valid_length_ptr = UnsafePointer[UInt32].alloc(bs)
    var cache_valid_length = NDBuffer[DType.uint32, 1](
        cache_valid_length_ptr, Index(1)
    )

    @parameter
    for type_idx in range(len(types)):
        alias type = types[type_idx]
        # Replit context encoding [testing even query valid lengths].
        valid_length[0] = 128
        valid_length[1] = 64
        cache_valid_length[0] = 0
        cache_valid_length[1] = 0
        execute_flash_attention[replit_num_q_heads, type, kv_params_replit](
            bs, valid_length, 128, 1024, cache_valid_length, ctx
        )

        # Replit context encoding [testing odd query valid length].
        valid_length[0] = 128
        valid_length[1] = 65
        cache_valid_length[0] = 0
        cache_valid_length[1] = 0
        execute_flash_attention[replit_num_q_heads, type, kv_params_replit](
            bs, valid_length, 128, 1024, cache_valid_length, ctx
        )

        # Replit token gen [testing even cache valid lengths].
        valid_length[0] = 1
        valid_length[1] = 1
        cache_valid_length[0] = 200
        cache_valid_length[1] = 256
        execute_flash_attention[replit_num_q_heads, type, kv_params_replit](
            bs, valid_length, 1, 1024, cache_valid_length, ctx
        )

        # Replit token gen [testing even cache valid lengths].
        valid_length[0] = 1
        valid_length[1] = 1
        cache_valid_length[0] = 200
        cache_valid_length[1] = 255

        execute_flash_attention[replit_num_q_heads, type, kv_params_replit](
            bs, valid_length, 1, 1024, cache_valid_length, ctx
        )


def main():
    try:
        with DeviceContext() as ctx:
            execute_flash_attention_suite(ctx)

        print("Success!")
    except e:
        print("CUDA ERROR:", String(e))
