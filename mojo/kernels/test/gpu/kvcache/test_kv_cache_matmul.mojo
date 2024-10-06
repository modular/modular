# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-no-debug %s -t | FileCheck %s
# CHECK-NOT: CUDA ERROR

from utils import Index
from gpu.host import DeviceContext
from runtime.asyncrt import (
    MojoCallContextPtr,
)
from buffer import NDBuffer, Dim, DimList
from kv_cache.types import KVCacheLayout, ContiguousKVCache, KVCacheStaticParams
from nn.kv_cache import (
    _matmul_kv_cache_impl,
    _fused_qkv_matmul_kv_cache_impl,
)
from linalg.matmul_gpu import _matmul_gpu
from utils import IndexList
from internal_utils import (
    HostNDBuffer,
    DeviceNDBuffer,
    fill,
    zero,
    linspace,
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


def execute_matmul[
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
    alias max_batch_size = 32

    debug_assert(
        batch_size < max_batch_size,
        "batch_size passed to unit test ("
        + str(batch_size)
        + ") is larger than configured max_batch_size ("
        + str(max_batch_size)
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
    var valid_lengths_host_ptr = UnsafePointer[UInt32].alloc(max_batch_size)
    for i in range(max_batch_size):
        valid_lengths_host_ptr[i] = -1
    for i in range(batch_size):
        if valid_lengths_host_ptr[i] != 0:
            is_context_encoding = False
        valid_lengths_host_ptr[i] = cache_size
    var valid_lengths_dev = ctx.create_buffer[DType.uint32](max_batch_size)
    ctx.enqueue_copy_to_device(valid_lengths_dev, valid_lengths_host_ptr)
    var valid_lengths_host = NDBuffer[DType.uint32, 1](
        valid_lengths_host_ptr, Index(batch_size)
    )
    var valid_lengths = NDBuffer[DType.uint32, 1](
        valid_lengths_dev.ptr, Index(batch_size)
    )

    kv_block_host = HostNDBuffer[
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
    kv_block_device = DeviceNDBuffer[
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
    var cache_device = ContiguousKVCache[type, kv_params,](
        kv_block_device.tensor,
        valid_lengths,
        is_context_encoding,
        batch_size,
    )
    var cache_host = ContiguousKVCache[type, kv_params,](
        kv_block_host.tensor,
        valid_lengths_host,
        is_context_encoding,
        batch_size,
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
                            bs, h, cache_size + s, hd
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
    _ = valid_lengths_dev^
    valid_lengths_host_ptr.free()


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

    debug_assert(
        batch_size < max_batch_size,
        "batch_size passed to unit test ("
        + str(batch_size)
        + ") is larger than configured max_batch_size ("
        + str(max_batch_size)
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
    var valid_lengths_host_ptr = UnsafePointer[UInt32].alloc(max_batch_size)
    for i in range(max_batch_size):
        valid_lengths_host_ptr[i] = -1
    for i in range(batch_size):
        if valid_lengths_host_ptr[i] != 0:
            is_context_encoding = False
        valid_lengths_host_ptr[i] = cache_size
    var valid_lengths_dev = ctx.create_buffer[DType.uint32](max_batch_size)
    ctx.enqueue_copy_to_device(valid_lengths_dev, valid_lengths_host_ptr)
    var valid_lengths_host = NDBuffer[DType.uint32, 1](
        valid_lengths_host_ptr, Index(batch_size)
    )
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
        IndexList[4](
            batch_size, max_seq_len, kv_params.num_heads, kv_params.head_size
        ),
    )
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
    var k_cache_device = ContiguousKVCache[type, kv_params,](
        k_block_device.tensor,
        valid_lengths,
        is_context_encoding,
        batch_size,
    )
    var k_cache_host = ContiguousKVCache[type, kv_params,](
        k_block_host.tensor,
        valid_lengths_host,
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
        IndexList[4](
            batch_size, max_seq_len, kv_params.num_heads, kv_params.head_size
        ),
    )
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
    v_cache_device = ContiguousKVCache[type, kv_params,](
        v_block_device.tensor,
        valid_lengths,
        is_context_encoding,
        batch_size,
    )
    var v_cache_host = ContiguousKVCache[type, kv_params,](
        v_block_host.tensor,
        valid_lengths_host,
        is_context_encoding,
        batch_size,
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
                    v_cache_host.load[type, width=1](
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


def execute_matmul_suite(ctx: DeviceContext):
    alias types = Tuple[DType, DType](DType.float32, DType.bfloat16)

    @parameter
    for type_idx in range(2):
        alias type = types.get[type_idx, DType]()
        for bs_ref in List[Int](1, 16):
            bs = bs_ref[]

            # Replit context encoding
            execute_matmul[replit_num_q_heads, type, kv_params_replit](
                bs, 128, 1024, 0, ctx
            )

            execute_matmul[replit_num_q_heads, type, kv_params_replit](
                bs, 512, 1024, 0, ctx
            )

            # Replit token gen
            execute_matmul[replit_num_q_heads, type, kv_params_replit](
                bs, 1, 1024, 10, ctx
            )

            execute_matmul[replit_num_q_heads, type, kv_params_replit](
                bs, 1, 1024, 128, ctx
            )

            # llama3 context encoding
            execute_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 128, 1024, 0, ctx
            )

            execute_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 512, 1024, 0, ctx
            )

            # llama3 token gen
            execute_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 1, 1024, 10, ctx
            )

            execute_matmul[llama_num_q_heads, type, kv_params_llama3](
                bs, 1, 1024, 128, ctx
            )


def execute_fused_matmul_suite(ctx: DeviceContext):
    alias types = Tuple[DType, DType](DType.float32, DType.bfloat16)

    @parameter
    for type_idx in range(2):
        alias type = types.get[type_idx, DType]()
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
    try:
        with DeviceContext() as ctx:
            execute_matmul_suite(ctx)
            execute_fused_matmul_suite(ctx)

        print("Success!")
    except e:
        print("CUDA ERROR:", str(e))
