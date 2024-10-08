# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from memory import memcpy
from utils import IndexList

from buffer import DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, assert_almost_equal
from nn.fused_qk_rope import fused_qk_rope
from nn.kv_cache import (
    ContiguousKVCache,
    KVCacheStaticParams,
    KVCacheT,
)

from testdata.fused_qk_rope_goldens import (
    k_cache_input,
    q_input,
    freqs_cis_table_input,
    q_out_golden,
    k_out_golden,
)


def _init_device_ndbuffer_from_goldens[
    type: DType, //, shape: DimList
](goldens: List[Scalar[type]], ctx: DeviceContext) -> DeviceNDBuffer[
    type, len(shape), shape=shape
]:
    """Initializes a device buffer with a set of golden values."""
    host_tensor = HostNDBuffer[type, len(shape), shape=shape]()
    memcpy(dest=host_tensor.tensor.data, src=goldens.data, count=goldens.size)

    # Copy tensor to device.
    device_tensor = DeviceNDBuffer[
        host_tensor.type, host_tensor.rank, shape = host_tensor.shape
    ](ctx=ctx)
    ctx.copy_to_device_sync(device_tensor.buffer, host_tensor.tensor.data)

    # Ensure the host buffer outlives the copy.
    _ = host_tensor^

    return device_tensor


def _fused_qk_rope[
    type: DType, q_shape: DimList, freqs_shape: DimList, //
](
    q_proj: DeviceNDBuffer[type, shape=q_shape],
    k_cache: ContiguousKVCache,
    freqs_cis: DeviceNDBuffer[type, shape=freqs_shape],
    output: DeviceNDBuffer[type, shape=q_shape],
    context: DeviceContext,
) -> None:
    """Wrapper that takes DeviceNDBuffer, to ensure lifetimes of data."""
    fused_qk_rope[target="cuda"](
        q_proj=rebind[NDBuffer[type, 4, shape=q_shape]](q_proj.tensor),
        k_cache=k_cache,
        freqs_cis=rebind[NDBuffer[type, 2, shape=freqs_shape]](
            freqs_cis.tensor
        ),
        output=rebind[NDBuffer[type, 4, shape=q_shape]](output.tensor),
        context=context,
    )

    # Synchronize here so that device buffers outlive the execution.
    context.synchronize()


def test_fused_qk_rope[type: DType](ctx: DeviceContext) -> None:
    """Verifies fused_qk_rope against golden values computed with PyTorch."""
    constrained[type == DType.float32, "goldens only for float32, currently"]()

    # Set up test hyperparameters.
    alias batch_size = 2
    alias start_positions = List[UInt32](0, 5)
    alias seq_len = 3
    alias max_seq_len = 16

    fn _max[type: DType, items: List[Scalar[type]]]() -> Scalar[type]:
        constrained[items.size > 0, "empty list in _max"]()
        max_item = items[0]
        for i in range(1, items.size):
            if items[i] > max_item:
                max_item = items[i]
        return max_item

    constrained[
        max_seq_len
        > (seq_len + int(_max[DType.uint32, items=start_positions]())),
        "KV cache size smaller than sum of sequence length and start pos",
    ]()
    alias num_heads = 2
    alias dim = 16
    alias head_dim = dim // num_heads

    # Create aliases for KV cache parameters.
    alias kv_params = KVCacheStaticParams(
        num_heads=num_heads, head_size=head_dim
    )
    alias block_shape = IndexList[4](
        batch_size, max_seq_len, num_heads, head_dim
    )
    alias BlockType = ContiguousKVCache[type, kv_params].BlockType

    # Construct backing buffer and the KV cache itself.
    k_cache_block_host = HostNDBuffer[
        type, shape = DimList(batch_size, max_seq_len, num_heads, head_dim)
    ](block_shape)

    # Initialize KV cache block buffer with golden values.
    k_cache_input_buffer = k_cache_input[type]()
    for batch_idx in range(batch_size):
        memcpy(
            dest=(
                k_cache_block_host.tensor.data
                + (batch_idx * max_seq_len * dim)
                + int(start_positions[batch_idx] * dim)
            ),
            src=k_cache_input_buffer.data + (batch_idx * seq_len * dim),
            count=seq_len * dim,
        )

    # Copy KV cache block to device.
    k_cache_block_dev = DeviceNDBuffer[type, shape = k_cache_block_host.shape](
        block_shape, ctx=ctx
    )
    ctx.enqueue_copy_to_device(
        k_cache_block_dev.buffer, k_cache_block_host.tensor.data
    )

    # Create the actual KV cache type.
    cache_lengths = DeviceNDBuffer[
        DType.uint32, 1, shape = DimList(start_positions.size)
    ](ctx=ctx)
    ctx.copy_to_device_sync(cache_lengths.buffer, start_positions.data)

    k_cache_block = BlockType(k_cache_block_dev.buffer.ptr, block_shape)
    k_cache = ContiguousKVCache[type, kv_params](
        block=k_cache_block,
        cache_lengths=rebind[NDBuffer[DType.uint32, 1]](cache_lengths.tensor),
        is_cache_empty=False,
        batch_size=batch_size,
    )

    # Create and initialize query buffer.
    q_dev = _init_device_ndbuffer_from_goldens[
        shape = DimList(batch_size, seq_len, num_heads, head_dim)
    ](q_input[type](), ctx)

    # Create and init rotary matrix (frequencies as cos(x) + i*sin(x)).
    freqs_cis_table_dev = _init_device_ndbuffer_from_goldens[
        shape = DimList(max_seq_len, head_dim)
    ](freqs_cis_table_input[type](), ctx)

    # Create and initialize golden outputs.
    expected_q_out_buffer = q_out_golden[type]()
    debug_assert(
        expected_q_out_buffer.size == q_dev.tensor.num_elements(),
        "invalid expected q out init",
    )
    expected_q_out = NDBuffer[type, rank = q_dev.rank, shape = q_dev.shape](
        expected_q_out_buffer.data
    )
    expected_k_out_buffer = k_out_golden[type]()
    debug_assert(
        expected_k_out_buffer.size == batch_size * seq_len * dim,
        "invalid expected k out init",
    )

    # Create output buffer.
    q_out_dev = DeviceNDBuffer[q_dev.type, q_dev.rank, shape = q_dev.shape](
        ctx=ctx
    )

    _fused_qk_rope(
        q_proj=q_dev,
        k_cache=k_cache,
        freqs_cis=freqs_cis_table_dev,
        output=q_out_dev,
        context=ctx,
    )

    # Compare output and expected query tensors.
    q_out_host = HostNDBuffer[q_dev.type, q_dev.rank, shape = q_dev.shape]()
    ctx.copy_from_device_sync(q_out_host.tensor.data, q_out_dev.buffer)

    assert_almost_equal(
        q_out_host.tensor.data,
        expected_q_out.data,
        expected_q_out.num_elements(),
    )
    _ = q_out_host^

    # Compare output and expected key cache buffers.
    for batch_idx in range(batch_size):
        k_cache_offset = (
            (batch_idx * max_seq_len * dim)
            # Account for the start_pos (cache_length) for this batch item.
            + int(start_positions[batch_idx] * dim)
        )
        k_cache_host_batch_item = (
            k_cache_block_host.tensor.data + k_cache_offset
        )
        k_cache_dev_batch_item = k_cache_block_dev.buffer.create_sub_buffer[
            type
        ](k_cache_offset, seq_len * dim)
        ctx.copy_from_device_sync(
            k_cache_host_batch_item, k_cache_dev_batch_item
        )

        assert_almost_equal(
            k_cache_host_batch_item,
            expected_k_out_buffer.data + (batch_idx * seq_len * dim),
            # Number of elements in one batch item.
            expected_k_out_buffer.size // batch_size,
        )

    # Ensure the lifetimes of the KV cache and output, since their data is
    # accessed through NDBuffers, which isn't parametrized on lifetime.
    _ = k_cache_block_dev^
    _ = q_out_dev^


def main() -> None:
    with DeviceContext() as ctx:
        test_fused_qk_rope[DType.float32](ctx)
