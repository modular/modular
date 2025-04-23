# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# RUN: %mojo-no-debug %s

from buffer import DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, assert_almost_equal
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from memory import UnsafePointer, memcpy
from nn.fused_qk_rope import fused_qk_rope
from testdata.fused_qk_rope_goldens import (
    freqs_cis_table_input,
    k_cache_input,
    k_out_golden,
    q_input,
    q_out_golden,
)

from utils import IndexList


def _init_device_ndbuffer_from_goldens[
    type: DType, //, shape: DimList
](goldens: List[Scalar[type]], ctx: DeviceContext) -> DeviceNDBuffer[
    type, len(shape), shape=shape
]:
    """Initializes a device buffer with a set of golden values."""
    host_tensor = HostNDBuffer[type, len(shape), shape=shape]()
    memcpy(dest=host_tensor.tensor.data, src=goldens.data, count=len(goldens))

    # Copy tensor to device.
    device_tensor = DeviceNDBuffer[
        host_tensor.type, host_tensor.rank, shape = host_tensor.shape
    ](ctx=ctx)
    ctx.enqueue_copy(device_tensor.buffer, host_tensor.tensor.data)
    ctx.synchronize()

    # Ensure the host buffer outlives the copy.
    _ = host_tensor^

    return device_tensor


def _fused_qk_rope[
    type: DType, q_shape: DimList, freqs_shape: DimList, //
](
    q_proj: DeviceNDBuffer[type, shape=q_shape],
    kv_collection: ContinuousBatchingKVCacheCollection,
    freqs_cis: DeviceNDBuffer[type, shape=freqs_shape],
    layer_idx: UInt32,
    output: DeviceNDBuffer[type, shape=q_shape],
    context: DeviceContext,
) -> None:
    """Wrapper that takes DeviceNDBuffer, to ensure lifetimes of data."""
    fused_qk_rope[kv_collection.CacheType, interleaved=True, target="gpu"](
        q_proj=rebind[NDBuffer[type, 4, q_proj.tensor.origin, shape=q_shape]](
            q_proj.tensor
        ),
        kv_collection=kv_collection,
        freqs_cis=rebind[
            NDBuffer[type, 2, freqs_cis.tensor.origin, shape=freqs_shape]
        ](freqs_cis.tensor),
        layer_idx=layer_idx,
        output=rebind[NDBuffer[type, 4, output.tensor.origin, shape=q_shape]](
            output.tensor
        ),
        context=context,
    )

    # Synchronize here so that device buffers outlive the execution.
    context.synchronize()


def test_fused_qk_rope[type: DType](ctx: DeviceContext) -> None:
    """Verifies fused_qk_rope against golden values computed with PyTorch."""
    constrained[type is DType.float32, "goldens only for float32, currently"]()

    # Set up test hyperparameters.
    alias batch_size = 2
    alias start_positions = List[UInt32](0, 5)
    alias lookup_table = List[UInt32](0, 1)
    alias seq_len = 3
    alias max_seq_len = 16
    alias num_layers = 1

    fn _max[type: DType, items: List[Scalar[type]]]() -> Scalar[type]:
        constrained[len(items) > 0, "empty list in _max"]()
        max_item = items[0]
        for i in range(1, len(items)):
            if items[i] > max_item:
                max_item = items[i]
        return max_item

    constrained[
        max_seq_len
        > (seq_len + Int(_max[DType.uint32, items=start_positions]())),
        "KV cache size smaller than sum of sequence length and start pos",
    ]()
    alias num_heads = 2
    alias dim = 16
    alias head_dim = dim // num_heads

    # Create aliases for KV cache parameters.
    alias kv_params = KVCacheStaticParams(
        num_heads=num_heads, head_size=head_dim
    )
    alias block_shape = DimList(
        batch_size, 2, num_layers, max_seq_len, num_heads, head_dim
    )

    # Construct backing buffer and the KV cache itself.
    kv_cache_block_host = HostNDBuffer[type, shape=block_shape](
        block_shape.into_index_list[6]()
    )

    # Initialize KV cache block buffer with golden values.
    k_cache_input_buffer = k_cache_input[type]()
    for batch_idx in range(batch_size):
        memcpy(
            dest=kv_cache_block_host.tensor._offset(
                IndexList[6](
                    batch_idx, 0, 0, Int(start_positions[batch_idx]), 0, 0
                )
            ),
            src=k_cache_input_buffer.data + (batch_idx * seq_len * dim),
            count=seq_len * dim,
        )

    # Copy KV cache block to device.
    kv_cache_block_dev = kv_cache_block_host.copy_to_device(ctx)

    # Create the actual KV cache type.
    var max_cache_len_in_batch = 0
    for i in range(batch_size):
        max_cache_len_in_batch = max(
            max_cache_len_in_batch, Int(start_positions[i])
        )
    cache_lengths = DeviceNDBuffer[
        DType.uint32, 1, shape = DimList(batch_size)
    ](ctx=ctx)
    ctx.enqueue_copy(cache_lengths.buffer, start_positions.data)

    lookup_table_dev = DeviceNDBuffer[
        DType.uint32, 1, shape = DimList(batch_size)
    ](ctx=ctx)
    ctx.enqueue_copy(lookup_table_dev.buffer, lookup_table.data)

    kv_collection = ContinuousBatchingKVCacheCollection[type, kv_params](
        blocks=kv_cache_block_dev.tensor,
        cache_lengths=rebind[NDBuffer[DType.uint32, 1, MutableAnyOrigin]](
            cache_lengths.tensor
        ),
        lookup_table=rebind[NDBuffer[DType.uint32, 1, MutableAnyOrigin]](
            lookup_table_dev.tensor
        ),
        max_seq_length=seq_len,
        max_cache_length=max_cache_len_in_batch,
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
        len(expected_q_out_buffer) == q_dev.tensor.num_elements(),
        "invalid expected q out init",
    )
    expected_q_out = NDBuffer[type, rank = q_dev.rank, shape = q_dev.shape](
        expected_q_out_buffer.data
    )
    expected_k_out_buffer = k_out_golden[type]()
    debug_assert(
        len(expected_k_out_buffer) == batch_size * seq_len * dim,
        "invalid expected k out init",
    )

    # Create output buffer.
    q_out_dev = DeviceNDBuffer[q_dev.type, q_dev.rank, shape = q_dev.shape](
        ctx=ctx
    )

    _fused_qk_rope(
        q_proj=q_dev,
        kv_collection=kv_collection,
        freqs_cis=freqs_cis_table_dev,
        layer_idx=UInt32(0),
        output=q_out_dev,
        context=ctx,
    )

    # Copy KV cache block from device.
    kv_cache_block_out_host = kv_cache_block_dev.copy_from_device(ctx)

    # Compare output and expected query tensors.
    q_out_host = HostNDBuffer[q_dev.type, q_dev.rank, shape = q_dev.shape]()
    ctx.enqueue_copy(q_out_host.tensor.data, q_out_dev.buffer)
    ctx.synchronize()

    assert_almost_equal(
        q_out_host.tensor.data,
        expected_q_out.data,
        expected_q_out.num_elements(),
    )

    # Compare output and expected key cache buffers.
    for batch_idx in range(batch_size):
        assert_almost_equal(
            kv_cache_block_out_host.tensor._offset(
                IndexList[6](
                    batch_idx, 0, 0, Int(start_positions[batch_idx]), 0, 0
                )
            ),
            expected_k_out_buffer.data + (batch_idx * seq_len * dim),
            # Number of elements in one batch item.
            len(expected_k_out_buffer) // batch_size,
        )

    # Ensure the lifetimes of the KV cache and output, since their data is
    # accessed through NDBuffers, which isn't parametrized on lifetime.
    _ = kv_cache_block_dev^
    _ = q_out_dev^
    _ = cache_lengths^
    _ = lookup_table_dev^


def main() -> None:
    with DeviceContext() as ctx:
        test_fused_qk_rope[DType.float32](ctx)
