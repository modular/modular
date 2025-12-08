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

from collections import Set
from math import ceildiv
from random import random_ui64

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import InitializationType
from internal_utils._utils import initialize
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from memory import LegacyUnsafePointer as UnsafePointer
from nn.kv_cache_ragged import kv_cache_2m_iadd_dispatch

from utils import IndexList


fn _create_kv_collection_from_host[
    dtype: DType,
    num_heads: Int,
    head_dim: Int,
    page_size: Int,
](
    kv_block_paged_host: NDBuffer[dtype, 6],
    cache_lengths_host: NDBuffer[DType.uint32, 1],
    paged_lut_host: NDBuffer[DType.uint32, 2],
    max_prompt_length: Int,
    max_full_context_length: Int,
) -> PagedKVCacheCollection[
    dtype,
    KVCacheStaticParams(num_heads=UInt(num_heads), head_size=UInt(head_dim)),
    page_size,
]:
    return PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(
            num_heads=UInt(num_heads), head_size=UInt(head_dim)
        ),
        page_size,
    ](
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            kv_block_paged_host.data,
            RuntimeLayout[Layout.row_major[6]()](
                kv_block_paged_host.dynamic_shape.canonicalize(),
                kv_block_paged_host.dynamic_stride.canonicalize(),
            ),
        ),
        LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
            cache_lengths_host.data,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                cache_lengths_host.dynamic_shape.canonicalize(),
                cache_lengths_host.dynamic_stride.canonicalize(),
            ),
        ),
        LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
            paged_lut_host.data,
            RuntimeLayout[Layout.row_major[2]()](
                paged_lut_host.dynamic_shape.canonicalize(),
                paged_lut_host.dynamic_stride.canonicalize(),
            ),
        ),
        max_prompt_length,
        max_full_context_length,
    )


fn _verify_kv_cache[
    dtype: DType,
    num_heads: Int,
    head_dim: Int,
    page_size: Int,
    batch_size: Int,
](
    kv_block_paged_host: NDBuffer[dtype, 6],
    cache_lengths_host: NDBuffer[DType.uint32, 1],
    paged_lut_host: NDBuffer[DType.uint32, 2],
    prompt_lens: IndexList[batch_size],
    cache_lens: IndexList[batch_size],
    num_active_loras: Int,
    total_slice_length: Int,
    max_prompt_length: Int,
    max_full_context_length: Int,
    layer_idx: Int,
) raises:
    var kv_collection_host = _create_kv_collection_from_host[
        dtype, num_heads, head_dim, page_size
    ](
        kv_block_paged_host,
        cache_lengths_host,
        paged_lut_host,
        max_prompt_length,
        max_full_context_length,
    )

    var k_cache_host = kv_collection_host.get_key_cache(layer_idx)
    var v_cache_host = kv_collection_host.get_value_cache(layer_idx)

    # First check that we didn't augment previous cache entries
    for i in range(batch_size):
        for c in range(cache_lens[i]):
            for h in range(num_heads):
                for d in range(head_dim):
                    var k_val = k_cache_host.load[width=1](i, h, c, d)
                    var v_val = v_cache_host.load[width=1](i, h, c, d)
                    if k_val != 1:
                        raise Error(
                            "Mismatch in output for k, expected 1, got "
                            + String(k_val)
                            + " in k_cache at index "
                            + String(IndexList[4](i, c, h, d))
                        )
                    if v_val != 1:
                        raise Error(
                            "Mismatch in output for v, expected 1, got "
                            + String(v_val)
                            + " in v_cache at index "
                            + String(IndexList[4](i, c, h, d))
                        )

    # Check that non-LoRA entries are not augmented
    for i in range(num_active_loras, batch_size):
        for c in range(prompt_lens[i]):
            var actual_len = c + cache_lens[i]
            for h in range(num_heads):
                for d in range(head_dim):
                    var k_val = k_cache_host.load[width=1](i, h, actual_len, d)
                    var v_val = v_cache_host.load[width=1](i, h, actual_len, d)
                    if k_val != 1:
                        raise Error(
                            "Mismatch in output for k, expected 1, got "
                            + String(k_val)
                            + " in k_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )
                    if v_val != 1:
                        raise Error(
                            "Mismatch in output for v, expected 1, got "
                            + String(v_val)
                            + " in v_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )

    # Check that the LoRA-augmented entries are correct
    var slice_row_offset = 0
    for i in range(num_active_loras):
        for c in range(prompt_lens[i]):
            var actual_len = c + cache_lens[i]
            var k_row_base = slice_row_offset * num_heads * head_dim
            for h in range(num_heads):
                for d in range(head_dim):
                    var k_val = k_cache_host.load[width=1](i, h, actual_len, d)
                    var k_idx = k_row_base + h * head_dim + d
                    var expected_k_val = 1 + k_idx
                    if k_val != expected_k_val:
                        raise Error(
                            "Mismatch in output for k, expected "
                            + String(expected_k_val)
                            + ", got "
                            + String(k_val)
                            + " in k_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )
            # V portion is stored in rows [total_slice_length, 2*total_slice_length)
            # of the input tensor, so the V row base starts at total_slice_length.
            var v_row_base = (
                (total_slice_length + slice_row_offset) * num_heads * head_dim
            )
            for h in range(num_heads):
                for d in range(head_dim):
                    var v_val = v_cache_host.load[width=1](i, h, actual_len, d)
                    var v_idx = v_row_base + h * head_dim + d
                    var expected_v_val = 1 + v_idx
                    if v_val != expected_v_val:
                        raise Error(
                            "Mismatch in output for v, expected "
                            + String(expected_v_val)
                            + ", got "
                            + String(v_val)
                            + " in v_cache at index "
                            + String(IndexList[4](i, h, actual_len, d))
                        )
            slice_row_offset += 1


fn test_kv_cache_2m_iadd_gpu[
    dtype: DType,
    num_heads: Int,
    head_dim: Int,
    page_size: Int,
    batch_size: Int,
](
    prompt_lens: IndexList[batch_size],
    cache_lens: IndexList[batch_size],
    num_active_loras: Int,
    ctx: DeviceContext,
) raises:
    comptime num_layers = 2
    debug_assert(
        num_active_loras <= batch_size,
        "num_active_loras must be less than or equal to batch_size",
    )
    var input_row_offsets_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size + 1
    )
    var input_row_offsets_host = NDBuffer[DType.uint32, 1](
        input_row_offsets_host_ptr, batch_size + 1
    )
    var cache_lengths_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size
    )
    var cache_lengths_host = NDBuffer[DType.uint32, 1](
        cache_lengths_host_ptr, batch_size
    )

    var input_row_offsets_slice_host_ptr = UnsafePointer[
        Scalar[DType.uint32]
    ].alloc(num_active_loras + 1)
    var input_row_offsets_slice_host = NDBuffer[DType.uint32, 1](
        input_row_offsets_slice_host_ptr, num_active_loras + 1
    )
    var total_length = 0
    var total_slice_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        input_row_offsets_host[i] = total_length
        cache_lengths_host[i] = cache_lens[i]
        max_full_context_length = max(
            max_full_context_length, cache_lens[i] + prompt_lens[i]
        )
        max_prompt_length = max(max_prompt_length, prompt_lens[i])

        if i < num_active_loras:
            input_row_offsets_slice_host[i] = total_length
            total_slice_length += prompt_lens[i]

        total_length += prompt_lens[i]

    input_row_offsets_host[batch_size] = total_length
    input_row_offsets_slice_host[num_active_loras] = total_slice_length

    num_paged_blocks = ceildiv(
        batch_size * max_full_context_length * 2, page_size
    )

    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host_ptr)
    var input_row_offsets_slice_device = ctx.enqueue_create_buffer[
        DType.uint32
    ](num_active_loras + 1)
    ctx.enqueue_copy(
        input_row_offsets_slice_device, input_row_offsets_slice_host_ptr
    )

    var lora_end_idx_host_ptr = UnsafePointer[Scalar[DType.int64]].alloc(1)
    var lora_end_idx_host = NDBuffer[DType.int64, 1](lora_end_idx_host_ptr, 1)
    lora_end_idx_host[0] = Int64(total_slice_length)

    var batch_seq_len_host_ptr = UnsafePointer[Scalar[DType.int64]].alloc(1)
    var batch_seq_len_host = NDBuffer[DType.int64, 1](batch_seq_len_host_ptr, 1)
    batch_seq_len_host[0] = Int64(total_length)

    var kv_block_paged_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        num_heads,
        head_dim,
    )
    var kv_block_paged_size = (
        num_paged_blocks * 2 * num_layers * page_size * num_heads * head_dim
    )
    var kv_block_paged_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        kv_block_paged_size
    )
    var kv_block_paged_host = NDBuffer[dtype, 6](
        kv_block_paged_host_ptr, kv_block_paged_shape
    )
    initialize(kv_block_paged_host, InitializationType.one)
    var paged_lut_shape = IndexList[2](
        batch_size, ceildiv(max_full_context_length, page_size)
    )
    var paged_lut_size = batch_size * ceildiv(
        max_full_context_length, page_size
    )
    var paged_lut_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        paged_lut_size
    )
    var paged_lut_host = NDBuffer[DType.uint32, 2](
        paged_lut_host_ptr, paged_lut_shape
    )
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = cache_lens[bs] + prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut_host[bs, block_idx] = randval

    var kv_block_paged_device = ctx.enqueue_create_buffer[dtype](
        kv_block_paged_size
    )
    ctx.enqueue_copy(kv_block_paged_device, kv_block_paged_host_ptr)
    var paged_lut_device = ctx.enqueue_create_buffer[DType.uint32](
        paged_lut_size
    )
    ctx.enqueue_copy(paged_lut_device, paged_lut_host_ptr)

    var kv_block_paged_device_nd = NDBuffer[dtype, 6](
        kv_block_paged_device.unsafe_ptr(), kv_block_paged_shape
    )
    var cache_lengths_device_nd = NDBuffer[DType.uint32, 1](
        cache_lengths_device.unsafe_ptr(), batch_size
    )
    var paged_lut_device_nd = NDBuffer[DType.uint32, 2](
        paged_lut_device.unsafe_ptr(), paged_lut_shape
    )
    var input_row_offsets_slice_device_nd = NDBuffer[DType.uint32, 1](
        input_row_offsets_slice_device.unsafe_ptr(), num_active_loras + 1
    )

    var kv_collection_device = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(
            num_heads=UInt(num_heads), head_size=UInt(head_dim)
        ),
        page_size,
    ](
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            kv_block_paged_device_nd.data,
            RuntimeLayout[Layout.row_major[6]()](
                kv_block_paged_device_nd.dynamic_shape.canonicalize(),
                kv_block_paged_device_nd.dynamic_stride.canonicalize(),
            ),
        ),
        LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
            cache_lengths_device_nd.data,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                cache_lengths_device_nd.dynamic_shape.canonicalize(),
                cache_lengths_device_nd.dynamic_stride.canonicalize(),
            ),
        ),
        LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
            paged_lut_device_nd.data,
            RuntimeLayout[Layout.row_major[2]()](
                paged_lut_device_nd.dynamic_shape.canonicalize(),
                paged_lut_device_nd.dynamic_stride.canonicalize(),
            ),
        ),
        max_prompt_length,
        max_full_context_length,
    )

    var a_shape = IndexList[2](2 * total_slice_length, num_heads * head_dim)
    var a_size = 2 * total_slice_length * num_heads * head_dim
    var a_host_ptr = UnsafePointer[Scalar[dtype]].alloc(a_size)
    var a_host = NDBuffer[dtype, 2, _, DimList(Dim(), num_heads * head_dim)](
        a_host_ptr, a_shape
    )
    initialize(a_host, InitializationType.arange)
    var a_device = ctx.enqueue_create_buffer[dtype](a_size)
    ctx.enqueue_copy(a_device, a_host_ptr)
    var a_device_nd = NDBuffer[
        dtype, 2, _, DimList(Dim(), num_heads * head_dim)
    ](a_device.unsafe_ptr(), a_shape)

    var layer_idx = 1
    kv_cache_2m_iadd_dispatch[target="gpu"](
        LayoutTensor[
            dtype,
            Layout.row_major(UNKNOWN_VALUE, num_heads * head_dim),
            MutAnyOrigin,
        ](
            a_device_nd.data,
            RuntimeLayout[
                Layout.row_major(UNKNOWN_VALUE, num_heads * head_dim)
            ](
                a_device_nd.dynamic_shape.canonicalize(),
                a_device_nd.dynamic_stride.canonicalize(),
            ),
        ),
        kv_collection_device,
        LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
            input_row_offsets_slice_device_nd.data,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                input_row_offsets_slice_device_nd.dynamic_shape.canonicalize(),
                input_row_offsets_slice_device_nd.dynamic_stride.canonicalize(),
            ),
        ),
        LayoutTensor[DType.int64, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
            lora_end_idx_host.data,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                lora_end_idx_host.dynamic_shape.canonicalize(),
                lora_end_idx_host.dynamic_stride.canonicalize(),
            ),
        ),
        LayoutTensor[DType.int64, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
            batch_seq_len_host.data,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                batch_seq_len_host.dynamic_shape.canonicalize(),
                batch_seq_len_host.dynamic_stride.canonicalize(),
            ),
        ),
        layer_idx,
        Optional(ctx),
    )
    ctx.synchronize()
    ctx.enqueue_copy(kv_block_paged_host_ptr, kv_block_paged_device)
    ctx.enqueue_copy(a_host_ptr, a_device)
    ctx.synchronize()

    _verify_kv_cache[dtype, num_heads, head_dim, page_size, batch_size](
        kv_block_paged_host,
        cache_lengths_host,
        paged_lut_host,
        prompt_lens,
        cache_lens,
        num_active_loras,
        total_slice_length,
        max_prompt_length,
        max_full_context_length,
        layer_idx,
    )

    input_row_offsets_host_ptr.free()
    cache_lengths_host_ptr.free()
    input_row_offsets_slice_host_ptr.free()
    lora_end_idx_host_ptr.free()
    batch_seq_len_host_ptr.free()
    kv_block_paged_host_ptr.free()
    paged_lut_host_ptr.free()
    a_host_ptr.free()
    _ = cache_lengths_device^
    _ = input_row_offsets_slice_device^
    _ = kv_block_paged_device^
    _ = paged_lut_device^
    _ = a_device^


fn test_kv_cache_2m_iadd_cpu[
    dtype: DType,
    num_heads: Int,
    head_dim: Int,
    page_size: Int,
    batch_size: Int,
](
    prompt_lens: IndexList[batch_size],
    cache_lens: IndexList[batch_size],
    num_active_loras: Int,
) raises:
    comptime num_layers = 2
    debug_assert(
        num_active_loras <= batch_size,
        "num_active_loras must be less than or equal to batch_size",
    )
    var input_row_offsets_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size + 1
    )
    var input_row_offsets_host = NDBuffer[DType.uint32, 1](
        input_row_offsets_host_ptr, batch_size + 1
    )
    var cache_lengths_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        batch_size
    )
    var cache_lengths_host = NDBuffer[DType.uint32, 1](
        cache_lengths_host_ptr, batch_size
    )

    var input_row_offsets_slice_host_ptr = UnsafePointer[
        Scalar[DType.uint32]
    ].alloc(num_active_loras + 1)
    var input_row_offsets_slice_host = NDBuffer[DType.uint32, 1](
        input_row_offsets_slice_host_ptr, num_active_loras + 1
    )
    var total_length = 0
    var total_slice_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        input_row_offsets_host[i] = total_length
        cache_lengths_host[i] = cache_lens[i]
        max_full_context_length = max(
            max_full_context_length, cache_lens[i] + prompt_lens[i]
        )
        max_prompt_length = max(max_prompt_length, prompt_lens[i])

        if i < num_active_loras:
            input_row_offsets_slice_host[i] = total_length
            total_slice_length += prompt_lens[i]

        total_length += prompt_lens[i]

    input_row_offsets_host[batch_size] = total_length
    input_row_offsets_slice_host[num_active_loras] = total_slice_length

    num_paged_blocks = ceildiv(
        batch_size * max_full_context_length * 2, page_size
    )

    var lora_end_idx_host_ptr = UnsafePointer[Scalar[DType.int64]].alloc(1)
    var lora_end_idx_host = NDBuffer[DType.int64, 1](lora_end_idx_host_ptr, 1)
    lora_end_idx_host[0] = Int64(total_slice_length)

    var batch_seq_len_host_ptr = UnsafePointer[Scalar[DType.int64]].alloc(1)
    var batch_seq_len_host = NDBuffer[DType.int64, 1](batch_seq_len_host_ptr, 1)
    batch_seq_len_host[0] = Int64(total_length)

    var kv_block_paged_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        num_heads,
        head_dim,
    )
    var kv_block_paged_size = (
        num_paged_blocks * 2 * num_layers * page_size * num_heads * head_dim
    )
    var kv_block_paged_host_ptr = UnsafePointer[Scalar[dtype]].alloc(
        kv_block_paged_size
    )
    var kv_block_paged_host = NDBuffer[dtype, 6](
        kv_block_paged_host_ptr, kv_block_paged_shape
    )
    initialize(kv_block_paged_host, InitializationType.one)
    var paged_lut_shape = IndexList[2](
        batch_size, ceildiv(max_full_context_length, page_size)
    )
    var paged_lut_size = batch_size * ceildiv(
        max_full_context_length, page_size
    )
    var paged_lut_host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(
        paged_lut_size
    )
    var paged_lut_host = NDBuffer[DType.uint32, 2](
        paged_lut_host_ptr, paged_lut_shape
    )
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = cache_lens[bs] + prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut_host[bs, block_idx] = randval

    var kv_collection_host = _create_kv_collection_from_host[
        dtype, num_heads, head_dim, page_size
    ](
        kv_block_paged_host,
        cache_lengths_host,
        paged_lut_host,
        max_prompt_length,
        max_full_context_length,
    )

    var a_shape = IndexList[2](2 * total_slice_length, num_heads * head_dim)
    var a_size = 2 * total_slice_length * num_heads * head_dim
    var a_host_ptr = UnsafePointer[Scalar[dtype]].alloc(a_size)
    var a_host = NDBuffer[dtype, 2, _, DimList(Dim(), num_heads * head_dim)](
        a_host_ptr, a_shape
    )
    initialize(a_host, InitializationType.arange)

    var layer_idx = 1
    kv_cache_2m_iadd_dispatch[target="cpu"](
        LayoutTensor[
            dtype,
            Layout.row_major(UNKNOWN_VALUE, num_heads * head_dim),
            MutAnyOrigin,
        ](
            a_host.data,
            RuntimeLayout[
                Layout.row_major(UNKNOWN_VALUE, num_heads * head_dim)
            ](
                a_host.dynamic_shape.canonicalize(),
                a_host.dynamic_stride.canonicalize(),
            ),
        ),
        kv_collection_host,
        LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
            input_row_offsets_slice_host.data,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                input_row_offsets_slice_host.dynamic_shape.canonicalize(),
                input_row_offsets_slice_host.dynamic_stride.canonicalize(),
            ),
        ),
        LayoutTensor[DType.int64, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
            lora_end_idx_host.data,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                lora_end_idx_host.dynamic_shape.canonicalize(),
                lora_end_idx_host.dynamic_stride.canonicalize(),
            ),
        ),
        LayoutTensor[DType.int64, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
            batch_seq_len_host.data,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                batch_seq_len_host.dynamic_shape.canonicalize(),
                batch_seq_len_host.dynamic_stride.canonicalize(),
            ),
        ),
        layer_idx,
        Optional[DeviceContext](),
    )

    _verify_kv_cache[dtype, num_heads, head_dim, page_size, batch_size](
        kv_block_paged_host,
        cache_lengths_host,
        paged_lut_host,
        prompt_lens,
        cache_lens,
        num_active_loras,
        total_slice_length,
        max_prompt_length,
        max_full_context_length,
        layer_idx,
    )

    # Cleanup
    input_row_offsets_host_ptr.free()
    cache_lengths_host_ptr.free()
    input_row_offsets_slice_host_ptr.free()
    lora_end_idx_host_ptr.free()
    batch_seq_len_host_ptr.free()
    kv_block_paged_host_ptr.free()
    paged_lut_host_ptr.free()
    a_host_ptr.free()


def main():
    # CPU tests
    test_kv_cache_2m_iadd_cpu[DType.float32, 8, 128, 128, 4](
        IndexList[4](10, 20, 30, 40),
        IndexList[4](40, 30, 20, 10),
        2,
    )
    test_kv_cache_2m_iadd_cpu[DType.float32, 8, 128, 128, 4](
        IndexList[4](10, 20, 30, 40),
        IndexList[4](40, 30, 20, 10),
        4,
    )
    test_kv_cache_2m_iadd_cpu[DType.float32, 8, 128, 128, 4](
        IndexList[4](10, 20, 30, 40),
        IndexList[4](40, 30, 20, 10),
        0,
    )
    test_kv_cache_2m_iadd_cpu[DType.float32, 8, 128, 128, 1](
        IndexList[1](10),
        IndexList[1](40),
        1,
    )

    # GPU tests
    with DeviceContext() as ctx:
        test_kv_cache_2m_iadd_gpu[DType.float32, 8, 128, 128, 4](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            2,
            ctx,
        )
        test_kv_cache_2m_iadd_gpu[DType.float32, 8, 128, 128, 4](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            4,
            ctx,
        )
        test_kv_cache_2m_iadd_gpu[DType.float32, 8, 128, 128, 4](
            IndexList[4](10, 20, 30, 40),
            IndexList[4](40, 30, 20, 10),
            0,
            ctx,
        )
        test_kv_cache_2m_iadd_gpu[DType.float32, 8, 128, 128, 1](
            IndexList[1](10),
            IndexList[1](40),
            1,
            ctx,
        )
