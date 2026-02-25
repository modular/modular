# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from nn.kv_cache_ragged import kv_cache_2m_iadd_dispatch

from utils import IndexList


fn _create_kv_collection_from_host[
    dtype: DType,
    num_heads: Int,
    head_dim: Int,
    page_size: Int,
](
    kv_block_paged_host: LayoutTensor[
        dtype, Layout.row_major[6](), MutAnyOrigin
    ],
    cache_lengths_host: LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    paged_lut_host: LayoutTensor[
        DType.uint32, Layout.row_major[2](), ImmutAnyOrigin
    ],
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
        kv_block_paged_host,
        cache_lengths_host,
        paged_lut_host,
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )


fn _verify_kv_cache[
    dtype: DType,
    num_heads: Int,
    head_dim: Int,
    page_size: Int,
    batch_size: Int,
](
    kv_block_paged_host: LayoutTensor[
        dtype, Layout.row_major[6](), MutAnyOrigin
    ],
    cache_lengths_host: LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    paged_lut_host: LayoutTensor[
        DType.uint32, Layout.row_major[2](), ImmutAnyOrigin
    ],
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
                    if k_val != Scalar[dtype](expected_k_val):
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
                    if v_val != Scalar[dtype](expected_v_val):
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
    var input_row_offsets = ManagedLayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE)
    ](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
            IndexList[1](batch_size + 1)
        ),
        ctx,
    )
    var input_row_offsets_host = input_row_offsets.tensor[update=False]()
    var cache_lengths = ManagedLayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE)
    ](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
            IndexList[1](batch_size)
        ),
        ctx,
    )
    var cache_lengths_host = NDBuffer[DType.uint32, 1](
        cache_lengths.tensor[update=False]().ptr, batch_size
    )

    var input_row_offsets_slice = ManagedLayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE)
    ](
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
            IndexList[1](num_active_loras + 1)
        ),
        ctx,
    )
    var input_row_offsets_slice_host = input_row_offsets_slice.tensor[
        update=False
    ]()
    var total_length = 0
    var total_slice_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        input_row_offsets_host[i] = UInt32(total_length)
        cache_lengths_host[i] = UInt32(cache_lens[i])
        max_full_context_length = max(
            max_full_context_length, cache_lens[i] + prompt_lens[i]
        )
        max_prompt_length = max(max_prompt_length, prompt_lens[i])

        if i < num_active_loras:
            input_row_offsets_slice_host[i] = UInt32(total_length)
            total_slice_length += prompt_lens[i]

        total_length += prompt_lens[i]

    input_row_offsets_host[batch_size] = UInt32(total_length)
    input_row_offsets_slice_host[num_active_loras] = UInt32(total_slice_length)

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
    var kv_block_paged = ManagedLayoutTensor[dtype, Layout.row_major[6]()](
        RuntimeLayout[Layout.row_major[6]()].row_major(kv_block_paged_shape),
        ctx,
    )
    var kv_block_paged_host = NDBuffer[dtype, 6](
        kv_block_paged.tensor[update=False]().ptr, kv_block_paged_shape
    )
    kv_block_paged_host.fill(1)
    var paged_lut_shape = IndexList[2](
        batch_size, ceildiv(max_full_context_length, page_size)
    )
    var paged_lut = ManagedLayoutTensor[DType.uint32, Layout.row_major[2]()](
        RuntimeLayout[Layout.row_major[2]()].row_major(paged_lut_shape), ctx
    )
    var paged_lut_host = NDBuffer[DType.uint32, 2](
        paged_lut.tensor[update=False]().ptr, paged_lut_shape
    )
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = cache_lens[bs] + prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))

            paged_lut_set.add(randval)
            paged_lut_host[bs, block_idx] = UInt32(randval)

    var kv_collection_device = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(
            num_heads=UInt(num_heads), head_size=UInt(head_dim)
        ),
        page_size,
    ](
        kv_block_paged.device_tensor(),
        LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
            cache_lengths.device_tensor().ptr,
            cache_lengths.device_tensor().runtime_layout,
        ),
        paged_lut.device_tensor(),
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    var a_shape = IndexList[2](2 * total_slice_length, num_heads * head_dim)
    var a = ManagedLayoutTensor[
        dtype, Layout.row_major(UNKNOWN_VALUE, num_heads * head_dim)
    ](
        RuntimeLayout[
            Layout.row_major(UNKNOWN_VALUE, num_heads * head_dim)
        ].row_major(a_shape),
        ctx,
    )
    var a_host = NDBuffer[dtype, 2, _, DimList(Dim(), num_heads * head_dim)](
        a.tensor[update=False]().ptr, a_shape
    )
    for i in range(a_host.num_elements()):
        a_host.data[i] = Scalar[dtype](i)

    var layer_idx = 1
    kv_cache_2m_iadd_dispatch[target="gpu"](
        a.device_tensor(),
        kv_collection_device,
        LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
            input_row_offsets_slice.device_tensor().ptr,
            input_row_offsets_slice.device_tensor().runtime_layout,
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
        UInt32(layer_idx),
        Optional(ctx),
    )
    kv_block_paged_host = NDBuffer[dtype, 6](
        kv_block_paged.tensor().ptr, kv_block_paged_shape
    )
    var cache_lengths_host_tensor = LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
    ](
        cache_lengths.tensor().ptr,
        cache_lengths.tensor().runtime_layout,
    )
    var paged_lut_host_tensor = LayoutTensor[
        DType.uint32, Layout.row_major[2](), ImmutAnyOrigin
    ](
        paged_lut.tensor().ptr,
        paged_lut.tensor().runtime_layout,
    )
    var kv_block_paged_host_tensor = kv_block_paged.tensor()

    _verify_kv_cache[dtype, num_heads, head_dim, page_size, batch_size](
        kv_block_paged_host_tensor,
        cache_lengths_host_tensor,
        paged_lut_host_tensor,
        prompt_lens,
        cache_lens,
        num_active_loras,
        total_slice_length,
        max_prompt_length,
        max_full_context_length,
        layer_idx,
    )

    lora_end_idx_host_ptr.free()
    batch_seq_len_host_ptr.free()


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
        input_row_offsets_host[i] = UInt32(total_length)
        cache_lengths_host[i] = UInt32(cache_lens[i])
        max_full_context_length = max(
            max_full_context_length, cache_lens[i] + prompt_lens[i]
        )
        max_prompt_length = max(max_prompt_length, prompt_lens[i])

        if i < num_active_loras:
            input_row_offsets_slice_host[i] = UInt32(total_length)
            total_slice_length += prompt_lens[i]

        total_length += prompt_lens[i]

    input_row_offsets_host[batch_size] = UInt32(total_length)
    input_row_offsets_slice_host[num_active_loras] = UInt32(total_slice_length)

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
    kv_block_paged_host.fill(1)
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
            var randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))

            paged_lut_set.add(randval)
            paged_lut_host[bs, block_idx] = UInt32(randval)

    var kv_collection_host = _create_kv_collection_from_host[
        dtype, num_heads, head_dim, page_size
    ](
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            kv_block_paged_host.data,
            RuntimeLayout[Layout.row_major[6]()].row_major(
                kv_block_paged_shape
            ),
        ),
        LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
            cache_lengths_host.data,
            RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
                IndexList[1](batch_size)
            ),
        ),
        LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
            paged_lut_host.data,
            RuntimeLayout[Layout.row_major[2]()].row_major(paged_lut_shape),
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
    for i in range(a_host.num_elements()):
        a_host.data[i] = Scalar[dtype](i)

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
        UInt32(layer_idx),
        Optional[DeviceContext](),
    )

    _verify_kv_cache[dtype, num_heads, head_dim, page_size, batch_size](
        LayoutTensor[dtype, Layout.row_major[6](), MutAnyOrigin](
            kv_block_paged_host.data,
            RuntimeLayout[Layout.row_major[6]()].row_major(
                kv_block_paged_shape
            ),
        ),
        LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
            cache_lengths_host.data,
            RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
                IndexList[1](batch_size)
            ),
        ),
        LayoutTensor[DType.uint32, Layout.row_major[2](), ImmutAnyOrigin](
            paged_lut_host.data,
            RuntimeLayout[Layout.row_major[2]()].row_major(paged_lut_shape),
        ),
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
