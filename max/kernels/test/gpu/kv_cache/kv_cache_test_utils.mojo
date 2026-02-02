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
"""Shared utilities for KV cache tests."""

from collections import Set
from math import ceildiv
from random import random_ui64

from gpu.host import DeviceBuffer, DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE

from utils import Index, IndexList


struct _KVCacheTestTensor[dtype: DType, layout: Layout, rank: Int](Copyable):
    comptime tensor_type = LayoutTensor[Self.dtype, Self.layout, ImmutAnyOrigin]

    var shape: IndexList[Self.rank]
    var host_ptr: UnsafePointer[Scalar[Self.dtype], MutExternalOrigin]
    var device_buf: Optional[DeviceBuffer[Self.dtype]]

    fn __init__(out self, shape: IndexList[Self.rank]):
        self.shape = shape
        self.host_ptr = alloc[Scalar[Self.dtype]](shape.flattened_length())
        self.device_buf = None

    fn __del__(deinit self):
        self.host_ptr.free()

    fn copy_to_device(mut self, ctx: DeviceContext) raises:
        self.device_buf = ctx.enqueue_create_buffer[Self.dtype](
            self.shape.flattened_length()
        )
        ctx.enqueue_copy(self.device_buf.value(), self.host_ptr)

    fn host_tensor(self) -> Self.tensor_type:
        return self._tensor(self.host_ptr)

    fn device_tensor(self) -> Self.tensor_type:
        return self._tensor(self.device_buf.value().unsafe_ptr())

    fn _runtime_layout(self) -> RuntimeLayout[Self.layout]:
        return RuntimeLayout[Self.layout].row_major(self.shape)

    fn _tensor(
        self, ptr: UnsafePointer[Scalar[Self.dtype]]
    ) -> Self.tensor_type:
        return Self.tensor_type(ptr, self._runtime_layout())


struct CacheLengthsTable(Copyable):
    var cache_lengths: _KVCacheTestTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), 1
    ]
    var input_row_offsets: _KVCacheTestTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), 1
    ]

    var batch_size: Int
    var max_full_context_length: Int
    var max_seq_length_batch: Int
    var total_length: Int

    fn __init__(out self, batch_size: Int):
        self.batch_size = batch_size
        self.cache_lengths = type_of(self.cache_lengths)(Index(batch_size))
        self.input_row_offsets = type_of(self.input_row_offsets)(
            Index(batch_size + 1)
        )
        self.max_full_context_length = 0
        self.max_seq_length_batch = 0
        self.total_length = 0

    fn _build(
        mut self,
        prompt_lens: List[Int],
        cache_lens: List[Int],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        var cache_lengths_ptr = self.cache_lengths.host_ptr
        var input_row_offsets_ptr = self.input_row_offsets.host_ptr

        var max_full_context_length = 0
        var max_seq_length_batch = 0
        var total_length = 0

        for batch, (prompt_len, cache_len) in enumerate(
            zip(prompt_lens, cache_lens)
        ):
            cache_lengths_ptr[batch] = cache_len
            input_row_offsets_ptr[batch] = total_length

            max_full_context_length = max(
                max_full_context_length, cache_len + prompt_len
            )
            max_seq_length_batch = max(max_seq_length_batch, prompt_len)
            total_length += prompt_len

        input_row_offsets_ptr[self.batch_size] = total_length

        self.max_full_context_length = max_full_context_length
        self.max_seq_length_batch = max_seq_length_batch
        self.total_length = total_length

        if ctx:
            self.cache_lengths.copy_to_device(ctx.value())
            self.input_row_offsets.copy_to_device(ctx.value())

    @staticmethod
    fn build(
        prompt_lens: List[Int],
        cache_lens: List[Int],
        ctx: Optional[DeviceContext],
    ) raises -> Self:
        var batch_size = len(prompt_lens)
        var cache_lengths_table = Self(batch_size)
        cache_lengths_table._build(prompt_lens, cache_lens, ctx)
        return cache_lengths_table^


struct PagedLookupTable[page_size: Int](Copyable):
    var paged_lut: _KVCacheTestTensor[DType.uint32, Layout.row_major[2](), 2]

    fn __init__(out self, batch_size: Int, max_full_context_length: Int) raises:
        self.paged_lut = type_of(self.paged_lut)(
            Index(batch_size, ceildiv(max_full_context_length, Self.page_size))
        )

    fn _build(
        mut self,
        prompt_lens: List[Int],
        cache_lens: List[Int],
        num_paged_blocks: Int,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        var batch_size = len(prompt_lens)

        var host_tensor = LayoutTensor[
            DType.uint32, type_of(self.paged_lut).layout
        ](
            self.paged_lut.host_ptr,
            self.paged_lut._runtime_layout(),
        )
        var used_set = Set[Int]()

        for batch in range(batch_size):
            var seq_len = prompt_lens[batch] + cache_lens[batch]

            for block_idx in range(0, ceildiv(seq_len, Self.page_size)):
                var randval = Int(random_ui64(0, num_paged_blocks - 1))
                while randval in used_set:
                    randval = Int(random_ui64(0, num_paged_blocks - 1))

                used_set.add(randval)
                host_tensor[batch, block_idx] = randval

        if ctx:
            self.paged_lut.copy_to_device(ctx.value())

    @staticmethod
    fn build(
        prompt_lens: List[Int],
        cache_lens: List[Int],
        max_full_context_length: Int,
        num_paged_blocks: Int,
        ctx: Optional[DeviceContext],
    ) raises -> Self:
        var batch_size = len(prompt_lens)
        var paged_lut = Self(batch_size, max_full_context_length)
        paged_lut._build(prompt_lens, cache_lens, num_paged_blocks, ctx)
        return paged_lut^

    @staticmethod
    fn build[
        batch_size: Int
    ](
        prompt_lens: IndexList[batch_size],
        cache_lens: IndexList[batch_size],
        max_full_context_length: Int,
        num_paged_blocks: Int,
        ctx: DeviceContext,
    ) raises -> Self:
        @parameter
        fn _to_list(idx_list: IndexList) -> List[Int]:
            var list = List[Int](capacity=idx_list.size)
            for i in range(idx_list.size):
                list.append(idx_list[i])
            return list^

        return Self.build(
            _to_list(prompt_lens),
            _to_list(cache_lens),
            max_full_context_length,
            num_paged_blocks,
            ctx,
        )

    fn host_tensor(self) -> type_of(self.paged_lut).tensor_type:
        return self.paged_lut.host_tensor()

    fn device_tensor(self) -> type_of(self.paged_lut).tensor_type:
        return self.paged_lut.device_tensor()
