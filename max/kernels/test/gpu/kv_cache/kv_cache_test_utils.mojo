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
"""Shared utilities for KV cache tests."""

from collections import Set
from math import ceildiv
from random import random_ui64

from gpu.host import DeviceBuffer, DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE

from utils import Index, IndexList


struct PagedLookupTable[page_size: Int](Copyable):
    comptime layout = Layout.row_major[2]()
    comptime tensor_type = LayoutTensor[
        DType.uint32,
        Self.layout,
        ImmutAnyOrigin,
    ]

    var shape: IndexList[2]
    var host_ptr: UnsafePointer[Scalar[DType.uint32], MutExternalOrigin]
    var device_buf: Optional[DeviceBuffer[DType.uint32]]

    fn __init__(out self, batch_size: Int, max_full_context_length: Int) raises:
        self.shape = Index(
            batch_size, ceildiv(max_full_context_length, Self.page_size)
        )
        self.host_ptr = alloc[Scalar[DType.uint32]](
            self.shape.flattened_length()
        )
        self.device_buf = None

    fn _build[
        seq_len_fn: fn(batch: Int) capturing -> Int
    ](
        mut self,
        batch_size: Int,
        num_paged_blocks: Int,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        var host_tensor = LayoutTensor[DType.uint32, Self.layout](
            self.host_ptr,
            self._runtime_layout(),
        )
        var used_set = Set[Int]()

        for batch in range(batch_size):
            var seq_len = seq_len_fn(batch)

            for block_idx in range(0, ceildiv(seq_len, Self.page_size)):
                var randval = Int(random_ui64(0, num_paged_blocks - 1))
                while randval in used_set:
                    randval = Int(random_ui64(0, num_paged_blocks - 1))

                used_set.add(randval)
                host_tensor[batch, block_idx] = randval

        if ctx:
            var ctx_value = ctx.value()
            self.device_buf = ctx_value.enqueue_create_buffer[DType.uint32](
                self.shape.flattened_length()
            )
            ctx_value.enqueue_copy(self.device_buf.value(), self.host_ptr)

    @staticmethod
    fn build(
        prompt_lens: List[Int],
        cache_sizes: List[Int],
        max_full_context_length: Int,
        num_paged_blocks: Int,
        ctx: Optional[DeviceContext],
    ) raises -> Self:
        @parameter
        fn seq_len_fn(batch: Int) -> Int:
            return cache_sizes[batch] + prompt_lens[batch]

        var batch_size = len(prompt_lens)
        var paged_lut = Self(batch_size, max_full_context_length)
        paged_lut._build[seq_len_fn](batch_size, num_paged_blocks, ctx)
        return paged_lut^

    @staticmethod
    fn build[
        batch_size: Int
    ](
        prompt_lens: IndexList[batch_size],
        cache_sizes: IndexList[batch_size],
        max_full_context_length: Int,
        num_paged_blocks: Int,
        ctx: DeviceContext,
    ) raises -> Self:
        @parameter
        fn seq_len_fn(batch: Int) -> Int:
            return cache_sizes[batch] + prompt_lens[batch]

        var paged_lut = Self(batch_size, max_full_context_length)
        paged_lut._build[seq_len_fn](batch_size, num_paged_blocks, ctx)
        return paged_lut^

    fn __del__(deinit self):
        self.host_ptr.free()

    fn host_tensor(self) -> Self.tensor_type:
        return self._tensor(self.host_ptr)

    fn device_tensor(self) -> Self.tensor_type:
        return self._tensor(self.device_buf.value().unsafe_ptr())

    fn _runtime_layout(self) -> RuntimeLayout[Self.layout]:
        return RuntimeLayout[Self.layout].row_major(self.shape)

    fn _tensor(
        self, ptr: UnsafePointer[Scalar[DType.uint32]]
    ) -> Self.tensor_type:
        return Self.tensor_type(ptr, self._runtime_layout())
