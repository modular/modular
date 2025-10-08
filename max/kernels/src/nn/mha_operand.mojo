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
from gpu.host import DeviceContext
from gpu.host._nvidia_cuda import TensorMapSwizzle
from kv_cache.types import KVCacheT
from layout import Layout, LayoutTensor
from layout.layout import UNKNOWN_VALUE, DimList
from layout.runtime_layout import RuntimeLayout
from layout.tma_async import TMANestedTensorTile, create_nested_tma_tile

from utils import Index, IndexList

from builtin.device_passable import DevicePassable


@register_passable("trivial")
trait MHAOperand(DevicePassable):
    """This serves as the trait to support arguments to our MHA kernel."""

    alias dtype: DType

    # TODO: change this to return a LayoutTensor once MOCO-1471 is fixed
    @always_inline
    fn block_paged_ptr[
        tile_size: Int,
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        head_dim_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.dtype]]:
        ...

    @always_inline
    fn cache_length(self, batch_idx: Int) -> Int:
        """Returns the length of the cache for a given batch index."""
        ...

    @always_inline
    fn max_context_length(self) -> UInt32:
        """Returns the maximum cache length in a given batch index."""
        ...

    @always_inline
    fn row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        ...

    @always_inline
    fn col_idx(self, head_idx: UInt32) -> UInt32:
        """Returns the col idx when viewing the memory as a matrix."""
        ...

    @always_inline
    fn create_tma_tile[
        tile_m: Int,
        tile_n: Int,
        swizzle_mode: TensorMapSwizzle,
        *,
        is_k_major: Bool,
    ](self, ctx: DeviceContext) raises -> TMANestedTensorTile[
        Self.dtype, tile_m, tile_n, swizzle_mode, is_k_major=is_k_major
    ]:
        """Creates a TMA tile for efficient GPU memory transfers."""
        ...


@register_passable("trivial")
struct KVCacheMHAOperand[cache_t: KVCacheT](MHAOperand):
    """An implementation for `mo.opaque` KVCacheT arguments to MHA kernels.

    We can eventually remove this trait and just add it as a sub-trait in the
    KVCacheT type, but we need to solve some cyclic dependencies first.
    """

    alias dtype = cache_t.dtype
    var cache: cache_t

    alias device_type: AnyTrivialRegType = Self

    fn _to_device_type(self, target: OpaquePointer):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "KVCacheMHAOperand"

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

    fn __init__(out self, cache: cache_t):
        self.cache = cache

    @always_inline
    fn block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        head_dim_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.dtype]]:
        return self.cache.block_paged_ptr[tile_size](
            Int(batch_idx), Int(start_tok_idx), Int(head_idx), Int(head_dim_idx)
        )

    @always_inline
    fn cache_length(self, batch_idx: Int) -> Int:
        return self.cache.cache_length(batch_idx)

    @always_inline
    fn max_context_length(self) -> UInt32:
        return self.cache.max_context_length()

    @always_inline
    fn row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        return self.cache.row_idx(batch_idx, start_tok_idx)

    @always_inline
    fn col_idx(self, head_idx: UInt32) -> UInt32:
        """Returns the col idx when viewing the memory as a matrix."""
        return self.cache.col_idx(head_idx)

    @always_inline
    fn create_tma_tile[
        tile_m: Int,
        tile_n: Int,
        swizzle_mode: TensorMapSwizzle,
        *,
        is_k_major: Bool,
    ](self, ctx: DeviceContext) raises -> TMANestedTensorTile[
        Self.dtype,
        tile_m,
        tile_n,
        swizzle_mode,
        is_k_major=is_k_major,
    ]:
        """Creates a TMA tile for efficient GPU memory transfers."""
        # Forward to the underlying cache's implementation
        return self.cache.create_tma_tile[
            tile_m, tile_n, swizzle_mode, is_k_major=is_k_major
        ](ctx)


@register_passable("trivial")
struct LayoutTensorMHAOperand[dtype_: DType, layout: Layout](MHAOperand):
    """An implementation for NDBuffer arguments to MHA kernels."""

    alias dtype = dtype_
    var buffer: LayoutTensor[Self.dtype, layout, MutableAnyOrigin]

    alias device_type: AnyTrivialRegType = Self

    fn _to_device_type(self, target: OpaquePointer):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "LayoutTensorMHAOperand"

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

    fn __init__(
        out self,
        buffer: LayoutTensor[Self.dtype, layout, MutableAnyOrigin],
    ):
        self.buffer = buffer

    @always_inline
    fn block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        head_dim_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.dtype]]:
        var ret_ptr = self.buffer.ptr + self.buffer._offset(
            IndexList[self.layout.rank()](
                Int(batch_idx),
                Int(start_tok_idx),
                Int(head_idx),
                Int(head_dim_idx),
            )
        )
        return rebind[UnsafePointer[Scalar[Self.dtype]]](ret_ptr)

    @always_inline
    fn cache_length(self, batch_idx: Int) -> Int:
        # NDBuffer path assumes BSHD layout and all cache entries have
        # the same length.
        return self.buffer.dim[1]()

    @always_inline
    fn max_context_length(self) -> UInt32:
        return self.buffer.dim[1]()

    @always_inline
    fn row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        return batch_idx * self.buffer.dim[1]() + start_tok_idx

    @always_inline
    fn col_idx(self, head_idx: UInt32) -> UInt32:
        """Returns the col idx when viewing the memory as a matrix."""
        return head_idx * self.buffer.dim[self.layout.rank() - 1]()

    @always_inline
    fn create_tma_tile[
        tile_m: Int,
        tile_n: Int,
        swizzle_mode: TensorMapSwizzle,
        *,
        is_k_major: Bool,
    ](self, ctx: DeviceContext) raises -> TMANestedTensorTile[
        Self.dtype,
        tile_m,
        tile_n,
        swizzle_mode,
        is_k_major=is_k_major,
    ]:
        """Creates a TMA tile for efficient GPU memory transfers."""
        # View the 4D buffer as a 2D matrix [batch*seq, heads*head_dim]
        var rows = self.buffer.dim[0]() * self.buffer.dim[1]()
        var cols = self.buffer.dim[2]() * self.buffer.dim[3]()
        alias layout_ = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

        rt_layout = RuntimeLayout[layout_].row_major(IndexList[2](rows, cols))

        var tensor = LayoutTensor[Self.dtype, layout_, MutableAnyOrigin](
            self.buffer.ptr, rt_layout
        )

        return create_nested_tma_tile[
            tile_m, tile_n, swizzle_mode, is_k_major=is_k_major
        ](ctx, tensor)


@register_passable("trivial")
struct RaggedMHAOperand[dtype_: DType, layout: Layout, cache_layout: Layout](
    MHAOperand
):
    """An implementation for ragged NDBuffer arguments to MHA kernels."""

    alias dtype = dtype_
    var buffer: LayoutTensor[Self.dtype, layout, MutableAnyOrigin]
    var cache_row_offsets: LayoutTensor[
        DType.uint32, cache_layout, MutableAnyOrigin
    ]

    alias device_type: AnyTrivialRegType = Self

    fn _to_device_type(self, target: OpaquePointer):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "RaggedMHAOperand"

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

    fn __init__(
        out self,
        buffer: LayoutTensor[Self.dtype, layout, MutableAnyOrigin],
        cache_row_offsets: LayoutTensor[
            DType.uint32, cache_layout, MutableAnyOrigin
        ],
    ):
        constrained[
            buffer.rank == 3, "only support rank 3 inputs for ragged inputs."
        ]()
        constrained[
            cache_row_offsets.rank == 1,
            "only support rank 1 inputs for cache offsets.",
        ]()
        self.buffer = buffer
        self.cache_row_offsets = cache_row_offsets

    @always_inline
    fn block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        head_dim_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.dtype]]:
        global_token_idx = Int(
            self.cache_row_offsets[Int(batch_idx)] + start_tok_idx
        )
        var ret_ptr = self.buffer.ptr + self.buffer._offset(
            IndexList[self.layout.rank()](
                Int(global_token_idx),
                Int(head_idx),
                Int(head_dim_idx),
            )
        )
        return rebind[UnsafePointer[Scalar[Self.dtype]]](ret_ptr)

    @always_inline
    fn cache_length(self, batch_idx: Int) -> Int:
        return Int(
            self.cache_row_offsets[batch_idx + 1]
            - self.cache_row_offsets[batch_idx]
        )

    @always_inline
    fn max_context_length(self) -> UInt32:
        # NotImplemented
        return 0

    @always_inline
    fn row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        return self.cache_row_offsets[Int(batch_idx)][0] + start_tok_idx

    @always_inline
    fn col_idx(self, head_idx: UInt32) -> UInt32:
        """Returns the col idx when viewing the memory as a matrix."""
        return head_idx * self.buffer.dim[2]()

    @always_inline
    fn create_tma_tile[
        tile_m: Int,
        tile_n: Int,
        swizzle_mode: TensorMapSwizzle,
        *,
        is_k_major: Bool,
    ](self, ctx: DeviceContext) raises -> TMANestedTensorTile[
        Self.dtype,
        tile_m,
        tile_n,
        swizzle_mode,
        is_k_major=is_k_major,
    ]:
        """Creates a TMA tile for efficient GPU memory transfers."""
        # View as [total_tokens, heads*head_dim]
        var rows = self.buffer.dim[0]()  # total tokens
        var cols = self.buffer.dim[1]() * self.buffer.dim[2]()

        alias layout_ = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

        rt_layout = RuntimeLayout[layout_].row_major(IndexList[2](rows, cols))
        var tensor = LayoutTensor[Self.dtype, layout_, MutableAnyOrigin](
            self.buffer.ptr, rt_layout
        )

        return create_nested_tma_tile[
            tile_m, tile_n, swizzle_mode, is_k_major=is_k_major
        ](ctx, tensor)
