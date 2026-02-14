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
from gpu.host import DeviceContext
from gpu.host.nvidia.tma import TensorMapSwizzle
from kv_cache.types import KVCacheT, swizzle_granularity, padded_depth
from layout import Layout, LayoutTensor
from layout.layout import UNKNOWN_VALUE, DimList
from layout.runtime_layout import RuntimeLayout
from layout.tma_async import (
    SplitLastDimTMATensorTile,
    create_split_tma,
    RaggedTMA3DTile,
)

from utils import Index, IndexList

from builtin.device_passable import DevicePassable


trait MHAOperand(DevicePassable, TrivialRegisterPassable):
    """This serves as the trait to support arguments to our MHA kernel."""

    comptime dtype: DType
    comptime page_size: Int

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
    ) -> UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]:
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
    fn create_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        depth: Int,
        BK: Int = padded_depth[Self.dtype, swizzle_mode, depth](),
    ](self, ctx: DeviceContext) raises -> SplitLastDimTMATensorTile[
        Self.dtype,
        IndexList[3](BN, 1, BK),
        swizzle_mode,
    ]:
        """Creates a TMA tile for efficient GPU memory transfers.
        This is useful for `k-major` MMA operations where we don't
        need to mask any extra rows."""
        ...

    @always_inline
    fn create_ragged_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        depth: Int,
        BK: Int = padded_depth[Self.dtype, swizzle_mode, depth](),
    ](self, ctx: DeviceContext) raises -> RaggedTMA3DTile[
        Self.dtype,
        swizzle_mode,
        BM=BN,
        BN=BK,
    ]:
        """Creates a TMA tile for efficient GPU memory transfers.
        This is useful for `mn-major` MMA operations where we need
        to mask extra rows to avoid adding `NaN` to the output
        through the MMA reduction."""
        ...


struct KVCacheMHAOperand[
    cache_t: KVCacheT,
](MHAOperand, TrivialRegisterPassable):
    """An implementation for `mo.opaque` KVCacheT arguments to MHA kernels.

    We can eventually remove this trait and just add it as a sub-trait in the
    KVCacheT type, but we need to solve some cyclic dependencies first.
    """

    comptime dtype = Self.cache_t.dtype
    comptime page_size = Self.cache_t.page_size_
    var cache: Self.cache_t

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "KVCacheMHAOperand"

    fn __init__(out self, cache: Self.cache_t):
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
    ) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
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
    fn create_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        depth: Int,
        BK: Int = padded_depth[Self.dtype, swizzle_mode, depth](),
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            Self.dtype,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Creates a TMA tile for efficient GPU memory transfers."""
        # Forward to the underlying cache's implementation
        # TODO: remove `comptime assert` when the `where` clause is enough
        constrained[
            (BK % swizzle_granularity[Self.dtype, swizzle_mode]()) == 0
        ]()
        tma = rebind[type_of(tma)](
            self.cache.create_tma_tile[swizzle_mode, BN=BN, BK=BK](ctx)
        )

    @always_inline
    fn create_ragged_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        depth: Int,
        BK: Int = padded_depth[Self.dtype, swizzle_mode, depth](),
    ](
        self,
        ctx: DeviceContext,
        out tma: RaggedTMA3DTile[
            Self.dtype,
            swizzle_mode,
            BM=BN,
            BN=BK,
        ],
    ) raises:
        # Forward to the underlying cache's implementation
        constrained[
            depth == Int(Self.cache_t.kv_params.head_size),
            "depth must match kv_params.head_size",
        ]()
        constrained[
            (BK % swizzle_granularity[Self.dtype, swizzle_mode]()) == 0,
            "BK must be a multiple of swizzle granularity",
        ]()
        tma = rebind[type_of(tma)](
            self.cache.create_ragged_tma_tile[swizzle_mode, BN=BN, BK=BK](ctx)
        )


struct LayoutTensorMHAOperand[dtype_: DType, layout: Layout](
    MHAOperand, TrivialRegisterPassable
):
    """An implementation for LayoutTensor arguments to MHA kernels."""

    comptime dtype = Self.dtype_
    comptime page_size = 0
    var buffer: LayoutTensor[Self.dtype, Self.layout, MutAnyOrigin]

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "LayoutTensorMHAOperand"

    fn __init__(
        out self,
        buffer: LayoutTensor[Self.dtype, Self.layout, MutAnyOrigin],
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
    ) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        var ret_ptr = self.buffer.ptr + self.buffer._offset(
            IndexList[self.layout.rank()](
                Int(batch_idx),
                Int(start_tok_idx),
                Int(head_idx),
                Int(head_dim_idx),
            )
        )
        return ret_ptr

    @always_inline
    fn cache_length(self, batch_idx: Int) -> Int:
        # LayoutTensor path assumes BSHD layout and all cache entries have
        # the same length.
        return self.buffer.dim[1]()

    @always_inline
    fn max_context_length(self) -> UInt32:
        return UInt32(self.buffer.dim[1]())

    @always_inline
    fn row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        return batch_idx * UInt32(self.buffer.dim[1]()) + start_tok_idx

    @always_inline
    fn create_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        depth: Int,
        BK: Int = padded_depth[Self.dtype, swizzle_mode, depth](),
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            Self.dtype,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Creates a TMA tile for efficient GPU memory transfers."""
        # View the 4D buffer as a 2D matrix [batch*seq, heads*head_dim]
        constrained[
            (BK % swizzle_granularity[Self.dtype, swizzle_mode]()) == 0
        ]()
        var rows = self.buffer.dim[0]() * self.buffer.dim[1]()
        comptime smem_shape = IndexList[3](BN, 1, BK)
        comptime gmem_shape = IndexList[3](UNKNOWN_VALUE, UNKNOWN_VALUE, depth)

        tma = create_split_tma[
            smem_shape,
            gmem_shape,
            swizzle_mode=swizzle_mode,
        ](ctx, self.buffer.ptr, rows, self.buffer.dim[2]())

    @always_inline
    fn create_ragged_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        depth: Int,
        BK: Int = padded_depth[Self.dtype, swizzle_mode, depth](),
    ](
        self,
        ctx: DeviceContext,
        out tma: RaggedTMA3DTile[
            Self.dtype,
            swizzle_mode,
            BM=BN,
            BN=BK,
        ],
    ) raises:
        constrained[
            (BK % swizzle_granularity[Self.dtype, swizzle_mode]()) == 0
        ]()
        var rows = self.buffer.dim[0]() * self.buffer.dim[1]()
        var num_heads = self.buffer.dim[2]()
        tma = type_of(tma).create[depth=depth](
            ctx, self.buffer.ptr, rows=rows, middle_dim=num_heads
        )


struct RaggedMHAOperand[dtype_: DType, layout: Layout, cache_layout: Layout](
    MHAOperand, TrivialRegisterPassable
):
    """An implementation for ragged LayoutTensor arguments to MHA kernels."""

    comptime dtype = Self.dtype_
    comptime page_size = 0
    var buffer: LayoutTensor[Self.dtype, Self.layout, ImmutAnyOrigin]
    var cache_row_offsets: LayoutTensor[
        DType.uint32, Self.cache_layout, ImmutAnyOrigin
    ]

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "RaggedMHAOperand"

    fn __init__(
        out self,
        buffer: LayoutTensor[Self.dtype, Self.layout, ImmutAnyOrigin],
        cache_row_offsets: LayoutTensor[
            DType.uint32, Self.cache_layout, ImmutAnyOrigin
        ],
    ):
        comptime assert (
            buffer.rank == 3
        ), "only support rank 3 inputs for ragged inputs."
        comptime assert (
            cache_row_offsets.rank == 1
        ), "only support rank 1 inputs for cache offsets."
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
    ) -> UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]:
        global_token_idx = Int(
            self.cache_row_offsets[Int(batch_idx)] + start_tok_idx
        )
        var ret_ptr = self.buffer.ptr + self.buffer._offset(
            IndexList[self.layout.rank()](
                global_token_idx,
                Int(head_idx),
                Int(head_dim_idx),
            )
        )
        return ret_ptr

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
    fn create_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        depth: Int,
        BK: Int = padded_depth[Self.dtype, swizzle_mode, depth](),
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            Self.dtype,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Creates a TMA tile for efficient GPU memory transfers."""
        # View as [total_tokens, heads*head_dim]
        constrained[
            (BK % swizzle_granularity[Self.dtype, swizzle_mode]()) == 0
        ]()
        var rows = self.buffer.dim[0]()  # total tokens
        comptime smem_shape = IndexList[3](BN, 1, BK)
        comptime gmem_shape = IndexList[3](UNKNOWN_VALUE, UNKNOWN_VALUE, depth)

        tma = create_split_tma[
            smem_shape,
            gmem_shape,
            swizzle_mode=swizzle_mode,
        ](ctx, self.buffer.ptr, rows, self.buffer.dim[1]())

    @always_inline
    fn create_ragged_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        depth: Int,
        BK: Int = padded_depth[Self.dtype, swizzle_mode, depth](),
    ](
        self,
        ctx: DeviceContext,
        out tma: RaggedTMA3DTile[
            Self.dtype,
            swizzle_mode,
            BM=BN,
            BN=BK,
        ],
    ) raises:
        constrained[
            (BK % swizzle_granularity[Self.dtype, swizzle_mode]()) == 0
        ]()
        var rows = self.buffer.dim[0]()  # total tokens
        var num_heads = self.buffer.dim[1]()
        tma = type_of(tma).create[depth=depth](
            ctx, self.buffer.ptr, rows=rows, middle_dim=num_heads
        )
