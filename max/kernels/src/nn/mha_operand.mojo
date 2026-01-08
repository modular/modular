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

from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
comptime OpaquePointer = LegacyUnsafePointer[
    mut=True, NoneType, origin=MutAnyOrigin
]
from utils import Index, IndexList

from builtin.device_passable import DevicePassable


@register_passable("trivial")
trait MHAOperand(DevicePassable):
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


@register_passable("trivial")
struct KVCacheMHAOperand[
    cache_t: KVCacheT,
](MHAOperand):
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

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

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
        # TODO: remove `__comptime_assert` when the `where` clause is enough
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
    ) raises where depth == Int(Self.cache_t.kv_params.head_size):
        # Forward to the underlying cache's implementation
        # TODO: remove `__comptime_assert` when the `where` clause is enough
        constrained[
            (BK % swizzle_granularity[Self.dtype, swizzle_mode]()) == 0
        ]()
        tma = rebind[type_of(tma)](
            self.cache.create_ragged_tma_tile[swizzle_mode, BN=BN, BK=BK](ctx)
        )


@register_passable("trivial")
struct LayoutTensorMHAOperand[dtype_: DType, layout: Layout](MHAOperand):
    """An implementation for NDBuffer arguments to MHA kernels."""

    comptime dtype = Self.dtype_
    comptime page_size = 0
    var buffer: LayoutTensor[Self.dtype, Self.layout, MutAnyOrigin]

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "LayoutTensorMHAOperand"

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

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
            ctx, self.buffer.ptr, rows=Int(rows), middle_dim=num_heads
        )


@register_passable("trivial")
struct RaggedMHAOperand[dtype_: DType, layout: Layout, cache_layout: Layout](
    MHAOperand
):
    """An implementation for ragged NDBuffer arguments to MHA kernels."""

    comptime dtype = Self.dtype_
    comptime page_size = 0
    var buffer: LayoutTensor[Self.dtype, Self.layout, MutAnyOrigin]
    var cache_row_offsets: LayoutTensor[
        DType.uint32, Self.cache_layout, MutAnyOrigin
    ]

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "RaggedMHAOperand"

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

    fn __init__(
        out self,
        buffer: LayoutTensor[Self.dtype, Self.layout, MutAnyOrigin],
        cache_row_offsets: LayoutTensor[
            DType.uint32, Self.cache_layout, MutAnyOrigin
        ],
    ):
        __comptime_assert (
            buffer.rank == 3
        ), "only support rank 3 inputs for ragged inputs."
        __comptime_assert (
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
            ctx, self.buffer.ptr, rows=Int(rows), middle_dim=num_heads
        )
