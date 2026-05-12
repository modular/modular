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
from std.gpu.host import DeviceContext
from std.gpu.host.nvidia.tma import TensorMapL2Promotion, TensorMapSwizzle
from kv_cache.types import (
    KVCacheT,
    PagedRowIndices,
    _populate_via_row_idx,
    kv_num_sub_tiles,
    kv_sub_tile_rows,
    padded_depth,
    swizzle_granularity,
)
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from layout.tile_layout import TensorLayout
from layout.tma_async import (
    SplitLastDimTMATensorTile,
    SharedMemBarrier,
    _gather4_box_width,
    create_split_tma,
    RaggedTMA3DTile,
    TMATensorTile,
    create_tensor_tile,
    create_tma_tile_gather4,
)
from layout.tile_tensor import TileTensor
from layout.tile_layout import row_major
from layout.coord import Idx, Coord
from std.math import ceildiv

from std.utils import Index, IndexList

from std.builtin.device_passable import DevicePassable


trait MHAOperand(DevicePassable, TrivialRegisterPassable):
    """This serves as the trait to support arguments to our MHA kernel."""

    comptime dtype: DType
    comptime scale_dtype: DType
    comptime page_size: Int
    comptime quantization_enabled: Bool = False
    comptime quantization_granularity: Int

    # TODO: change this to return a LayoutTensor once MOCO-1471 is fixed
    @always_inline
    def block_paged_ptr[
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
    def block_paged_tile[
        layout_t: TensorLayout,
        //,
        tile_size: Int,
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        layout_val: layout_t,
        head_dim_idx: UInt32 = 0,
    ) -> TileTensor[Self.dtype, layout_t, ImmutAnyOrigin]:
        """Wraps block_paged_ptr in a TileTensor with the caller's layout."""
        return TileTensor[Self.dtype, layout_t, ImmutAnyOrigin](
            ptr=self.block_paged_ptr[tile_size](
                batch_idx, start_tok_idx, head_idx, head_dim_idx
            ),
            layout=layout_val,
        )

    @always_inline
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], ImmutAnyOrigin]:
        ...

    @always_inline
    def load_scale[
        width: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[Self.scale_dtype, width]:
        ...

    @always_inline
    def cache_length(self, batch_idx: Int) -> Int:
        """Returns the length of the cache for a given batch index."""
        ...

    @always_inline
    def max_context_length(self) -> UInt32:
        """Returns the maximum cache length in a given batch index."""
        ...

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of virtual rows in the KV memory view.

        For paged caches this accounts for the paging stride so that TMA
        descriptors can be sized to cover the entire address space.
        """
        ...

    @always_inline
    def row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        ...

    @always_inline
    def populate[
        BN: Int,
        base_alignment: Int,
        pair_cta: Bool = False,
        is_leader: Bool = True,
    ](self, batch_idx: UInt32, base_kv_row: UInt32) -> PagedRowIndices[
        BN, Self.page_size, pair_cta, is_leader
    ]:
        """Populate a full `PagedRowIndices[BN, ...]` for a BN-row tile.

        Returns the precomputed physical row indices for the `num_pages`
        sub-tile pages covering the BN-row range starting at
        `base_kv_row` for `batch_idx`. Both K's TMA (which may cover only
        a subset in `pair_cta` mode) and V's TMA (which covers the full
        range) can then consume the result without any lazy LUT lookup.

        `base_alignment` is a comptime promise that
        `base_kv_row % base_alignment == 0` at runtime — typically
        `mask.start_column_alignment[...]()`. The `PagedKVCache`
        override uses it to pick the largest legal SIMD chunk for its
        LUT vector load and to skip the intra-page divmod when
        `base_alignment % page_size == 0`.

        Default implementation: scalar loop over `num_pages` calls to
        `row_idx`. Overrides (e.g. `PagedKVCache`) replace this with a
        single SIMD load from the underlying lookup table.
        """

        @parameter
        def _row(batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
            return self.row_idx(batch_idx, start_tok_idx)

        return _populate_via_row_idx[
            BN, Self.page_size, pair_cta, is_leader, _row
        ](batch_idx, base_kv_row)

    @always_inline
    def get_tma_row(self, encoded_index: Int32) -> Int32:
        """Convert an encoded sparse index to a physical TMA row.

        For paged caches the encoded index is
        ``physical_block * page_size + offset`` and this method returns
        ``physical_block * stride + offset``.  Non-paged operands return
        the encoded index unchanged.
        """
        ...

    @always_inline
    def create_tma_tile[
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
    def create_scale_tma_tile[
        BMN: Int
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        Self.scale_dtype,
        2,
        Index(1, BMN),
        Index(1, BMN),
    ]:
        """Creates a TMA tile for efficient GPU memory transfers.
        This is useful for `m-major` MMA operations where we don't
        need to mask any extra rows."""
        ...

    @always_inline
    def create_ragged_tma_tile[
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

    @always_inline
    def create_rope_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int,
        padded_depth: Int,
    ](self, ctx: DeviceContext) raises -> SplitLastDimTMATensorTile[
        DType.bfloat16,
        IndexList[3](BN, 1, BK),
        swizzle_mode,
    ]:
        """Creates a BF16 TMA tile for the rope portion of the per-tensor rope-aware KV cache.
        """
        ...

    @always_inline
    def create_gather4_tma_tile[
        tile_width: Int,
        tile_stride: Int = tile_width,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        tma_dtype: DType = Self.dtype,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        tma_dtype,
        2,
        tile_shape=IndexList[2](
            tile_height,
            _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
        ),
    ]:
        """Creates a 2D TMA gather4 descriptor for this operand.

        The descriptor views the data as a flat 2D matrix of
        ``[num_kv_rows, tile_width]`` and is configured for gather4 operations
        that load 4 non-contiguous rows per TMA instruction. The box width
        is derived from the swizzle mode; for SWIZZLE_NONE it equals
        ``tile_width``.

        When ``tma_dtype`` differs from ``Self.dtype``, the underlying data
        pointer is bitcast to ``tma_dtype`` at descriptor creation time.

        Parameters:
            tile_width: Number of elements per row to load (box width) in
                ``tma_dtype`` elements.
            tile_stride: Row stride in elements in global memory. Defaults to
                ``tile_width``. Use a larger value when the global row is wider
                than the portion to load.
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
                Defaults to SWIZZLE_NONE.
            tile_height: Number of rows in the tile. Must be a multiple of 4.
                Defaults to 4 for backward compatibility.
            tma_dtype: The data type used for the TMA descriptor. Defaults to
                ``Self.dtype``. When different, the pointer is bitcast.
            l2_promotion: L2 cache promotion hint for TMA loads. Defaults to
                NONE.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.

        Returns:
            A TMATensorTile with box width derived from the swizzle mode.
        """
        ...

    @always_inline
    def create_rope_gather4_tma_tile[
        tile_width: Int,
        padded_depth: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        DType.bfloat16,
        2,
        tile_shape=IndexList[2](
            tile_height,
            _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
        ),
    ]:
        """Creates a BF16 gather4 TMA descriptor for the rope portion of the
        KV cache.

        For the per-tensor rope-aware layout each token row is
        ``padded_depth`` FP8 bytes (content) followed by BF16 rope elements.
        This method offsets the base pointer by ``padded_depth`` bytes,
        reinterprets as BF16, and creates a gather4 TMA descriptor.

        Parameters:
            tile_width: Number of BF16 elements per row in global memory.
            padded_depth: Byte offset from row start to the rope data.
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
            tile_height: Number of rows in the tile. Must be a multiple of 4.
            l2_promotion: L2 cache promotion hint for TMA loads. Defaults to
                NONE.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.

        Returns:
            A BF16 TMATensorTile configured for gather4.
        """
        ...

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Scalar[DType.float32], MutAnyOrigin]:
        """Returns the base pointer to the quantization scales tensor.

        Returns a null pointer for operands without quantization support.
        """
        ...


struct KVCacheMHAOperand[
    cache_t: KVCacheT,
](MHAOperand, TrivialRegisterPassable):
    """An implementation for `mo.opaque` KVCacheT arguments to MHA kernels.

    We can eventually remove this trait and just add it as a sub-trait in the
    KVCacheT type, but we need to solve some cyclic dependencies first.
    """

    comptime dtype = Self.cache_t.dtype
    comptime scale_dtype = Self.cache_t.scale_dtype
    comptime page_size = Self.cache_t.page_size_
    comptime quantization_enabled = Self.cache_t.quantization_enabled
    comptime quantization_granularity = Self.cache_t.quantization_granularity
    var cache: Self.cache_t

    comptime device_type: AnyType = Self

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "KVCacheMHAOperand"

    def __init__(out self, cache: Self.cache_t):
        self.cache = cache

    @always_inline
    def block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        head_dim_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]:
        return self.cache.block_paged_ptr[tile_size](
            Int(batch_idx), Int(start_tok_idx), Int(head_idx), Int(head_dim_idx)
        )

    @always_inline
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], ImmutAnyOrigin]:
        return self.cache.scales_block_paged_ptr(
            batch_idx, start_tok_idx, head_idx, head_dim_idx
        )

    @always_inline
    def load_scale[
        width: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[Self.scale_dtype, width]:
        return self.cache.load_scale[width=width](
            batch_idx, head_idx, start_tok_idx, head_dim_idx
        )

    @always_inline
    def cache_length(self, batch_idx: Int) -> Int:
        return self.cache.cache_length(batch_idx)

    @always_inline
    def max_context_length(self) -> UInt32:
        return self.cache.max_context_length()

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of virtual rows in the KV memory view."""
        return self.cache.num_kv_rows()

    @always_inline
    def row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        return self.cache.row_idx(batch_idx, start_tok_idx)

    @always_inline
    def populate[
        BN: Int,
        base_alignment: Int,
        pair_cta: Bool = False,
        is_leader: Bool = True,
    ](self, batch_idx: UInt32, base_kv_row: UInt32) -> PagedRowIndices[
        BN, Self.page_size, pair_cta, is_leader
    ]:
        """Delegate to the underlying cache's `populate`.

        `PagedKVCache.populate` overrides with a SIMD lookup-table read;
        other cache types fall through to the scalar default on
        `KVCacheT.populate`. `base_alignment` is the comptime alignment
        of `base_kv_row` (typically `mask.start_column_alignment[...]()`).
        """
        return self.cache.populate[BN, base_alignment, pair_cta, is_leader](
            batch_idx, base_kv_row
        )

    @always_inline
    def get_tma_row(self, encoded_index: Int32) -> Int32:
        """Convert an encoded sparse index to a physical TMA row."""
        return self.cache.get_tma_row(encoded_index)

    @always_inline
    def create_tma_tile[
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
        comptime assert (
            BK % swizzle_granularity[Self.dtype, swizzle_mode]()
        ) == 0
        tma = rebind[type_of(tma)](
            self.cache.create_tma_tile[swizzle_mode, BN=BN, BK=BK](ctx)
        )

    @always_inline
    def create_scale_tma_tile[
        BMN: Int
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            Self.scale_dtype,
            2,
            Index(1, BMN),
            Index(1, BMN),
        ],
    ) raises:
        """Creates a TMA tile for efficient GPU memory transfers.
        This is useful for `m-major` MMA operations where we don't
        need to mask any extra rows."""
        comptime assert False, "create_scale_tma_tile is not implemented"

    @always_inline
    def create_ragged_tma_tile[
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
        comptime assert (
            depth == Self.cache_t.kv_params.head_size
        ), "depth must match kv_params.head_size"
        comptime assert (
            BK % swizzle_granularity[Self.dtype, swizzle_mode]()
        ) == 0, "BK must be a multiple of swizzle granularity"
        tma = rebind[type_of(tma)](
            self.cache.create_ragged_tma_tile[swizzle_mode, BN=BN, BK=BK](ctx)
        )

    @always_inline
    def create_rope_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int,
        padded_depth: Int,
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            DType.bfloat16,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Delegates to the underlying KVCache to create a BF16 rope TMA tile.
        """
        tma = self.cache.create_rope_tma_tile[
            swizzle_mode, BN=BN, BK=BK, padded_depth=padded_depth
        ](ctx)

    @always_inline
    def create_gather4_tma_tile[
        tile_width: Int,
        tile_stride: Int = tile_width,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        tma_dtype: DType = Self.dtype,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            tma_dtype,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Creates a 2D TMA gather4 descriptor for this KV cache operand."""
        tma = rebind[type_of(tma)](
            self.cache.create_gather4_tma_tile[
                tile_height=tile_height,
                tile_width=tile_width,
                tile_stride=tile_stride,
                swizzle_mode=swizzle_mode,
                tma_dtype=tma_dtype,
                l2_promotion=l2_promotion,
            ](ctx)
        )

    @always_inline
    def create_rope_gather4_tma_tile[
        tile_width: Int,
        padded_depth: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            DType.bfloat16,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Delegates to the underlying KVCache to create a BF16 rope gather4
        TMA tile."""
        tma = rebind[type_of(tma)](
            self.cache.create_rope_gather4_tma_tile[
                tile_height=tile_height,
                tile_width=tile_width,
                padded_depth=padded_depth,
                swizzle_mode=swizzle_mode,
                l2_promotion=l2_promotion,
            ](ctx)
        )

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Scalar[DType.float32], MutAnyOrigin]:
        """Returns the base pointer to the quantization scales tensor."""
        return rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](
            self.cache.scales_raw_ptr()
        )


struct KVCacheScalesMHAOperand[
    cache_t: KVCacheT,
](MHAOperand, TrivialRegisterPassable):
    """An MHAOperand that accesses the scales field of a KVCache.

    This is useful for MLA attention where k_s (per-token scales) are stored
    in the scales field of the k cache with quantization_granularity = head_size.
    The scales have shape [num_blocks, page_size, num_heads, head_dim_granularity].
    """

    comptime dtype = Self.cache_t.scale_dtype
    comptime scale_dtype = DType.invalid
    comptime page_size = Self.cache_t.page_size_
    comptime quantization_enabled = Self.cache_t.quantization_enabled
    comptime quantization_granularity = Self.cache_t.quantization_granularity
    var cache: Self.cache_t

    comptime device_type: AnyType = Self

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "KVCacheScalesMHAOperand"

    def __init__(out self, cache: Self.cache_t):
        self.cache = cache

    @always_inline
    def block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        head_dim_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]:
        # Forward to scales_block_paged_ptr instead of block_paged_ptr
        return self.cache.scales_block_paged_ptr(
            Int(batch_idx), Int(start_tok_idx), Int(head_idx), Int(head_dim_idx)
        )

    @always_inline
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], ImmutAnyOrigin]:
        # SAFETY: KVCacheScalesMHAOperand doesn't support block-paged scales;
        # callers only dereference behind comptime quantization guards.
        return UnsafePointer[
            Scalar[Self.scale_dtype], ImmutAnyOrigin
        ].unsafe_dangling()

    @always_inline
    def load_scale[
        width: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[Self.scale_dtype, width]:
        return SIMD[Self.scale_dtype, width](0)

    @always_inline
    def cache_length(self, batch_idx: Int) -> Int:
        return self.cache.cache_length(batch_idx)

    @always_inline
    def max_context_length(self) -> UInt32:
        return self.cache.max_context_length()

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of virtual rows in the KV memory view."""
        return self.cache.num_kv_rows()

    @always_inline
    def row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        return self.cache.row_idx(batch_idx, start_tok_idx)

    @always_inline
    def get_tma_row(self, encoded_index: Int32) -> Int32:
        """Convert an encoded sparse index to a physical TMA row."""
        return self.cache.get_tma_row(encoded_index)

    @always_inline
    def create_tma_tile[
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
        """TMA not supported for KVCacheScalesMHAOperand."""
        comptime assert False, "TMA not supported for KVCacheScalesMHAOperand"

    @always_inline
    def create_scale_tma_tile[
        BMN: Int
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            Self.scale_dtype,
            2,
            Index(1, BMN),
            Index(1, BMN),
        ],
    ) raises:
        comptime assert False, "create_scale_tma_tile is not implemented"

    @always_inline
    def create_ragged_tma_tile[
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
        """TMA not supported for KVCacheScalesMHAOperand."""
        comptime assert False, "TMA not supported for KVCacheScalesMHAOperand"

    @always_inline
    def create_rope_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int,
        padded_depth: Int,
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            DType.bfloat16,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Not supported for KVCacheScalesMHAOperand."""
        comptime assert (
            False
        ), "create_rope_tma_tile is not supported for KVCacheScalesMHAOperand"

    @always_inline
    def create_gather4_tma_tile[
        tile_width: Int,
        tile_stride: Int = tile_width,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        tma_dtype: DType = Self.dtype,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            tma_dtype,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Not supported for KVCacheScalesMHAOperand."""
        comptime assert False, (
            "create_gather4_tma_tile is not supported for"
            " KVCacheScalesMHAOperand"
        )

    @always_inline
    def create_rope_gather4_tma_tile[
        tile_width: Int,
        padded_depth: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            DType.bfloat16,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Not supported for KVCacheScalesMHAOperand."""
        comptime assert False, (
            "create_rope_gather4_tma_tile is not supported for"
            " KVCacheScalesMHAOperand"
        )

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Scalar[DType.float32], MutAnyOrigin]:
        """Returns a dangling pointer. KVCacheScalesMHAOperand already points to the
        scales pointer."""
        # SAFETY: Callers access scales through the operand's own pointer, not
        # this raw_ptr; only used behind comptime quantization guards.
        return UnsafePointer[
            Scalar[DType.float32], MutAnyOrigin
        ].unsafe_dangling()


struct LayoutTensorMHAOperand[
    origin: Origin[mut=False],
    scale_origin: Origin[mut=False],
    //,
    dtype_: DType,
    layout: Layout,
    scale_dtype_: DType = DType.float32,
    scale_layout: Layout = Layout(),
](MHAOperand, TrivialRegisterPassable):
    """An implementation for LayoutTensor arguments to MHA kernels."""

    comptime dtype = Self.dtype_
    comptime scale_dtype = Self.scale_dtype_
    comptime page_size = 0
    comptime layout_rank = Self.layout.rank()
    comptime scale_rank = Self.scale_layout.rank()
    comptime layout_dim: Int = Self.layout.shape[Self.layout_rank - 1].value()
    comptime scale_dim: Int = Self.scale_layout.shape[
        Self.scale_rank - 1
    ].value()
    comptime quantization_granularity: Int = ceildiv(
        Self.layout_dim, Self.scale_dim
    )
    comptime quantization_enabled: Bool = Self.scale_layout.rank() != 0

    var buffer: LayoutTensor[Self.dtype, Self.layout, Self.origin]
    var scale_buffer: LayoutTensor[
        Self.scale_dtype, Self.scale_layout, Self.scale_origin
    ]
    comptime device_type: AnyType = Self

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "LayoutTensorMHAOperand"

    def __init__(
        out self,
        buffer: LayoutTensor[Self.dtype, Self.layout, Self.origin],
        scale_buffer: LayoutTensor[
            Self.scale_dtype, Self.scale_layout, Self.scale_origin
        ] = LayoutTensor[Self.scale_dtype, Self.scale_layout, ImmutAnyOrigin](
            None
        ),
    ):
        self.buffer = buffer
        self.scale_buffer = scale_buffer

    @always_inline
    def block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        head_dim_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.dtype], ImmutAnyOrigin]:
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
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], ImmutAnyOrigin]:
        var ret_ptr = self.scale_buffer.ptr + self.scale_buffer._offset(
            IndexList[self.scale_layout.rank()](
                Int(batch_idx),
                Int(start_tok_idx),
                Int(head_idx),
                Int(head_dim_idx // Self.quantization_granularity),
            )
        )
        return ret_ptr

    @always_inline
    def load_scale[
        width: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[Self.scale_dtype, width]:
        return self.scale_buffer.load[width=width](
            Index(
                Int(batch_idx),
                Int(start_tok_idx),
                Int(head_idx),
                Int(head_dim_idx // Self.quantization_granularity),
            )
        )

    @always_inline
    def cache_length(self, batch_idx: Int) -> Int:
        # LayoutTensor path assumes BSHD layout and all cache entries have
        # the same length.
        return self.buffer.dim[1]()

    @always_inline
    def max_context_length(self) -> UInt32:
        return UInt32(self.buffer.dim[1]())

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of virtual rows (batch * seq_len)."""
        return self.buffer.dim[0]() * self.buffer.dim[1]()

    @always_inline
    def row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        return batch_idx * UInt32(self.buffer.dim[1]()) + start_tok_idx

    @always_inline
    def get_tma_row(self, encoded_index: Int32) -> Int32:
        """Convert an encoded sparse index to a physical TMA row.

        Non-paged operand: identity (no paging translation needed).
        """
        return encoded_index

    @always_inline
    def create_tma_tile[
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
        comptime assert (
            BK % swizzle_granularity[Self.dtype, swizzle_mode]()
        ) == 0, String("BN = ", BN, "\ndepth = ", depth, "\nBK = ", BK)
        var rows = self.buffer.dim[0]() * self.buffer.dim[1]()
        comptime smem_shape = IndexList[3](BN, 1, BK)
        comptime gmem_shape = IndexList[3](UNKNOWN_VALUE, UNKNOWN_VALUE, depth)

        tma = create_split_tma[
            smem_shape,
            gmem_shape,
            swizzle_mode=swizzle_mode,
        ](ctx, self.buffer.ptr, rows, self.buffer.dim[2]())

    @always_inline
    def create_scale_tma_tile[
        BMN: Int
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            Self.scale_dtype,
            2,
            Index(1, BMN),
            Index(1, BMN),
        ],
    ) raises:
        var total_elements = self.scale_buffer.size()
        debug_assert(
            total_elements % 4 == 0,
            (
                "Total_elements must be divisible by 4. Otherwise, the Nvidia"
                " Driver will crash."
            ),
        )
        var scale_tensor = TileTensor(
            self.scale_buffer.ptr,
            row_major(Coord(Idx[1](), Idx(total_elements))),
        )
        return create_tensor_tile[
            Index(1, BMN),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
            __desc_shape=Index(1, BMN),
        ](ctx, scale_tensor)

    @always_inline
    def create_ragged_tma_tile[
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
        comptime assert (
            BK % swizzle_granularity[Self.dtype, swizzle_mode]()
        ) == 0
        var rows = self.buffer.dim[0]() * self.buffer.dim[1]()
        var num_heads = self.buffer.dim[2]()
        tma = type_of(tma).create[depth=depth](
            ctx, self.buffer.ptr, rows=rows, middle_dim=num_heads
        )

    @always_inline
    def create_rope_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int,
        padded_depth: Int,
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            DType.bfloat16,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Not supported for LayoutTensorMHAOperand."""
        comptime assert (
            False
        ), "create_rope_tma_tile is not supported for LayoutTensorMHAOperand"

    @always_inline
    def create_gather4_tma_tile[
        tile_width: Int,
        tile_stride: Int = tile_width,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        tma_dtype: DType = Self.dtype,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            tma_dtype,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Creates a 2D TMA gather4 descriptor for this LayoutTensor operand."""
        # View the 4D buffer as a 2D matrix [batch*seq, tile_width]
        var rows = self.buffer.dim[0]() * self.buffer.dim[1]()
        tma = create_tma_tile_gather4[
            tma_dtype,
            tile_height=tile_height,
            tile_width=tile_width,
            tile_stride=tile_stride,
            swizzle_mode=swizzle_mode,
            l2_promotion=l2_promotion,
        ](ctx, self.buffer.ptr.bitcast[Scalar[tma_dtype]](), rows)

    @always_inline
    def create_rope_gather4_tma_tile[
        tile_width: Int,
        padded_depth: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            DType.bfloat16,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Not supported for LayoutTensorMHAOperand."""
        comptime assert False, (
            "create_rope_gather4_tma_tile is not supported for"
            " LayoutTensorMHAOperand"
        )

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Scalar[DType.float32], MutAnyOrigin]:
        """Returns a dangling pointer. LayoutTensor operands do not support
        quantization."""
        # SAFETY: LayoutTensor operands are never quantized; callers only
        # dereference behind comptime quantization guards.
        return UnsafePointer[
            Scalar[DType.float32], MutAnyOrigin
        ].unsafe_dangling()


struct RaggedMHAOperand[
    origin: Origin[mut=False],
    cache_origin: Origin[mut=False],
    //,
    dtype_: DType,
    layout: Layout,
    cache_layout: Layout,
    scale_dtype_: DType = DType.invalid,
    scale_layout: Layout = Layout(),
](MHAOperand, TrivialRegisterPassable):
    """An implementation for ragged LayoutTensor arguments to MHA kernels."""

    comptime dtype = Self.dtype_
    comptime scale_dtype = Self.scale_dtype_
    comptime page_size = 0
    comptime quantization_granularity = 0
    var buffer: LayoutTensor[Self.dtype, Self.layout, Self.origin]
    var scale_buffer: LayoutTensor[
        Self.scale_dtype, Self.scale_layout, ImmutAnyOrigin
    ]
    var cache_row_offsets: LayoutTensor[
        DType.uint32, Self.cache_layout, Self.cache_origin
    ]

    comptime device_type: AnyType = Self

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "RaggedMHAOperand"

    def __init__(
        out self,
        buffer: LayoutTensor[Self.dtype, Self.layout, Self.origin],
        cache_row_offsets: LayoutTensor[
            DType.uint32, Self.cache_layout, Self.cache_origin
        ],
    ):
        comptime assert (
            buffer.rank == 3
        ), "only support rank 3 inputs for ragged inputs."
        comptime assert (
            cache_row_offsets.rank == 1
        ), "only support rank 1 inputs for cache offsets."
        self.buffer = buffer
        self.cache_row_offsets = rebind[type_of(self.cache_row_offsets)](
            cache_row_offsets
        )
        self.scale_buffer = LayoutTensor[
            Self.scale_dtype, Self.scale_layout, ImmutAnyOrigin
        ](None)

    def __init__(
        out self,
        buffer: LayoutTensor[Self.dtype, Self.layout, Self.origin],
        scale_buffer: LayoutTensor[
            Self.scale_dtype, Self.scale_layout, ImmutAnyOrigin
        ],
        cache_row_offsets: LayoutTensor[
            DType.uint32, Self.cache_layout, Self.cache_origin
        ],
    ):
        self.buffer = buffer
        self.cache_row_offsets = rebind[type_of(self.cache_row_offsets)](
            cache_row_offsets
        )
        self.scale_buffer = scale_buffer

    @always_inline
    def block_paged_ptr[
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
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], ImmutAnyOrigin]:
        # SAFETY: Ragged operands don't support quantization; callers only
        # dereference behind comptime quantization guards.
        return UnsafePointer[
            Scalar[Self.scale_dtype], ImmutAnyOrigin
        ].unsafe_dangling()

    @always_inline
    def load_scale[
        width: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[Self.scale_dtype, width]:
        return SIMD[Self.scale_dtype, width](0)

    @always_inline
    def cache_length(self, batch_idx: Int) -> Int:
        return Int(
            self.cache_row_offsets[batch_idx + 1]
            - self.cache_row_offsets[batch_idx]
        )

    @always_inline
    def max_context_length(self) -> UInt32:
        comptime assert (
            False
        ), "For RaggedMHAOperand, max_context_length is not implemented."

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of tokens in the ragged buffer."""
        return self.buffer.dim[0]()

    @always_inline
    def row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        return self.cache_row_offsets[Int(batch_idx)][0] + start_tok_idx

    @always_inline
    def get_tma_row(self, encoded_index: Int32) -> Int32:
        """Convert an encoded sparse index to a physical TMA row.

        Non-paged operand: identity (no paging translation needed).
        """
        return encoded_index

    @always_inline
    def create_tma_tile[
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
        comptime assert (
            BK % swizzle_granularity[Self.dtype, swizzle_mode]()
        ) == 0
        var rows = self.buffer.dim[0]()  # total tokens
        comptime smem_shape = IndexList[3](BN, 1, BK)
        comptime gmem_shape = IndexList[3](UNKNOWN_VALUE, UNKNOWN_VALUE, depth)

        tma = create_split_tma[
            smem_shape,
            gmem_shape,
            swizzle_mode=swizzle_mode,
        ](ctx, self.buffer.ptr, rows, self.buffer.dim[1]())

    @always_inline
    def create_scale_tma_tile[
        BMN: Int
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            Self.scale_dtype,
            2,
            Index(1, BMN),
            Index(1, BMN),
        ],
    ) raises:
        # if per token scale, treat as 1D tensor
        comptime if Self.scale_layout.rank() == 2:
            var total_elements = self.scale_buffer.size()
            debug_assert(
                total_elements % 4 == 0,
                (
                    "Total_elements must be divisible by 4. Otherwise, the"
                    " Nvidia Driver will crash."
                ),
            )
            var scale_tensor = TileTensor(
                self.scale_buffer.ptr,
                row_major(Coord(Idx[1](), Idx(total_elements))),
            )
            return create_tensor_tile[
                Index(1, BMN),
                swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
                __desc_shape=Index(1, BMN),
            ](ctx, scale_tensor)

        # if per token per head scale, treat as 2D tensor with shape [num_heads, total_seq_len]
        elif Self.scale_layout.rank() == 3:
            comptime num_heads = Self.scale_layout.shape[0].value()
            var total_seq_len = self.buffer.dim[1]()

            var scale_tensor = TileTensor(
                self.scale_buffer.ptr,
                row_major(Coord(Idx[num_heads](), Idx(total_seq_len))),
            )

            return create_tensor_tile[
                Index(1, BMN),
                swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
                __desc_shape=Index(1, BMN),
            ](ctx, scale_tensor)

        else:
            comptime assert False, (
                "scale_layout must be 2D(per token) or 3D(per token per head)"
                " tensor."
            )

    @always_inline
    def create_ragged_tma_tile[
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
        comptime assert (
            BK % swizzle_granularity[Self.dtype, swizzle_mode]()
        ) == 0
        var rows = self.buffer.dim[0]()  # total tokens
        var num_heads = self.buffer.dim[1]()
        tma = type_of(tma).create[depth=depth](
            ctx, self.buffer.ptr, rows=rows, middle_dim=num_heads
        )

    @always_inline
    def create_rope_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int,
        padded_depth: Int,
    ](
        self,
        ctx: DeviceContext,
        out tma: SplitLastDimTMATensorTile[
            DType.bfloat16,
            IndexList[3](BN, 1, BK),
            swizzle_mode,
        ],
    ) raises:
        """Not supported for RaggedMHAOperand."""
        comptime assert (
            False
        ), "create_rope_tma_tile is not supported for RaggedMHAOperand"

    @always_inline
    def create_gather4_tma_tile[
        tile_width: Int,
        tile_stride: Int = tile_width,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        tma_dtype: DType = Self.dtype,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            tma_dtype,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[tma_dtype, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Creates a 2D TMA gather4 descriptor for this ragged operand."""
        # View the ragged buffer as a 2D matrix [total_tokens, tile_width]
        var rows = self.buffer.dim[0]()
        tma = create_tma_tile_gather4[
            tma_dtype,
            tile_height=tile_height,
            tile_width=tile_width,
            tile_stride=tile_stride,
            swizzle_mode=swizzle_mode,
            l2_promotion=l2_promotion,
        ](ctx, self.buffer.ptr.bitcast[Scalar[tma_dtype]](), rows)

    @always_inline
    def create_rope_gather4_tma_tile[
        tile_width: Int,
        padded_depth: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
        tile_height: Int = 4,
        l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    ](
        self,
        ctx: DeviceContext,
        out tma: TMATensorTile[
            DType.bfloat16,
            2,
            tile_shape=IndexList[2](
                tile_height,
                _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
            ),
            desc_shape=IndexList[2](
                1,
                _gather4_box_width[DType.bfloat16, tile_width, swizzle_mode](),
            ),
        ],
    ) raises:
        """Not supported for RaggedMHAOperand."""
        comptime assert (
            False
        ), "create_rope_gather4_tma_tile is not supported for RaggedMHAOperand"

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Scalar[DType.float32], MutAnyOrigin]:
        """Returns a dangling pointer. Ragged operands do not support
        quantization."""
        # SAFETY: Ragged operands don't support quantization; callers only
        # dereference behind comptime quantization guards.
        return UnsafePointer[
            Scalar[DType.float32], MutAnyOrigin
        ].unsafe_dangling()
