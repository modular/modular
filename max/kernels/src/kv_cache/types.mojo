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
"""
This module contains the types for the key-value cache APIs.

The module includes structs implementing several different types of
[KV caches](/glossary/ai/kv-cache).

This module defines two traits that define the roles of the different structs

- `KVCacheT`: Defines the interface for a single (key or value) cache.
- `KVCollectionT`: Defines the interface for a pair of caches (keys and values).
"""

from std.math import align_up
from std.gpu.host import DeviceContext
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    IntTuple,
    LTToTTLayout,
    Layout,
    LayoutTensor,
    RuntimeInt,
    TensorLayout,
    TileTensor,
    UNKNOWN_VALUE,
    coord,
    lt_to_tt,
)
from layout.tma_async import (
    SplitLastDimTMATensorTile,
    TMATensorTile,
    _gather4_box_width,
    create_split_tma,
    create_tma_tile_gather4,
    RaggedTMA3DTile,
)
from layout.tile_layout import RowMajorLayout, Layout as InternalLayout
from layout.coord import DynamicCoord

from std.collections import OptionalReg
from std.utils import Index, IndexList
from std.sys import size_of
from std.builtin.device_passable import DevicePassable
from std.math import ceildiv


@always_inline
def swizzle_granularity[dtype: DType, swizzle_mode: TensorMapSwizzle]() -> Int:
    comptime sg = swizzle_mode.bytes() // size_of[dtype]()
    return sg


@always_inline
def padded_depth[
    dtype: DType, swizzle_mode: TensorMapSwizzle, depth: Int
]() -> Int:
    comptime padded_depth = align_up(
        depth, swizzle_mode.bytes() // size_of[dtype]()
    )
    return padded_depth


@always_inline
def _compute_kv_cache_dynamic_shape_strides[
    dtype: DType, //, kv_cache_rank: Int, drop_list: Tuple
](blocks: TileTensor[dtype, ...]) -> Tuple[
    IndexList[kv_cache_rank],
    IndexList[kv_cache_rank],
]:
    var kv_cache_shape = IndexList[kv_cache_rank]()
    var kv_cache_strides = IndexList[kv_cache_rank]()
    var out_index = kv_cache_rank - 1
    var stride = 1

    comptime for i in reversed(range(blocks.flat_rank)):
        var dim = Int(blocks.dim[i]())

        # Skip dimensions in the drop list (kv_idx and layer_idx).
        comptime if i not in drop_list:
            kv_cache_shape[out_index] = dim
            kv_cache_strides[out_index] = stride
            out_index = out_index - 1

        stride *= dim

    return (kv_cache_shape, kv_cache_strides)


@always_inline
def _make_cache_tt[
    dtype: DType,
    ResultLayout: TensorLayout,
    rank: Int,
](
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    shape: IndexList[rank],
    strides: IndexList[rank],
) -> TileTensor[
    dtype,
    InternalLayout[
        shape_types=ResultLayout._shape_types,
        stride_types=ResultLayout._stride_types,
    ],
    MutAnyOrigin,
]:
    """Construct a TileTensor from a pointer and IndexList shape/strides.

    Static dims in ResultLayout are left at their compile-time values;
    dynamic dims are filled from the IndexList arguments.
    """
    comptime ConcLayout = InternalLayout[
        shape_types=ResultLayout._shape_types,
        stride_types=ResultLayout._stride_types,
    ]
    var shape_c = Coord[*ConcLayout.shape_types]()
    var stride_c = Coord[*ConcLayout.stride_types]()
    comptime for i in range(rank):
        comptime if not shape_c.element_types[i].is_static_value:
            shape_c[i] = rebind[shape_c.element_types[i]](
                Scalar[DType.int64](shape[i])
            )
        comptime if not stride_c.element_types[i].is_static_value:
            stride_c[i] = rebind[stride_c.element_types[i]](
                Scalar[DType.int64](strides[i])
            )
    return TileTensor[dtype, ConcLayout, MutAnyOrigin](
        ptr=ptr, layout=ConcLayout(shape_c, stride_c)
    )


struct KVCacheStaticParams(Equatable, TrivialRegisterPassable):
    var num_heads: UInt
    var head_size: UInt
    var is_mla: Bool

    def __init__(
        out self, num_heads: UInt, head_size: UInt, is_mla: Bool = False
    ):
        """
        Initialize KVCacheStaticParams.
        Args:
            num_heads (UInt): Number of attention heads.
            head_size (UInt): Size of each attention head.
            is_mla (Bool, optional): Whether to use Multi-Linear Attention (MLA) mode.
                If true, we only store k cache. If False, we store k and v cache.
                Defaults to False.
        """
        self.num_heads = num_heads
        self.head_size = head_size
        self.is_mla = is_mla


# Explicit 1D TileTensor layout that lets the compiler prove flat_rank == 1,
# bypassing the LTToTTLayout comptime alias chain where the compiler can't
# simplify Variadic.size(_Flattened[...]) to 1.
comptime _1d_tt_layout = InternalLayout[
    shape_types=Variadic.types[T=CoordLike, RuntimeInt[DType.int64]],
    stride_types=Variadic.types[T=CoordLike, ComptimeInt[1]],
]

comptime _2d_row_major_tt_layout = InternalLayout[
    shape_types=Variadic.types[
        T=CoordLike, RuntimeInt[DType.int64], RuntimeInt[DType.int64]
    ],
    stride_types=Variadic.types[
        T=CoordLike, RuntimeInt[DType.int64], ComptimeInt[1]
    ],
]


trait KVCacheT(DevicePassable, TrivialRegisterPassable):
    """Trait for different KVCache types and implementations.

    Represents a single (key or value) cache.
    """

    comptime dtype: DType
    comptime kv_params: KVCacheStaticParams
    comptime page_size_: Int
    comptime scale_dtype: DType = DType.invalid
    comptime quantization_enabled: Bool = False
    comptime quantization_granularity: Int = 1

    def cache_lengths_nd(
        self,
    ) -> TileTensor[DType.uint32, _1d_tt_layout, ImmutAnyOrigin,]:
        """Returns the cache lengths as a TileTensor."""
        ...

    def cache_length(self, batch_idx: Int) -> Int:
        """Returns the length of the cache for a given batch index."""
        ...

    def load[
        width: Int,
        output_dtype: DType = Self.dtype,
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        output_dtype, width
    ]:
        """Loads an element from the given index."""
        ...

    def store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[Self.dtype, ...],
    ):
        """Stores an element at the given index."""
        ...

    def store_scale(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        scales: SIMD[Self.scale_dtype, ...],
    ):
        """Stores the quantization scales at the given index."""
        ...

    def load_scale[
        width: Int
    ](
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[
        Self.scale_dtype, width
    ]:
        """Loads the quantization scales from the given index."""
        ...

    def load_quantized[
        width: Int
    ](
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[
        Self.dtype, width
    ]:
        """Loads a quantized element from the given index."""
        ...

    def empty_cache(self) -> Bool:
        """Returns true if the cache_lengths for all requests is 0,
        false otherwise."""
        ...

    def max_prompt_length(self) -> UInt32:
        """Returns the maximum sequence length across all batches of the current
        request."""
        ...

    def max_context_length(self) -> UInt32:
        """Returns the maximum cache length used across all batches of the
        current request."""
        ...

    @always_inline
    def block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        """Returns a pointer to the KVCache block at the given index.

        Paged KVCache implementations must have a block_size which is a multiple of the
        and greater than the layout's first dimension.
        """
        ...

    @always_inline
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]:
        """Returns a pointer to the scales block at the requested indices."""
        ...

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]:
        """Returns the base pointer to the scales tensor.

        For PagedKVCache with quantization enabled, this returns the raw
        base pointer of the scales TileTensor. For caches without
        quantization, returns a null pointer.
        """
        ...

    @staticmethod
    def max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
        ...

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of virtual rows in this KV cache view.

        For paged caches this accounts for the paging stride:
        ``(total_blocks - 1) * stride + page_size``.
        """
        ...

    @always_inline
    def row_idx(self, batch_idx: UInt32, start_tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        ...

    @always_inline
    def create_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int = padded_depth[
            Self.dtype, swizzle_mode, Int(Self.kv_params.head_size)
        ](),
    ](self, ctx: DeviceContext) raises -> SplitLastDimTMATensorTile[
        Self.dtype,
        IndexList[3](BN, 1, BK),
        swizzle_mode,
    ]:
        """Creates a TMA tile for this KV cache.
        This is useful for `k-major` MMA operations where we don't
        need to mask any extra rows."""
        ...

    @always_inline
    def create_ragged_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int = padded_depth[
            Self.dtype, swizzle_mode, Int(Self.kv_params.head_size)
        ](),
    ](self, ctx: DeviceContext) raises -> RaggedTMA3DTile[
        Self.dtype,
        swizzle_mode,
        BM=BN,
        BN=BK,
    ]:
        """Creates a TMA tile for this KV cache.
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
        """Creates a BF16 TMA tile for the rope portion of the KV cache.

        For the per-tensor rope-aware layout, each token row in the KV cache is
        stored as `padded_depth` FP8 bytes (content) followed by `BK` BF16
        elements (rope). This method returns a TMA descriptor that points at
        the rope data starting at byte offset `padded_depth` within each row,
        reinterpreted as BF16.
        """
        ...

    @always_inline
    def create_gather4_tma_tile[
        row_width: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        Self.dtype,
        2,
        tile_shape=IndexList[2](
            4,
            _gather4_box_width[Self.dtype, row_width, swizzle_mode](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[Self.dtype, row_width, swizzle_mode](),
        ),
    ]:
        """Creates a 2D TMA gather4 descriptor for this KV cache.

        The descriptor views the KV cache as a flat 2D matrix of
        ``[num_kv_rows, row_width]`` and is configured for gather4 operations
        that load 4 non-contiguous rows per TMA instruction. The box width
        is derived from the swizzle mode; for SWIZZLE_NONE it equals
        ``row_width``.

        Parameters:
            row_width: Number of elements per row (innermost dimension).
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
                Defaults to SWIZZLE_NONE.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.

        Returns:
            A TMATensorTile with box width derived from the swizzle mode.
        """
        ...


struct ContinuousBatchingKVCache[
    dtype_: DType,
    kv_params_: KVCacheStaticParams,
](KVCacheT, TrivialRegisterPassable):
    """Wrapper for the ContinuousKVCache of a given layer in the transformer
    model.

    Parameters:
        dtype_: The dtype of the kv-cache.
        kv_params_: The kv-cache static parameters.

    This abstracts the Pointer indirection for accessing the ContinuousKVCache
    for a given batch entry.

    THIS IS THE TYPE THAT IS PASSED TO KV PROJECTION AND FLASH ATTENTION
    KERNELS.
    """

    comptime dtype = Self.dtype_
    comptime kv_params = Self.kv_params_
    comptime page_size_ = 0
    # Note: quantization not supported for `ContinuousBatchingKVCache`.
    comptime scale_dtype = DType.float32
    comptime quantization_granularity = 1
    # Shape is [num_blocks, max_seq_len, num_heads, head_size].
    comptime blocks_shape = IntTuple(
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
        Int(Self.kv_params.num_heads),
        Int(Self.kv_params.head_size),
    )
    comptime blocks_layout = Layout.row_major(Self.blocks_shape)

    comptime blocks_tt_layout = LTToTTLayout[Self.blocks_layout]
    comptime blocks_tt_type = TileTensor[
        Self.dtype, Self.blocks_tt_layout, MutAnyOrigin
    ]

    comptime cache_lengths_tt_layout = _1d_tt_layout
    comptime cache_lengths_tt_type = TileTensor[
        DType.uint32, Self.cache_lengths_tt_layout, ImmutAnyOrigin
    ]

    comptime lookup_table_tt_layout = _1d_tt_layout
    comptime lookup_table_tt_type = TileTensor[
        DType.uint32, Self.lookup_table_tt_layout, ImmutAnyOrigin
    ]

    var blocks: Self.blocks_tt_type
    var cache_lengths: Self.cache_lengths_tt_type
    var lookup_table: Self.lookup_table_tt_type

    # The length of the longest sequence in the current request.
    # This length only considers tokens not in the KVCache.
    var max_seq_length: UInt32

    # The length of the longest context in the current request.
    # This is effectively:
    #   max(cache_lengths[i] + prompt_lengths[i] for i in range(batch_size)
    var max_cache_length: UInt32

    comptime device_type: AnyType = Self

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "ContinuousBatchingKVCache"

    @always_inline
    def _get_idx_tuple(
        self, block_idx: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int
    ) -> DynamicCoord[DType.int64, 4]:
        assert (
            UInt(head_idx) < Self.kv_params.num_heads
        ), "KVCache head_idx out of range"
        assert (
            UInt(head_dim_idx) < Self.kv_params.head_size
        ), "KVCache head_dim_idx is out of range"
        assert tok_idx < Int(
            self.blocks.dim[1]()
        ), "KVCache tok_idx out of range"
        return coord[DType.int64](
            Tuple(block_idx, tok_idx, head_idx, head_dim_idx)
        )

    @staticmethod
    def max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
        return -1

    def __init__(
        out self,
        blocks: Self.blocks_tt_type,
        cache_lengths: Self.cache_lengths_tt_type,
        lookup_table: Self.lookup_table_tt_type,
        max_seq_length: UInt32,
        max_cache_length: UInt32,
    ):
        comptime assert (
            not self.quantization_enabled
        ), "ContinuousBatchingKVCache does not support quantization"
        assert Int(blocks.dim[2]()) == Int(
            Self.kv_params.num_heads
        ), "blocks.dim[2]() must be equal to kv_params.num_heads"
        assert Int(blocks.dim[3]()) == Int(
            Self.kv_params.head_size
        ), "blocks.dim[3]() must be equal to kv_params.head_size"

        self.blocks = blocks
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length

    @always_inline
    def _batch_size(self) -> Int:
        return Int(self.cache_lengths.dim[0]())

    @always_inline
    def cache_lengths_nd(self) -> Self.cache_lengths_tt_type:
        return self.cache_lengths

    @always_inline
    def cache_length(self, batch_idx: Int) -> Int:
        assert (
            batch_idx < self._batch_size()
        ), "KVCache batch_idx is out of bounds"
        return Int(self.cache_lengths[batch_idx])

    @always_inline
    def load[
        width: Int,
        output_dtype: DType = Self.dtype,
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        output_dtype, width
    ]:
        assert bs < self._batch_size(), "KVCache::load batch_size out of range"

        var block_idx = self.lookup_table[bs]
        var idx = self._get_idx_tuple(
            Int(block_idx), head_idx, tok_idx, head_dim_idx
        )
        # Bypass TileTensor.load's `where` constraint by using ptr directly.
        return self.blocks.load[width=width](idx).cast[output_dtype]()

    @always_inline
    def store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[Self.dtype, ...],
    ):
        assert bs < self._batch_size(), "KVCache::store batch_size out of range"
        var block_idx = self.lookup_table[bs]
        var idx = self._get_idx_tuple(
            Int(block_idx), head_idx, tok_idx, head_dim_idx
        )
        # Bypass TileTensor.store's `where` constraint by using ptr directly.
        self.blocks.store(idx, val)

    @always_inline
    def load_scale[
        width: Int
    ](
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[
        Self.scale_dtype, width
    ]:
        """Loads a quantization scale from the given index.

        Note: ContinuousBatchingKVCache does not support KVCache quantization.
        """
        return SIMD[Self.scale_dtype, width](0)

    @always_inline
    def store_scale(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        scales: SIMD[Self.scale_dtype, ...],
    ):
        """Stores the quantization scales at the given index.

        Note: ContinuousBatchingKVCache does not support KVCache quantization.
        """
        ...

    @always_inline
    def load_quantized[
        width: Int
    ](
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[
        Self.dtype, width
    ]:
        """Loads a quantized element from the given index.

        Note: ContinuousBatchingKVCache does not support KVCache quantization.
        """
        return SIMD[Self.dtype, width](0)

    def empty_cache(self) -> Bool:
        """Returns true if the cache_lengths for all requests is 0,
        false otherwise."""
        return self.max_cache_length == 0

    def max_prompt_length(self) -> UInt32:
        """Returns the maximum sequence length across all batches of the current
        request."""
        return self.max_seq_length

    def max_context_length(self) -> UInt32:
        """Returns the maximum cache length used across all batches of the
        current request."""
        return self.max_cache_length

    @always_inline
    def _stride(self) -> UInt32:
        return UInt32(self.blocks.layout.stride[0]().value()) // UInt32(
            self.kv_params.num_heads * self.kv_params.head_size
        )

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of virtual rows in this KV cache view."""
        var total_blocks = self.blocks.dim[0]()
        return Int(
            UInt32(total_blocks - 1) * self._stride()
            + UInt32(self.blocks.dim[1]())
        )

    @always_inline
    def row_idx(self, batch_idx: UInt32, tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        block_idx = self.lookup_table[Int(batch_idx)]
        return block_idx * self._stride() + tok_idx

    @always_inline
    def create_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int = padded_depth[
            Self.dtype, swizzle_mode, Int(Self.kv_params.head_size)
        ](),
    ](self, ctx: DeviceContext) raises -> SplitLastDimTMATensorTile[
        Self.dtype,
        IndexList[3](BN, 1, BK),
        swizzle_mode,
    ]:
        """Creates a TMA tile for this KV cache."""
        comptime assert (
            BK % swizzle_granularity[Self.dtype, swizzle_mode]()
        ) == 0, "BK must be a multiple of swizzle granularity"
        # The continuous cache is laid out as [num_blocks, num_layers, seq_len, num_heads, head_size]
        # We create a view of the data as a flattened 2D tensor
        var total_blocks = Int(self.blocks.dim[0]())
        # An axis's size is 1 + maximum valid idx
        # Idx calc is:
        # block_idx * self._stride() + tok_idx
        # max values
        # (total_blocks - 1) * self._stride() + self.blocks.dim[1]() - 1
        # yields number of rows:
        # (total_blocks - 1) * self._stride() + self.blocks.dim[1]()
        var rows = UInt32(total_blocks - 1) * self._stride() + UInt32(
            self.blocks.dim[1]()
        )

        comptime smem_dim = IndexList[3](BN, 1, BK)
        comptime gmem_dim = IndexList[3](
            UNKNOWN_VALUE,
            Int(Self.kv_params.num_heads),
            Int(Self.kv_params.head_size),
        )
        return create_split_tma[smem_dim, gmem_dim, swizzle_mode](
            ctx, self.blocks.ptr, Int(rows)
        )

    @always_inline
    def create_gather4_tma_tile[
        row_width: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        Self.dtype,
        2,
        tile_shape=IndexList[2](
            4,
            _gather4_box_width[Self.dtype, row_width, swizzle_mode](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[Self.dtype, row_width, swizzle_mode](),
        ),
    ]:
        """Creates a 2D TMA gather4 descriptor for this KV cache.

        The descriptor views the KV cache as a flat 2D matrix of
        ``[num_kv_rows, row_width]`` and is configured for gather4 operations
        that load 4 non-contiguous rows per TMA instruction. The box width
        is derived from the swizzle mode; for SWIZZLE_NONE it equals
        ``row_width``.

        Parameters:
            row_width: Number of elements per row (innermost dimension).
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
                Defaults to SWIZZLE_NONE.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.

        Returns:
            A TMATensorTile with box width derived from the swizzle mode.
        """
        return create_tma_tile_gather4[Self.dtype, row_width, swizzle_mode](
            ctx, self.blocks.ptr, self.num_kv_rows()
        )

    @always_inline
    def create_ragged_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int = padded_depth[
            Self.dtype, swizzle_mode, Int(Self.kv_params.head_size)
        ](),
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
        ) == 0, "BK must be a multiple of swizzle granularity"
        var total_blocks = Int(self.blocks.dim[0]())
        var rows = UInt32(total_blocks - 1) * self._stride() + UInt32(
            self.blocks.dim[1]()
        )
        tma = type_of(tma).create[depth=Int(Self.kv_params.head_size)](
            ctx,
            self.blocks.ptr,
            rows=Int(rows),
            middle_dim=Int(Self.kv_params.num_heads),
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
        """Not supported for ContinuousBatchingKVCache."""
        comptime assert (
            False
        ), "create_rope_tma_tile is not supported for ContinuousBatchingKVCache"

    @always_inline
    def block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        var block_idx = Int(self.lookup_table[batch_idx])
        var full_block_idx = self._get_idx_tuple(
            block_idx, head_idx, start_tok_idx, head_dim_idx
        )
        var offset_ptr = self.blocks.ptr + Int(
            self.blocks.layout(full_block_idx)
        )
        return offset_ptr

    @always_inline
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]:
        """Returns a pointer to the scales block at the requested indices.

        Note: ContinuousBatchingKVCache does not support KVCache quantization.
        This function returns a NULL pointer.
        """
        return UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]()

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]:
        """Returns a null pointer. ContinuousBatchingKVCache does not support
        quantization."""
        return UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]()


struct PagedKVCache[
    dtype_: DType,
    kv_params_: KVCacheStaticParams,
    page_size: Int,
    scale_dtype_: DType = DType.invalid,
    quantization_granularity_: Int = 1,
](KVCacheT, TrivialRegisterPassable):
    """The PagedKVCache is a wrapper around the KVCache blocks for a given layer.
    It is used to access the KVCache blocks for PagedAttention.

    Note: This struct represents a 4D view of a 6D `PagedKVCacheCollection`
    tensor. The compile-time layout has `UNKNOWN_VALUE` for stride[0] because
    the actual stride depends on `num_layers` from the parent tensor, which is
    only known at runtime. This ensures offset calculations use the correct
    runtime strides rather than incorrect compile-time values.

    Parameters:
        dtype_: The dtype of the kv-cache.
        kv_params_: The kv-cache static parameters.
        page_size: The size of the page.
        scale_dtype_: Dtype of the quantization scales (if quantization enabled).
        quantization_granularity_:  Block size used for quantization (e.g. 128).
    """

    comptime dtype = Self.dtype_
    comptime kv_params = Self.kv_params_
    comptime page_size_ = Self.page_size
    comptime scale_dtype = Self.scale_dtype_
    comptime quantization_enabled = Self.scale_dtype_ != DType.invalid
    comptime quantization_granularity = Self.quantization_granularity_

    # Shape is [total_num_blocks, page_size, num_heads, head_size].
    # This tensor is a view of a 6D parent tensor with shape
    # [num_blocks, 2, num_layers, page_size, num_heads, head_size].
    # The outer stride depends on num_layers (unknown), so stride[0] must be
    # UNKNOWN_VALUE to ensure we use runtime strides for offset calculations.
    comptime blocks_shape = IntTuple(
        UNKNOWN_VALUE,
        Self.page_size,
        Int(Self.kv_params.num_heads),
        Int(Self.kv_params.head_size),
    )
    comptime blocks_strides = IntTuple(
        # Runtime value: 2 * num_layers * page_size * num_heads * head_size
        UNKNOWN_VALUE,
        Int(Self.kv_params.num_heads) * Int(Self.kv_params.head_size),
        Int(Self.kv_params.head_size),
        1,
    )
    comptime blocks_layout = Layout(Self.blocks_shape, Self.blocks_strides)

    # TileTensor layout for blocks.
    comptime blocks_tt_layout = LTToTTLayout[Self.blocks_layout]
    comptime blocks_tt_type = TileTensor[
        Self.dtype, Self.blocks_tt_layout, MutAnyOrigin
    ]

    comptime cache_lengths_tt_layout = _1d_tt_layout
    comptime cache_lengths_tt_type = TileTensor[
        DType.uint32, Self.cache_lengths_tt_layout, ImmutAnyOrigin
    ]

    comptime lookup_table_tt_layout = _2d_row_major_tt_layout
    comptime lookup_table_tt_type = TileTensor[
        DType.uint32, Self.lookup_table_tt_layout, ImmutAnyOrigin
    ]

    var blocks: Self.blocks_tt_type
    var cache_lengths: Self.cache_lengths_tt_type
    var lookup_table: Self.lookup_table_tt_type

    # The length of the longest sequence in the current request.
    # This length only considers tokens not in the KVCache.
    var max_seq_length: UInt32

    # The length of the longest context in the current request.
    # This is effectively:
    #   max(cache_lengths[i] + prompt_lengths[i] for i in range(batch_size)
    var max_cache_length: UInt32

    # Number of quantization scale values per token.
    comptime head_dim_granularity = ceildiv(
        Int(Self.kv_params.head_size),
        Self.quantization_granularity,
    )
    comptime scales_tt_layout = RowMajorLayout[
        RuntimeInt[DType.int64],
        ComptimeInt[Self.page_size],
        ComptimeInt[Int(Self.kv_params.num_heads)],
        ComptimeInt[Self.head_dim_granularity],
    ]
    comptime scales_tt_type = TileTensor[
        Self.scale_dtype, Self.scales_tt_layout, MutAnyOrigin
    ]

    # KV Cache quantization scales
    var scales: OptionalReg[Self.scales_tt_type]

    comptime device_type: AnyType = Self

    def _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "PagedKVCache"

    def __init__(
        out self,
        blocks: Self.blocks_tt_type,
        cache_lengths: Self.cache_lengths_tt_type,
        lookup_table: Self.lookup_table_tt_type,
        max_seq_length: UInt32,
        max_cache_length: UInt32,
        scales: OptionalReg[Self.scales_tt_type] = None,
    ):
        assert (
            Int(blocks.dim[1]()) == Self.page_size
        ), "blocks.dim[1]() must be equal to page_size"
        assert Int(blocks.dim[2]()) == Int(
            Self.kv_params.num_heads
        ), "blocks.dim[2]() must be equal to kv_params.num_heads"
        assert Int(blocks.dim[3]()) == Int(
            Self.kv_params.head_size
        ), "blocks.dim[3]() must be equal to kv_params.head_size"

        self.blocks = blocks
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.scales = scales

    @staticmethod
    def max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
        return Self.page_size

    @always_inline
    def cache_lengths_nd(self) -> Self.cache_lengths_tt_type:
        return self.cache_lengths

    def cache_length(self, batch_idx: Int) -> Int:
        """Returns the length of the cache for a given batch index."""
        return Int(self.cache_lengths[batch_idx])

    @always_inline
    def _stride(self) -> UInt32:
        return UInt32(self.blocks.layout.stride[0]().value()) // UInt32(
            self.kv_params.num_heads * self.kv_params.head_size
        )

    @always_inline
    def num_kv_rows(self) -> Int:
        """Returns the total number of virtual rows in this KV cache view."""
        var total_blocks = self.blocks.dim[0]()
        return Int(
            UInt32(total_blocks - 1) * self._stride() + UInt32(Self.page_size)
        )

    @always_inline
    def row_idx(self, batch_idx: UInt32, tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        var lut_block_index, tok_in_block_idx = divmod(
            Int(tok_idx), Self.page_size
        )
        assert tok_in_block_idx < Int(
            self.blocks.dim[1]()
        ), "KVCache tok_idx out of range"

        assert batch_idx < UInt32(
            self.cache_lengths.num_elements()
        ), "batch_idx is oob"
        debug_assert(
            lut_block_index < Int(self.blocks.dim[0]()),
            "block_idx is OOB. Attempted to access block index ",
            lut_block_index,
            " with num_blocks ",
            Int(self.blocks.dim[0]()),
        )
        block_idx = self.lookup_table[Int(batch_idx), lut_block_index]
        # alias row_stride = Int(num_heads * head_size * Self.collection_size)
        return block_idx * self._stride() + UInt32(tok_in_block_idx)

    @always_inline
    def create_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int = padded_depth[
            Self.dtype, swizzle_mode, Int(Self.kv_params.head_size)
        ](),
    ](self, ctx: DeviceContext) raises -> SplitLastDimTMATensorTile[
        Self.dtype,
        IndexList[3](BN, 1, BK),
        swizzle_mode,
    ]:
        """Creates a TMA tile for this KV cache."""
        comptime assert (
            BK % swizzle_granularity[Self.dtype, swizzle_mode]()
        ) == 0, "BK must be a multiple of swizzle granularity"
        # Paged cache collection is (where `$idx` means subsetting that idx):
        # [total_num_blocks, $kv_idx, $layer_idx, page_size, num_heads, head_size]
        #
        # An axis's size is 1 + maximum valid idx
        # Idx calc is:
        # block_idx * self._stride() + tok_in_block_idx
        # max values
        # (total_blocks - 1) * self._stride() + Self.page_size - 1
        # yields number of rows:
        # (total_blocks - 1) * self._stride() + Self.page_size
        #
        # Create a view that accounts for the paged layout
        var total_blocks = Int(self.blocks.dim[0]())
        var rows = UInt32(total_blocks - 1) * self._stride() + UInt32(
            Self.page_size
        )
        comptime smem_dim = IndexList[3](BN, 1, BK)
        comptime gmem_dim = IndexList[3](
            UNKNOWN_VALUE,
            Int(Self.kv_params.num_heads),
            Int(Self.kv_params.head_size),
        )
        return create_split_tma[smem_dim, gmem_dim, swizzle_mode](
            ctx, self.blocks.ptr, Int(rows)
        )

    @always_inline
    def create_gather4_tma_tile[
        row_width: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    ](self, ctx: DeviceContext) raises -> TMATensorTile[
        Self.dtype,
        2,
        tile_shape=IndexList[2](
            4,
            _gather4_box_width[Self.dtype, row_width, swizzle_mode](),
        ),
        desc_shape=IndexList[2](
            1,
            _gather4_box_width[Self.dtype, row_width, swizzle_mode](),
        ),
    ]:
        """Creates a 2D TMA gather4 descriptor for this KV cache.

        The descriptor views the KV cache as a flat 2D matrix of
        ``[num_kv_rows, row_width]`` and is configured for gather4 operations
        that load 4 non-contiguous rows per TMA instruction. The box width
        is derived from the swizzle mode; for SWIZZLE_NONE it equals
        ``row_width``.

        Parameters:
            row_width: Number of elements per row (innermost dimension).
            swizzle_mode: TMA swizzle mode for shared memory access pattern.
                Defaults to SWIZZLE_NONE.

        Args:
            ctx: The CUDA device context used to create the TMA descriptor.

        Returns:
            A TMATensorTile with box width derived from the swizzle mode.
        """
        return create_tma_tile_gather4[Self.dtype, row_width, swizzle_mode](
            ctx, self.blocks.ptr, self.num_kv_rows()
        )

    @always_inline
    def create_ragged_tma_tile[
        swizzle_mode: TensorMapSwizzle,
        *,
        BN: Int,
        BK: Int = padded_depth[
            Self.dtype, swizzle_mode, Int(Self.kv_params.head_size)
        ](),
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
        ) == 0, "BK must be a multiple of swizzle granularity"
        var total_blocks = Int(self.blocks.dim[0]())
        var rows = UInt32(total_blocks - 1) * self._stride() + UInt32(
            Self.page_size
        )
        tma = type_of(tma).create[depth=Int(Self.kv_params.head_size)](
            ctx,
            self.blocks.ptr,
            rows=Int(rows),
            middle_dim=Int(Self.kv_params.num_heads),
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
        """Creates a BF16 TMA tile for the rope portion of the per-tensor rope-aware KV cache.

        In the per-tensor rope-aware layout each token row is:
          `padded_depth` FP8 bytes (content) | `BK` BF16 elements (rope)
        Total row bytes = padded_depth + BK * 2.

        The TMA descriptor points at the rope data by offsetting `blocks.ptr`
        by `padded_depth` bytes, then reinterpreting as BF16.  The global
        memory stride dimension (last dim of gmem_shape) is the total row size
        expressed in BF16 units: (padded_depth + BK * 2) // 2.
        """
        comptime assert (
            BK % swizzle_granularity[DType.bfloat16, swizzle_mode]()
        ) == 0, "BK must be a multiple of swizzle granularity for BF16"
        # Compute the total row width in BF16 elements:
        #   padded_depth FP8 bytes + BK BF16 elements
        #   = (padded_depth + BK * 2) bytes total
        #   = (padded_depth + BK * 2) // 2 BF16 elements per row
        comptime bf16_row_stride = (padded_depth + BK * 2) // 2

        var total_blocks = self.blocks.dim[0]()
        var rows = UInt32(total_blocks - 1) * self._stride() + UInt32(
            Self.page_size
        )
        # Offset past the FP8 content to reach the BF16 rope data,
        # then reinterpret the pointer as BF16.
        var rope_ptr = (self.blocks.ptr + padded_depth).bitcast[
            Scalar[DType.bfloat16]
        ]()
        comptime smem_dim = IndexList[3](BN, 1, BK)
        comptime gmem_dim = IndexList[3](
            UNKNOWN_VALUE,
            Int(Self.kv_params.num_heads),
            bf16_row_stride,
        )
        tma = create_split_tma[smem_dim, gmem_dim, swizzle_mode](
            ctx, rope_ptr, Int(rows)
        )

    @always_inline
    def _get_idx(
        self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int
    ) -> DynamicCoord[DType.int64, 4]:
        debug_assert(
            UInt(head_idx) < Self.kv_params.num_heads,
            "KVCache head_idx out of range (",
            head_idx,
            ")",
        )
        assert (
            UInt(head_dim_idx) < Self.kv_params.head_size
        ), "KVCache head_dim_idx is out of range"

        var lut_block_index, tok_in_block_idx = divmod(tok_idx, self.page_size)

        assert tok_in_block_idx < Int(
            self.blocks.dim[1]()
        ), "KVCache tok_idx out of range"

        assert bs < self.cache_lengths.num_elements(), "batch_idx is oob"
        debug_assert(
            lut_block_index < Int(self.blocks.dim[0]()),
            "block_idx is OOB. Attempted to access block index ",
            lut_block_index,
            " with num_blocks ",
            Int(self.blocks.dim[0]()),
        )
        block_idx = Int(self.lookup_table[bs, lut_block_index])
        return coord[DType.int64](
            Tuple(block_idx, tok_in_block_idx, head_idx, head_dim_idx)
        )

    @always_inline
    def _get_scale_idx(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
    ) -> DynamicCoord[DType.int64, 4]:
        debug_assert(
            UInt(head_idx) < Self.kv_params.num_heads,
            "KVCache head_idx out of range (",
            head_idx,
            ")",
        )

        var lut_block_index, tok_in_block_idx = divmod(tok_idx, self.page_size)

        assert tok_in_block_idx < Int(
            self.blocks.dim[1]()
        ), "KVCache tok_idx out of range"

        assert bs < self.cache_lengths.num_elements(), "batch_idx is oob"
        debug_assert(
            lut_block_index < Int(self.blocks.dim[0]()),
            "block_idx is OOB. Attempted to access block index ",
            lut_block_index,
            " with num_blocks ",
            Int(self.blocks.dim[0]()),
        )

        block_idx = Int(self.lookup_table[bs, lut_block_index])
        var head_dim_granularity = ceildiv(
            head_dim_idx,
            Self.quantization_granularity,
        )
        return coord[DType.int64](
            Tuple(
                block_idx,
                tok_in_block_idx,
                head_idx,
                head_dim_granularity,
            )
        )

    @always_inline
    def load[
        width: Int,
        output_dtype: DType = Self.dtype,
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        output_dtype, width
    ]:
        """Loads an element from the given index."""

        comptime if Self.quantization_enabled:
            comptime assert output_dtype != Self.dtype, (
                "Output type should not be FP8 when KVCache quantization is"
                " disabled"
            )

        var idx = self._get_idx(bs, head_idx, tok_idx, head_dim_idx)

        # Bypass TileTensor.load's `where` constraint by using ptr directly.
        comptime if Self.quantization_enabled:
            var quantized_val = self.blocks.load[width=width](idx)
            var scale = self.load_scale[width=1](
                bs, head_idx, tok_idx, head_dim_idx
            )
            var dequantized = quantized_val.cast[Self.scale_dtype]() * scale
            return dequantized.cast[output_dtype]()
        else:
            return self.blocks.load[width=width](idx).cast[output_dtype]()

    @always_inline
    def store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[Self.dtype, ...],
    ):
        """Stores an element at the given index."""
        var idx = self._get_idx(bs, head_idx, tok_idx, head_dim_idx)
        # Bypass TileTensor.store's `where` constraint by using ptr directly.
        self.blocks.store(idx, val)

    @always_inline
    def load_scale[
        width: Int
    ](
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[
        Self.scale_dtype, width
    ]:
        """Loads a quantization scale from the given index."""
        comptime assert (
            Self.quantization_enabled
        ), "Scales only exist for quantized KVCache"
        comptime assert (
            Self.scale_dtype != DType.invalid
        ), "Invalid scale data type"
        assert (
            self.scales is not None
        ), "Scales missing, yet KVCache quantization enabled"
        var idx = self._get_scale_idx(bs, head_idx, tok_idx, head_dim_idx)
        # Bypass TileTensor.load's `where` constraint by using ptr directly.
        return self.scales.value().load[width=width](idx)

    @always_inline
    def store_scale(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        scales: SIMD[Self.scale_dtype, ...],
    ):
        """Stores the quantization scales at the given index."""

        comptime if Self.quantization_enabled:
            comptime assert (
                Self.scale_dtype != DType.invalid
            ), "Valid quantization scale data type needed"

        var scale_idx = self._get_scale_idx(bs, head_idx, tok_idx, head_dim_idx)
        # Bypass TileTensor.store's `where` constraint by using ptr directly.
        self.scales.value().store(scale_idx, scales)

    @always_inline
    def load_quantized[
        width: Int
    ](
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
    ) -> SIMD[
        Self.dtype, width
    ]:
        """Loads a quantized element from the given index."""
        comptime assert Self.quantization_enabled, (
            "Output type should not be quantized when KVCache quantization is"
            " disabled"
        )
        var idx = self._get_idx(bs, head_idx, tok_idx, head_dim_idx)
        # Bypass TileTensor.load's `where` constraint by using ptr directly.
        return self.blocks.load[width=width](idx)

    def empty_cache(self) -> Bool:
        """Returns true if the cache_lengths for all requests is 0,
        false otherwise."""
        return self.max_cache_length == 0

    def max_prompt_length(self) -> UInt32:
        """Returns the maximum sequence length across all batches of the current
        request."""
        return self.max_seq_length

    def max_context_length(self) -> UInt32:
        """Returns the maximum cache length used across all batches of the
        current request."""
        return self.max_cache_length

    @always_inline
    def block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        comptime assert (
            tile_size <= Self.page_size and Self.page_size % tile_size == 0
        ), (
            "Invalid tile size for PagedKVCache. tile_size must be less"
            " than or equal to the page size and divisible by the page size"
        )

        var full_block_idx = self._get_idx(
            batch_idx, head_idx, start_tok_idx, head_dim_idx
        )

        var ptr = self.blocks.ptr + Int(self.blocks.layout(full_block_idx))
        return ptr

    @always_inline
    def scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]:
        """Returns a pointer to the scales block at the requested indices."""
        comptime assert (
            self.quantization_enabled
        ), "Quantization must be enabled to request scales block"
        var full_scale_block_idx = self._get_scale_idx(
            batch_idx, head_idx, start_tok_idx, head_dim_idx
        )
        assert self.scales is not None, "Quantization scale factors not set."
        var scales_block = self.scales.value()

        var scales_ptr = scales_block.ptr + Int(
            scales_block.layout(full_scale_block_idx)
        )
        return scales_ptr

    @always_inline
    def scales_raw_ptr(
        self,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]:
        """Returns the base pointer to the scales tensor, or null if scales
        are not set."""

        comptime if Self.quantization_enabled:
            return self.scales.value().ptr
        return UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]()


trait KVCollectionT(ImplicitlyCopyable):
    """Trait for a pair of caches (keys and values)."""

    comptime CacheType: KVCacheT
    comptime name_str: StaticString
    comptime dtype: DType
    comptime kv_params: KVCacheStaticParams

    def get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        ...

    def get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        ...

    def cache_length(self, bs_idx: Int) -> Int:
        ...


struct ContinuousBatchingKVCacheCollection[
    dtype_: DType,
    kv_params_: KVCacheStaticParams,
](KVCollectionT):
    """This is a "view" of the cache for the given sequences
    in the batch.

    Parameters:
        dtype_: The dtype of the kv-cache.
        kv_params_: The kv-cache static parameters.

    This object does not own the underlying buffers in k_cache and v_cache,
    it's borrowing them from the BlockWrappers in our KVCacheManager.
    """

    comptime name_str = "continuous_batching"
    comptime dtype = Self.dtype_
    comptime kv_params = Self.kv_params_
    comptime CacheType = ContinuousBatchingKVCache[Self.dtype, Self.kv_params]
    comptime scale_dtype: DType = DType.invalid

    # Shape is [num_blocks, 2, num_layers, max_seq_len, num_heads, head_size].
    comptime blocks_shape = IntTuple(
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
        Int(Self.kv_params.num_heads),
        Int(Self.kv_params.head_size),
    )
    comptime blocks_layout = Layout.row_major(Self.blocks_shape)
    comptime blocks_tt_layout = LTToTTLayout[Self.blocks_layout]
    comptime blocks_tt_type = TileTensor[
        Self.dtype, Self.blocks_tt_layout, MutAnyOrigin
    ]

    var blocks: Self.blocks_tt_type
    var cache_lengths: Self.CacheType.cache_lengths_tt_type
    var lookup_table: Self.CacheType.lookup_table_tt_type
    var max_seq_length: UInt32
    var max_cache_length: UInt32
    var kv_cache_dynamic_shape: IndexList[4]
    var kv_cache_dynamic_strides: IndexList[4]

    def __init__(
        out self,
        blocks: LayoutTensor[Self.dtype, Layout.row_major[6](), MutAnyOrigin],
        cache_lengths: LayoutTensor[
            DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ],
        lookup_table: LayoutTensor[
            DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
        scales: OptionalReg[
            LayoutTensor[Self.scale_dtype, Layout.row_major[6](), MutAnyOrigin]
        ] = None,
    ):
        """Construct from LayoutTensor params (MOGG boundary)."""
        comptime assert blocks.rank == 6
        self.blocks = lt_to_tt[ResultLayout=Self.blocks_tt_layout](blocks)
        self.cache_lengths = lt_to_tt[
            ResultLayout=Self.CacheType.cache_lengths_tt_layout
        ](cache_lengths)
        self.lookup_table = lt_to_tt[
            ResultLayout=Self.CacheType.lookup_table_tt_layout
        ](lookup_table)
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.kv_cache_dynamic_shape, self.kv_cache_dynamic_strides = (
            _compute_kv_cache_dynamic_shape_strides[4, (1, 2)](self.blocks)
        )

    def __init__(
        out self,
        blocks: Self.blocks_tt_type,
        cache_lengths: Self.CacheType.cache_lengths_tt_type,
        lookup_table: Self.CacheType.lookup_table_tt_type,
        max_seq_length: UInt32,
        max_cache_length: UInt32,
    ):
        """Construct from TileTensor fields directly."""
        self.blocks = blocks
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.kv_cache_dynamic_shape, self.kv_cache_dynamic_strides = (
            _compute_kv_cache_dynamic_shape_strides[4, (1, 2)](self.blocks)
        )

    @always_inline
    def get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        return self._get_cache[0](layer_idx)

    @always_inline
    def get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        return self._get_cache[1](layer_idx)

    @always_inline
    def _get_cache[kv_idx: Int](self, layer_idx: Int) -> Self.CacheType:
        assert (
            kv_idx == 0 or self.blocks.dim[1]() > 1
        ), "invalid kv_idx for MLA cache"
        var offset = Int(
            self.blocks.layout(
                coord[DType.int64](Tuple(0, kv_idx, layer_idx, 0, 0, 0))
            )
        )
        return self.CacheType(
            _make_cache_tt[
                Self.CacheType.dtype,
                Self.CacheType.blocks_tt_layout,
                4,
            ](
                self.blocks.ptr + offset,
                self.kv_cache_dynamic_shape,
                self.kv_cache_dynamic_strides,
            ),
            self.cache_lengths,
            self.lookup_table,
            self.max_seq_length,
            self.max_cache_length,
        )

    def cache_length(self, bs_idx: Int) -> Int:
        return Int(self.cache_lengths[bs_idx])


struct PagedKVCacheCollection[
    dtype_: DType,
    kv_params_: KVCacheStaticParams,
    page_size: Int,
    scale_dtype_: DType = DType.invalid,
    quantization_granularity_: Int = 1,
](KVCollectionT):
    comptime name_str = "paged"
    comptime dtype = Self.dtype_
    comptime kv_params = Self.kv_params_
    comptime scale_dtype = Self.scale_dtype_
    comptime CacheType = PagedKVCache[
        Self.dtype,
        Self.kv_params,
        Self.page_size,
        Self.scale_dtype,
        Self.quantization_granularity_,
    ]

    # Shape is [total_num_blocks, 2, num_layers, page_size, num_heads, head_size].
    # Matrix view is
    # (total_num_blocks, 2, num_layers, page_size) x (num_heads, head_size)
    comptime blocks_shape = IntTuple(
        UNKNOWN_VALUE,
        2 if not Self.kv_params.is_mla else 1,
        UNKNOWN_VALUE,
        Self.page_size,
        Int(Self.kv_params.num_heads),
        Int(Self.kv_params.head_size),
    )
    comptime blocks_layout = Layout.row_major(Self.blocks_shape)
    comptime blocks_tt_layout = LTToTTLayout[Self.blocks_layout]
    comptime blocks_tt_type = TileTensor[
        Self.dtype, Self.blocks_tt_layout, MutAnyOrigin
    ]

    # Match PagedKVCache.head_dim_granularity.
    comptime head_dim_granularity = ceildiv(
        Int(Self.kv_params.head_size),
        Self.CacheType.quantization_granularity,
    )
    # Define scales tensor with shape [total_num_blocks, 2, num_layers, page_size, num_heads, granularity]
    comptime scales_shape = IntTuple(
        UNKNOWN_VALUE,  # total_num_blocks
        2 if not Self.kv_params.is_mla else 1,
        UNKNOWN_VALUE,  # num_layers
        Self.page_size,  # page_size
        Int(Self.kv_params.num_heads),  # num_heads
        Self.head_dim_granularity,  # scales per token
    )
    comptime scales_layout = Layout.row_major(Self.scales_shape)
    comptime scales_tt_layout = LTToTTLayout[Self.scales_layout]
    comptime scales_tt_type = TileTensor[
        Self.scale_dtype, Self.scales_tt_layout, MutAnyOrigin
    ]
    var scales: OptionalReg[Self.scales_tt_type]
    var kv_cache_scales_dynamic_shape: IndexList[4]
    var kv_cache_scales_dynamic_strides: IndexList[4]

    var blocks: Self.blocks_tt_type
    var cache_lengths: Self.CacheType.cache_lengths_tt_type
    var lookup_table: Self.CacheType.lookup_table_tt_type
    var max_seq_length: UInt32
    var max_cache_length: UInt32
    var kv_cache_dynamic_shape: IndexList[4]
    var kv_cache_dynamic_strides: IndexList[4]

    def __init__(
        out self,
        blocks: LayoutTensor[Self.dtype, Layout.row_major[6](), MutAnyOrigin],
        cache_lengths: LayoutTensor[
            DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ],
        lookup_table: LayoutTensor[
            DType.uint32, Layout.row_major[2](), ImmutAnyOrigin
        ],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
        scales: OptionalReg[
            LayoutTensor[Self.scale_dtype, Layout.row_major[6](), MutAnyOrigin]
        ] = None,
    ):
        """Construct from LayoutTensor params (MOGG boundary)."""
        comptime assert blocks.rank == 6
        self.blocks = lt_to_tt[ResultLayout=Self.blocks_tt_layout](blocks)
        self.cache_lengths = lt_to_tt[
            ResultLayout=Self.CacheType.cache_lengths_tt_layout
        ](cache_lengths)
        self.lookup_table = lt_to_tt[
            ResultLayout=Self.CacheType.lookup_table_tt_layout
        ](lookup_table)
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.kv_cache_dynamic_shape, self.kv_cache_dynamic_strides = (
            _compute_kv_cache_dynamic_shape_strides[4, (1, 2)](self.blocks)
        )
        if scales is not None:
            self.scales = lt_to_tt[ResultLayout=Self.scales_tt_layout](
                scales.value()
            )
            self.kv_cache_scales_dynamic_shape, self.kv_cache_scales_dynamic_strides = _compute_kv_cache_dynamic_shape_strides[
                4, (1, 2)
            ](
                self.scales.value()
            )
        else:
            self.scales = None
            self.kv_cache_scales_dynamic_shape = IndexList[4](0, 0, 0, 0)
            self.kv_cache_scales_dynamic_strides = IndexList[4](0, 0, 0, 0)

    def __init__(
        out self,
        blocks: Self.blocks_tt_type,
        cache_lengths: Self.CacheType.cache_lengths_tt_type,
        lookup_table: Self.CacheType.lookup_table_tt_type,
        max_seq_length: UInt32,
        max_cache_length: UInt32,
        scales: OptionalReg[Self.scales_tt_type] = None,
    ):
        """Construct from TileTensor fields directly."""
        self.blocks = blocks
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.kv_cache_dynamic_shape, self.kv_cache_dynamic_strides = (
            _compute_kv_cache_dynamic_shape_strides[4, (1, 2)](self.blocks)
        )
        if scales is not None:
            self.scales = scales.value()
            self.kv_cache_scales_dynamic_shape, self.kv_cache_scales_dynamic_strides = _compute_kv_cache_dynamic_shape_strides[
                4, (1, 2)
            ](
                self.scales.value()
            )
        else:
            self.scales = None
            self.kv_cache_scales_dynamic_shape = IndexList[4](0, 0, 0, 0)
            self.kv_cache_scales_dynamic_strides = IndexList[4](0, 0, 0, 0)

    @always_inline
    def get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        return self._get_cache[0](layer_idx)

    @always_inline
    def get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        comptime assert (
            not Self.kv_params.is_mla
        ), "Cannot call get_value_cache for MLA cache"
        return self._get_cache[1](layer_idx)

    @always_inline
    def _get_cache[kv_idx: Int](self, layer_idx: Int) -> Self.CacheType:
        comptime assert (
            kv_idx >= 0 and kv_idx < 2
        ), "Invalid kv_idx for KV cache"

        var kv_layer_coord = coord[DType.int64](
            Tuple(0, kv_idx, layer_idx, 0, 0, 0)
        )

        var scales_tt: OptionalReg[Self.CacheType.scales_tt_type] = None
        comptime if Self.CacheType.quantization_enabled:
            if self.scales is not None:
                var scale_offset = Int(
                    self.scales.value().layout(kv_layer_coord)
                )
                scales_tt = _make_cache_tt[
                    Self.CacheType.scale_dtype,
                    Self.CacheType.scales_tt_layout,
                    4,
                ](
                    self.scales.value().ptr + scale_offset,
                    self.kv_cache_scales_dynamic_shape,
                    self.kv_cache_scales_dynamic_strides,
                )

        var blocks_offset = Int(self.blocks.layout(kv_layer_coord))
        return self.CacheType(
            _make_cache_tt[
                Self.CacheType.dtype,
                Self.CacheType.blocks_tt_layout,
                4,
            ](
                self.blocks.ptr + blocks_offset,
                self.kv_cache_dynamic_shape,
                self.kv_cache_dynamic_strides,
            ),
            self.cache_lengths,
            self.lookup_table,
            self.max_seq_length,
            self.max_cache_length,
            scales_tt,
        )

    def cache_length(self, bs_idx: Int) -> Int:
        return Int(self.cache_lengths[bs_idx])
