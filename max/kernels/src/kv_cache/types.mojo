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

from math import align_up
from gpu.host import DeviceContext
from gpu.host.nvidia.tma import TensorMapSwizzle
from layout import UNKNOWN_VALUE, Layout, LayoutTensor, IntTuple
from layout.runtime_layout import RuntimeLayout
from layout.tma_async import (
    SplitLastDimTMATensorTile,
    create_split_tma,
    RaggedTMA3DTile,
)

from memory import LegacyUnsafePointer
from collections import OptionalReg
from utils import Index, IndexList
from sys import size_of
from builtin.device_passable import DevicePassable
from math import ceildiv


@always_inline
fn swizzle_granularity[dtype: DType, swizzle_mode: TensorMapSwizzle]() -> Int:
    comptime sg = swizzle_mode.bytes() // size_of[dtype]()
    return sg


@always_inline
fn padded_depth[
    dtype: DType, swizzle_mode: TensorMapSwizzle, depth: Int
]() -> Int:
    comptime padded_depth = align_up(
        depth, swizzle_mode.bytes() // size_of[dtype]()
    )
    return padded_depth


@always_inline
fn _compute_kv_cache_dynamic_shape_strides[
    dtype: DType, //, kv_cache_rank: Int, drop_list: Tuple
](blocks: LayoutTensor[dtype, ...]) -> Tuple[
    IndexList[kv_cache_rank],
    IndexList[kv_cache_rank],
]:
    var kv_cache_shape = IndexList[kv_cache_rank]()
    var kv_cache_strides = IndexList[kv_cache_rank]()
    var out_index = kv_cache_rank - 1
    var stride = 1

    comptime for i in reversed(range(blocks.rank)):
        var dim = blocks.dim[i]()

        # Skip dimensions in the drop list (kv_idx and layer_idx).
        comptime if i not in drop_list:
            kv_cache_shape[out_index] = dim
            kv_cache_strides[out_index] = stride
            out_index = out_index - 1

        stride *= dim

    return (kv_cache_shape, kv_cache_strides)


struct KVCacheStaticParams(Equatable, TrivialRegisterPassable):
    var num_heads: UInt
    var head_size: UInt
    var is_mla: Bool

    fn __init__(
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


trait KVCacheT(DevicePassable, TrivialRegisterPassable):
    """Trait for different KVCache types and implementations.

    Represents a single (key or value) cache.
    """

    comptime dtype: DType
    comptime kv_params: KVCacheStaticParams
    comptime page_size_: Int
    comptime scale_dtype: DType = DType.invalid
    comptime quantization_enabled: Bool = False

    fn cache_lengths_nd(
        self,
    ) -> LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin]:
        """Returns the cache lengths as a LayoutTensor."""
        ...

    fn cache_length(self, batch_idx: Int) -> Int:
        """Returns the length of the cache for a given batch index."""
        ...

    fn load[
        width: Int,
        output_dtype: DType = Self.dtype,
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        output_dtype, width
    ]:
        """Loads an element from the given index."""
        ...

    fn store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[Self.dtype, ...],
    ):
        """Stores an element at the given index."""
        ...

    fn store_scale(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        scales: SIMD[Self.scale_dtype, ...],
    ):
        """Stores the quantization scales at the given index."""
        ...

    fn load_scale[
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

    fn load_quantized[
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

    fn empty_cache(self) -> Bool:
        """Returns true if the cache_lengths for all requests is 0,
        false otherwise."""
        ...

    fn max_prompt_length(self) -> UInt32:
        """Returns the maximum sequence length across all batches of the current
        request."""
        ...

    fn max_context_length(self) -> UInt32:
        """Returns the maximum cache length used across all batches of the
        current request."""
        ...

    # TODO: change this to return a LayoutTensor once MOCO-1471 is fixed
    @always_inline
    fn block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        """Returns a LayoutTensor pointing to the KVCache block at the given index.

        Paged KVCache implementations must have a block_size which is a multiple of the
        and greater than the layout's first dimension.
        """
        ...

    @always_inline
    fn scales_block_paged_ptr(
        self,
        batch_idx: Int,
        start_tok_idx: Int,
        head_idx: Int,
        head_dim_idx: Int = 0,
    ) -> UnsafePointer[Scalar[Self.scale_dtype], MutAnyOrigin]:
        """Returns a pointer to the scales block at the requested indices."""
        ...

    @staticmethod
    fn max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
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
    fn create_ragged_tma_tile[
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
    # Shape is [num_blocks, max_seq_len, num_heads, head_size].
    comptime blocks_shape = IntTuple(
        UNKNOWN_VALUE,
        UNKNOWN_VALUE,
        Int(Self.kv_params.num_heads),
        Int(Self.kv_params.head_size),
    )
    comptime blocks_layout = Layout.row_major(Self.blocks_shape)
    comptime blocks_type = LayoutTensor[
        Self.dtype, Self.blocks_layout, MutAnyOrigin
    ]

    var blocks: Self.blocks_type
    var cache_lengths: LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
    ]
    var lookup_table: LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
    ]

    # The length of the longest sequence in the current request.
    # This length only considers tokens not in the KVCache.
    var max_seq_length: UInt32

    # The length of the longest context in the current request.
    # This is effectively:
    #   max(cache_lengths[i] + prompt_lengths[i] for i in range(batch_size)
    var max_cache_length: UInt32

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "ContinuousBatchingKVCache"

    @always_inline
    fn _get_idx_tuple(
        self, block_idx: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int
    ) -> IndexList[4]:
        debug_assert(
            UInt(head_idx) < Self.kv_params.num_heads,
            "KVCache head_idx out of range",
        )
        debug_assert(
            UInt(head_dim_idx) < Self.kv_params.head_size,
            "KVCache head_dim_idx is out of range",
        )
        debug_assert(
            tok_idx < self.blocks.dim[1](),
            "KVCache tok_idx out of range",
        )
        return Index(block_idx, tok_idx, head_idx, head_dim_idx)

    @staticmethod
    fn max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
        return -1

    fn __init__(
        out self,
        blocks: Self.blocks_type,
        cache_lengths: LayoutTensor[
            DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ],
        lookup_table: LayoutTensor[
            DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
    ):
        comptime assert (
            not self.quantization_enabled
        ), "ContinuousBatchingKVCache does not support quantization"
        debug_assert(
            blocks.dim[2]() == Int(Self.kv_params.num_heads),
            "blocks.dim[2]() must be equal to kv_params.num_heads",
        )
        debug_assert(
            blocks.dim[3]() == Int(Self.kv_params.head_size),
            "blocks.dim[3]() must be equal to kv_params.head_size",
        )

        self.blocks = blocks
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length

    @always_inline
    fn _batch_size(self) -> Int:
        return self.cache_lengths.dim[0]()

    @always_inline
    fn cache_lengths_nd(
        self,
    ) -> LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin]:
        return self.cache_lengths

    @always_inline
    fn cache_length(self, batch_idx: Int) -> Int:
        debug_assert(
            batch_idx < self._batch_size(), "KVCache batch_idx is out of bounds"
        )
        return Int(self.cache_lengths[batch_idx][0])

    @always_inline
    fn load[
        width: Int,
        output_dtype: DType = Self.dtype,
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        output_dtype, width
    ]:
        debug_assert(
            bs < self._batch_size(),
            "KVCache::load batch_size out of range",
        )

        var block_idx = self.lookup_table[bs]
        var idx = self._get_idx_tuple(
            Int(block_idx), head_idx, tok_idx, head_dim_idx
        )
        return self.blocks.load[width=width](idx).cast[output_dtype]()

    @always_inline
    fn store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[Self.dtype, ...],
    ):
        debug_assert(
            bs < self._batch_size(),
            "KVCache::store batch_size out of range",
        )
        var block_idx = self.lookup_table[bs]
        var idx = self._get_idx_tuple(
            Int(block_idx), head_idx, tok_idx, head_dim_idx
        )
        self.blocks.store(idx, val)

    @always_inline
    fn load_scale[
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
    fn store_scale(
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
    fn load_quantized[
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

    fn empty_cache(self) -> Bool:
        """Returns true if the cache_lengths for all requests is 0,
        false otherwise."""
        return self.max_cache_length == 0

    fn max_prompt_length(self) -> UInt32:
        """Returns the maximum sequence length across all batches of the current
        request."""
        return self.max_seq_length

    fn max_context_length(self) -> UInt32:
        """Returns the maximum cache length used across all batches of the
        current request."""
        return self.max_cache_length

    @always_inline
    fn _stride(self) -> UInt32:
        return UInt32(self.blocks.runtime_layout.stride.value[0]) // UInt32(
            self.kv_params.num_heads * self.kv_params.head_size
        )

    @always_inline
    fn row_idx(self, batch_idx: UInt32, tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        block_idx = self.lookup_table[Int(batch_idx)][0]
        return block_idx * self._stride() + tok_idx

    @always_inline
    fn create_tma_tile[
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
        constrained[
            (BK % swizzle_granularity[Self.dtype, swizzle_mode]()) == 0,
            "BK must be a multiple of swizzle granularity",
        ]()
        # The continuous cache is laid out as [num_blocks, num_layers, seq_len, num_heads, head_size]
        # We create a view of the data as a flattened 2D tensor
        var total_blocks = self.blocks.dim[0]()
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
    fn create_ragged_tma_tile[
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
        constrained[
            (BK % swizzle_granularity[Self.dtype, swizzle_mode]()) == 0,
            "BK must be a multiple of swizzle granularity",
        ]()
        var total_blocks = self.blocks.dim[0]()
        var rows = UInt32(total_blocks - 1) * self._stride() + UInt32(
            self.blocks.dim[1]()
        )
        tma = type_of(tma).create[depth = Int(Self.kv_params.head_size)](
            ctx,
            self.blocks.ptr,
            rows=Int(rows),
            middle_dim=Int(Self.kv_params.num_heads),
        )

    @always_inline
    fn block_paged_ptr[
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
        var offset_ptr = self.blocks.ptr + self.blocks._offset(full_block_idx)
        return offset_ptr

    @always_inline
    fn scales_block_paged_ptr(
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


struct PagedKVCache[
    dtype_: DType,
    kv_params_: KVCacheStaticParams,
    page_size: Int,
    scale_dtype_: DType = DType.invalid,
    quantization_granularity: Int = 1,
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
        quantization_granularity:  Block size used for quantization (e.g. 128).
    """

    comptime dtype = Self.dtype_
    comptime kv_params = Self.kv_params_
    comptime page_size_ = Self.page_size
    comptime scale_dtype = Self.scale_dtype_
    comptime quantization_enabled = Self.scale_dtype_ != DType.invalid

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
    comptime blocks_type = LayoutTensor[
        Self.dtype, Self.blocks_layout, MutAnyOrigin
    ]

    var blocks: Self.blocks_type
    var cache_lengths: LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
    ]
    var lookup_table: LayoutTensor[
        DType.uint32, Layout.row_major[2](), ImmutAnyOrigin
    ]

    # The length of the longest sequence in the current request.
    # This length only considers tokens not in the KVCache.
    var max_seq_length: UInt32

    # The length of the longest context in the current request.
    # This is effectively:
    #   max(cache_lengths[i] + prompt_lengths[i] for i in range(batch_size)
    var max_cache_length: UInt32

    comptime head_dim_granularity = ceildiv(
        Int(Self.kv_params.head_size),
        Self.quantization_granularity,
    )
    # Scales shape for KV Cache quantization is [num_blocks, page_size, num_heads].
    comptime scales_shape = IntTuple(
        UNKNOWN_VALUE,  # num_blocks
        Self.page_size,  # page_size
        Int(Self.kv_params.num_heads),  # num_heads
        Self.head_dim_granularity,  # block size
    )
    comptime scales_layout = Layout.row_major(Self.scales_shape)
    comptime scales_block_type = LayoutTensor[
        Self.scale_dtype, Self.scales_layout, MutAnyOrigin
    ]

    # KV Cache quantization scales
    var scales: OptionalReg[Self.scales_block_type]

    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "PagedKVCache"

    fn __init__(
        out self,
        blocks: Self.blocks_type,
        cache_lengths: LayoutTensor[
            DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ],
        lookup_table: LayoutTensor[
            DType.uint32, Layout.row_major[2](), ImmutAnyOrigin
        ],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
        scales: OptionalReg[Self.scales_block_type] = None,
    ):
        debug_assert(
            blocks.dim[1]() == Self.page_size,
            "blocks.dim[1]() must be equal to page_size",
        )
        debug_assert(
            blocks.dim[2]() == Int(Self.kv_params.num_heads),
            "blocks.dim[2]() must be equal to kv_params.num_heads",
        )
        debug_assert(
            blocks.dim[3]() == Int(Self.kv_params.head_size),
            "blocks.dim[3]() must be equal to kv_params.head_size",
        )

        self.blocks = blocks
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.scales = scales

    @staticmethod
    fn max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
        return Self.page_size

    @always_inline
    fn cache_lengths_nd(
        self,
    ) -> LayoutTensor[DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin]:
        return self.cache_lengths

    fn cache_length(self, batch_idx: Int) -> Int:
        """Returns the length of the cache for a given batch index."""
        return Int(self.cache_lengths[batch_idx])

    @always_inline
    fn _stride(self) -> UInt32:
        return UInt32(self.blocks.runtime_layout.stride.value[0]) // UInt32(
            self.kv_params.num_heads * self.kv_params.head_size
        )

    @always_inline
    fn row_idx(self, batch_idx: UInt32, tok_idx: UInt32) -> UInt32:
        """Returns the row idx when viewing the memory as a matrix."""
        var lut_block_index, tok_in_block_idx = divmod(
            Int(tok_idx), Self.page_size
        )
        debug_assert(
            tok_in_block_idx < self.blocks.dim[1](),
            "KVCache tok_idx out of range",
        )

        debug_assert(
            batch_idx < UInt32(self.cache_lengths.size()), "batch_idx is oob"
        )
        debug_assert(
            lut_block_index < self.blocks.dim[0](),
            "block_idx is OOB. Attempted to access block index ",
            lut_block_index,
            " with num_blocks ",
            self.blocks.dim[0](),
        )
        block_idx = self.lookup_table[Int(batch_idx), lut_block_index][0]
        # alias row_stride = Int(num_heads * head_size * Self.collection_size)
        return block_idx * self._stride() + UInt32(tok_in_block_idx)

    @always_inline
    fn create_tma_tile[
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
        constrained[
            (BK % swizzle_granularity[Self.dtype, swizzle_mode]()) == 0,
            "BK must be a multiple of swizzle granularity",
        ]()
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
        var total_blocks = self.blocks.dim[0]()
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
    fn create_ragged_tma_tile[
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
        constrained[
            (BK % swizzle_granularity[Self.dtype, swizzle_mode]()) == 0,
            "BK must be a multiple of swizzle granularity",
        ]()
        var total_blocks = self.blocks.dim[0]()
        var rows = UInt32(total_blocks - 1) * self._stride() + UInt32(
            Self.page_size
        )
        tma = type_of(tma).create[depth = Int(Self.kv_params.head_size)](
            ctx,
            self.blocks.ptr,
            rows=Int(rows),
            middle_dim=Int(Self.kv_params.num_heads),
        )

    @always_inline
    fn _get_idx(
        self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int
    ) -> IndexList[4]:
        debug_assert(
            UInt(head_idx) < Self.kv_params.num_heads,
            "KVCache head_idx out of range (",
            head_idx,
            ")",
        )
        debug_assert(
            UInt(head_dim_idx) < Self.kv_params.head_size,
            "KVCache head_dim_idx is out of range",
        )

        var lut_block_index, tok_in_block_idx = divmod(tok_idx, self.page_size)

        debug_assert(
            tok_in_block_idx < self.blocks.dim[1](),
            "KVCache tok_idx out of range",
        )

        debug_assert(bs < self.cache_lengths.size(), "batch_idx is oob")
        debug_assert(
            lut_block_index < self.blocks.dim[0](),
            "block_idx is OOB. Attempted to access block index ",
            lut_block_index,
            " with num_blocks ",
            self.blocks.dim[0](),
        )
        block_idx = Int(self.lookup_table[bs, lut_block_index])
        return Index(block_idx, tok_in_block_idx, head_idx, head_dim_idx)

    @always_inline
    fn _get_scale_idx(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
    ) -> IndexList[4]:
        debug_assert(
            UInt(head_idx) < Self.kv_params.num_heads,
            "KVCache head_idx out of range (",
            head_idx,
            ")",
        )

        var lut_block_index, tok_in_block_idx = divmod(tok_idx, self.page_size)

        debug_assert(
            tok_in_block_idx < self.blocks.dim[1](),
            "KVCache tok_idx out of range",
        )

        debug_assert(bs < self.cache_lengths.size(), "batch_idx is oob")
        debug_assert(
            lut_block_index < self.blocks.dim[0](),
            "block_idx is OOB. Attempted to access block index ",
            lut_block_index,
            " with num_blocks ",
            self.blocks.dim[0](),
        )

        block_idx = Int(self.lookup_table[bs, lut_block_index])
        var head_dim_granularity = ceildiv(
            head_dim_idx,
            Self.quantization_granularity,
        )
        return Index(
            block_idx,
            tok_in_block_idx,
            head_idx,
            head_dim_granularity,
        )

    @always_inline
    fn load[
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
    fn store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[Self.dtype, ...],
    ):
        """Stores an element at the given index."""
        var idx = self._get_idx(bs, head_idx, tok_idx, head_dim_idx)
        self.blocks.store(idx, val)

    @always_inline
    fn load_scale[
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
        debug_assert(
            self.scales is not None,
            "Scales missing, yet KVCache quantization enabled",
        )
        var idx = self._get_scale_idx(bs, head_idx, tok_idx, head_dim_idx)
        return self.scales.value().load[width=width](idx)

    @always_inline
    fn store_scale(
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
        self.scales.value().store(scale_idx, scales)

    @always_inline
    fn load_quantized[
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
        return self.blocks.load[width=width](idx)

    fn empty_cache(self) -> Bool:
        """Returns true if the cache_lengths for all requests is 0,
        false otherwise."""
        return self.max_cache_length == 0

    fn max_prompt_length(self) -> UInt32:
        """Returns the maximum sequence length across all batches of the current
        request."""
        return self.max_seq_length

    fn max_context_length(self) -> UInt32:
        """Returns the maximum cache length used across all batches of the
        current request."""
        return self.max_cache_length

    @always_inline
    fn block_paged_ptr[
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

        var ptr = self.blocks.ptr + self.blocks._offset(full_block_idx)
        return ptr

    @always_inline
    fn scales_block_paged_ptr(
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
        debug_assert(
            self.scales is not None, "Quantization scale factors not set."
        )
        var scales_block = self.scales.value()

        var scales_ptr = scales_block.ptr + scales_block._offset(
            full_scale_block_idx
        )
        return scales_ptr


trait KVCollectionT(ImplicitlyCopyable):
    """Trait for a pair of caches (keys and values)."""

    comptime CacheType: KVCacheT
    comptime name_str: StaticString
    comptime dtype: DType
    comptime kv_params: KVCacheStaticParams

    fn get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        ...

    fn get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        ...

    fn cache_length(self, bs_idx: Int) -> Int:
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
    It does own the Pointer[LayoutTensor[dtype, Layout.row_major[3]()]] and valid_lengths buffer
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
    comptime blocks_type = LayoutTensor[
        Self.dtype, Self.blocks_layout, MutAnyOrigin
    ]

    var cache_lengths: LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
    ]
    var lookup_table: LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
    ]
    var blocks: Self.blocks_type
    var max_seq_length: UInt32
    var max_cache_length: UInt32
    var kv_cache_dynamic_shape: IndexList[4]
    var kv_cache_dynamic_strides: IndexList[4]

    fn __init__(
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
        comptime assert blocks.rank == 6
        self.blocks = rebind[self.blocks_type](blocks)
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.kv_cache_dynamic_shape, self.kv_cache_dynamic_strides = (
            _compute_kv_cache_dynamic_shape_strides[4, (1, 2)](self.blocks)
        )

    @always_inline
    fn get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        return self._get_cache[0](layer_idx)

    @always_inline
    fn get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        return self._get_cache[1](layer_idx)

    @always_inline
    fn _get_cache[kv_idx: Int](self, layer_idx: Int) -> Self.CacheType:
        debug_assert(
            kv_idx == 0 or self.blocks.runtime_layout.shape.value[1] > 1,
            "invalid kv_idx for MLA cache",
        )
        return self.CacheType(
            self.CacheType.blocks_type(
                self.blocks.ptr
                + self.blocks._offset(
                    IndexList[6](0, kv_idx, layer_idx, 0, 0, 0)
                ),
                RuntimeLayout[self.CacheType.blocks_layout](
                    self.kv_cache_dynamic_shape,
                    self.kv_cache_dynamic_strides,
                ),
            ),
            self.cache_lengths,
            self.lookup_table,
            self.max_seq_length,
            self.max_cache_length,
        )

    fn cache_length(self, bs_idx: Int) -> Int:
        return Int(self.cache_lengths[bs_idx])


struct PagedKVCacheCollection[
    dtype_: DType,
    kv_params_: KVCacheStaticParams,
    page_size: Int,
    scale_dtype_: DType = DType.invalid,
](KVCollectionT):
    comptime name_str = "paged"
    comptime dtype = Self.dtype_
    comptime kv_params = Self.kv_params_
    comptime scale_dtype = Self.scale_dtype_
    comptime CacheType = PagedKVCache[
        Self.dtype, Self.kv_params, Self.page_size, Self.scale_dtype
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
    comptime blocks_type = LayoutTensor[
        Self.dtype, Self.blocks_layout, MutAnyOrigin
    ]

    comptime head_dim_granularity = ceildiv(
        Int(Self.kv_params.head_size),
        Self.CacheType.quantization_granularity,
    )
    # Define scales tensor with shape [total_num_blocks, 2, num_layers, page_size, num_heads]
    comptime scales_shape = IntTuple(
        UNKNOWN_VALUE,  # total_num_blocks
        2 if not Self.kv_params.is_mla else 1,
        UNKNOWN_VALUE,  # num_layers
        Self.page_size,  # page_size
        Int(Self.kv_params.num_heads),  # num_heads
        Self.head_dim_granularity,  # block size
    )
    comptime scales_layout = Layout.row_major(Self.scales_shape)
    comptime scales_type = LayoutTensor[
        Self.scale_dtype, Self.scales_layout, MutAnyOrigin
    ]
    var scales: OptionalReg[Self.scales_type]
    var kv_cache_scales_dynamic_shape: IndexList[4]
    var kv_cache_scales_dynamic_strides: IndexList[4]

    var blocks: Self.blocks_type
    comptime cache_lengths_type = LayoutTensor[
        DType.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
    ]
    var cache_lengths: Self.cache_lengths_type
    comptime lookup_table_type = LayoutTensor[
        DType.uint32, Layout.row_major[2](), ImmutAnyOrigin
    ]
    var lookup_table: Self.lookup_table_type
    var max_seq_length: UInt32
    var max_cache_length: UInt32
    var kv_cache_dynamic_shape: IndexList[4]
    var kv_cache_dynamic_strides: IndexList[4]

    fn __init__(
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
        comptime assert blocks.rank == 6
        self.blocks = rebind[Self.blocks_type](blocks)
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.kv_cache_dynamic_shape, self.kv_cache_dynamic_strides = (
            _compute_kv_cache_dynamic_shape_strides[4, (1, 2)](self.blocks)
        )
        if scales is not None:
            self.scales = rebind[Self.scales_type](scales.value())
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
    fn get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        return self._get_cache[0](layer_idx)

    @always_inline
    fn get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        comptime assert (
            not Self.kv_params.is_mla
        ), "Cannot call get_value_cache for MLA cache"
        return self._get_cache[1](layer_idx)

    @always_inline
    fn _get_cache[kv_idx: Int](self, layer_idx: Int) -> Self.CacheType:
        comptime assert (
            kv_idx >= 0 and kv_idx < 2
        ), "Invalid kv_idx for KV cache"

        var scales_block: OptionalReg[
            LayoutTensor[
                Self.CacheType.scale_dtype,
                Self.CacheType.scales_layout,
                MutAnyOrigin,
            ]
        ] = None

        comptime if Self.CacheType.quantization_enabled:
            if self.scales is not None:
                scales_block = Self.CacheType.scales_block_type(
                    self.scales.value().ptr
                    + self.scales.value()._offset(
                        IndexList[6](0, kv_idx, layer_idx, 0, 0, 0)
                    ),
                    RuntimeLayout[self.CacheType.scales_layout](
                        self.kv_cache_scales_dynamic_shape,
                        self.kv_cache_scales_dynamic_strides,
                    ),
                )

        return self.CacheType(
            Self.CacheType.blocks_type(
                self.blocks.ptr
                + self.blocks._offset(
                    IndexList[6](0, kv_idx, layer_idx, 0, 0, 0)
                ),
                RuntimeLayout[self.CacheType.blocks_layout](
                    self.kv_cache_dynamic_shape,
                    self.kv_cache_dynamic_strides,
                ),
            ),
            self.cache_lengths,
            self.lookup_table,
            self.max_seq_length,
            self.max_cache_length,
            scales_block,
        )

    fn cache_length(self, bs_idx: Int) -> Int:
        return Int(self.cache_lengths[bs_idx])
