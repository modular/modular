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
"""
This module contains the types for the key-value cache APIs.

The module includes structs implementing several different types of
[KV caches](/glossary/ai/kv-cache).

This module defines two traits that define the roles of the different structs

- `KVCacheT`: Defines the interface for a single (key or value) cache.
- `KVCollectionT`: Defines the interface for a pair of caches (keys and values).
"""

from buffer import Dim, DimList, NDBuffer
from layout import LayoutTensor

from utils import Index, IndexList


@parameter
fn _strides_from_shape[shape: DimList, *, skip: Int = 0]() -> DimList:
    alias rank = len(shape)
    var strides = List[Dim](length=rank, fill=Dim())
    var stride = Dim(1)

    # Skip over dimensions that are not contiguous. This occurs when computing the
    # strides for a buffer slice where some intermediate dimensions needed to
    # compute the full stride are not available. In the current use case, one of
    # these dimensions is `num_layers` and this is not a statically known value at
    # this time.
    @parameter
    for i in reversed(range(skip, rank)):
        strides[i] = stride
        stride *= shape.at[i]()

    @parameter
    if rank == 4:
        return DimList(strides[0], strides[1], strides[2], strides[3])
    elif rank == 6:
        return DimList(
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            strides[4],
            strides[5],
        )
    else:
        constrained[False, "Extend to support additional ranks."]()
        return DimList.create_unknown[rank]()


@always_inline
fn _compute_kv_cache_dynamic_shape_strides[
    dtype: DType, rank: Int, //, kv_cache_rank: Int, drop_list: Tuple
](blocks: NDBuffer[dtype, rank, **_]) -> (
    IndexList[kv_cache_rank],
    IndexList[kv_cache_rank],
):
    var kv_cache_shape = IndexList[kv_cache_rank]()
    var kv_cache_strides = IndexList[kv_cache_rank]()
    var out_index = kv_cache_rank - 1
    var stride = 1

    @parameter
    for i in reversed(range(rank)):
        var dim = blocks.dim[i]()

        # Skip dimensions in the drop list (kv_idx and layer_idx).
        @parameter
        if i not in drop_list:
            kv_cache_shape[out_index] = dim
            kv_cache_strides[out_index] = stride
            out_index = out_index - 1

        stride *= dim

    return (kv_cache_shape, kv_cache_strides)


@fieldwise_init
@register_passable("trivial")
struct KVCacheStaticParams(Copyable, EqualityComparable, Movable):
    var num_heads: UInt
    var head_size: UInt

    @always_inline("nodebug")
    fn __eq__(self, rhs: KVCacheStaticParams) -> Bool:
        return (
            self.num_heads == rhs.num_heads and self.head_size == rhs.head_size
        )

    @always_inline("nodebug")
    fn __ne__(self, rhs: KVCacheStaticParams) -> Bool:
        return not (self == rhs)


@register_passable("trivial")
trait KVCacheT(Copyable, Movable):
    """Trait for different KVCache types and implementations.

    Represents a single (key or value) cache.
    """

    alias dtype: DType
    alias kv_params: KVCacheStaticParams

    fn cache_lengths_nd(self) -> NDBuffer[DType.uint32, 1, MutableAnyOrigin]:
        """Returns the cache lengths as a NDBuffer."""
        ...

    fn cache_length(self, batch_idx: Int) -> Int:
        """Returns the length of the cache for a given batch index."""
        ...

    fn load[
        width: Int
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        dtype, width
    ]:
        """Loads an element from the given index."""
        ...

    fn store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[dtype, *_],
    ):
        """Stores an element at the given index."""
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
    ) -> UnsafePointer[Scalar[dtype]]:
        """Returns a LayoutTensor pointing to the KVCache block at the given index.

        Paged KVCache implementations must have a block_size which is a multiple of the
        and greater than the layout's first dimension.
        """
        ...

    @staticmethod
    fn max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
        ...


@register_passable("trivial")
struct ContinuousBatchingKVCache[
    dtype_: DType,
    kv_params_: KVCacheStaticParams,
](KVCacheT):
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

    alias dtype = dtype_
    alias kv_params = kv_params_

    # Shape is [num_blocks, max_seq_len, num_heads, head_size].
    alias blocks_shape = DimList(
        Dim(),
        Dim(),
        Dim(Int(Self.kv_params.num_heads)),
        Dim(Int(Self.kv_params.head_size)),
    )
    alias blocks_stride = _strides_from_shape[Self.blocks_shape, skip=1]()
    alias blocks_type = NDBuffer[
        Self.dtype, 4, MutableAnyOrigin, Self.blocks_shape, Self.blocks_stride
    ]

    var blocks: Self.blocks_type
    var cache_lengths: NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    var lookup_table: NDBuffer[DType.uint32, 1, MutableAnyOrigin]

    # The length of the longest sequence in the current request.
    # This length only considers tokens not in the KVCache.
    var max_seq_length: UInt32

    # The length of the longest context in the current request.
    # This is effectively:
    #   max(cache_lengths[i] + prompt_lengths[i] for i in range(batch_size)
    var max_cache_length: UInt32

    @always_inline
    fn _get_idx_tuple(
        self, block_idx: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int
    ) -> IndexList[4]:
        debug_assert(
            head_idx < Self.kv_params.num_heads, "KVCache head_idx out of range"
        )
        debug_assert(
            head_dim_idx < Self.kv_params.head_size,
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
        cache_lengths: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
        lookup_table: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
    ):
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
    fn cache_lengths_nd(self) -> NDBuffer[DType.uint32, 1, MutableAnyOrigin]:
        return self.cache_lengths

    @always_inline
    fn cache_length(self, batch_idx: Int) -> Int:
        debug_assert(
            batch_idx < self._batch_size(), "KVCache batch_idx is out of bounds"
        )
        return Int(self.cache_lengths[batch_idx][0])

    @always_inline
    fn load[
        width: Int
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        Self.dtype, width
    ]:
        debug_assert(
            bs < self._batch_size(),
            "KVCache::load batch_size out of range",
        )

        var block_idx = self.lookup_table[bs]
        var idx = self._get_idx_tuple(
            Int(block_idx), head_idx, tok_idx, head_dim_idx
        )
        return self.blocks.load[width=width](idx)

    @always_inline
    fn store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[Self.dtype, *_],
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
    ) -> UnsafePointer[Scalar[Self.dtype]]:
        var block_idx = Int(self.lookup_table[batch_idx])
        var full_block_idx = self._get_idx_tuple(
            block_idx, head_idx, start_tok_idx, head_dim_idx
        )
        var offset_ptr = self.blocks._offset(full_block_idx)
        return offset_ptr


@register_passable("trivial")
struct PagedKVCache[
    dtype_: DType,
    kv_params_: KVCacheStaticParams,
    page_size: Int,
](KVCacheT):
    """The PagedKVCache is a wrapper around the KVCache blocks for a given layer.
    It is used to access the KVCache blocks for PagedAttention.

    Parameters:
        dtype_: The dtype of the kv-cache.
        kv_params_: The kv-cache static parameters.
        page_size: The size of the page.
    """

    alias dtype = dtype_
    alias kv_params = kv_params_

    # Shape is [total_num_blocks, page_size, num_heads, head_size].
    alias blocks_shape = DimList(
        Dim(),
        Dim(page_size),
        Dim(Int(Self.kv_params.num_heads)),
        Dim(Int(Self.kv_params.head_size)),
    )
    alias blocks_stride = _strides_from_shape[Self.blocks_shape, skip=1]()
    alias blocks_type = NDBuffer[
        Self.dtype, 4, MutableAnyOrigin, Self.blocks_shape, Self.blocks_stride
    ]

    var blocks: Self.blocks_type
    var cache_lengths: NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    var lookup_table: NDBuffer[DType.uint32, 2, MutableAnyOrigin]

    # The length of the longest sequence in the current request.
    # This length only considers tokens not in the KVCache.
    var max_seq_length: UInt32

    # The length of the longest context in the current request.
    # This is effectively:
    #   max(cache_lengths[i] + prompt_lengths[i] for i in range(batch_size)
    var max_cache_length: UInt32

    fn __init__(
        out self,
        blocks: Self.blocks_type,
        cache_lengths: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
        lookup_table: NDBuffer[DType.uint32, 2, MutableAnyOrigin],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
    ):
        debug_assert(
            blocks.dim[1]() == page_size,
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

    @staticmethod
    fn max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
        return page_size

    @always_inline
    fn cache_lengths_nd(self) -> NDBuffer[DType.uint32, 1, MutableAnyOrigin]:
        return self.cache_lengths

    fn cache_length(self, batch_idx: Int) -> Int:
        """Returns the length of the cache for a given batch index."""
        return Int(self.cache_lengths[batch_idx])

    @always_inline
    fn _get_idx(
        self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int
    ) -> IndexList[4]:
        debug_assert(
            head_idx < Self.kv_params.num_heads,
            "KVCache head_idx out of range (",
            head_idx,
            ")",
        )
        debug_assert(
            head_dim_idx < Self.kv_params.head_size,
            "KVCache head_dim_idx is out of range",
        )

        var lut_block_index, tok_in_block_idx = divmod(tok_idx, self.page_size)

        debug_assert(
            tok_in_block_idx < self.blocks.dim[1](),
            "KVCache tok_idx out of range",
        )

        debug_assert(bs < len(self.cache_lengths), "batch_idx is oob")
        debug_assert(
            lut_block_index < self.page_size,
            "block_idx is OOB. Attempted to access block index ",
            lut_block_index,
            " with page size ",
            self.page_size,
        )
        block_idx = Int(self.lookup_table[bs, lut_block_index])
        return Index(block_idx, tok_in_block_idx, head_idx, head_dim_idx)

    @always_inline
    fn load[
        width: Int
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        Self.dtype, width
    ]:
        """Loads an element from the given index."""
        var idx = self._get_idx(bs, head_idx, tok_idx, head_dim_idx)
        return self.blocks.load[width=width](idx)

    @always_inline
    fn store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[Self.dtype, *_],
    ):
        """Stores an element at the given index."""
        var idx = self._get_idx(bs, head_idx, tok_idx, head_dim_idx)
        self.blocks.store(idx, val)

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
    ) -> UnsafePointer[Scalar[Self.dtype]]:
        constrained[
            tile_size <= page_size and page_size % tile_size == 0,
            (
                "Invalid tile size for PagedKVCache. tile_size must be less"
                " than or equal to the page size and divisible by the page size"
            ),
        ]()

        var full_block_idx = self._get_idx(
            batch_idx, head_idx, start_tok_idx, head_dim_idx
        )

        var ptr = self.blocks._offset(full_block_idx)
        return ptr


trait KVCollectionT(Copyable, Movable):
    """Trait for a pair of caches (keys and values)."""

    alias CacheType: KVCacheT
    alias name_str: StaticString
    alias dtype: DType
    alias kv_params: KVCacheStaticParams

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
    It does own the Pointer[NDBuffer[dtype, 3]] and valid_lengths buffer
    """

    alias name_str = "continuous_batching"
    alias dtype = dtype_
    alias kv_params = kv_params_
    alias CacheType = ContinuousBatchingKVCache[Self.dtype, Self.kv_params]

    # Shape is [num_blocks, 2, num_layers, max_seq_len, num_heads, head_size].
    alias blocks_shape = DimList(
        Dim(),
        Dim(),
        Dim(),
        Dim(),
        Dim(Int(Self.kv_params.num_heads)),
        Dim(Int(Self.kv_params.head_size)),
    )
    alias blocks_stride = _strides_from_shape[Self.blocks_shape]()
    alias blocks_type = NDBuffer[
        Self.dtype, 6, MutableAnyOrigin, Self.blocks_shape, Self.blocks_stride
    ]

    var cache_lengths: NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    var lookup_table: NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    var blocks: Self.blocks_type
    var max_seq_length: UInt32
    var max_cache_length: UInt32
    var kv_cache_dynamic_shape: IndexList[4]
    var kv_cache_dynamic_strides: IndexList[4]

    fn __init__(
        out self,
        blocks: NDBuffer[Self.dtype, 6, MutableAnyOrigin],
        cache_lengths: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
        lookup_table: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
    ):
        self.blocks = rebind[self.blocks_type](blocks)
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.kv_cache_dynamic_shape, self.kv_cache_dynamic_strides = (
            _compute_kv_cache_dynamic_shape_strides[4, (1, 2)](self.blocks)
        )

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

    @always_inline
    fn get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        return self._get_cache[0](layer_idx)

    @always_inline
    fn get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        return self._get_cache[1](layer_idx)

    @always_inline
    fn _get_cache[kv_idx: Int](self, layer_idx: Int) -> Self.CacheType:
        return self.CacheType(
            self.CacheType.blocks_type(
                self.blocks._offset(
                    IndexList[6](0, kv_idx, layer_idx, 0, 0, 0)
                ),
                self.kv_cache_dynamic_shape,
                self.kv_cache_dynamic_strides,
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
](KVCollectionT):
    alias name_str = "paged"
    alias dtype = dtype_
    alias kv_params = kv_params_
    alias CacheType = PagedKVCache[Self.dtype, Self.kv_params, page_size]

    # Shape is [total_num_blocks, 2, num_layers, page_size, num_heads, head_size].
    alias blocks_shape = DimList(
        Dim(),
        Dim(),
        Dim(),
        Dim(page_size),
        Dim(Int(Self.kv_params.num_heads)),
        Dim(Int(Self.kv_params.head_size)),
    )
    alias blocks_stride = _strides_from_shape[Self.blocks_shape]()
    alias blocks_type = NDBuffer[
        Self.dtype, 6, MutableAnyOrigin, Self.blocks_shape, Self.blocks_stride
    ]

    var blocks: Self.blocks_type
    var cache_lengths: NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    var lookup_table: NDBuffer[DType.uint32, 2, MutableAnyOrigin]
    var max_seq_length: UInt32
    var max_cache_length: UInt32
    var kv_cache_dynamic_shape: IndexList[4]
    var kv_cache_dynamic_strides: IndexList[4]

    fn __init__(
        out self,
        blocks: NDBuffer[Self.dtype, 6, MutableAnyOrigin],
        cache_lengths: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
        lookup_table: NDBuffer[DType.uint32, 2, MutableAnyOrigin],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
    ):
        self.blocks = rebind[Self.blocks_type](blocks)
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.kv_cache_dynamic_shape, self.kv_cache_dynamic_strides = (
            _compute_kv_cache_dynamic_shape_strides[4, (1, 2)](self.blocks)
        )

    fn __copyinit__(out self, other: Self):
        self.blocks = other.blocks
        self.cache_lengths = other.cache_lengths
        self.lookup_table = other.lookup_table
        self.max_seq_length = other.max_seq_length
        self.max_cache_length = other.max_cache_length
        self.kv_cache_dynamic_shape = other.kv_cache_dynamic_shape
        self.kv_cache_dynamic_strides = other.kv_cache_dynamic_strides

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

    fn __moveinit__(out self, owned other: Self):
        self.blocks = other.blocks
        self.cache_lengths = other.cache_lengths
        self.lookup_table = other.lookup_table
        self.max_seq_length = other.max_seq_length
        self.max_cache_length = other.max_cache_length
        self.kv_cache_dynamic_shape = other.kv_cache_dynamic_shape
        self.kv_cache_dynamic_strides = other.kv_cache_dynamic_strides

    @always_inline
    fn get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        return self._get_cache[0](layer_idx)

    @always_inline
    fn get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        return self._get_cache[1](layer_idx)

    @always_inline
    fn _get_cache[kv_idx: Int](self, layer_idx: Int) -> Self.CacheType:
        return self.CacheType(
            Self.CacheType.blocks_type(
                self.blocks._offset(
                    IndexList[6](0, kv_idx, layer_idx, 0, 0, 0)
                ),
                self.kv_cache_dynamic_shape,
                self.kv_cache_dynamic_strides,
            ),
            self.cache_lengths,
            self.lookup_table,
            self.max_seq_length,
            self.max_cache_length,
        )

    fn cache_length(self, bs_idx: Int) -> Int:
        return Int(self.cache_lengths[bs_idx])
