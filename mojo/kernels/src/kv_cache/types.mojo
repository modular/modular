# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.intrinsics import _type_is_eq

from buffer import Dim, DimList, NDBuffer
from layout import Layout, LayoutTensor
from memory import UnsafePointer

from utils import IndexList


@value
@register_passable("trivial")
struct KVCacheStaticParams(EqualityComparable):
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
trait KVCacheT(CollectionElement):
    """Trait for different KVCache types and implementations.

    We have to expose a super-set of constructors to help with genericizing
    the KVCollectionT and KVManagerT. Some trait implementations may
    constrained-guard unused constructors.
    """

    alias type: DType
    alias kv_params: KVCacheStaticParams

    @staticmethod
    fn id() -> String:
        """Returns a string id describing the type, this is used when defining
        mo.opaque symbols."""
        ...

    fn cache_lengths_nd(self) -> NDBuffer[DType.uint32, 1, MutableAnyOrigin]:
        """Returns the cache lengths as a NDBuffer."""
        ...

    fn cache_length(self, batch_idx: Int) -> Int:
        """Returns the length of the cache for a given batch index."""
        ...

    fn load[
        width: Int
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        Self.type, width
    ]:
        """Loads an element from the given index."""
        ...

    fn store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[Self.type, *_],
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
    ) -> UnsafePointer[Scalar[Self.type]]:
        """Returns a LayoutTensor pointing to the KVCache block at the given index.

        Paged KVCache implementations must have a block_size which is a multiple of the
        and greater than the layout's first dimension.
        """
        ...

    @staticmethod
    fn max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
        ...


@value
@register_passable("trivial")
struct ContiguousKVCache[
    type_: DType,
    kv_params_: KVCacheStaticParams,
](KVCacheT):
    """Wrapper for the ContiguousKVCache of a given layer in the transformer model.

    This abstracts the Pointer indirection for accessing the ContiguousKVCache for a
    given batch entry.

    THIS IS THE TYPE THAT IS PASSED TO KV PROJECTION AND FLASH ATTENTION KERNELS.
    """

    alias type = type_
    alias kv_params = kv_params_

    alias _internal_block_shape = DimList(
        Dim(), Dim(), Self.kv_params.num_heads, Self.kv_params.head_size
    )
    alias single_block_shape = DimList(
        Self._internal_block_shape.get[1](),
        Self._internal_block_shape.get[2](),
        Self._internal_block_shape.get[3](),
    )
    alias BlockType = NDBuffer[
        Self.type, 4, MutableAnyOrigin, Self._internal_block_shape
    ]
    var _block: Self.BlockType
    var is_cache_empty: Bool
    var cache_lengths: NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    var batch_size: Int
    var max_seq_length: UInt32
    var max_cache_length: UInt32

    fn __init__(
        out self,
        block: Self.BlockType,
        cache_lengths: NDBuffer[DType.uint32, 1],
        is_cache_empty: Bool,
        batch_size: Int,
        max_seq_length: UInt32,
        max_cache_length: UInt32,
    ):
        self._block = block
        self.cache_lengths = cache_lengths
        self.is_cache_empty = is_cache_empty
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length

    @staticmethod
    fn id() -> String:
        return String(
            "ContiguousKVCache+",
            Self.type,
            "+",
            Self.kv_params.num_heads,
            "+",
            Self.kv_params.head_size,
        )

    @staticmethod
    fn max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
        return -1

    @always_inline
    fn _get_idx_tuple(
        self, bs_idx: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int
    ) -> IndexList[4]:
        debug_assert(
            bs_idx < self.batch_size, "KVCache batch_size out of range"
        )
        debug_assert(
            head_idx < Self.kv_params.num_heads, "KVCache head_idx out of range"
        )
        debug_assert(
            head_dim_idx < Self.kv_params.head_size,
            "KVCache head_dim_idx is out of range",
        )
        debug_assert(
            tok_idx < self._block.dim[1](), "KVCache tok_idx out of range"
        )
        return (
            bs_idx,
            tok_idx,
            head_idx,
            head_dim_idx,
        )

    @always_inline
    fn cache_lengths_nd(self) -> NDBuffer[DType.uint32, 1, MutableAnyOrigin]:
        return self.cache_lengths

    @always_inline
    fn cache_length(self, batch_idx: Int) -> Int:
        debug_assert(
            batch_idx < self.batch_size, "KVCache batch_idx is out of bounds"
        )
        return Int(self.cache_lengths[batch_idx])

    @always_inline
    fn load[
        width: Int
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        Self.type, width
    ]:
        var idx = self._get_idx_tuple(bs, head_idx, tok_idx, head_dim_idx)
        return self._block.load[width=width](idx)

    @always_inline
    fn store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[Self.type, *_],
    ):
        var idx = self._get_idx_tuple(bs, head_idx, tok_idx, head_dim_idx)
        self._block.store(idx, val)

    fn empty_cache(self) -> Bool:
        return self.is_cache_empty

    fn max_prompt_length(self) -> UInt32:
        return self.max_seq_length

    fn max_context_length(self) -> UInt32:
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
    ) -> UnsafePointer[Scalar[Self.type]]:
        var idx = self._get_idx_tuple(
            batch_idx, head_idx, start_tok_idx, head_dim_idx
        )
        return self._block._offset(idx)


@value
@register_passable("trivial")
struct ContinuousBatchingKVCache[
    type_: DType,
    kv_params_: KVCacheStaticParams,
](KVCacheT):
    """Wrapper for the ContinuousKVCache of a given layer in the transformer model.

    This abstracts the Pointer indirection for accessing the ContinuousKVCache for a
    given batch entry.

    THIS IS THE TYPE THAT IS PASSED TO KV PROJECTION AND FLASH ATTENTION KERNELS.
    """

    alias KeyIdx = 0
    alias ValueIdx = 1

    alias type = type_
    alias kv_params = kv_params_

    alias single_block_shape = DimList(
        Dim(), Self.kv_params.num_heads, Self.kv_params.head_size
    )

    # shape is
    # - BSHD: [num_blocks, 2, num_layers, max_seq_len, num_heads, head_size]
    # - BHSD: [num_blocks, 2, num_layers, num_heads, max_seq_len, head_size]
    alias BlocksType = NDBuffer[Self.type, 6, MutableAnyOrigin]
    var blocks: Self.BlocksType
    var cache_lengths: NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    var lookup_table: NDBuffer[DType.uint32, 1, MutableAnyOrigin]

    # The length of the longest sequence in the current request.
    # This length only considers tokens not in the KVCache.
    var max_seq_length: UInt32

    # The length of the longest context in the current request.
    # This is effectively:
    #   max(cache_lengths[i] + prompt_lengths[i] for i in range(batch_size)
    var max_cache_length: UInt32
    var batch_size: Int
    var layer_idx: Int
    var kv_idx: Int

    @always_inline
    fn _get_idx_tuple(
        self, block_idx: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int
    ) -> IndexList[6]:
        debug_assert(
            head_idx < Self.kv_params.num_heads, "KVCache head_idx out of range"
        )
        debug_assert(
            head_dim_idx < Self.kv_params.head_size,
            "KVCache head_dim_idx is out of range",
        )
        debug_assert(
            tok_idx < self.blocks.dim[3](),
            "KVCache tok_idx out of range",
        )
        return IndexList[6](
            block_idx,
            self.kv_idx,
            self.layer_idx,
            tok_idx,
            head_idx,
            head_dim_idx,
        )

    @staticmethod
    fn max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
        return -1

    fn __init__(
        out self,
        blocks: Self.BlocksType,
        cache_lengths: NDBuffer[DType.uint32, 1],
        lookup_table: NDBuffer[DType.uint32, 1],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
        layer_idx: Int,
        kv_idx: Int,
    ):
        self.blocks = blocks
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.batch_size = cache_lengths.dim[0]()
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.layer_idx = layer_idx
        self.kv_idx = kv_idx

    @staticmethod
    fn id() -> String:
        return "KVCacheRegisterPassable"

    @always_inline
    fn cache_lengths_nd(self) -> NDBuffer[DType.uint32, 1, MutableAnyOrigin]:
        return self.cache_lengths

    @always_inline
    fn cache_length(self, batch_idx: Int) -> Int:
        debug_assert(
            batch_idx < self.batch_size, "KVCache batch_idx is out of bounds"
        )
        return Int(self.cache_lengths[batch_idx][0])

    @always_inline
    fn load[
        width: Int
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        Self.type, width
    ]:
        debug_assert(
            bs < self.batch_size,
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
        val: SIMD[Self.type, *_],
    ):
        debug_assert(
            bs < self.batch_size,
            "KVCache::store batch_size out of range",
        )
        var block_idx = self.lookup_table[bs]
        var idx = self._get_idx_tuple(
            Int(block_idx), head_idx, tok_idx, head_dim_idx
        )
        self.blocks.store(idx, val)

    fn incr_cache_length(mut self, batch_idx: Int, inc: Int):
        debug_assert(
            batch_idx < self.batch_size, "KVCache batch_idx is out of bounds"
        )
        self.cache_lengths[batch_idx] += inc

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
    ) -> UnsafePointer[Scalar[Self.type]]:
        var block_idx = Int(self.lookup_table[batch_idx])
        var full_block_idx = self._get_idx_tuple(
            block_idx, head_idx, start_tok_idx, head_dim_idx
        )
        var offset_ptr = self.blocks._offset(full_block_idx)
        return offset_ptr


@value
@register_passable("trivial")
struct PagedKVCache[
    type_: DType,
    kv_params_: KVCacheStaticParams,
    page_size: Int,
](KVCacheT):
    """The PagedKVCache is a wrapper around the KVCache blocks for a given layer.
    It is used to access the KVCache blocks for PagedAttention.
    """

    alias type = type_
    alias kv_params = kv_params_

    alias KeyIdx = 0
    alias ValueIdx = 1

    """The entire region of memory for KVCache blocks with shape:
    [total_num_blocks, 2, num_layers, page_size, num_heads, head_size].
    """
    var blocks: NDBuffer[Self.type, 6, MutableAnyOrigin]
    var cache_lengths: NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    var lookup_table: NDBuffer[DType.uint32, 2, MutableAnyOrigin]

    # The length of the longest sequence in the current request.
    # This length only considers tokens not in the KVCache.
    var max_seq_length: UInt32

    # The length of the longest context in the current request.
    # This is effectively:
    #   max(cache_lengths[i] + prompt_lengths[i] for i in range(batch_size)
    var max_cache_length: UInt32
    var layer_idx: Int
    var kv_idx: Int

    fn __init__(
        out self,
        blocks: NDBuffer[Self.type, 6],
        cache_lengths: NDBuffer[DType.uint32, 1],
        lookup_table: NDBuffer[DType.uint32, 2],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
        layer_idx: Int,
        kv_idx: Int,
    ):
        debug_assert(
            blocks.dim[3]() == page_size,
            "blocks.dim[3]() must be equal to page_size",
        )
        self.blocks = blocks
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.layer_idx = layer_idx
        self.kv_idx = kv_idx

    @staticmethod
    fn max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
        return page_size

    @staticmethod
    fn id() -> String:
        """Returns a string id describing the type, this is used when defining
        mo.opaque symbols."""
        return String("PagedKVCache+", Self.type)

    @always_inline
    fn cache_lengths_nd(self) -> NDBuffer[DType.uint32, 1, MutableAnyOrigin]:
        return self.cache_lengths

    fn cache_length(self, batch_idx: Int) -> Int:
        """Returns the length of the cache for a given batch index."""
        return Int(self.cache_lengths[batch_idx])

    @always_inline
    fn _get_idx(
        self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int
    ) -> IndexList[6]:
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

        lut_block_index, tok_in_block_idx = divmod(tok_idx, self.page_size)

        debug_assert(
            tok_in_block_idx < self.blocks.dim[3](),
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
        return IndexList[6](
            block_idx,
            self.kv_idx,
            self.layer_idx,
            tok_in_block_idx,
            head_idx,
            head_dim_idx,
        )

    fn load[
        width: Int
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        Self.type, width
    ]:
        """Loads an element from the given index."""
        var idx = self._get_idx(bs, head_idx, tok_idx, head_dim_idx)
        return self.blocks.load[width=width](idx)

    fn store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[Self.type, *_],
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
    ) -> UnsafePointer[Scalar[Self.type]]:
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


@value
@register_passable("trivial")
struct PagedKVCacheFA3Fallback[
    type_: DType,
    kv_params_: KVCacheStaticParams,
    page_size: Int,
](KVCacheT):
    """The PagedKVCache is a wrapper around the KVCache blocks for a given layer.
    It is used to access the KVCache blocks for PagedAttention.
    """

    alias type = type_
    alias kv_params = kv_params_

    alias KeyIdx = 0
    alias ValueIdx = 1

    """The entire region of memory for KVCache blocks with shape:
    [2, total_num_blocks, page_size, num_heads, head_size].
    """
    var blocks: NDBuffer[Self.type, 5, MutableAnyOrigin]
    var cache_lengths: NDBuffer[DType.int32, 1, MutableAnyOrigin]

    """The lookup table with shape:
    [num_layers, batch_size, max_num_blocks_in_batch].

    This is to conform to the expected layout in the DaoLabs FA3 kernel.
    We have a different lookup table for each layer.
    """
    var lookup_table: NDBuffer[DType.int32, 3, MutableAnyOrigin]

    # The length of the longest sequence in the current request.
    # This length only considers tokens not in the KVCache.
    var max_seq_length: UInt32

    # The length of the longest context in the current request.
    # This is effectively:
    #   max(cache_lengths[i] + prompt_lengths[i] for i in range(batch_size)
    var max_cache_length: UInt32
    var layer_idx: Int
    var kv_idx: Int

    fn __init__(
        out self,
        blocks: NDBuffer[Self.type, 5],
        cache_lengths: NDBuffer[DType.int32, 1],
        lookup_table: NDBuffer[DType.int32, 3],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
        layer_idx: Int,
        kv_idx: Int,
    ):
        debug_assert(
            blocks.dim[3]() == page_size,
            "blocks.dim[3]() must be equal to page_size",
        )
        self.blocks = blocks
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.layer_idx = layer_idx
        self.kv_idx = kv_idx

    @staticmethod
    fn max_tile_size() -> Int:
        """Returns the maximum tile size for the KVCache."""
        return page_size

    @staticmethod
    fn id() -> String:
        """Returns a string id describing the type, this is used when defining
        mo.opaque symbols."""
        return String("PagedKVCache+", Self.type)

    @always_inline
    fn cache_lengths_nd(self) -> NDBuffer[DType.uint32, 1, MutableAnyOrigin]:
        return rebind[NDBuffer[DType.uint32, 1, MutableAnyOrigin]](
            self.cache_lengths
        )

    fn cache_length(self, batch_idx: Int) -> Int:
        """Returns the length of the cache for a given batch index."""
        return Int(self.cache_lengths[batch_idx])

    @always_inline
    fn _get_idx(
        self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int
    ) -> IndexList[5]:
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

        lut_block_index, tok_in_block_idx = divmod(tok_idx, self.page_size)

        debug_assert(
            tok_in_block_idx < self.blocks.dim[2](),
            "KVCache tok_idx out of range",
        )

        debug_assert(bs < len(self.cache_lengths), "batch_idx is oob")
        debug_assert(
            lut_block_index < self.lookup_table.dim[2](),
            "block_idx is OOB. Attempted to access block index ",
            lut_block_index,
            " with length ",
            self.lookup_table.dim[2](),
        )
        block_idx = Int(self.lookup_table[self.layer_idx, bs, lut_block_index])
        return IndexList[5](
            self.kv_idx,
            block_idx,
            tok_in_block_idx,
            head_idx,
            head_dim_idx,
        )

    fn load[
        width: Int
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        Self.type, width
    ]:
        """Loads an element from the given index."""
        var idx = self._get_idx(bs, head_idx, tok_idx, head_dim_idx)
        return self.blocks.load[width=width](idx)

    fn store(
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[Self.type, *_],
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
    ) -> UnsafePointer[Scalar[Self.type]]:
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


trait KVCollectionT(CollectionElement):
    alias CacheType: KVCacheT
    alias type: DType
    alias kv_params: KVCacheStaticParams

    @staticmethod
    fn id() -> String:
        ...

    fn get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        ...

    fn get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        ...

    fn cache_length(self, bs_idx: Int) -> Int:
        ...


struct ContiguousKVCacheCollection[
    type_: DType,
    kv_params_: KVCacheStaticParams,
](KVCollectionT):
    """This is a "view" of the cache for the given sequences
    in the batch.

    This object does not own the underlying buffers in k_cache and v_cache,
    it's borrowing them from the BlockWrappers in our KVCacheManager.
    It does own the Pointer[NDBuffer[type, 3]] and valid_lengths buffer
    """

    alias type = type_
    alias kv_params = kv_params_
    alias CacheType = ContiguousKVCache[Self.type, Self.kv_params]

    var key_cache: NDBuffer[Self.type, 5, MutableAnyOrigin]
    var value_cache: NDBuffer[Self.type, 5, MutableAnyOrigin]
    var cache_lengths: NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    var is_context_encoding: Bool
    var num_layers: Int
    var batch_size: Int
    var max_seq_len_limit: Int

    # The length of the longest sequence in the current request.
    # This length only considers tokens not in the KVCache.
    var max_seq_len_in_batch: UInt32

    # The length of the longest context in the current request.
    # This is effectively:
    #   max(cache_lengths[i] + prompt_lengths[i] for i in range(batch_size)
    var max_cache_len_in_batch: UInt32

    fn __init__(
        out self,
        key_cache: NDBuffer[Self.type, 5],
        value_cache: NDBuffer[Self.type, 5],
        cache_lengths: NDBuffer[DType.uint32, 1],
        is_context_encoding: Bool,
        num_layers: Int,
        batch_size: Int,
        max_seq_len_in_batch: Int,
        max_cache_len_in_batch: Int,
    ):
        debug_assert(key_cache.dim[0]() == num_layers, "invalid key_cache size")
        debug_assert(
            value_cache.dim[0]() == num_layers, "invalid value_cache size "
        )

        self.key_cache = key_cache
        self.value_cache = value_cache
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_seq_len_limit = key_cache.dim[2]()
        self.max_seq_len_in_batch = max_seq_len_in_batch
        self.max_cache_len_in_batch = max_cache_len_in_batch
        self.cache_lengths = cache_lengths
        self.is_context_encoding = is_context_encoding

    fn __copyinit__(out self, other: Self):
        self.key_cache = other.key_cache
        self.value_cache = other.value_cache
        self.cache_lengths = other.cache_lengths
        self.is_context_encoding = other.is_context_encoding
        self.num_layers = other.num_layers
        self.batch_size = other.batch_size
        self.max_seq_len_limit = other.max_seq_len_limit
        self.max_seq_len_in_batch = other.max_seq_len_in_batch
        self.max_cache_len_in_batch = other.max_cache_len_in_batch

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

    fn __moveinit__(out self, owned other: Self):
        self.key_cache = other.key_cache
        self.value_cache = other.value_cache
        self.cache_lengths = other.cache_lengths
        self.is_context_encoding = other.is_context_encoding
        self.num_layers = other.num_layers
        self.batch_size = other.batch_size
        self.max_seq_len_limit = other.max_seq_len_limit
        self.max_seq_len_in_batch = other.max_seq_len_in_batch
        self.max_cache_len_in_batch = other.max_cache_len_in_batch

    @staticmethod
    fn id() -> String:
        return String(
            "ContiguousKVCacheCollection+",
            Self.type,
            "+",
            Self.kv_params.num_heads,
            "+",
            Self.kv_params.head_size,
        )

    fn get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        var layer_size = self.batch_size * self.kv_params.num_heads * self.max_seq_len_limit * self.kv_params.head_size
        var k_shape = IndexList[4](
            self.key_cache.dim[1](),
            self.key_cache.dim[2](),
            self.key_cache.dim[3](),
            self.key_cache.dim[4](),
        )

        var layer_key_cache = Self.CacheType.BlockType(
            self.key_cache.data + (layer_idx * layer_size), k_shape
        )
        return self.CacheType(
            layer_key_cache,
            self.cache_lengths,
            self.is_context_encoding,
            self.batch_size,
            self.max_seq_len_in_batch,
            self.max_cache_len_in_batch,
        )

    fn get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        var layer_size = self.batch_size * self.kv_params.num_heads * self.max_seq_len_limit * self.kv_params.head_size
        var v_shape = IndexList[4](
            self.value_cache.dim[1](),
            self.value_cache.dim[2](),
            self.value_cache.dim[3](),
            self.value_cache.dim[4](),
        )

        var layer_value_cache = self.CacheType.BlockType(
            self.value_cache.data + (layer_idx * layer_size), v_shape
        )

        return self.CacheType(
            layer_value_cache,
            self.cache_lengths,
            self.is_context_encoding,
            self.batch_size,
            self.max_seq_len_in_batch,
            self.max_cache_len_in_batch,
        )

    fn cache_length(self, batch_idx: Int) -> Int:
        debug_assert(
            batch_idx < self.batch_size, "KVCache batch_idx is out of bounds"
        )
        return Int(self.cache_lengths[batch_idx])


struct ContinuousBatchingKVCacheCollection[
    type_: DType,
    kv_params_: KVCacheStaticParams,
](KVCollectionT):
    """This is a "view" of the cache for the given sequences
    in the batch.

    This object does not own the underlying buffers in k_cache and v_cache,
    it's borrowing them from the BlockWrappers in our KVCacheManager.
    It does own the Pointer[NDBuffer[type, 3]] and valid_lengths buffer
    """

    alias type = type_
    alias kv_params = kv_params_
    alias CacheType = ContinuousBatchingKVCache[Self.type, Self.kv_params]

    var cache_lengths: NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    var lookup_table: NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    var blocks: Self.CacheType.BlocksType
    var max_seq_length: UInt32
    var max_cache_length: UInt32
    var num_layers: Int
    var batch_size: Int

    fn __init__(
        out self,
        blocks: Self.CacheType.BlocksType,
        cache_lengths: NDBuffer[DType.uint32, 1],
        lookup_table: NDBuffer[DType.uint32, 1],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
    ):
        self.blocks = rebind[self.CacheType.BlocksType](blocks)
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length
        self.num_layers = blocks.dim[2]()
        self.batch_size = cache_lengths.dim[0]()

    fn __init__(
        out self,
        blocks: Self.CacheType.BlocksType,
        cache_lengths: NDBuffer[DType.uint32, 1],
        lookup_table: NDBuffer[DType.uint32, 1],
        is_cache_empty: Bool,
    ):
        self.blocks = rebind[self.CacheType.BlocksType](blocks)
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = blocks.dim[3]()
        self.max_cache_length = 0 if is_cache_empty else self.max_seq_length
        self.num_layers = blocks.dim[2]()
        self.batch_size = cache_lengths.dim[0]()

    fn __moveinit__(out self, owned other: Self):
        self.blocks = other.blocks
        self.cache_lengths = other.cache_lengths
        self.lookup_table = other.lookup_table
        self.num_layers = other.num_layers
        self.batch_size = other.batch_size
        self.max_seq_length = other.max_seq_length
        self.max_cache_length = other.max_cache_length

    fn __copyinit__(out self, other: Self):
        self.blocks = other.blocks
        self.cache_lengths = other.cache_lengths
        self.lookup_table = other.lookup_table
        self.num_layers = other.num_layers
        self.batch_size = other.batch_size
        self.max_seq_length = other.max_seq_length
        self.max_cache_length = other.max_cache_length

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

    @staticmethod
    fn id() -> String:
        return "KVCacheCollection"

    fn get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        return self.CacheType(
            self.blocks,
            self.cache_lengths,
            self.lookup_table,
            self.max_seq_length,
            self.max_cache_length,
            layer_idx,
            Self.CacheType.KeyIdx,
        )

    fn get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        return self.CacheType(
            self.blocks,
            self.cache_lengths,
            self.lookup_table,
            self.max_seq_length,
            self.max_cache_length,
            layer_idx,
            Self.CacheType.ValueIdx,
        )

    fn cache_length(self, bs_idx: Int) -> Int:
        return Int(self.cache_lengths[bs_idx])


struct PagedKVCacheCollection[
    type_: DType,
    kv_params_: KVCacheStaticParams,
    page_size: Int,
](KVCollectionT):
    alias type = type_
    alias kv_params = kv_params_
    alias CacheType = PagedKVCache[Self.type, Self.kv_params, page_size]

    var blocks: NDBuffer[Self.type, 6, MutableAnyOrigin]
    var cache_lengths: NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    var lookup_table: NDBuffer[DType.uint32, 2, MutableAnyOrigin]
    var max_seq_length: UInt32
    var max_cache_length: UInt32

    fn __init__(
        out self,
        blocks: NDBuffer[Self.type, 6],
        cache_lengths: NDBuffer[DType.uint32, 1],
        lookup_table: NDBuffer[DType.uint32, 2],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
    ):
        self.blocks = blocks
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length

    fn __copyinit__(out self, other: Self):
        self.blocks = other.blocks
        self.cache_lengths = other.cache_lengths
        self.lookup_table = other.lookup_table
        self.max_seq_length = other.max_seq_length
        self.max_cache_length = other.max_cache_length

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

    @staticmethod
    fn id() -> String:
        return String("PagedKVCacheCollection+", Self.type)

    fn get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        return self.CacheType(
            self.blocks,
            self.cache_lengths,
            self.lookup_table,
            self.max_seq_length,
            self.max_cache_length,
            layer_idx,
            Self.CacheType.KeyIdx,
        )

    fn get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        return self.CacheType(
            self.blocks,
            self.cache_lengths,
            self.lookup_table,
            self.max_seq_length,
            self.max_cache_length,
            layer_idx,
            Self.CacheType.ValueIdx,
        )

    fn cache_length(self, bs_idx: Int) -> Int:
        return Int(self.cache_lengths[bs_idx])


struct PagedKVCacheCollectionFA3Fallback[
    type_: DType,
    kv_params_: KVCacheStaticParams,
    page_size: Int,
](KVCollectionT):
    alias type = type_
    alias kv_params = kv_params_
    alias CacheType = PagedKVCacheFA3Fallback[
        Self.type, Self.kv_params, page_size
    ]

    var blocks: NDBuffer[Self.type, 5, MutableAnyOrigin]
    var cache_lengths: NDBuffer[DType.int32, 1, MutableAnyOrigin]
    var lookup_table: NDBuffer[DType.int32, 3, MutableAnyOrigin]
    var max_seq_length: UInt32
    var max_cache_length: UInt32

    fn __init__(
        out self,
        blocks: NDBuffer[Self.type, 5],
        cache_lengths: NDBuffer[DType.int32, 1],
        lookup_table: NDBuffer[DType.int32, 3],
        max_seq_length: UInt32,
        max_cache_length: UInt32,
    ):
        self.blocks = blocks
        self.cache_lengths = cache_lengths
        self.lookup_table = lookup_table
        self.max_seq_length = max_seq_length
        self.max_cache_length = max_cache_length

    fn __copyinit__(out self, other: Self):
        self.blocks = other.blocks
        self.cache_lengths = other.cache_lengths
        self.lookup_table = other.lookup_table
        self.max_seq_length = other.max_seq_length
        self.max_cache_length = other.max_cache_length

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

    @staticmethod
    fn id() -> String:
        return String("PagedKVCacheCollection+", Self.type)

    fn get_key_cache(self, layer_idx: Int) -> Self.CacheType:
        return self.CacheType(
            self.blocks,
            self.cache_lengths,
            self.lookup_table,
            self.max_seq_length,
            self.max_cache_length,
            layer_idx,
            Self.CacheType.KeyIdx,
        )

    fn get_value_cache(self, layer_idx: Int) -> Self.CacheType:
        return self.CacheType(
            self.blocks,
            self.cache_lengths,
            self.lookup_table,
            self.max_seq_length,
            self.max_cache_length,
            layer_idx,
            Self.CacheType.ValueIdx,
        )

    fn cache_length(self, bs_idx: Int) -> Int:
        return Int(self.cache_lengths[bs_idx])
