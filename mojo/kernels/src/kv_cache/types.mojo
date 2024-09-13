# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from buffer import NDBuffer, DimList, Dim
from os import abort
from utils import StaticIntTuple

# TODO this is to make moving this value around easier. We should realistically
# make this a buffer, but lifetime management of buffers is challenging.
# TODO BEFORE MERGE cut linear issue to clean this up.
alias _default_max_batch_size = 32


@value
@register_passable("trivial")
struct KVCacheLayout(EqualityComparable):
    var _val: Int

    alias BHSD = KVCacheLayout(0)
    alias BSHD = KVCacheLayout(1)

    @always_inline("nodebug")
    fn __eq__(self, rhs: KVCacheLayout) -> Bool:
        return self._val == rhs._val

    @always_inline("nodebug")
    fn __ne__(self, rhs: KVCacheLayout) -> Bool:
        return not (self == rhs)


@value
@register_passable("trivial")
struct KVCacheStaticParams(EqualityComparable):
    var num_heads: Int
    var head_size: Int
    var layout: KVCacheLayout

    @always_inline("nodebug")
    fn __eq__(self, rhs: KVCacheStaticParams) -> Bool:
        return (
            self.num_heads == rhs.num_heads
            and self.head_size == rhs.head_size
            and self.layout == rhs.layout
        )

    @always_inline("nodebug")
    fn __ne__(self, rhs: KVCacheStaticParams) -> Bool:
        return not (self == rhs)


# TODO rename this to contiguous
@value
@register_passable("trivial")
struct ContiguousKVCache[
    type: DType,
    kv_params: KVCacheStaticParams,
    _max_batch_size: Int = _default_max_batch_size,
]():
    """Wrapper for the ContiguousKVCache of a given layer in the transformer model.

    This abstracts the Pointer indirection for accessing the ContiguousKVCache for a
    given batch entry.

    THIS IS THE TYPE THAT IS PASSED TO KV PROJECTION AND FLASH ATTENTION KERNELS.
    """

    alias _internal_block_shape = DimList(
        Dim(), Dim(), kv_params.num_heads, kv_params.head_size
    ) if kv_params.layout == KVCacheLayout.BSHD else DimList(
        Dim(), kv_params.num_heads, Dim(), kv_params.head_size
    )
    alias single_block_shape = DimList(
        Self._internal_block_shape.get[1](),
        Self._internal_block_shape.get[2](),
        Self._internal_block_shape.get[3](),
    )
    var _block: NDBuffer[type, 4, Self._internal_block_shape]
    var cache_lengths: StaticIntTuple[_max_batch_size]
    var batch_size: Int

    @always_inline
    fn _get_idx_tuple(
        self, bs_idx: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int
    ) -> StaticIntTuple[4]:
        debug_assert(
            bs_idx < self.batch_size, "KVCache batch_size out of range"
        )
        debug_assert(
            head_idx < kv_params.num_heads, "KVCache head_idx out of range"
        )
        debug_assert(
            head_dim_idx < kv_params.head_size,
            "KVCache head_dim_idx is out of range",
        )

        @parameter
        if kv_params.layout == KVCacheLayout.BHSD:
            debug_assert(
                tok_idx < self._block.dim[2](), "KVCache tok_idx out of range"
            )
            return (
                bs_idx,
                head_idx,
                tok_idx,
                head_dim_idx,
            )
        elif kv_params.layout == KVCacheLayout.BSHD:
            debug_assert(
                tok_idx < self._block.dim[1](), "KVCache tok_idx out of range"
            )
            return (
                bs_idx,
                tok_idx,
                head_idx,
                head_dim_idx,
            )
        else:
            constrained[False, "unsupported layout"]()
            return StaticIntTuple[4]()

    fn __init__(
        inout self,
        block: NDBuffer[type, 4, Self._internal_block_shape],
        cache_lengths: StaticIntTuple[_max_batch_size],
        batch_size: Int,
        num_heads: Int,
        head_size: Int,
    ):
        debug_assert(
            batch_size <= _max_batch_size,
            "Expected batch_size ("
            + str(batch_size)
            + ") <= _max_batch_size ("
            + str(_max_batch_size)
            + ")",
        )
        self._block = block
        self.cache_lengths = cache_lengths
        self.batch_size = batch_size

    @staticmethod
    fn id() -> String:
        return (
            "ContiguousKVCache+"
            + str(type)
            + "+"
            + str(kv_params.num_heads)
            + "+"
            + str(kv_params.head_size)
        )

    @always_inline
    fn cache_length(self, batch_idx: Int) -> Int:
        debug_assert(
            batch_idx < self.batch_size, "KVCache batch_idx is out of bounds"
        )
        return self.cache_lengths[batch_idx]

    # TODO could we load entire blocks as tiles? Would that be more performant?
    @always_inline
    fn load[
        width: Int
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        type, width
    ]:
        var idx = self._get_idx_tuple(bs, head_idx, tok_idx, head_dim_idx)
        return rebind[SIMD[type, width]](self._block.load[width=width](idx))

    @always_inline
    fn store[
        width: Int
    ](
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[type, width],
    ):
        var idx = self._get_idx_tuple(bs, head_idx, tok_idx, head_dim_idx)
        self._block.store[width=width](idx, rebind[SIMD[type, width]](val))

    @always_inline
    fn block(
        self,
        batch_idx: Int,
        new_seq_len: Int,
    ) -> NDBuffer[type, 3, Self.single_block_shape]:
        """Retrieve the NDBuffer containing the cache for a given batch index.

        The caller is responsible for passing `new_seq_len` which is the
        length of the newly encoded tokens. The internal cache_len is not updated
        until we complete the forward pass, so the actual seq_length of the cache should
        be `cache_len[batch_idx] + new_seq_len`.
        """
        var idx = self._get_idx_tuple(batch_idx, 0, 0, 0)
        var offset_ptr = self._block._offset(idx)
        var ret_shape: StaticIntTuple[3]
        var ret_strides = StaticIntTuple[3](
            self._block.dynamic_stride[1],
            self._block.dynamic_stride[2],
            self._block.dynamic_stride[3],
        )

        @parameter
        if kv_params.layout == KVCacheLayout.BSHD:
            ret_shape = StaticIntTuple[3](
                self.cache_length(batch_idx) + new_seq_len,
                kv_params.num_heads,
                kv_params.head_size,
            )
        elif kv_params.layout == KVCacheLayout.BHSD:
            ret_shape = StaticIntTuple[3](
                kv_params.num_heads,
                self.cache_length(batch_idx) + new_seq_len,
                kv_params.head_size,
            )
        else:
            constrained[False, "Unsupported layout"]()
            ret_shape = StaticIntTuple[3](0)

        return NDBuffer[type, 3, Self.single_block_shape](
            offset_ptr, ret_shape, ret_strides
        )


struct ContiguousKVCacheCollection[
    type: DType,
    kv_params: KVCacheStaticParams,
    _max_batch_size: Int = _default_max_batch_size,
](CollectionElement):
    """This is a "view" of the cache for the given sequences
    in the batch.

    This object does not own the underlying buffers in k_cache and v_cache,
    it's borrowing them from the BlockWrappers in our KVCacheManager.
    It does own the Pointer[NDBuffer[type, 3]] and valid_lengths buffer
    """

    # TODO outer should be list, inner can be Pointer?
    alias KeyCacheType = ContiguousKVCache[type, kv_params, _max_batch_size]
    alias ValueCacheType = ContiguousKVCache[type, kv_params, _max_batch_size]
    var key_cache: NDBuffer[type, 5]
    var value_cache: NDBuffer[type, 5]
    var cache_lengths: StaticIntTuple[_max_batch_size]
    var seq_ids: List[Int]
    var num_layers: Int
    var batch_size: Int
    var max_seq_len: Int

    fn __init__(
        inout self,
        key_cache: NDBuffer[type, 5],
        value_cache: NDBuffer[type, 5],
        cache_lengths: StaticIntTuple[_max_batch_size],
        seq_ids: List[Int],
        num_layers: Int,
        batch_size: Int,
    ):
        debug_assert(key_cache.dim[0]() == num_layers, "invalid key_cache size")
        debug_assert(
            value_cache.dim[0]() == num_layers, "invalid value_cache size "
        )
        debug_assert(
            batch_size < _max_batch_size, "KVCache batch_size > _max_batch_size"
        )

        self.key_cache = key_cache
        self.value_cache = value_cache
        self.seq_ids = seq_ids

        self.num_layers = num_layers
        self.batch_size = batch_size

        @parameter
        if kv_params.layout == KVCacheLayout.BHSD:
            self.max_seq_len = key_cache.dim[3]()
        elif kv_params.layout == KVCacheLayout.BSHD:
            self.max_seq_len = key_cache.dim[2]()
        else:
            self.max_seq_len = abort[Int]("unsupported layout")

        self.cache_lengths = cache_lengths

    fn __copyinit__(inout self, other: Self):
        self.key_cache = other.key_cache
        self.value_cache = other.value_cache
        self.seq_ids = other.seq_ids
        self.cache_lengths = other.cache_lengths
        self.num_layers = other.num_layers
        self.batch_size = other.batch_size
        self.max_seq_len = other.max_seq_len

    fn __moveinit__(inout self, owned other: Self):
        self.key_cache = other.key_cache
        self.value_cache = other.value_cache
        self.seq_ids = other.seq_ids
        self.cache_lengths = other.cache_lengths
        self.num_layers = other.num_layers
        self.batch_size = other.batch_size
        self.max_seq_len = other.max_seq_len

    @staticmethod
    fn id() -> String:
        return (
            "KVCacheCollection+"
            + str(type)
            + "+"
            + str(kv_params.num_heads)
            + "+"
            + str(kv_params.head_size)
        )

    fn get_key_cache(self, layer_idx: Int) -> Self.KeyCacheType:
        var layer_size = self.batch_size * kv_params.num_heads * self.max_seq_len * kv_params.head_size
        var k_shape = StaticIntTuple[4](
            self.key_cache.dim[1](),
            self.key_cache.dim[2](),
            self.key_cache.dim[3](),
            self.key_cache.dim[4](),
        )
        var layer_key_cache = NDBuffer[
            type, 4, Self.KeyCacheType._internal_block_shape
        ](self.key_cache.data + (layer_idx * layer_size), k_shape)
        return Self.KeyCacheType(
            layer_key_cache,
            self.cache_lengths,
            self.batch_size,
        )

    fn get_value_cache(self, layer_idx: Int) -> Self.ValueCacheType:
        var layer_size = self.batch_size * kv_params.num_heads * self.max_seq_len * kv_params.head_size
        var v_shape = StaticIntTuple[4](
            self.value_cache.dim[1](),
            self.value_cache.dim[2](),
            self.value_cache.dim[3](),
            self.value_cache.dim[4](),
        )

        var layer_value_cache = NDBuffer[
            type, 4, Self.ValueCacheType._internal_block_shape
        ](self.value_cache.data + (layer_idx * layer_size), v_shape)

        return Self.ValueCacheType(
            layer_value_cache, self.cache_lengths, self.batch_size
        )

    fn get_seq_ids(self) -> List[Int]:
        return self.seq_ids

    fn cache_length(self, batch_idx: Int) -> Int:
        debug_assert(
            batch_idx < self.batch_size, "KVCache batch_idx is out of bounds"
        )
        return self.cache_lengths[batch_idx]

    fn incr_cache_length(inout self, batch_idx: Int, inc: Int):
        debug_assert(
            batch_idx < self.batch_size, "KVCache batch_idx is out of bounds"
        )
        self.cache_lengths[batch_idx] += inc
