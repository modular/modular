# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from buffer import NDBuffer
from kv_cache.types import KVCacheT
from layout.layout import DimList
from memory import UnsafePointer


@register_passable("trivial")
trait MHAOperand:
    """This serves as the trait to support arguments to our MHA kernel."""

    alias type: DType

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
    ) -> UnsafePointer[Scalar[Self.type]]:
        ...

    @always_inline
    fn cache_length(self, batch_idx: Int) -> Int:
        """Returns the length of the cache for a given batch index."""
        ...

    @always_inline
    fn max_context_length(self) -> UInt32:
        """Returns the maximum cache length in a given batch index."""
        ...


@register_passable("trivial")
struct KVCacheMHAOperand[cache_t: KVCacheT](MHAOperand):
    """An implementation for `mo.opaque` KVCacheT arguments to MHA kernels.

    We can eventually remove this trait and just add it as a sub-trait in the
    KVCacheT type, but we need to solve some cyclic dependencies first.
    """

    alias type = cache_t.type
    var cache: cache_t

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
    ) -> UnsafePointer[Scalar[Self.type]]:
        return self.cache.block_paged_ptr[tile_size](
            Int(batch_idx), Int(start_tok_idx), Int(head_idx), Int(head_dim_idx)
        )

    @always_inline
    fn cache_length(self, batch_idx: Int) -> Int:
        return self.cache.cache_length(batch_idx)

    @always_inline
    fn max_context_length(self) -> UInt32:
        return self.cache.max_context_length()


@register_passable("trivial")
struct NDBufferMHAOperand[
    type_: DType, rank: Int, shape: DimList, stride: DimList
](MHAOperand):
    """An implementation for NDBuffer arguments to MHA kernels."""

    alias type = type_
    var buffer: NDBuffer[Self.type, rank, MutableAnyOrigin, shape, stride]

    fn __init__(
        out self,
        buffer: NDBuffer[Self.type, rank, MutableAnyOrigin, shape, stride],
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
    ) -> UnsafePointer[Scalar[Self.type]]:
        var ret_ptr = self.buffer._offset(
            (
                Int(batch_idx),
                Int(start_tok_idx),
                Int(head_idx),
                Int(head_dim_idx),
            )
        )
        return rebind[UnsafePointer[Scalar[Self.type]]](ret_ptr)

    @always_inline
    fn cache_length(self, batch_idx: Int) -> Int:
        # NDBuffer path assumes BSHD layout and all cache entries have
        # the same length.
        return self.buffer.dim[1]()

    @always_inline
    fn max_context_length(self) -> UInt32:
        return self.buffer.dim[1]()
