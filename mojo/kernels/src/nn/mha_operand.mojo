# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from kv_cache.types import KVCacheT
from memory import UnsafePointer
from buffer import NDBuffer
from layout.layout import DimList


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
            int(batch_idx), int(start_tok_idx), int(head_idx), int(head_dim_idx)
        )


struct NDBufferMHAOperand[type_: DType, rank: Int, shape: DimList](MHAOperand):
    """An implementation for NDBuffer arguments to MHA kernels."""

    alias type = type_
    var buffer: NDBuffer[Self.type, rank, shape]

    fn __init__(out self, buffer: NDBuffer[Self.type, rank, shape]):
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
                int(batch_idx),
                int(start_tok_idx),
                int(head_idx),
                int(head_dim_idx),
            )
        )
        return rebind[UnsafePointer[Scalar[Self.type]]](ret_ptr)
