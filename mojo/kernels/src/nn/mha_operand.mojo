# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from buffer import NDBuffer
from collections import OptionalReg
from kv_cache.types import KVCacheT
from layout.layout import DimList
from memory import UnsafePointer
from utils import Variant


@register_passable("trivial")
struct _LengthHelper[ragged: Bool]:
    var valid_lengths: NDBuffer[DType.uint32, 1, *_]

    fn __init__(mut self, valid_lengths: NDBuffer[DType.uint32, 1, *_]):
        self.valid_lengths = valid_lengths

    @always_inline
    fn batch_size(self) -> UInt32:
        """Returns the batch size for the given operand."""
        return self.valid_lengths.dim[0]() - (1 if ragged else 0)

    @always_inline
    fn length(self, batch_idx: Int) -> UInt32:
        debug_assert(
            batch_idx < Int(self.batch_size()),
            "batch_idx out of bounds",
        )

        @parameter
        if ragged:
            return (
                self.valid_lengths[batch_idx + 1]
                - self.valid_lengths[batch_idx]
            )
        else:
            return self.valid_lengths[batch_idx]

    fn offset(self, batch_idx: Int) -> UInt32:
        constrained[
            ragged,
            "_LengthHelper::offset should not be invoked if ragged==True",
        ]()
        return self.valid_lengths[batch_idx]


@register_passable("trivial")
trait MHAOperand:
    """This serves as the trait to support arguments to our MHA kernel."""

    alias type: DType
    alias num_heads: Int
    alias depth: Int

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
        partition_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.type]]:
        ...

    @always_inline
    fn batch_size(self) -> UInt32:
        """Returns the batch size for the given operand."""
        ...

    @always_inline
    fn start_pos(self, batch_idx: Int) -> UInt32:
        """Returns the start position of the cache for a given batch index."""
        ...

    @always_inline
    fn length(self, batch_idx: Int) -> UInt32:
        """Returns the length of the operand for a given batch index."""
        ...

    @always_inline
    fn max_length(self) -> UInt32:
        """Returns the maximum length for the given operand."""
        ...


@register_passable("trivial")
struct KVCacheMHAOperand[
    cache_t: KVCacheT, //,
    ragged: Bool,
](MHAOperand):
    """An implementation for `mo.opaque` KVCacheT arguments to MHA kernels.

    We can eventually remove this trait and just add it as a sub-trait in the
    KVCacheT type, but we need to solve some cyclic dependencies first.
    """

    alias type = cache_t.type
    alias num_heads = cache_t.kv_params.num_heads
    alias depth = cache_t.kv_params.head_size

    var cache: cache_t
    var valid_lengths: NDBuffer[DType.uint32, 1]
    var length_helper: _LengthHelper[ragged]

    fn __init__(
        out self,
        cache: cache_t,
        valid_lengths: NDBuffer[DType.uint32, 1],
    ):
        self.cache = cache
        self.valid_lengths = valid_lengths
        self.length_helper = _LengthHelper[ragged](valid_lengths)

    @always_inline
    fn batch_size(self) -> UInt32:
        """Returns the batch size for the given operand."""
        return self.length_helper.batch_size()

    @always_inline
    fn block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        head_dim_idx: UInt32 = 0,
        partition_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.type]]:
        debug_assert(
            partition_idx == 0,
            "partition_idx must be 0 for KVCacheMHAOperand",
        )
        return self.cache.block_paged_ptr[tile_size](
            Int(batch_idx),
            Int(start_tok_idx),
            Int(head_idx),
            Int(head_dim_idx),
        )

    @always_inline
    fn start_pos(self, batch_idx: Int) -> UInt32:
        return self.cache.cache_length(batch_idx)

    @always_inline
    fn length(self, batch_idx: Int) -> UInt32:
        return self.length_helper.length(batch_idx) + self.cache.cache_length(
            batch_idx
        )

    @always_inline
    fn max_length(self) -> UInt32:
        return self.cache.max_context_length()


@register_passable("trivial")
struct NDBufferMHAOperand[
    type_: DType, rank: Int, shape: DimList, stride: DimList, //, ragged: Bool
](MHAOperand):
    """An implementation for NDBuffer arguments to MHA kernels."""

    alias type = type_
    alias num_heads = shape.get[rank - 2]()
    alias depth = shape.get[rank - 1]()

    var buffer: NDBuffer[Self.type, rank, shape, stride]
    var length_helper: OptionalReg[_LengthHelper[ragged]]
    var start_pos_nd: OptionalReg[NDBuffer[DType.uint32, 1]]
    var start_pos_padded: OptionalReg[UInt32]
    var max_length_val: OptionalReg[UInt32]

    # TODO create separate entry point for ragged and non-ragged cases
    fn __init__(
        out self,
        buffer: NDBuffer[Self.type, rank, shape, stride],
        valid_lengths: OptionalReg[NDBuffer[DType.uint32, 1]] = None,
        max_length: OptionalReg[UInt32] = None,
        start_pos: Variant[NDBuffer[DType.uint32, 1], UInt32] = UInt32(0),
    ):
        constrained[
            (ragged and rank == 3) or rank == 4,
            "Expected rank==3 if ragged=True, otherwise rank==4",
        ]()
        debug_assert(
            not ragged or valid_lengths,
            "Expected valid_lengths to be passed if ragged=True",
        )
        debug_assert(
            not ragged or max_length,
            "Expected max_length to be passed if ragged=True",
        )
        self.buffer = buffer
        if start_pos.isa[NDBuffer[DType.uint32, 1]]():
            self.start_pos_nd = start_pos[NDBuffer[DType.uint32, 1]]
            self.start_pos_padded = None
        else:
            self.start_pos_nd = None
            self.start_pos_padded = start_pos[UInt32]

        self.max_length_val = max_length
        if valid_lengths:
            self.length_helper = _LengthHelper[ragged](
                rebind[valid_lengths.T](valid_lengths.value())
            )
        else:
            self.length_helper = None

    @always_inline
    fn batch_size(self) -> UInt32:
        """Returns the batch size for the given operand."""
        if self.length_helper:
            return self.length_helper.value().batch_size()
        else:
            return self.buffer.dim[0]()

    @always_inline
    fn block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        head_dim_idx: UInt32 = 0,
        partition_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.type]]:
        debug_assert(
            partition_idx == 0,
            "partition_idx must be 0 for NDBufferMHAOperand",
        )

        @parameter
        if ragged:
            tok_offset = self.length_helper.value().offset(Int(batch_idx))
            ret_ptr = self.buffer._offset(
                (
                    Int(tok_offset + start_tok_idx),
                    Int(head_idx),
                    Int(head_dim_idx),
                )
            )
        else:
            ret_ptr = self.buffer._offset(
                (
                    Int(batch_idx),
                    Int(start_tok_idx),
                    Int(head_idx),
                    Int(head_dim_idx),
                )
            )
        return rebind[UnsafePointer[Scalar[Self.type]]](ret_ptr)

    @always_inline
    fn length(self, batch_idx: Int) -> UInt32:
        if self.length_helper:
            return self.length_helper.value().length(batch_idx)
        else:
            # NDBuffer path assumes BSHD layout and all cache entries have
            # the same length.
            return self.buffer.dim[1]()

    @always_inline
    fn start_pos(self, batch_idx: Int) -> UInt32:
        if self.start_pos_nd:
            debug_assert(batch_idx < self.start_pos_nd.value().dim[0]())
            return self.start_pos_nd.value()[batch_idx]

        return self.start_pos_padded.value()

    @always_inline
    fn max_length(self) -> UInt32:
        if self.max_length_val:
            return self.max_length_val.value()
        else:
            debug_assert(
                not ragged,
                "This branch should not be hit for ragged inputs",
            )
            return self.buffer.dim[1]()


@register_passable("trivial")
struct PartitionedNDBufferMHAOperand[
    type_: DType,
    rank: Int,
    shape: DimList,
    stride: DimList,
](MHAOperand):
    """An implementation for NDBuffer arguments to MHA kernels for partitioned
    decoding.
    """

    alias type = type_
    alias num_heads = shape.get[rank - 2]()
    alias depth = shape.get[rank - 1]()

    var buffer: NDBuffer[Self.type, rank, shape, stride]
    var start_pos_nd: OptionalReg[NDBuffer[DType.uint32, 1]]
    var max_length_val: UInt32

    fn __init__(
        out self,
        buffer: NDBuffer[Self.type, rank, shape, stride],
        max_length: UInt32,
        start_pos: OptionalReg[NDBuffer[DType.uint32, 1]] = None,
    ):
        constrained[
            rank == 4,
            "Expected rank==4 for PartitionedNDBufferMHAOperand",
        ]()
        self.buffer = buffer
        self.start_pos_nd = start_pos
        self.max_length_val = max_length

    @always_inline
    fn batch_size(self) -> UInt32:
        """Returns the batch size for the given operand."""
        return self.buffer.dim[1]()

    @always_inline
    fn block_paged_ptr[
        tile_size: Int
    ](
        self,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        head_dim_idx: UInt32 = 0,
        partition_idx: UInt32 = 0,
    ) -> UnsafePointer[Scalar[Self.type]]:
        debug_assert(
            start_tok_idx == 0,
            "start_tok_idx must be 0 during partitioned decoding",
        )
        return self.buffer._offset(
            (
                Int(partition_idx),
                Int(batch_idx),
                Int(head_idx),
                Int(head_dim_idx),
            )
        )

    @always_inline
    fn length(self, batch_idx: Int) -> UInt32:
        return 1  # during token gen, this is always 1

    @always_inline
    fn start_pos(self, batch_idx: Int) -> UInt32:
        if self.start_pos_nd:
            debug_assert(batch_idx < self.start_pos_nd.value().dim[0]())
            return self.start_pos_nd.value()[batch_idx]
        else:
            return 0

    @always_inline
    fn max_length(self) -> UInt32:
        return self.max_length_val
