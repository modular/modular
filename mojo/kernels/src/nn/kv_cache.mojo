# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from sys import has_neon
from buffer.list import DimList
from random import rand
from buffer.buffer import NDBuffer, _compute_nd_index
from math import ceildiv, isclose
from nn.softmax import softmax
from utils.index import Index
from testing import assert_true
from nn.mha import _naive_attention
from algorithm import elementwise
from nn.flash_attention import flash_attention
from benchmark import run
from collections import Set
from driver import Device, CPUDescriptor, get_cuda_device, DeviceMemory
from driver.tensor import Tensor
from driver.tensor_slice import TensorSlice
from gpu.host import Function, Context
from tensor import TensorSpec, TensorShape
from gpu import (
    WARP_SIZE,
    BlockIdx,
    ThreadIdx,
    barrier,
    lane_id,
)


trait KVCache:
    fn get_valid_lengths(self) -> NDBuffer[DType.int64, 1]:
        ...

    fn load[
        type: DType, width: Int
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        type, width
    ]:
        ...

    @always_inline
    fn store[
        type: DType, width: Int
    ](
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[type, width],
    ):
        ...


# TODO: too many parameters, cut back on that.
# @value
# @register_passable
struct PagedKVCache[
    type: DType,
    transpose_blocks: Bool = False,
](KVCache):
    alias block_t = NDBuffer[type, 3]
    # TODO make this a Pointer, not a List.
    alias block_list_t = List[Self.block_t]
    var blocks: List[Self.block_list_t]
    var valid_lengths: NDBuffer[DType.int64, 1]
    var num_heads: Int
    var head_size: Int
    var block_size: Int

    @always_inline
    fn _split_indices_on_block(self, tok_idx: Int) -> StaticIntTuple[2]:
        return divmod(tok_idx, self.block_size)

    @staticmethod
    @always_inline
    fn _get_idx_within_block(
        head_idx: Int, idx_in_block: Int, head_dim_idx: Int
    ) -> StaticIntTuple[3]:
        @parameter
        if transpose_blocks:
            return (head_idx, head_dim_idx, idx_in_block)
        else:
            return (head_idx, idx_in_block, head_dim_idx)

    fn __init__(
        inout self,
        blocks: List[Self.block_list_t],
        valid_lengths: NDBuffer[DType.int64, 1],
        num_heads: Int,
        head_size: Int,
        block_size: Int,
    ):
        self.blocks = blocks
        self.valid_lengths = valid_lengths
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size

    fn __moveinit__(inout self, owned other: Self):
        self.blocks = other.blocks^
        self.valid_lengths = other.valid_lengths
        self.num_heads = other.num_heads
        self.head_size = other.head_size
        self.block_size = other.block_size

    fn __copyinit__(inout self, other: Self):
        constrained[False, "don't copy"]()
        self.blocks = other.blocks
        self.valid_lengths = other.valid_lengths
        self.num_heads = other.num_heads
        self.head_size = other.head_size
        self.block_size = other.block_size

    fn get_valid_lengths(self) -> NDBuffer[DType.int64, 1]:
        return self.valid_lengths

    # TODO could we load entire blocks as tiles? Would that be more performant?
    @always_inline
    fn load[
        _type: DType, width: Int
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        _type, width
    ]:
        constrained[_type == type, "Expected type to match"]()

        var block_split_idx = self._split_indices_on_block(tok_idx)
        var block_idx = block_split_idx[0]
        var idx_in_block = block_split_idx[1]

        var batch = self.blocks[bs]
        var block = batch[block_idx]

        var idx = Self._get_idx_within_block(
            head_idx, idx_in_block, head_dim_idx
        )

        return rebind[SIMD[_type, width]](block.load[width=width](idx))

    @always_inline
    fn store[
        _type: DType, width: Int
    ](
        self,
        bs: Int,
        head_idx: Int,
        tok_idx: Int,
        head_dim_idx: Int,
        val: SIMD[_type, width],
    ):
        constrained[_type == type, "Expected type to match"]()
        var block_split_idx = self._split_indices_on_block(tok_idx)
        var block_idx = block_split_idx[0]
        var idx_in_block = block_split_idx[1]

        var batch = self.blocks[bs]
        var block = batch[block_idx]

        var idx = Self._get_idx_within_block(
            head_idx, idx_in_block, head_dim_idx
        )

        block.store[width=width](idx, rebind[SIMD[type, width]](val))


@value
@register_passable
struct SplitBatchKVCache[type: DType, transpose: Bool]:
    # struct SplitBatchKVCache[type: DType, transpose: Bool](KVCache):
    """Wrapper for the KVCache of a given layer in the transformer model.

    This abstracts the Pointer indirection for accessing the KVCache for a
    given batch entry.

    THIS IS THE TYPE THAT IS PASSED TO KV PROJECTION AND FLASH ATTENTION KERNELS.
    """

    alias block_t = NDBuffer[type, 3]
    var blocks: Pointer[Self.block_t]
    var valid_lengths: NDBuffer[DType.int64, 1]
    var batch_size: Int
    var num_heads: Int
    var head_size: Int

    @staticmethod
    @always_inline
    fn _get_idx_tuple(
        head_idx: Int, tok_idx: Int, head_dim_idx: Int
    ) -> StaticIntTuple[3]:
        @parameter
        if transpose:
            return (head_idx, head_dim_idx, tok_idx)
        else:
            return (head_idx, tok_idx, head_dim_idx)

    fn __init__(
        inout self,
        blocks: Pointer[Self.block_t],
        valid_lengths: NDBuffer[DType.int64, 1],
        num_heads: Int,
        head_size: Int,
    ):
        self.blocks = blocks
        self.valid_lengths = valid_lengths
        self.batch_size = len(valid_lengths)
        self.num_heads = num_heads
        self.head_size = head_size

    fn get_valid_lengths(self) -> NDBuffer[DType.int64, 1]:
        return self.valid_lengths

    # TODO could we load entire blocks as tiles? Would that be more performant?
    @always_inline
    fn load[
        width: Int
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        type, width
    ]:
        if bs > self.batch_size:
            # don't want to raise right now, TODO Raise
            print("invalid batch size")

        var batch = self.blocks[bs]

        var idx = Self._get_idx_tuple(head_idx, tok_idx, head_dim_idx)
        return rebind[SIMD[type, width]](batch.load[width=width](idx))

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
        var batch = self.blocks[bs]

        var idx = Self._get_idx_tuple(head_idx, tok_idx, head_dim_idx)
        batch.store[width=width](idx, rebind[SIMD[type, width]](val))


struct SplitBatchKVCacheCollection[
    type: DType,
    is_cpu: Bool,
    transpose_k: Bool,
]:
    """This is a "view" of the cache for the given sequences
    in the batch.

    This object does not own the underlying buffers in k_cache and v_cache,
    it's borrowing them from the BlockWrappers in our KVCacheManager.
    It does own the Pointer[NDBuffer[type, 3]] and valid_lengths buffer
    """

    # TODO outer should be list, inner can be Pointer?
    alias cache_t = List[DeviceMemory]
    var key_cache: Self.cache_t
    var value_cache: Self.cache_t
    var valid_lengths_tensor: Tensor[DType.int64, 1]
    var valid_lengths: NDBuffer[DType.int64, 1]
    var seq_ids: List[Int]
    var num_layers: Int
    var batch_size: Int
    var head_size: Int
    var num_heads: Int

    fn __init__(
        inout self,
        owned key_cache: self.cache_t,
        owned value_cache: self.cache_t,
        owned valid_lengths: Tensor[DType.int64, 1],
        seq_ids: List[Int],
        num_layers: Int,
        batch_size: Int,
        head_size: Int,
        num_heads: Int,
    ) raises:
        debug_assert(len(key_cache) == num_layers, "invalid key_cache size")
        debug_assert(len(value_cache) == num_layers, "invalid value_cache size")

        self.key_cache = key_cache^
        self.value_cache = value_cache^
        self.seq_ids = seq_ids

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.valid_lengths_tensor = valid_lengths^
        self.valid_lengths = ndbuffer_view_from_tensor[DType.int64, 1, 1,](
            self.valid_lengths_tensor,
            0,
            StaticIntTuple[1](
                batch_size,
            ),
        )

    fn __moveinit__(inout self, owned other: Self):
        self.key_cache = other.key_cache^
        self.value_cache = other.value_cache^
        self.seq_ids = other.seq_ids
        self.valid_lengths = other.valid_lengths
        self.num_layers = other.num_layers
        self.head_size = other.head_size
        self.num_heads = other.num_heads
        self.batch_size = other.batch_size
        self.valid_lengths_tensor = other.valid_lengths_tensor^

    fn __copyinit__(inout self, other: Self):
        constrained[False, "don't copy!"]()
        self.key_cache = other.key_cache
        self.value_cache = other.value_cache
        self.seq_ids = other.seq_ids
        self.valid_lengths = other.valid_lengths
        self.num_layers = other.num_layers
        self.head_size = other.head_size
        self.num_heads = other.num_heads
        self.batch_size = other.batch_size
        self.valid_lengths_tensor = other.valid_lengths_tensor

    fn get_key_cache(
        self, layer_idx: Int
    ) -> SplitBatchKVCache[type, transpose_k]:
        var device_mem_ref = self.key_cache.__get_ref(layer_idx)
        var k_cache_ptr = device_mem_ref[].data_unsafe().address.bitcast[
            NDBuffer[type, 3]
        ]()
        return SplitBatchKVCache[type, transpose_k](
            k_cache_ptr,
            self.valid_lengths,
            self.num_heads,
            self.head_size,
        )

    fn get_value_cache(self, layer_idx: Int) -> SplitBatchKVCache[type, False]:
        var device_mem_ref = self.value_cache.__get_ref(layer_idx)
        var v_cache_ptr = device_mem_ref[].data_unsafe().address.bitcast[
            NDBuffer[type, 3]
        ]()
        return SplitBatchKVCache[type, False](
            v_cache_ptr,
            self.valid_lengths,
            self.num_heads,
            self.head_size,
        )

    fn get_seq_ids(self) -> List[Int]:
        return self.seq_ids

    fn get_valid_lengths(self) -> NDBuffer[DType.int64, 1]:
        return self.valid_lengths


struct SplitBatchCacheBlockWrapper[
    type: DType,
    transpose_k: Bool,
    token_type: DType = DType.int64,
](CollectionElement):
    """Wrapper for a KV Cache object for a single sequence in the cache.

    This stores all of the KV Cache entries for each layer for a given sequence.
    It stores state related to:
    - whether the block is in use by an active request
    - the length of the context in the cache (we always overallocate to max_seq_len)
    - the tokens assigned to the context in the cache
    """

    var k_blocks: NDBuffer[type, 4]
    var v_blocks: NDBuffer[type, 4]

    # TODO make this tensor too.
    var tokens: NDBuffer[token_type, 1]
    var in_use: Bool
    var valid_length: Int
    var max_seq_len: Int
    var num_layers: Int
    var head_size: Int
    var num_heads: Int
    var this_device: Device
    var other_device: Device

    fn __init__(
        inout self,
        max_seq_len: Int,
        num_layers: Int,
        head_size: Int,
        num_heads: Int,
        k_blocks: NDBuffer[type, 4],
        v_blocks: NDBuffer[type, 4],
        inout other_device: Device,
        inout this_device: Device,
    ) raises:
        self.v_blocks = k_blocks
        self.k_blocks = v_blocks

        self.in_use = False

        # TODO use this_device.alloc
        var tokens_ptr = DTypePointer[token_type].alloc(max_seq_len)
        self.tokens = NDBuffer[token_type, 1](tokens_ptr, DimList(max_seq_len))
        self.valid_length = 0
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.head_size = head_size
        self.num_heads = num_heads
        self.this_device = this_device
        self.other_device = other_device

    fn __copyinit__(inout self, other: Self):
        constrained[False, "don't copy!"]()
        self.v_blocks = other.v_blocks
        self.k_blocks = other.k_blocks
        self.in_use = other.in_use

        # TODO tokens is not copy-safe, we need to change this.
        self.tokens = other.tokens
        self.valid_length = other.valid_length
        self.max_seq_len = other.max_seq_len
        self.num_layers = other.num_layers
        self.head_size = other.head_size
        self.num_heads = other.num_heads
        self.this_device = other.this_device
        self.other_device = other.other_device

    fn __moveinit__(inout self, owned other: Self):
        self.v_blocks = other.v_blocks
        self.k_blocks = other.k_blocks
        self.in_use = other.in_use
        self.tokens = other.tokens
        self.valid_length = other.valid_length
        self.max_seq_len = other.max_seq_len
        self.num_layers = other.num_layers
        self.head_size = other.head_size
        self.num_heads = other.num_heads
        self.this_device = other.this_device^
        self.other_device = other.other_device^

    fn get_key_cache(self, layer_idx: Int) raises -> NDBuffer[type, 3]:
        var offset_ptr = self.k_blocks.data + (
            layer_idx * self.max_seq_len * self.head_size * self.num_heads
        )
        var k_block_shape = StaticIntTuple[3](
            (self.num_heads, self.max_seq_len, self.head_size)
        ) if transpose_k else (
            self.num_heads,
            self.head_size,
            self.max_seq_len,
        )
        return NDBuffer[type, 3](offset_ptr, k_block_shape)

    fn get_value_cache(self, layer_idx: Int) raises -> NDBuffer[type, 3]:
        var offset_ptr = self.v_blocks.data + (
            layer_idx * self.max_seq_len * self.head_size * self.num_heads
        )
        var v_block_shape = StaticIntTuple[3](
            self.num_heads,
            self.head_size,
            self.max_seq_len,
        )
        return NDBuffer[type, 3](offset_ptr, v_block_shape)

    fn get_valid_length(self) -> Int:
        return self.valid_length

    fn __del__(owned self):
        """NOTE: This should be the only thing that deallocates blocks in the
        cache.

        All other uses of these blocks are considered borrowed.
        """
        self.tokens.data.free()

    fn reset(inout self):
        """Release this block from its existing sequence."""

        self.valid_length = 0
        self.in_use = False

    fn mark_in_use(inout self) raises:
        """Mark that this block is in use."""
        if self.in_use:
            raise "Already in use!"
        self.in_use = True

    fn step(inout self, token_ids: NDBuffer[token_type, 1]) raises:
        """Update the cache to note recent additions."""
        if not self.in_use:
            raise "Not in use!"

        var new_seq_len = token_ids.dim[0]()
        memcpy(
            self.tokens.data.offset(self.valid_length),
            token_ids.data,
            new_seq_len,
        )
        self.valid_length += new_seq_len


struct SplitBatchKVCacheManager[
    type: DType,
    is_cpu: Bool,  # TODO remove this in favor of passing device
    transpose_k: Bool,
]:
    """Manages a Batch-split KV cache across multiple user sessions.

    Each request is assigned a seq_id, which is associated with a set of buffers
    to store the key and value projections per layer.

    The order of calls for an active request is expected to be:
    * claim -- assigned blocks to the sequence and give it a unique id
    * step -- commit context encoding projections
    * foreach token generated:
        * fetch -- retrieve blocks based on a seq_id
        * step -- commit token generation projections
    * release -- mark blocks as not in use

    TODO this is not currently threadsafe, make it so
    """

    alias wrapper_t = SplitBatchCacheBlockWrapper[
        type,
        transpose_k,
    ]
    alias collection_t = SplitBatchKVCacheCollection[
        type,
        is_cpu,
        transpose_k,
    ]
    var blocks_buf: Tensor[type, 5]
    var blocks: List[Self.wrapper_t]
    var unused_blocks: Set[Int]
    var num_blocks: Int
    var max_batch_size: Int
    var max_seq_len: Int
    var seq_id_counter: Int
    var seq_id_to_block: Dict[Int, Int]
    var num_layers: Int
    var head_size: Int
    var num_heads: Int
    var other_device: Device
    var this_device: Device

    # TODO currently we allocate the exact right amount, we should
    # extend this to allow for over-allocation
    fn __init__(
        inout self,
        max_batch_size: Int,
        max_seq_len: Int,
        num_layers: Int,
        head_size: Int,
        num_heads: Int,
        inout other_device: Device,
        inout this_device: Device,
    ) raises:
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # TODO make a different argument for the number of blocks.
        self.num_blocks = self.max_batch_size
        self.blocks = List[self.wrapper_t](capacity=self.num_blocks * 2)
        self.blocks_buf = other_device.allocate(
            TensorSpec(
                type,
                (
                    self.num_blocks * 2,
                    num_layers,
                    num_heads,
                    max_seq_len,
                    head_size,
                ),
            )
        ).get_tensor[type, 5]()
        self.unused_blocks = Set[Int]()
        self.seq_id_counter = 0
        self.seq_id_to_block = Dict[Int, Int]()
        self.num_layers = num_layers
        self.head_size = head_size
        self.num_heads = num_heads
        self.this_device = this_device
        self.other_device = other_device

        var block_size = self.num_layers * self.max_seq_len * self.num_heads * self.head_size
        var fa_k_shape: StaticIntTuple[4]
        var fa_v_shape = StaticIntTuple[4](
            self.num_layers, self.num_heads, self.max_seq_len, self.head_size
        )

        @parameter
        if not transpose_k:
            fa_k_shape = StaticIntTuple[4](
                self.num_layers,
                self.num_heads,
                self.head_size,
                self.max_seq_len,
            )
        else:
            fa_k_shape = fa_v_shape

        for i in range(self.num_blocks):
            var k_offset = 2 * i * block_size

            # TODO use tensor slices
            var k_block = ndbuffer_view_from_tensor(
                self.blocks_buf,
                k_offset,
                fa_k_shape,
            )

            var v_offset = k_offset + block_size
            var v_block = ndbuffer_view_from_tensor(
                self.blocks_buf,
                v_offset,
                fa_v_shape,
            )
            self.blocks.append(
                Self.wrapper_t(
                    self.max_seq_len,
                    self.num_layers,
                    self.head_size,
                    self.num_heads,
                    k_block,
                    v_block,
                    self.other_device,
                    self.this_device,
                )
            )

            self.unused_blocks.add(i)

    fn claim(inout self, batch_size: Int) raises -> Self.collection_t:
        """Assign `batch_size` blocks for incoming requests.

        This returns a SplitBatchKVCacheCollection, which has buffers
        for each layer in the network. Each sequence is assigned a seq_id
        which uniquely identifies that sequence's entries in this cache.
        """
        if batch_size > self.max_batch_size:
            raise "batch size too large"

        var seq_ids = List[Int]()

        for bs in range(batch_size):
            # TODO make this atomic, account for collisions once we overflow
            var new_seq_id = self.seq_id_counter
            self.seq_id_counter += 1
            seq_ids.append(new_seq_id)

            var block_id = self.unused_blocks.pop()
            self.seq_id_to_block[new_seq_id] = block_id
            var block = self.blocks.__get_ref(block_id)
            block[].mark_in_use()
        return self.fetch(seq_ids)

    fn fetch(inout self, seq_ids: List[Int]) raises -> Self.collection_t:
        """Retrieves the pre-assigned blocks for the given seq_ids.

        if any of the seq_ids are not valid (e.g. no assigned blocks) then
        and error is raised.
        """
        var batch_size = len(seq_ids)

        # TODO check that we have the blocks to satisfy this request
        # outer dimension: num_layers. inner_dimension: batch_size
        var host_key_caches = List[DeviceMemory](capacity=self.num_layers)
        var host_value_caches = List[DeviceMemory](capacity=self.num_layers)
        for l in range(self.num_layers):
            host_key_caches.append(
                self.this_device.allocate(
                    sizeof[NDBuffer[type, 3]]() * batch_size
                )
            )
            host_value_caches.append(
                self.this_device.allocate(
                    sizeof[NDBuffer[type, 3]]() * batch_size
                )
            )
        var host_valid_lengths = self.this_device.allocate(
            TensorSpec(DType.int64, (batch_size,))
        ).get_tensor[DType.int64, 1]()
        for bs in range(batch_size):
            var seq_id = seq_ids.__get_ref(bs)
            if seq_id[] not in self.seq_id_to_block:
                raise "Unknown seq_id " + str(seq_id[])

            var block_id = self.seq_id_to_block[seq_id[]]
            var block = self.blocks.__get_ref(block_id)

            host_valid_lengths[bs] = block[].get_valid_length()

            for l in range(self.num_layers):
                var host_key_ref = host_key_caches.__get_ref(l)
                var host_key_ptr = host_key_ref[].data_unsafe().address.bitcast[
                    NDBuffer[type, 3]
                ]()
                var value_key_ref = host_value_caches.__get_ref(l)
                var host_value_ptr = value_key_ref[].data_unsafe().address.bitcast[
                    NDBuffer[type, 3]
                ]()

                host_key_ptr[bs] = block[].get_key_cache(l)
                host_value_ptr[bs] = block[].get_value_cache(l)
        var device_key_caches = List[DeviceMemory](capacity=self.num_layers)
        var device_value_caches = List[DeviceMemory](capacity=self.num_layers)
        var device_valid_lengths: Tensor[DType.int64, 1]

        # TODO remove this branch once we have no-op copy_tos on device_memory.
        @parameter
        if is_cpu:
            device_key_caches = host_key_caches^
            device_value_caches = host_value_caches^
            device_valid_lengths = host_valid_lengths^
        else:
            for l in range(self.num_layers):
                device_key_caches.append(
                    host_key_caches.__get_ref(l)[].copy_to(self.other_device)
                )
                device_value_caches.append(
                    host_value_caches.__get_ref(l)[].copy_to(self.other_device)
                )

            # copy valid lengths from CPU to other device
            device_valid_lengths = (
                host_valid_lengths.get_device_memory()
                .copy_to(self.other_device)
                .get_tensor[DType.int64, 1]()
            )

        return Self.collection_t(
            device_key_caches^,
            device_value_caches^,
            device_valid_lengths^,
            seq_ids,
            self.num_layers,
            batch_size,
            self.head_size,
            self.num_heads,
        )

    fn step(
        inout self,
        token_ids: NDBuffer[DType.int64, 2],
        inflight_cache: SplitBatchKVCacheCollection,
    ) raises:
        """Commits changes to the KVCache blocks.

        This is used to note that a KV projection step has occured and
        the values in these buffers have been written to. We note the new tokens
        in the blocks and update the valid_length counter.
        """
        var batch_size = token_ids.dim[0]()
        var new_seq_len = token_ids.dim[1]()
        var seq_ids = inflight_cache.get_seq_ids()
        for bs in range(batch_size):
            var token_id_offset = token_ids.data.offset(bs * new_seq_len)
            var token_view = NDBuffer[DType.int64, 1](
                token_id_offset, DimList(new_seq_len)
            )
            var seq_id = seq_ids[bs]
            var block = self.blocks.__get_ref(self.seq_id_to_block[seq_id])
            block[].step(token_view)

    fn release(inout self, seq_id: Int) raises:
        """Marks `seq_id` as no longer necessary, their blocks are reintroduced
        to the pool.

        """
        if seq_id not in self.seq_id_to_block:
            raise "seq_id not found: " + str(seq_id)

        var block_idx = self.seq_id_to_block.pop(seq_id)
        var block = self.blocks.__get_ref(block_idx)
        block[].reset()
        self.unused_blocks.add(block_idx)


fn test_generation_loop[
    type: DType = DType.float32, target: StringLiteral = "cpu"
]() raises:
    alias num_layers = 4
    alias head_size = 64
    alias num_heads = 4
    var max_seq_len = 100
    var max_batch_size = 8
    var num_iters = 10
    var this_device = Device(CPUDescriptor())
    var other_device: Device

    @parameter
    if target == "cpu":
        other_device = this_device
    else:
        other_device = get_cuda_device()
    var manager = SplitBatchKVCacheManager[type, target == "cpu", False](
        max_batch_size,
        max_seq_len,
        num_layers,
        head_size,
        num_heads,
        other_device,
        this_device,
    )
    var num_tokens = 10
    var batch_size = 4
    var token_ptr = DTypePointer[DType.int64].alloc(num_tokens * batch_size)
    var token_ids = NDBuffer[DType.int64, 2](
        token_ptr, DimList(batch_size, num_tokens)
    )

    # create Pointer for "next" tokens during token generation
    # TODO make this device.allocate
    var next_token_ptr = DTypePointer[DType.int64].alloc(1 * batch_size)
    var next_token_ids = NDBuffer[DType.int64, 2](
        next_token_ptr, DimList(batch_size, 1)
    )

    # STEP 1: Tell the manager that we have new requests, retrieve the cache and unique IDs
    var empty_kv = manager.claim(batch_size)
    # STEP 2: do a forward pass, fill in KV projections in the cache buffers for our prompt
    fake_kv_projection[type, target=target](empty_kv, token_ids)
    # STEP 3: notify the manager we modified empty_kv, update the state in the manager
    manager.step(token_ids, empty_kv)
    var cache_seq_ids = empty_kv.get_seq_ids()

    # TODO add more dynamism to each iteration
    var cache_size = 0
    for i in range(num_iters):
        # STEP 4: retrieve the kv cache for the given sequence ids.
        # Order of seq_ids shouldn't matter.
        # Whether they were claimed together shouldn't matter.
        var prefilled_kv = manager.fetch(cache_seq_ids)

        # STEP 5: do a forward pass, fill in KV projections in the cache buffers
        # for the token we generated on the previous iteration
        fake_kv_projection[type, target=target](prefilled_kv, next_token_ids)

        # STEP 6: notify the KV Cache manager we modified prefilled_kv, update the state in the manager
        manager.step(next_token_ids, prefilled_kv)

        # STEP 7: repeat...
    # assert on some internal state of the manager:
    assert_true(len(manager.seq_id_to_block) == 4)
    assert_true(manager.num_blocks - len(manager.unused_blocks) == 4)

    # assert on the values inside of our KV cache
    # TODO update this to work with GPUs
    @parameter
    if target == "cpu":
        for l in range(num_layers):
            var layer_cache = manager.fetch(cache_seq_ids)
            var k_cache = layer_cache.get_key_cache(l)
            var v_cache = layer_cache.get_value_cache(l)
            for bs in range(batch_size):
                for t in range(num_tokens + num_iters):
                    for h in range(num_heads):
                        for dh in range(head_size):
                            var k_c = k_cache.load[width=1](bs, h, t, dh)
                            var v_c = v_cache.load[width=1](bs, h, t, dh)
                            assert_true(k_c == t)
                            assert_true(v_c == t)
            _ = layer_cache^
    # STEP 8: notify the manager that these sequences have completed generation
    # evict them from our cache.
    for seq_id in cache_seq_ids:
        manager.release(seq_id[])
    assert_true(len(manager.seq_id_to_block) == 0)
    assert_true(manager.num_blocks - len(manager.unused_blocks) == 0)

    token_ptr.free()
    next_token_ptr.free()
    _ = manager^


fn fake_attn[
    type: DType, target: String = "cpu"
](
    cache: SplitBatchKVCacheCollection[type], hidden_state: NDBuffer[type, 3]
) raises:
    raise "fake attn not implemented"


fn fake_kv_projection[
    type: DType, target: String = "cpu"
](
    cache: SplitBatchKVCacheCollection[type],
    token_ids: NDBuffer[DType.int64, 2],
) raises:
    @parameter
    if target == "cuda":
        return fake_kv_projection_cuda[type](cache, token_ids)
    else:
        return fake_kv_projection_cpu[type](cache, token_ids)


fn fake_kv_projection_cuda[
    type: DType
](
    cache: SplitBatchKVCacheCollection[type],
    token_ids: NDBuffer[DType.int64, 2],
) raises:
    for l in range(cache.num_layers):
        var layer_key_cache = cache.get_key_cache(l)
        var layer_value_cache = cache.get_value_cache(l)
        fake_kv_projection_layer_cuda(
            layer_key_cache, layer_value_cache, token_ids
        )
        _ = layer_key_cache^
        _ = layer_value_cache^


fn fake_kv_projection_layer_cuda[
    type: DType,
](
    k_cache: SplitBatchKVCache[type],
    v_cache: SplitBatchKVCache[type],
    token_ids: NDBuffer[DType.int64, 2],
) raises:
    @parameter
    @__copy_capture(
        k_cache,
        v_cache,
        token_ids,
    )
    fn kernel():
        var new_seq_len = token_ids.dim[1]()

        var batch_idx = BlockIdx.x()
        var cached_seq_len = k_cache.get_valid_lengths()[batch_idx]

        var head_idx = BlockIdx.y()
        var head_dim_idx = ThreadIdx.x()
        for seq_idx in range(cached_seq_len, cached_seq_len + new_seq_len):
            k_cache.store(
                batch_idx,
                head_idx,
                seq_idx,
                head_dim_idx,
                SIMD[type, 1](seq_idx),
            )
            v_cache.store(
                batch_idx,
                head_idx,
                seq_idx,
                head_dim_idx,
                SIMD[type, 1](seq_idx),
            )

    var func = Function[kernel]()
    # TODO expand to more valid lengths
    var batch_size = token_ids.dim[0]()

    func(
        grid_dim=(batch_size, k_cache.num_heads),
        block_dim=k_cache.head_size,
    )


fn fake_kv_projection_cpu[
    type: DType
](
    cache: SplitBatchKVCacheCollection[type],
    token_ids: NDBuffer[DType.int64, 2],
):
    for l in range(cache.num_layers):
        var layer_key_cache = cache.get_key_cache(l)
        var layer_value_cache = cache.get_value_cache(l)
        fake_kv_projection_layer_cpu[type](
            layer_key_cache, layer_value_cache, token_ids
        )


fn fake_kv_projection_layer_cpu[
    type: DType
](
    k_cache: SplitBatchKVCache[type],
    v_cache: SplitBatchKVCache[type],
    token_ids: NDBuffer[DType.int64, 2],
):
    alias width = 1
    var cached_seq_lens = k_cache.get_valid_lengths()
    var batch_size = token_ids.dim[0]()
    var new_seq_len = token_ids.dim[1]()
    for bs in range(batch_size):
        var cached_seq_len = cached_seq_lens[bs]
        for head_idx in range(k_cache.num_heads):
            for seq_idx in range(cached_seq_len, cached_seq_len + new_seq_len):
                for head_dim_idx in range(k_cache.head_size):
                    k_cache.store(
                        bs,
                        head_idx,
                        seq_idx,
                        head_dim_idx,
                        Scalar[type](seq_idx),
                    )
                    v_cache.store(
                        bs,
                        head_idx,
                        seq_idx,
                        head_dim_idx,
                        Scalar[type](seq_idx),
                    )


# TODO specialized to block size, pass as parameter and include as known
# dimension in k/v_block_list
# TODO pass in position token to handle unrounded inputs
fn _naive_block_list_attention[
    type: DType,
    k_cache_t: KVCache,
    v_cache_t: KVCache,
](
    output: NDBuffer[type, 4],
    q: NDBuffer[type, 4],
    k_cache: k_cache_t,
    v_cache: v_cache_t,
    mask: NDBuffer[type, 2],
    kv_seq_len: Int,
    scale: Float32,
):
    """This kernel provides a simple block-list attention implementation for
    transformer models. It's intended as a POC and shouldn't be used in prod.
    """

    alias simd_size = simdwidthof[type]()

    var batch_size = q.dim[0]()
    var num_heads = q.dim[1]()
    var seq_len = q.dim[2]()
    var head_dim = q.dim[3]()

    # Allocate intermediate memory buffer.
    var score_size = batch_size * num_heads * seq_len * kv_seq_len
    var score_ptr = DTypePointer[type].alloc(score_size)
    var score = NDBuffer[type, 4](
        score_ptr, Index(batch_size, num_heads, seq_len, kv_seq_len)
    )

    batched_matmul[
        4,
        type,
        type,
        type,
        True,
    ](score, q, k_cache)

    @__copy_capture(score)
    @parameter
    @always_inline
    fn scale_and_mask[width: Int, _rank: Int](coords: StaticIntTuple[_rank]):
        var vec = score.load[width=width](rebind[StaticIntTuple[4]](coords))
        vec = vec * scale.cast[type]()
        vec = vec + mask.load[width=width](
            Index(coords[_rank - 2], coords[_rank - 1])
        )
        score.store[width=width](rebind[StaticIntTuple[4]](coords), vec)

    elementwise[scale_and_mask, simd_size, 4](score.dynamic_shape)

    try:
        softmax[type, simd_size, 4](
            score,
            score,
            axis=3,
        )
    except e:
        abort(e)

    batched_matmul[4, type, type, type, False](output, score, v_cache)
    score_ptr.free()


# TODO this is pretty specialized, let's give it a different name
@always_inline
fn batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    trans_b: Bool,
    cache_t: KVCache,
](
    c_buf: NDBuffer[c_type, rank],
    a_buf: NDBuffer[a_type, rank],
    b_buf: cache_t,
):
    var outer_batch_size = c_buf.dim[0]()
    var inner_batch_size = c_buf.dim[1]()
    var M = c_buf.dim[2]()
    var N = c_buf.dim[3]()
    var K = a_buf.dim[3]()

    # TODO this is hella slow. parallelize, vectorize, and tile
    # TODO also collapse obs/ibs into one thing, we can unpack in the KVCache obj
    for obs in range(outer_batch_size):
        for ibs in range(inner_batch_size):
            for m in range(M):
                for n in range(N):
                    var accum = Scalar[c_type](0.0)
                    for k in range(K):
                        var a_val = a_buf[obs, ibs, m, k].cast[c_type]()
                        var b_val = SIMD[c_type, 1]()

                        @parameter
                        if trans_b:
                            b_val = b_buf.load[b_type, 1](obs, ibs, n, k).cast[
                                c_type
                            ]()
                        else:
                            b_val = b_buf.load[b_type, 1](obs, ibs, k, n).cast[
                                c_type
                            ]()

                        accum += a_val * b_val
                    c_buf[(obs, ibs, m, n)] = accum


@always_inline
fn is_ndbuffer_close[
    rank: Int, type: DType
](
    a: NDBuffer[type, rank],
    b: NDBuffer[type, rank],
    abs_tol: Scalar[type] = 1e-5,
    rel_tol: Scalar[type] = 1e-4,
    print_wrong_value: Bool = True,
    max_num_print: Int = 1,
) -> Bool:
    """Compare if two NDBuffers are close within input tolerance.

    It prints out up to `max_num_print` difference values if `print_wrong_value`
    is set to True.

    Returns:
        Returns True if they are within tolerance.
    """
    debug_assert(
        a.dynamic_shape == b.dynamic_shape
        and a.dynamic_stride == b.dynamic_stride,
        "Input buffers must have the same shape and stride.",
    )

    var num_errs = 0
    var is_close = True

    for i in range(a.num_elements()):
        var nd_idx = _compute_nd_index(a, i)
        var expect = a.load[width=1](nd_idx)
        var actual = b.load[width=1](nd_idx)
        if not isclose(expect, actual, atol=abs_tol, rtol=rel_tol):
            is_close = False
            if print_wrong_value and num_errs < max_num_print:
                print("At ", nd_idx, "expect", expect, "but get", actual)
                num_errs += 1
            else:
                return False

    return is_close


# this needs to align with tile sizes for target arch, otherwise we will
# try to load beyond the block size for a given tile.
# TODO figure out a way to retrieve max tile size from single source
alias block_size = 32 if has_neon() else 64


def execute_mha_and_compare[
    type: DType,
    seq_len: Int,
    transpose_k: Bool,
    k_cache_t: KVCache,
    v_cache_t: KVCache,
](
    q_tens: Tensor[type, 4],
    k_tens: Tensor[type, 4],
    v_tens: Tensor[type, 4],
    mask_tens: Tensor[type, 2],
    naive_output_tens: Tensor[type, 4],
    test_output_tens: Tensor[type, 4],
    fa_output_tens: Tensor[type, 4],
    k_cache: k_cache_t,
    v_cache: v_cache_t,
):
    # TODO rewrite as Tensors instead of NDBuffers
    var q = ndbuffer_view_from_tensor_with_tensor_shape[
        q_tens.type, q_tens.rank, q_tens.rank
    ](q_tens, 0, q_tens.get_tensor_spec().shape)
    var k = ndbuffer_view_from_tensor_with_tensor_shape[
        k_tens.type, k_tens.rank, k_tens.rank
    ](k_tens, 0, k_tens.get_tensor_spec().shape)
    var v = ndbuffer_view_from_tensor_with_tensor_shape[
        v_tens.type, v_tens.rank, v_tens.rank
    ](v_tens, 0, v_tens.get_tensor_spec().shape)
    var mask = ndbuffer_view_from_tensor_with_tensor_shape[
        mask_tens.type, mask_tens.rank, mask_tens.rank
    ](mask_tens, 0, mask_tens.get_tensor_spec().shape)

    var naive_output = ndbuffer_view_from_tensor_with_tensor_shape[
        naive_output_tens.type, naive_output_tens.rank, naive_output_tens.rank
    ](naive_output_tens, 0, naive_output_tens.get_tensor_spec().shape)

    var test_output = ndbuffer_view_from_tensor_with_tensor_shape[
        test_output_tens.type, test_output_tens.rank, test_output_tens.rank
    ](test_output_tens, 0, test_output_tens.get_tensor_spec().shape)

    var fa_output = ndbuffer_view_from_tensor_with_tensor_shape[
        fa_output_tens.type, fa_output_tens.rank, fa_output_tens.rank
    ](fa_output_tens, 0, fa_output_tens.get_tensor_spec().shape)

    alias scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    var batch_size = q.dim[0]()
    var num_heads = q.dim[1]()
    var depth = q.dim[3]()

    @parameter
    fn do_naive_attn():
        _naive_attention[type, transpose_k](
            naive_output.make_dims_unknown(),
            q.make_dims_unknown(),
            k.make_dims_unknown(),
            v.make_dims_unknown(),
            mask,
            scale,
        )

    var naive_results = run[do_naive_attn](
        num_warmup=1,
        max_iters=10,
    )
    var naive_mean = naive_results.mean()
    print("Naive Attn:", naive_mean, "s")

    @parameter
    fn do_naive_block_list_attn():
        _naive_block_list_attention[type](
            test_output.make_dims_unknown(),
            q.make_dims_unknown(),
            k_cache,
            v_cache,
            mask,
            seq_len,
            scale,
        )

    var naive_bl_results = run[do_naive_block_list_attn](
        num_warmup=1,
        max_iters=10,
    )
    var naive_bl_mean = naive_bl_results.mean()
    print("Naive BlockList Attn:", naive_bl_mean, "s")
    print("Naive/BlockList:", naive_mean / naive_bl_mean, "X")

    assert_true(
        is_ndbuffer_close(
            naive_output.make_dims_unknown(),
            test_output.make_dims_unknown(),
        )
    )

    var mask_4d = NDBuffer[type, 4](
        mask.data,
        Index(batch_size, num_heads, seq_len, seq_len),
        Index(0, 0, seq_len, 1),
    )

    @parameter
    fn input_k_fn[
        width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
        constrained[
            width <= block_size, "Expected width to be less than block size"
        ]()
        var bs = idx[0]
        var head_idx = idx[1]
        var head_d_idx: Int
        var seq: Int

        @parameter
        if not transpose_k:
            head_d_idx = idx[2]
            seq = idx[3]
        else:
            head_d_idx = idx[3]
            seq = idx[2]

        return k_cache.load[type, width=width](bs, head_idx, seq, head_d_idx)

    @parameter
    fn input_v_fn[
        width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
        # idx == bs, num_heads, kv_len, head_dim
        constrained[
            width <= block_size, "Expected width to be less than block size"
        ]()
        var bs = idx[0]
        var head_idx = idx[1]
        var seq = idx[2]
        var head_d_idx = idx[3]
        return v_cache.load[type, width=width](bs, head_idx, seq, head_d_idx)

    @parameter
    fn input_mask_fn[
        width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
        return mask_4d.load[width=width](rebind[StaticIntTuple[4]](idx))

    var fa_k_shape: StaticIntTuple[4]
    var fa_v_shape = StaticIntTuple[4](batch_size, num_heads, seq_len, depth)

    @parameter
    if not transpose_k:
        fa_k_shape = StaticIntTuple[4](batch_size, num_heads, depth, seq_len)
    else:
        fa_k_shape = fa_v_shape

    @parameter
    fn do_flash_attn():
        flash_attention[
            type,
            4,
            input_k_fn,
            input_v_fn,
            input_mask_fn,
            transpose_k=transpose_k,
        ](
            q.make_dims_unknown(),
            fa_k_shape,
            fa_v_shape,
            fa_output,
            scale,
        )

    var flash_attn_results = run[do_flash_attn](
        num_warmup=1,
        max_iters=10,
    )
    var flash_attn_mean = flash_attn_results.mean()
    print("Flash Attn:", flash_attn_mean, "s")
    print("Naive/FlashAttn:", naive_mean / flash_attn_mean, "X")
    assert_true(
        is_ndbuffer_close(
            naive_output.make_dims_unknown(),
            fa_output.make_dims_unknown(),
            max_num_print=100,
        )
    )
    print()


# TODO confusing overload, fix this.
def ndbuffer_view_from_tensor_with_tensor_shape[
    type: DType, in_rank: Int, out_rank: Int
](
    tensor: Tensor[type, in_rank],
    offset: Int,
    out_shape: TensorShape,
) -> NDBuffer[type, out_rank]:
    var cvt_shape = StaticIntTuple[out_rank]()

    for i in range(out_rank):
        cvt_shape[i] = out_shape[i]

    return ndbuffer_view_from_tensor(tensor, offset, cvt_shape)


def ndbuffer_view_from_tensor[
    type: DType, in_rank: Int, out_rank: Int
](
    tensor: Tensor[type, in_rank],
    offset: Int,
    out_shape: StaticIntTuple[out_rank],
) -> NDBuffer[type, out_rank]:
    var ptr = tensor._ptr + offset
    return NDBuffer[type, out_rank](ptr, out_shape)


def _create_contig_kv_cache[
    type: DType,
    batch_size: Int,
    num_heads: Int,
    seq_len: Int,
    depth: Int,
    transpose: Bool,
](ref_: Tensor[type, 4], backing_tensor: Tensor[type, 4]) -> SplitBatchKVCache[
    type, transpose
]:
    var batch_tensor_size = num_heads * seq_len * depth
    var batch_list = Pointer[NDBuffer[type, 3]].alloc(batch_size)
    var batch_tensor_shape = StaticIntTuple[3](
        num_heads, depth, seq_len
    ) if transpose else StaticIntTuple[3](num_heads, seq_len, depth)

    for b in range(batch_size):
        # TODO allocate top-level buffer and take slice of it here.
        var offset = batch_tensor_size * b
        batch_list[b] = ndbuffer_view_from_tensor[type, 4, 3](
            backing_tensor, offset, batch_tensor_shape
        )
        memcpy(
            batch_list[b].data,
            ref_._ptr.offset(offset),
            batch_tensor_size,
        )

    var valid_lengths_ptr = DTypePointer[DType.int64].alloc(batch_size)
    var valid_lengths = NDBuffer[DType.int64, 1](
        valid_lengths_ptr, (batch_size,)
    )
    for i in range(batch_size):
        valid_lengths[i] = seq_len
    return SplitBatchKVCache[type, transpose](
        batch_list, valid_lengths, num_heads, depth
    )


def _create_paged_kv_cache[
    type: DType,
    transpose_blocks: Bool,
](
    ref_: Tensor[type, 4],
    batch_size: Int,
    num_heads: Int,
    seq_len: Int,
    depth: Int,
) -> PagedKVCache[type, transpose_blocks]:
    var kv_block_size = num_heads * block_size * depth
    var num_full_blocks = seq_len // block_size
    var tokens_in_last_block = seq_len % block_size
    var k_block_shape = DimList(
        num_heads, depth, block_size
    ) if transpose_blocks else DimList(num_heads, block_size, depth)
    var block_list = List[List[NDBuffer[type, 3]]]()
    for bs in range(batch_size):
        block_list.append(List[NDBuffer[type, 3]]())
        for block_idx in range(ceildiv(seq_len, block_size)):
            var block_ptr = DTypePointer[type].alloc(kv_block_size)

            if block_idx < num_full_blocks:
                for h in range(num_heads):

                    @parameter
                    if transpose_blocks:
                        for d in range(depth):
                            for b in range(block_size):
                                block_ptr[
                                    h * depth * block_size + d * block_size + b
                                ] = ref_[
                                    Index(bs, h, d, block_idx * block_size + b)
                                ]
                    else:
                        var ptr_offset = bs * num_heads * seq_len * depth + h * seq_len * depth + block_idx * block_size * depth
                        memcpy(
                            block_ptr + h * block_size * depth,
                            ref_._ptr + ptr_offset,
                            depth * block_size,
                        )
            else:
                for s in range(tokens_in_last_block):
                    for h in range(num_heads):

                        @parameter
                        if transpose_blocks:
                            for d in range(depth):
                                block_ptr[
                                    h * depth * block_size + d * block_size + s
                                ] = ref_[
                                    Index(bs, h, d, block_idx * block_size + s)
                                ]
                        else:
                            var ptr_offset = bs * num_heads * seq_len * depth + h * seq_len * depth + (
                                block_idx * block_size + s
                            ) * depth
                            memcpy(
                                block_ptr + h * block_size * depth + s * depth,
                                ref_._ptr + ptr_offset,
                                depth,
                            )

            block_list[-1].append(
                NDBuffer[type, 3](block_ptr, k_block_shape).make_dims_unknown()
            )

    var valid_lengths_ptr = DTypePointer[DType.int64].alloc(batch_size)
    var valid_lengths = NDBuffer[DType.int64, 1](
        valid_lengths_ptr, (batch_size,)
    )
    for i in range(batch_size):
        valid_lengths[i] = seq_len
    return PagedKVCache[type, transpose_blocks](
        block_list,
        valid_lengths,
        num_heads,
        depth,
        block_size,
    )


def test_mha_block_list[
    type: DType, seq_len: Int, do_paged: Bool, transpose_k: Bool
]():
    print("seq_len:", seq_len, "k_trans?:", transpose_k)
    # Query, key, value dimensions.
    alias batch_size = 1
    alias num_heads = 10
    alias depth = 20
    alias mask_val = Float32(-1e10)
    var host_driver = Device(CPUDescriptor())

    # Q, K, V shapes.
    alias BHSD = StaticIntTuple[4](batch_size, num_heads, seq_len, depth)
    alias BHDS = StaticIntTuple[4](batch_size, num_heads, depth, seq_len)

    alias qkv_size = batch_size * num_heads * seq_len * depth

    alias v_block_shape = StaticIntTuple[3](num_heads, block_size, depth)
    alias k_block_shape = v_block_shape if transpose_k else StaticIntTuple[3](
        num_heads, depth, block_size
    )

    # Allocate memory for all variables.
    # var q_ptr = DTypePointer[type].alloc(qkv_size)
    # var k_ptr = DTypePointer[type].alloc(qkv_size)
    # var v_ptr = DTypePointer[type].alloc(qkv_size)
    # var mask_ptr = DTypePointer[type].alloc(seq_len * seq_len)
    # var output_ptr = DTypePointer[type].alloc(qkv_size)
    # var block_list_output_ptr = DTypePointer[type].alloc(qkv_size)
    # var fa_output_ptr = DTypePointer[type].alloc(qkv_size)

    # Q, K, V are randomly initialized.
    # rand(q_ptr, qkv_size)
    # rand(k_ptr, qkv_size)
    # rand(v_ptr, qkv_size)
    var q = host_driver.allocate(TensorSpec(type, BHSD)).get_tensor[type, 4]()
    rand(q._ptr, qkv_size)
    # var q = NDBuffer[type, 4, BHSD](q_ptr)
    # var k = NDBuffer[type, 4, BHSD if transpose_k else BHDS](k_ptr)
    var k = host_driver.allocate(
        TensorSpec(type, BHSD if transpose_k else BHDS)
    ).get_tensor[type, 4]()
    rand(k._ptr, qkv_size)
    # var v = NDBuffer[type, 4, BHSD](v_ptr)
    var v = host_driver.allocate(TensorSpec(type, BHSD)).get_tensor[type, 4]()
    rand(v._ptr, qkv_size)

    var mask = host_driver.allocate(
        TensorSpec(type, Index(seq_len, seq_len))
    ).get_tensor[type, 2]()

    # Set triangular mask
    for b in range(seq_len):
        for i in range(b + 1):
            mask[Index(b, i)] = 0
        for i in range(b + 1, seq_len):
            mask[Index(b, i)] = mask_val.cast[type]()

    # Contruct buffers.
    # var mask = NDBuffer[type, 2](mask_ptr, Index(seq_len, seq_len))
    # var naive_output = NDBuffer[type, 4, BHSD](output_ptr)
    var naive_output = host_driver.allocate(TensorSpec(type, BHSD)).get_tensor[
        type, 4
    ]()
    # var test_output = NDBuffer[type, 4, BHSD](block_list_output_ptr)
    var test_output = host_driver.allocate(TensorSpec(type, BHSD)).get_tensor[
        type, 4
    ]()
    # var fa_output = NDBuffer[type, 4, BHSD](fa_output_ptr)
    var fa_output = host_driver.allocate(TensorSpec(type, BHSD)).get_tensor[
        type, 4
    ]()

    # TODO add copy to device here.
    @parameter
    if do_paged:
        var k_cache = _create_paged_kv_cache[type, not transpose_k](
            k,
            batch_size,
            num_heads,
            seq_len,
            depth,
        )
        var v_cache = _create_paged_kv_cache[type, False](
            v,
            batch_size,
            num_heads,
            seq_len,
            depth,
        )

        execute_mha_and_compare[
            type,
            seq_len,
            transpose_k,
            __type_of(k_cache),
            __type_of(v_cache),
        ](
            q,
            k,
            v,
            mask,
            naive_output,
            test_output,
            fa_output,
            k_cache,
            v_cache,
        )
        for i in range(len(k_cache.blocks)):
            for j in range(len(k_cache.blocks[i])):
                k_cache.blocks[i][j].data.free()
                v_cache.blocks[i][j].data.free()
        k_cache.valid_lengths.data.free()
        v_cache.valid_lengths.data.free()
        _ = k_cache^
        _ = v_cache^
    else:
        raise "re-enable once MOCO-779 is fixed"
        # var v_cache_shape = StaticIntTuple[4](
        #     batch_size, num_heads, seq_len, depth
        # )
        # var k_cache_shape = v_cache_shape if transpose_k else StaticIntTuple[4](
        #     batch_size, num_heads, seq_len, depth
        # )

        # var k_cache_backing = host_driver.allocate(
        #     TensorSpec(type, k_cache_shape)
        # ).get_tensor[type, 4]()
        # var v_cache_backing = host_driver.allocate(
        #     TensorSpec(type, v_cache_shape)
        # ).get_tensor[type, 4]()

        # var k_cache = _create_contig_kv_cache[
        #     type, batch_size, num_heads, seq_len, depth, not transpose_k
        # ](k, k_cache_backing)

        # var v_cache = _create_contig_kv_cache[
        #     type, batch_size, num_heads, seq_len, depth, False
        # ](v, v_cache_backing)

        # execute_mha_and_compare[
        #     type,
        #     seq_len,
        #     transpose_k,
        #     __type_of(k_cache),
        #     __type_of(v_cache),
        # ](
        #     q,
        #     k,
        #     v,
        #     mask,
        #     naive_output,
        #     test_output,
        #     fa_output,
        #     k_cache,
        #     v_cache,
        # )

        # # NOTE: this is usually handled by the collection wrapper.
        # k_cache.valid_lengths.data.free()
        # v_cache.valid_lengths.data.free()
        # _ = k_cache^
        # _ = v_cache^
        # _ = k_cache_backing^
        # _ = v_cache_backing^


def main():
    # TODO reintroduce contiguous tensors in this test once MOCO-779 is fixed
    test_mha_block_list[DType.float32, 128, True, False]()
    test_mha_block_list[DType.float32, 2, True, False]()
    test_mha_block_list[DType.float32, 135, True, False]()
    test_mha_block_list[DType.float32, 128, True, True]()
    test_mha_block_list[DType.float32, 2, True, True]()
    test_mha_block_list[DType.float32, 135, True, True]()
    test_generation_loop()
    with Context() as ctx:
        test_generation_loop[target="cuda"]()
