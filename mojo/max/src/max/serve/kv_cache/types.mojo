# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from buffer import NDBuffer
from utils import StaticIntTuple
from kv_cache.types import (
    ContiguousKVCache,
    ContiguousKVCacheCollection,
    KVCacheStaticParams,
    KVCacheLayout,
    _default_max_batch_size,
)

from max.driver import Device, DeviceMemory, Tensor, TensorSlice
from max.tensor import TensorShape, TensorSpec


struct ContiguousKVCacheManager[
    type: DType,
    kv_params: KVCacheStaticParams,
    _max_batch_size: Int = _default_max_batch_size,
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

    alias CollectionType = ContiguousKVCacheCollection[
        type, kv_params, _max_batch_size
    ]
    var blocks_buf: Tensor[type, 5]
    var num_blocks: Int
    var max_batch_size: Int
    var max_seq_len: Int
    var active_seq_ids: List[Int]
    var cache_lengths: List[Int]
    var seq_id_counter: Int
    var num_layers: Int
    var other_device: Device
    var this_device: Device

    # TODO currently we allocate the exact right amount, we should
    # extend this to allow for over-allocation
    fn __init__(
        inout self,
        max_batch_size: Int,
        max_seq_len: Int,
        num_layers: Int,
        inout other_device: Device,
        inout this_device: Device,
    ) raises:
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # TODO make a different argument for the number of blocks.
        self.num_blocks = self.max_batch_size
        var blocks_shape: StaticIntTuple[6]

        @parameter
        if kv_params.layout == KVCacheLayout.BHSD:
            blocks_shape = StaticIntTuple[6](
                2,
                num_layers,
                max_batch_size,
                kv_params.num_heads,
                max_seq_len,
                kv_params.head_size,
            )
        elif kv_params.layout == KVCacheLayout.BSHD:
            blocks_shape = StaticIntTuple[6](
                2,
                num_layers,
                max_batch_size,
                max_seq_len,
                kv_params.num_heads,
                kv_params.head_size,
            )
        else:
            constrained[False, "unsupported layout"]()
            raise "failure"

        self.blocks_buf = Tensor[type, 5](blocks_shape, other_device)
        self.active_seq_ids = List[Int]()
        self.cache_lengths = List[Int]()
        self.seq_id_counter = 0
        self.num_layers = num_layers
        self.this_device = this_device
        self.other_device = other_device

    fn claim(inout self, batch_size: Int) raises -> Self.CollectionType:
        """Assign `batch_size` blocks for incoming requests.

        This returns a KVCacheCollection, which has buffers
        for each layer in the network. Each sequence is assigned a seq_id
        which uniquely identifies that sequence's entries in this cache.
        """
        if batch_size > self.max_batch_size:
            raise "batch size too large"

        if self.active_seq_ids:
            raise "active sequence already in flight"

        var seq_ids = List[Int]()

        for _ in range(batch_size):
            var new_seq_id = self.seq_id_counter
            self.seq_id_counter += 1
            seq_ids.append(new_seq_id)
            self.cache_lengths.append(0)

        self.active_seq_ids = seq_ids
        return self.fetch(seq_ids)

    fn fetch(inout self, seq_ids: List[Int]) raises -> Self.CollectionType:
        """Retrieves the pre-assigned blocks for the given seq_ids.

        if any of the seq_ids are not valid (e.g. no assigned blocks) then
        and error is raised.
        """

        for idx in range(len(self.active_seq_ids)):
            if seq_ids[idx] != self.active_seq_ids[idx]:
                raise "Invalid seq idx at batch " + str(
                    idx
                ) + ", expected " + str(
                    self.active_seq_ids[idx]
                ) + " but got " + str(
                    seq_ids[idx]
                )

        var batch_size = len(seq_ids)
        var cache_lengths = StaticIntTuple[_max_batch_size](-1)

        for bs in range(batch_size):
            cache_lengths[bs] = self.cache_lengths[bs]

        var block_size = self.num_layers * batch_size * kv_params.num_heads * self.max_seq_len * kv_params.head_size
        var v_cache_shape: StaticIntTuple[5]
        var k_cache_shape: StaticIntTuple[5]

        @parameter
        if kv_params.layout == KVCacheLayout.BHSD:
            v_cache_shape = StaticIntTuple[5](
                self.num_layers,
                batch_size,
                kv_params.num_heads,
                self.max_seq_len,
                kv_params.head_size,
            )
            k_cache_shape = v_cache_shape
        elif kv_params.layout == KVCacheLayout.BSHD:
            v_cache_shape = StaticIntTuple[5](
                self.num_layers,
                batch_size,
                self.max_seq_len,
                kv_params.num_heads,
                kv_params.head_size,
            )
            k_cache_shape = v_cache_shape
        else:
            constrained[False, "unsupported layout"]()
            raise "failure"

        var key_cache = NDBuffer[type, 5](
            self.blocks_buf.unsafe_ptr(),
            k_cache_shape,
        )
        var value_cache = NDBuffer[type, 5](
            self.blocks_buf.unsafe_ptr() + block_size, v_cache_shape
        )

        return Self.CollectionType(
            key_cache,
            value_cache,
            cache_lengths,
            seq_ids,
            self.num_layers,
            batch_size,
        )

    fn step(
        inout self,
        valid_lengths: List[Int],
        inout inflight_cache: ContiguousKVCacheCollection,
    ) raises:
        """Commits changes to the ContiguousKVCache blocks.

        This is used to note that a KV projection step has occured and
        the values in these buffers have been written to. We note the new tokens
        in the blocks and update the valid_length counter.
        """

        var batch_size = len(valid_lengths)
        debug_assert(
            batch_size == len(self.cache_lengths),
            (
                "Invalid valid_lengths passed, expected to match active"
                " requests batch size."
            ),
        )
        for bs in range(batch_size):
            var new_seq_len = valid_lengths[bs]
            self.cache_lengths[bs] += new_seq_len
            inflight_cache.incr_cache_length(bs, new_seq_len)

    fn release(inout self, seq_id: Int) raises:
        """Marks `seq_id` as no longer necessary, their blocks are reintroduced
        to the pool.

        """
        # TODO there are a lot of edge cases to figure out here, for now just
        # invalidate the cache.
        self.active_seq_ids = List[Int]()
        self.cache_lengths = List[Int]()
