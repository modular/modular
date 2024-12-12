# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import Set, Dict, Optional
from buffer import NDBuffer, DimList
from utils import IndexList
from sys import sizeof
from sys.intrinsics import _type_is_eq

from kv_cache.types import (
    ContiguousKVCache,
    ContiguousKVCacheCollection,
    KVCacheStaticParams,
    KVCollectionT,
)

from max.driver import (
    Device,
    DeviceMemory,
    Tensor,
    TensorSlice,
    DeviceTensor,
    ManagedTensorSlice,
)
from max.driver import Device, DeviceMemory, Tensor, TensorSlice, DeviceTensor
from max.tensor import TensorShape, TensorSpec
from collections import Optional


trait KVCacheManagerT:
    alias CollectionType: KVCollectionT

    fn claim(mut self, batch_size: Int) raises -> List[Int]:
        ...

    fn fetch[
        collection_t: KVCollectionT
    ](mut self, seq_ids: List[Int]) raises -> collection_t:
        ...

    fn step[
        collection_t: KVCollectionT
    ](
        mut self,
        valid_lengths: List[Int],
        owned inflight_cache: collection_t,
    ) raises:
        ...

    fn release(mut self, seq_id: Int) raises:
        ...


struct ContiguousKVCacheManager[
    type: DType,
    kv_params: KVCacheStaticParams,
](KVCacheManagerT):
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

    alias CollectionType = ContiguousKVCacheCollection[type, kv_params]
    var blocks_buf: Tensor[type, 6]
    var num_blocks: Int
    var max_batch_size: Int
    var max_seq_len: Int
    var active_seq_ids: List[Int]
    var cache_lengths: List[Int]
    var seq_id_counter: Int

    var num_layers: Int
    var other_device: Device
    var this_device: Device
    var cache_lengths_tensor_host: DeviceTensor
    var cache_lengths_tensor_dev: DeviceTensor

    # TODO currently we allocate the exact right amount, we should
    # extend this to allow for over-allocation
    fn __init__(
        mut self,
        max_batch_size: Int,
        max_seq_len: Int,
        num_layers: Int,
        mut other_device: Device,
        mut this_device: Device,
    ) raises:
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # TODO make a different argument for the number of blocks.
        self.num_blocks = self.max_batch_size

        var blocks_shape = IndexList[6](
            2,
            num_layers,
            max_batch_size,
            max_seq_len,
            kv_params.num_heads,
            kv_params.head_size,
        )

        self.blocks_buf = Tensor[type, 6](blocks_shape, other_device)
        self.active_seq_ids = List[Int]()
        self.cache_lengths = List[Int]()
        self.seq_id_counter = 0
        self.num_layers = num_layers
        self.this_device = this_device
        self.other_device = other_device

        self.cache_lengths_tensor_host = self.this_device.allocate(
            TensorSpec(DType.uint32, (max_batch_size))
        )
        self.cache_lengths_tensor_dev = self.other_device.allocate(
            TensorSpec(DType.uint32, (max_batch_size))
        )

    fn claim(mut self, batch_size: Int) raises -> List[Int]:
        """Assign `batch_size` blocks for incoming requests.

        This returns a List of seq_ids, which can be passed to `fetch` to
        retrieve the KVCollection for the given batch.
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
        return self.active_seq_ids

    fn fetch[
        collection_t: KVCollectionT
    ](mut self, seq_ids: List[Int]) raises -> collection_t:
        """Retrieves the pre-assigned blocks for the given seq_ids.

        if any of the seq_ids are not valid (e.g. no assigned blocks) then
        and error is raised.
        """
        constrained[_type_is_eq[collection_t, self.CollectionType]()]()

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

        # Have 2 copies of cache_lengths and copy to device from host here
        cache_lengths_host_ptr = (
            self.cache_lengths_tensor_host.unsafe_ptr().bitcast[UInt32]()
        )
        var is_context_encoding = True
        for bs in range(batch_size):
            if self.cache_lengths[bs] != 0:
                is_context_encoding = False
            cache_lengths_host_ptr[bs] = self.cache_lengths[bs]

        self.cache_lengths_tensor_host.copy_into(self.cache_lengths_tensor_dev)

        var block_size = self.num_layers * batch_size * kv_params.num_heads * self.max_seq_len * kv_params.head_size

        var v_cache_shape = IndexList[5](
            self.num_layers,
            batch_size,
            self.max_seq_len,
            kv_params.num_heads,
            kv_params.head_size,
        )
        var k_cache_shape = v_cache_shape

        var key_cache = NDBuffer[type, 5](
            self.blocks_buf.unsafe_ptr(),
            k_cache_shape,
        )
        var value_cache = NDBuffer[type, 5](
            self.blocks_buf.unsafe_ptr() + block_size, v_cache_shape
        )

        return rebind[collection_t](
            Self.CollectionType(
                key_cache,
                value_cache,
                NDBuffer[DType.uint32, 1](
                    self.cache_lengths_tensor_dev.unsafe_ptr().bitcast[
                        UInt32
                    ](),
                    (batch_size,),
                ),
                is_context_encoding,
                seq_ids,
                self.num_layers,
                batch_size,
            )
        )

    fn step[
        collection_t: KVCollectionT
    ](
        mut self,
        valid_lengths: List[Int],
        owned inflight_cache: collection_t,
    ) raises:
        """Commits changes to the ContiguousKVCache blocks.

        This is used to note that a KV projection step has occured and
        the values in these buffers have been written to. We note the new tokens
        in the blocks and update the valid_length counter.
        """
        constrained[_type_is_eq[collection_t, self.CollectionType]()]()

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

    fn release(mut self, seq_id: Int) raises:
        """Marks `seq_id` as no longer necessary, their blocks are reintroduced
        to the pool.

        """
        # TODO there are a lot of edge cases to figure out here, for now just
        # invalidate the cache.
        self.active_seq_ids = List[Int]()
        self.cache_lengths = List[Int]()
