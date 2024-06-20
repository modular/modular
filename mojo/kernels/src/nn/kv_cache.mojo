# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from buffer import NDBuffer
from buffer.list import DimList
from driver import Device, DeviceMemory, Tensor, TensorSlice
from tensor import TensorShape, TensorSpec

from .util import (
    ndbuffer_view_from_tensor,
    ndbuffer_view_from_tensor_with_tensor_shape,
)


# register_passable isn't compatible with custom kernels via extensibility, requirement for mo.opaque
# register_passable is required to pass the kernel to GPU
# :(
@value
@register_passable
struct KVCacheRegisterPassable[type: DType, transpose: Bool]():
    """Wrapper for the KVCache of a given layer in the transformer model.

    This abstracts the Pointer indirection for accessing the KVCache for a
    given batch entry.

    THIS IS THE TYPE THAT IS PASSED TO KV PROJECTION AND FLASH ATTENTION KERNELS.
    """

    alias BlockType = NDBuffer[type, 3]
    var blocks: Pointer[Self.BlockType]
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
        blocks: Pointer[Self.BlockType],
        valid_lengths: NDBuffer[DType.int64, 1],
        num_heads: Int,
        head_size: Int,
    ):
        self.blocks = blocks
        self.valid_lengths = valid_lengths
        self.batch_size = valid_lengths.dim[0]()
        self.num_heads = num_heads
        self.head_size = head_size

    @staticmethod
    fn id() -> String:
        return "KVCacheRegisterPassable"

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


# TODO make register passable again!
# GPU kernels won't work without it.
struct KVCache[type: DType, transpose: Bool](Movable):
    var register_passable_cache: KVCacheRegisterPassable[type, transpose]

    fn __init__(
        inout self,
        blocks: Pointer[KVCacheRegisterPassable[type, transpose].BlockType],
        valid_lengths: NDBuffer[DType.int64, 1],
        num_heads: Int,
        head_size: Int,
    ):
        self.register_passable_cache = KVCacheRegisterPassable[type, transpose](
            blocks, valid_lengths, num_heads, head_size
        )

    fn __moveinit__(inout self, owned other: KVCache[type, transpose]):
        self.register_passable_cache = other.register_passable_cache^

    fn __copyinit__(inout self, other: KVCache[type, transpose]):
        self.register_passable_cache = other.register_passable_cache

    @staticmethod
    fn id() -> String:
        return "KVCache"

    fn get_valid_lengths(self) -> NDBuffer[DType.int64, 1]:
        return self.register_passable_cache.valid_lengths

    # TODO could we load entire blocks as tiles? Would that be more performant?
    @always_inline
    fn load[
        width: Int
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        type, width
    ]:
        return self.register_passable_cache.load[width](
            bs, head_idx, tok_idx, head_dim_idx
        )

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
        return self.register_passable_cache.store[width](
            bs, head_idx, tok_idx, head_dim_idx, val
        )

    fn as_register_passable(self) -> KVCacheRegisterPassable[type, transpose]:
        return self.register_passable_cache


struct KVCacheCollection[
    type: DType,
    transpose_k: Bool,
]:
    """This is a "view" of the cache for the given sequences
    in the batch.

    This object does not own the underlying buffers in k_cache and v_cache,
    it's borrowing them from the BlockWrappers in our KVCacheManager.
    It does own the Pointer[NDBuffer[type, 3]] and valid_lengths buffer
    """

    # TODO outer should be list, inner can be Pointer?
    alias CacheType = List[DeviceMemory]
    var key_cache: Self.CacheType
    var value_cache: Self.CacheType
    var valid_lengths_tensor: Tensor[DType.int64, 1]
    var valid_lengths: NDBuffer[DType.int64, 1]
    var seq_ids: List[Int]
    var num_layers: Int
    var batch_size: Int
    var head_size: Int
    var num_heads: Int

    fn __init__(
        inout self,
        owned key_cache: self.CacheType,
        owned value_cache: self.CacheType,
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

    @staticmethod
    fn id() -> String:
        return "KVCacheCollection"

    fn get_key_cache(self, layer_idx: Int) -> KVCache[type, transpose_k]:
        var device_mem_ref = self.key_cache.__get_ref(layer_idx)
        var k_cache_ptr = device_mem_ref[].unsafe_ptr().address.bitcast[
            NDBuffer[type, 3]
        ]()
        return KVCache[type, transpose_k](
            k_cache_ptr,
            self.valid_lengths,
            self.num_heads,
            self.head_size,
        )

    fn get_value_cache(self, layer_idx: Int) -> KVCache[type, False]:
        var device_mem_ref = self.value_cache.__get_ref(layer_idx)
        var v_cache_ptr = device_mem_ref[].unsafe_ptr().address.bitcast[
            NDBuffer[type, 3]
        ]()
        return KVCache[type, False](
            v_cache_ptr,
            self.valid_lengths,
            self.num_heads,
            self.head_size,
        )

    fn get_seq_ids(self) -> List[Int]:
        return self.seq_ids

    fn get_valid_lengths(self) -> NDBuffer[DType.int64, 1]:
        return self.valid_lengths


struct CacheBlockWrapper[
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


struct KVCacheManager[
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

    alias WrapperType = CacheBlockWrapper[
        type,
        transpose_k,
    ]
    alias CollectionType = KVCacheCollection[
        type,
        transpose_k,
    ]
    var blocks_buf: Tensor[type, 5]
    var blocks: List[Self.WrapperType]
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
        self.blocks = List[self.WrapperType](capacity=self.num_blocks * 2)
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
        ).to_tensor[type, 5]()
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
                Self.WrapperType(
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

    fn claim(inout self, batch_size: Int) raises -> Self.CollectionType:
        """Assign `batch_size` blocks for incoming requests.

        This returns a KVCacheCollection, which has buffers
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

    fn fetch(inout self, seq_ids: List[Int]) raises -> Self.CollectionType:
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
        ).to_tensor[DType.int64, 1]()
        for bs in range(batch_size):
            var seq_id = seq_ids.__get_ref(bs)
            if seq_id[] not in self.seq_id_to_block:
                raise "Unknown seq_id " + str(seq_id[])

            var block_id = self.seq_id_to_block[seq_id[]]
            var block = self.blocks.__get_ref(block_id)

            host_valid_lengths[bs] = block[].get_valid_length()

            for l in range(self.num_layers):
                var host_key_ref = host_key_caches.__get_ref(l)
                var host_key_ptr = host_key_ref[].unsafe_ptr().address.bitcast[
                    NDBuffer[type, 3]
                ]()
                var value_key_ref = host_value_caches.__get_ref(l)
                var host_value_ptr = value_key_ref[].unsafe_ptr().address.bitcast[
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
                host_valid_lengths.to_device_memory()
                .copy_to(self.other_device)
                .to_tensor[DType.int64, 1]()
            )

        return Self.CollectionType(
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
        inflight_cache: KVCacheCollection,
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
