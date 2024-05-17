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


trait KVCache:
    fn get_valid_lengths(self) -> List[Int]:
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


struct ContiguousKVCache[
    type: DType,
    num_heads: Int,
    head_size: Int,
](CollectionElement, KVCache):
    """Wrapper for the KVCache of a given layer in the transformer model.

    This abstracts the pointer indirection for accessing the KVCache for a
    given batch entry.

    THIS IS THE TYPE THAT IS PASSED TO KV PROJECTION AND FLASH ATTENTION KERNELS.
    """

    alias block_t = NDBuffer[type, 3]
    var blocks: List[Self.block_t]
    var valid_lengths: List[Int]

    fn __init__(
        inout self, blocks: List[Self.block_t], valid_lengths: List[Int]
    ):
        self.blocks = blocks
        self.valid_lengths = valid_lengths

    fn __moveinit__(inout self, owned other: Self):
        self.blocks = other.blocks^
        self.valid_lengths = other.valid_lengths^

    fn __copyinit__(inout self, other: Self):
        self.blocks = other.blocks
        self.valid_lengths = other.valid_lengths

    fn get_valid_lengths(self) -> List[Int]:
        return self.valid_lengths

    # TODO could we load entire blocks as tiles? Would that be more performant?
    @always_inline
    fn load[
        _type: DType, width: Int
    ](self, bs: Int, head_idx: Int, tok_idx: Int, head_dim_idx: Int) -> SIMD[
        _type, width
    ]:
        constrained[_type == type, "Expected type to match"]()
        var batch = self.blocks[bs]

        return rebind[SIMD[_type, width]](
            batch.load[width=width](head_idx, tok_idx, head_dim_idx)
        )

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
        var batch = self.blocks[bs]

        batch.store[width=width](
            (head_idx, tok_idx, head_dim_idx), rebind[SIMD[type, width]](val)
        )


struct ContiguousKVCacheCollection[
    type: DType,
    num_layers: Int,
    head_size: Int,
    num_heads: Int,
](CollectionElement):
    """This is a "view" of the cache for the given sequences
    in the batch.

    This object does not own the underlying buffers, it's borrowing them from
    the BlockWrappers in our KVCacheManager.
    """

    alias cache_t = List[List[NDBuffer[type, 3]]]
    var key_cache: Self.cache_t
    var value_cache: Self.cache_t
    var valid_lengths: List[Int]
    var seq_ids: List[Int]

    fn __init__(
        inout self,
        key_cache: List[List[NDBuffer[type, 3]]],
        value_cache: List[List[NDBuffer[type, 3]]],
        valid_lengths: List[Int],
        seq_ids: List[Int],
    ):
        debug_assert(len(key_cache) == num_layers, "invalid key_cache size")
        debug_assert(len(value_cache) == num_layers, "invalid value_cache size")
        debug_assert(
            len(key_cache[0]) == len(valid_lengths), "invalid valid lengths"
        )

        # TODO remove when we have padding/batching enabled
        var first_valid_length = valid_lengths[0]
        for i in range(1, len(valid_lengths)):
            debug_assert(
                first_valid_length == valid_lengths[i],
                "Mismatch in valid lengths",
            )

        self.key_cache = key_cache
        self.value_cache = value_cache
        self.seq_ids = seq_ids
        self.valid_lengths = valid_lengths

    fn __moveinit__(inout self, owned other: Self):
        self.key_cache = other.key_cache
        self.value_cache = other.value_cache
        self.seq_ids = other.seq_ids
        self.valid_lengths = other.valid_lengths

    fn __copyinit__(inout self, other: Self):
        self.key_cache = other.key_cache
        self.value_cache = other.value_cache
        self.seq_ids = other.seq_ids
        self.valid_lengths = other.valid_lengths

    fn get_key_cache(
        self, layer_idx: Int
    ) -> ContiguousKVCache[type, num_heads, head_size]:
        return ContiguousKVCache[type, num_heads, head_size](
            self.key_cache[layer_idx], self.valid_lengths
        )

    fn get_value_cache(
        self, layer_idx: Int
    ) -> ContiguousKVCache[type, num_heads, head_size]:
        return ContiguousKVCache[type, num_heads, head_size](
            self.value_cache[layer_idx], self.valid_lengths
        )

    fn get_seq_ids(self) -> List[Int]:
        return self.seq_ids

    fn get_valid_lengths(self) -> List[Int]:
        return self.valid_lengths


@value
struct ContiguousCacheBlockWrapper[
    type: DType,
    num_layers: Int,
    head_size: Int,
    num_heads: Int,
    token_type: DType = DType.int64,
](CollectionElement):
    """Wrapper for a KV Cache object for a single sequence in the cache.

    This stores all of the KV Cache entries for each layer for a given sequence.
    It stores state related to:
    - whether the block is in use by an active request
    - the length of the context in the cache (we always overallocate to max_seq_len)
    - the tokens assigned to the context in the cache
    """

    # TODO pass malloc fn so we can get GPU tensors?
    alias block_t = NDBuffer[type, 3]
    var k_blocks: List[NDBuffer[type, 3]]
    var v_blocks: List[NDBuffer[type, 3]]
    var tokens: NDBuffer[token_type, 1]
    var in_use: Bool
    var valid_length: Int
    var max_seq_len: Int

    fn __init__(inout self, max_seq_len: Int):
        self.k_blocks = List[NDBuffer[type, 3]]()
        self.v_blocks = List[NDBuffer[type, 3]]()

        @unroll
        for _ in range(num_layers):
            var k_block_ptr = DTypePointer[type].alloc(
                num_heads * max_seq_len * head_size
            )
            self.k_blocks.append(
                NDBuffer[type, 3](
                    k_block_ptr, DimList(num_heads, max_seq_len, head_size)
                )
            )
            var v_block_ptr = DTypePointer[type].alloc(
                num_heads * max_seq_len * head_size
            )
            self.v_blocks.append(
                NDBuffer[type, 3](
                    v_block_ptr, DimList(num_heads, max_seq_len, head_size)
                )
            )

        self.in_use = False
        var tokens_ptr = DTypePointer[token_type].alloc(max_seq_len)
        self.tokens = NDBuffer[token_type, 1](tokens_ptr, DimList(max_seq_len))
        self.valid_length = 0
        self.max_seq_len = max_seq_len

    fn __moveinit__(inout self, owned other: Self):
        self.k_blocks = other.k_blocks
        self.v_blocks = other.v_blocks
        self.tokens = other.tokens
        self.in_use = other.in_use
        self.valid_length = other.valid_length
        self.max_seq_len = other.max_seq_len

    fn __copyinit__(inout self, other: Self):
        self.k_blocks = other.k_blocks
        self.v_blocks = other.v_blocks
        self.tokens = other.tokens
        self.in_use = other.in_use
        self.valid_length = other.valid_length
        self.max_seq_len = other.max_seq_len

    fn get_key_cache(self, layer_idx: Int) -> Self.block_t:
        return self.k_blocks[layer_idx]

    fn get_value_cache(self, layer_idx: Int) -> Self.block_t:
        return self.v_blocks[layer_idx]

    fn get_valid_length(self) -> Int:
        return self.valid_length

    fn __del__(owned self):
        """NOTE: This should be the only thing that deallocates blocks in the
        cache.

        All other uses of these blocks are considered borrowed.
        """
        for k_block in self.k_blocks:
            k_block[].data.free()

        for v_block in self.v_blocks:
            v_block[].data.free()

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


struct ContiguousKVCacheManager[
    type: DType,
    num_layers: Int,
    head_size: Int,
    num_heads: Int,
]:
    """Manages a contiguous KV cache across multiple user sessions.

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

    alias wrapper_t = ContiguousCacheBlockWrapper[
        type, num_layers, head_size, num_heads
    ]
    alias collection_t = ContiguousKVCacheCollection[
        type, num_layers, head_size, num_heads
    ]
    var blocks: List[Self.wrapper_t]
    var unused_blocks: Set[Int]
    var max_batch_size: Int
    var max_seq_len: Int
    var seq_id_counter: Int
    var seq_id_to_block: Dict[Int, Int]

    # TODO currently we allocate the exact right amount, we should
    # extend this to allow for over-allocation
    fn __init__(inout self, max_batch_size: Int, max_seq_len: Int):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.blocks = List[Self.wrapper_t]()
        self.unused_blocks = Set[Int]()
        self.seq_id_counter = 0
        self.seq_id_to_block = Dict[Int, Int]()

        for i in range(self.max_batch_size * self.num_layers):
            self.blocks.append(Self.wrapper_t(max_seq_len))
            self.unused_blocks.add(i)

    fn claim(inout self, batch_size: Int) raises -> Self.collection_t:
        """Assign `batch_size` blocks for incoming requests.

        This returns a ContiguousKVCacheCollection, which has buffers
        for each layer in the network. Each sequence is assigned a seq_id
        which uniquely identifies that sequence's entries in this cache.
        """
        if batch_size > self.max_batch_size:
            raise "batch size too large"

        # TODO check that we have the blocks to satisfy this request
        # outer dimension: num_layers. inner_dimension: batch_size
        var key_caches = List[List[NDBuffer[type, 3]]]()
        var value_caches = List[List[NDBuffer[type, 3]]]()
        for l in range(num_layers):
            key_caches.append(List[NDBuffer[type, 3]]())
            value_caches.append(List[NDBuffer[type, 3]]())

        var seq_ids = List[Int]()

        # temp list to avoid lookups in second loop
        var block_ids = List[Int]()
        var valid_lengths = List[Int]()
        for bs in range(batch_size):
            # TODO make this atomic, account for collisions once we overflow
            var new_seq_id = self.seq_id_counter
            self.seq_id_counter += 1
            seq_ids.append(new_seq_id)

            var block_id = self.unused_blocks.pop()
            block_ids.append(block_id)
            self.seq_id_to_block[new_seq_id] = block_id
            var block = self.blocks.__get_ref(block_ids[bs])
            block[].mark_in_use()

            valid_lengths.append(block[].get_valid_length())
            for l in range(num_layers):
                key_caches[l].append(block[].get_key_cache(l))
                value_caches[l].append(block[].get_value_cache(l))

        return Self.collection_t(
            key_caches, value_caches, valid_lengths, seq_ids
        )

    fn fetch(inout self, seq_ids: List[Int]) raises -> Self.collection_t:
        """Retrieves the pre-assigned blocks for the given seq_ids.

        if any of the seq_ids are not valid (e.g. no assigned blocks) then
        and error is raised.
        """
        var key_caches = List[List[NDBuffer[type, 3]]]()
        var value_caches = List[List[NDBuffer[type, 3]]]()
        for l in range(num_layers):
            key_caches.append(List[NDBuffer[type, 3]]())
            value_caches.append(List[NDBuffer[type, 3]]())

        # TODO temporary constraint, enforces that each item in the batch has
        # similar "valid_length". We should add padding to fix this.
        var block_length = -1
        var valid_lengths = List[Int]()
        for seq_id in seq_ids:
            if seq_id[] not in self.seq_id_to_block:
                raise "Unknown seq_id " + str(seq_id[])

            var block_id = self.seq_id_to_block[seq_id[]]
            var block = self.blocks.__get_ref(block_id)
            if block_length < 0:
                block_length = block[].valid_length
            elif block_length != block[].valid_length:
                raise "Uneven block lengths, got " + str(
                    block_length
                ) + " vs " + str(block[].valid_length)

            valid_lengths.append(block[].valid_length)

            for l in range(num_layers):
                key_caches[l].append(block[].get_key_cache(l))
                value_caches[l].append(block[].get_value_cache(l))

        return Self.collection_t(
            key_caches, value_caches, valid_lengths, seq_ids
        )

    fn step(
        inout self,
        token_ids: NDBuffer[DType.int64, 2],
        inflight_cache: ContiguousKVCacheCollection,
    ) raises:
        """Commits changes to the KVCache blocks

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
        to the pool

        """
        if seq_id not in self.seq_id_to_block:
            raise "seq_id not found: " + str(seq_id)

        var block_idx = self.seq_id_to_block.pop(seq_id)
        var block = self.blocks.__get_ref(block_idx)
        block[].reset()
        self.unused_blocks.add(block_idx)


fn test_generation_loop() raises:
    alias type = DType.float32
    alias num_layers = 4
    alias head_size = 64
    alias num_heads = 4
    var max_seq_len = 100
    var max_batch_size = 8
    var num_iters = 10

    var manager = ContiguousKVCacheManager[
        type, num_layers, head_size, num_heads
    ](max_batch_size, max_seq_len)

    var num_tokens = 10
    var batch_size = 4
    var token_ptr = DTypePointer[DType.int64].alloc(num_tokens * batch_size)
    var token_ids = NDBuffer[DType.int64, 2](
        token_ptr, DimList(batch_size, num_tokens)
    )

    # create pointer for "next" tokens during token generation
    var next_token_ptr = DTypePointer[DType.int64].alloc(1 * batch_size)
    var next_token_ids = NDBuffer[DType.int64, 2](
        next_token_ptr, DimList(batch_size, 1)
    )

    # STEP 1: Tell the manager that we have new requests, retrieve the cache and unique IDs
    var empty_kv = manager.claim(batch_size)

    # STEP 2: do a forward pass, fill in KV projections in the cache buffers for our prompt
    fake_kv_projection[type](empty_kv, token_ids)

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
        fake_kv_projection[type](prefilled_kv, next_token_ids)

        # STEP 6: notify the KV Cache manager we modified prefilled_kv, update the state in the manager
        manager.step(next_token_ids, prefilled_kv)

        # STEP 7: repeat...

    # assert on some internal state of the manager:
    assert_true(len(manager.seq_id_to_block) == 4)
    assert_true(len(manager.blocks) - len(manager.unused_blocks) == 4)

    # STEP 8: notify the manager that these sequences have completed generation
    # evict them from our cache.
    for seq_id in cache_seq_ids:
        manager.release(seq_id[])

    token_ptr.free()
    next_token_ptr.free()


fn fake_kv_projection[
    type: DType
](
    cache: ContiguousKVCacheCollection[type],
    token_ids: NDBuffer[DType.int64, 2],
):
    var batch_size = token_ids.dim[0]()
    var new_seq_len = token_ids.dim[1]()
    for l in range(cache.num_layers):
        var layer_key_cache = cache.get_key_cache(l)
        var layer_value_cache = cache.get_value_cache(l)
        fake_kv_projection_layer[type](
            layer_key_cache, layer_value_cache, token_ids
        )


fn fake_kv_projection_layer[
    type: DType
](
    k_cache: ContiguousKVCache[type],
    v_cache: ContiguousKVCache[type],
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
    type: DType, transpose_k: Bool
](
    output: NDBuffer[type, 4],
    q: NDBuffer[type, 4],
    k_block_list: List[List[NDBuffer[type, 3]]],
    v_block_list: List[List[NDBuffer[type, 3]]],
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
    var block_size = k_block_list[0][0].dim[1]()
    var head_dim = q.dim[3]()

    # Allocate intermediate memory buffer.
    var score_size = batch_size * num_heads * seq_len * kv_seq_len
    var score_ptr = DTypePointer[type].alloc(score_size)
    var score = NDBuffer[type, 4](
        score_ptr, Index(batch_size, num_heads, seq_len, kv_seq_len)
    )

    batched_matmul[4, type, type, type, True, transpose_k](
        score, q, k_block_list
    )

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

    batched_matmul[4, type, type, type, False, False](
        output, score, v_block_list
    )
    score_ptr.free()


# TODO this is pretty specialized, let's give it a different name
@always_inline
fn batched_matmul[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    trans_b: Bool,
    trans_blocks: Bool,
](
    c_buf: NDBuffer[c_type, rank],
    a_buf: NDBuffer[a_type, rank],
    b_buf: List[
        List[NDBuffer[b_type, rank - 1]]
    ],  # TODO different params for rank and block_rank
):
    var outer_batch_size = c_buf.dim[0]()
    var inner_batch_size = c_buf.dim[1]()
    var M = c_buf.dim[2]()
    var N = c_buf.dim[3]()
    var K = a_buf.dim[3]()
    alias block_size_dim = 2 if trans_blocks else 1
    var block_size = b_buf[0][0].dim[block_size_dim]()

    fn _b_input(obs: Int, ibs: Int, n: Int, k: Int) -> Scalar[b_type]:
        var block_idx: Int
        var idx_in_block: Int
        var i: Int

        @parameter
        if trans_b:
            block_idx = n // block_size
            idx_in_block = n % block_size
            i = k
        else:
            block_idx = k // block_size
            idx_in_block = k % block_size
            i = n

        @parameter
        if trans_blocks:
            return b_buf[obs][block_idx][ibs, i, idx_in_block]
        else:
            return b_buf[obs][block_idx][ibs, idx_in_block, i]

    # TODO this is hella slow. parallelize, vectorize, and tile
    for obs in range(outer_batch_size):
        for ibs in range(inner_batch_size):
            for m in range(M):
                for n in range(N):
                    var accum = Scalar[c_type](0.0)
                    for k in range(K):
                        var a_val = a_buf[obs, ibs, m, k].cast[c_type]()
                        var b_val = _b_input(obs, ibs, n, k).cast[c_type]()
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


def test_mha_block_list[type: DType, seq_len: Int, is_k_transposed: Bool]():
    print("seq_len:", seq_len, "k_trans?:", is_k_transposed)
    # Query, key, value dimensions.
    alias batch_size = 1
    alias num_heads = 12
    alias depth = 64
    alias mask_val = Float32(-1e10)
    alias scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    # this needs to align with tile sizes for target arch, otherwise we will
    # try to load beyond the block size for a given tile.
    # TODO figure out a way to retrieve max tile size from single source
    alias block_size = 32 if has_neon() else 64

    # Q, K, V shapes.
    alias BHSD = DimList(batch_size, num_heads, seq_len, depth)

    alias qkv_size = batch_size * num_heads * seq_len * depth
    alias kv_block_size = num_heads * block_size * depth

    alias v_block_shape = DimList(num_heads, block_size, depth)
    alias k_block_shape = DimList(
        num_heads, depth, block_size
    ) if is_k_transposed else v_block_shape

    # Allocate memory for all variables.
    var q_ptr = DTypePointer[type].alloc(qkv_size)
    var k_ptr = DTypePointer[type].alloc(qkv_size)
    var v_ptr = DTypePointer[type].alloc(qkv_size)
    var mask_ptr = DTypePointer[type].alloc(seq_len * seq_len)
    var output_ptr = DTypePointer[type].alloc(qkv_size)
    var block_list_output_ptr = DTypePointer[type].alloc(qkv_size)

    # Q, K, V are randomly initialized.
    rand(q_ptr, qkv_size)
    rand(k_ptr, qkv_size)
    rand(v_ptr, qkv_size)
    var q = NDBuffer[type, 4, BHSD](q_ptr)
    var k = NDBuffer[type, 4, BHSD](k_ptr)
    var v = NDBuffer[type, 4, BHSD](v_ptr)

    # intialize block list cache  with values from the contiguous tensor
    var num_full_blocks = seq_len // block_size
    var tokens_in_last_block = seq_len % block_size
    var k_block_list = List[List[NDBuffer[type, 3]]]()
    var v_block_list = List[List[NDBuffer[type, 3]]]()
    for bs in range(batch_size):
        k_block_list.append(List[NDBuffer[type, 3]]())
        v_block_list.append(List[NDBuffer[type, 3]]())
        for block_idx in range(ceildiv(seq_len, block_size)):
            var k_block_ptr = DTypePointer[type].alloc(kv_block_size)
            var v_block_ptr = DTypePointer[type].alloc(kv_block_size)

            if block_idx < num_full_blocks:
                for h in range(num_heads):
                    memcpy(
                        v_block_ptr + h * block_size * depth,
                        v._offset((bs, h, block_idx * block_size, 0)),
                        depth * block_size,
                    )

                    @parameter
                    if is_k_transposed:
                        for d in range(depth):
                            for b in range(block_size):
                                k_block_ptr[
                                    h * depth * block_size + d * block_size + b
                                ] = k[
                                    Index(bs, h, block_idx * block_size + b, d)
                                ]
                    else:
                        memcpy(
                            k_block_ptr + h * block_size * depth,
                            k._offset((bs, h, block_idx * block_size, 0)),
                            depth * block_size,
                        )
            else:
                for s in range(tokens_in_last_block):
                    for h in range(num_heads):
                        memcpy(
                            v_block_ptr + h * block_size * depth + s * depth,
                            v._offset((bs, h, block_idx * block_size + s, 0)),
                            depth,
                        )

                        @parameter
                        if is_k_transposed:
                            for d in range(depth):
                                k_block_ptr[
                                    h * depth * block_size + d * block_size + s
                                ] = k[
                                    Index(bs, h, block_idx * block_size + s, d)
                                ]
                        else:
                            memcpy(
                                k_block_ptr
                                + h * block_size * depth
                                + s * depth,
                                k._offset(
                                    (bs, h, block_idx * block_size + s, 0)
                                ),
                                depth,
                            )

            k_block_list[-1].append(
                NDBuffer[type, 3](
                    k_block_ptr, k_block_shape
                ).make_dims_unknown()
            )
            v_block_list[-1].append(
                NDBuffer[type, 3](
                    v_block_ptr, v_block_shape
                ).make_dims_unknown()
            )

    # Set triangular mask
    for b in range(seq_len):
        for i in range(b + 1):
            mask_ptr[b * seq_len + i] = 0
        for i in range(b + 1, seq_len):
            mask_ptr[b * seq_len + i] = mask_val.cast[type]()

    # Contruct buffers.

    var mask = NDBuffer[type, 2](mask_ptr, Index(seq_len, seq_len))
    var mask_4d = NDBuffer[type, 4](
        mask_ptr,
        Index(batch_size, num_heads, seq_len, seq_len),
        Index(0, 0, seq_len, 1),
    )
    var output = NDBuffer[type, 4, BHSD](output_ptr)
    var mha_output = NDBuffer[type, 4, BHSD](block_list_output_ptr)

    @parameter
    fn do_naive_attn():
        _naive_attention[type, True](
            output.make_dims_unknown(),
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
        _naive_block_list_attention[type, is_k_transposed](
            mha_output.make_dims_unknown(),
            q.make_dims_unknown(),
            k_block_list,
            v_block_list,
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
            output.make_dims_unknown(),
            mha_output.make_dims_unknown(),
        )
    )

    @parameter
    if is_k_transposed:
        var fa_output_ptr = DTypePointer[type].alloc(qkv_size)
        var fa_output = NDBuffer[type, 4, BHSD](fa_output_ptr)

        @parameter
        fn input_k_fn[
            width: Int, rank: Int
        ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
            constrained[
                width <= block_size, "Expected width to be less than block size"
            ]()
            # idx == bs, num_heads, head_dim, kv_len
            var bs = idx[0]
            var head_idx = idx[1]
            var head_d_idx = idx[2]
            var seq = idx[3]
            var block_idx = seq // block_size
            var idx_in_block = seq % block_size

            return k_block_list[bs][block_idx].load[width=width](
                (head_idx, head_d_idx, idx_in_block)
            )

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
            var head_d_idx = idx[3]
            var seq = idx[2]
            var block_idx = seq // block_size
            var idx_in_block = seq % block_size
            return v_block_list[bs][block_idx].load[width=width](
                head_idx, idx_in_block, head_d_idx
            )

        @parameter
        fn input_mask_fn[
            width: Int, rank: Int
        ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
            return mask_4d.load[width=width](rebind[StaticIntTuple[4]](idx))

        var fa_k_shape = StaticIntTuple[4](
            batch_size, num_heads, depth, seq_len
        )
        var fa_v_shape = StaticIntTuple[4](
            batch_size, num_heads, seq_len, depth
        )

        @parameter
        fn do_flash_attn():
            flash_attention[type, 4, input_k_fn, input_v_fn, input_mask_fn,](
                q.make_dims_unknown(),
                fa_k_shape,
                fa_v_shape,
                fa_output.make_dims_unknown(),
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
                output.make_dims_unknown(),
                fa_output.make_dims_unknown(),
            )
        )
        fa_output_ptr.free()

    print()

    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    mask_ptr.free()
    output_ptr.free()
    block_list_output_ptr.free()
    for i in range(len(v_block_list)):
        for j in range(len(v_block_list[i])):
            v_block_list[i][j].data.free()
            k_block_list[i][j].data.free()


def main():
    test_mha_block_list[DType.float32, 128, False]()
    test_mha_block_list[DType.float32, 2, False]()
    test_mha_block_list[DType.float32, 135, False]()
    test_mha_block_list[DType.float32, 128, True]()
    test_mha_block_list[DType.float32, 2, True]()
    test_mha_block_list[DType.float32, 135, True]()
    test_generation_loop()
