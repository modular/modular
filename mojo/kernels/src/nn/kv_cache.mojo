# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from buffer import DimList, NDBuffer
from extensibility import Tensor as ExtensibilityTensor, empty_tensor
from gpu.host import Stream, DeviceContext
from linalg.matmul_gpu import _matmul_gpu
from linalg.matmul import matmul
from max.extensibility import Tensor
from register import mogg_register
from utils.numerics import min_finite, isnan
from nn.flash_attention import flash_attention
from register import mogg_register
from buffer import NDBuffer, DimList

from .types import KVCache, KVCacheCollection


@mogg_register("kv_cache_length")
@export
fn kv_cache_length(
    kv_collection: KVCacheCollection[DType.float32, False]
) -> ExtensibilityTensor[DType.int64, 1]:
    """Returns the size of the cache in a KVCacheCollection mo.opaque object."""
    var retval = empty_tensor[DType.int64](StaticIntTuple[1](1))
    var valid_length = kv_collection.get_valid_lengths()[0]

    retval.store[1](0, valid_length)
    return retval^


@mogg_register("key_cache_for_layer")
@export
fn key_cache_for_layer(
    layer_idx: ExtensibilityTensor[DType.int64, 1],
    kv_collection: KVCacheCollection[DType.float32, False],
) -> KVCache[DType.float32, False]:
    """Retrieves the Key cache for the given layer."""
    var val = layer_idx.simd_load[simd_width=1](0)

    return kv_collection.get_key_cache(int(val))


@mogg_register("value_cache_for_layer")
@export
fn value_cache_for_layer(
    layer_idx: ExtensibilityTensor[DType.int64, 1],
    kv_collection: KVCacheCollection[DType.float32, False],
) -> KVCache[DType.float32, False]:
    """Retrieves the Value cache for the given layer."""
    return kv_collection.get_value_cache(
        int(layer_idx.simd_load[simd_width=1](0))
    )


@mogg_register("matmul_kv_cache")
@export
fn matmul_kv_cache(
    hidden_state: ExtensibilityTensor[DType.float32, 3],
    weight: ExtensibilityTensor[DType.float32, 2],
    cache: KVCache[DType.float32, False],
) -> KVCache[DType.float32, False]:
    """Performs a matmul, writing the output into a mutable KVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical KVCache, with logical shape:
            (batch_size, num_kv_heads, max_seq_len, head_size).
    """
    return _matmul_kv_cache(hidden_state, weight, cache)


@mogg_register("matmul_kv_cache_with_rope")
@export
fn matmul_kv_cache_with_rope(
    hidden_state: ExtensibilityTensor[DType.float32, 3],
    weight: ExtensibilityTensor[DType.float32, 2],
    freqs: ExtensibilityTensor[DType.float32, 2],
    cache: KVCache[DType.float32, False],
) -> KVCache[DType.float32, False]:
    """Performs a matmul, writing the output w/ rope embeddings into a
    mutable KVCache object.
    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        freqs: RoPE embeddings with shape (seq_len, head_size).
        cache: The historical KVCache, with logical shape:
            (batch_size, num_kv_heads, max_seq_len, head_size).
    """

    @always_inline
    @parameter
    fn rope_fn[
        type: DType, width: Int
    ](idx: StaticIntTuple[4], val: SIMD[type, width]) -> SIMD[type, width]:
        @parameter
        if width == 1:
            print("ROPE KERNEL CALLED WITH SINGLE VALUE, EXPECTED AT LEAST 2")
            return val
        else:
            var val_cast = rebind[SIMD[DType.float32, width]](val)

            var x_c = val_cast.deinterleave()
            var x_re = x_c[0]
            var x_im = x_c[1]

            var f_idx = StaticIntTuple[2](idx[2], idx[3])
            var f_c_temp = freqs.simd_load[width](f_idx)

            var f_c = f_c_temp.deinterleave()
            var f_re = f_c[0]
            var f_im = f_c[1]

            var r_re = (x_re * f_re) - (x_im * f_im)
            var r_im = (x_re * f_im) + (x_im * f_re)

            var r_c = r_re.interleave(r_im)
            return rebind[SIMD[type, width]](r_c)

    return _matmul_kv_cache[rope_fn](hidden_state, weight, cache)


alias elementwise_fn_type = fn[type: DType, width: Int] (
    StaticIntTuple[4], SIMD[type, width]
) capturing -> SIMD[type, width]


@always_inline
fn _matmul_kv_cache[
    elemwise_fn: OptionalReg[elementwise_fn_type] = None
](
    hidden_state: ExtensibilityTensor[DType.float32, 3],
    weight: ExtensibilityTensor[DType.float32, 2],
    cache: KVCache[DType.float32, False],
) -> KVCache[DType.float32, False]:
    """Helper for performing matmul with custom KVCache types.

    Parameters:
        elemwise_fn: An optional functor to execute before writing output to
            cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size)
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        cache: The historical KVCache, with logical shape:
            (batch_size, num_kv_heads, max_seq_len, head_size)
    """
    # TODO internalize this info in the cache object?
    var BS = hidden_state.shape[0]
    var SEQ_LEN = hidden_state.shape[1]
    var N = weight.shape[1]
    var K = weight.shape[0]

    @parameter
    @__copy_capture(SEQ_LEN)
    fn write_to_cache[
        type_: DType, width: Int
    ](idx: StaticIntTuple[2], val: SIMD[type_, width]):
        var bs_and_seq = divmod(idx[0], SEQ_LEN)
        var b_idx = bs_and_seq[0]
        var t_idx = bs_and_seq[1]
        var head_and_dim = divmod(
            idx[1], cache.register_passable_cache.head_size
        )
        var h_idx = head_and_dim[0]
        var hd_idx = head_and_dim[1]

        var result = val

        @parameter
        if elemwise_fn:
            alias func = elemwise_fn.value()
            result = func(
                StaticIntTuple[4](b_idx, h_idx, t_idx, hd_idx), result
            )

        var cache_t_idx = t_idx + int(cache.get_valid_lengths()[b_idx])
        cache.store[width](
            b_idx,
            h_idx,
            cache_t_idx,
            hd_idx,
            rebind[SIMD[DType.float32, width]](result),
        )

    var hidden_state_nd = NDBuffer[DType.float32, 2](
        hidden_state.data,
        StaticIntTuple[2](BS * SEQ_LEN, K),
    )
    var weight_nd = NDBuffer[DType.float32, 2](weight.data, weight.shape)

    # TODO figure out how to avoid this malloc.
    var c_nd = NDBuffer[DType.float32, 2](
        DTypePointer[DType.float32].alloc(BS * SEQ_LEN * N),
        StaticIntTuple[2](BS * SEQ_LEN, N),
    )
    matmul[
        DType.float32,
        hidden_state_nd.shape,
        DType.float32,
        weight_nd.shape,
        DType.float32,
        c_nd.shape,
        transpose_b=False,
        elementwise_lambda_fn=write_to_cache,
    ](c_nd, hidden_state_nd, weight_nd)
    c_nd.data.free()
    return cache


# TODO onboard to mogg_register after GRA-607
fn matmul_kv_cache_gpu[
    type: DType, transpose_block: Bool
](
    hidden_state: NDBuffer[type, 2],
    weight: NDBuffer[type, 2],
    cache: KVCache[type, transpose_block],
    stream: Stream,
) raises:
    """Performs a GPU matmul, writing the output into a mutable KVCache object.

    TODO unify with CPU implementation above.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical KVCache, with logical shape:
            (batch_size, num_kv_heads, max_seq_len, head_size).
        stream: The stream to use when scheduling the kernel launch.
    """
    var M = hidden_state.dim[0]()
    var seq_len = M // cache.register_passable_cache.batch_size

    @parameter
    @__copy_capture(seq_len)
    fn write_to_cache[
        type_: DType, width: Int
    ](idx: StaticIntTuple[2], val: SIMD[type_, width]):
        var bs_and_seq = divmod(idx[0], seq_len)
        var b_idx = bs_and_seq[0]
        var t_idx = bs_and_seq[1] + int(cache.get_valid_lengths()[b_idx])
        var head_and_dim = divmod(
            idx[1], cache.register_passable_cache.head_size
        )
        var h_idx = head_and_dim[0]
        var hd_idx = head_and_dim[1]

        cache.store[width](
            b_idx, h_idx, t_idx, hd_idx, rebind[SIMD[type, width]](val)
        )

    # TODO we need to allocate a buffer here, C is used to store intermediate outputs
    var c_nd = NDBuffer[type, 2](
        DTypePointer[type](),
        StaticIntTuple[2](hidden_state.dim[0](), weight.dim[1]()),
    )

    _matmul_gpu[elementwise_lambda_fn=write_to_cache,](
        c_nd, hidden_state, weight, DeviceContext(stream)
    )


@mogg_register("flash_attention_kv_cache")
@export
fn flash_attention_kv_cache(
    q: ExtensibilityTensor[DType.float32, 4],
    k: KVCache[DType.float32, False],
    v: KVCache[DType.float32, False],
    mask: ExtensibilityTensor[DType.float32, 2],
    scale: ExtensibilityTensor[DType.float32, 1],
) -> ExtensibilityTensor[DType.float32, 4]:
    """Performs flash attention using k and v caches from KVCache custom types.

    Args:
        q: Tensor with shape (batch_size, num_heads, seq_len, head_size).
        k: KVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        v: KVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        mask: The attention mask to apply to the score matrix.
        scale: The scaled factor in scaled-dot product attention. Usually rsqrt(head_size).

    Returns:
        Symbol tensor with shape (batch_size, num_heads, seq_len, head_size).
    """
    var output = empty_tensor[DType.float32](q.shape)
    var mask_nd = NDBuffer[DType.float32, 2](mask.data, mask.shape)

    @parameter
    fn input_k_fn[
        width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[DType.float32, width]:
        var bs = idx[0]
        var head_idx = idx[1]
        var seq = idx[2]
        var head_d_idx = idx[3]

        var retval = k.load[width=width](bs, head_idx, seq, head_d_idx)
        return retval

    @parameter
    fn input_v_fn[
        width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[DType.float32, width]:
        var bs = idx[0]
        var head_idx = idx[1]
        var seq = idx[2]
        var head_d_idx = idx[3]
        var retval = v.load[width=width](bs, head_idx, seq, head_d_idx)
        return retval

    @parameter
    @__copy_capture(mask_nd)
    fn input_mask_fn[
        width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[DType.float32, width]:
        return mask_nd.load[width=width]((idx[2], idx[3]))

    var batch_size = q.shape[0]
    var num_heads = q.shape[1]
    var depth = q.shape[3]
    var new_seq_len = q.shape[2]
    var cache_seq_len = int(k.get_valid_lengths()[0])
    var seq_len = new_seq_len + cache_seq_len
    var fa_k_shape = StaticIntTuple[4](batch_size, num_heads, seq_len, depth)
    var fa_v_shape = StaticIntTuple[4](batch_size, num_heads, seq_len, depth)

    var q_nd = NDBuffer[DType.float32, 4](q.data, q.shape)
    var output_nd = NDBuffer[DType.float32, 4](output.data, output.shape)

    flash_attention[
        DType.float32,
        4,
        input_k_fn,
        input_v_fn,
        input_mask_fn,
        transpose_k=True,
    ](
        q_nd.make_dims_unknown(),
        fa_k_shape,
        fa_v_shape,
        output_nd,
        scale.simd_load[1](0),
    )
    return output^


# TODO onboard to mogg_register after GRA-607
fn kv_cache_flash_attention_gpu[
    rank: Int,
    q_shape: DimList,
    mask_shape: DimList,
    output_shape: DimList,
    q_type: DType,
    k_type: DType,
    v_type: DType,
    mask_type: DType,
    output_type: DType,
    transpose_k: Bool,
    # llama 2 has attention mask but not causal mask.
    add_attn_mask: Bool = True,
](
    output: NDBuffer[output_type, rank, output_shape],
    q: NDBuffer[q_type, rank, q_shape],
    k: KVCache[k_type, transpose_k],
    v: KVCache[v_type, False],
    mask: NDBuffer[mask_type, 3, mask_shape],
    scale: Float32,
) raises:
    """Performs GPU flash attention using k and v caches from KVCache custom types.

    TODO implement and unify behind the same interface as CPU.

    Args:
        output: The pre-allocated output buffer with shape (batch_size, num_heads, seq_len, head_size).
        q: Tensor with shape (batch_size, num_heads, seq_len, head_size).
        k: KVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        v: KVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        mask: The attention mask to apply to the score matrix.
        scale: The scaled factor in scaled-dot product attention. Usually rsqrt(head_size).
    """
    # tracked by KERN-496
    raise "TODO implement flash attention kernel on GPU"
