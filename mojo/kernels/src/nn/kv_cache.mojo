# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from buffer import DimList, NDBuffer
from gpu.host import Stream, DeviceContext
from linalg.matmul_gpu import _matmul_gpu
from linalg.matmul import matmul
from register import mogg_register, mogg_register_shape_func
from utils.numerics import min_finite, isnan
from nn.flash_attention import flash_attention
from register import mogg_register
from buffer import NDBuffer, DimList
from utils import Index
from .types import ContiguousKVCache, ContiguousKVCacheCollection
from runtime.llcl import (
    MojoCallContextPtr,
)


@mogg_register("kv_cache_length")
@export
fn kv_cache_length(
    kv_collection: ContiguousKVCacheCollection[DType.float32, False],
    output: NDBuffer[DType.int64, 1],
):
    """Returns the size of the cache in a ContiguousKVCacheCollection mo.opaque object.
    """
    var valid_length = kv_collection.get_valid_lengths()[0]

    output.store[width=1](Index(0), valid_length)


@mogg_register("key_cache_for_layer")
@export
fn key_cache_for_layer(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[DType.float32, False],
) -> ContiguousKVCache[DType.float32, False]:
    """Retrieves the Key cache for the given layer."""
    return kv_collection.get_key_cache(int(layer_idx))


@mogg_register("value_cache_for_layer")
@export
fn value_cache_for_layer(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[DType.float32, False],
) -> ContiguousKVCache[DType.float32, False]:
    """Retrieves the Value cache for the given layer."""
    return kv_collection.get_value_cache(int(layer_idx))


@mogg_register("matmul_kv_cache")
@export
fn matmul_kv_cache[
    target: StringLiteral = "cpu"
](
    hidden_state: NDBuffer[DType.float32, 3],
    weight: NDBuffer[DType.float32, 2],
    cache: ContiguousKVCache[DType.float32, False],
) -> ContiguousKVCache[DType.float32, False]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, num_kv_heads, max_seq_len, head_size).
    """
    return _matmul_kv_cache[target=target](hidden_state, weight, cache)


@mogg_register("matmul_kv_cache_with_rope")
@export
fn matmul_kv_cache_with_rope[
    target: StringLiteral = "cpu"
](
    hidden_state: NDBuffer[DType.float32, 3],
    weight: NDBuffer[DType.float32, 2],
    freqs: NDBuffer[DType.float32, 2],
    cache: ContiguousKVCache[DType.float32, False],
) -> ContiguousKVCache[DType.float32, False]:
    """Performs a matmul, writing the output w/ rope embeddings into a
    mutable ContiguousKVCache object.
    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        freqs: RoPE embeddings with shape (seq_len, head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, num_kv_heads, max_seq_len, head_size).
    """

    @always_inline
    @parameter
    @__copy_capture(freqs)
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
            var f_c_temp = freqs.load[width=width](f_idx)

            var f_c = f_c_temp.deinterleave()
            var f_re = f_c[0]
            var f_im = f_c[1]

            var r_re = (x_re * f_re) - (x_im * f_im)
            var r_im = (x_re * f_im) + (x_im * f_re)

            var r_c = r_re.interleave(r_im)
            return rebind[SIMD[type, width]](r_c)

    return _matmul_kv_cache[target=target, elemwise_fn=rope_fn](
        hidden_state, weight, cache
    )


alias elementwise_fn_type = fn[type: DType, width: Int] (
    StaticIntTuple[4], SIMD[type, width]
) capturing -> SIMD[type, width]


@always_inline
fn _matmul_kv_cache[
    *,
    target: StringLiteral = "cpu",
    elemwise_fn: OptionalReg[elementwise_fn_type] = None,
](
    hidden_state: NDBuffer[DType.float32, 3],
    weight: NDBuffer[DType.float32, 2],
    cache: ContiguousKVCache[DType.float32, False],
    context: MojoCallContextPtr = MojoCallContextPtr(),
) -> ContiguousKVCache[DType.float32, False]:
    """Helper for performing matmul with custom ContiguousKVCache types.

    Parameters:
        target: StringLiteral identifying the device target (cpu vs cuda)
        elemwise_fn: An optional functor to execute before writing output to
            cache.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size)
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, num_kv_heads, max_seq_len, head_size)
        context: Pointer containing the runtime context for the target device.
    """
    # TODO internalize this info in the cache object?
    var BS = hidden_state.dim[0]()
    var SEQ_LEN = hidden_state.dim[1]()
    var N = weight.dim[1]()
    var K = weight.dim[0]()

    @parameter
    @__copy_capture(cache, SEQ_LEN)
    fn write_to_cache[
        type_: DType, width: Int
    ](idx: StaticIntTuple[2], val: SIMD[type_, width]):
        var bs_and_seq = divmod(idx[0], SEQ_LEN)
        var b_idx = bs_and_seq[0]
        var t_idx = bs_and_seq[1]
        var head_and_dim = divmod(idx[1], cache.head_size)
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

    var hidden_state_2d = NDBuffer[DType.float32, 2](
        hidden_state.data,
        StaticIntTuple[2](BS * SEQ_LEN, K),
    )

    # TODO figure out how to avoid this malloc.
    var c_nd = NDBuffer[DType.float32, 2](
        DTypePointer[DType.float32].alloc(BS * SEQ_LEN * N),
        StaticIntTuple[2](BS * SEQ_LEN, N),
    )
    matmul[
        DType.float32,
        hidden_state_2d.shape,
        DType.float32,
        weight.shape,
        DType.float32,
        c_nd.shape,
        transpose_b=False,
        elementwise_lambda_fn=write_to_cache,
        target=target,
    ](c_nd, hidden_state_2d, weight)
    c_nd.data.free()
    return cache


@mogg_register_shape_func("flash_attention_kv_cache")
@export
fn flash_attention_kv_cache_shape_func[
    single_thread_blocking_override: Bool
](
    q: NDBuffer[DType.float32, 4],
    k: ContiguousKVCache[DType.float32, False],
    v: ContiguousKVCache[DType.float32, False],
    mask: NDBuffer[DType.float32, 2],
    scale: NDBuffer[DType.float32, 1],
) -> StaticIntTuple[4]:
    return q.dynamic_shape


@mogg_register("flash_attention_kv_cache")
@export
fn flash_attention_kv_cache[
    target: StringLiteral = "cpu",
](
    q: NDBuffer[DType.float32, 4],
    k: ContiguousKVCache[DType.float32, False],
    v: ContiguousKVCache[DType.float32, False],
    mask: NDBuffer[DType.float32, 2],
    scale: NDBuffer[DType.float32, 1],
    output: NDBuffer[DType.float32, 4],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    """Performs flash attention using k and v caches from ContiguousKVCache custom types.

    Args:
        q: NDBuffer with shape (batch_size, num_heads, seq_len, head_size).
        k: ContiguousKVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        v: ContiguousKVCache type with logical shape (batch_size, num_heads, max_seq_len, head_size).
        mask: The attention mask to apply to the score matrix.
        scale: The scaled factor in scaled-dot product attention. Usually rsqrt(head_size).
        output: The Pre-allocated output buffer to write results to. Has shape:
            (batch_size, num_heads, seq_len, head_size).
        context: Pointer containing the runtime context for the target device.
    """

    @parameter
    if target == "cpu":
        return _flash_attention_kv_cache_cpu(
            q, k, v, mask, scale, output, context
        )
    else:
        return _flash_attention_kv_cache_gpu(
            q, k, v, mask, scale, output, context
        )


fn _flash_attention_kv_cache_cpu(
    q: NDBuffer[DType.float32, 4],
    k: ContiguousKVCache[DType.float32, False],
    v: ContiguousKVCache[DType.float32, False],
    mask: NDBuffer[DType.float32, 2],
    scale: NDBuffer[DType.float32, 1],
    output: NDBuffer[DType.float32, 4],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    @parameter
    @__copy_capture(k)
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
    @__copy_capture(v)
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
    @__copy_capture(mask)
    fn input_mask_fn[
        width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[DType.float32, width]:
        return mask.load[width=width]((idx[2], idx[3]))

    var batch_size = q.dim[0]()
    var num_heads = q.dim[1]()
    var depth = q.dim[3]()
    var new_seq_len = q.dim[2]()
    var cache_seq_len = int(k.get_valid_lengths()[0])
    var seq_len = new_seq_len + cache_seq_len
    var fa_k_shape = StaticIntTuple[4](batch_size, num_heads, seq_len, depth)
    var fa_v_shape = StaticIntTuple[4](batch_size, num_heads, seq_len, depth)

    flash_attention[
        DType.float32,
        4,
        input_k_fn,
        input_v_fn,
        input_mask_fn,
        transpose_k=True,
    ](
        q.make_dims_unknown(),
        fa_k_shape,
        fa_v_shape,
        output,
        scale.load[width=1](0),
    )


fn _flash_attention_kv_cache_gpu(
    q: NDBuffer[DType.float32, 4],
    k: ContiguousKVCache[DType.float32, False],
    v: ContiguousKVCache[DType.float32, False],
    mask: NDBuffer[DType.float32, 2],
    scale: NDBuffer[DType.float32, 1],
    output: NDBuffer[DType.float32, 4],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    pass
