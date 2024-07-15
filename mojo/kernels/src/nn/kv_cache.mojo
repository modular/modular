# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from buffer import DimList, NDBuffer, Dim
from math import rsqrt
from sys.info import _current_target
from utils import Index
from utils.numerics import min_finite, isnan

from algorithm.functional import _elementwise_impl
from gpu.host import Stream, DeviceContext, DeviceBuffer
from gpu.host._compile import _get_nvptx_target
from linalg import transpose
from linalg.matmul_gpu import _matmul_gpu
from linalg.matmul import matmul
from nn.flash_attention import flash_attention as cpu_flash_attention
from nn.mha import flash_attention as gpu_flash_attention
from register import mogg_register, mogg_register_shape_func
from runtime.llcl import (
    MojoCallContextPtr,
)

from .types import ContiguousKVCache, ContiguousKVCacheCollection, KVCacheLayout


@mogg_register("kv_cache_length")
@export
fn kv_cache_length(
    kv_collection: ContiguousKVCacheCollection[
        DType.float32, KVCacheLayout.BSHD
    ],
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
    kv_collection: ContiguousKVCacheCollection[
        DType.float32, KVCacheLayout.BSHD
    ],
) -> ContiguousKVCache[DType.float32, KVCacheLayout.BSHD]:
    """Retrieves the Key cache for the given layer."""
    return kv_collection.get_key_cache(int(layer_idx))


@mogg_register("value_cache_for_layer")
@export
fn value_cache_for_layer(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32, KVCacheLayout.BSHD
    ],
) -> ContiguousKVCache[DType.float32, KVCacheLayout.BSHD]:
    """Retrieves the Value cache for the given layer."""
    return kv_collection.get_value_cache(int(layer_idx))


@mogg_register("matmul_kv_cache")
@export
fn matmul_kv_cache[
    hidden_state_shape: DimList,
    weight_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[DType.float32, 3, hidden_state_shape],
    weight: NDBuffer[DType.float32, 2, weight_shape],
    cache: ContiguousKVCache[DType.float32, KVCacheLayout.BSHD],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[DType.float32, KVCacheLayout.BSHD]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, num_kv_heads, max_seq_len, head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    return _matmul_kv_cache[hidden_state_shape, weight_shape, target=target](
        hidden_state, weight, cache, ctx
    )


@mogg_register("rope_kv_cache")
@export
fn rope_kv_cache[
    freqs_shape: DimList, target: StringLiteral = "cpu"
](
    cache: ContiguousKVCache[DType.float32, KVCacheLayout.BSHD],
    freqs: NDBuffer[DType.float32, 2, freqs_shape],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[DType.float32, KVCacheLayout.BSHD]:
    var valid_len = int(cache.get_valid_lengths()[0])

    @always_inline
    @parameter
    @__copy_capture(freqs, valid_len)
    fn rope_fn[width: Int, rank: Int](idx: StaticIntTuple[rank]):
        @parameter
        if width == 1:
            print("ROPE KERNEL CALLED WITH SINGLE VALUE, EXPECTED AT LEAST 2")
        else:
            var bs = idx[0]
            var head_idx = idx[1]
            var t_cache_idx = idx[2] + valid_len
            var hd_idx = idx[3]
            var val = cache.load[width=width](bs, head_idx, t_cache_idx, hd_idx)
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

            var r_c = rebind[SIMD[DType.float32, width]](r_re.interleave(r_im))
            cache.store[width=width](bs, head_idx, t_cache_idx, hd_idx, r_c)

    alias compile_target = _current_target() if target == "cpu" else _get_nvptx_target()
    alias simd_width = simdwidthof[DType.float32, target=compile_target]()
    var launch_shape = StaticIntTuple[4](
        cache.batch_size, cache.num_heads, freqs.dim[0](), freqs.dim[1]()
    )
    _elementwise_impl[rope_fn, simd_width, 4, target=target](launch_shape, ctx)
    return cache


alias elementwise_fn_type = fn[type: DType, width: Int] (
    StaticIntTuple[4], SIMD[type, width]
) capturing -> SIMD[type, width]


@always_inline
fn _matmul_kv_cache[
    hidden_state_shape: DimList,
    weight_shape: DimList,
    *,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[DType.float32, 3, hidden_state_shape],
    weight: NDBuffer[DType.float32, 2, weight_shape],
    cache: ContiguousKVCache[DType.float32, KVCacheLayout.BSHD],
    context: MojoCallContextPtr = MojoCallContextPtr(),
) -> ContiguousKVCache[DType.float32, KVCacheLayout.BSHD]:
    """Helper for performing matmul with custom ContiguousKVCache types.

    Parameters:
        hidden_state_shape: The static shapes for the hidden_state tensor
        weight_shape: The static shapes for the weight tensor
        target: StringLiteral identifying the device target (cpu vs cuda)

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

    var valid_len = int(cache.get_valid_lengths()[0])

    @parameter
    @__copy_capture(cache, SEQ_LEN, valid_len)
    fn write_to_cache[
        type_: DType, width: Int
    ](idx: StaticIntTuple[2], val: SIMD[type_, width]):
        var bs_and_seq = divmod(idx[0], SEQ_LEN)
        var b_idx = bs_and_seq[0]
        var t_idx = bs_and_seq[1]
        var head_and_dim = divmod(idx[1], cache.head_size)
        var h_idx = head_and_dim[0]
        var hd_idx = head_and_dim[1]

        var cache_t_idx = t_idx + valid_len
        cache.store[width](
            b_idx,
            h_idx,
            cache_t_idx,
            hd_idx,
            rebind[SIMD[DType.float32, width]](val),
        )

    var hidden_state_2d = NDBuffer[
        DType.float32, 2, DimList(Dim(), hidden_state.shape.get[2]())
    ](
        hidden_state.data,
        StaticIntTuple[2](BS * SEQ_LEN, K),
    )

    var c_ptr = DTypePointer[DType.float32].alloc(
        BS * SEQ_LEN * N
    ) if target == "cpu" else DTypePointer[DType.float32]()

    var c_nd = NDBuffer[
        DType.float32, 2, DimList(Dim(), weight.shape.get[1]())
    ](
        c_ptr,
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
    ](c_nd, hidden_state_2d, weight, ctx=context)
    c_nd.data.free()
    return cache


@mogg_register_shape_func("flash_attention_kv_cache")
@export
fn flash_attention_kv_cache_shape_func[
    q_shape: DimList,
    mask_shape: DimList,
    scale_shape: DimList,
    single_thread_blocking_override: Bool,
](
    q: NDBuffer[DType.float32, 4, q_shape],
    k: ContiguousKVCache[DType.float32, KVCacheLayout.BSHD],
    v: ContiguousKVCache[DType.float32, KVCacheLayout.BSHD],
    mask: NDBuffer[DType.float32, 2, mask_shape],
    scale: NDBuffer[DType.float32, 1, scale_shape],
) -> StaticIntTuple[4]:
    return q.dynamic_shape


@mogg_register("flash_attention_kv_cache")
@export
fn flash_attention_kv_cache[
    q_shape: DimList,
    mask_shape: DimList,
    scale_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[DType.float32, 4, q_shape],
    k: ContiguousKVCache[DType.float32, KVCacheLayout.BSHD],
    v: ContiguousKVCache[DType.float32, KVCacheLayout.BSHD],
    mask: NDBuffer[DType.float32, 2, mask_shape],
    scale: NDBuffer[DType.float32, 1, scale_shape],
    output: NDBuffer[DType.float32, 4, output_shape],
    context: MojoCallContextPtr,
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
        return _flash_attention_kv_cache_cpu[
            q_shape,
            mask_shape,
            scale_shape,
            output_shape,
        ](q, k, v, mask, scale, output, context)
    else:
        return _flash_attention_kv_cache_gpu[
            q_shape,
            mask_shape,
            scale_shape,
            output_shape,
        ](q, k, v, mask, scale, output, context)


fn _flash_attention_kv_cache_cpu[
    q_shape: DimList,
    mask_shape: DimList,
    scale_shape: DimList,
    output_shape: DimList,
](
    q: NDBuffer[DType.float32, 4, q_shape],
    k: ContiguousKVCache[DType.float32, KVCacheLayout.BSHD],
    v: ContiguousKVCache[DType.float32, KVCacheLayout.BSHD],
    mask: NDBuffer[DType.float32, 2, mask_shape],
    scale: NDBuffer[DType.float32, 1, scale_shape],
    output: NDBuffer[DType.float32, 4, output_shape],
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

    cpu_flash_attention[
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


fn _flash_attention_kv_cache_gpu[
    q_shape: DimList,
    mask_shape: DimList,
    scale_shape: DimList,
    output_shape: DimList,
](
    q: NDBuffer[DType.float32, 4, q_shape],
    k: ContiguousKVCache[DType.float32, KVCacheLayout.BSHD],
    v: ContiguousKVCache[DType.float32, KVCacheLayout.BSHD],
    mask: NDBuffer[DType.float32, 2, mask_shape],
    scale: NDBuffer[DType.float32, 1, scale_shape],
    output: NDBuffer[DType.float32, 4, output_shape],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    # TODO fix scale, right now it's passed in as GPU buffer but it's needed as a scalar
    # GRA-750
    var k_nd = NDBuffer[DType.float32, 4, DimList(Dim(), Dim(), 6, 48)](
        k.block.data, q.dynamic_shape
    )
    var v_nd = NDBuffer[DType.float32, 4, DimList(Dim(), Dim(), 6, 48)](
        v.block.data, q.dynamic_shape
    )
    var mask_nd = NDBuffer[DType.float32, 3, DimList(Dim(), Dim(), Dim()),](
        mask.data,
        StaticIntTuple[3](q.dim[0](), mask.dim[0](), mask.dim[1]()),
    )
    try:
        gpu_flash_attention[
            4,
            3,
            q.shape,
            k_nd.shape,
            v_nd.shape,
            mask_nd.shape,
            output.shape,
            q.type,
            k_nd.type,
            v_nd.type,
            mask_nd.type,
            output.type,
            add_attn_mask=True,
            target="cuda",
        ](
            output,
            q,
            k_nd,
            v_nd,
            mask_nd,
            # TODO take scale from argument GRA-750
            rsqrt(Float32(k.head_size)),
            context,
        )
    except e:
        print("Error in GPU Flash Attention:", e)
