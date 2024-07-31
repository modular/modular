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
from register import mogg_register
from runtime.asyncrt import (
    MojoCallContextPtr,
)

from .types import (
    ContiguousKVCache,
    ContiguousKVCacheCollection,
    KVCacheLayout,
    KVCacheStaticParams,
)


@value
struct KVCacheKernelNames:
    var matmul_kernel: StringLiteral
    var rope_kernel: StringLiteral
    var flash_attention_kernel: StringLiteral
    var kv_cache_length_kernel: StringLiteral
    var key_cache_for_layer_kernel: StringLiteral
    var value_cache_for_layer_kernel: StringLiteral


fn _kv_cache_kernel_names[params: KVCacheStaticParams]() -> KVCacheKernelNames:
    @parameter
    if params == KVCacheStaticParams(
        num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
    ):
        return KVCacheKernelNames(
            matmul_kernel="matmul_kv_cache_h6_d48_bshd",
            rope_kernel="rope_kv_cache_h6_d48_bshd",
            flash_attention_kernel="flash_attention_kv_cache_h6_d48_bshd",
            kv_cache_length_kernel="kv_cache_length_h6_d48_bshd",
            key_cache_for_layer_kernel="key_cache_for_layer_h6_d48_bshd",
            value_cache_for_layer_kernel="value_cache_for_layer_h6_d48_bshd",
        )
    elif params == KVCacheStaticParams(
        num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
    ):
        return KVCacheKernelNames(
            matmul_kernel="matmul_kv_cache_h6_d48_bhsd",
            rope_kernel="rope_kv_cache_h6_d48_bhsd",
            flash_attention_kernel="flash_attention_kv_cache_h6_d48_bhsd",
            kv_cache_length_kernel="kv_cache_length_h6_d48_bhsd",
            key_cache_for_layer_kernel="key_cache_for_layer_h6_d48_bhsd",
            value_cache_for_layer_kernel="value_cache_for_layer_h6_d48_bhsd",
        )
    else:
        constrained[False, "Unsupported KV Cache configuration"]()

    return KVCacheKernelNames(
        matmul_kernel="",
        rope_kernel="",
        flash_attention_kernel="",
        kv_cache_length_kernel="",
        key_cache_for_layer_kernel="",
        value_cache_for_layer_kernel="",
    )


@mogg_register("kv_cache_length_h6_d48_bshd")
@export
fn kv_cache_length_h6_d48_bshd(
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    output: NDBuffer[DType.int64, 1],
):
    return _kv_cache_length(kv_collection, output)


@mogg_register("kv_cache_length_h6_d48_bhsd")
@export
fn kv_cache_length_h6_d48_bhsd(
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    output: NDBuffer[DType.int64, 1],
):
    return _kv_cache_length(kv_collection, output)


fn _kv_cache_length[
    kv_params: KVCacheStaticParams
](
    kv_collection: ContiguousKVCacheCollection[DType.float32, kv_params],
    output: NDBuffer[DType.int64, 1],
):
    """Returns the size of the cache in a ContiguousKVCacheCollection mo.opaque object.
    """
    var valid_length = kv_collection.get_valid_lengths()[0]
    output.store[width=1](Index(0), valid_length)


@mogg_register("key_cache_for_layer_h6_d48_bshd")
@export
fn key_cache_for_layer_h6_d48_bshd(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BSHD),
]:
    return _key_cache_for_layer(layer_idx, kv_collection)


@mogg_register("key_cache_for_layer_h6_d48_bhsd")
@export
fn key_cache_for_layer_h6_d48_bhsd(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BHSD),
]:
    return _key_cache_for_layer(layer_idx, kv_collection)


fn _key_cache_for_layer[
    kv_params: KVCacheStaticParams
](
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[DType.float32, kv_params],
) -> ContiguousKVCache[DType.float32, kv_params]:
    """Retrieves the Key cache for the given layer."""
    return kv_collection.get_key_cache(int(layer_idx))


@mogg_register("value_cache_for_layer_h6_d48_bshd")
@export
fn value_cache_for_layer_h6_d48_bshd(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BSHD),
]:
    return _value_cache_for_layer(layer_idx, kv_collection)


@mogg_register("value_cache_for_layer_h6_d48_bhsd")
@export
fn value_cache_for_layer_h6_d48_bhsd(
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BHSD),
]:
    return _value_cache_for_layer(layer_idx, kv_collection)


fn _value_cache_for_layer[
    kv_params: KVCacheStaticParams,
](
    layer_idx: Int64,
    kv_collection: ContiguousKVCacheCollection[DType.float32, kv_params],
) -> ContiguousKVCache[DType.float32, kv_params]:
    """Retrieves the Value cache for the given layer."""
    return kv_collection.get_value_cache(int(layer_idx))


@mogg_register("matmul_kv_cache_h6_d48_bshd")
@export
fn matmul_kv_cache_h6_d48_bshd[
    hidden_state_shape: DimList,
    weight_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[DType.float32, 3, hidden_state_shape],
    weight: NDBuffer[DType.float32, 2, weight_shape],
    cache: ContiguousKVCache[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BSHD),
]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, max_seq_len, num_kv_heads, head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    return _matmul_kv_cache[target=target](hidden_state, weight, cache, ctx)


@mogg_register("matmul_kv_cache_h6_d48_bhsd")
@export
fn matmul_kv_cache_h6_d48_bhsd[
    hidden_state_shape: DimList,
    weight_shape: DimList,
    target: StringLiteral = "cpu",
](
    hidden_state: NDBuffer[DType.float32, 3, hidden_state_shape],
    weight: NDBuffer[DType.float32, 2, weight_shape],
    cache: ContiguousKVCache[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BHSD),
]:
    """Performs a matmul, writing the output into a mutable ContiguousKVCache object.

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        cache: The historical ContiguousKVCache, with logical shape:
            (batch_size, num_kv_heads, max_seq_len, head_size).
        ctx: The call context pointer, passed by the graph compiler.
    """
    return _matmul_kv_cache[target=target](hidden_state, weight, cache, ctx)


@always_inline
fn _matmul_kv_cache[
    hidden_state_shape: DimList,
    weight_shape: DimList,
    kv_params: KVCacheStaticParams,
    *,
    target: StringLiteral,
](
    hidden_state: NDBuffer[DType.float32, 3, hidden_state_shape],
    weight: NDBuffer[DType.float32, 2, weight_shape],
    cache: ContiguousKVCache[DType.float32, kv_params],
    context: MojoCallContextPtr = MojoCallContextPtr(),
) -> ContiguousKVCache[DType.float32, kv_params]:
    """Helper for performing matmul with custom ContiguousKVCache types.

    Parameters:
        hidden_state_shape: The static shapes for the hidden_state tensor
        weight_shape: The static shapes for the weight tensor
        kv_params: The static parameters for our KVCache object
        target: StringLiteral identifying the device target (cpu vs cuda)

    Args:
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size)
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size)
        cache: The ContiguousKVCache, with shape determined by `kv_params.layout`
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
        var head_and_dim = divmod(idx[1], kv_params.head_size)
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

    var c_ptr = UnsafePointer[Float32].alloc(
        BS * SEQ_LEN * N
    ) if target == "cpu" else UnsafePointer[Float32]()

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


@mogg_register("rope_kv_cache_h6_d48_bshd")
@export
fn rope_kv_cache_h6_d48_bshd[
    freqs_shape: DimList, target: StringLiteral = "cpu"
](
    cache: ContiguousKVCache[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    freqs: NDBuffer[DType.float32, 2, freqs_shape],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BSHD),
]:
    return _rope_kv_cache[target=target](cache, freqs, ctx)


@mogg_register("rope_kv_cache_h6_d48_bhsd")
@export
fn rope_kv_cache_h6_d48_bhsd[
    freqs_shape: DimList, target: StringLiteral = "cpu"
](
    cache: ContiguousKVCache[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    freqs: NDBuffer[DType.float32, 2, freqs_shape],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[
    DType.float32,
    KVCacheStaticParams(num_heads=6, head_size=48, layout=KVCacheLayout.BHSD),
]:
    return _rope_kv_cache[target=target](cache, freqs, ctx)


@always_inline
fn _rope_kv_cache[
    freqs_shape: DimList,
    kv_params: KVCacheStaticParams,
    target: StringLiteral,
](
    cache: ContiguousKVCache[DType.float32, kv_params],
    freqs: NDBuffer[DType.float32, 2, freqs_shape],
    ctx: MojoCallContextPtr,
) -> ContiguousKVCache[DType.float32, kv_params]:
    """Unfused kernel to apply rope embeddings to KV Cache objects

    Parameters:
        freqs_shape: The shape of freqs, (seq_len, head_dim)
        kv_params: The KVCache parameters, including num_kv_heads, head_size, and layout
        target: The compute target, could be cuda or cpu

    Arguments:
        cache: The KVCache object, with the layout of the underlying data determined by kv_params.layout
        freqs: The RoPE frequencies, with shape (seq_len, head_dim).
        ctx: The call context as passed by the graph compiler.
    """

    var valid_len = int(cache.get_valid_lengths()[0])

    @always_inline
    @parameter
    @__copy_capture(freqs, valid_len)
    fn rope_fn[width: Int, rank: Int](idx: StaticIntTuple[rank]):
        @parameter
        if width == 1:
            print("ROPE KERNEL CALLED WITH SINGLE VALUE, EXPECTED AT LEAST 2")
        else:
            var bs: Int
            var head_idx: Int
            var t_idx: Int
            var hd_idx: Int

            @parameter
            if kv_params.layout == KVCacheLayout.BSHD:
                bs = idx[0]
                t_idx = idx[1]
                head_idx = idx[2]
                hd_idx = idx[3]
            elif kv_params.layout == KVCacheLayout.BHSD:
                bs = idx[0]
                head_idx = idx[1]
                t_idx = idx[2]
                hd_idx = idx[3]
            else:
                constrained[False, "Unsupported KVCache Layout"]()
                return
            var t_cache_idx = t_idx + valid_len
            var val = cache.load[width=width](bs, head_idx, t_cache_idx, hd_idx)
            var val_cast = rebind[SIMD[DType.float32, width]](val)

            var x_c = val_cast.deinterleave()
            var x_re = x_c[0]
            var x_im = x_c[1]

            var f_idx = StaticIntTuple[2](t_idx, hd_idx)
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
        cache.batch_size, kv_params.num_heads, freqs.dim[0](), freqs.dim[1]()
    ) if kv_params.layout == KVCacheLayout.BHSD else StaticIntTuple[4](
        cache.batch_size, freqs.dim[0](), kv_params.num_heads, freqs.dim[1]()
    )
    _elementwise_impl[rope_fn, simd_width, 4, target=target](launch_shape, ctx)
    return cache


@mogg_register("flash_attention_kv_cache_h6_d48_bshd")
@export
fn flash_attention_kv_cache_h6_d48_bshd[
    q_shape: DimList,
    mask_shape: DimList,
    scale_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[DType.float32, 4, q_shape],
    k: ContiguousKVCache[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    v: ContiguousKVCache[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BSHD
        ),
    ],
    mask: NDBuffer[DType.float32, 2, mask_shape],
    scale: NDBuffer[DType.float32, 1, scale_shape],
    output: NDBuffer[DType.float32, 4, output_shape],
    context: MojoCallContextPtr,
):
    return _flash_attention_kv_cache[target=target](
        q, k, v, mask, scale, output, context
    )


@mogg_register("flash_attention_kv_cache_h6_d48_bhsd")
@export
fn flash_attention_kv_cache_h6_d48_bhsd[
    q_shape: DimList,
    mask_shape: DimList,
    scale_shape: DimList,
    output_shape: DimList,
    target: StringLiteral = "cpu",
](
    q: NDBuffer[DType.float32, 4, q_shape],
    k: ContiguousKVCache[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    v: ContiguousKVCache[
        DType.float32,
        KVCacheStaticParams(
            num_heads=6, head_size=48, layout=KVCacheLayout.BHSD
        ),
    ],
    mask: NDBuffer[DType.float32, 2, mask_shape],
    scale: NDBuffer[DType.float32, 1, scale_shape],
    output: NDBuffer[DType.float32, 4, output_shape],
    context: MojoCallContextPtr,
):
    return _flash_attention_kv_cache[target=target](
        q, k, v, mask, scale, output, context
    )


fn _flash_attention_kv_cache[
    mask_rank: Int,
    q_shape: DimList,
    mask_shape: DimList,
    scale_shape: DimList,
    output_shape: DimList,
    kv_params: KVCacheStaticParams,
    target: StringLiteral,
](
    q: NDBuffer[DType.float32, 4, q_shape],
    k: ContiguousKVCache[DType.float32, kv_params],
    v: ContiguousKVCache[DType.float32, kv_params],
    mask: NDBuffer[DType.float32, mask_rank, mask_shape],
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
            mask_rank,
            q_shape,
            mask_shape,
            scale_shape,
            output_shape,
        ](q, k, v, mask, scale, output, context)
    else:
        return _flash_attention_kv_cache_gpu[
            mask_rank,
            q_shape,
            mask_shape,
            scale_shape,
            output_shape,
        ](q, k, v, mask, scale, output, context)


fn _flash_attention_kv_cache_cpu[
    mask_rank: Int,
    q_shape: DimList,
    mask_shape: DimList,
    scale_shape: DimList,
    output_shape: DimList,
    kv_params: KVCacheStaticParams,
](
    q: NDBuffer[DType.float32, 4, q_shape],
    k: ContiguousKVCache[DType.float32, kv_params],
    v: ContiguousKVCache[DType.float32, kv_params],
    mask: NDBuffer[DType.float32, mask_rank, mask_shape],
    scale: NDBuffer[DType.float32, 1, scale_shape],
    output: NDBuffer[DType.float32, 4, output_shape],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    constrained[
        kv_params.layout == KVCacheLayout.BHSD,
        "CPU flash attention only supports BHSD layout",
    ]()

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
        @parameter
        if mask_rank == 4:
            return mask.load[width=width](
                rebind[StaticIntTuple[mask_rank]](idx)
            )
        else:
            return mask.load[width=width]((idx[2], idx[3]))

    var batch_size = q.dim[0]()
    var num_heads = q.dim[1]()
    var depth = q.dim[3]()
    var new_seq_len = q.dim[2]()
    var cache_seq_len = int(k.get_valid_lengths()[0])
    var seq_len = new_seq_len + cache_seq_len
    var fa_k_shape = StaticIntTuple[4](
        batch_size, kv_params.num_heads, seq_len, depth
    )
    var fa_v_shape = StaticIntTuple[4](
        batch_size, kv_params.num_heads, seq_len, depth
    )

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
    mask_rank: Int,
    q_shape: DimList,
    mask_shape: DimList,
    scale_shape: DimList,
    output_shape: DimList,
    kv_params: KVCacheStaticParams,
](
    q: NDBuffer[DType.float32, 4, q_shape],
    k: ContiguousKVCache[DType.float32, kv_params],
    v: ContiguousKVCache[DType.float32, kv_params],
    mask: NDBuffer[DType.float32, mask_rank, mask_shape],
    scale: NDBuffer[DType.float32, 1, scale_shape],
    output: NDBuffer[DType.float32, 4, output_shape],
    context: MojoCallContextPtr = MojoCallContextPtr(),
):
    constrained[
        kv_params.layout == KVCacheLayout.BSHD,
        "GPU Flash attention only supports the BSHD layout.",
    ]()

    alias wrapped_mask_rank = mask_rank if mask_rank == 4 else 3
    var mask_nd: NDBuffer[
        DType.float32,
        wrapped_mask_rank,
        DimList.create_unknown[wrapped_mask_rank](),
    ]

    @parameter
    if mask_rank == 2:
        mask_nd = NDBuffer[
            DType.float32,
            wrapped_mask_rank,
            DimList.create_unknown[wrapped_mask_rank](),
        ](
            mask.data,
            StaticIntTuple[wrapped_mask_rank](
                q.dim[0](), mask.dim[0](), mask.dim[1]()
            ),
        )
    else:
        mask_nd = rebind[
            NDBuffer[
                DType.float32,
                wrapped_mask_rank,
                DimList.create_unknown[wrapped_mask_rank](),
            ]
        ](mask)

    # GPU flash attention kernel gets the cache length from the k tensor shape
    # TODO remove this an instead pass in explicit KVCache lengths to the GPU kernel.
    var valid_length = int(k.get_valid_lengths()[0] + q.dim[1]())
    var k_shape = k.block.dynamic_shape
    var kv_shape = StaticIntTuple[4](
        k_shape[0], valid_length, k_shape[2], k_shape[3]
    )
    var k_nd = NDBuffer[DType.float32, 4, k.block_shape](k.block.data, kv_shape)
    var v_nd = NDBuffer[DType.float32, 4, v.block_shape](v.block.data, kv_shape)

    try:
        gpu_flash_attention[
            4,
            wrapped_mask_rank,
            q.shape,
            k_nd.shape,
            v_nd.shape,
            mask_nd.shape,
            output.shape,
            q.type,
            k.type,
            v.type,
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
            rsqrt(Float32(kv_params.head_size)),
            context,
        )
    except e:
        print("Error in GPU Flash Attention:", e)
