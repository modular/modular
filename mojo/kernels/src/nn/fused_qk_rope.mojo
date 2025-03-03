# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import Optional
from math import gcd
from sys.info import _current_target, simdwidthof

from algorithm.functional import elementwise
from buffer import NDBuffer
from complex import ComplexSIMD
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.host.info import is_cpu
from kv_cache.types import KVCacheT, KVCollectionT
from nn._ragged_utils import get_batch_from_row_offsets

from utils import IndexList, StaticTuple


@always_inline
fn _rope(val: SIMD, freq: __type_of(val)) -> __type_of(val):
    x_re, x_im = val.deinterleave()
    f_re, f_im = freq.deinterleave()
    var r = ComplexSIMD(x_re, x_im) * ComplexSIMD(f_re, f_im)
    return rebind[__type_of(val)](r.re.interleave(r.im))


# In GGUF, weights are organized as real, imag, real, imag, real, imag, …,
# while in safetensors, the data is stored as real, …, real, imag, …, imag.
# This function return the indices for the real and imaginary part.
@always_inline
fn get_safetensors_idx(head_dim_idx: Int, head_size: Int) -> (Int, Int):
    return (head_dim_idx // 2, head_dim_idx // 2 + head_size // 2)


@always_inline
fn rope_q_proj[
    type: DType, rank: Int, width: Int, //, *, interleaved: Bool
](
    q_proj: NDBuffer[type, rank, *_],
    output: NDBuffer[type, rank, *_],
    idx: IndexList[rank],
    freq_val: SIMD[type, width],
    head_size: Int,
):
    var indices = get_safetensors_idx(idx[rank - 1], head_size)
    var pos_re = idx
    var pos_im = idx
    pos_re[rank - 1] = indices[0]
    pos_im[rank - 1] = indices[1]
    alias width_2 = width // 2

    var val: SIMD[type, width]

    @parameter
    if interleaved:
        val = q_proj.load[width=width](idx)
    else:
        val = rebind[SIMD[type, width]](
            q_proj.load[width=width_2](pos_re).interleave(
                q_proj.load[width=width_2](pos_im)
            )
        )

    var res = _rope(val, freq_val)

    @parameter
    if interleaved:
        output.store(idx, res)
    else:
        output_re, output_im = res.deinterleave()
        output.store(pos_re, output_re)
        output.store(pos_im, output_im)


@always_inline
fn rope_k_cache[
    type: DType, cache_t: KVCacheT, width: Int, //, *, interleaved: Bool
](
    k_cache: cache_t,
    b_idx: Int,
    h_idx: Int,
    s_idx: Int,
    d_idx: Int,
    freq_val: SIMD[type, width],
    head_size: Int,
):
    h_re, h_im = get_safetensors_idx(d_idx, head_size)
    alias width_2 = width // 2
    alias cache_type = cache_t.type

    constrained[
        cache_type == type,
        String(
            "Expected cache type ", cache_type, " to match input type ", type
        ),
    ]()

    var val: SIMD[type, width]

    @parameter
    if interleaved:
        val = rebind[SIMD[type, width]](
            k_cache.load[width=width](b_idx, h_idx, s_idx, d_idx)
        )
    else:
        val = rebind[SIMD[type, width]](
            k_cache.load[width=width_2](b_idx, h_idx, s_idx, h_re).interleave(
                k_cache.load[width=width_2](b_idx, h_idx, s_idx, h_im)
            )
        )

    var res = _rope(val, freq_val)

    @parameter
    if interleaved:
        k_cache.store(
            b_idx, h_idx, s_idx, d_idx, rebind[SIMD[cache_type, width]](res)
        )
    else:
        output_re, output_im = res.deinterleave()
        k_cache.store(
            b_idx,
            h_idx,
            s_idx,
            h_re,
            rebind[SIMD[cache_type, width_2]](output_re),
        )
        k_cache.store(
            b_idx,
            h_idx,
            s_idx,
            h_im,
            rebind[SIMD[cache_type, width_2]](output_im),
        )


@always_inline
fn fused_qk_rope[
    type: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    *,
    interleaved: Bool,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    kv_collection: collection_t,
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 4, *_],
    context: Optional[DeviceContext],
) raises:
    alias kv_params = cache_t.kv_params

    var batch_size = q_proj.dim[0]()
    var new_seq_len = q_proj.dim[1]()
    alias num_q_heads = q_proj.shape.get[2]()
    alias num_k_heads = kv_params.num_heads
    alias head_size = q_proj.shape.get[3]()

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    @always_inline
    @parameter
    @__copy_capture(k_cache)
    fn rope_fn[width: Int, rank: Int](idx_arg: IndexList[rank]):
        constrained[rank == 4, "Invalid rank passed to rope kernel"]()

        @parameter
        if width == 1:
            return
        else:
            var idx = rebind[IndexList[4]](idx_arg)
            var bs_idx = idx[0]
            # post_seq_idx: sum of start_pos (cache_lengths[batch_idx]) and
            # seq_idx (idx[1]).
            var post_seq_idx = k_cache.cache_length(bs_idx) + idx[1]
            var head_idx = idx[2]
            var head_dim_idx = idx[3]

            # WARN assumes head_size % simd_width == 0
            # guarded by constrained statement below
            var is_q_proj = head_idx < num_q_heads
            var f_idx = IndexList[2](post_seq_idx, head_dim_idx)
            var f_c_temp = freqs_cis.load[width=width](f_idx)

            if is_q_proj:
                rope_q_proj[interleaved=interleaved](
                    q_proj, output, idx, f_c_temp, head_size
                )
            else:
                head_idx -= num_q_heads
                rope_k_cache[interleaved=interleaved](
                    k_cache,
                    bs_idx,
                    head_idx,
                    post_seq_idx,
                    head_dim_idx,
                    f_c_temp,
                    head_size,
                )

    var launch_shape = IndexList[4](
        batch_size,
        new_seq_len,
        num_q_heads + num_k_heads,  # concat q and k along head dim
        head_size,
    )
    alias compile_target = _current_target() if is_cpu[
        target
    ]() else _get_gpu_target()
    alias target_simd_width = simdwidthof[type, target=compile_target]()
    alias kernel_simd_width = gcd(target_simd_width, kv_params.head_size)
    constrained[kernel_simd_width >= 2, "invalid simd_width and head size"]()

    @parameter
    if is_cpu[target]():
        elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
            launch_shape
        )
    else:
        elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
            launch_shape, context.value()
        )


@always_inline
fn fused_qk_rope_ragged[
    type: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    *,
    interleaved: Bool,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 3, *_],
    input_row_offsets: NDBuffer[DType.uint32, 1, *_],
    kv_collection: collection_t,
    freqs_cis: NDBuffer[type, 2, *_],
    layer_idx: UInt32,
    output: NDBuffer[type, 3, *_],
    context: Optional[DeviceContext],
) raises:
    alias kv_params = cache_t.kv_params
    alias num_q_heads = q_proj.shape.get[1]()
    alias num_k_heads = kv_params.num_heads
    alias head_size = q_proj.shape.get[2]()
    var batch_size = input_row_offsets.dim[0]() - 1

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    @always_inline
    @parameter
    @__copy_capture(k_cache, batch_size, input_row_offsets)
    fn rope_fn[width: Int, rank: Int](idx_arg: IndexList[rank]):
        constrained[rank == 3, "Invalid rank passed to rope kernel"]()

        @parameter
        if width == 1:
            return
        else:
            var idx = rebind[IndexList[3]](idx_arg)

            var global_token_idx = idx[0]

            var batch_idx: Int = get_batch_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var token_idx = Int(global_token_idx - input_row_offsets[batch_idx])

            var post_seq_idx = k_cache.cache_length(batch_idx) + token_idx
            var head_idx = idx[1]
            var head_dim_idx = idx[2]

            # WARN assumes head_size % simd_width == 0
            # guarded by constrained statement below
            var is_q_proj = head_idx < num_q_heads
            var f_idx = IndexList[2](post_seq_idx, head_dim_idx)
            var f_c_temp = freqs_cis.load[width=width](f_idx)

            if is_q_proj:
                rope_q_proj[interleaved=interleaved](
                    q_proj, output, idx, f_c_temp, head_size
                )
            else:
                head_idx -= num_q_heads
                rope_k_cache[interleaved=interleaved](
                    k_cache,
                    batch_idx,
                    head_idx,
                    post_seq_idx,
                    head_dim_idx,
                    f_c_temp,
                    head_size,
                )

    var launch_shape = IndexList[3](
        q_proj.dim[0](),
        num_q_heads + num_k_heads,  # concat q and k along head dim
        head_size,
    )
    alias compile_target = _current_target() if is_cpu[
        target
    ]() else _get_gpu_target()
    alias target_simd_width = simdwidthof[type, target=compile_target]()
    alias kernel_simd_width = gcd(target_simd_width, kv_params.head_size)
    constrained[kernel_simd_width >= 2, "invalid simd_width and head size"]()

    @parameter
    if is_cpu[target]():
        elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
            launch_shape
        )
    else:
        elementwise[func=rope_fn, simd_width=kernel_simd_width, target=target](
            launch_shape, context.value()
        )
