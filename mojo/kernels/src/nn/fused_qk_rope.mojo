# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import Optional
from math import gcd
from sys.info import _current_target, simdwidthof
from sys.intrinsics import _type_is_eq

from algorithm.functional import elementwise
from buffer import NDBuffer
from gpu.host import DeviceContext
from gpu.host._compile import _get_nvptx_target
from kv_cache.types import (
    ContiguousKVCache,
    ContinuousBatchingKVCache,
    KVCacheT,
)
from runtime.asyncrt import (
    MojoCallContextPtr,
)
from utils import IndexList


@always_inline
fn fused_qk_rope[
    type: DType,
    cache_t: KVCacheT, //,
    *,
    target: StringLiteral,
](
    q_proj: NDBuffer[type, 4, *_],
    k_cache: cache_t,
    freqs_cis: NDBuffer[type, 2, *_],
    output: NDBuffer[type, 4, *_],
    context: Optional[DeviceContext],
):
    alias kv_params = cache_t.get_kv_params()

    var batch_size = q_proj.dim[0]()
    var new_seq_len = q_proj.dim[1]()
    alias num_q_heads = q_proj.shape.get[2]()
    alias num_k_heads = kv_params.num_heads
    alias head_size = q_proj.shape.get[3]()

    @always_inline
    @parameter
    @__copy_capture(freqs_cis, q_proj, output)
    fn rope_fn_common[
        rank: Int,
        cache_t_: KVCacheT, //,
        width: Int,
    ](k_cache: cache_t_, idx_arg: IndexList[rank]):
        constrained[rank == 4, "Invalid rank passed to rope kernel"]()

        @parameter
        if width == 1:
            print("ROPE KERNEL CALLED WITH SINGLE VALUE, EXPECTED AT LEAST 2")
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
            var val: SIMD[type, width]

            if is_q_proj:
                val = q_proj.load[width=width](idx)
            else:
                head_idx -= num_q_heads

                val = k_cache.load[type, width=width](
                    bs_idx, head_idx, post_seq_idx, head_dim_idx
                )

            var x_c = val.deinterleave()
            var x_re = x_c[0]
            var x_im = x_c[1]

            var f_idx = IndexList[2](post_seq_idx, head_dim_idx)
            var f_c_temp = freqs_cis.load[width=width](f_idx)

            var f_c = f_c_temp.deinterleave()
            var f_re = f_c[0]
            var f_im = f_c[1]

            var r_re = (x_re * f_re) - (x_im * f_im)
            var r_im = (x_re * f_im) + (x_im * f_re)

            var result = r_re.interleave(r_im)
            if is_q_proj:
                output.store(idx, result)
            else:
                k_cache.store(
                    bs_idx, head_idx, post_seq_idx, head_dim_idx, result
                )

    alias compile_target = _current_target() if target == "cpu" else _get_nvptx_target()
    alias target_simd_width = simdwidthof[type, target=compile_target]()
    alias kernel_simd_width = gcd(target_simd_width, kv_params.head_size)
    constrained[kernel_simd_width >= 2, "invalid simd_width and head size"]()

    var launch_shape = IndexList[4](
        batch_size,
        new_seq_len,
        num_q_heads + num_k_heads,  # concat q and k along head dim
        head_size,
    )

    # TODO this is necessary due to traits not having a notion of being register_passable
    # remove this forking after MOCO-1205 (or after we get rid of mo.opaque)
    @parameter
    if _type_is_eq[cache_t, ContiguousKVCache[type, kv_params]]():
        # cast to a register passable type so the function closure works on GPU
        var k_cache_reg = rebind[ContiguousKVCache[type, kv_params]](k_cache)

        @parameter
        @__copy_capture(k_cache_reg)
        fn rope_fn_contig[
            width: Int,
            rank: Int,
        ](idx: IndexList[rank]):
            rope_fn_common[width](k_cache_reg, idx)

        @parameter
        if target == "cpu":
            elementwise[
                func=rope_fn_contig, simd_width=kernel_simd_width, target=target
            ](launch_shape)
        else:
            elementwise[
                func=rope_fn_contig, simd_width=kernel_simd_width, target=target
            ](launch_shape, context.value())
    elif _type_is_eq[cache_t, ContinuousBatchingKVCache[type, kv_params]]():
        # cast to a register passable type so the function closure works on GPU
        var k_cache_reg = rebind[ContinuousBatchingKVCache[type, kv_params]](
            k_cache
        )

        @parameter
        @__copy_capture(k_cache_reg)
        fn rope_fn_continuous[
            width: Int,
            rank: Int,
        ](idx: IndexList[rank]):
            rope_fn_common[width](k_cache_reg, idx)

        @parameter
        if target == "cpu":
            elementwise[
                func=rope_fn_continuous,
                simd_width=kernel_simd_width,
                target=target,
            ](launch_shape)
        else:
            elementwise[
                func=rope_fn_continuous,
                simd_width=kernel_simd_width,
                target=target,
            ](launch_shape, context.value())
