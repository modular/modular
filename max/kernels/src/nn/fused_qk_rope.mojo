# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from math import gcd
from sys.info import _current_target, simd_width_of

from algorithm.functional import elementwise
from complex import ComplexSIMD
from gpu.host import DeviceContext, get_gpu_target
from gpu.host.info import is_cpu
from kv_cache.types import KVCacheT, KVCollectionT
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout import IntTuple
from nn._ragged_utils import get_batch_from_row_offsets

from utils import IndexList


@always_inline
fn _rope[
    dtype: DType,
    freq_dtype: DType,
    width: Int,
](val: SIMD[dtype, width], freq: SIMD[freq_dtype, width]) -> SIMD[dtype, width]:
    x_re, x_im = val.cast[freq_dtype]().deinterleave()
    f_re, f_im = freq.deinterleave()
    var r = ComplexSIMD(x_re, x_im) * ComplexSIMD(f_re, f_im)
    return rebind[SIMD[dtype, width]](r.re.interleave(r.im).cast[dtype]())


# In GGUF, weights are organized as real, imag, real, imag, real, imag, …,
# while in safetensors, the data is stored as real, …, real, imag, …, imag.
# This function return the indices for the real and imaginary part.
@always_inline
fn get_safetensors_idx(head_dim_idx: Int, head_size: Int) -> (Int, Int):
    return (head_dim_idx // 2, head_dim_idx // 2 + head_size // 2)


@always_inline
fn get_identity_rope_coeff[width: Int, dtype: DType]() -> SIMD[dtype, width]:
    # Creates a SIMD vector with real parts set to 1 and imaginary parts to
    # 0, effectively making the RoPE transformation an identity operation.
    return rebind[SIMD[dtype, width]](
        SIMD[dtype, width // 2](1).interleave(SIMD[dtype, width // 2](0))
    )


@always_inline
fn rope_q_proj[
    dtype: DType,
    freq_dtype: DType,
    rank: Int,
    width: Int, //,
    *,
    interleaved: Bool,
](
    q_proj: LayoutTensor[dtype, **_],
    output: LayoutTensor[mut=True, dtype, **_],
    idx: IndexList[rank],
    freq_val: SIMD[freq_dtype, width],
    head_size: Int,
):
    constrained[q_proj.rank == rank]()
    constrained[output.rank == rank]()
    var indices = get_safetensors_idx(idx[rank - 1], head_size)
    var pos_re = idx
    var pos_im = idx
    pos_re[rank - 1] = indices[0]
    pos_im[rank - 1] = indices[1]
    alias width_2 = width // 2

    var val: SIMD[dtype, width]

    @parameter
    if interleaved:
        val = q_proj.load[width=width](idx)
    else:
        val = rebind[SIMD[dtype, width]](
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
    freq_dtype: DType, cache_t: KVCacheT, width: Int, //, *, interleaved: Bool
](
    k_cache: cache_t,
    b_idx: Int,
    h_idx: Int,
    s_idx: Int,
    d_idx: Int,
    freq_val: SIMD[freq_dtype, width],
    head_size: Int,
):
    h_re, h_im = get_safetensors_idx(d_idx, head_size)
    alias width_2 = width // 2
    alias cache_type = cache_t.dtype

    var val: SIMD[cache_type, width]

    @parameter
    if interleaved:
        val = k_cache.load[width=width](b_idx, h_idx, s_idx, d_idx)
    else:
        val = rebind[SIMD[cache_type, width]](
            k_cache.load[width=width_2](b_idx, h_idx, s_idx, h_re).interleave(
                k_cache.load[width=width_2](b_idx, h_idx, s_idx, h_im)
            )
        )

    var res = _rope(val, freq_val)

    @parameter
    if interleaved:
        k_cache.store(b_idx, h_idx, s_idx, d_idx, res)
    else:
        output_re, output_im = res.deinterleave()
        k_cache.store(b_idx, h_idx, s_idx, h_re, output_re)
        k_cache.store(b_idx, h_idx, s_idx, h_im, output_im)


@always_inline
fn fused_qk_rope[
    dtype: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    *,
    interleaved: Bool,
    target: StaticString,
](
    q_proj: LayoutTensor[dtype, **_],
    kv_collection: collection_t,
    freqs_cis: LayoutTensor[dtype, **_],
    layer_idx: UInt32,
    output: LayoutTensor[mut=True, dtype, **_],
    context: Optional[DeviceContext],
) raises:
    constrained[q_proj.rank == 4]()
    constrained[freqs_cis.rank == 2]()
    constrained[output.rank == 4]()

    alias kv_params = cache_t.kv_params

    var batch_size = q_proj.dim[0]()
    var new_seq_len = q_proj.dim[1]()
    alias num_q_heads = Int(q_proj.layout.shape[2])
    alias num_k_heads = kv_params.num_heads
    alias head_size = Int(q_proj.layout.shape[3])

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    @always_inline
    @parameter
    @__copy_capture(k_cache)
    fn rope_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
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
    ]() else get_gpu_target()
    alias target_simd_width = simd_width_of[dtype, target=compile_target]()
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
    dtype: DType,
    freq_dtype: DType,
    collection_t: KVCollectionT, //,
    cache_t: KVCacheT,
    *,
    interleaved: Bool,
    target: StaticString,
    mrope_section: Optional[IntTuple] = None,
](
    q_proj: LayoutTensor[dtype, **_],
    input_row_offsets: LayoutTensor[DType.uint32, **_],
    kv_collection: collection_t,
    freqs_cis: LayoutTensor[freq_dtype, **_],
    position_ids: OptionalReg[
        LayoutTensor[DType.uint32, Layout.row_major[2](), MutableAnyOrigin]
    ],
    layer_idx: UInt32,
    output: LayoutTensor[mut=True, dtype, **_],
    context: Optional[DeviceContext],
) raises:
    """Applies RoPE (Rotary Position Embedding) to query and key tensors.

    This function can applies RoPE only to the last `rope_dim` elements of each
    head, leaving the first `unroped_dim` elements unchanged. This is required
    for DeepSeek models where only part of each head undergoes rotary
    transformation.
    """
    constrained[q_proj.rank == 3, "q_proj must be rank 3"]()
    constrained[freqs_cis.rank == 2, "freqs_cis must be rank 2"]()
    constrained[output.rank == 3, "output must be rank 3"]()
    constrained[
        input_row_offsets.rank == 1, "input_row_offsets must be rank 1"
    ]()
    alias kv_params = cache_t.kv_params
    alias num_q_heads = Int(q_proj.layout.shape[1])
    alias num_k_heads = kv_params.num_heads
    alias q_head_size = Int(q_proj.layout.shape[2])
    alias k_head_size = kv_params.head_size
    var batch_size = input_row_offsets.dim[0]() - 1

    # Add rope dimension parameters
    alias rope_dim = Int(freqs_cis.layout.shape[1])

    # Check if shape of freqs_cis matches head_size.
    # If not, we only rope the last `rope_dim` dimensions of each head.
    alias unroped_dim = q_head_size - rope_dim
    alias has_nope = unroped_dim > 0

    constrained[
        freqs_cis.layout.shape[1] != UNKNOWN_VALUE,
        "Need static shape for freqs_cis",
    ]()
    constrained[
        rope_dim <= q_head_size and rope_dim <= k_head_size,
        "rope_dim must be smaller or equal to head size, but got rope_dim = "
        + String(rope_dim)
        + " and head_size = "
        + String(k_head_size),
    ]()
    constrained[
        (rope_dim == q_head_size and rope_dim == k_head_size) or interleaved,
        "Partial RoPE operation only supported for interleaved pattern",
    ]()
    constrained[
        collection_t.dtype == dtype,
        "Expected cache dtype "
        + String(collection_t.dtype)
        + " to match input dtype "
        + String(dtype),
    ]()

    var k_cache = kv_collection.get_key_cache(Int(layer_idx))

    @always_inline
    @parameter
    @__copy_capture(k_cache, batch_size, input_row_offsets, position_ids)
    fn rope_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
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
            var head_idx = idx[1]
            var head_dim_idx = idx[2]

            # Use position_ids if provided, otherwise fall back to cache calculation
            var post_seq_idx = k_cache.cache_length(batch_idx) + token_idx

            var position_ids_idx = post_seq_idx
            if position_ids:

                @parameter
                if mrope_section:
                    var section_idx = 0

                    @parameter
                    for i in range(len(mrope_section.value())):
                        alias val = mrope_section.value().value(i)
                        if head_dim_idx < val:
                            section_idx = i
                            break
                    position_ids_idx = Int(
                        position_ids.value()[section_idx, global_token_idx]
                    )
                else:
                    position_ids_idx = Int(
                        position_ids.value()[0, global_token_idx]
                    )

            # WARN assumes head_size % simd_width == 0
            # guarded by constrained statement below
            var is_q_proj = head_idx < num_q_heads
            var is_unroped_region = head_dim_idx < unroped_dim

            var f_c_temp: SIMD[freq_dtype, width]

            @parameter
            if has_nope:
                if is_unroped_region:
                    f_c_temp = get_identity_rope_coeff[width, freq_dtype]()
                else:
                    var f_idx = IndexList[2](
                        position_ids_idx, head_dim_idx - unroped_dim
                    )
                    f_c_temp = freqs_cis.load[width=width](f_idx)
            else:
                var f_idx = IndexList[2](position_ids_idx, head_dim_idx)
                f_c_temp = freqs_cis.load[width=width](f_idx)

            if is_q_proj:
                rope_q_proj[interleaved=interleaved](
                    q_proj, output, idx, f_c_temp, q_head_size
                )
            else:

                @parameter
                if has_nope:
                    if is_unroped_region:
                        return

                head_idx -= num_q_heads
                # in case k_head_size != q_head_size
                head_dim_idx += k_head_size - UInt(q_head_size)
                rope_k_cache[interleaved=interleaved](
                    k_cache,
                    batch_idx,
                    head_idx,
                    post_seq_idx,
                    head_dim_idx,
                    f_c_temp,
                    k_head_size,
                )

    var launch_shape = IndexList[3](
        q_proj.dim[0](),
        num_q_heads + num_k_heads,  # concat q and k along head dim
        q_head_size,
    )
    alias compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    alias target_simd_width = simd_width_of[dtype, target=compile_target]()
    alias kernel_simd_width = gcd(target_simd_width, rope_dim)

    @parameter
    if mrope_section:

        @parameter
        for i in range(len(mrope_section.value())):
            constrained[
                Int(mrope_section.value()[i]) % kernel_simd_width == 0,
                "mrope_section must be divisible by rope kernel simd_width",
            ]()

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
