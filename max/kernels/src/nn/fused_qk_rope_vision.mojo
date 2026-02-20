# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

"""Fused Q/K RoPE kernel for vision models (no KV cache).

This kernel applies Rotary Position Embedding (RoPE) to both query and key
tensors in a single fused operation, optimized for BF16 on GPU.
"""

from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from algorithm.functional import elementwise
from gpu.host.info import is_cpu
from register import register_internal

from utils.index import IndexList


@always_inline
fn _rope[
    dtype: DType,
    freq_dtype: DType,
    width: Int,
](
    val: SIMD[dtype, width],
    cos: SIMD[freq_dtype, width],
    sin: SIMD[freq_dtype, width],
) -> SIMD[dtype, width]:
    """Apply RoPE rotation using complex multiplication.

    val is complex (interleaved real/imag), cos/sin are the rotation coefficients.
    Complex multiplication for rotation:
        out_re = x_re * cos - x_im * sin
        out_im = x_re * sin + x_im * cos
    """
    var x_complex = val.cast[freq_dtype]().deinterleave()
    var x_re = x_complex[0]
    var x_im = x_complex[1]

    # cos/sin are repeated [c0, c0, c1, c1], so deinterleaving gives [c0, c1] in both parts
    # We need the values corresponding to x_re (even indices) and x_im (odd indices)
    # Since they are repeated, the even and odd parts are identical.
    var cos_parts = cos.deinterleave()
    var sin_parts = sin.deinterleave()

    var cos_half = cos_parts[0]
    var sin_half = sin_parts[0]

    # Apply rotation
    var out_re = x_re * cos_half - x_im * sin_half
    var out_im = x_re * sin_half + x_im * cos_half

    return rebind[SIMD[dtype, width]](out_re.interleave(out_im).cast[dtype]())


@register_internal("mo.fused_qk_rope_vision")
fn fused_qk_rope_vision[
    target: StaticString,
](
    q_out: OutputTensor,
    k_out: OutputTensor,
    query: InputTensor[dtype = q_out.dtype, rank = q_out.rank],
    key: InputTensor[dtype = k_out.dtype, rank = k_out.rank],
    freqs_cos: InputTensor,
    freqs_sin: InputTensor,
    ctx: DeviceContextPtr,
) raises:
    """Fused Q/K RoPE for vision models.

    Applies RoPE to query and key tensors simultaneously.
    Input shapes:
        - query: [B, S, num_heads, head_dim]
        - key: [B, S, num_heads, head_dim]
        - freqs_cos: [S, head_dim]  (cos values, interleaved pairs)
        - freqs_sin: [S, head_dim]  (sin values, interleaved pairs)
    Output shapes:
        - q_out: same as query
        - k_out: same as key
    """

    # Extract dimensions
    var batch_size = query.dim_size(0)
    var seq_len = query.dim_size(1)
    var num_q_heads = query.dim_size(2)
    var head_dim = query.dim_size(3)
    var num_k_heads = key.dim_size(2)

    debug_assert(head_dim % 2 == 0, "head_dim must be even for RoPE pairs")

    @parameter
    @__copy_capture(query, freqs_cos, freqs_sin, q_out)
    @always_inline
    fn rope_fn_q[
        width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank],):
        comptime assert rank == 4, "Expected rank 4 for query tensor"

        @parameter
        if width == 1:
            return
        else:
            var s = idx[1]
            var v = idx[3]
            var cos_val = freqs_cos.load[width=width](IndexList[2](s, v))
            var sin_val = freqs_sin.load[width=width](IndexList[2](s, v))
            var val = query.load[width=width](idx)
            var res = _rope(
                val,
                cos_val.cast[query.dtype](),
                sin_val.cast[query.dtype](),
            )
            q_out.store[width=width](idx, res)

    @parameter
    @__copy_capture(key, freqs_cos, freqs_sin, k_out)
    @always_inline
    fn rope_fn_k[
        width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank],):
        comptime assert rank == 4, "Expected rank 4 for key tensor"

        @parameter
        if width == 1:
            return
        else:
            var s = idx[1]
            var v = idx[3]
            var cos_val = freqs_cos.load[width=width](IndexList[2](s, v))
            var sin_val = freqs_sin.load[width=width](IndexList[2](s, v))
            var val = key.load[width=width](idx)
            var res = _rope(
                val,
                cos_val.cast[key.dtype](),
                sin_val.cast[key.dtype](),
            )
            k_out.store[width=width](idx, res)

    var q_shape = IndexList[4](batch_size, seq_len, num_q_heads, head_dim)
    var k_shape = IndexList[4](batch_size, seq_len, num_k_heads, head_dim)

    # Keep SIMD width conservative to work with a wide range of head sizes.
    comptime kernel_simd_width = 2

    @parameter
    if is_cpu[target]():
        elementwise[
            func=rope_fn_q, simd_width=kernel_simd_width, target=target
        ](q_shape)
        elementwise[
            func=rope_fn_k, simd_width=kernel_simd_width, target=target
        ](k_shape)
    else:
        var dev_ctx = ctx.get_device_context()
        elementwise[
            func=rope_fn_q, simd_width=kernel_simd_width, target=target
        ](q_shape, dev_ctx)
        elementwise[
            func=rope_fn_k, simd_width=kernel_simd_width, target=target
        ](k_shape, dev_ctx)
