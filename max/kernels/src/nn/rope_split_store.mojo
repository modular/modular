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

"""Fused rope + split + KV store kernel.

Reads a flat QKV matmul output, applies RoPE to Q and K regions, stores
K/V to the paged KV cache, and writes roped Q to the output buffer — all
in a single GPU kernel to eliminate intermediate tensor round-trips.
"""

from std.algorithm.functional import elementwise
from std.collections import OptionalReg
from std.gpu.host import DeviceContext, get_gpu_target
from std.gpu.host.info import is_cpu, is_gpu
from std.math import gcd
from std.sys.info import _current_target, simd_width_of

from kv_cache.types import KVCacheT, PagedKVCacheCollection
from layout import TileTensor
from nn._ragged_utils import get_batch_from_row_offsets
from nn.fused_qk_rope import rope_value
from nn.rope import get_safetensors_idx
from std.runtime.asyncrt import DeviceContextPtr
from std.utils.index import IndexList


@always_inline
def _rope_split_store_ragged[
    dtype: DType,
    freq_dtype: DType,
    cache_t: KVCacheT,
    //,
    *,
    target: StaticString,
    interleaved: Bool = True,
](
    qkv: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[DType.uint32, ...],
    freqs_cis: TileTensor[freq_dtype, ...],
    k_cache: cache_t,
    v_cache: OptionalReg[cache_t],
    q_output: TileTensor[mut=True, dtype, ...],
    context: Optional[DeviceContext],
) raises:
    """Read flat QKV buffer, apply rope to Q and K, store K/V to cache.

    Args:
        qkv: Flat matmul output [total_seq_len, q_dim + k_dim + v_dim].
        input_row_offsets: [batch_size + 1] ragged offsets.
        freqs_cis: [max_seq_len, head_dim] interleaved rope frequencies.
        k_cache: Key cache to store roped K.
        v_cache: Value cache to store V.
        q_output: Output buffer for roped Q [total_seq_len, q_dim].
        context: DeviceContext for GPU.
    """
    comptime kv_params = cache_t.kv_params
    comptime kv_type = cache_t.dtype
    comptime head_size = Int(kv_params.head_size)
    comptime num_kv_heads = Int(kv_params.num_heads)

    comptime assert qkv.flat_rank == 2, "qkv must be rank 2"
    comptime assert q_output.flat_rank == 2, "q_output must be rank 2"
    comptime assert freqs_cis.flat_rank == 2, "freqs_cis must be rank 2"

    var q_dim = Int(q_output.dim[1]())
    var k_dim = head_size * num_kv_heads
    var total_seq_len = Int(qkv.dim[0]())
    var batch_size = Int(input_row_offsets.dim[0]()) - 1

    if batch_size == 0:
        return

    var freqs_ptr = freqs_cis.ptr
    var freqs_stride0 = head_size
    var qkv_ptr = qkv.ptr
    var q_out_ptr = q_output.ptr

    var combined_dim = Int(qkv.dim[1]())
    var qk_offset = q_dim + k_dim

    @parameter
    @__copy_capture(
        q_dim,
        k_dim,
        qk_offset,
        combined_dim,
        batch_size,
        freqs_ptr,
        freqs_stride0,
        qkv_ptr,
        q_out_ptr,
        k_cache,
        v_cache,
        input_row_offsets,
    )
    def rope_split_store_fn[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
        comptime assert rank == 2
        var idx = rebind[IndexList[2]](idx_arg)
        var global_token_idx = idx[0]
        var col = idx[1]

        # Guard: simd_width >= 2 is guaranteed by the comptime assert after
        # this function. We wrap in comptime if because elementwise
        # instantiates at all simd widths and width_2 = simd_width // 2 = 0
        # causes rebind errors.
        comptime if simd_width >= 2:
            # Hoist batch index lookup (binary search) before Q/K/V branch
            # so each thread does it once instead of per-region.
            var bi: Int = get_batch_from_row_offsets(
                input_row_offsets, global_token_idx
            )
            var ti = Int(UInt32(global_token_idx) - input_row_offsets.ptr[bi])

            if col < q_dim:
                # Q region: apply rope, write to q_output.
                var hdi = col % head_size
                var pos = k_cache.cache_length(bi) + ti

                # Base offset of this element in the flat buffers.
                var qkv_base = global_token_idx * combined_dim + col
                var q_base = global_token_idx * q_dim + col

                comptime if interleaved:
                    var val = (qkv_ptr + qkv_base).load[width=simd_width]()
                    var freq = (freqs_ptr + pos * freqs_stride0 + hdi).load[
                        width=simd_width
                    ]()
                    (q_out_ptr + q_base).store(rope_value(val, freq))
                else:
                    # Non-interleaved: gather re/im halves, rope, scatter.
                    comptime width_2 = simd_width // 2
                    var head_start_qkv = qkv_base - hdi
                    var head_start_q = q_base - hdi
                    var re_idx, im_idx = get_safetensors_idx(hdi, head_size)
                    var val_re = (qkv_ptr + head_start_qkv + re_idx).load[
                        width=width_2
                    ]()
                    var val_im = (qkv_ptr + head_start_qkv + im_idx).load[
                        width=width_2
                    ]()
                    var val = rebind[SIMD[dtype, simd_width]](
                        val_re.interleave(val_im)
                    )
                    var freq = (freqs_ptr + pos * freqs_stride0 + hdi).load[
                        width=simd_width
                    ]()
                    var res = rope_value(val, freq)
                    var res_re: SIMD[dtype, width_2]
                    var res_im: SIMD[dtype, width_2]
                    res_re, res_im = res.deinterleave()
                    (q_out_ptr + head_start_q + re_idx).store(res_re)
                    (q_out_ptr + head_start_q + im_idx).store(res_im)
                return

            if col < qk_offset:
                # K region: apply rope, store to k_cache.
                var kv_col = col - q_dim
                var hi, di = divmod(UInt(kv_col), kv_params.head_size)
                var cl = k_cache.cache_length(bi)
                var pos = cl + ti

                comptime if interleaved:
                    var qkv_base = global_token_idx * combined_dim + col
                    var val = (qkv_ptr + qkv_base).load[width=simd_width]()
                    var freq = (freqs_ptr + pos * freqs_stride0 + Int(di)).load[
                        width=simd_width
                    ]()
                    k_cache.store(
                        bi,
                        Int(hi),
                        pos,
                        Int(di),
                        rebind[SIMD[kv_type, simd_width]](
                            rope_value(val, freq)
                        ),
                    )
                else:
                    # Non-interleaved K: gather re/im, rope, deinterleave, store.
                    comptime width_2 = simd_width // 2
                    var k_head_base = (
                        global_token_idx * combined_dim
                        + q_dim
                        + Int(hi) * head_size
                    )
                    var re_idx, im_idx = get_safetensors_idx(Int(di), head_size)
                    var val_re = (qkv_ptr + k_head_base + re_idx).load[
                        width=width_2
                    ]()
                    var val_im = (qkv_ptr + k_head_base + im_idx).load[
                        width=width_2
                    ]()
                    var val = rebind[SIMD[dtype, simd_width]](
                        val_re.interleave(val_im)
                    )
                    var freq = (freqs_ptr + pos * freqs_stride0 + Int(di)).load[
                        width=simd_width
                    ]()
                    var roped = rope_value(val, freq)
                    var roped_re: SIMD[dtype, width_2]
                    var roped_im: SIMD[dtype, width_2]
                    roped_re, roped_im = roped.deinterleave()
                    k_cache.store(
                        bi,
                        Int(hi),
                        pos,
                        re_idx,
                        rebind[SIMD[kv_type, width_2]](roped_re),
                    )
                    k_cache.store(
                        bi,
                        Int(hi),
                        pos,
                        im_idx,
                        rebind[SIMD[kv_type, width_2]](roped_im),
                    )
                return

            # V region: store directly to v_cache (no rope).
            var qkv_base = global_token_idx * combined_dim + col
            var val = (qkv_ptr + qkv_base).load[width=simd_width]()
            var v_col = col - qk_offset
            var hi, di = divmod(UInt(v_col), kv_params.head_size)
            var cl = v_cache.value().cache_length(bi)
            v_cache.value().store(
                bi,
                Int(hi),
                ti + cl,
                Int(di),
                rebind[SIMD[kv_type, simd_width]](val),
            )

    var launch_shape = IndexList[2](total_seq_len, combined_dim)
    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[dtype, target=compile_target]()
    comptime kernel_simd_width = gcd(target_simd_width, head_size)
    comptime assert (
        kernel_simd_width >= 2
    ), "rope_split_store requires simd_width >= 2"
    comptime assert (
        head_size % kernel_simd_width == 0
    ), "head_size must be divisible by simd_width"

    comptime if is_cpu[target]():
        elementwise[
            func=rope_split_store_fn,
            simd_width=kernel_simd_width,
            target=target,
        ](launch_shape)
    else:
        elementwise[
            func=rope_split_store_fn,
            simd_width=kernel_simd_width,
            target=target,
        ](launch_shape, context.value())


@always_inline
def rope_split_store_paged_ragged[
    dtype: DType,
    freq_dtype: DType,
    target: StaticString = "cpu",
    interleaved: Bool = True,
](
    qkv: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[DType.uint32, ...],
    freqs_cis: TileTensor[freq_dtype, ...],
    kv_collection: PagedKVCacheCollection,
    layer_idx: UInt32,
    q_output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContextPtr,
) raises:
    """Rope+split+store with paged KV cache collection."""
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache: OptionalReg[type_of(k_cache)] = kv_collection.get_value_cache(
        layer_idx_cast
    )

    comptime if is_gpu[target]():
        cuda_ctx = ctx.get_device_context()

    return _rope_split_store_ragged[
        target=target,
        interleaved=interleaved,
    ](qkv, input_row_offsets, freqs_cis, k_cache, v_cache, q_output, cuda_ctx)
