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
from std.utils.numerics import max_finite

from kv_cache.types import KVCacheT, PagedKVCacheCollection
from layout import (
    Coord,
    CoordLike,
    Idx,
    RowMajorLayout,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
)
from nn._ragged_utils import get_batch_from_row_offsets
from nn.fused_qk_rope import rope_value
from nn.rope import get_safetensors_idx
from std.utils.index import IndexList


# ===-----------------------------------------------------------------------===#
# Core kernel
# ===-----------------------------------------------------------------------===#


@always_inline
def _rope_split_store_ragged_impl[
    dtype: DType,
    freq_dtype: DType,
    cache_t: KVCacheT,
    //,
    *,
    target: StaticString,
    interleaved: Bool = True,
    get_freq_pos: def(Int, Int, Int) capturing -> Int,
](
    qkv: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[DType.uint32, ...],
    freqs_cis: TileTensor[freq_dtype, ...],
    k_cache: cache_t,
    v_cache: OptionalReg[cache_t],
    q_output: TileTensor[mut=True, dtype, ...],
    context: Optional[DeviceContext],
) raises:
    """Read flat QKV buffer, apply RoPE to Q and K, store K/V to cache.

    The ``get_freq_pos`` closure resolves a head-dimension index and
    token position to the row of ``freqs_cis`` that supplies the RoPE
    coefficients.  Callers swap in different closures to get
    cache-derived positions vs. explicit position-ID lookups.

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
    comptime head_size = kv_params.head_size
    comptime num_kv_heads = kv_params.num_heads

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

    # When `kv_type` is an FP8 type, dispatch to the fp8 quantize-and-store
    # body further down. Keeping the bf16 path in its own comptime branch
    # ensures the existing (non-quantized) flow is byte-identical to
    # pre-fp8 behaviour: the rebind[SIMD[kv_type, simd_width]] only sees
    # `dtype == kv_type` here, and the body emitted is unchanged.
    #
    # SM100 (B200) target. NVFP4 / FP8 e4m3fn. The NVIDIA SM100 path here
    # uses `x.cast[DType.float8_e4m3fn]()` directly because the write path
    # is HBM-bound, not conversion-rate bound — the packed
    # `cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4` is wired into the
    # MLA fp8 decode kernel for the read path only.
    comptime if not kv_type.is_float8():

        @parameter
        @__copy_capture(
            q_dim,
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
            simd_width: Int, alignment: Int = 1
        ](idx_arg: Coord):
            comptime assert idx_arg.rank == 2
            var idx = rebind[IndexList[2]](coord_to_index_list(idx_arg))
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
                var ti = Int(
                    UInt32(global_token_idx) - input_row_offsets.raw_load(bi)
                )

                # Cache position: used for cache stores and as the default
                # freq_pos when the caller doesn't supply explicit position IDs.
                var cache_pos = k_cache.cache_length(bi) + ti

                if col < q_dim:
                    # Q region: apply rope, write to q_output.
                    var hdi = col % head_size
                    var freq_pos = get_freq_pos(
                        hdi, global_token_idx, cache_pos
                    )

                    var qkv_base = global_token_idx * combined_dim + col
                    var q_base = global_token_idx * q_dim + col

                    comptime if interleaved:
                        var val = (qkv_ptr + qkv_base).load[width=simd_width]()
                        var freq = (
                            freqs_ptr + freq_pos * freqs_stride0 + hdi
                        ).load[width=simd_width]()
                        (q_out_ptr + q_base).store(rope_value(val, freq))
                    else:
                        # Non-interleaved: gather re/im halves, rope, scatter.
                        comptime width_2 = simd_width / 2
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
                        var freq = (
                            freqs_ptr + freq_pos * freqs_stride0 + hdi
                        ).load[width=simd_width]()
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
                    var hi, di = divmod(UInt(kv_col), UInt(kv_params.head_size))
                    var freq_pos = get_freq_pos(
                        Int(di), global_token_idx, cache_pos
                    )

                    comptime if interleaved:
                        var qkv_base = global_token_idx * combined_dim + col
                        var val = (qkv_ptr + qkv_base).load[width=simd_width]()
                        var freq = (
                            freqs_ptr + freq_pos * freqs_stride0 + Int(di)
                        ).load[width=simd_width]()
                        k_cache.store(
                            bi,
                            Int(hi),
                            cache_pos,
                            Int(di),
                            rebind[SIMD[kv_type, simd_width]](
                                rope_value(val, freq)
                            ),
                        )
                    else:
                        # Non-interleaved K: gather re/im, rope, deinterleave,
                        # store.
                        comptime width_2 = simd_width / 2
                        var k_head_base = (
                            global_token_idx * combined_dim
                            + q_dim
                            + Int(hi) * head_size
                        )
                        var re_idx, im_idx = get_safetensors_idx(
                            Int(di), head_size
                        )
                        var val_re = (qkv_ptr + k_head_base + re_idx).load[
                            width=width_2
                        ]()
                        var val_im = (qkv_ptr + k_head_base + im_idx).load[
                            width=width_2
                        ]()
                        var val = rebind[SIMD[dtype, simd_width]](
                            val_re.interleave(val_im)
                        )
                        var freq = (
                            freqs_ptr + freq_pos * freqs_stride0 + Int(di)
                        ).load[width=simd_width]()
                        var roped = rope_value(val, freq)
                        var roped_re: SIMD[dtype, width_2]
                        var roped_im: SIMD[dtype, width_2]
                        roped_re, roped_im = roped.deinterleave()
                        k_cache.store(
                            bi,
                            Int(hi),
                            cache_pos,
                            re_idx,
                            rebind[SIMD[kv_type, width_2]](roped_re),
                        )
                        k_cache.store(
                            bi,
                            Int(hi),
                            cache_pos,
                            im_idx,
                            rebind[SIMD[kv_type, width_2]](roped_im),
                        )
                    return

                # V region: store directly to v_cache (no rope).
                var qkv_base = global_token_idx * combined_dim + col
                var val = (qkv_ptr + qkv_base).load[width=simd_width]()
                var v_col = col - qk_offset
                var hi, di = divmod(UInt(v_col), UInt(kv_params.head_size))
                var cl = v_cache.value().cache_length(bi)
                v_cache.value().store(
                    bi,
                    Int(hi),
                    ti + cl,
                    Int(di),
                    rebind[SIMD[kv_type, simd_width]](val),
                )

        var launch_shape = (total_seq_len, combined_dim)
        comptime compile_target = _current_target() if is_cpu[
            target
        ]() else get_gpu_target()
        comptime target_simd_width = simd_width_of[
            dtype, target=compile_target
        ]()
        comptime kernel_simd_width = gcd(target_simd_width, head_size)
        comptime assert (
            kernel_simd_width >= 2
        ), "rope_split_store requires simd_width >= 2"
        comptime assert (
            head_size % kernel_simd_width == 0
        ), "head_size must be divisible by simd_width"

        var device_ctx = context.value() if context else DeviceContext(
            api="cpu"
        )
        elementwise[
            func=rope_split_store_fn,
            simd_width=kernel_simd_width,
            target=target,
        ](launch_shape, device_ctx)
    else:
        # ============================================================
        # FP8 quantize-and-store path.
        # ============================================================
        # When the KV cache dtype is FP8 (e4m3fn), each thread processes
        # one full quantization block (size = `quantization_granularity`)
        # along head_dim, computes the per-block max-abs in registers,
        # derives an fp32 scale, quantizes the bf16 SIMD lane, and stores
        # both the fp8 values and the fp32 scale via the KV cache's
        # `store` / `store_scale` APIs.
        #
        # We launch a separate `elementwise` over the K and V regions
        # only (the Q region is dispatched in its own launch using the
        # bf16-friendly simd width, since Q is never quantized).
        #
        # Conversion pattern adapted from MLA fp8 decode (which does
        # bf16→fp8 via `cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4`);
        # on the store path we use the per-lane SIMD cast
        # (`x.cast[fp8]()`) — this path is HBM-write bound, not
        # conversion-rate bound, so the packed intrinsic is unnecessary
        # here.
        comptime kv_params_const = cache_t.kv_params
        comptime cache_scale_dtype = cache_t.scale_dtype
        comptime quant_g = cache_t.quantization_granularity
        comptime assert (
            kv_type == DType.float8_e4m3fn
        ), "FP8 KV path supports float8_e4m3fn only."
        comptime assert (
            cache_scale_dtype == DType.float32
        ), "FP8 KV path expects fp32 scales."
        comptime assert quant_g > 1 and head_size % quant_g == 0, (
            "FP8 KV path requires quantization_granularity > 1 and"
            " head_size divisible by it"
        )
        # NOTE on `interleaved`:
        #
        # `interleaved` controls how `rope_value` PAIRS dimensions for
        # rotation: when True it rotates `(2k, 2k+1)`; when False it
        # rotates `(k, k+head_size/2)` (the HuggingFace `rotate_half`
        # convention all Gemma checkpoints are trained against).  This
        # is a MODEL-WEIGHT semantic, not a storage option.
        #
        # In the fp8 path we apply the user-selected rotation (correct
        # HF math) and STORE the rope output as a contiguous block at
        # `[di, di+g)` in BOTH Q and K (i.e. write the interleaved-
        # register-order result without deinterleaving).  The Q·K dot
        # product `Σ_d Q[d]·K[d]` is invariant to the head_dim
        # permutation as long as Q and K share it, so FA4 sums the same
        # scalar product.  Per-block scales align with the contiguous
        # storage block at `[di, di+g)`.  Non-interleaved RoPE math is
        # fully supported.

        # ---------- Q launch (bf16, unchanged math) ----------
        # We reuse the bf16 simd width for Q since Q is not quantized.
        comptime q_compile_target = _current_target() if is_cpu[
            target
        ]() else get_gpu_target()
        comptime q_target_simd_width = simd_width_of[
            dtype, target=q_compile_target
        ]()
        comptime q_kernel_simd_width = gcd(q_target_simd_width, head_size)
        comptime assert (
            q_kernel_simd_width >= 2
        ), "rope_split_store Q lane requires simd_width >= 2"

        @parameter
        @__copy_capture(
            q_dim,
            combined_dim,
            batch_size,
            freqs_ptr,
            freqs_stride0,
            qkv_ptr,
            q_out_ptr,
            input_row_offsets,
        )
        def rope_q_fn[simd_width: Int, alignment: Int = 1](idx_arg: Coord):
            comptime assert idx_arg.rank == 2
            var idx = rebind[IndexList[2]](coord_to_index_list(idx_arg))
            var global_token_idx = idx[0]
            var col = idx[1]
            comptime if simd_width >= 2:
                var bi: Int = get_batch_from_row_offsets(
                    input_row_offsets, global_token_idx
                )
                var cache_pos = k_cache.cache_length(bi) + Int(
                    UInt32(global_token_idx) - input_row_offsets.raw_load(bi)
                )
                var hdi = col % head_size
                var freq_pos = get_freq_pos(hdi, global_token_idx, cache_pos)
                var qkv_base = global_token_idx * combined_dim + col
                var q_base = global_token_idx * q_dim + col

                comptime if interleaved:
                    var val = (qkv_ptr + qkv_base).load[width=simd_width]()
                    var freq = (
                        freqs_ptr + freq_pos * freqs_stride0 + hdi
                    ).load[width=simd_width]()
                    (q_out_ptr + q_base).store(rope_value(val, freq))
                else:
                    # Non-interleaved: apply HF `rotate_half` rotation by
                    # pairing `(d, d + head_size/2)` (re_idx/im_idx
                    # loads), then store the result as a contiguous block
                    # at `[di, di+g)` (= `q_base`).  Q·K dot product is
                    # permutation-invariant; the fp8 K-store below uses
                    # the same contiguous layout.
                    comptime width_2 = simd_width / 2
                    var head_start_qkv = qkv_base - hdi
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
                    var freq = (
                        freqs_ptr + freq_pos * freqs_stride0 + hdi
                    ).load[width=simd_width]()
                    var res = rope_value(val, freq)
                    (q_out_ptr + q_base).store(res)

        var q_launch_shape = (total_seq_len, q_dim)
        var q_device_ctx = context.value() if context else DeviceContext(
            api="cpu"
        )
        elementwise[
            func=rope_q_fn,
            simd_width=q_kernel_simd_width,
            target=target,
        ](q_launch_shape, q_device_ctx)

        # ---------- K+V launch (fp8 quantize+store, one thread per block) ----------
        # Launch space is [total_seq_len, 2 * num_kv_heads * head_size /
        # quant_g] — one thread per (token, kv_id, kv_head, block).
        # Each thread handles a full `quant_g`-wide block.
        comptime kv_simd_width = quant_g
        comptime assert (
            head_size % kv_simd_width == 0
        ), "FP8 KV path: head_size must be divisible by quant_g"
        comptime fp8_max_val: Float32 = Float32(
            max_finite[DType.float8_e4m3fn]()
        )

        # Inner stride: simd_width-units per row in launch space.
        # We multiply the column index by `kv_simd_width` so consecutive
        # threads cover consecutive blocks (one block per thread).
        var num_kv_blocks_per_token = (
            2 * num_kv_heads * head_size
        )  # in elements, will be /quant_g in launch

        @parameter
        @__copy_capture(
            q_dim,
            qk_offset,
            combined_dim,
            batch_size,
            freqs_ptr,
            freqs_stride0,
            qkv_ptr,
            k_cache,
            v_cache,
            input_row_offsets,
        )
        def rope_split_store_fp8_fn[
            simd_width: Int, alignment: Int = 1
        ](idx_arg: Coord):
            comptime assert idx_arg.rank == 2
            # `elementwise` may instantiate this function at smaller
            # widths (notably width=1) for the uneven-SIMD tail. The fp8
            # block-quantize body is only meaningful at the launch width
            # (== `kv_simd_width`); compile other widths to a no-op so the
            # type system doesn't have to handle width=1 cases that
            # would break rope_value's width/2 internals.
            comptime if simd_width == kv_simd_width:
                var idx = rebind[IndexList[2]](coord_to_index_list(idx_arg))
                var global_token_idx = idx[0]
                # col here ranges over flattened [k_dim + v_dim] in element
                # units (elementwise scaled by simd_width so col is the
                # base element index of this thread's block).
                var col_kv = idx[1]

                var bi: Int = get_batch_from_row_offsets(
                    input_row_offsets, global_token_idx
                )
                var ti = Int(
                    UInt32(global_token_idx) - input_row_offsets.raw_load(bi)
                )
                var cache_pos = k_cache.cache_length(bi) + ti

                if col_kv < (kv_params_const.num_heads * head_size):
                    # K region.
                    var hi, di = divmod(
                        UInt(col_kv), UInt(kv_params_const.head_size)
                    )
                    var freq_pos = get_freq_pos(
                        Int(di), global_token_idx, cache_pos
                    )
                    var qkv_base = (
                        global_token_idx * combined_dim
                        + q_dim
                        + Int(hi) * head_size
                        + Int(di)
                    )

                    # 1) Gather rope-applied K block (bf16 SIMD).
                    var roped_bf16: SIMD[dtype, simd_width]
                    comptime if interleaved:
                        var val = (qkv_ptr + qkv_base).load[width=simd_width]()
                        var freq = (
                            freqs_ptr + freq_pos * freqs_stride0 + Int(di)
                        ).load[width=simd_width]()
                        roped_bf16 = rope_value(val, freq)
                    else:
                        comptime width_2 = simd_width / 2
                        var k_head_base = (
                            global_token_idx * combined_dim
                            + q_dim
                            + Int(hi) * head_size
                        )
                        var re_idx, im_idx = get_safetensors_idx(
                            Int(di), head_size
                        )
                        var val_re = (qkv_ptr + k_head_base + re_idx).load[
                            width=width_2
                        ]()
                        var val_im = (qkv_ptr + k_head_base + im_idx).load[
                            width=width_2
                        ]()
                        var val = rebind[SIMD[dtype, simd_width]](
                            val_re.interleave(val_im)
                        )
                        var freq = (
                            freqs_ptr + freq_pos * freqs_stride0 + Int(di)
                        ).load[width=simd_width]()
                        roped_bf16 = rope_value(val, freq)

                    # 2) Per-block max-abs and scale = max_abs / FP8_MAX.
                    var roped_f32 = roped_bf16.cast[DType.float32]()
                    var max_abs = abs(roped_f32).reduce_max()
                    var scale: Float32 = max_abs / fp8_max_val
                    var inv_scale: Float32 = (
                        Float32(0.0) if scale
                        == Float32(0.0) else Float32(1.0) / scale
                    )
                    # 3) Quantize to fp8.
                    var fp8_val = (roped_f32 * inv_scale).cast[
                        DType.float8_e4m3fn
                    ]()

                    # 4) Store fp8 block and scale.  Regardless of
                    # `interleaved`, the rope output occupies the full
                    # `g`-wide register block; we store it contiguously
                    # at `[di, di+g)`.  In the non-interleaved (HF
                    # rotate_half) case the K cache layout is therefore
                    # a per-rope-block PERMUTATION of the original
                    # head_dim positions.  Q·K dot product is invariant
                    # to that permutation when Q is stored the same way
                    # (= the contiguous q_out path above).
                    k_cache.store(
                        bi,
                        Int(hi),
                        cache_pos,
                        Int(di),
                        rebind[SIMD[kv_type, simd_width]](fp8_val),
                    )

                    # Emit one fp32 scale at the block-start head_dim_idx.
                    # Scale block covers `[di, di+g)`, matching the
                    # contiguous storage block written above.  The
                    # dequant kernel reads `scale_block_idx = d // g`
                    # at `d ∈ [di, di+g)` and recovers `di`.
                    k_cache.store_scale(
                        bi,
                        Int(hi),
                        cache_pos,
                        Int(di),
                        SIMD[cache_scale_dtype, 1](scale),
                    )
                else:
                    # V region.
                    var v_col = col_kv - (kv_params_const.num_heads * head_size)
                    var hi, di = divmod(
                        UInt(v_col), UInt(kv_params_const.head_size)
                    )
                    var qkv_base = (
                        global_token_idx * combined_dim
                        + qk_offset
                        + Int(hi) * head_size
                        + Int(di)
                    )
                    var val_bf16 = (qkv_ptr + qkv_base).load[width=simd_width]()
                    var val_f32 = val_bf16.cast[DType.float32]()
                    var max_abs = abs(val_f32).reduce_max()
                    var scale: Float32 = max_abs / fp8_max_val
                    var inv_scale: Float32 = (
                        Float32(0.0) if scale
                        == Float32(0.0) else Float32(1.0) / scale
                    )
                    var fp8_val = (val_f32 * inv_scale).cast[
                        DType.float8_e4m3fn
                    ]()
                    var cl = v_cache.value().cache_length(bi)
                    v_cache.value().store(
                        bi,
                        Int(hi),
                        ti + cl,
                        Int(di),
                        rebind[SIMD[kv_type, simd_width]](fp8_val),
                    )
                    v_cache.value().store_scale(
                        bi,
                        Int(hi),
                        ti + cl,
                        Int(di),
                        SIMD[cache_scale_dtype, 1](scale),
                    )

        # KV launch: one thread per block. Range scaled by simd_width.
        var kv_launch_shape = (total_seq_len, num_kv_blocks_per_token)
        var device_ctx = context.value() if context else DeviceContext(
            api="cpu"
        )
        elementwise[
            func=rope_split_store_fp8_fn,
            simd_width=kv_simd_width,
            target=target,
        ](kv_launch_shape, device_ctx)


# ===-----------------------------------------------------------------------===#
# Without position IDs
# ===-----------------------------------------------------------------------===#


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

    @parameter
    def get_freq_pos(
        dim_idx: Int, global_token_idx: Int, cache_pos: Int
    ) -> Int:
        return cache_pos

    return _rope_split_store_ragged_impl[
        target=target,
        interleaved=interleaved,
        get_freq_pos=get_freq_pos,
    ](
        qkv,
        input_row_offsets,
        freqs_cis,
        k_cache,
        v_cache,
        q_output,
        context,
    )


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
    ctx: DeviceContext,
) raises:
    """Rope+split+store with paged KV cache collection."""
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache: OptionalReg[type_of(k_cache)] = kv_collection.get_value_cache(
        layer_idx_cast
    )

    comptime if is_gpu[target]():
        cuda_ctx = ctx

    return _rope_split_store_ragged[
        target=target,
        interleaved=interleaved,
    ](qkv, input_row_offsets, freqs_cis, k_cache, v_cache, q_output, cuda_ctx)


# ===-----------------------------------------------------------------------===#
# With position IDs
# ===-----------------------------------------------------------------------===#


@always_inline
def _rope_split_store_ragged_with_position_ids[
    dtype: DType,
    freq_dtype: DType,
    cache_t: KVCacheT,
    //,
    *,
    target: StaticString,
    interleaved: Bool = True,
    mrope_types: TypeList[Trait=CoordLike, ...] = TypeList.of[
        Trait=CoordLike
    ](),
    mrope_section: Optional[Coord[*mrope_types]] = None,
    PositionIdsLayoutType: TensorLayout = RowMajorLayout[
        *Coord[Int64, Int64].element_types
    ],
](
    qkv: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[DType.uint32, ...],
    freqs_cis: TileTensor[freq_dtype, ...],
    k_cache: cache_t,
    v_cache: OptionalReg[cache_t],
    position_ids: TileTensor[DType.uint32, PositionIdsLayoutType, ...],
    q_output: TileTensor[mut=True, dtype, ...],
    context: Optional[DeviceContext],
) raises:
    """Read flat QKV buffer, apply rope (with explicit position IDs) to Q and K,
    store K/V to cache.

    Like ``_rope_split_store_ragged`` but looks up RoPE frequencies using
    ``position_ids`` instead of ``cache_length + token_offset``.  When
    ``mrope_section`` is provided, different head-dimension sections use
    different rows of ``position_ids`` (multi-axis RoPE for VL models).

    Args:
        qkv: Flat matmul output [total_seq_len, q_dim + k_dim + v_dim].
        input_row_offsets: [batch_size + 1] ragged offsets.
        freqs_cis: [max_seq_len, head_dim] interleaved rope frequencies.
        k_cache: Key cache to store roped K.
        v_cache: Value cache to store V.
        position_ids: [num_sections, total_seq_len] explicit position IDs.
        q_output: Output buffer for roped Q [total_seq_len, q_dim].
        context: DeviceContext for GPU.
    """
    comptime assert PositionIdsLayoutType.rank == 2

    # Validate mrope_section alignment with kernel SIMD width.
    comptime kv_params = cache_t.kv_params
    comptime head_size = kv_params.head_size
    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[dtype, target=compile_target]()
    comptime kernel_simd_width = gcd(target_simd_width, head_size)
    comptime if mrope_section:
        comptime for i in range(len(mrope_section.value())):
            comptime assert (
                Int(mrope_section.value()[i].value()) % kernel_simd_width == 0
            ), "mrope_section must be divisible by rope kernel simd_width"

    var pos_ids_ptr = position_ids.ptr
    var pos_ids_stride = Int(position_ids.dim[1]())

    @parameter
    @__copy_capture(pos_ids_ptr, pos_ids_stride)
    def get_freq_pos(
        dim_idx: Int, global_token_idx: Int, cache_pos: Int
    ) -> Int:
        comptime if mrope_section:
            var section_idx = 0
            comptime for i in range(len(mrope_section.value())):
                comptime val = Int(mrope_section.value()[i].value())
                if dim_idx < val:
                    section_idx = i
                    break
            return Int(
                pos_ids_ptr[section_idx * pos_ids_stride + global_token_idx]
            )
        else:
            return Int(pos_ids_ptr[global_token_idx])

    return _rope_split_store_ragged_impl[
        target=target,
        interleaved=interleaved,
        get_freq_pos=get_freq_pos,
    ](
        qkv,
        input_row_offsets,
        freqs_cis,
        k_cache,
        v_cache,
        q_output,
        context,
    )


@always_inline
def rope_split_store_paged_ragged_with_position_ids[
    dtype: DType,
    freq_dtype: DType,
    target: StaticString = "cpu",
    interleaved: Bool = True,
    mrope_types: TypeList[Trait=CoordLike, ...] = TypeList.of[
        Trait=CoordLike
    ](),
    mrope_section: Optional[Coord[*mrope_types]] = None,
    PositionIdsLayoutType: TensorLayout = RowMajorLayout[
        *Coord[Int64, Int64].element_types
    ],
](
    qkv: TileTensor[dtype, ...],
    input_row_offsets: TileTensor[DType.uint32, ...],
    freqs_cis: TileTensor[freq_dtype, ...],
    kv_collection: PagedKVCacheCollection,
    position_ids: TileTensor[DType.uint32, PositionIdsLayoutType, ...],
    layer_idx: UInt32,
    q_output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """Rope+split+store with paged KV cache and explicit position IDs."""
    var cuda_ctx: Optional[DeviceContext] = None
    var layer_idx_cast = Int(layer_idx)
    var k_cache = kv_collection.get_key_cache(layer_idx_cast)
    var v_cache: OptionalReg[type_of(k_cache)] = kv_collection.get_value_cache(
        layer_idx_cast
    )

    comptime if is_gpu[target]():
        cuda_ctx = ctx

    return _rope_split_store_ragged_with_position_ids[
        target=target,
        interleaved=interleaved,
        mrope_types=mrope_types,
        mrope_section=mrope_section,
        PositionIdsLayoutType=PositionIdsLayoutType,
    ](
        qkv,
        input_row_offsets,
        freqs_cis,
        k_cache,
        v_cache,
        position_ids,
        q_output,
        cuda_ctx,
    )
