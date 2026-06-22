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
"""Host (numpy) replica of the GPU NVFP4 weight repack.

This reproduces ``repack_nvfp4_for_sm8x`` (see
``max/kernels/src/quantization/qmatmul_gpu.mojo``) BYTE-FOR-BYTE on the host so
the skeleton's combined weight buffer can be produced at load time without ever
uploading the original packed weight to the GPU (which doubles GPU memory and
risks OOM).

The combined output buffer has shape ``[N, K/2 + (K/16)*2]`` uint8:

    [ repacked 4-bit weights | bf16 block scales ]

* The weight region is byte-identical to ``repack_GPTQ_for_sm8x`` (same
  ``pack_Q_tile`` bit-shuffle and the nested ``repacked_weights_layout``).
* The scale region is ``row_major(K/16, N)`` bf16 where
  ``scale[g, n] = bf16( f32(fp8_block_scale[n, g]) * global )``.

The mapping is derived directly from the kernel's warp/lane tile distribution;
see the docstrings on the individual helpers.
"""

from __future__ import annotations

import numpy as np

__all__ = ["nvfp4_repack_host", "e4m3fn_to_f32", "f32_to_bf16_bits"]

# --------------------------------------------------------------------------- #
# Scalar casts, matched bit-for-bit to Mojo.
# --------------------------------------------------------------------------- #


def e4m3fn_to_f32(codes: np.ndarray) -> np.ndarray:
    """Decode float8_e4m3fn bytes (uint8) to float32, bit-exact.

    e4m3fn: 1 sign, 4 exponent (bias 7), 3 mantissa. No infinities; the only
    NaN encoding is sign + all-ones exponent + all-ones mantissa (0x7F/0xFF).
    Subnormals (exp == 0) are ``mantissa/8 * 2^-6``.
    """
    u = codes.astype(np.uint8)
    sign = (u >> 7) & 0x1
    exp = (u >> 3) & 0xF
    mant = u & 0x7

    out = np.zeros(u.shape, dtype=np.float32)

    # Normals: exp in [1, 15]. value = 2^(exp-7) * (1 + mant/8).
    normal = exp != 0
    exp_f = exp.astype(np.int32) - 7
    out_normal = (1.0 + mant.astype(np.float32) / 8.0) * np.exp2(
        exp_f.astype(np.float32)
    )
    out = np.where(normal, out_normal, out)

    # Subnormals: exp == 0. value = mant/8 * 2^-6.
    sub = exp == 0
    out_sub = mant.astype(np.float32) / 8.0 * np.float32(2.0**-6)
    out = np.where(sub, out_sub, out)

    # NaN: 0x7F / 0xFF (exp==15 & mant==7). e4m3fn has no inf.
    nan = (exp == 0xF) & (mant == 0x7)
    out = np.where(nan, np.float32(np.nan), out)

    out = np.where(sign.astype(bool), -out, out)
    return out.astype(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Convert float32 -> bfloat16 raw 16-bit pattern (uint16), round-to-nearest
    -even. Matches LLVM ``fptrunc f32 -> bf16`` (what Mojo ``.cast[bfloat16]()``
    lowers to).
    """
    x = x.astype(np.float32)
    bits = x.view(np.uint32)

    # NaN: force a quiet bf16 NaN (top mantissa bit set), preserving sign. LLVM
    # produces 0x7FC0-style payloads; matching the exact payload isn't needed
    # for the representable test values, but handle it deterministically.
    is_nan = np.isnan(x)

    # round-to-nearest-even on the discarded low 16 bits.
    lsb = (bits >> 16) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    rounded = bits + rounding_bias
    out = (rounded >> 16).astype(np.uint16)

    nan_pattern = ((bits >> 16).astype(np.uint16) | np.uint16(0x0040)).astype(
        np.uint16
    )
    out = np.where(is_nan, nan_pattern, out)
    return out.astype(np.uint16)


# --------------------------------------------------------------------------- #
# pack_Q_tile: the per-Q-tile bit shuffle (qmatmul_gpu.mojo ~line 1075).
# --------------------------------------------------------------------------- #


def _pack_Q_tile(tmp: np.ndarray) -> np.ndarray:
    """Vectorized ``pack_Q_tile``.

    ``tmp`` has trailing dim 16 (uint8 bytes, 2 nibble-codes each). Returns a
    uint32 array with trailing dim 4. Operates over all leading dims at once.
    """
    t = tmp.astype(np.uint32)
    res = np.zeros(t.shape[:-1] + (4,), dtype=np.uint32)
    for i in range(4):
        a = t[..., i * 4 + 0]
        b = t[..., i * 4 + 1]
        c = t[..., i * 4 + 2]
        d = t[..., i * 4 + 3]
        v = np.zeros(t.shape[:-1], dtype=np.uint32)
        v |= a & 0x0F
        v |= (a & 0xF0) << 12
        v |= (b & 0x0F) << 4
        v |= (b & 0xF0) << 16
        v |= (c & 0x0F) << 8
        v |= (c & 0xF0) << 20
        v |= (d & 0x0F) << 12
        v |= (d & 0xF0) << 24
        res[..., i] = v
    return res


# --------------------------------------------------------------------------- #
# Repacked-weight destination layout (the nested repacked_weights_layout).
# --------------------------------------------------------------------------- #


def _repacked_weight_flat_index(
    n: np.ndarray, ku: np.ndarray, N: int, uint_K: int
) -> np.ndarray:
    """Flat uint32 index into the repacked weight region for logical (n, ku).

    Mirrors::

        repacked_weights_layout = Layout(
            IntTuple(IntTuple(64, N//64), IntTuple(2, uint_K//2)),
            IntTuple(IntTuple(2, 128*(uint_K//2)), IntTuple(1, 128)),
        )

    where ``n`` is the global row (0..N) and ``ku`` is the global uint32 column
    (0..uint_K). The layout splits each coordinate into (inner, outer):

        n  -> (n % 64, n // 64)     strides (2, 128*(uint_K//2))
        ku -> (ku % 2, ku // 2)     strides (1, 128)
    """
    n_in = n % 64
    n_out = n // 64
    ku_in = ku % 2
    ku_out = ku // 2
    return (
        n_in * 2
        + n_out * (128 * (uint_K // 2))
        + ku_in * 1
        + ku_out * 128
    )


# --------------------------------------------------------------------------- #
# Weight repack.
# --------------------------------------------------------------------------- #


def _repack_weights(weight_u8: np.ndarray, N: int, K: int) -> np.ndarray:
    """Produce the repacked 4-bit weight region as a flat uint8 array of length
    ``N * K // 2``.

    ``weight_u8`` is ``[N, K//2]`` uint8 (canonical NVFP4: 2 E2M1 nibble codes
    per byte, low nibble = even k).
    """
    pack_factor = 8
    repack_tile = (64, 16)
    WGROUP = 128
    weights_bytes_per_group = WGROUP // 2  # 64 bytes
    uint_K = K // pack_factor
    WK_groups = K // WGROUP
    n_q_tiles = WGROUP // repack_tile[1]  # 8
    q_uint = repack_tile[1] // pack_factor  # 2

    # Output uint32 buffer (flat), reinterpreted from the uint8 region.
    out_u32 = np.zeros(N * uint_K, dtype=np.uint32)

    # `weight_u8` row n holds K//2 bytes contiguously. The kernel views this as
    # uint32 [N, uint_K] (raw_weights). The smem staging is a pure copy, so we
    # can index `weight_u8` directly with the warp/lane tile math.
    #
    # Each warp owns 64 rows (warp_x in {0,1} within a 128-row block) and one
    # WGROUP block of 64 bytes of K (warp_y selects the block). Globally, a
    # warp's 64-row half is `n_base = block_n*128 + warp_x*64`, and its WGROUP
    # block is `wgroup_idx` covering bytes [wgroup_idx*64, +64).
    #
    # We enumerate every (global 64-row tile, global WGROUP block, q-tile, lane)
    # and place the 4 packed uint32 directly. Vectorize over lane (0..31).
    lane = np.arange(32)
    lane_row = lane // 4  # row within distribute[row_major(8,4)]
    lane_col = lane % 4

    n_row_tiles = N // 64

    # Fully vectorized over (nt, wgroup_idx, i_Q_tile, lane, i_e). The scalar
    # math below is IDENTICAL to the per-iteration version above (kept in the
    # comments); only the Python loops are replaced by broadcasting so the
    # repack is feasible at 31B scale.
    nt_a = np.arange(n_row_tiles)
    wg_a = np.arange(WK_groups)
    iq_a = np.arange(n_q_tiles)
    ie_a = np.arange(16)

    # Per-axis decompositions.
    n_base = nt_a * 64  # [nt]
    byte_group_base = wg_a * weights_bytes_per_group  # [wg]
    ku_group_base = wg_a * (WGROUP // pack_factor)  # [wg]
    fr = ie_a // 2  # [ie]
    fc = ie_a % 2  # [ie]

    # rows[nt, lane, ie] = n_base + fr*8 + lane_row
    rows = (
        n_base[:, None, None]
        + (fr * 8)[None, None, :]
        + lane_row[None, :, None]
    )  # [nt, 32, ie]
    # cols[wg, iq, lane, ie] = byte_group_base + iq*(repack_tile[1]//2)
    #                          + fc*4 + lane_col
    cols = (
        byte_group_base[:, None, None, None]
        + (iq_a * (repack_tile[1] // 2))[None, :, None, None]
        + (fc * 4)[None, None, None, :]
        + lane_col[None, None, :, None]
    )  # [wg, iq, 32, ie]

    # tmp[nt, wg, iq, lane, ie] = weight_u8[rows, cols]
    rows_b = rows[:, None, None, :, :]  # [nt,1,1,32,ie]
    cols_b = cols[None, :, :, :, :]  # [1,wg,iq,32,ie]
    rows_b, cols_b = np.broadcast_arrays(rows_b, cols_b)
    tmp = weight_u8[rows_b, cols_b]  # [nt, wg, iq, 32, 16]

    packed = _pack_Q_tile(tmp)  # [nt, wg, iq, 32, 4]

    # ku_base[wg, iq] = ku_group_base + iq*q_uint
    ku_base = ku_group_base[:, None] + (iq_a * q_uint)[None, :]  # [wg, iq]
    r0 = n_base[:, None] + 2 * lane[None, :]  # [nt, 32]
    r1 = r0 + 1

    def scatter(r_axis: np.ndarray, ku_off: int, p_idx: int) -> None:
        # r_axis: [nt, 32]; ku: [wg, iq]. Broadcast to [nt, wg, iq, 32].
        r_b = r_axis[:, None, None, :]
        ku_b = (ku_base + ku_off)[None, :, :, None]
        r_b, ku_b = np.broadcast_arrays(r_b, ku_b)
        idx = _repacked_weight_flat_index(r_b, ku_b, N, uint_K)
        # packed dims [nt, wg, iq, 32, 4] -> select component p_idx -> move 32
        # to last axis to align with [nt, wg, iq, 32].
        out_u32[idx] = packed[..., p_idx]

    scatter(r0, 0, 0)
    scatter(r0, 1, 1)
    scatter(r1, 0, 2)
    scatter(r1, 1, 3)

    return out_u32.view(np.uint8)


# --------------------------------------------------------------------------- #
# Scale repack.
# --------------------------------------------------------------------------- #


def _repack_scales(
    block_scales_fp8: np.ndarray, global_scale: float, N: int, K: int, group_size: int
) -> np.ndarray:
    """Produce the bf16 scale region as a flat uint8 array.

    The destination is ``row_major(K_groups, N)`` bf16, holding
    ``bf16( f32(fp8[n, g]) * global )``, but the kernel writes it with an
    N-PERMUTATION inside every 64-column warp span -- it is NOT a plain
    transpose. Per the SCALE-repack loop (qmatmul_gpu.mojo ~line 1733):

      - BN=128 over N, BK=1024 over K; BK_groups = BK//group_size.
      - For each block and warp (warp_x in {0,1} = 64-row half of N,
        warp_y in {0,1} = which of the 2 groups in this step), lane writes
        destination N-columns ``2*lane`` and ``2*lane+1`` within its 64-col
        span, but reads SOURCE rows given by the thread layout
        ``Layout((4,8),(16,1))``:

            src_row(lane) = (lane % 4) * 16 + (lane // 4)

        dest col 2*lane   <- src_row(lane)
        dest col 2*lane+1 <- src_row(lane) + 8

      - Global dest N = block_n*128 + warp_x*64 + {2*lane, 2*lane+1}
      - Global dest group g = block_k*BK_groups + i*2 + warp_y
      - Global source N = block_n*128 + warp_x*64 + src_row(+8)
        Global source group g = block_k*BK_groups + i*2 + warp_y (same g)

    Net effect: within each 64-row block of N, the scale rows are permuted by
    P where dest position p maps from source row src_of(p):
        for lane in 0..31:
            src_of(2*lane)   = (lane%4)*16 + lane//4
            src_of(2*lane+1) = (lane%4)*16 + lane//4 + 8
    The group index g is preserved (same row of the row_major(K_groups, N)).
    """
    BN = 128
    BK = 1024
    BK_groups = BK // group_size
    K_groups = K // group_size

    f32 = e4m3fn_to_f32(block_scales_fp8)  # [N, K_groups]
    scaled = (f32 * np.float32(global_scale)).astype(np.float32)  # [N, K_groups]

    # Build the per-64 N-permutation: dest column p (0..63) <- source row p64.
    src_of = np.empty(64, dtype=np.int64)
    for lane in range(32):
        base = (lane % 4) * 16 + (lane // 4)
        src_of[2 * lane] = base
        src_of[2 * lane + 1] = base + 8

    # Destination row_major(K_groups, N) f32 (we cast to bf16 at the very end).
    dest = np.empty((K_groups, N), dtype=np.float32)

    # Each 64-row span of N is one (block_n, warp_x) pair. Global column base
    # is span*64 == block_n*128 + warp_x*64, so iterating spans directly is
    # equivalent to the kernel's BN-block / warp_x decomposition and also
    # handles N < BN (e.g. N=64) cleanly.
    n_spans = N // 64
    for s in range(n_spans):
        col_base = s * 64
        for p in range(64):
            dest[:, col_base + p] = scaled[col_base + src_of[p], :]

    # Note: the group index g maps identically (block_k*BK_groups + i*2 + warp_y
    # walks every g in 0..K_groups), so no group permutation -- dest row == g.
    bf16_bits = f32_to_bf16_bits(dest)  # uint16 [K_groups, N]
    return bf16_bits.reshape(K_groups * N).view(np.uint16).view(np.uint8)


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #


def nvfp4_repack_host(
    weight_u8: np.ndarray,
    block_scales_fp8: np.ndarray,
    global_scale: float,
    group_size: int = 16,
) -> np.ndarray:
    """Host replica of ``repack_nvfp4_for_sm8x``.

    Args:
        weight_u8: ``[N, K//2]`` uint8 packed NVFP4 weights (2 E2M1 codes/byte,
            low nibble = even k).
        block_scales_fp8: ``[N, K//group_size]`` uint8 bytes of float8_e4m3fn
            block scales.
        global_scale: per-tensor f32 multiplier (weight_scale_2).
        group_size: scale group size (16 for NVFP4 g16).

    Returns:
        Combined ``[N, K//2 + (K//group_size)*2]`` uint8 buffer:
        ``[repacked 4-bit weights][bf16 scales]``.
    """
    weight_u8 = np.ascontiguousarray(weight_u8.astype(np.uint8))
    block_scales_fp8 = np.ascontiguousarray(block_scales_fp8.astype(np.uint8))

    N = weight_u8.shape[0]
    K = weight_u8.shape[1] * 2
    assert N % 64 == 0, "N must be a multiple of 64"
    assert K % 128 == 0, "K must be a multiple of WGROUP=128"

    weight_region = _repack_weights(weight_u8, N, K)  # flat uint8 [N*K//2]
    scale_region = _repack_scales(
        block_scales_fp8, global_scale, N, K, group_size
    )  # flat uint8

    K_groups = K // group_size
    out_cols = K // 2 + K_groups * 2
    # The output buffer is two CONTIGUOUS blocks in the flat byte stream:
    #   bytes [0, N*K//2)        = repacked weights  (repacked_weights_layout)
    #   bytes [N*K//2, end)      = bf16 scales        (row_major(K_groups, N))
    # The declared [N, out_cols] tensor shape is just the nominal shape the
    # custom op returns; the content is NOT interleaved per row. So build the
    # flat stream and reshape to [N, out_cols] at the end.
    flat = np.concatenate([weight_region, scale_region])
    assert flat.size == N * out_cols
    return flat.reshape(N, out_cols)
