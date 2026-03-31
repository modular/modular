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
"""Correction warp group logic for depth=512 pair-CTA SM100 attention.

Rescales the O accumulator in TMEM when the per-row maximum changes during
online softmax. O is split into two halves (O_lo, O_hi) produced by separate
P@V MMA instructions (MMA_N=ov_depth/2 each). The correction warp processes
O_lo first, releasing PO_lo to unblock the next iteration's P@V_lo, then
processes O_hi and releases PO_hi.

Pair-CTA TMEM column layout for O (4 quadrants):
    First MMA (O_lo, TMEM base = TMEM_O, ov_depth/4 physical cols):
        Rows 0-63:   O logical cols 0 .. ov_depth/4 - 1
        Rows 64-127: O logical cols ov_depth/4 .. ov_depth/2 - 1
    Second MMA (O_hi, TMEM base = TMEM_O_hi, ov_depth/4 physical cols):
        Rows 0-63:   O logical cols ov_depth/2 .. 3*ov_depth/4 - 1
        Rows 64-127: O logical cols 3*ov_depth/4 .. ov_depth - 1

Both row groups access the SAME physical TMEM column range within each MMA
region — the pair-CTA layout distinguishes them by row, not column address.
All 128 threads participate in both phases, each processing ov_depth/4
physical columns per phase.
"""

from std.sys import size_of
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_ld,
    tcgen05_st,
    tcgen05_store_wait,
    tcgen05_fence_before,
    tcgen05_fence_after,
)
from std.gpu.primitives.warp import _vote_nvidia_helper
from linalg.matmul.gpu.sm100_structured.structured_kernels.tmem import (
    TmemAddress,
)
from std.gpu import thread_idx_uint as thread_idx
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    mul_ftz,
)
from nn.attention.mha_mask import MHAMask
from .barriers import Depth512MBars
from .config import Depth512SM100Config
from .smem import Depth512AttentionSMem


@always_inline
def depth512_correction[
    MaskType: MHAMask,
    qkv_dtype: DType,
    config: Depth512SM100Config[qkv_dtype],
    page_size: Int,
](
    smem: Depth512AttentionSMem[config=config],
    score_row: UInt32,
    num_keys: UInt32,
    mask: MaskType,
):
    comptime accum_type = DType.float32
    comptime assert size_of[accum_type]() == 4
    comptime BM = config.BM
    comptime BN = config.BN
    comptime PairBM = BM * 2

    # Columns per thread per O phase. Each MMA (MMA_N=ov_depth/2) maps:
    #   rows 0-63 → first ov_depth/4 cols, rows 64-127 → second ov_depth/4.
    comptime ov_quarter = config.ov_depth // 4
    var mbars = Depth512MBars[config.num_kv_stages](smem.mbar_base())

    # Dummy arrives for the prologue iteration (no previous O to protect).
    # This satisfies the correction half of PO_lo and PO_hi for the first P@V.
    _ = mbars.po_lo_mbar()[].arrive()
    _ = mbars.po_hi_mbar()[].arrive()

    # ---- Thread identity -----------------------------------------------------
    var tid: UInt32 = UInt32(thread_idx.x)
    var row: UInt32 = tid % 128
    var m_row = row % UInt32(BM)

    # TMEM base for each phase.  Both row groups (0-63, 64-127) read/write
    # the same physical columns — pair-CTA layout maps different rows to
    # different logical output columns but uses the same physical addresses.
    var tmem_addr: UInt32 = smem.tmem_addr_ptr()[]
    var o_lo_tmem = TmemAddress(tmem_addr + UInt32(config.TMEM_O))
    var o_hi_tmem = TmemAddress(tmem_addr + UInt32(config.TMEM_O_hi))

    var correction_smem = smem.correction_smem() + m_row

    pipeline_c = mbars.consumer_c()
    pipeline_o_lo = mbars.consumer_o_lo()
    pipeline_o_hi = mbars.consumer_o_hi()

    var iter_count: UInt32 = (
        mask.total_iters[PairBM, BN, page_size](score_row, num_keys) - 1
    )

    # ---- Double-buffer constants for the TMEM rescale loop -------------------
    # Each phase processes ov_quarter columns.  Follows the FA4 pattern:
    # alternate between o_b0 and o_b1 loads so masking overlaps TMEM access.
    comptime batch_size = 16 if ov_quarter % 16 == 0 else 8
    comptime assert ov_quarter % batch_size == 0
    comptime load_iters = ov_quarter // (2 * batch_size)
    comptime load_remainder = ov_quarter % (2 * batch_size)
    comptime assert load_iters > 1
    comptime assert (load_remainder == batch_size) or (load_remainder == 0)

    # ---- Rescale helper (inlined for O_lo and O_hi) --------------------------

    @parameter
    @always_inline
    def rescale_o(o_tmem: TmemAddress, c_pair: SIMD[DType.float32, 2]):
        """Double-buffered TMEM load/scale/store over ov_quarter columns."""
        var o_b0: InlineArray[Scalar[accum_type], batch_size]
        var o_b1: InlineArray[Scalar[accum_type], batch_size]
        o_b0 = tcgen05_ld[
            datapaths=32,
            bits=32,
            repeat=batch_size,
            dtype=accum_type,
            pack=False,
            width=batch_size,
        ](o_tmem.addr)

        comptime for b in range(load_iters):
            comptime b0_offset0 = 2 * b * batch_size
            comptime b1_offset = b0_offset0 + batch_size
            comptime b0_offset1 = b1_offset + batch_size
            o_b1 = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=batch_size,
                dtype=accum_type,
                pack=False,
                width=batch_size,
            ]((o_tmem + b1_offset).addr)
            var o_b0_scaled = InlineArray[Scalar[accum_type], batch_size](
                uninitialized=True
            )

            comptime for _i in range(0, batch_size, 2):
                var pair = mul_ftz(
                    SIMD[DType.float32, 2](
                        rebind[Scalar[DType.float32]](o_b0[_i]),
                        rebind[Scalar[DType.float32]](o_b0[_i + 1]),
                    ),
                    c_pair,
                )
                o_b0_scaled[_i] = pair[0]
                o_b0_scaled[_i + 1] = pair[1]
            tcgen05_st[
                datapaths=32,
                bits=32,
                repeat=batch_size,
                pack=False,
            ]((o_tmem + b0_offset0).addr, o_b0_scaled)

            comptime if b0_offset1 + batch_size <= ov_quarter:
                o_b0 = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=batch_size,
                    dtype=accum_type,
                    pack=False,
                    width=batch_size,
                ]((o_tmem + b0_offset1).addr)
            var o_b1_scaled = InlineArray[Scalar[accum_type], batch_size](
                uninitialized=True
            )

            comptime for _i in range(0, batch_size, 2):
                var pair = mul_ftz(
                    SIMD[DType.float32, 2](
                        rebind[Scalar[DType.float32]](o_b1[_i]),
                        rebind[Scalar[DType.float32]](o_b1[_i + 1]),
                    ),
                    c_pair,
                )
                o_b1_scaled[_i] = pair[0]
                o_b1_scaled[_i + 1] = pair[1]
            tcgen05_st[
                datapaths=32,
                bits=32,
                repeat=batch_size,
                pack=False,
            ]((o_tmem + b1_offset).addr, o_b1_scaled)

        comptime if load_remainder > 0:
            comptime offset = 2 * batch_size * load_iters
            var o_b0_scaled_rem = InlineArray[
                Scalar[accum_type], load_remainder
            ](uninitialized=True)

            comptime for _i in range(0, load_remainder, 2):
                var pair = mul_ftz(
                    SIMD[DType.float32, 2](
                        rebind[Scalar[DType.float32]](o_b0[_i]),
                        rebind[Scalar[DType.float32]](o_b0[_i + 1]),
                    ),
                    c_pair,
                )
                o_b0_scaled_rem[_i] = pair[0]
                o_b0_scaled_rem[_i + 1] = pair[1]
            tcgen05_st[
                datapaths=32,
                bits=32,
                repeat=load_remainder,
                pack=False,
            ]((o_tmem + offset).addr, o_b0_scaled_rem)
        tcgen05_store_wait()
        tcgen05_fence_before()

    # ---- Main loop -----------------------------------------------------------

    while iter_count != 0:
        iter_count -= 1

        # Read correction factor from softmax.
        pipeline_c.wait()
        var c_scalar: Scalar[accum_type] = correction_smem[0]

        change = _vote_nvidia_helper(c_scalar < 1.0) != 0

        # Phase 1: rescale O_lo (all 128 threads, ov_quarter cols each).
        pipeline_o_lo.wait()
        if change:
            tcgen05_fence_after()
            var c_pair = SIMD[DType.float32, 2](c_scalar, c_scalar)
            rescale_o(o_lo_tmem, c_pair)
            pipeline_o_lo.release()

            # Phase 2: rescale O_hi (all 128 threads, ov_quarter cols each).
            pipeline_o_hi.wait()
            tcgen05_fence_after()
            rescale_o(o_hi_tmem, c_pair)
        else:
            pipeline_o_lo.release()
            pipeline_o_hi.wait()

        pipeline_o_hi.release()
        pipeline_c.release()
