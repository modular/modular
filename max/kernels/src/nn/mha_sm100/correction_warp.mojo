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
"""Correction warp group logic for FA4 (SM100 Flash Attention)."""

from sys import size_of
from gpu import thread_idx
from gpu.compute.arch.tcgen05 import (
    tcgen05_ld,
    tcgen05_st,
    tcgen05_store_wait,
    tcgen05_fence_before,
)
from gpu.primitives.warp import _vote_nvidia_helper
from linalg.matmul.gpu.sm100_structured.structured_kernels.tmem import (
    TmemAddress,
)
from layout.tma_async import SharedMemBarrier
from nn.fa4_config import FA4Config, EnableForcedOrdering
from nn.sm100_attention_utils import (
    SharedMemPointer,
    FA4MiscMBars,
)
from nn.mha_mask import MHAMask


@always_inline
fn fa4_correction[
    qkv_type: DType,
    MaskType: MHAMask,
    config: FA4Config,
    page_size: Int,
](
    mbars: FA4MiscMBars[
        num_qk_stages = config.num_qk_stages,
        num_pv_stages = config.num_pv_stages,
        num_kv_stages = config.num_kv_stages,
        separate_kv=True,
        use_order_barriers=EnableForcedOrdering,
    ],
    score_row: UInt32,
    num_keys: UInt32,
    mask: MaskType,
):
    comptime accum_type = DType.float32
    comptime assert size_of[accum_type]() == 4
    comptime BM = config.BM
    comptime BN = config.BN

    # Offset calculations (matching SM100MHA2Q layout)
    comptime q_offset: Int32 = 0
    comptime kv_offset: Int32 = q_offset + Int32(
        config.BM * config.padded_depth
    )
    comptime correction_offset: Int32 = (
        kv_offset
        + Int32(2 * config.num_kv_stages * config.padded_depth * config.BN)
    ) * Int32(size_of[qkv_type]()) // Int32(size_of[DType.float32]())
    comptime mbar_offset = (correction_offset + Int32(config.BM)) * Int32(
        size_of[DType.float32]()
    ) // Int32(size_of[SharedMemBarrier]())

    comptime MiscMBarsType = type_of(mbars)

    # Dummy arrives for the prologue iteration (no previous O to protect).
    # This satisfies the combined barrier's correction half for the first P@V.
    _ = mbars.combined_p_o_consumer(0)[].arrive()
    _ = mbars.combined_p_o_consumer(1)[].arrive()

    var tmem_addr: UInt32 = (
        mbars.mbar_base + MiscMBarsType.num_mbars()
    ).bitcast[UInt32]()[]
    o0_tmem = TmemAddress(tmem_addr + UInt32(config.TMEM_O0))
    o1_tmem = TmemAddress(tmem_addr + UInt32(config.TMEM_O1))
    var correction_smem_arg: SharedMemPointer[Scalar[accum_type]] = (
        mbars.mbar_base - mbar_offset
    ).bitcast[Float32]() + correction_offset

    pipeline_c0 = mbars.consumer_c0()
    pipeline_c1 = mbars.consumer_c1()
    pipeline_o = mbars.consumer_o()

    var iter_count: UInt32 = (
        mask.total_iters[BM, BN, page_size](score_row, num_keys) - 1
    )

    comptime batch_size = 16 if config.depth % 16 == 0 else 8
    comptime assert config.depth % batch_size == 0
    # output is BM x depth
    comptime load_iters = config.depth // (2 * batch_size)
    comptime load_remainder = config.depth % (2 * batch_size)
    comptime assert load_iters > 1
    comptime assert (load_remainder == batch_size) or (load_remainder == 0)
    var correction_smem_0 = correction_smem_arg + UInt32(thread_idx.x) % 128
    var correction_smem_1 = correction_smem_0 + (BM // 2)

    while iter_count != 0:
        iter_count -= 1

        @parameter
        for i in range(2):
            # correct
            var c_scalar: Scalar[accum_type]

            @parameter
            if i == 0:
                pipeline_c0.wait()
                c_scalar = correction_smem_0[0]
            else:
                pipeline_c1.wait()
                c_scalar = correction_smem_1[0]

            change = _vote_nvidia_helper(c_scalar < 1.0) != 0
            pipeline_o.wait()
            if change:
                # TODO: experiment with different batch sizes.
                # The idea here is to both pipeline, and reduce peak register use.

                var o_tmem: TmemAddress

                @parameter
                if i == 0:
                    o_tmem = o0_tmem
                else:
                    o_tmem = o1_tmem

                var o_b0: SIMD[accum_type, batch_size]
                var o_b1: SIMD[accum_type, batch_size]
                o_b0 = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=batch_size,
                    dtype=accum_type,
                    pack=False,
                    width=batch_size,
                ](o_tmem.addr)

                @parameter
                for b in range(load_iters):
                    # BN=64 or BN=80, load_iters=2
                    # b=0
                    # b0_offset0=0
                    # b1_offset =16
                    # b0_offset1=32
                    # b=1
                    # b0_offset0=32
                    # b1_offset =48
                    # b0_offset1=64
                    comptime b0_offset0 = 2 * b * batch_size
                    comptime b1_offset = b0_offset0 + batch_size
                    comptime b0_offset1 = b1_offset + batch_size
                    o_b1 = tcgen05_ld[  # 0b1 start
                        datapaths=32,
                        bits=32,
                        repeat=batch_size,
                        dtype=accum_type,
                        pack=False,
                        width=batch_size,
                    ]((o_tmem + b1_offset).addr)
                    tcgen05_st[  # 0b0*c_scalar store
                        datapaths=32,
                        bits=32,
                        repeat=batch_size,
                        pack=False,
                    ]((o_tmem + b0_offset0).addr, o_b0 * c_scalar)

                    @parameter
                    if b0_offset1 + batch_size <= config.depth:
                        o_b0 = tcgen05_ld[  # 0b0 start
                            datapaths=32,
                            bits=32,
                            repeat=batch_size,
                            dtype=accum_type,
                            pack=False,
                            width=batch_size,
                        ]((o_tmem + b0_offset1).addr)
                    tcgen05_st[  # 0b0*c_scalar store
                        datapaths=32,
                        bits=32,
                        repeat=batch_size,
                        pack=False,
                    ]((o_tmem + b1_offset).addr, o_b1 * c_scalar)

                @parameter
                if load_remainder > 0:
                    comptime offset = 2 * batch_size * load_iters
                    tcgen05_st[  # 0b0*c_scalar store
                        datapaths=32,
                        bits=32,
                        repeat=load_remainder,
                        pack=False,
                    ]((o_tmem + offset).addr, o_b0 * c_scalar)
                tcgen05_store_wait()
                tcgen05_fence_before()

            pipeline_o.release()

            @parameter
            if i == 0:
                pipeline_c0.release()
            else:
                pipeline_c1.release()
