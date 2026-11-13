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
"""Large-M router-gate GEMM for SM100 (B200): bf16 activation, fp32 weight.

Target family: Blackwell sm_100. Shape of record: MiniMax-M3 prefill,
`[16384, 6144] bf16 x [128, 6144] fp32^T -> [16384, 128] fp32`.

The baseline widens the activation to fp32 in a standalone pass and then runs
an fp32 matmul, which the SM100 dispatch executes with TF32 tensor cores. That
moves the `[M, K]` activation through HBM three times (read bf16, write fp32,
read fp32). A bf16 value is exact in TF32, so only the weight needs TF32
rounding, and the router weight is tiny (`N * K`). This route therefore
splits each fp32 weight into two bf16 values whose sum is that weight
truncated to TF32's 10-bit mantissa, runs one bf16 tensor-core GEMM over the
interleaved `[2N, K]` weight, and adds each output pair in the GEMM epilogue.
The activation is read once, as bf16.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.sys import size_of

from layout import Idx, TensorEngine, TensorLayout, TileTensor, row_major
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.utils.coord import Coord

# Mask that keeps sign, exponent and the top 7 mantissa bits: the bf16 value
# nearest below the fp32 input in magnitude.
comptime _BF16_MASK = UInt32(0xFFFF0000)
# Mask that keeps the top 10 mantissa bits, TF32's operand precision.
comptime _TF32_MASK = UInt32(0xFFFFE000)


def _split_weight_tf32_to_bf16_pairs(
    b: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    b_pairs: MutPointer[Scalar[DType.bfloat16], MutUntrackedOrigin],
    num_elements: Int32,
    k: Int32,
):
    """Writes `b_pairs[2 * n]` = hi and `b_pairs[2 * n + 1]` = mid of `b[n]`.

    Rows interleave so one MMA output pair `(2 * n, 2 * n + 1)` holds both
    partial products of output column `n`. `hi + mid` is `b` rounded to TF32
    (to nearest), and both terms are exact bf16 values.
    """
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i >= Int(num_elements):
        return
    var bits = bitcast[DType.uint32](b.unsafe_load(i))
    # Half an ulp of the 10-bit mantissa; a carry out of the mantissa bumps
    # the exponent, which is the correct rounding.
    bits += UInt32(0x1000)
    var hi = bitcast[DType.float32](bits & _BF16_MASK)
    var tf32 = bitcast[DType.float32](bits & _TF32_MASK)
    # `tf32 - hi` is exact: it keeps at most 3 mantissa bits.
    var mid = tf32 - hi
    var kk = Int(k)
    var row = i // kk
    var dst = 2 * row * kk + (i - row * kk)
    b_pairs.unsafe_store(dst, hi.cast[DType.bfloat16]())
    b_pairs.unsafe_store(dst + kk, mid.cast[DType.bfloat16]())


def router_gate_tf32_gemm[
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    //,
    *,
    cta_group: Int = 2,
](
    c: TileTensor[.float32, c_layout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[.bfloat16, a_layout, ImmutAnyOrigin, Engine=a_engine],
    b: TileTensor[.float32, b_layout, ImmutAnyOrigin, Engine=b_engine],
    m: Int,
    ctx: DeviceContext,
) raises:
    """Computes `c = a @ b^T` for bf16 `a` and fp32 `b` with TF32-class numerics.

    Every product is exact (a bf16 activation times a TF32-truncated weight
    fits in fp32), as in the fp32 matmul's TF32 tensor-core route; only the
    fp32 accumulation order differs.

    Parameters:
        c_layout: Layout of the fp32 output.
        a_layout: Layout of the bf16 activation.
        b_layout: Layout of the fp32 weight.
        c_engine: Engine of the fp32 output.
        a_engine: Engine of the bf16 activation.
        b_engine: Engine of the fp32 weight.
        cta_group: MMA CTA group, 1 or 2.

    Args:
        c: Output `[M, N]` fp32 tensor.
        a: Activation `[M, K]` bf16 tensor.
        b: Weight `[N, K]` fp32 tensor (transpose_b layout), static shape.
        m: Runtime row count.
        ctx: The device context.
    """
    comptime N = type_of(b).static_shape[0]
    comptime K = type_of(b).static_shape[1]
    comptime assert N > 0 and K > 0, "the weight needs a static [N, K] shape"
    comptime assert (2 * N * size_of[DType.float32]()) % 16 == 0
    if m == 0:
        return

    var b_pairs = ctx.enqueue_create_buffer[DType.bfloat16](2 * N * K)
    comptime split = _split_weight_tf32_to_bf16_pairs
    comptime threads = 256
    ctx.enqueue_function[split](
        b.unsafe_ptr(),
        b_pairs.unsafe_ptr(),
        Int32(N * K),
        Int32(K),
        grid_dim=ceildiv(N * K, threads),
        block_dim=threads,
    )
    var b_pairs_tt = TileTensor(b_pairs, row_major(Idx[2 * N], Idx[K])).as_imm()

    comptime config = MatmulConfig[
        DType.bfloat16, DType.bfloat16, DType.float32, True
    ](
        mma_shape=(128 * cta_group, 2 * N, 16),
        cluster_shape=(cta_group, 1, 1),
        cta_group=cta_group,
    )

    def sum_pairs[
        dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[dtype, width]) {var c}:
        comptime assert idx.rank == 2, "the matmul epilogue contract is rank 2"
        comptime assert (
            width % 2 == 0
        ), "the epilogue gets whole (hi, mid) pairs"
        var row = Int(idx[0])
        var col = Int(idx[1]) // 2
        comptime for i in range(0, width, 2):
            c.store[alignment=size_of[DType.float32]()](
                (row, col + i // 2),
                SIMD[.float32, 1](
                    rebind[Scalar[.float32]](val[i] + val[i + 1])
                ),
            )

    blackwell_matmul_tma_umma_warp_specialized[True, config=config](
        (m, 2 * N), a, b_pairs_tt, sum_pairs, ctx
    )
    _ = b_pairs^
