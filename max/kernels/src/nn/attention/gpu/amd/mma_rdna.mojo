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
"""RDNA-specific MMA wrapper for Wave32 WMMA operations.

This module provides MMA operations optimized for RDNA GPUs with:
- 16x16x16 WMMA shape
- 16-element A/B fragments
- 8-element C/D fragments
- Wave32 execution model
"""

from collections import OptionalReg
from math import ceildiv

from gpu import barrier
from gpu.compute.mma import mma as _mma_intrinsic
from layout import IntTuple, Layout
from layout.tensor_core import TiledTensorCore

from .buffers import KVBuffer, RegisterBuffer, RegisterMMABuffer
from .buffers_rdna import RDNA_AB_FRAG_SIZE, RDNA_CD_FRAG_SIZE


@parameter
fn _noop_copy_fn[i: Int]():
    """Default no-op function for A copy."""
    pass


@always_inline
fn mma_rdna[
    c_register_buffer_type: RegisterBuffer,
    a_register_buffer_type: RegisterMMABuffer,
    b_buffer_type: KVBuffer,
    //,
    tensor_core_mma: TiledTensorCore,
    BK: Int,
    prefetch_function: OptionalReg[fn() capturing -> None],
    swap_a_b: Bool = False,
    beg_iter: Int = 0,
    num_iters: Int = 1,
    prefetched_b_tile: Bool = False,
    a_copy_fn: fn[i: Int]() capturing -> None = _noop_copy_fn,
](
    c: c_register_buffer_type,
    mut a_tile: a_register_buffer_type,
    mut b_tile: b_buffer_type,
):
    """RDNA-specific MMA operation for Wave32 WMMA.

    This function performs matrix multiply-accumulate operations using
    RDNA's 16x16x16 WMMA instructions. It handles the K-dimension tiling
    and manages shared memory staging for the B operand.

    Parameters:
        c_register_buffer_type: Type for C accumulator buffer (8-element fragments).
        a_register_buffer_type: Type for A input buffer (16-element fragments).
        b_buffer_type: Type for B input buffer loaded from shared memory.
        tensor_core_mma: The TiledTensorCore configuration for RDNA.
        BK: Block size in K dimension.
        prefetch_function: Optional function to prefetch next tile.
        swap_a_b: Whether to swap A and B operands.
        beg_iter: Starting iteration index.
        num_iters: Number of iterations over tiles.
        prefetched_b_tile: Whether B tile is already prefetched.
        a_copy_fn: Callback to copy A (P) chunk i to shared memory.

    Args:
        c: Accumulator register buffer.
        a_tile: A operand register buffer.
        b_tile: B operand buffer (loaded to shared memory).
    """
    comptime assert (
        b_buffer_type._num_stages == 2
    ), "b_tile.num_stages must be 2"

    # For RDNA, group_size is always 1, so num_k_mmas2 is simpler
    comptime num_k_mmas2 = ceildiv(
        BK, tensor_core_mma.shape[2] * tensor_core_mma.group_size
    )

    comptime if not prefetched_b_tile:
        b_tile.load_from_dram()

    comptime for i in range(beg_iter, beg_iter + num_iters):
        comptime if i < beg_iter + num_iters - 1:
            b_tile.load_from_dram()

            comptime if i == beg_iter + num_iters - 2:
                comptime if prefetch_function:
                    comptime prefetch_func = prefetch_function.value()
                    prefetch_func()

        # Copy P chunk i to shared memory (writes to tile 0, reusing K's memory)
        # This must be done before V's copy_to_shared since they use different memory
        a_copy_fn[i]()

        b_tile.copy_to_shared[i % 2]()

        barrier()

        comptime for k_mma in range(num_k_mmas2):
            var a_reg_tile = a_tile.get_mma_tile[i, k_mma]()

            b_tile.load_from_shared[k_mma,]()

            var b_mma_tile = b_tile.get_mma_tile()
            var c_reg = c.get_reg_tile()

            # RDNA WMMA: bypass TensorCoreMMA.mma() which uses wrong
            # fragment sizes (M*K/WARP_SIZE=8 vs actual RDNA WMMA=16 for A/B).
            # Call the MMA intrinsic directly with correct RDNA sizes.
            var a_vec = a_reg_tile.vectorize[1, RDNA_AB_FRAG_SIZE]()
            var b_vec = b_mma_tile.vectorize[1, RDNA_AB_FRAG_SIZE]()
            var c_vec = c_reg.vectorize[1, RDNA_CD_FRAG_SIZE]()

            comptime num_m_mmas = a_vec.shape[0]()
            comptime num_n_mmas = b_vec.shape[0]()

            comptime for m_mma in range(num_m_mmas):
                comptime for n_mma in range(num_n_mmas):
                    # C register indexing: n_mma * num_m_mmas + m_mma
                    # regardless of swap_a_b. The swap only changes which
                    # operand goes to which MMA argument position.
                    comptime c_idx = Layout.col_major(num_m_mmas, num_n_mmas)(
                        IntTuple(m_mma, n_mma)
                    )

                    comptime if swap_a_b:
                        _mma_intrinsic(
                            c_vec[c_idx, 0],
                            b_vec[n_mma, 0],
                            a_vec[m_mma, 0],
                            c_vec[c_idx, 0],
                        )
                    else:
                        _mma_intrinsic(
                            c_vec[c_idx, 0],
                            a_vec[m_mma, 0],
                            b_vec[n_mma, 0],
                            c_vec[c_idx, 0],
                        )

        barrier()
