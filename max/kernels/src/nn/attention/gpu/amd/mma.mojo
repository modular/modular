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

from std.gpu.compute.mma import mma as gpu_mma
from std.gpu import lane_id, WARP_SIZE
from std.utils import IndexList
from layout import TileTensor
from layout.swizzle import Swizzle
from layout.tensor_core import num_matrix_reg
from layout.tile_layout import row_major, col_major


struct TiledMmaOp[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    group_size: Int,
    transpose_b: Bool = False,
]:
    """TileTensor-native MMA operation for AMD attention kernels.

    Wraps the raw GPU MMA intrinsic and operates directly on TileTensor
    register tiles.

    Parameters:
        out_type: Accumulator data type.
        in_type: Input matrix element data type.
        shape: MMA instruction shape [M, N, K].
        group_size: Number of MMA operations along the K dimension.
        transpose_b: Whether to transpose the B matrix.
    """

    comptime a_frag_size = num_matrix_reg[Self.shape[0], Self.shape[2]]()
    comptime b_frag_size = num_matrix_reg[Self.shape[2], Self.shape[1]]()
    comptime c_frag_size = num_matrix_reg[Self.shape[0], Self.shape[1]]()

    @staticmethod
    @always_inline
    def mma[
        swap_a_b: Bool = False
    ](
        a: TileTensor[_, _, address_space=AddressSpace.LOCAL, ...],
        b: TileTensor[_, _, address_space=AddressSpace.LOCAL, ...],
        c: TileTensor[mut=True, _, _, address_space=AddressSpace.LOCAL, ...],
    ):
        """Perform grouped MMA on TileTensor operands.

        Tiles down to individual MMA fragments so the compiler can
        prove static shapes, then calls the raw gpu_mma intrinsic
        directly.

        Parameters:
            swap_a_b: Whether to swap A and B operands.

        Args:
            a: A operand tile [num_m_mmas, group_size * a_frag_size].
            b: B operand tile [num_n_mmas, group_size * b_frag_size].
            c: Accumulator tile [num_m_mmas * num_n_mmas, c_frag_size],
                modified in-place.
        """
        comptime num_m_mmas = type_of(a).static_shape[0]
        comptime num_n_mmas = type_of(b).static_shape[0]

        comptime a_frag = Self.a_frag_size
        comptime b_frag = Self.b_frag_size
        comptime c_frag = Self.c_frag_size

        comptime for k in range(Self.group_size):
            var a_k = a.tile[num_m_mmas, a_frag](0, k).vectorize[1, a_frag]()
            var b_k = b.tile[num_n_mmas, b_frag](0, k).vectorize[1, b_frag]()

            comptime for m_mma in range(num_m_mmas):
                comptime for n_mma in range(num_n_mmas):
                    # col_major(M, N): m + n*M; row_major(N, M): m*M + n.
                    comptime c_idx = (
                        m_mma * num_m_mmas + n_mma
                    ) if swap_a_b else (m_mma + n_mma * num_m_mmas)
                    # Tile to a single [1, c_frag] fragment, then
                    # vectorize to [1, 1] — provably rank-2.
                    var c_frag_vec = c.tile[1, c_frag](c_idx, 0).vectorize[
                        1, c_frag
                    ]()
                    gpu_mma(
                        c_frag_vec[0, 0],
                        b_k[n_mma, 0],
                        a_k[m_mma, 0],
                        c_frag_vec[0, 0],
                    )

    @staticmethod
    @always_inline
    def load_b[
        swizzle: Optional[Swizzle] = None,
    ](
        warp_tile: TileTensor[
            Self.in_type, _, _, address_space=AddressSpace.SHARED, ...
        ],
        reg_tile: TileTensor[
            mut=True, Self.in_type, _, _, address_space=AddressSpace.LOCAL, ...
        ],
        k_group_idx: Int = 0,
    ):
        """Load B-matrix fragments from SMEM to registers.

        Distributes the warp tile across threads with optional swizzle,
        loading one MMA fragment per iteration. Handles both transposed
        and non-transposed B layouts via comptime dispatch.

        Parameters:
            swizzle: Optional swizzle for SMEM bank-conflict avoidance.

        Args:
            warp_tile: Source warp tile in shared memory.
            reg_tile: Destination register tile for MMA fragments.
            k_group_idx: K-dimension group index within the warp tile.
        """
        comptime mma_n = Self.shape[1]
        comptime mma_k = Self.shape[2]
        comptime simd_width = (num_matrix_reg[mma_k, mma_n]() * Self.group_size)

        comptime num_frags = type_of(reg_tile).static_shape[0]
        var reg_vec = reg_tile.vectorize[1, simd_width]()
        comptime assert type_of(reg_vec).flat_rank == 2

        comptime if Self.transpose_b:
            comptime for i in range(num_frags):
                var mma_tile = warp_tile.tile[mma_n, mma_k * Self.group_size](
                    i, k_group_idx
                )
                var dist = mma_tile.vectorize[1, simd_width]().distribute[
                    col_major[mma_n, WARP_SIZE // mma_n](),
                    swizzle=swizzle,
                ](lane_id())
                comptime assert type_of(dist).flat_rank == 2
                reg_vec[i, 0] = dist[0, 0]
        else:
            comptime for i in range(num_frags):
                var mma_tile = warp_tile.tile[mma_k * Self.group_size, mma_n](
                    k_group_idx, i
                )
                var dist = mma_tile.vectorize[simd_width, 1]().distribute[
                    row_major[WARP_SIZE // mma_n, mma_n](),
                    swizzle=swizzle,
                ](lane_id())
                comptime assert type_of(dist).flat_rank == 2
                reg_vec[i, 0] = dist[0, 0]

    @staticmethod
    @always_inline
    def load_a[
        swizzle: Optional[Swizzle] = None,
    ](
        warp_tile: TileTensor[
            Self.in_type, _, _, address_space=AddressSpace.SHARED, ...
        ],
        reg_tile: TileTensor[
            mut=True, Self.in_type, _, _, address_space=AddressSpace.LOCAL, ...
        ],
        k_group_idx: Int = 0,
    ):
        """Load A-matrix fragments from SMEM to registers.

        Distributes the warp tile across threads with optional swizzle,
        loading one MMA fragment per iteration. Always uses col_major
        thread distribution.

        Parameters:
            swizzle: Optional swizzle for SMEM access.

        Args:
            warp_tile: Source warp tile in shared memory.
            reg_tile: Destination register tile for MMA fragments.
            k_group_idx: K-dimension group index within the warp tile.
        """
        comptime mma_m = Self.shape[0]
        comptime mma_k = Self.shape[2]
        comptime simd_width = (num_matrix_reg[mma_m, mma_k]() * Self.group_size)
        comptime num_frags = type_of(reg_tile).static_shape[0]
        var reg_vec = reg_tile.vectorize[1, simd_width]()
        comptime assert type_of(reg_vec).flat_rank == 2

        comptime for i in range(num_frags):
            var mma_tile = warp_tile.tile[mma_m, mma_k * Self.group_size](
                i, k_group_idx
            )
            var dist = mma_tile.vectorize[1, simd_width]().distribute[
                col_major[mma_m, WARP_SIZE // mma_m](),
                swizzle=swizzle,
            ](lane_id())
            comptime assert type_of(dist).flat_rank == 2
            reg_vec[i, 0] = dist[0, 0]
