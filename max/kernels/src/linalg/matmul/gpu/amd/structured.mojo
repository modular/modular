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

from memory import LegacyUnsafePointer as UnsafePointer
from sys import align_of
from collections import OptionalReg
from gpu import WARP_SIZE
from gpu.mma import mma
from itertools import product
from layout import Layout, LayoutTensor
from layout.int_tuple import product as prod
from layout.layout import blocked_product
from layout.swizzle import Swizzle
from layout.tensor_core import num_matrix_reg, TensorCore
from linalg.structuring import SMemTileType, RegTileType
from sys._assembly import inlined_assembly
from utils import IndexList, StaticTuple


# NOTE: this struct might be a little overkill. may be consider simplifying this
@fieldwise_init
@register_passable("trivial")
struct ThreadRole(Stringable, Writable):
    var _value: UInt8

    alias PRODUCER = Self(0)
    alias CONSUMER = Self(1)
    alias PRODUCER_CONSUMER = Self(2)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @always_inline
    fn __is__(self, other: Self) -> Bool:
        return self == other

    @always_inline
    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @always_inline
    fn __str__(self) -> String:
        """Returns the string representation of this algorithm.

        Returns:
            String: A human-readable string representation of the algorithm.
        """
        if self is Self.PRODUCER:
            return "PRODUCER"
        elif self is Self.CONSUMER:
            return "CONSUMER"
        elif self is Self.PRODUCER_CONSUMER:
            return "PRODUCER_CONSUMER"
        else:
            return String("UNKNOWN_ROLE: ", self._value)

    @always_inline
    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write(String(self))


@parameter
@always_inline
fn pipeline_layout[layout: Layout, pipeline_stages: Int]() -> Layout:
    constrained[layout.rank() == 2]()
    return blocked_product(
        layout, Layout.row_major(1, pipeline_stages), coalesce_output=True
    )


# TODO: replace with Fabio's implementation
@register_passable("trivial")
struct SMemBuffer[
    dtype: DType,
    layout: Layout,
    pipeline_stages: Int,
    BM: Int,
    BN: Int,
    WM: Int,
    WN: Int,
]:

    """Manages shared memory and returns 2D tile slices of the buffer."""

    alias SMemTileType = SMemTileType[
        dtype, pipeline_layout[layout, pipeline_stages](), alignment=128
    ]

    alias BlockTileType = Self.SMemTileType.TileType[BM, BN]
    alias WarpTileType = Self.BlockTileType.TileType[WM, WN]

    var buffer: Self.SMemTileType

    @always_inline
    fn __init__(out self):
        constrained[
            layout.rank() == 2,
            "layout must be 2D",
        ]()

        constrained[
            prod(layout.shape[0]) == BM and prod(layout.shape[1]) == BN,
            (
                "shared memory rows must match block_rows and columns must"
                " match BN"
            ),
        ]()

        constrained[
            BM % WM == 0 and BN % WN == 0,
            "BM and BN must be a multiple of WM and WN",
        ]()

        self.buffer = Self.SMemTileType.stack_allocation()

    @always_inline
    fn get_tile(self, stage: Int) -> Self.BlockTileType:
        return self.buffer.tile[BM, BN](0, stage)


@register_passable("trivial")
struct AMDSharedMemoryBarrier[size: Int]:
    var __repr: StaticTuple[Int32, size]

    @always_inline
    fn initialize(ref [AddressSpace.SHARED, MutAnyOrigin]self):
        self.__repr = StaticTuple[Int32, size](fill=0)

    @always_inline
    fn value(ref [AddressSpace.SHARED]self) -> Int32:
        var sum: Int32 = 0

        @parameter
        for i in range(size):
            sum += self.__repr[i]
        return sum

    @always_inline
    fn increment(ref [AddressSpace.SHARED, MutAnyOrigin]self, warp_id: Int):
        var bar = rebind[
            UnsafePointer[
                Scalar[DType.int32], address_space = AddressSpace.SHARED
            ]
        ](Pointer(to=self.__repr))
        bar[warp_id] += 1

    @always_inline
    fn wait_until_equal_to(ref [AddressSpace.SHARED]self, v: Int32):
        while self.value() != v:
            inlined_assembly[
                "s_sleep 0", NoneType, constraints="", has_side_effect=True
            ]()

    @always_inline
    fn wait_until_greater_or_equal_to(ref [AddressSpace.SHARED]self, v: Int32):
        while self.value() < v:
            inlined_assembly[
                "s_sleep 0", NoneType, constraints="", has_side_effect=True
            ]()


@register_passable("trivial")
struct MMAConfig[
    InType: DType,
    OutType: DType,
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
]:
    alias mma = TensorCore[
        OutType,
        InType,
        mma_shape,
        transpose_b,
    ]()

    alias simd_width = simd_width_of[InType]()
    alias registers_per_thread_a = num_matrix_reg[mma_shape[0], mma_shape[2]]()
    alias registers_per_thread_b = num_matrix_reg[mma_shape[1], mma_shape[2]]()

    alias k_group_size_a = Self.simd_width // Self.registers_per_thread_a
    alias k_group_size_b = Self.simd_width // Self.registers_per_thread_b

    @staticmethod
    @always_inline
    fn adjusted_mma_k_shape_a() -> Int:
        return mma_shape[2] * Self.k_group_size_a

    @staticmethod
    @always_inline
    fn adjusted_mma_k_shape_b() -> Int:
        return mma_shape[2] * Self.k_group_size_b


@register_passable("trivial")
struct AmdTileOperator[
    InType: DType,
    OutType: DType,
    warp_block_layout_a: Layout,
    warp_block_layout_b: Layout,
    mma_shape: IndexList[3],
    swizzle: OptionalReg[Swizzle] = None,
    transpose_b: Bool = True,
]:
    """Manages tensor core operations for matrix multiplication on AMD GPUs.

    This operator handles loading matrix fragments from shared memory to registers
    and performing matrix multiply-accumulate operations using tensor cores.

    Parameters:
        InType: Input data type.
        OutType: Output data type.
        warp_block_layout_a: Layout for matrix A warp tiles.
        warp_block_layout_b: Layout for matrix B warp tiles.
        mma_shape: Shape of the MMA operation [M, N, K].
        swizzle: Optional swizzle pattern for memory access.
        transpose_b: Whether matrix B is transposed.

    Requirements:
        - warp_block_layout_a.shape[0] must be divisible by mma_shape[0]
        - warp_block_layout_b.shape[0] must be divisible by mma_shape[1]
        - warp_block_layout_a.shape[1] must be divisible by mma_shape[2]
        - warp_block_layout_b.shape[1] must be divisible by mma_shape[2]
        - The K dimension must align such that num_k_tiles is divisible by k_group_size
    """

    alias simd_width = simd_width_of[InType]()
    alias _type_alignment = align_of[SIMD[InType, Self.simd_width]]()

    # Create tensor core instance
    alias tensor_core = TensorCore[
        OutType,
        InType,
        mma_shape,
        transpose_b,
    ]()

    alias num_m_mmas = prod(warp_block_layout_a.shape[0]) // mma_shape[0]
    alias num_n_mmas = prod(warp_block_layout_b.shape[0]) // mma_shape[1]

    alias _out_frag_rows = Self.num_m_mmas * Self.num_n_mmas
    alias _out_frag_cols = Self.tensor_core.c_reg_type.size

    alias _out_layout = Layout.row_major(
        Self._out_frag_rows, Self._out_frag_cols
    )

    alias WK = prod(warp_block_layout_a.shape[1])
    alias num_k_tiles = Self.WK // mma_shape[2]

    alias _registers_per_thread_a = num_matrix_reg[mma_shape[0], mma_shape[2]]()
    alias _registers_per_thread_b = num_matrix_reg[mma_shape[1], mma_shape[2]]()
    alias k_group_size_a = Self.simd_width // Self._registers_per_thread_a
    alias k_group_size_b = Self.simd_width // Self._registers_per_thread_b

    alias _k_tiles_per_simd_a = Self.num_k_tiles // Self.k_group_size_a
    alias _k_tiles_per_simd_b = Self.num_k_tiles // Self.k_group_size_b

    # Total number of K tiles for MMA operations
    alias total_k_tiles = Self.num_k_tiles
    alias out_frag_size = mma_shape[0] * mma_shape[1] // WARP_SIZE

    alias _in_layout[
        num_mmas: Int,
        _k_tiles_per_simd: Int,
    ] = Layout.row_major(_k_tiles_per_simd * num_mmas, Self.simd_width)

    alias ARegTileType = RegTileType[
        InType, Self._in_layout[Self.num_m_mmas, Self._k_tiles_per_simd_a]
    ]

    alias BRegTileType = RegTileType[
        InType, Self._in_layout[Self.num_n_mmas, Self._k_tiles_per_simd_b]
    ]

    alias OutRegTileType = LayoutTensor[
        OutType,
        Self._out_layout,
        MutAnyOrigin,
        *_,
        alignment = Self._type_alignment,
        address_space = AddressSpace.LOCAL,
    ]

    alias OutRegTileFragmentType = Self.OutRegTileType.TileType[
        Self._out_frag_rows, Self._out_frag_cols
    ]

    # Register storage for matrix data
    var _a_reg_tile: Self.ARegTileType
    var _b_reg_tile: Self.BRegTileType
    var out_reg_tile: Self.OutRegTileType

    @always_inline
    fn __init__(out self):
        constrained[
            Self.simd_width >= Self._registers_per_thread_a
            and Self.simd_width >= Self._registers_per_thread_b,
            (
                "simd_width must be greater than or equal to required mma"
                " fragments size"
            ),
        ]()

        constrained[
            Self.num_k_tiles % Self.k_group_size_a == 0,
            "num_k_tiles must be divisible by k_group_size",
        ]()

        constrained[
            Self._k_tiles_per_simd_a == Self._k_tiles_per_simd_b,
            "k_tiles_per_simd must be equal for A and B",
        ]()

        self._a_reg_tile = Self.ARegTileType.stack_allocation()
        self._b_reg_tile = Self.BRegTileType.stack_allocation()

        # Initialize output accumulator to zero
        self.out_reg_tile = Self.OutRegTileType.stack_allocation().fill(0)

    @always_inline
    fn a_reg_tile(
        self, k_tile_idx: Int
    ) -> Self.ARegTileType.TileType[Self.num_m_mmas, Self.simd_width]:
        """Get A register tile for a specific K tile."""
        return self._a_reg_tile.tile[Self.num_m_mmas, Self.simd_width](
            k_tile_idx, 0
        )

    @always_inline
    fn b_reg_tile(
        self, k_tile_idx: Int
    ) -> Self.BRegTileType.TileType[Self.num_n_mmas, Self.simd_width]:
        """Get B register tile for a specific K tile."""
        return self._b_reg_tile.tile[Self.num_n_mmas, Self.simd_width](
            k_tile_idx, 0
        )

    @always_inline
    fn reset_accumulator(self):
        """Reset the accumulator to zero for a new tile computation."""
        _ = self.out_reg_tile.fill(0)

    # Helper aliases for K-tile indexing
    alias k_tile_group_index[
        k_tile_idx: Int
    ] = k_tile_idx // Self.k_group_size_a

    alias k_tile_fragment_index[
        k_tile_idx: Int
    ] = k_tile_idx % Self.k_group_size_a

    @always_inline
    fn load_tile_fragment[
        k_tile_idx: Int
    ](self, smem_tile_a: LayoutTensor, smem_tile_b: LayoutTensor):
        """Load fragments from shared memory to registers for a specific K tile.

        Parameters:
            k_tile_idx: K-tile index (0 to total_k_tiles-1).

        Args:
            smem_tile_a: Shared memory tile for matrix A.
            smem_tile_b: Shared memory tile for matrix B.
        """
        alias group_idx = Self.k_tile_group_index[k_tile_idx]
        alias fragment_idx = Self.k_tile_fragment_index[k_tile_idx]

        # Only load if this is the first fragment in the group
        # (tensor core loads k_group_size tiles at once)
        @parameter
        if fragment_idx == 0:
            Self.tensor_core.load_a[swizzle=swizzle](
                smem_tile_a,
                self._a_reg_tile.tile[Self.num_m_mmas, Self.simd_width](
                    group_idx, 0
                ).vectorize[1, Self.simd_width](),
                UInt(group_idx),
            )

            Self.tensor_core.load_b[swizzle=swizzle](
                smem_tile_b,
                self._b_reg_tile.tile[Self.num_n_mmas, Self.simd_width](
                    group_idx, 0
                ).vectorize[1, Self.simd_width](),
                UInt(group_idx),
            )

    @always_inline
    fn mma_compute[k_tile_idx: Int](self):
        """Perform matrix multiply-accumulate for a specific K tile.

        This method assumes fragments are already loaded via load_tile_fragment.

        Parameters:
            k_tile_idx: K-tile index (0 to total_k_tiles-1).
        """
        alias group_idx = Self.k_tile_group_index[k_tile_idx]
        alias fragment_idx = Self.k_tile_fragment_index[k_tile_idx]

        var c_slice = self.out_reg_tile

        # Get the tiles for this group
        var a_tile = self.a_reg_tile(group_idx)
        var b_tile = self.b_reg_tile(group_idx)

        # Perform MMA for this specific fragment within the group
        @parameter
        for mma_m_idx in range(Self.num_m_mmas):
            var a_fragment = a_tile.tile[1, Self._registers_per_thread_a](
                mma_m_idx, fragment_idx
            )

            @parameter
            for mma_n_idx in range(Self.num_n_mmas):
                var b_fragment = b_tile.tile[1, Self._registers_per_thread_b](
                    mma_n_idx, fragment_idx
                )

                # Storage scheme is column major for efficient write-back
                var c_fragment = c_slice.tile[1, Self._registers_per_thread_a](
                    mma_n_idx * Self.num_m_mmas + mma_m_idx, 0
                )

                mma(
                    c_fragment.vectorize[1, Self._registers_per_thread_a]()[
                        0, 0
                    ],
                    b_fragment.vectorize[1, Self._registers_per_thread_b]()[
                        0, 0
                    ],
                    a_fragment.vectorize[1, Self._registers_per_thread_a]()[
                        0, 0
                    ],
                    c_fragment.vectorize[1, Self._registers_per_thread_a]()[
                        0, 0
                    ],
                )
