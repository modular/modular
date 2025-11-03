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

from sys import align_of
from collections import OptionalReg
from gpu import WARP_SIZE, thread_idx
from layout import Layout, LayoutTensor
from layout.layout_tensor import (
    ThreadScope,
    copy_local_to_shared,
)
from layout.swizzle import Swizzle
from layout.tensor_core import TensorCore
from utils import IndexList
from layout.swizzle import Swizzle
from sys._assembly import inlined_assembly
from gpu.mma import mma
from itertools import product

from gpu import warp_id as get_warp_id
from layout.int_tuple import product as prod
from layout.tensor_core import num_matrix_reg
from layout.layout import blocked_product
from linalg.structuring import SMemArrayType, SMemTileType


# NOTE: this struct might be a little overkill. may be consider simplifying this
@fieldwise_init
@register_passable("trivial")
struct ThreadRole(Stringable, Writable):
    var _value: UInt8

    alias PRODUCER = Self(0)
    alias CONSUMER = Self(1)
    alias PRODUCER_CONSUMER = Self(2)

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

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

    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write(String(self))


@parameter
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

    fn get_tile(self, stage: Int) -> Self.BlockTileType:
        return self.buffer.tile[BM, BN](0, stage)


@register_passable("trivial")
struct AMDSharedMemoryBarrier[size: Int]:
    var __repr: SIMD[DType.int32, size]

    @always_inline
    fn initialize(ref [AddressSpace.SHARED, MutableAnyOrigin]self):
        self.__repr = 0

    @always_inline
    fn value(ref [AddressSpace.SHARED]self) -> Int32:
        return self.__repr.reduce_add()

    @always_inline
    fn increment(ref [AddressSpace.SHARED, MutableAnyOrigin]self, warp_id: Int):
        # Generate scalar access instructions, self.__repr[warp_id] += 1 is a vector instruction
        var scalar_elements = rebind[
            UnsafePointer[
                Scalar[DType.int32], address_space = AddressSpace.SHARED
            ]
        ](Pointer(to=self.__repr))
        scalar_elements[warp_id] += 1

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


struct RingBuffer[
    dtype: DType,
    layout: Layout,
    pipeline_stages: Int,
    B_rows: Int,  # BM for A, BN for B
    B_cols: Int,  # BK for both
    W_rows: Int,  # WM for A, WN for B
    W_cols: Int,  # WK for both
    consumer_warps: Int,
]:

    """Manages access to shared memory tiles using barriers based in shared memory.
    """

    # NOTE: smem can be 3D if pipelined, in that case we need a way to extract
    # the 2D tiles that's what this does

    # The barrier consists of integers. Producers and
    # consumers should wait if the barrier integer value does not fit into their expected range.
    # The rows of the barrier represent the warp tile desired. the columns consist of pipeline stages
    # each with consumer_warps slots. If pipeline_stages is > 1 then shared memory buffering is being used.
    # There are also consumer_warps slots for each pipeline stage, since each warp can write to the barrier
    # at the same time causing race conditions.

    alias block_warps = B_rows // W_rows

    alias BarrierArray = SMemArrayType[
        AMDSharedMemoryBarrier[consumer_warps],
        Self.block_warps * pipeline_stages,
    ]

    var barrier: Self.BarrierArray
    alias SmemBufferType = SMemBuffer[
        dtype, layout, pipeline_stages, B_rows, B_cols, W_rows, W_cols
    ]

    var smem_buffer: Self.SmemBufferType

    fn __init__(
        out self,
        smem_buffer: Self.SmemBufferType,
    ):
        self.smem_buffer = smem_buffer
        self.barrier = Self.BarrierArray.stack_allocation[alignment=32]()

        @parameter
        for i in range(type_of(self.barrier).size):
            self.barrier[i][].initialize()

    @always_inline
    fn _producer_wait(
        self, barrier: Self.BarrierArray, phase: Int, tile_idx: Int, stage: Int
    ):
        barrier[tile_idx * pipeline_stages + stage][].wait_until_equal_to(phase)

    @always_inline
    fn _consumer_wait(
        self, barrier: Self.BarrierArray, phase: Int, tile_idx: Int, stage: Int
    ):
        barrier[
            tile_idx * pipeline_stages + stage
        ][].wait_until_greater_or_equal_to(phase)

    @always_inline
    fn await_shared_memory_warp_tile[
        is_producer: Bool
    ](
        mut self,
        mut phase: Int,
        stage: Int,
        tile_idx: Int,
    ) -> Self.SmemBufferType.WarpTileType:
        @parameter
        if is_producer:
            self._producer_wait(self.barrier, phase, tile_idx, stage)
        else:
            self._consumer_wait(self.barrier, phase, tile_idx, stage)

        phase += 1 + Self.block_warps
        var staged_smem_tile = self.smem_buffer.get_tile(stage)
        return staged_smem_tile.tile[W_rows, W_cols](tile_idx, 0)

    @always_inline
    fn commit(mut self, stage: Int, tile_idx: Int):
        self.barrier[tile_idx * pipeline_stages + stage][].increment(
            Int(get_warp_id() % UInt(consumer_warps))
        )


struct AmdWarpBlockScatterGather[
    dtype: DType,
    thread_layout: Layout,
    warp_tile_layout: Layout,
    simd_width: Int,
    is_a: Bool,
    warp_rows: Int,
    warp_cols: Int,
    swizzle: OptionalReg[Swizzle] = None,
]:

    """
    Transports data from global -> register -> shared memory. Does this by warp tile
    each warp is responsible for moving one warp block of smem.
    """

    alias total_participating_threads = thread_layout.size()
    alias elements_loaded_per_thread = warp_tile_layout.size() // Self.total_participating_threads
    alias simd_loads_per_thread = Self.elements_loaded_per_thread // Self.simd_width

    alias LoadFragmentType = LayoutTensor[
        dtype,
        Layout.row_major(Self.simd_loads_per_thread, Self.simd_width),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]

    var fragment: Self.LoadFragmentType

    fn __init__(out self):
        constrained[
            Self.simd_loads_per_thread > 0,
            "simd_loads_per_thread must be greater than 0",
        ]()

        self.fragment = Self.LoadFragmentType.stack_allocation()

    @always_inline
    fn load_compute_tile(
        mut self,
        mut cache_manager: RingBuffer,
        mut phase: Int,
        gmem_tile: LayoutTensor[
            dtype,
            _,
            MutableAnyOrigin,
            address_space = AddressSpace.GLOBAL, **_,
        ],
        stage: Int,
        tile_idx: Int,
    ):
        var gmem_warp_tile = gmem_tile.tile[warp_rows, warp_cols](tile_idx, 0)

        load_from_gmem_to_reg[src_thread_layout=thread_layout](
            self.fragment.vectorize[1, Self.simd_width](),
            gmem_warp_tile.vectorize[1, Self.simd_width](),
        )

        var vectorized_fragment = self.fragment.vectorize[1, Self.simd_width]()

        var warp_tile = cache_manager.await_shared_memory_warp_tile[True](
            phase, stage, tile_idx
        )

        copy_local_to_shared[
            thread_layout=thread_layout,
            swizzle=swizzle,
            thread_scope = ThreadScope.WARP,
            row_major=True,
        ](
            warp_tile.vectorize[1, Self.simd_width](),
            vectorized_fragment,
        )

        inlined_assembly[
            "s_waitcnt lgkmcnt(0)",
            NoneType,
            constraints="",
            has_side_effect=True,
        ]()

        cache_manager.commit(stage, tile_idx)


fn load_from_gmem_to_reg[
    dtype: DType,
    src_thread_layout: Layout,
](
    dst: LayoutTensor[
        dtype,
        _,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL, **_,
    ],
    src: LayoutTensor[
        dtype,
        _,
        MutableAnyOrigin,
        address_space = AddressSpace.GLOBAL, **_,
    ],
):
    var worker_idx = thread_idx.x

    var src_fragments = src.distribute[src_thread_layout](worker_idx)
    alias M = src_fragments.shape[0]()
    alias N = src_fragments.shape[1]()

    constrained[
        src_fragments.layout.rank() == 2,
        "src_fragments must be rank 2.",
    ]()

    constrained[
        src_fragments.layout.all_dims_known(),
        "src_fragments must have known layout.",
    ]()

    @parameter
    for i in range(M):

        @parameter
        for j in range(N):
            alias idx = src_fragments.layout([i, j])
            alias dst_frag_idx = Layout.col_major(M, N)([i, j])
            dst[dst_frag_idx, 0] = rebind[
                SIMD[dst.dtype, dst.element_layout.size()]
            ](src_fragments[i, j])


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

    @parameter
    @staticmethod
    fn adjusted_mma_k_shape_a() -> Int:
        return mma_shape[2] * Self.k_group_size_a

    @parameter
    @staticmethod
    fn adjusted_mma_k_shape_b() -> Int:
        return mma_shape[2] * Self.k_group_size_b


# needs warp rows and cols to be passed in
struct AmdTileOperator[
    InType: DType,
    OutType: DType,
    mma_shape: IndexList[3],
    transpose_b: Bool, //,
    mma_config: type_of(MMAConfig[InType, OutType, mma_shape, transpose_b]),
    warp_block_layout_a: Layout,
    warp_block_layout_b: Layout,
    swizzle: OptionalReg[Swizzle] = None,
    tile_being_processed_per_warp: Int = 1,
]:
    alias type_alignment = align_of[SIMD[InType, mma_config.simd_width]]()

    alias num_m_mmas = prod(warp_block_layout_a.shape[0]) // mma_shape[0]
    alias num_n_mmas = prod(warp_block_layout_b.shape[0]) // mma_shape[1]

    alias out_frag_rows = Self.num_m_mmas * Self.num_n_mmas
    alias out_frag_cols = mma_config.mma.c_reg_type.size

    alias out_mma_fragment_layout = pipeline_layout[
        Layout.row_major(Self.out_frag_rows, Self.out_frag_cols),
        tile_being_processed_per_warp,
    ]()

    alias WK = prod(warp_block_layout_a.shape[1])
    alias num_k_tiles = Self.WK // mma_shape[2]

    alias k_tiles_per_simd_a = Self.num_k_tiles // mma_config.k_group_size_a
    alias k_tiles_per_simd_b = Self.num_k_tiles // mma_config.k_group_size_b

    alias in_layout[
        num_mmas: Int,
        k_tiles_per_simd: Int,
    ] = Layout.row_major(k_tiles_per_simd * num_mmas, mma_config.simd_width)

    alias InMmaFragmentTypeA = LayoutTensor[
        InType,
        Self.in_layout[Self.num_m_mmas, Self.k_tiles_per_simd_a],
        MutableAnyOrigin,
        *_,
        alignment = Self.type_alignment,
        address_space = AddressSpace.LOCAL,
    ]

    alias InMmaFragmentTypeB = LayoutTensor[
        InType,
        Self.in_layout[Self.num_n_mmas, Self.k_tiles_per_simd_b],
        MutableAnyOrigin,
        *_,
        alignment = Self.type_alignment,
        address_space = AddressSpace.LOCAL,
    ]

    alias OutMmaFragmentType = LayoutTensor[
        OutType,
        Self.out_mma_fragment_layout,
        MutableAnyOrigin,
        *_,
        alignment = Self.type_alignment,
        address_space = AddressSpace.LOCAL,
    ]

    alias OutMmaFragmentTileType = Self.OutMmaFragmentType.TileType[
        Self.out_frag_rows, Self.out_frag_cols
    ]

    var full_c_reg_tile: Self.OutMmaFragmentType
    var a_reg_tile: Self.InMmaFragmentTypeA
    var b_reg_tile: Self.InMmaFragmentTypeB

    fn __init__(out self):
        constrained[
            mma_config.simd_width >= mma_config.registers_per_thread_a
            and mma_config.simd_width >= mma_config.registers_per_thread_b,
            (
                "simd_width must be greater than or equal to required mma"
                " fragments size"
            ),
        ]()

        self.a_reg_tile = Self.InMmaFragmentTypeA.stack_allocation()
        self.b_reg_tile = Self.InMmaFragmentTypeB.stack_allocation()

        # BUG: this operation fails for some blocks see KERN-2090 for more details.
        self.full_c_reg_tile = Self.OutMmaFragmentType.stack_allocation().fill(
            0
        )

    fn get_c_reg_tile_slice(self, tile_idx: Int) -> Self.OutMmaFragmentTileType:
        return self.full_c_reg_tile.tile[
            Self.out_frag_rows, Self.out_frag_cols
        ](0, tile_idx)

    @always_inline
    fn mma[
        swap_a_b: Bool = True,
    ](
        mut self,
        mut cache_manager_a: RingBuffer,
        mut cache_manager_b: RingBuffer,
        mut phase_a: Int,
        mut phase_b: Int,
        stage: Int,
        smem_warp_tile_idx_a: Int,
        smem_warp_tile_idx_b: Int,
        linear_warp_idx: Int,  # tells us which set of registers to use
        block_tile_num: Int,
    ):
        var smem_tile_a = cache_manager_a.await_shared_memory_warp_tile[False](
            phase_a, stage, smem_warp_tile_idx_a
        )
        var smem_tile_b = cache_manager_b.await_shared_memory_warp_tile[False](
            phase_b, stage, smem_warp_tile_idx_b
        )

        @parameter
        for k_tile_idx in range(Self.k_tiles_per_simd_a):
            mma_config.mma.load_a[swizzle=swizzle](
                smem_tile_a,
                self.a_reg_tile.tile[Self.num_m_mmas, mma_config.simd_width](
                    k_tile_idx, 0
                ).vectorize[1, mma_config.simd_width](),
                UInt(k_tile_idx),
            )

        @parameter
        for k_tile_idx in range(Self.k_tiles_per_simd_b):
            mma_config.mma.load_b[swizzle=swizzle](
                smem_tile_b,
                self.b_reg_tile.tile[Self.num_n_mmas, mma_config.simd_width](
                    k_tile_idx, 0
                ).vectorize[1, mma_config.simd_width](),
                UInt(k_tile_idx),
            )

        # TODO: remove this constraint
        constrained[
            Self.k_tiles_per_simd_a == Self.k_tiles_per_simd_b,
            (
                "num_m_mmas * num_n_mmas must be equal to"
                " full_c_reg_tile.layout.size()"
            ),
        ]()

        var c_slice = self.get_c_reg_tile_slice(linear_warp_idx)

        # NOTE: maybe you can use TensorCoreKGrouo
        @parameter
        for k_tile_idx in range(Self.k_tiles_per_simd_a):
            var a_tile = self.a_reg_tile.tile[
                Self.num_m_mmas, mma_config.simd_width
            ](k_tile_idx, 0)
            var b_tile = self.b_reg_tile.tile[
                Self.num_n_mmas, mma_config.simd_width
            ](k_tile_idx, 0)

            @parameter
            for fragment, mma_m_idx in product(
                range(mma_config.k_group_size_a), range(Self.num_m_mmas)
            ):
                var a_fragment = a_tile.tile[
                    1, mma_config.registers_per_thread_a
                ](mma_m_idx, fragment)

                @parameter
                for mma_n_idx in range(Self.num_n_mmas):
                    var b_fragment = b_tile.tile[
                        1, mma_config.registers_per_thread_b
                    ](mma_n_idx, fragment)

                    # NOTE: this storage scheme is column major, because distribute needs it
                    # when writing back to global memory

                    var c_vector: SIMD[
                        OutType, mma_config.registers_per_thread_a
                    ]
                    var c_fragment = c_slice.tile[
                        1, mma_config.registers_per_thread_a
                    ](mma_n_idx * Self.num_m_mmas + mma_m_idx, 0)

                    # required because of BUG: where fill fails for some blocks
                    if (
                        k_tile_idx == 0
                        and fragment == 0
                        and block_tile_num == 0
                    ):
                        c_vector = SIMD[
                            OutType, mma_config.registers_per_thread_a
                        ](0)
                    else:
                        c_vector = rebind[type_of(c_vector)](
                            c_fragment.vectorize[
                                1, mma_config.registers_per_thread_a
                            ]()[0, 0]
                        )

                    mma(
                        c_fragment.vectorize[
                            1, mma_config.registers_per_thread_a
                        ]()[0, 0],
                        b_fragment.vectorize[
                            1, mma_config.registers_per_thread_b
                        ]()[0, 0],
                        a_fragment.vectorize[
                            1, mma_config.registers_per_thread_a
                        ]()[0, 0],
                        c_vector,
                    )

        cache_manager_a.commit(stage, smem_warp_tile_idx_a)
        cache_manager_b.commit(stage, smem_warp_tile_idx_b)
