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


# NOTE: this struct might be a little overkill. may be consider simplifying this
@fieldwise_init
struct ThreadRole(Copyable, ImplicitlyCopyable, Movable, Stringable, Writable):
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
struct SharedMemoryBuffer[
    BufferType: DType,
    smem_tile_layout: Layout,
    pipeline_stages: Int,
    block_rows: Int,
    block_cols: Int,
    warp_rows: Int,
    warp_cols: Int,
](ImplicitlyCopyable):

    """Manages shared memory and returns 2D tile slices of the buffer."""

    alias SmemTensorType = LayoutTensor[
        BufferType,
        pipeline_layout[smem_tile_layout, pipeline_stages](),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    alias SmemTileType = Self.SmemTensorType.TileType[block_rows, block_cols]
    alias WarpTileType = Self.SmemTileType.TileType[warp_rows, warp_cols]

    var buffer: Self.SmemTensorType

    fn __init__(out self):
        constrained[
            smem_tile_layout.rank() == 2,
            "layout must be 2D",
        ]()

        constrained[
            prod(smem_tile_layout.shape[0]) == block_rows
            and prod(smem_tile_layout.shape[1]) == block_cols,
            (
                "shared memory rows must match block_rows and columns must"
                " match block_cols"
            ),
        ]()

        constrained[
            block_rows % warp_rows == 0 and block_cols % warp_cols == 0,
            (
                "block_rows and block_cols must be a multiple of warp_rows and"
                " warp_cols"
            ),
        ]()

        self.buffer = Self.SmemTensorType.stack_allocation()

    fn get_tile(self, stage: Int) -> Self.SmemTileType:
        return self.buffer.tile[block_rows, block_cols](0, stage)


struct RingBuffer[
    pipeline_stages: Int,
    a_buffer_layout: Layout,
    b_buffer_layout: Layout,
    SmemBufferDataType: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    WK: Int, //,
    SmemBufferTypeA: type_of(
        SharedMemoryBuffer[
            SmemBufferDataType, a_buffer_layout, pipeline_stages, BM, BK, WM, WK
        ]
    ),
    SmemBufferTypeB: type_of(
        SharedMemoryBuffer[
            SmemBufferDataType, b_buffer_layout, pipeline_stages, BN, BK, WN, WK
        ]
    ),
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

    alias warps_per_block_m = BM // WM
    alias warps_per_block_n = BN // WN

    alias BarrierTensorType[warp_tile_count: Int] = LayoutTensor[
        DType.int32,
        Layout.row_major(warp_tile_count, consumer_warps * pipeline_stages),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=32,
    ]

    var barrier_a: Self.BarrierTensorType[Self.warps_per_block_m]
    var barrier_b: Self.BarrierTensorType[Self.warps_per_block_n]

    alias SharedMemoryBufferType[is_a: Bool] = SharedMemoryBuffer[
        SmemBufferDataType,
        a_buffer_layout if is_a else b_buffer_layout,
        pipeline_stages,
        BM if is_a else BN,
        BK,
        WM if is_a else WN,
        WK,
    ]

    var smem_buffer_a: SmemBufferTypeA
    var smem_buffer_b: SmemBufferTypeB

    fn __init__(
        out self,
        smem_buffer_a: SmemBufferTypeA,
        smem_buffer_b: SmemBufferTypeB,
    ):
        self.smem_buffer_a = smem_buffer_a
        self.smem_buffer_b = smem_buffer_b

        self.barrier_a = (
            Self.BarrierTensorType[Self.warps_per_block_m]
            .stack_allocation()
            .fill(0)
        )
        self.barrier_b = (
            Self.BarrierTensorType[Self.warps_per_block_n]
            .stack_allocation()
            .fill(0)
        )

    @always_inline
    fn _get_barrier[
        is_a: Bool
    ](
        self,
        out bar: Self.BarrierTensorType[
            Self.warps_per_block_m if is_a else Self.warps_per_block_n
        ],
    ):
        @parameter
        if is_a:
            return rebind[type_of(bar)](self.barrier_a)
        else:
            return rebind[type_of(bar)](self.barrier_b)

    @always_inline
    fn _get_current_barrier_value[
        is_a: Bool
    ](self, tile_idx: Int, stage: Int) -> Int:
        var bar = self._get_barrier[is_a]()
        var warp_vector = bar.vectorize[1, consumer_warps]()[tile_idx, stage]
        return Int(warp_vector.reduce_add())

    @always_inline
    fn _producer_wait[is_a: Bool](self, phase: Int, tile_idx: Int, stage: Int):
        while self._get_current_barrier_value[is_a](tile_idx, stage) != phase:
            inlined_assembly[
                "s_sleep 0", NoneType, constraints="", has_side_effect=True
            ]()

    @always_inline
    fn _consumer_wait[is_a: Bool](self, phase: Int, tile_idx: Int, stage: Int):
        while self._get_current_barrier_value[is_a](tile_idx, stage) < phase:
            inlined_assembly[
                "s_sleep 0", NoneType, constraints="", has_side_effect=True
            ]()

    @always_inline
    fn _get_shared_memory_tile[
        is_a: Bool
    ](
        self,
        stage: Int,
        out smem_tile: Self.SharedMemoryBufferType[is_a].SmemTileType,
    ):
        @parameter
        if is_a:
            return rebind[type_of(smem_tile)](
                self.smem_buffer_a.get_tile(stage)
            )
        else:
            return rebind[type_of(smem_tile)](
                self.smem_buffer_b.get_tile(stage)
            )

    @always_inline
    fn _get_shared_memory_warp_tile[
        is_a: Bool, is_producer: Bool
    ](
        mut self,
        mut phase: Int,
        tile_idx: Int,
        stage: Int,
        phase_inc: Int,
        out smem_warp_tile: Self.SharedMemoryBufferType[is_a].WarpTileType,
    ):
        @parameter
        if is_producer:
            self._producer_wait[is_a](phase, tile_idx, stage)
        else:
            self._consumer_wait[is_a](phase, tile_idx, stage)

        phase += phase_inc
        var staged_smem_tile = self._get_shared_memory_tile[is_a](stage)
        return rebind[type_of(smem_warp_tile)](
            staged_smem_tile.tile[WM if is_a else WN, WK](tile_idx, 0)
        )

    @always_inline
    fn await_shared_memory_warp_tile[
        is_a: Bool, is_producer: Bool
    ](
        mut self, mut phase: Int, stage: Int, tile_idx: Int
    ) -> Self.SharedMemoryBufferType[is_a].WarpTileType:
        var phase_inc: Int

        @parameter
        if is_a:
            phase_inc = 1 + Self.warps_per_block_n
        else:
            phase_inc = 1 + Self.warps_per_block_m

        return self._get_shared_memory_warp_tile[is_a, is_producer](
            phase, tile_idx, stage, phase_inc
        )

    @always_inline
    fn commit[is_a: Bool](mut self, stage: Int, tile_idx: Int):
        var bar = self._get_barrier[is_a]()
        var bar_tile = bar.tile[1, consumer_warps](tile_idx, stage)
        var warp_id = get_warp_id() % UInt(consumer_warps)
        bar_tile[1, warp_id] = bar_tile[1, warp_id] + 1


struct AmdWarpBlockScatterGather[
    SmemType: DType,
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
        SmemType,
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
    fn load_compute_tile[
        GmemType: DType,
        GmemLayout: Layout,
    ](
        mut self,
        mut cache_manager: RingBuffer,
        mut phase: Int,
        gmem_tile: LayoutTensor[
            GmemType,
            GmemLayout,
            MutableAnyOrigin,
            address_space = AddressSpace.GLOBAL,
        ],
        stage: Int,
        tile_idx: Int,
    ):
        var gmem_warp_tile = gmem_tile.tile[warp_rows, warp_cols](tile_idx, 0)

        load_from_gmem_to_reg[
            simd_width = Self.simd_width,
            src_thread_layout=thread_layout,
        ](
            self.fragment.vectorize[1, Self.simd_width](),
            gmem_warp_tile.vectorize[1, Self.simd_width](),
        )

        var vectorized_fragment = self.fragment.vectorize[1, Self.simd_width]()

        var warp_tile = cache_manager.await_shared_memory_warp_tile[is_a, True](
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

        cache_manager.commit[is_a](stage, tile_idx)


fn load_from_gmem_to_reg[
    simd_width: Int,
    src_thread_layout: Layout,
](dst: LayoutTensor, src: LayoutTensor):
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
        address_space = AddressSpace.LOCAL,
        alignment = Self.type_alignment,
    ]

    alias InMmaFragmentTypeB = LayoutTensor[
        InType,
        Self.in_layout[Self.num_n_mmas, Self.k_tiles_per_simd_b],
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        alignment = Self.type_alignment,
    ]

    alias OutMmaFragmentType = LayoutTensor[
        OutType,
        Self.out_mma_fragment_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        alignment = Self.type_alignment,
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
        mut cache_manager: RingBuffer,
        mut phase_a: Int,
        mut phase_b: Int,
        stage: Int,
        smem_warp_tile_idx_a: Int,
        smem_warp_tile_idx_b: Int,
        linear_warp_idx: Int,  # tells us which set of registers to use
        block_tile_num: Int,
    ):
        var smem_tile_a = cache_manager.await_shared_memory_warp_tile[
            True, False
        ](phase_a, stage, smem_warp_tile_idx_a)
        var smem_tile_b = cache_manager.await_shared_memory_warp_tile[
            False, False
        ](phase_b, stage, smem_warp_tile_idx_b)

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

        cache_manager.commit[True](stage, smem_warp_tile_idx_a)
        cache_manager.commit[False](stage, smem_warp_tile_idx_b)
