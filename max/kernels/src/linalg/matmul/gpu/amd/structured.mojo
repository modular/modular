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
from gpu import (
    WARP_SIZE,
    thread_idx,
)
from gpu.memory import AddressSpace
from layout import IntTuple, Layout, LayoutTensor
from layout.layout_tensor import (
    ThreadScope,
    copy_local_to_shared,
)
from layout.swizzle import Swizzle
from layout.tensor_core import TensorCore
from memory import stack_allocation
from utils import IndexList, StaticTuple
from layout.swizzle import Swizzle
from sys._assembly import inlined_assembly
from gpu.mma import mma
from sys import size_of, is_amd_gpu, _RegisterPackType
from itertools import product

from layout.layout_tensor import _get_worker_idx


@always_inline
fn wait_vmcount[inflight_transfers: Int]():
    """Waits until vector memory count is less than or equal to inflight_transfers. VM_CNT gets decremented when data is returned
    to VGPRS, or memory writes have completed.

    Parameters:
        inflight_transfers: The maximum number of transfers that should remain after the wait.
    """

    constrained[is_amd_gpu(), "s_waitcnt is amd specific"]()

    alias asm = "s_waitcnt vmcnt(" + String(inflight_transfers) + ")\n\t"

    inlined_assembly[
        asm,
        NoneType,
        constraints="",
        has_side_effect=True,
    ]()


@always_inline
fn to_vgpr(register: UInt32) -> UInt32:
    """Moves data from any register to a VGPR.

    Args:
        register: The register to move to a VGPR.
    """

    constrained[is_amd_gpu(), "VGPRs are amd specific"]()

    return inlined_assembly[
        "v_mov_b32 $0, $1\n\t",
        UInt32,
        constraints="=v,r",
        has_side_effect=True,
    ](register)


@always_inline
fn to_consecutive_sgprs(
    vgpr_one: UInt32, vgpr_two: UInt32
) -> SIMD[DType.uint32, 2]:
    """Moves data from two VGPRS to two consecutive SGPRS, e.g. s[0:1], s[2:3].
    This move is performed by the first thread in a warp, and the registers are shared
    by all threads in the warp.

    Args:
        vgpr_one: The register to move to a SGPR n.
        vgpr_two: The register to move to a SGPR n+1.
    """

    constrained[is_amd_gpu(), "SGPRs are amd specific"]()

    # nop is very important
    # 4.5. Manually Inserted Wait States (NOPs)
    # https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf

    var sgpr_pack = inlined_assembly[
        "v_readfirstlane_b32 $0, $2\n\t"
        + "v_readfirstlane_b32 $1, $3\n\t"
        + "s_nop 4\n\t",
        _RegisterPackType[
            SIMD[DType.uint32, 1],
            SIMD[DType.uint32, 1],
        ],
        constraints="=s,=s,v,v",
        has_side_effect=True,
    ](vgpr_one, vgpr_two)

    return SIMD[DType.uint32, 2](sgpr_pack[0], sgpr_pack[1])


@always_inline
fn global_load_dword[
    dtype: DType, width: Int
](sgpr_address: SIMD[DType.uint32, 2], vgpr_offset: UInt32) -> SIMD[
    dtype, width
]:
    """Loads 8 or 16 bytes from global memory to a SIMD vector. Each load is asynchronous and needs
    to be waited on with wait_vmcount.

    Parameters:
        dtype: The data type of global memory.
        width: The width of the SIMD vector to load.

    Args:
        sgpr_address: Two consecutive SGPRs that contain the address of the global memory ptr.
        vgpr_offset: The offset of the address that this thread will load from.
    """

    alias bytes_per_load = width * size_of[dtype]()

    constrained[is_amd_gpu(), "global_load_dword is amd specific"]()
    constrained[bytes_per_load in (8, 16), "bytes_per_load must be 8 or 16"]()

    # we load the data and supply different offsets per load
    alias load_asm = "global_load_dwordx4 $0, $1, $2 offset:0\n\t" if bytes_per_load == 16 else "global_load_dwordx2 $0, $1, $2 offset:0\n\t"

    return inlined_assembly[
        load_asm,
        SIMD[dtype, width],
        constraints="=&v,v,s,~{memory}",
        has_side_effect=True,
    ](vgpr_offset, sgpr_address)


@parameter
fn _get_batch_size[total_loads: Int]() -> Int:
    if total_loads % 4 == 0:
        return 4
    elif total_loads % 2 == 0:
        return 2
    else:
        return 1


fn batched_copy_dram_to_local[
    src_thread_layout: Layout,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](dst: LayoutTensor, src: LayoutTensor):
    """Copies data from global memory (DRAM) to registers (LOCAL) in a batched manner.

    This function utilizes global_load_dwordx4/global_load_dwordx2 to load data from global memory to local memory in a batched manner.

    Parameters:
        src_thread_layout: The layout used to distribute the threads for coalesced loads.
        num_threads: Total number of threads participating in the copy
            operation. Defaults to the size of thread_layout.
        thread_scope: Defines whether operations are performed at `BLOCK` or
            `WARP` level. `BLOCK` scope involves all threads in a thread block,
            while `WARP` scope restricts operations to threads within the same
            warp. Defaults to `ThreadScope.BLOCK`.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor in register memory (LOCAL address space).
        src: The source tensor in global memory (DRAM) to be copied.

    Notes:

    - This function is particularly used for warp specialization.
    """
    constrained[is_amd_gpu(), "This function is only supported on AMD GPUs."]()

    alias num_busy_threads = src_thread_layout.size()
    var worker_idx = _get_worker_idx[thread_scope, block_dim_count]()

    @parameter
    if num_threads > num_busy_threads:
        if worker_idx >= UInt(num_busy_threads):
            return

    var src_fragments = src.distribute[src_thread_layout](worker_idx)

    alias M = src_fragments.shape[0]()
    alias N = src_fragments.shape[1]()

    alias total_loads = M * N
    alias batch_size = _get_batch_size[total_loads]()

    var src_frag_offset = UInt32(
        src_fragments.distance(src.ptr) * size_of[src.dtype]()
    )

    alias batches = total_loads // batch_size

    var ptr_addr = Int(src.ptr)
    var address = bitcast[DType.uint32, 2](SIMD[DType.uint64, 1](ptr_addr))
    var addr_low = address[0]  # Lower 32 bits
    var addr_high = address[1]  # Upper 32 bits

    var vgpr_low = to_vgpr(addr_low)
    var vgpr_high = to_vgpr(addr_high)
    var sgpr_address = to_consecutive_sgprs(vgpr_low, vgpr_high)

    @parameter
    for batch in range(batches):

        @parameter
        for ld_num in range(batch_size):
            alias ld = batch * batch_size + ld_num
            alias row = ld // N
            alias col = ld % N

            alias src_frag_idx_bytes = UInt32(
                src_fragments.layout([row, col])
            ) * size_of[src.dtype]()

            var offset = src_frag_offset + src_frag_idx_bytes
            dst[row, col] = global_load_dword[
                dst.dtype, dst.element_layout.size()
            ](sgpr_address, offset)

        wait_vmcount[0]()


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


@fieldwise_init
struct SharedMemoryBuffer[
    BufferType: DType,
    layout: Layout,
    pipeline_stages: Int,
    warp_rows: Int,
    warp_cols: Int,
](ImplicitlyCopyable):
    alias SmemTileType = LayoutTensor[
        BufferType,
        layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    alias smem_2D_tile_layout[tensor_layout: Layout] = Layout(
        IntTuple(tensor_layout.shape[1], tensor_layout.shape[2]),
        IntTuple(tensor_layout.stride[1], tensor_layout.stride[2]),
    ) if tensor_layout.rank() != 2 else tensor_layout

    alias SmemTileType2D = LayoutTensor[
        BufferType,
        Self.smem_2D_tile_layout[layout],
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    alias WarpTileType = Self.SmemTileType2D.TileType[warp_rows, warp_cols]

    var buffer: Self.SmemTileType

    fn __init__(out self):
        constrained[
            (pipeline_stages == 1 and layout.rank() == 2)
            or (pipeline_stages == layout.shape[0].value()),
            (
                "shared memory buffers must either be 2 dimensional with 1"
                " pipeline"
            ),
        ]()
        self.buffer = Self.SmemTileType.stack_allocation()

    fn get_tile(self, stage: Int) -> Self.SmemTileType2D:
        var buffer_ptr = self.buffer.ptr

        alias size = layout.size()

        var offset = size * stage
        buffer_ptr += offset

        return Self.SmemTileType2D(buffer_ptr)


struct RingBuffer[
    pipeline_stages: Int,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    TileTypeA: DType,
    TileTypeB: DType,
    WM: Int,
    WN: Int,
    WK: Int,
    warps_per_block_m: Int,
    warps_per_block_n: Int,
]:

    """Simulates A TMA barriers on AMD. This pipeline can be used for both multistage
    (double buffering) where all threads are producers and consumers, and for warp
    specialized pipelines."""

    # NOTE: smem can be 3D if pipelined, in that case we need a way to extract
    # the 2D tiles that's what this does

    alias BarrierTensorType[warp_count: Int] = LayoutTensor[
        DType.int32,
        Layout.row_major(warp_count, pipeline_stages),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=32,
    ]

    var barrier_a: Self.BarrierTensorType[warps_per_block_m]
    var barrier_b: Self.BarrierTensorType[warps_per_block_n]

    alias SharedMemoryBufferType[is_a: Bool] = SharedMemoryBuffer[
        TileTypeA if is_a else TileTypeB,
        a_tile_layout if is_a else b_tile_layout,
        pipeline_stages,
        WM if is_a else WN,
        WK,
    ]

    var smem_tile_a: Self.SharedMemoryBufferType[True]
    var smem_tile_b: Self.SharedMemoryBufferType[False]

    fn __init__(
        out self,
        smem_tile_a: SharedMemoryBuffer[
            TileTypeA, a_tile_layout, pipeline_stages, WM, WK
        ],
        smem_tile_b: SharedMemoryBuffer[
            TileTypeB, b_tile_layout, pipeline_stages, WN, WK
        ],
    ):
        self.smem_tile_a = smem_tile_a
        self.smem_tile_b = smem_tile_b

        self.barrier_a = (
            Self.BarrierTensorType[warps_per_block_m].stack_allocation().fill(0)
        )
        self.barrier_b = (
            Self.BarrierTensorType[warps_per_block_n].stack_allocation().fill(0)
        )

    @always_inline
    fn _get_barrier[
        is_a: Bool
    ](
        self,
        out bar: Self.BarrierTensorType[
            warps_per_block_m if is_a else warps_per_block_n
        ],
    ):
        @parameter
        if is_a:
            return rebind[__type_of(bar)](self.barrier_a)
        else:
            return rebind[__type_of(bar)](self.barrier_b)

    @always_inline
    fn _producer_wait[is_a: Bool](self, phase: Int, tile_idx: Int, stage: Int):
        while self._get_barrier[is_a]()[tile_idx, stage] != phase:
            inlined_assembly[
                "s_sleep 0", NoneType, constraints="", has_side_effect=True
            ]()

    @always_inline
    fn _consumer_wait[is_a: Bool](self, phase: Int, tile_idx: Int, stage: Int):
        while self._get_barrier[is_a]()[tile_idx, stage] < phase:
            inlined_assembly[
                "s_sleep 0", NoneType, constraints="", has_side_effect=True
            ]()

    @always_inline
    fn _get_shared_memory_tile[
        is_a: Bool
    ](
        self,
        stage: Int,
        out smem_tile: Self.SharedMemoryBufferType[is_a].SmemTileType2D,
    ):
        @parameter
        if is_a:
            return rebind[__type_of(smem_tile)](
                self.smem_tile_a.get_tile(stage)
            )
        else:
            return rebind[__type_of(smem_tile)](
                self.smem_tile_b.get_tile(stage)
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
        return rebind[__type_of(smem_warp_tile)](
            staged_smem_tile.tile[WM if is_a else WN, WK](tile_idx, 0)
        )

    @always_inline
    fn put_tile[
        is_a: Bool
    ](
        mut self,
        mut phase: Int,
        stage: Int,
        tile_idx: Int,
    ) -> Self.SharedMemoryBufferType[is_a].WarpTileType:
        var phase_inc: Int

        @parameter
        if is_a:
            phase_inc = 1 + warps_per_block_n
        else:
            phase_inc = 1 + warps_per_block_m

        return self._get_shared_memory_warp_tile[is_a, True](
            phase, tile_idx, stage, phase_inc
        )

    @always_inline
    fn get_tile[
        is_a: Bool
    ](
        mut self, mut phase: Int, stage: Int, tile_idx: Int
    ) -> Self.SharedMemoryBufferType[is_a].WarpTileType:
        var phase_inc: Int

        @parameter
        if is_a:
            phase_inc = warps_per_block_m
        else:
            phase_inc = warps_per_block_n

        return self._get_shared_memory_warp_tile[is_a, False](
            phase, tile_idx, stage, phase_inc
        )

    @always_inline
    fn commit[is_a: Bool](mut self, stage: Int, tile_idx: Int):
        var bar: Self.BarrierTensorType

        # NOTE this may need atomics

        @parameter
        if is_a:
            self.barrier_a[tile_idx, stage] = (
                self.barrier_a[tile_idx, stage] + 1
            )
        else:
            self.barrier_b[tile_idx, stage] = (
                self.barrier_b[tile_idx, stage] + 1
            )


struct AmdWarpBlockScatterGather[
    SmemType: DType,
    thread_layout: Layout,
    smem_layout: Layout,
    simd_width: Int,
    is_a: Bool,
    warp_rows: Int,
    warp_cols: Int,
    swizzle: OptionalReg[Swizzle] = None,
]:

    """
    Transports data from global -> register -> shared memory. Does this by warp tile
    each warp is resposible for moving one warp block of smem.
    """

    alias SmemTensorType = LayoutTensor[
        SmemType,
        smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]

    alias LoadFragmentShape = Self.SmemTensorType.DistributeType[
        thread_layout
    ].layout.shape
    alias LoadFragmentLayout = Layout.row_major(
        Self.LoadFragmentShape[0].value(), Self.LoadFragmentShape[1].value()
    )
    alias LoadFragmentType = LayoutTensor[
        SmemType,
        Self.LoadFragmentLayout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]

    var fragment: Self.LoadFragmentType

    fn __init__(out self):
        constrained[
            len(Self.LoadFragmentShape) == 2,
            (
                "LoadFragmentShape must be equal to"
                " SmemTensorType.DistributeType[thread_layout].layout.shape"
            ),
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
        alias load_length = 4 if size_of[
            GmemType
        ]() == 2 and Self.simd_width == 8 else 2
        var gmem_warp_tile = gmem_tile.tile[warp_rows, warp_cols](tile_idx, 0)

        load_from_gmem_to_reg_no_waitcnt[
            src_thread_layout=thread_layout,
            load_length=load_length,
        ](
            self.fragment.vectorize[1, Self.simd_width](),
            gmem_warp_tile.vectorize[1, Self.simd_width](),
        )

        var vectorized_fragment = self.fragment.vectorize[1, Self.simd_width]()

        inlined_assembly[
            "s_waitcnt vmcnt(0)",
            NoneType,
            constraints="",
            has_side_effect=True,
        ]()

        var warp_tile = cache_manager.put_tile[is_a](phase, stage, tile_idx)

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


fn load_from_gmem_to_reg_no_waitcnt[
    src_thread_layout: Layout,
    num_threads: Int = src_thread_layout.size(),
    load_length: Int = 4,
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

    # NOTE: maybe instructions need to be immediate like here:
    # https://github.com/huggingface/hf-rocm-kernels/blob/23b89ef66f1f0d170dcd141af84b12aec458ebc4/csrc/op_src/skinny_gemm/utils.cuh#L98-L125

    alias asm = "global_load_dwordx4 $0, $1, off offset:0\n\t" if load_length == 4 else "global_load_dwordx2 $0, $1, off offset:0\n\t"

    @parameter
    for i in range(M):

        @parameter
        for j in range(N):
            alias idx = src_fragments.layout([i, j])

            dst[i, j] = inlined_assembly[
                asm,
                SIMD[dst.dtype, dst.element_layout.size()],
                constraints="=v,v,~{memory}",
                has_side_effect=True,
            ](src_fragments.ptr + idx)


# needs warp rows and cols to be passed in
struct AmdTileOperator[
    InType: DType,
    OutType: DType,
    warp_block_layout_a: Layout,
    warp_block_layout_b: Layout,
    mma_shape: IndexList[3],
    transpose_b: Bool,
    swizzle: OptionalReg[Swizzle] = None,
    simd_width: Int = 1,
    warps_being_processed: Int = 1,
]:
    alias type_alignment = align_of[SIMD[InType, Self.simd_width]]()
    alias tensor_core_mma = TensorCore[
        OutType,
        InType,
        mma_shape,
        transpose_b,
    ]()

    alias num_m_mmas = warp_block_layout_a.shape[0].value() // mma_shape[0]
    alias num_n_mmas = warp_block_layout_b.shape[0].value() // mma_shape[1]

    alias full_out_mma_fragment_layout = Layout.row_major(
        warps_being_processed,
        Self.num_m_mmas * Self.num_n_mmas,
        Self.tensor_core_mma.c_reg_type.size,
    )

    alias out_mma_fragment_layout = Layout.row_major(
        Self.num_m_mmas * Self.num_n_mmas, Self.tensor_core_mma.c_reg_type.size
    )

    alias a_matrix_size = mma_shape[0] * mma_shape[2]
    alias b_matrix_size = mma_shape[1] * mma_shape[2]

    alias register_count_a = Self.a_matrix_size // WARP_SIZE
    alias register_count_b = Self.b_matrix_size // WARP_SIZE

    alias WK = warp_block_layout_a.shape[1].value()
    alias num_k_tiles = Self.WK // mma_shape[2]

    alias fragments_per_simd_a = Self.simd_width // Self.register_count_a
    alias fragments_per_simd_b = Self.simd_width // Self.register_count_b

    alias k_tiles_per_simd_a = Self.num_k_tiles // Self.fragments_per_simd_a
    alias k_tiles_per_simd_b = Self.num_k_tiles // Self.fragments_per_simd_b

    alias in_layout[
        num_mmas: Int,
        k_tiles_per_simd: Int,
    ] = Layout.row_major(k_tiles_per_simd * num_mmas, Self.simd_width)

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

    alias FullOutMmaFragmentType = LayoutTensor[
        OutType,
        Self.full_out_mma_fragment_layout,
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

    var full_c_reg_tile: Self.FullOutMmaFragmentType
    var a_reg_tile: Self.InMmaFragmentTypeA
    var b_reg_tile: Self.InMmaFragmentTypeB

    fn __init__(out self):
        constrained[
            Self.WK % Self.simd_width == 0, "WK must be divisible by simd_width"
        ]()
        constrained[
            Self.simd_width >= Self.register_count_a
            and Self.simd_width >= Self.register_count_b,
            (
                "simd_width must be greater than or equal to required mma"
                " fragments size"
            ),
        ]()

        self.a_reg_tile = Self.InMmaFragmentTypeA.stack_allocation()
        self.b_reg_tile = Self.InMmaFragmentTypeB.stack_allocation()
        self.full_c_reg_tile = (
            Self.FullOutMmaFragmentType.stack_allocation().fill(0)
        )

    fn get_c_reg_tile_slice(self, warp_idx: Int) -> Self.OutMmaFragmentType:
        alias size = Self.out_mma_fragment_layout.size()
        return Self.OutMmaFragmentType(
            self.full_c_reg_tile.ptr + warp_idx * size
        )

    @always_inline
    fn mma[
        swap_a_b: Bool = True,
    ](
        mut self,
        mut cache_manager: RingBuffer,
        mut phase_a: Int,
        mut phase_b: Int,
        stage: Int,
        tile_idx_a: Int,
        tile_idx_b: Int,
        linear_warp_idx: Int,
    ):
        var smem_tile_a = cache_manager.get_tile[True](
            phase_a, stage, tile_idx_a
        )
        var smem_tile_b = cache_manager.get_tile[False](
            phase_b, stage, tile_idx_b
        )

        @parameter
        for k_tile_idx in range(Self.k_tiles_per_simd_a):
            Self.tensor_core_mma.load_a[swizzle=swizzle](
                smem_tile_a,
                self.a_reg_tile.tile[Self.num_m_mmas, Self.simd_width](
                    k_tile_idx, 0
                ).vectorize[1, Self.simd_width](),
                UInt(k_tile_idx),
            )

        @parameter
        for k_tile_idx in range(Self.k_tiles_per_simd_b):
            Self.tensor_core_mma.load_b[swizzle=swizzle](
                smem_tile_b,
                self.b_reg_tile.tile[Self.num_n_mmas, Self.simd_width](
                    k_tile_idx, 0
                ).vectorize[1, Self.simd_width](),
                UInt(k_tile_idx),
            )

        inlined_assembly[
            "s_waitcnt lgkmcnt(0)",
            NoneType,
            constraints="",
            has_side_effect=True,
        ]()

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
            var a_tile = self.a_reg_tile.tile[Self.num_m_mmas, Self.simd_width](
                k_tile_idx, 0
            )
            var b_tile = self.b_reg_tile.tile[Self.num_n_mmas, Self.simd_width](
                k_tile_idx, 0
            )

            @parameter
            for fragment, mma_m_idx in product(
                range(Self.fragments_per_simd_a), range(Self.num_m_mmas)
            ):
                var a_fragment = a_tile.tile[1, Self.register_count_a](
                    mma_m_idx, fragment
                )

                @parameter
                for mma_n_idx in range(Self.num_n_mmas):
                    var b_fragment = b_tile.tile[1, Self.register_count_b](
                        mma_n_idx, fragment
                    )

                    # NOTE: this storage scheme is column major, because distribute needs it
                    # when writing back to global memory

                    # TODO make c_layout column major
                    var c_fragment = c_slice.tile[1, Self.register_count_a](
                        mma_n_idx * Self.num_m_mmas + mma_m_idx, 0
                    )

                    mma(
                        c_fragment.vectorize[1, Self.register_count_a]()[0, 0],
                        b_fragment.vectorize[1, Self.register_count_b]()[0, 0],
                        a_fragment.vectorize[1, Self.register_count_a]()[0, 0],
                        c_fragment.vectorize[1, Self.register_count_a]()[0, 0],
                    )

        cache_manager.commit[True](stage, tile_idx_a)
        cache_manager.commit[False](stage, tile_idx_b)
