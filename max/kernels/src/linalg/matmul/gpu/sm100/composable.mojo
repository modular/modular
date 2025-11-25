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

from hashlib import default_comp_time_hasher
from math import ceildiv
from memory import (
    LegacyOpaquePointer as OpaquePointer,
    LegacyUnsafePointer as UnsafePointer,
)
from sys import size_of

from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE, barrier
from gpu import lane_id as get_lane_id
from gpu.cluster import block_rank_in_cluster
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu import block_idx, lane_id, thread_idx
from gpu.memory import external_memory
from gpu.mma_sm100 import *
from gpu.tcgen05 import *

# Additional imports for testing
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout._utils import ManagedLayoutTensor
from layout.int_tuple import IntTuple
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _tma_desc_tile_layout,
    create_tma_tile,
)

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple
from builtin.device_passable import DevicePassable

import ...vendor
from ....arch.sm100 import MmaOpSM100_SS


# @register_passable("trivial")
trait OpArgs(ImplicitlyCopyable):
    pass


# @register_passable("trivial")
trait LoadOp(DevicePassable):
    comptime args_type: OpArgs

    fn __init__(out self, args: Self.args_type):
        ...

    fn __call__(
        self,
        a_smem_tile: LayoutTensor[
            _, _, address_space = AddressSpace.SHARED, *_, **_
        ],
        b_smem_tile: LayoutTensor[
            _, _, address_space = AddressSpace.SHARED, *_, **_
        ],
        m: UInt32,
        n: UInt32,
        k: UInt32,
        ref [AddressSpace.SHARED]mbar: SharedMemBarrier,
    ):
        ...


trait MmaOp:
    fn __call__(self):
        ...


struct TMALoadOpArgs[
    a_type: DType,
    b_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_desc_layout: Layout = a_layout,
    b_desc_layout: Layout = b_layout,
](OpArgs):
    var a_tma_op: TMATensorTile[Self.a_type, Self.a_layout, Self.a_desc_layout]
    var b_tma_op: TMATensorTile[Self.b_type, Self.b_layout, Self.b_desc_layout]

    @always_inline
    fn __init__(
        out self,
        a: TMATensorTile[Self.a_type, Self.a_layout, Self.a_desc_layout],
        b: TMATensorTile[Self.b_type, Self.b_layout, Self.b_desc_layout],
    ):
        self.a_tma_op = a
        self.b_tma_op = b

    @always_inline
    fn __copyinit__(out self, other: Self):
        self.a_tma_op = other.a_tma_op
        self.b_tma_op = other.b_tma_op


struct TMALoadOp[
    a_type: DType,
    b_type: DType,
    block_tile_shape: IndexList[3],
    cluster_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](ImplicitlyCopyable, LoadOp):
    comptime a_tma_layout = Layout.row_major(
        Self.block_tile_shape[0] // Self.cluster_shape[0],
        Self.block_tile_shape[2],
    )
    comptime b_tma_layout = Layout.row_major(
        Self.block_tile_shape[1] // Self.cluster_shape[1],
        Self.block_tile_shape[2],
    )
    comptime a_tma_desc_layout = _tma_desc_tile_layout[
        Self.a_type,
        2,
        Index(
            Self.block_tile_shape[0] // Self.cluster_shape[0],
            Self.block_tile_shape[2],
        ),
        True,
        Self.a_swizzle,
    ]()
    comptime b_tma_desc_layout = _tma_desc_tile_layout[
        Self.b_type,
        2,
        Index(
            Self.block_tile_shape[1] // Self.cluster_shape[1],
            Self.block_tile_shape[2],
        ),
        True,
        Self.b_swizzle,
    ]()

    comptime args_type = TMALoadOpArgs[
        Self.a_type,
        Self.b_type,
        Self.a_tma_layout,
        Self.b_tma_layout,
        Self.a_tma_desc_layout,
        Self.b_tma_desc_layout,
    ]

    comptime a_tma_type = TMATensorTile[
        Self.a_type, Self.a_tma_layout, Self.a_tma_desc_layout
    ]
    comptime b_tma_type = TMATensorTile[
        Self.b_type, Self.b_tma_layout, Self.b_tma_desc_layout
    ]

    comptime device_type = Self

    fn _to_device_type(self, target: OpaquePointer):
        """Device type mapping is the identity function."""
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        """
        Gets this type's name, for use in error messages when handing arguments
        to kernels.
        """
        return String(
            "TMALoadOp[a_type = ",
            String(Self.a_type),
            ", b_type = ",
            String(Self.b_type),
            ", block_tile_shape = ",
            String(Self.block_tile_shape),
            ", cluster_shape = ",
            String(Self.cluster_shape),
            ", a_swizzle = ",
            String(Self.a_swizzle),
            ", b_swizzle = ",
            String(Self.b_swizzle),
            "]",
        )

    @staticmethod
    fn get_device_type_name() -> String:
        """
        Gets device_type's name, for use in error messages when handing arguments
        to kernels.
        """
        return String(
            "TMALoadOp[a_type = ",
            String(Self.a_type),
            ", b_type = ",
            String(Self.b_type),
            ", block_tile_shape = ",
            String(Self.block_tile_shape),
            ", cluster_shape = ",
            String(Self.cluster_shape),
            ", a_swizzle = ",
            String(Self.a_swizzle),
            ", b_swizzle = ",
            String(Self.b_swizzle),
            "]",
        )

    var a_tma_ptr: UnsafePointer[Self.a_tma_type]
    var b_tma_ptr: UnsafePointer[Self.b_tma_type]

    @always_inline
    fn __init__(out self, args: Self.args_type):
        self.a_tma_ptr = UnsafePointer(
            to=rebind[Self.a_tma_type](args.a_tma_op)
        )
        self.b_tma_ptr = UnsafePointer(
            to=rebind[Self.b_tma_type](args.b_tma_op)
        )

    @staticmethod
    @always_inline
    def to_kernel_args(
        a: LayoutTensor[Self.a_type, *_, **_],
        b: LayoutTensor[Self.b_type, *_, **_],
        ctx: DeviceContext,
    ) -> Self.args_type:
        var a_tma_op = create_tma_tile[
            Index(
                Self.block_tile_shape[0] // Self.cluster_shape[0],
                Self.block_tile_shape[2],
            ),
            swizzle_mode = Self.a_swizzle,
        ](ctx, a)

        var b_tma_op = create_tma_tile[
            Index(
                Self.block_tile_shape[1] // Self.cluster_shape[1],
                Self.block_tile_shape[2],
            ),
            swizzle_mode = Self.b_swizzle,
        ](ctx, b)

        return Self.args_type(
            rebind[Self.a_tma_type](a_tma_op),
            rebind[Self.b_tma_type](b_tma_op),
        )

    @always_inline
    fn __call__(
        self,
        a_smem_tile: LayoutTensor[
            _, _, address_space = AddressSpace.SHARED, *_, **_
        ],
        b_smem_tile: LayoutTensor[
            _, _, address_space = AddressSpace.SHARED, *_, **_
        ],
        m: UInt32,
        n: UInt32,
        k: UInt32,
        ref [AddressSpace.SHARED]mbar: SharedMemBarrier,
    ):
        self.a_tma_ptr[].async_copy(
            a_smem_tile,
            mbar,
            (UInt(k), UInt(m)),
        )

        self.b_tma_ptr[].async_copy(
            b_smem_tile,
            mbar,
            (UInt(k), UInt(n)),
        )


trait OutputOp(DevicePassable):
    comptime args_type: OpArgs

    fn __init__(out self, args: Self.args_type):
        ...

    fn __call__(self, tmem_addr: UInt32):
        ...


struct STOutputOpArgs[
    dtype: DType,
    layout: Layout,
    sb: Origin,
](OpArgs):
    var c: LayoutTensor[Self.dtype, Self.layout, Self.sb]

    @always_inline
    fn __init__(
        out self,
        c: LayoutTensor[Self.dtype, Self.layout, Self.sb],
    ):
        self.c = c

    @always_inline
    fn __copyinit__(out self, other: Self):
        self.c = other.c


struct R2GOutputOp[
    accum_type: DType,
    dtype: DType,
    layout: Layout,
    num_threads: Int,
    mma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    o: Origin,
](ImplicitlyCopyable, OutputOp):
    comptime args_type = STOutputOpArgs[Self.dtype, Self.layout, Self.o]

    var c: LayoutTensor[Self.dtype, Self.layout, Self.o]

    comptime device_type = Self

    fn _to_device_type(self, target: OpaquePointer):
        """Device type mapping is the identity function."""
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        """
        Gets this type's name, for use in error messages when handing arguments
        to kernels.
        """
        return String(
            "R2GOutputOp[accum_type = ",
            String(Self.accum_type),
            ", dtype = ",
            String(Self.dtype),
            ", layout = ",
            String(Self.layout),
            ", num_threads = ",
            String(Self.num_threads),
            ", mma_shape = ",
            String(Self.mma_shape),
            ", block_tile_shape = ",
            String(Self.block_tile_shape),
            "]",
        )

    @staticmethod
    fn get_device_type_name() -> String:
        """
        Gets device_type's name, for use in error messages when handing arguments
        to kernels.
        """
        return String(
            "R2GOutputOp[accum_type = ",
            String(Self.accum_type),
            ", dtype = ",
            String(Self.dtype),
            ", layout = ",
            String(Self.layout),
            ", num_threads = ",
            String(Self.num_threads),
            ", mma_shape = ",
            String(Self.mma_shape),
            ", block_tile_shape = ",
            String(Self.block_tile_shape),
            "]",
        )

    @always_inline
    fn __init__(out self, args: Self.args_type):
        self.c = args.c

    @staticmethod
    @always_inline
    fn to_kernel_args(
        c: LayoutTensor[Self.dtype, Self.layout, Self.o], ctx: DeviceContext
    ) -> Self.args_type:
        return Self.args_type(c)

    @always_inline
    fn __call__(self, tmem_addr: UInt32):
        comptime BM = Self.block_tile_shape[0]
        comptime BN = Self.block_tile_shape[1]
        comptime MMA_M = Self.mma_shape[0]
        comptime MMA_N = Self.mma_shape[1]

        comptime num_m_mmas = BM // MMA_M
        comptime num_n_mmas = BN // MMA_N
        comptime c_frag_size = MMA_M * MMA_N // Self.num_threads

        c_frag = tcgen05_ld[
            datapaths=16,
            bits=256,
            repeat = BN // 8,
            dtype = Self.accum_type,
            pack=False,
            width=c_frag_size,
        ](tmem_addr)

        tcgen05_load_wait()  # wait for the load to finish

        comptime num_warps = Self.num_threads // WARP_SIZE
        # Extract last 2 bits so that warp_id is 0-3.
        var warp_id = (thread_idx.x // UInt(WARP_SIZE)) & 3

        var ctile = self.c.tile[BM, BN](Int(block_idx.y), Int(block_idx.x))

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                comptime mma_id = n_mma * num_m_mmas + m_mma

                c_gmem_warp_tile = ctile.tile[MMA_M // num_warps, MMA_N](
                    4 * m_mma + Int(warp_id), n_mma
                )

                c_gmem_frag = c_gmem_warp_tile.vectorize[1, 2]().distribute[
                    Layout.row_major(8, 4)
                ](lane_id())

                comptime num_vecs_m = c_gmem_frag.layout.shape[0].value()
                comptime num_vecs_n = c_gmem_frag.layout.shape[1].value()

                @parameter
                for n_vec in range(num_vecs_n):

                    @parameter
                    for m_vec in range(num_vecs_m):
                        comptime i_vec = n_vec * num_vecs_m + m_vec

                        c_gmem_frag[m_vec, n_vec] = rebind[
                            c_gmem_frag.element_type
                        ](
                            SIMD[Self.accum_type, 2](
                                c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                            ).cast[Self.dtype]()
                        )


trait PipelineOp:
    comptime args_type: OpArgs

    @staticmethod
    fn run(args: Self.args_type):
        ...


struct PipelineArgs[
    loadop_t: LoadOp,
    outputop_t: OutputOp,
](DevicePassable, OpArgs):
    comptime device_type = Self

    fn _to_device_type(self, target: OpaquePointer):
        """Device type mapping is the identity function."""
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        """
        Gets this type's name, for use in error messages when handing arguments
        to kernels.
        """
        return String(
            "PipelineArgs[loadop_t = ",
            Self.loadop_t.get_type_name(),
            ", outputop_t = ",
            Self.outputop_t.get_type_name(),
            "]",
        )

    @staticmethod
    fn get_device_type_name() -> String:
        """
        Gets device_type's name, for use in error messages when handing arguments
        to kernels.
        """
        return String(
            "PipelineArgs[loadop_t = ",
            Self.loadop_t.get_device_type_name(),
            ", outputop_t = ",
            Self.outputop_t.get_device_type_name(),
            "]",
        )

    var load_args: Self.loadop_t.args_type
    var output_args: Self.outputop_t.args_type
    var num_iters: UInt

    fn __init__(
        out self,
        load_args: Self.loadop_t.args_type,
        output_args: Self.outputop_t.args_type,
        num_iters: UInt,
    ):
        self.load_args = load_args
        self.output_args = output_args
        self.num_iters = num_iters


struct Pipeline[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    loadop_t: LoadOp,
    outputop_t: OutputOp,
](PipelineOp):
    comptime args_type = PipelineArgs[Self.loadop_t, Self.outputop_t]

    @staticmethod
    @always_inline
    fn run(args: Self.args_type):
        var loadop = Self.loadop_t(args.load_args)
        var outputop = Self.outputop_t(args.output_args)

        comptime BM = Self.block_tile_shape[0]
        comptime BN = Self.block_tile_shape[1]
        comptime BK = Self.block_tile_shape[2]

        comptime a_smem_layout = tile_layout_k_major[
            Self.a_type, BM, BK, swizzle_mode = Self.a_swizzle
        ]()
        comptime b_smem_layout = tile_layout_k_major[
            Self.b_type, BN, BK, swizzle_mode = Self.b_swizzle
        ]()

        a_smem = rebind[
            UnsafePointer[
                Scalar[Self.a_type],
                address_space = AddressSpace.SHARED,
            ]
        ](
            external_memory[
                Scalar[Self.a_type],
                address_space = AddressSpace.SHARED,
                alignment=128,
                name="tmem_test_dynamic_shared_memory",
            ]()
        )

        # a_smem_layout is a description of how tile is arranged in memory, and LayoutTensor is a pointer to memory + a layout, taking in a_smem as its pointer
        comptime a_smem_tile_t = LayoutTensor[
            Self.a_type,
            a_smem_layout,
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]
        comptime b_smem_tile_t = LayoutTensor[
            Self.b_type,
            b_smem_layout,
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]

        comptime a_size = a_smem_layout.size()
        comptime b_size = b_smem_layout.size()

        constrained[
            ((a_size * size_of[Self.a_type]()) % 128) == 0, "preserve alignment"
        ]()
        constrained[
            ((b_size * size_of[Self.b_type]()) % 16) == 0, "preserve alignment"
        ]()
        var b_smem = (a_smem + a_size).bitcast[Scalar[Self.b_type]]()

        var a_smem_tile = a_smem_tile_t(a_smem)
        var b_smem_tile = b_smem_tile_t(b_smem)

        # Shared memory pointer to hold tensor memory address, after last smem pointer and expected smem size
        var ptr_tmem_addr = (b_smem + b_size).bitcast[UInt32]()

        comptime accum_type = get_accum_type[Self.a_type]()

        comptime a_expected_bytes = a_size * size_of[Self.a_type]()
        comptime b_expected_bytes = b_size * size_of[Self.b_type]()
        comptime expected_bytes = a_expected_bytes + b_expected_bytes

        tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
        mma_mbar = tma_mbar + 1

        if thread_idx.x == 0:
            tma_mbar[0].init()
            mma_mbar[0].init()

        var tma_phase: UInt32 = 0
        var mma_phase: UInt32 = 0

        var elect_one_warp = thread_idx.x // UInt(WARP_SIZE) == 0
        var elect_one_thread = thread_idx.x == 0
        var elect_one_cta = block_rank_in_cluster() % 2 == 0
        comptime max_tmem_cols = 512

        # allocate all 2^18 bytes of smem for tcgen05, all 512 cols allocated
        if elect_one_warp:
            tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

        # Ensure all threads sees initialized mbarrier and
        # tensor memory allocation
        barrier()

        tmem_addr = ptr_tmem_addr[0]

        # Create MmaOpSM100_SS instance to handle MMA operations
        var mma_op = MmaOpSM100_SS[
            Self.c_type,
            Self.a_type,
            Self.b_type,
            Self.block_tile_shape,
            Self.mma_shape,
            accum_type=accum_type,
            cta_group=1,
            a_swizzle = Self.a_swizzle,
            b_swizzle = Self.b_swizzle,
            transpose_b=True,
        ]()

        var num_iters = args.num_iters
        var m: UInt32 = block_idx.y * UInt(BM)
        var n: UInt32 = block_idx.x * UInt(BN)

        for i in range(num_iters):
            if elect_one_thread:
                tma_mbar[0].expect_bytes(expected_bytes)
                loadop(
                    a_smem_tile,
                    b_smem_tile,
                    m,
                    n,
                    UInt32(i * UInt(BK)),
                    tma_mbar[0],
                )

            tma_mbar[0].wait(tma_phase)
            tma_phase ^= 1

            if elect_one_thread:
                mma_op.mma(a_smem_tile, b_smem_tile, tmem_addr, init_c=(i == 0))
                mma_op.commit(mma_mbar)

            mma_mbar[0].wait(mma_phase)
            mma_phase ^= 1

        outputop(tmem_addr)

        barrier()
        if elect_one_warp:
            tcgen05_release_allocation_lock[1]()
            tcgen05_dealloc[1](tmem_addr, max_tmem_cols)


@__llvm_arg_metadata(args, `nvvm.grid_constant`)
fn matmul_kernel[pipeline_t: PipelineOp](args: pipeline_t.args_type):
    pipeline_t.run(args)


fn matmul_sm100[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    *,
    mma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](
    c_device: NDBuffer[c_type, 2, _, c_shape],
    a_device: NDBuffer[a_type, 2, _, a_shape],
    b_device: NDBuffer[b_type, 2, _, b_shape],
    ctx: DeviceContext,
) raises:
    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    var M = c.dim[0]()
    var N = c.dim[1]()
    comptime K = a_shape.get[1]()
    comptime num_iters = UInt(K // BK)

    comptime smem_use = (
        BM * size_of[a_type]() + BN * size_of[b_type]()
    ) * BK + 24

    comptime accum_type = get_accum_type[a_type]()

    comptime block_dim = 128
    comptime cluster_shape = Index(1, 1, 1)
    comptime loadop_t = TMALoadOp[
        a_type,
        b_type,
        block_tile_shape,
        cluster_shape,
        a_swizzle,
        b_swizzle,
    ]
    comptime outputop_t = R2GOutputOp[
        accum_type,
        c_type,
        c.layout,
        block_dim,
        mma_shape,
        block_tile_shape,
        c.origin,
    ]
    comptime pipeline_t = Pipeline[
        a_type,
        b_type,
        c_type,
        block_tile_shape,
        mma_shape,
        a_swizzle,
        b_swizzle,
        loadop_t,
        outputop_t,
    ]
    comptime pipeline_args_t = pipeline_t.args_type
    var args = pipeline_args_t(
        loadop_t.to_kernel_args(a, b, ctx),
        outputop_t.to_kernel_args(c, ctx),
        num_iters,
    )

    comptime kernel = matmul_kernel[pipeline_t]
    ctx.enqueue_function_checked[kernel, kernel](
        args,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(block_dim),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
    )
