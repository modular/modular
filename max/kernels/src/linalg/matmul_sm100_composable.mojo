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

from sys import size_of
from math import ceildiv
from hashlib import default_comp_time_hasher
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList

from gpu import WARP_SIZE, barrier
from gpu import lane_id as get_lane_id
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_idx, lane_id, thread_idx
from gpu.memory import AddressSpace, external_memory
from gpu.mma_sm100 import *
from gpu.tcgen05 import *
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.int_tuple import IntTuple
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from layout._ndbuffer_stub import from_ndbuffer_row_major
from gpu.cluster import block_rank_in_cluster
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tma_tile,
    _tma_desc_tile_layout,
)
from linalg import vendor_blas
from linalg.mmaop_sm100 import MmaOpSM100_SS

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

# Additional imports for testing
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static


# @register_passable("trivial")
trait OpArgs(Copyable):
    pass


# @register_passable("trivial")
trait LoadOp:
    alias args_type: OpArgs

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
    var a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout]
    var b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout]

    @always_inline
    fn __init__(
        out self,
        a: TMATensorTile[a_type, a_layout, a_desc_layout],
        b: TMATensorTile[b_type, b_layout, b_desc_layout],
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
](LoadOp):
    alias a_tma_layout = Layout.row_major(
        block_tile_shape[0] // cluster_shape[0], block_tile_shape[2]
    )
    alias b_tma_layout = Layout.row_major(
        block_tile_shape[1] // cluster_shape[1], block_tile_shape[2]
    )
    alias a_tma_desc_layout = _tma_desc_tile_layout[
        a_type,
        2,
        Index(block_tile_shape[0] // cluster_shape[0], block_tile_shape[2]),
        True,
        a_swizzle,
    ]()
    alias b_tma_desc_layout = _tma_desc_tile_layout[
        b_type,
        2,
        Index(block_tile_shape[1] // cluster_shape[1], block_tile_shape[2]),
        True,
        b_swizzle,
    ]()

    alias args_type = TMALoadOpArgs[
        a_type,
        b_type,
        Self.a_tma_layout,
        Self.b_tma_layout,
        Self.a_tma_desc_layout,
        Self.b_tma_desc_layout,
    ]

    alias a_tma_type = TMATensorTile[
        a_type, Self.a_tma_layout, Self.a_tma_desc_layout
    ]
    alias b_tma_type = TMATensorTile[
        b_type, Self.b_tma_layout, Self.b_tma_desc_layout
    ]

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
        a: LayoutTensor[a_type, *_, **_],
        b: LayoutTensor[b_type, *_, **_],
        ctx: DeviceContext,
    ) -> Self.args_type:
        var a_tma_op = create_tma_tile[
            a_type,
            2,
            Index(block_tile_shape[0] // cluster_shape[0], block_tile_shape[2]),
            swizzle_mode=a_swizzle,
        ](ctx, a)

        var b_tma_op = create_tma_tile[
            b_type,
            2,
            Index(block_tile_shape[1] // cluster_shape[1], block_tile_shape[2]),
            swizzle_mode=b_swizzle,
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


trait OutputOp:
    alias args_type: OpArgs

    fn __init__(out self, args: Self.args_type):
        ...

    fn __call__(self, tmem_addr: UInt32):
        ...


struct STOutputOpArgs[
    type: DType,
    layout: Layout,
    sb: Origin,
](OpArgs):
    var c: LayoutTensor[type, layout, sb]

    @always_inline
    fn __init__(
        out self,
        c: LayoutTensor[type, layout, sb],
    ):
        self.c = c

    @always_inline
    fn __copyinit__(out self, other: Self):
        self.c = other.c


struct R2GOutputOp[
    accum_type: DType,
    type: DType,
    layout: Layout,
    num_threads: Int,
    mma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    o: Origin,
](OutputOp):
    alias args_type = STOutputOpArgs[type, layout, o]

    var c: LayoutTensor[type, layout, o]

    @always_inline
    fn __init__(out self, args: Self.args_type):
        self.c = args.c

    @staticmethod
    @always_inline
    fn to_kernel_args(
        c: LayoutTensor[type, layout, o], ctx: DeviceContext
    ) -> Self.args_type:
        return Self.args_type(c)

    @always_inline
    fn __call__(self, tmem_addr: UInt32):
        alias BM = block_tile_shape[0]
        alias BN = block_tile_shape[1]
        alias MMA_M = mma_shape[0]
        alias MMA_N = mma_shape[1]

        alias num_m_mmas = BM // MMA_M
        alias num_n_mmas = BN // MMA_N
        alias c_frag_size = MMA_M * MMA_N // num_threads

        c_frag = tcgen05_ld[
            datapaths=16,
            bits=256,
            repeat = BN // 8,
            dtype=accum_type,
            pack=False,
            width=c_frag_size,
        ](tmem_addr)

        tcgen05_load_wait()  # wait for the load to finish

        alias num_warps = num_threads // WARP_SIZE
        # Extract last 2 bits so that warp_id is 0-3.
        var warp_id = (thread_idx.x // WARP_SIZE) & 3

        var ctile = self.c.tile[BM, BN](block_idx.y, block_idx.x)

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                alias mma_id = n_mma * num_m_mmas + m_mma

                c_gmem_warp_tile = ctile.tile[MMA_M // num_warps, MMA_N](
                    4 * m_mma + warp_id, n_mma
                )

                c_gmem_frag = c_gmem_warp_tile.vectorize[1, 2]().distribute[
                    Layout.row_major(8, 4)
                ](lane_id())

                alias num_vecs_m = c_gmem_frag.layout.shape[0].value()
                alias num_vecs_n = c_gmem_frag.layout.shape[1].value()

                @parameter
                for n_vec in range(num_vecs_n):

                    @parameter
                    for m_vec in range(num_vecs_m):
                        alias i_vec = n_vec * num_vecs_m + m_vec

                        c_gmem_frag[m_vec, n_vec] = rebind[
                            c_gmem_frag.element_type
                        ](
                            SIMD[accum_type, 2](
                                c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                            ).cast[type]()
                        )


trait PipelineOp:
    alias args_type: OpArgs

    @staticmethod
    fn run(args: Self.args_type):
        ...


struct PipelineArgs[
    loadop_t: LoadOp,
    outputop_t: OutputOp,
](OpArgs):
    var load_args: loadop_t.args_type
    var output_args: outputop_t.args_type
    var num_iters: UInt

    fn __init__(
        out self,
        load_args: loadop_t.args_type,
        output_args: outputop_t.args_type,
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
    alias args_type = PipelineArgs[loadop_t, outputop_t]

    @staticmethod
    @always_inline
    fn run(args: Self.args_type):
        var loadop = loadop_t(args.load_args)
        var outputop = outputop_t(args.output_args)

        alias BM = block_tile_shape[0]
        alias BN = block_tile_shape[1]
        alias BK = block_tile_shape[2]

        alias a_smem_layout = tile_layout_k_major[
            a_type, BM, BK, swizzle_mode=a_swizzle
        ]()
        alias b_smem_layout = tile_layout_k_major[
            b_type, BN, BK, swizzle_mode=b_swizzle
        ]()

        a_smem = rebind[
            UnsafePointer[
                Scalar[a_type],
                address_space = AddressSpace.SHARED,
                alignment=128,
            ]
        ](
            external_memory[
                Scalar[a_type],
                address_space = AddressSpace.SHARED,
                alignment=128,
                name="tmem_test_dynamic_shared_memory",
            ]()
        )

        # a_smem_layout is a description of how tile is arranged in memory, and LayoutTensor is a pointer to memory + a layout, taking in a_smem as its pointer
        alias a_smem_tile_t = LayoutTensor[
            a_type,
            a_smem_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]
        alias b_smem_tile_t = LayoutTensor[
            b_type,
            b_smem_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]

        alias a_size = a_smem_layout.size()
        alias b_size = b_smem_layout.size()

        constrained[
            ((a_size * size_of[a_type]()) % 128) == 0, "preserve alignment"
        ]()
        constrained[
            ((b_size * size_of[b_type]()) % 16) == 0, "preserve alignment"
        ]()
        var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()

        var a_smem_tile = a_smem_tile_t(a_smem)
        var b_smem_tile = b_smem_tile_t(b_smem)

        # Shared memory pointer to hold tensor memory address, after last smem pointer and expected smem size
        var ptr_tmem_addr = (
            (b_smem + b_size)
            .bitcast[UInt32]()
            .static_alignment_cast[alignment=16]()
        )

        alias accum_type = get_accum_type[a_type]()

        alias a_expected_bytes = a_size * size_of[a_type]()
        alias b_expected_bytes = b_size * size_of[b_type]()
        alias expected_bytes = a_expected_bytes + b_expected_bytes

        tma_mbar = (
            (ptr_tmem_addr + 2)
            .bitcast[SharedMemBarrier]()
            .static_alignment_cast[alignment=8]()
        )
        mma_mbar = (tma_mbar + 1).static_alignment_cast[alignment=8]()

        if thread_idx.x == 0:
            tma_mbar[0].init()
            mma_mbar[0].init()

        var tma_phase: UInt32 = 0
        var mma_phase: UInt32 = 0

        var elect_one_warp = thread_idx.x // WARP_SIZE == 0
        var elect_one_thread = thread_idx.x == 0
        var elect_one_cta = block_rank_in_cluster() % 2 == 0
        alias max_tmem_cols = 512

        # allocate all 2^18 bytes of smem for tcgen05, all 512 cols allocated
        if elect_one_warp:
            tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

        # Ensure all threads sees initialized mbarrier and
        # tensor memory allocation
        barrier()

        tmem_addr = ptr_tmem_addr[0]

        # Create MmaOpSM100_SS instance to handle MMA operations
        var mma_op = MmaOpSM100_SS[
            c_type,
            a_type,
            b_type,
            block_tile_shape,
            mma_shape,
            accum_type=accum_type,
            cta_group=1,
            a_swizzle=a_swizzle,
            b_swizzle=b_swizzle,
            transpose_b=True,
        ]()

        var num_iters = args.num_iters
        var m: UInt32 = block_idx.y * BM
        var n: UInt32 = block_idx.x * BN

        for i in range(num_iters):
            if elect_one_thread:
                tma_mbar[0].expect_bytes(expected_bytes)
                loadop(
                    a_smem_tile, b_smem_tile, m, n, UInt32(i * BK), tma_mbar[0]
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

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    var M = c.dim[0]()
    var N = c.dim[1]()
    alias K = a_shape.get[1]()
    alias num_iters: UInt = K // BK

    alias smem_use = (BM * size_of[a_type]() + BN * size_of[b_type]()) * BK + 24

    alias accum_type = get_accum_type[a_type]()

    alias block_dim = 128
    alias cluster_shape = Index(1, 1, 1)
    alias loadop_t = TMALoadOp[
        a_type,
        b_type,
        block_tile_shape,
        cluster_shape,
        a_swizzle,
        b_swizzle,
    ]
    alias outputop_t = R2GOutputOp[
        accum_type,
        c_type,
        c.layout,
        block_dim,
        mma_shape,
        block_tile_shape,
        c.origin,
    ]
    alias pipeline_t = Pipeline[
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
    alias pipeline_args_t = pipeline_t.args_type
    var args = pipeline_args_t(
        loadop_t.to_kernel_args(a, b, ctx),
        outputop_t.to_kernel_args(c, ctx),
        num_iters,
    )

    alias kernel = matmul_kernel[pipeline_t]
    ctx.enqueue_function[kernel](
        args,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(block_dim),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
    )
