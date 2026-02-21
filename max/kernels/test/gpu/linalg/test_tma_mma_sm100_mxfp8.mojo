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

from sys import size_of, argv
from utils.numerics import min_finite, max_finite
from gpu import (
    WARP_SIZE,
    barrier,
    warp_id as get_warp_id,
    block_idx,
    lane_id,
    thread_idx,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.memory import AddressSpace, external_memory
from gpu.compute.arch.mma_nvidia_sm100 import *
from gpu.compute.arch.tcgen05 import *
from layout import Layout, LayoutTensor
from layout._utils import ManagedLayoutTensor
from layout.int_tuple import IntTuple
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
    tile_sf_layout_k_major,
)
from layout.layout import tile_to_shape
from layout import Layout, LayoutTensor, UNKNOWN_VALUE, RuntimeLayout
from gpu.primitives.cluster import block_rank_in_cluster
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
    create_tma_tile,
)
from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple
from math import ceildiv
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList, Dim
from internal_utils._utils import ValOrDim, dynamic, static
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from internal_utils import assert_almost_equal
from random import rand
from layout._ndbuffer_stub import from_ndbuffer_row_major
from logger import Logger
from collections import Optional
from linalg.utils import elementwise_epilogue_type
from builtin.simd import _convert_f32_to_float8_ue8m0
from gpu.sync import syncwarp
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from random import random_ui64
from linalg.fp4_utils import (
    convert_ref_scales_to_mxfp8_format,
    MXFP8_SF_VECTOR_SIZE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    MXFP8_SF_DTYPE,
)
from linalg.matmul.vendor.blas import matmul


fn simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(a_scales_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_scales_tma_op, `nvvm.grid_constant`)
fn block_scaled_mxfp8_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    a_scales_tile_layout: Layout,
    b_scales_tile_layout: Layout,
    c_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    a_scales_desc_layout: Layout,
    b_scales_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    umma_shape: IndexList[3],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    num_threads: UInt = 256,
](
    a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
    a_scales_tma_op: TMATensorTile[
        a_scales_type, a_scales_tile_layout, a_scales_desc_layout
    ],
    b_scales_tma_op: TMATensorTile[
        b_scales_type, b_scales_tile_layout, b_scales_desc_layout
    ],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    num_iters: UInt,
):
    comptime assert num_threads == 256
    comptime assert (
        a_type == b_type and a_type == DType.float8_e4m3fn
    ), "Only support float8_e4m3fn"

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = umma_shape[0]
    comptime MMA_N = umma_shape[1]
    comptime MMA_K = umma_shape[2]
    comptime num_m_mmas = BM // MMA_M
    comptime num_n_mmas = BN // MMA_N
    comptime num_k_mmas = BK // MMA_K

    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    var smem = external_memory[
        UInt8, address_space = AddressSpace.SHARED, alignment=8
    ]()
    var a_smem = smem.bitcast[Scalar[a_type]]()

    comptime a_smem_tile_t = LayoutTensor[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    comptime b_smem_tile_t = LayoutTensor[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    comptime assert BM == BK == 128 and BN in (
        128,
        256,
    ), "Only support 128x128x128 or 128x256x128 block size"

    comptime a_scales_smem_layout = tile_sf_layout_k_major[
        BM, BK, MXFP8_SF_VECTOR_SIZE
    ]()
    comptime b_scales_smem_layout = tile_sf_layout_k_major[
        BN, BK, MXFP8_SF_VECTOR_SIZE
    ]()

    comptime a_scales_smem_tile_t = LayoutTensor[
        a_scales_type,
        a_scales_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    comptime b_scales_smem_tile_t = LayoutTensor[
        b_scales_type,
        b_scales_smem_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    comptime a_size = a_smem_layout.size()
    comptime b_size = b_smem_layout.size()
    comptime a_scales_size = a_scales_smem_layout.size()
    comptime b_scales_size = b_scales_smem_layout.size()

    comptime assert (
        (a_size * size_of[a_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (b_size * size_of[b_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (a_scales_size * size_of[a_scales_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (b_scales_size * size_of[b_scales_type]()) % 16
    ) == 0, "preserve alignment"

    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()
    var a_scales_smem = (b_smem + b_size).bitcast[Scalar[a_scales_type]]()
    var b_scales_smem = (a_scales_smem + a_scales_size).bitcast[
        Scalar[b_scales_type]
    ]()

    var a_smem_tile = a_smem_tile_t(a_smem)
    var b_smem_tile = b_smem_tile_t(b_smem)
    var a_scales_smem_tile = a_scales_smem_tile_t(a_scales_smem)
    var b_scales_smem_tile = b_scales_smem_tile_t(b_scales_smem)

    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (b_scales_smem + b_scales_size).bitcast[UInt32]()

    comptime accum_type = get_accum_type[a_type]()

    comptime c_frag_size = MMA_M * MMA_N // Int(num_threads)
    var c_frag = SIMD[accum_type, c_frag_size]()

    comptime a_expected_bytes = a_size * size_of[a_type]()
    comptime b_expected_bytes = b_size * size_of[b_type]()
    comptime a_scales_expected_bytes = a_scales_size * size_of[a_scales_type]()
    comptime b_scales_expected_bytes = b_scales_size * size_of[b_scales_type]()
    comptime expected_bytes = a_expected_bytes + b_expected_bytes + a_scales_expected_bytes + b_scales_expected_bytes

    tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    mma_mbar = tma_mbar + 1

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var elect_one_warp = get_warp_id() == 0
    var elect_one_thread = thread_idx.x == 0
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    comptime max_tmem_cols = 512

    var tmem_addr_ptr = (mma_mbar + 1).bitcast[UInt32]()
    if elect_one_warp:
        tcgen05_alloc[1](tmem_addr_ptr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    tmem_addr = tmem_addr_ptr[0]

    comptime SFA_NUM_COLS = BM // 32
    comptime SFB_NUM_COLS = BN // 32
    var a_scales_tmem_addr_start = tmem_addr + UInt32(BN)
    var b_scales_tmem_addr_start = a_scales_tmem_addr_start + UInt32(
        SFA_NUM_COLS
    )

    if thread_idx.x >= 128:
        tmem_addr += 16 << 16  # offset for lane 16

    comptime a_canonical_layout = tile_to_descriptor[a_type, a_smem_layout]()
    comptime b_canonical_layout = tile_to_descriptor[
        b_type, b_smem_layout, is_k_major=transpose_b
    ]()
    comptime aSBO = a_canonical_layout[0].stride[1].value() * size_of[a_type]()
    comptime aLBO = a_canonical_layout[1].stride[1].value() * size_of[a_type]()
    comptime b_stride01 = b_canonical_layout[0].stride[1].value()
    comptime b_stride11 = b_canonical_layout[1].stride[1].value()
    comptime bSBO = (b_stride01 if transpose_b else b_stride11) * size_of[
        b_type
    ]()
    comptime bLBO = (b_stride11 if transpose_b else b_stride01) * size_of[
        b_type
    ]()

    adesc = MMASmemDescriptor.create[aSBO, aLBO, a_swizzle](a_smem_tile.ptr)
    bdesc = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](b_smem_tile.ptr)

    idesc = UMMAInsDescriptor[UMMAKind.KIND_MXF8F6F4].create[
        accum_type,
        a_type,
        b_type,
        a_scales_type,
        Index[dtype = DType.uint32](umma_shape[0], umma_shape[1]),
        transpose_b=transpose_b,
    ]()

    for k_iter in range(num_iters):
        if elect_one_thread:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

            a_tma_op.async_copy(
                a_smem_tile,
                tma_mbar[0],
                (Int(k_iter) * BK, Int(block_idx.y) * BM),
            )
            b_tma_op.async_copy(
                b_smem_tile,
                tma_mbar[0],
                (
                    Int(k_iter) * BK,
                    Int(block_idx.x) * BN,
                ) if transpose_b else (
                    Int(block_idx.x) * BN,
                    Int(k_iter) * BK,
                ),
            )
            a_scales_tma_op.async_copy_4d(
                a_scales_smem_tile,
                tma_mbar[0],
                (
                    0,
                    0,
                    Int(k_iter),
                    Int(block_idx.y) * (BM // SF_MN_GROUP_SIZE),
                ),
            )
            b_scales_tma_op.async_copy_4d(
                b_scales_smem_tile,
                tma_mbar[0],
                (
                    0,
                    0,
                    Int(k_iter),
                    Int(block_idx.x) * (BN // SF_MN_GROUP_SIZE),
                ),
            )

        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        if elect_one_thread:
            comptime for i in range(BM // SF_MN_GROUP_SIZE):
                comptime idx = IntTuple(i * SF_ATOM_M[0], 0)
                comptime a_scales_offset = a_scales_smem_layout(idx) * size_of[
                    a_scales_type
                ]()
                var a_scales_tmem_addr = a_scales_tmem_addr_start + UInt32(
                    i * (SF_MN_GROUP_SIZE // 32)
                )
                var a_scales_desc = MMASmemDescriptor.create[
                    8 * 16, 0, TensorMapSwizzle.SWIZZLE_NONE
                ](a_scales_smem_tile.ptr + a_scales_offset)
                tcgen05_cp[
                    cta_group=1, datapaths=32, bits=128, multicast="warpx4"
                ](a_scales_tmem_addr, a_scales_desc)

            comptime for i in range(BN // SF_MN_GROUP_SIZE):
                comptime idx = IntTuple(i * SF_ATOM_M[0], 0)
                comptime b_scales_offset = b_scales_smem_layout(idx) * size_of[
                    b_scales_type
                ]()
                var b_scales_tmem_addr = b_scales_tmem_addr_start + UInt32(
                    i * (SF_MN_GROUP_SIZE // 32)
                )
                var b_scales_desc = MMASmemDescriptor.create[
                    8 * 16, 0, TensorMapSwizzle.SWIZZLE_NONE
                ](b_scales_smem_tile.ptr + b_scales_offset)
                tcgen05_cp[
                    cta_group=1, datapaths=32, bits=128, multicast="warpx4"
                ](b_scales_tmem_addr, b_scales_desc)

        syncwarp()

        barrier()

        if elect_one_thread:
            if k_iter == 0:
                var runtime_desc = UMMAInsDescriptor[
                    UMMAKind.KIND_MXF8F6F4
                ].update_desc_with_sf_id[0](
                    idesc,
                )
                mma(
                    adesc,
                    bdesc,
                    tmem_addr,
                    runtime_desc,
                    a_scales_tmem_addr_start,
                    b_scales_tmem_addr_start,
                    c_scale=0,
                )

                comptime for j in range(1, num_k_mmas):
                    runtime_desc = UMMAInsDescriptor[
                        UMMAKind.KIND_MXF8F6F4
                    ].update_desc_with_sf_id[UInt32(j)](
                        idesc,
                    )
                    comptime idx = IntTuple(0, MMA_K * j)
                    comptime a_offset = a_smem_layout(idx) * size_of[a_type]()
                    comptime b_offset = b_smem_layout(idx) * size_of[b_type]()
                    mma(
                        adesc + a_offset,
                        bdesc + b_offset,
                        tmem_addr,
                        runtime_desc,
                        a_scales_tmem_addr_start,
                        b_scales_tmem_addr_start,
                        c_scale=1,
                    )
            else:
                comptime for j in range(num_k_mmas):
                    var runtime_desc = UMMAInsDescriptor[
                        UMMAKind.KIND_MXF8F6F4
                    ].update_desc_with_sf_id[UInt32(j)](
                        idesc,
                    )
                    comptime idx = IntTuple(0, MMA_K * j)
                    comptime a_offset = a_smem_layout(idx) * size_of[a_type]()
                    comptime b_offset = b_smem_layout(idx) * size_of[b_type]()
                    mma(
                        adesc + a_offset,
                        bdesc + b_offset,
                        tmem_addr,
                        runtime_desc,
                        a_scales_tmem_addr_start,
                        b_scales_tmem_addr_start,
                        c_scale=1,
                    )

            mma_arrive(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat = BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)

    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    comptime num_warps = num_threads // UInt(WARP_SIZE)
    var warp_id = get_warp_id()
    warp_id = 2 * (warp_id % 4) + warp_id // 4

    ctile = c.tile[BM, BN](Int(block_idx.y), Int(block_idx.x))

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma

            c_gmem_warp_tile = ctile.tile[MMA_M // Int(num_warps), MMA_N](
                4 * m_mma + Int(warp_id), n_mma
            )

            c_gmem_frag = c_gmem_warp_tile.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](lane_id())

            comptime num_vecs_m = c_gmem_frag.layout.shape[0].value()
            comptime num_vecs_n = c_gmem_frag.layout.shape[1].value()

            comptime for n_vec in range(num_vecs_n):
                comptime for m_vec in range(num_vecs_m):
                    comptime i_vec = n_vec * num_vecs_m + m_vec

                    c_gmem_frag[m_vec, n_vec] = rebind[
                        c_gmem_frag.element_type
                    ](
                        SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()
                    )


fn sm100_block_scaled_mxfp8[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_scales_layout: Layout,
    b_scales_layout: Layout,
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    SF_VECTOR_SIZE: Int,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, MutAnyOrigin],
    a_scales: LayoutTensor[a_scales_type, a_scales_layout, MutAnyOrigin],
    b_scales: LayoutTensor[b_scales_type, b_scales_layout, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        a_type == b_type and a_type == DType.float8_e4m3fn
    ), "Only support float8_e4m3fn"

    var M = c.dim(0)
    comptime N = c_layout.shape[1].value()
    comptime K = a_layout.shape[1].value()

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime assert BM == BK == 128 and BN in (
        128,
        256,
    ), "Only support 128x128x128 or 128x256x128 block size"

    a_tma_op = create_tensor_tile[Index(BM, BK), swizzle_mode=a_swizzle](ctx, a)
    b_tma_op = create_tensor_tile[
        Index(BN, BK),
        swizzle_mode=b_swizzle,
    ](ctx, b)

    comptime assert (
        a_scales_type == b_scales_type and a_scales_type == MXFP8_SF_DTYPE
    ), "Only support F8-UE8M0 scales"
    comptime assert (
        a_scales.rank == b_scales.rank == 5
    ), "a_scales and b_scales must be 5D tensors"
    comptime assert (
        a_scales_layout.shape[2].value()
        == b_scales_layout.shape[2].value()
        == SF_ATOM_M[0]
    ), ""
    comptime assert (
        a_scales_layout.shape[3].value()
        == b_scales_layout.shape[3].value()
        == SF_ATOM_M[1]
    ), ""
    comptime assert (
        a_scales_layout.shape[4].value()
        == b_scales_layout.shape[4].value()
        == SF_ATOM_K
    ), ""

    comptime scales_4d_layout[layout: Layout] = Layout.row_major(
        layout.shape[0].value(),
        layout.shape[1].value(),
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    comptime a_scales_4d_layout = scales_4d_layout[a_scales_layout]
    comptime b_scales_4d_layout = scales_4d_layout[b_scales_layout]

    var a_scales_4d = LayoutTensor[
        a_scales_type, a_scales_4d_layout, MutAnyOrigin
    ](
        a_scales.ptr,
        RuntimeLayout[a_scales_4d_layout].row_major(
            IndexList[4](
                a_scales.dim(0),
                a_scales.dim(1),
                a_scales.dim(2),
                a_scales.dim(3) * a_scales.dim(4),
            ),
        ),
    )
    var b_scales_4d = LayoutTensor[
        b_scales_type, b_scales_4d_layout, MutAnyOrigin
    ](
        b_scales.ptr,
        RuntimeLayout[b_scales_4d_layout].row_major(
            IndexList[4](
                b_scales.dim(0),
                b_scales.dim(1),
                b_scales.dim(2),
                b_scales.dim(3) * b_scales.dim(4),
            ),
        ),
    )

    var a_scales_tma_op = create_tensor_tile[
        Index(
            BM // SF_MN_GROUP_SIZE, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
        ),
        swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        __tile_layout = Layout.row_major(
            BM // SF_MN_GROUP_SIZE, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
        ),
    ](ctx, a_scales_4d)

    var b_scales_tma_op = create_tensor_tile[
        Index(
            BN // SF_MN_GROUP_SIZE, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
        ),
        swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        __tile_layout = Layout.row_major(
            BN // SF_MN_GROUP_SIZE, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
        ),
    ](ctx, b_scales_4d)

    comptime block_dim = 256
    comptime sf_block_atom_size = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K

    comptime smem_use = (BM + BN) * size_of[a_type]() * BK + (
        (BM // SF_MN_GROUP_SIZE + BN // SF_MN_GROUP_SIZE)
        * sf_block_atom_size
        * size_of[a_scales_type]()
    ) + 24
    comptime kernel = block_scaled_mxfp8_kernel[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        b_scales_type,
        type_of(a_tma_op).layout,
        type_of(b_tma_op).layout,
        type_of(a_scales_tma_op).layout,
        type_of(b_scales_tma_op).layout,
        c_layout,
        type_of(a_tma_op).desc_layout,
        type_of(b_tma_op).desc_layout,
        type_of(a_scales_tma_op).desc_layout,
        type_of(b_scales_tma_op).desc_layout,
        block_tile_shape,
        umma_shape,
        transpose_b=transpose_b,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=block_dim,
    ]
    ctx.enqueue_function[kernel, kernel](
        a_tma_op,
        b_tma_op,
        a_scales_tma_op,
        b_scales_tma_op,
        c,
        UInt(ceildiv(K, BK)),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(block_dim),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )


def test_block_scaled_mxfp8[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    umma_shape: IndexList[3],
    transpose_b: Bool = True,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim):
    comptime assert transpose_b, "transpose_b must be true"

    var M = m.value
    var N = n.value
    var K = k.value

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime MMA_M = umma_shape[0]
    comptime MMA_N = umma_shape[1]
    comptime MMA_K = umma_shape[2]

    if N % BN != 0:
        raise Error("N must be divisible by BN")

    comptime scales_type = MXFP8_SF_DTYPE
    comptime ref_scales_type = DType.float32

    # Initialize reference scales
    comptime REF_BLOCK_SCALE = 128
    comptime static_ref_a_scales_shape = DimList(
        ceildiv(Int(k.dim), REF_BLOCK_SCALE), m.dim
    )
    comptime static_ref_b_scales_shape = DimList(
        ceildiv(Int(n.dim), REF_BLOCK_SCALE),
        ceildiv(Int(k.dim), REF_BLOCK_SCALE),
    )

    var dynamic_ref_a_scales_shape = DimList(
        ceildiv(k.value, REF_BLOCK_SCALE), m.value
    )
    var dynamic_ref_b_scales_shape = DimList(
        ceildiv(n.value, REF_BLOCK_SCALE), ceildiv(k.value, REF_BLOCK_SCALE)
    )

    var ref_a_scales_size = ceildiv(k.value, REF_BLOCK_SCALE) * m.value
    var ref_b_scales_size = ceildiv(n.value, REF_BLOCK_SCALE) * ceildiv(
        k.value, REF_BLOCK_SCALE
    )

    var a_scales_host_ref_ptr = UnsafePointer[Scalar[ref_scales_type]].alloc(
        ref_a_scales_size
    )
    var a_scales_host_ref = NDBuffer[
        ref_scales_type, 2, _, static_ref_a_scales_shape
    ](a_scales_host_ref_ptr, dynamic_ref_a_scales_shape)
    var b_scales_host_ref_ptr = UnsafePointer[Scalar[ref_scales_type]].alloc(
        ref_b_scales_size
    )
    var b_scales_host_ref = NDBuffer[
        ref_scales_type, 2, _, static_ref_b_scales_shape
    ](b_scales_host_ref_ptr, dynamic_ref_b_scales_shape)

    var a_scales_device_ref = ctx.enqueue_create_buffer[ref_scales_type](
        ref_a_scales_size
    )
    var a_scales_device_ref_nd = NDBuffer[
        ref_scales_type, 2, _, static_ref_a_scales_shape
    ](a_scales_device_ref.unsafe_ptr(), dynamic_ref_a_scales_shape)
    var b_scales_device_ref = ctx.enqueue_create_buffer[ref_scales_type](
        ref_b_scales_size
    )
    var b_scales_device_ref_nd = NDBuffer[
        ref_scales_type, 2, _, static_ref_b_scales_shape
    ](b_scales_device_ref.unsafe_ptr(), dynamic_ref_b_scales_shape)

    a_scales_host_ref.fill(Scalar[ref_scales_type](1.0))
    b_scales_host_ref.fill(Scalar[ref_scales_type](1.0))

    for i in range(a_scales_host_ref.dim(0)):
        for j in range(a_scales_host_ref.dim(1) // 32):
            for k in range(32):
                a_scales_host_ref[i, j * 32 + k] = (
                    1 << random_ui64(0, 3)
                ).cast[ref_scales_type]()

    for i in range(b_scales_host_ref.dim(0)):
        for j in range(b_scales_host_ref.dim(1)):
            b_scales_host_ref[i, j] = (1 << random_ui64(0, 3)).cast[
                ref_scales_type
            ]()

    ctx.enqueue_copy(a_scales_device_ref, a_scales_host_ref_ptr)
    ctx.enqueue_copy(b_scales_device_ref, b_scales_host_ref_ptr)

    print(
        String(a_type)
        + "_"
        + String(b_type)
        + "_"
        + String(c_type)
        + " Problem Shape: "
        + String(Index(M, N, K))
        + " Block Tile Shape: "
        + String(block_tile_shape)
        + " UMMA Shape: "
        + String(umma_shape)
    )

    comptime static_a_shape = DimList(m.dim, k.dim)
    comptime static_b_shape = DimList(n.dim, k.dim)
    comptime static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value)
    var dynamic_c_shape = DimList(m.value, n.value)

    comptime SF_VECTOR_SIZE = 32
    comptime atom_m = (32, 4)
    comptime atom_k = 4
    comptime sf_k = ceildiv(k.dim, SF_VECTOR_SIZE)
    comptime static_a_scales_shape = DimList(
        ceildiv(m.dim, atom_m[0] * atom_m[1]),
        ceildiv(sf_k, atom_k),
        Dim(atom_m[0]),
        Dim(atom_m[1]),
        Dim(atom_k),
    )
    comptime static_b_scales_shape = DimList(
        ceildiv(n.dim, atom_m[0] * atom_m[1]),
        ceildiv(sf_k, atom_k),
        Dim(atom_m[0]),
        Dim(atom_m[1]),
        Dim(atom_k),
    )

    var dynamic_a_scales_shape = DimList(
        ceildiv(m.value, atom_m[0] * atom_m[1]),
        ceildiv(sf_k, atom_k),
        Dim(atom_m[0]),
        Dim(atom_m[1]),
        Dim(atom_k),
    )
    var dynamic_b_scales_shape = DimList(
        ceildiv(n.value, atom_m[0] * atom_m[1]),
        ceildiv(sf_k, atom_k),
        Dim(atom_m[0]),
        Dim(atom_m[1]),
        Dim(atom_k),
    )

    var a_scales_total = (
        ceildiv(m.value, atom_m[0] * atom_m[1])
        * Int(ceildiv(sf_k, atom_k))
        * atom_m[0]
        * atom_m[1]
        * atom_k
    )
    var b_scales_total = (
        ceildiv(n.value, atom_m[0] * atom_m[1])
        * Int(ceildiv(sf_k, atom_k))
        * atom_m[0]
        * atom_m[1]
        * atom_k
    )

    var a_scales_host_ptr = UnsafePointer[Scalar[scales_type]].alloc(
        a_scales_total
    )
    var a_scales_host = NDBuffer[scales_type, 5, _, static_a_scales_shape](
        a_scales_host_ptr, dynamic_a_scales_shape
    )
    var b_scales_host_ptr = UnsafePointer[Scalar[scales_type]].alloc(
        b_scales_total
    )
    var b_scales_host = NDBuffer[scales_type, 5, _, static_b_scales_shape](
        b_scales_host_ptr, dynamic_b_scales_shape
    )

    var a_scales_device = ctx.enqueue_create_buffer[scales_type](a_scales_total)
    var a_scales_device_nd = NDBuffer[scales_type, 5, _, static_a_scales_shape](
        a_scales_device.unsafe_ptr(), dynamic_a_scales_shape
    )
    var b_scales_device = ctx.enqueue_create_buffer[scales_type](b_scales_total)
    var b_scales_device_nd = NDBuffer[scales_type, 5, _, static_b_scales_shape](
        b_scales_device.unsafe_ptr(), dynamic_b_scales_shape
    )

    var a_size = m.value * k.value
    var b_size = n.value * k.value
    var c_size = m.value * n.value

    var a_host_ptr = UnsafePointer[Scalar[a_type]].alloc(a_size)
    var a_host = NDBuffer[a_type, 2, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host_ptr = UnsafePointer[Scalar[b_type]].alloc(b_size)
    var b_host = NDBuffer[b_type, 2, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )
    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ptr, dynamic_c_shape
    )
    var c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_ref = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ref_ptr, dynamic_c_shape
    )

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_device_nd = NDBuffer[a_type, 2, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_device_nd = NDBuffer[b_type, 2, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device_ref.unsafe_ptr(), dynamic_c_shape
    )

    convert_ref_scales_to_mxfp8_format[
        REF_BLOCK_SIZE=REF_BLOCK_SCALE, SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE
    ](
        m,
        n,
        k,
        from_ndbuffer_row_major(a_scales_host_ref),
        from_ndbuffer_row_major(b_scales_host_ref),
        from_ndbuffer_row_major(a_scales_host),
        from_ndbuffer_row_major(b_scales_host),
    )
    # Initialize matmul operands
    if simple_init():
        for m in range(M):
            for k in range(K):
                a_host[m, k] = Float32(k).cast[a_type]()
        for n in range(N):
            for k in range(K):
                b_host[n, k] = Float32(1 if n == k else 0).cast[b_type]()
    else:
        rand(a_host.data, a_host.num_elements())
        rand(b_host.data, b_host.num_elements())

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    var a = from_ndbuffer_row_major(a_device_nd)
    var b = from_ndbuffer_row_major(b_device_nd)
    var c = from_ndbuffer_row_major(c_device_nd)
    var a_scales = from_ndbuffer_row_major(a_scales_device_nd)
    var b_scales = from_ndbuffer_row_major(b_scales_device_nd)
    var c_ref = from_ndbuffer_row_major(c_device_ref_nd)

    sm100_block_scaled_mxfp8[
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
    ](c, a, b, a_scales, b_scales, ctx)

    matmul[scales_type=scales_type](
        ctx,
        c_ref,
        a,
        b,
        a_scales=a_scales.get_immutable(),
        b_scales=b_scales.get_immutable(),
        transpose_b=True,
        c_row_major=True,
    )

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)

    ctx.synchronize()

    comptime rtol = 1e-2
    assert_almost_equal(
        c_host.data,
        c_host_ref.data,
        c_host.num_elements(),
        atol=0.0001,
        rtol=rtol,
    )

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()
    a_scales_host_ref_ptr.free()
    b_scales_host_ref_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^
    _ = a_scales_device_ref^
    _ = b_scales_device_ref^


def main():
    with DeviceContext() as ctx:
        comptime dtype = DType.float8_e4m3fn
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_M = 128
        comptime MMA_K = 32

        test_block_scaled_mxfp8[
            dtype,
            dtype,
            DType.bfloat16,
            Index(MMA_M, 256, BK),
            Index(MMA_M, 256, MMA_K),
            transpose_b=True,
        ](ctx, dynamic(256), static[256](), static[BK * 3]())
        test_block_scaled_mxfp8[
            dtype,
            dtype,
            DType.bfloat16,
            Index(MMA_M, 256, BK),
            Index(MMA_M, 256, MMA_K),
            transpose_b=True,
        ](ctx, dynamic(256), static[256 * 2](), static[BK * 3]())
        test_block_scaled_mxfp8[
            dtype,
            dtype,
            DType.bfloat16,
            Index(MMA_M, 256, BK),
            Index(MMA_M, 256, MMA_K),
            transpose_b=True,
        ](ctx, dynamic(1000), static[256 * 4](), static[BK * 3]())

        test_block_scaled_mxfp8[
            dtype,
            dtype,
            DType.bfloat16,
            Index(MMA_M, 128, BK),
            Index(MMA_M, 128, MMA_K),
            transpose_b=True,
        ](ctx, dynamic(256), static[2 * 128](), static[BK * 3]())
        test_block_scaled_mxfp8[
            dtype,
            dtype,
            DType.bfloat16,
            Index(MMA_M, 128, BK),
            Index(MMA_M, 128, MMA_K),
            transpose_b=True,
        ](ctx, dynamic(256), static[3 * 128](), static[BK * 2]())
        test_block_scaled_mxfp8[
            dtype,
            dtype,
            DType.bfloat16,
            Index(MMA_M, 128, BK),
            Index(MMA_M, 128, MMA_K),
            transpose_b=True,
        ](ctx, dynamic(1000), static[3 * 128](), static[BK * 3]())
