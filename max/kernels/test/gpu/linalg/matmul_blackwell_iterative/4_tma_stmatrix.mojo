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

from sys import size_of, argv
from math import ceildiv

from buffer.buffer import NDBuffer
from buffer.dimlist import DimList

from gpu import WARP_SIZE, barrier
from gpu import lane_id as get_lane_id
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_idx, lane_id, thread_idx
from gpu.memory import AddressSpace, external_memory, fence_async_view_proxy
from gpu.mma_sm100 import *
from gpu.tcgen05 import *
from gpu.mma import st_matrix
from layout import (
    Layout,
    RuntimeLayout,
    LayoutTensor,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout.swizzle import make_ldmatrix_swizzle, make_swizzle
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.int_tuple import IntTuple
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    st_matrix_n_layout,
    tile_to_descriptor,
)
from layout._ndbuffer_stub import from_ndbuffer_row_major
from gpu.cluster import block_rank_in_cluster
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tma_tile
from linalg import vendor_blas

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple
from stdlib.bit import log2_floor

# Additional imports for testing
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark":
            return True
    return False


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
fn kernel_4[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: UInt = 128,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    num_iters: UInt,
):
    constrained[num_threads == 128 or num_threads == 256]()
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias num_m_mmas = BM // MMA_M
    alias num_n_mmas = BN // MMA_N
    alias num_k_mmas = BK // MMA_K

    alias TMA_BN = c_tma_op.layout.shape[1].value()

    alias a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()
    alias sub_a_smem_layout = tile_layout_k_major[
        a_type, BM, 64, swizzle_mode=a_swizzle
    ]()
    alias sub_b_smem_layout = tile_layout_k_major[
        b_type, BN, 64, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, 64, swizzle_mode=b_swizzle
    ]()
    alias c_smem_layout = Layout.row_major(BM, BN)

    a_smem = rebind[
        UnsafePointer[
            Scalar[a_type], address_space = AddressSpace.SHARED, alignment=128
        ]
    ](
        external_memory[
            Scalar[a_type],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="tmem_test_dynamic_shared_memory",
        ]()
    )
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
    alias c_smem_tile_t = LayoutTensor[
        c_type,
        c_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    alias sub_a_smem_tile_t = LayoutTensor[
        a_type,
        sub_a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    alias sub_b_smem_tile_t = LayoutTensor[
        b_type,
        sub_b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    alias a_size = a_smem_layout.size()
    alias b_size = b_smem_layout.size()
    alias c_size = c_smem_layout.size()

    constrained[
        ((a_size * size_of[a_type]()) % 128) == 0, "preserve alignment"
    ]()
    constrained[
        ((b_size * size_of[b_type]()) % 16) == 0, "preserve alignment"
    ]()
    constrained[
        ((c_size * size_of[c_type]()) % 128) == 0, "preserve alignment"
    ]()

    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()
    var c_smem = (b_smem + b_size).bitcast[Scalar[c_type]]()

    var a_smem_tile = a_smem_tile_t(a_smem)
    var b_smem_tile = b_smem_tile_t(b_smem)
    var c_smem_tile = c_smem_tile_t(c_smem)

    alias accum_type = get_accum_type[a_type]()

    alias c_frag_size = MMA_M * MMA_N // num_threads
    var c_frag = SIMD[accum_type, c_frag_size]()

    alias a_expected_bytes = a_size * size_of[a_type]()
    alias b_expected_bytes = b_size * size_of[b_type]()
    alias expected_bytes = a_expected_bytes + b_expected_bytes

    tma_mbar = (
        (c_smem + c_size)
        .bitcast[SharedMemBarrier]()
        .static_alignment_cast[alignment=8]()
    )
    mma_mbar = (tma_mbar + 1).static_alignment_cast[alignment=8]()

    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (
        (mma_mbar + 1).bitcast[UInt32]().static_alignment_cast[alignment=16]()
    )

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var elect_one_warp = thread_idx.x // WARP_SIZE == 0
    var elect_one_thread = thread_idx.x == 0
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    alias max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    tmem_addr = ptr_tmem_addr[0]

    alias a_canonical_layout = tile_to_descriptor[a_type, a_smem_layout]()
    alias b_canonical_layout = tile_to_descriptor[
        b_type, b_smem_layout, is_k_major=transpose_b
    ]()
    alias aSBO = a_canonical_layout[0].stride[1].value() * size_of[a_type]()
    alias aLBO = a_canonical_layout[1].stride[1].value() * size_of[a_type]()
    alias b_stride01 = b_canonical_layout[0].stride[1].value()
    alias b_stride11 = b_canonical_layout[1].stride[1].value()
    alias bSBO = (b_stride01 if transpose_b else b_stride11) * size_of[b_type]()
    alias bLBO = (b_stride11 if transpose_b else b_stride01) * size_of[b_type]()

    adesc = MMASmemDescriptor.create[aSBO, aLBO, a_swizzle](a_smem_tile.ptr)
    bdesc = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](b_smem_tile.ptr)

    idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        accum_type,
        a_type,
        b_type,
        Index[dtype = DType.uint32](mma_shape[0], mma_shape[1]),
        transpose_b=transpose_b,
    ]()

    # finish mma and store result in tensor memory
    for i in range(num_iters):
        # load A and B from global memory to shared memory
        if elect_one_thread:
            tma_mbar[0].expect_bytes(expected_bytes)

            @parameter
            for j in range(BK // 64):
                alias k = 64 * j
                alias a_offset = a_smem_layout(IntTuple(0, k))
                alias b_offset = b_smem_layout(IntTuple(0, k))
                constrained[((a_offset * size_of[a_type]()) % 128) == 0]()
                constrained[((b_offset * size_of[b_type]()) % 128) == 0]()
                sub_a_smem_tile = sub_a_smem_tile_t(a_smem + a_offset)
                a_tma_op.async_copy(
                    sub_a_smem_tile,
                    tma_mbar[0],
                    (UInt(i * BK + k), UInt(block_idx.y * BM)),
                )
                sub_b_smem_tile = sub_b_smem_tile_t(b_smem + b_offset)
                b_tma_op.async_copy(
                    sub_b_smem_tile,
                    tma_mbar[0],
                    (
                        UInt(i * BK + k),
                        UInt(block_idx.x * BN),
                    ) if transpose_b else (
                        UInt(block_idx.x * BN),
                        UInt(i * BK + k),
                    ),
                )

        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        if elect_one_thread:

            @parameter
            for j in range(num_k_mmas):
                alias idx = IntTuple(0, MMA_K * j)
                alias a_offset = a_smem_layout(idx) * size_of[a_type]()
                alias b_offset = b_smem_layout(idx) * size_of[b_type]()

                # use c_scale=0 for the first mma only on the first iteration to initialize
                var c_scale_value: UInt32 = 0 if (i == 0 and j == 0) else 1
                mma(
                    adesc + a_offset,
                    bdesc + b_offset,
                    tmem_addr,
                    idesc,
                    c_scale=c_scale_value,
                )

            mma_arrive(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    # load result from tensor memory to registers
    c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat = BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)

    tcgen05_load_wait()

    # store from tensor memory to smem using the swizzling pattern

    alias num_warps = num_threads // WARP_SIZE
    warp_id = thread_idx.x // WARP_SIZE

    var st_matrix_rt_layout = RuntimeLayout[
        st_matrix_n_layout[c_type, TMA_BN, num_m_mmas, 1](),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()

    alias st_matrix_swizzle = make_swizzle[c_type, c_swizzle]()

    @parameter
    for tma_n in range(BN // TMA_BN):

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for i in range(TMA_BN // 16):
                var d_reg = c_frag.slice[
                    8, offset = (i + tma_n * (TMA_BN // 16)) * 8
                ]().cast[DType.bfloat16]()

                var st_matrix_args = RuntimeTuple[
                    IntTuple(
                        UNKNOWN_VALUE,
                        IntTuple(
                            i,
                            m_mma,
                            UNKNOWN_VALUE,
                        ),
                    )
                ](thread_idx.x, i, m_mma, 0)
                var offset = c_smem_tile.ptr.offset(
                    st_matrix_swizzle(st_matrix_rt_layout(st_matrix_args))
                    + BM * TMA_BN * tma_n
                )

                var d_reg_f32_packed = bitcast[DType.float32, 4](d_reg)

                st_matrix[simd_width=4](offset, d_reg_f32_packed)
    barrier()

    # SMEM -> GMEM: Direct TMA store
    # UMMA (tensor memory) → registers → shared memory → global memory
    #           c_frag                   c_smem_tile      c_tma_op

    if elect_one_warp and thread_idx.x < UInt(BN // TMA_BN):
        fence_async_view_proxy()

        var smem_offset = c_smem_tile.ptr.offset(BM * TMA_BN * thread_idx.x)

        c_tma_tile = LayoutTensor[
            c_type,
            c_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            alignment=128,
        ](smem_offset)

        c_tma_op.async_store(
            c_tma_tile,
            (
                UInt(block_idx.x * BN + thread_idx.x * TMA_BN),
                UInt(block_idx.y * BM),
            ),
        )
        c_tma_op.commit_group()
        # wait for the store to complete
        c_tma_op.wait_group[0]()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)


fn blackwell_kernel_4[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](
    c_device: NDBuffer[c_type, 2, _, c_shape],
    a_device: NDBuffer[a_type, 2, _, a_shape],
    b_device: NDBuffer[b_type, 2, _, b_shape],
    ctx: DeviceContext,
) raises:
    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)
    var M = c.dim[0]()
    var N = c.dim[1]()
    var K = a.dim[1]()

    constrained[
        transpose_b,
        "Only support transposed B",
    ]()

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    a_tma_op = create_tma_tile[
        a_type, 2, Index(BM, 64), swizzle_mode=a_swizzle
    ](ctx, a)
    b_tma_op = create_tma_tile[
        b_type,
        2,
        Index(BN, 64) if transpose_b else Index(64, BN),
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
    ](ctx, b)
    c_tma_op = create_tma_tile[BM, 64, swizzle_mode=c_swizzle](ctx, c)

    alias smem_use = (
        BM * BK * size_of[a_type]()
        + BN * BK * size_of[b_type]()
        + BM * BN * size_of[c_type]()
        + 24
    )

    alias block_dim = 128

    alias kernel = kernel_4[
        a_type,
        b_type,
        c_type,
        __type_of(a_tma_op).layout,
        __type_of(b_tma_op).layout,
        __type_of(c_tma_op).layout,
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        __type_of(c_tma_op).desc_layout,
        block_tile_shape,
        umma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        num_threads=block_dim,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        K // BK,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(block_dim),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
    )


alias WARP_GROUP_SIZE = 128
alias NumWarpPerWarpGroup = 4


fn get_dict_of_shapes(
    index: Int, dict: Dict[Int, Tuple[Int, Int, Int]]
) -> Tuple[Int, Int, Int]:
    try:
        return dict[index]
    except error:
        print("error")
        return (128, 128, 128)


fn make_dict_of_shapes() -> Dict[Int, Tuple[Int, Int, Int]]:
    var dic = Dict[Int, Tuple[Int, Int, Int]]()
    dic[0] = (4096, 4096, 4096)
    return dic^


fn benchmark_blackwell_matmul(ctx: DeviceContext) raises:
    alias a_type = DType.bfloat16
    alias b_type = DType.bfloat16
    alias c_type = DType.bfloat16
    alias umma_shape = Index(64, 256, 16)
    alias transpose_b = True
    alias BK = 64

    alias dict_of_shapes = make_dict_of_shapes()

    print("Benchmarking kernel_4")
    print("============================================")
    print("Shapes: [M, N, K]")
    print("Data types: a=", a_type, ", b=", b_type, ", c=", c_type)
    print("UMMA shape:", umma_shape[0], "x", umma_shape[1], "x", umma_shape[2])
    print("BK:", BK)
    print("transpose_b:", transpose_b)
    print()

    @parameter
    for i in range(len(dict_of_shapes)):
        alias shape = get_dict_of_shapes(i, dict_of_shapes)
        try:
            print(
                "Benchmarking shape: [",
                shape[0],
                ",",
                shape[1],
                ",",
                shape[2],
                "]",
            )
            test_blackwell_kernel_4[
                a_type,
                b_type,
                c_type,
                umma_shape,
                transpose_b,
                BK,
                benchmark=True,
            ](ctx, dynamic(shape[0]), static[shape[1]](), static[shape[2]]())
        except e:
            print("Error: Failed to run benchmark for this shape")


def test_blackwell_kernel_4[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    umma_shape: IndexList[3],
    transpose_b: Bool = True,
    BK: Int = 64,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    benchmark: Bool = False,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim):
    var M = m.value
    var N = n.value
    var K = k.value

    print(
        M,
        "x",
        N,
        "x",
        K,
    )

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    alias static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[b_type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[b_type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    # Initialize matmul operands
    var at = a_host.tensor
    var bt = b_host.tensor
    for m in range(M):
        for k in range(K):
            at[m, k] = Float32(k).cast[a_type]()
    for n in range(N):
        for k in range(K):
            bt[n, k] = Float32(1 if n == k else 0).cast[b_type]()
    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    # Move operands to the Device

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    ctx.enqueue_copy(c_device.buffer, c_host.tensor.data)
    ctx.enqueue_copy(c_device_ref.buffer, c_host_ref.tensor.data)

    var a = from_ndbuffer_row_major(a_device.tensor)
    var b = from_ndbuffer_row_major(b_device.tensor)
    var c = from_ndbuffer_row_major(c_device.tensor)

    alias block_tile_shape = Index(umma_shape[0], umma_shape[1], BK)

    blackwell_kernel_4[
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
    ](
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        ctx,
    )

    ctx.synchronize()
    if benchmark:
        alias num_runs = 100
        alias num_warmup = 10

        @always_inline
        @parameter
        fn run_kernel(ctx: DeviceContext) raises:
            blackwell_kernel_4[
                transpose_b=transpose_b,
                umma_shape=umma_shape,  # 64, 128, 16
                block_tile_shape=block_tile_shape,  # 64, 128, 64 (BM, BN, entirety of BK)
                a_swizzle=a_swizzle,
                b_swizzle=b_swizzle,
                c_swizzle=c_swizzle,
            ](
                c_device.tensor,
                a_device.tensor,
                b_device.tensor,
                ctx,
            )

        # Warmup
        for _ in range(num_warmup):
            run_kernel(ctx)
        ctx.synchronize()
        print("finished warmup")

        var nstime = ctx.execution_time[run_kernel](num_runs) / num_runs
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12

        print("  Average time: ", sectime * 1000, " ms")
        print("  Performance: ", TFlop / sectime, " TFLOPS")
        print()
    else:
        vendor_blas.matmul(
            ctx,
            c_device_ref.tensor,
            a_device.tensor,
            b_device.tensor,
            c_row_major=True,
            transpose_b=transpose_b,
        )

        ctx.synchronize()

        ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
        ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)
        ctx.synchronize()
        alias rtol = 1e-2

        assert_almost_equal(
            c_host.tensor,
            c_host_ref.tensor,
            atol=0.0001,
            rtol=rtol,
        )

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device

    _ = a
    _ = b
    _ = c


def main():
    with DeviceContext() as ctx:
        if is_benchmark():
            # Run the benchmark
            print("\n\n========== Running Benchmarks ==========\n")
            benchmark_blackwell_matmul(ctx)
            return

        test_blackwell_kernel_4[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            umma_shape = Index(64, 256, 16),
            a_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            c_swizzle = TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
            BK=64,
        ](ctx, dynamic(4096), static[4096](), static[4096]())
