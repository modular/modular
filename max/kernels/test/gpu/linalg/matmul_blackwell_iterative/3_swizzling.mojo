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
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tma_tile
from linalg import vendor_blas

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


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark":
            return True
    return False


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
fn kernel_3[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: UInt = 128,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    num_iters: UInt,
):
    constrained[num_threads == 128 or num_threads == 256]()
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]  # BM
    alias MMA_N = mma_shape[1]  # BN
    alias MMA_K = mma_shape[2]  # 16
    alias num_m_mmas = BM // MMA_M
    alias num_n_mmas = BN // MMA_N
    alias num_k_mmas = BK // MMA_K

    # we don't do the whole mma_shape_A vibes, rather, we directly declare it
    # tile_layout_k_major is cutlass equiv of tile_to_mma_shape
    # and sA_layout gets computed directly, by hand
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

    constrained[
        ((a_size * size_of[a_type]()) % 128) == 0, "preserve alignment"
    ]()
    constrained[
        ((b_size * size_of[b_type]()) % 16) == 0, "preserve alignment"
    ]()
    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()

    var a_smem_tile = a_smem_tile_t(a_smem)
    var b_smem_tile = b_smem_tile_t(b_smem)

    alias accum_type = get_accum_type[a_type]()

    alias c_frag_size = MMA_M * MMA_N // num_threads  # MMA_M * MMA_N is the size of the accumulator, num_threads is the number of threads in the warp, c_frag_size is the num of elements in the accumulator per thread
    var c_frag = SIMD[
        accum_type, c_frag_size
    ]()  # array of accumulator elements

    alias a_expected_bytes = a_size * size_of[a_type]()
    alias b_expected_bytes = b_size * size_of[b_type]()
    alias expected_bytes = a_expected_bytes + b_expected_bytes

    tma_mbar = (
        (b_smem + b_size)
        .bitcast[SharedMemBarrier]()
        .static_alignment_cast[alignment=8]()
    )
    mma_mbar = (tma_mbar + 1).static_alignment_cast[alignment=8]()

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

    # allocate all 2^18 bytes of smem for tcgen05, all 512 cols allocated
    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    tmem_addr = ptr_tmem_addr[0]

    # give me a tensor for matrices A and B, sliced down to the portion that this CTA is responsible for, and referring to original global tensor
    # so it's the 128 x 64 portion the CTA will copy from global tensor into smem tile A and B
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

    for i in range(
        num_iters
    ):  # K // BK, which is K // 64 or K // 128 depending on BK
        # so only one thread per CTA does the copy
        if elect_one_thread:
            tma_mbar[0].expect_bytes(expected_bytes)

            @parameter
            for j in range(
                BK // 64
            ):  # so we do the copy in 64 chunks or 64 elements at a time (BK // 64). but hmm, we said that the K atom can only be 32 bytes (16 elements)
                alias k = 64 * j
                alias a_offset = a_smem_layout(IntTuple(0, k))
                alias b_offset = b_smem_layout(IntTuple(0, k))
                constrained[((a_offset * size_of[a_type]()) % 128) == 0]()
                constrained[((b_offset * size_of[b_type]()) % 128) == 0]()
                sub_a_smem_tile = sub_a_smem_tile_t(a_smem + a_offset)
                # the answer to the above comment. # The descriptor layout i.e. data per copy can be smaller than the shared memory
                # tile shape due to WGMMA requirement. E.g. k-major no swizzle WGMMA BM x 16B to be
                # one continuous chunk in shared memory. We need to break down tile shape in K by 16B.
                # so the async_copy takes care of that. TMA engine will copy the data from global tensor into smem tile A
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
        # wait for the copy to finish
        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        # now we do the mma, again only one thread issues the instruction
        if elect_one_thread:

            @parameter
            for j in range(
                num_k_mmas
            ):  # BK by MMA_K chunks for the mma acc required this time, since you can only do MMA_K at a time (16 elements)
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

    # eventually all of c has been accumulated, so we load it from tmem_addr into c_frag registers using tcgen05_ld
    c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat = BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)

    tcgen05_load_wait()  # wait for the load to finish

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    alias num_warps = num_threads // WARP_SIZE
    warp_id = thread_idx.x // WARP_SIZE

    ctile = c.tile[BM, BN](block_idx.y, block_idx.x)

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
                        ).cast[c_type]()
                    )


fn blackwell_kernel_3[
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

    # hard coded 64 for BK

    # equivalent of cutlass tma atom a, it is a handle that is passed to async_copy, to accurately tell the TMA engine how to copy from global tensor a into smem tile A
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

    alias smem_use = (BM * size_of[a_type]() + BN * size_of[b_type]()) * BK + 24

    alias block_dim = 128

    alias kernel = kernel_3[
        a_type,
        b_type,
        c_type,
        a_tma_op.layout,
        b_tma_op.layout,
        c.layout,
        a_tma_op.desc_layout,
        b_tma_op.desc_layout,
        block_tile_shape,
        umma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=block_dim,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c,
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

    print("Benchmarking kernel_3")
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
            test_blackwell_kernel_3[
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


def test_blackwell_kernel_3[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    umma_shape: IndexList[3],
    transpose_b: Bool = True,
    BK: Int = 64,
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

    blackwell_kernel_3[
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
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
            blackwell_kernel_3[
                transpose_b=transpose_b,
                umma_shape=umma_shape,  # 64, 128, 16
                block_tile_shape=block_tile_shape,  # 64, 128, 64 (BM, BN, entirety of BK)
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

        test_blackwell_kernel_3[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            umma_shape = Index(64, 256, 16),
            transpose_b=True,
            BK=64,
        ](ctx, dynamic(4096), static[4096](), static[4096]())
