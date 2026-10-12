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

from std.builtin._closure import __ownership_keepalive
from std.math import ceildiv
from std.memory import bitcast
from std.sys import argv, size_of

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu import WARP_SIZE
from max.gpu.sync import barrier
from max.gpu import warp_id, block_idx, thread_idx
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import external_memory, fence_async_view_proxy
from max.gpu.compute.mma import st_matrix
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.compute.arch.tcgen05 import *

# Additional imports for testing
from internal_utils import assert_almost_equal
from layout import (
    Coord,
    Idx,
    IntTuple,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
    TileTensor,
    coord,
    row_major,
)
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.swizzle import make_swizzle
from layout.tensor_core_async import (
    st_matrix_n_layout,
    tile_layout_k_major_typed,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
    create_tma_tile,
)

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark":
            return True
    return False


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
def kernel_4[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_tile_shape: Coord,
    b_tile_shape: Coord,
    c_tile_shape: Coord,
    a_desc_shape: Coord,
    b_desc_shape: Coord,
    c_desc_shape: Coord,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: Int = 128,
](
    a_tma_op: TMATensorTile[a_type, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_shape, b_desc_shape],
    c_tma_op: TMATensorTile[c_type, c_tile_shape, c_desc_shape],
    num_iters_dev: Int32,
):
    var num_iters = Int(num_iters_dev)
    comptime assert num_threads == 128 or num_threads == 256
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_m_mmas = BM // MMA_M
    comptime num_n_mmas = BN // MMA_N
    comptime num_k_mmas = BK // MMA_K

    comptime TMA_BN = c_tile_shape.element_types[1].static_value

    comptime assert transpose_b, "Only support transposed B"
    comptime a_smem_layout = tile_layout_k_major_typed[
        a_type, BM, BK, a_swizzle
    ]
    comptime b_smem_layout = tile_layout_k_major_typed[
        b_type, BN, BK, b_swizzle
    ]
    comptime sub_a_smem_layout = tile_layout_k_major_typed[
        a_type, BM, 64, a_swizzle
    ]
    comptime sub_b_smem_layout = tile_layout_k_major_typed[
        b_type, BN, 64, b_swizzle
    ]

    var a_smem = rebind[
        MutPointer[Scalar[a_type], address_space=.SHARED, MutUntrackedOrigin]
    ](
        external_memory[
            Scalar[a_type],
            address_space=.SHARED,
            alignment=128,
            name="tmem_test_dynamic_shared_memory",
        ]()
    )
    comptime a_size = BM * BK
    comptime b_size = BN * BK
    comptime c_size = BM * BN

    comptime assert (
        (a_size * size_of[a_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (b_size * size_of[b_type]()) % 16
    ) == 0, "preserve alignment"
    comptime assert (
        (c_size * size_of[c_type]()) % 128
    ) == 0, "preserve alignment"

    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()
    var c_smem = (b_smem + b_size).bitcast[Scalar[c_type]]()

    var a_smem_tile = TileTensor(a_smem, a_smem_layout)
    var b_smem_tile = TileTensor(b_smem, b_smem_layout)
    var c_smem_tile = TileTensor(c_smem, row_major[BM, BN]())

    comptime accum_type = get_accum_type[a_type]()

    comptime c_frag_size = MMA_M * MMA_N // num_threads
    var c_frag: Array[Scalar[accum_type], c_frag_size]

    comptime a_expected_bytes = a_size * size_of[a_type]()
    comptime b_expected_bytes = b_size * size_of[b_type]()
    comptime expected_bytes = a_expected_bytes + b_expected_bytes

    var tma_mbar = (c_smem + c_size).bitcast[SharedMemBarrier]()
    var mma_mbar = tma_mbar + 1

    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (mma_mbar + 1).bitcast[UInt32]()

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var elect_one_warp = warp_id() == 0
    var elect_one_thread = thread_idx.x == 0
    comptime max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    # The K-major smem layout is ((8, BM/8), (sw_K, BK/sw_K)): the descriptor
    # SBO is the stride between 8-row core-matrix groups and the LBO is the
    # stride between swizzle atoms along K, i.e. flattened strides 1 and 3.
    comptime aSBO = type_of(a_smem_layout).static_stride[1] * size_of[a_type]()
    comptime aLBO = type_of(a_smem_layout).static_stride[3] * size_of[a_type]()
    comptime bSBO = type_of(b_smem_layout).static_stride[1] * size_of[b_type]()
    comptime bLBO = type_of(b_smem_layout).static_stride[3] * size_of[b_type]()

    var adesc = MMASmemDescriptor.create[aSBO, aLBO, a_swizzle](
        a_smem_tile.unsafe_ptr()
    )
    var bdesc = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](
        b_smem_tile.unsafe_ptr()
    )

    var idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        accum_type,
        a_type,
        b_type,
        Index[dtype=.uint32](mma_shape[0], mma_shape[1]),
        transpose_b=transpose_b,
    ]()

    # finish mma and store result in tensor memory
    for i in range(num_iters):
        # load A and B from global memory to shared memory
        if elect_one_thread:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

            comptime for j in range(BK // 64):
                comptime k = 64 * j
                comptime a_offset = Int(a_smem_layout(Coord(Idx[0], Idx[k])))
                comptime b_offset = Int(b_smem_layout(Coord(Idx[0], Idx[k])))
                comptime assert ((a_offset * size_of[a_type]()) % 128) == 0
                comptime assert ((b_offset * size_of[b_type]()) % 128) == 0
                var sub_a_smem_tile = TileTensor(
                    a_smem + a_offset, sub_a_smem_layout
                )
                a_tma_op.async_copy(
                    sub_a_smem_tile,
                    tma_mbar[0],
                    (i * BK + k, block_idx.y * BM),
                )
                var sub_b_smem_tile = TileTensor(
                    b_smem + b_offset, sub_b_smem_layout
                )
                b_tma_op.async_copy(
                    sub_b_smem_tile,
                    tma_mbar[0],
                    (
                        i * BK + k,
                        block_idx.x * BN,
                    ) if transpose_b else (
                        block_idx.x * BN,
                        i * BK + k,
                    ),
                )

        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        if elect_one_thread:
            comptime for j in range(num_k_mmas):
                comptime idx = Coord(Idx[0], Idx[MMA_K * j])
                comptime a_offset = Int(a_smem_layout(idx)) * size_of[a_type]()
                comptime b_offset = Int(b_smem_layout(idx)) * size_of[b_type]()

                # use c_scale=0 for the first mma only on the first iteration to initialize
                var c_scale_value: UInt32 = UInt32(
                    0 if (i == 0 and j == 0) else 1
                )
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
        repeat=BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)

    tcgen05_load_wait()

    # store from tensor memory to smem using the swizzling pattern

    var st_matrix_rt_layout = RuntimeLayout[
        st_matrix_n_layout[c_type, TMA_BN, num_m_mmas, 1](),
        element_type=.int32,
        linear_idx_type=.int32,
    ]()

    comptime st_matrix_swizzle = make_swizzle[c_type, c_swizzle]()

    comptime for tma_n in range(BN // TMA_BN):
        comptime for m_mma in range(num_m_mmas):
            comptime for i in range(TMA_BN // 16):
                var d_reg = SIMD[.bfloat16, 8]()

                comptime for _ei in range(4):
                    comptime _src_offset = (
                        i + tma_n * (TMA_BN // 16)
                    ) * 8 + 2 * _ei
                    var pair = SIMD[.float32, 2](
                        rebind[Float32](c_frag[_src_offset]),
                        rebind[Float32](c_frag[_src_offset + 1]),
                    )
                    var casted = pair.cast[.bfloat16]()
                    d_reg[2 * _ei] = casted[0]
                    d_reg[2 * _ei + 1] = casted[1]

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
                var offset = (
                    c_smem_tile.unsafe_ptr()
                    + st_matrix_swizzle(st_matrix_rt_layout(st_matrix_args))
                    + BM * TMA_BN * tma_n
                )

                var d_reg_f32_packed = bitcast[.float32, 4](d_reg)

                st_matrix[simd_width=4](offset, d_reg_f32_packed)
    barrier()

    # SMEM -> GMEM: Direct TMA store
    # UMMA (tensor memory) → registers → shared memory → global memory
    #           c_frag                   c_smem_tile      c_tma_op

    if elect_one_warp and thread_idx.x < BN // TMA_BN:
        fence_async_view_proxy()

        var smem_offset = c_smem_tile.unsafe_ptr() + BM * TMA_BN * thread_idx.x

        var c_tma_tile = TileTensor(
            smem_offset,
            row_major[
                c_tile_shape.element_types[0].static_value,
                c_tile_shape.element_types[1].static_value,
            ](),
        )

        c_tma_op.async_store(
            c_tma_tile,
            (
                block_idx.x * BN + thread_idx.x * TMA_BN,
                block_idx.y * BM,
            ),
        )
        c_tma_op.commit_group()
        # wait for the store to complete
        c_tma_op.wait_group[0]()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)


def blackwell_kernel_4[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    ctx: DeviceContext,
) raises:
    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())
    var K = Int(a.dim[1]())

    comptime assert transpose_b, "Only support transposed B"

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    var a_tma_op = create_tensor_tile[coord[BM, 64], swizzle_mode=a_swizzle](
        ctx, a
    )
    var b_tma_op = create_tensor_tile[
        coord[BN, 64],
        swizzle_mode=b_swizzle,
    ](ctx, b)
    var c_tma_op = create_tma_tile[BM, 64, swizzle_mode=c_swizzle](ctx, c)

    comptime smem_use = (
        BM * BK * size_of[a_type]()
        + BN * BK * size_of[b_type]()
        + BM * BN * size_of[c_type]()
        + 24
    )

    comptime block_dim = 128

    comptime kernel = kernel_4[
        a_type,
        b_type,
        c_type,
        type_of(a_tma_op).tile_shape,
        type_of(b_tma_op).tile_shape,
        type_of(c_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).desc_shape,
        type_of(c_tma_op).desc_shape,
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
        Int32(K // BK),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(block_dim),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )


comptime WARP_GROUP_SIZE = 128
comptime NumWarpPerWarpGroup = 4


def get_dict_of_shapes(
    index: Int, dict: Dict[Int, Tuple[Int, Int, Int]]
) -> Tuple[Int, Int, Int]:
    try:
        return dict[index]
    except error:
        print("error")
        return (128, 128, 128)


def make_dict_of_shapes() -> Dict[Int, Tuple[Int, Int, Int]]:
    var dic = Dict[Int, Tuple[Int, Int, Int]]()
    dic[0] = (4096, 4096, 4096)
    return dic^


def benchmark_blackwell_matmul(ctx: DeviceContext) raises:
    comptime a_type = DType.bfloat16
    comptime b_type = DType.bfloat16
    comptime c_type = DType.bfloat16
    comptime umma_shape = Index(64, 256, 16)
    comptime transpose_b = True
    comptime BK = 64

    comptime dict_of_shapes = make_dict_of_shapes()

    print("Benchmarking kernel_4")
    print("============================================")
    print("Shapes: [M, N, K]")
    print("Data types: a=", a_type, ", b=", b_type, ", c=", c_type)
    print("UMMA shape:", umma_shape[0], "x", umma_shape[1], "x", umma_shape[2])
    print("BK:", BK)
    print("transpose_b:", transpose_b)
    print()

    comptime for i in range(len(dict_of_shapes)):
        comptime shape = get_dict_of_shapes(i, dict_of_shapes)
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
                M=shape[0],
                N=shape[1],
                K=shape[2],
            ](ctx)
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
    M: Int = 4096,
    N: Int = 4096,
    K: Int = 4096,
](ctx: DeviceContext) raises:
    print(M, "x", N, "x", K)

    var a = HostDeviceTileTensor[a_type](row_major[M, K](), ctx)
    var a_host = a.host_tensor()
    var b = HostDeviceTileTensor[b_type](
        row_major[N if transpose_b else K, K if transpose_b else N](), ctx
    )
    var b_host = b.host_tensor()
    var c = HostDeviceTileTensor[c_type](row_major[M, N](), ctx)
    var c_host = c.host_tensor()
    var c_ref = HostDeviceTileTensor[c_type](row_major[M, N](), ctx)
    var c_host_ref = c_ref.host_tensor()

    # Initialize matmul operands
    for m_idx in range(M):
        for k_idx in range(K):
            a_host[m_idx, k_idx] = Float32(k_idx).cast[a_type]()
    for n_idx in range(N):
        for k_idx in range(K):
            b_host[n_idx, k_idx] = Float32(1 if n_idx == k_idx else 0).cast[
                b_type
            ]()
    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    a.to_device()
    b.to_device()
    c.to_device()
    c_ref.to_device()

    comptime block_tile_shape = Index(umma_shape[0], umma_shape[1], BK)

    blackwell_kernel_4[
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
    ](
        c.device_tensor(),
        a.device_tensor(),
        b.device_tensor(),
        ctx,
    )

    ctx.synchronize()
    if benchmark:
        comptime num_runs = 100
        comptime num_warmup = 10

        @inline(.always)
        def run_kernel(ctx: DeviceContext) raises {mut c, imm}:
            blackwell_kernel_4[
                transpose_b=transpose_b,
                umma_shape=umma_shape,  # 64, 128, 16
                block_tile_shape=block_tile_shape,  # 64, 128, 64 (BM, BN, entirety of BK)
                a_swizzle=a_swizzle,
                b_swizzle=b_swizzle,
                c_swizzle=c_swizzle,
            ](
                c.device_tensor(),
                a.device_tensor(),
                b.device_tensor(),
                ctx,
            )

        # Warmup
        for _ in range(num_warmup):
            run_kernel(ctx)
        ctx.synchronize()
        print("finished warmup")

        var nstime = (
            Float64(ctx.execution_time(run_kernel, num_runs)) / num_runs
        )
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * Float64(M) * Float64(N) * Float64(K) * 1e-12

        print("  Average time: ", sectime * 1000, " ms")
        print("  Performance: ", TFlop / sectime, " TFLOPS")
        print()
    else:
        vendor_blas.matmul(
            ctx,
            c_ref.device_tensor(),
            a.device_tensor(),
            b.device_tensor(),
            c_row_major=True,
            transpose_b=transpose_b,
        )

        ctx.synchronize()

        c.to_host()
        c_ref.to_host()
        comptime rtol = 1e-2

        assert_almost_equal(
            c_host.unsafe_ptr(),
            c_host_ref.unsafe_ptr(),
            M * N,
            atol=0.0001,
            rtol=rtol,
        )


def main() raises:
    with DeviceContext() as ctx:
        if is_benchmark():
            # Run the benchmark
            print("\n\n========== Running Benchmarks ==========\n")
            benchmark_blackwell_matmul(ctx)
            return

        test_blackwell_kernel_4[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            umma_shape=Index(64, 256, 16),
            a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            c_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
            BK=64,
            M=4096,
            N=4096,
            K=4096,
        ](ctx)
