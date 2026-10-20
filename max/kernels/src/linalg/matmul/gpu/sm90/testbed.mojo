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

"""Provides a testbed for validating SM90 warp-specialized matmul kernels against a vendor reference."""

from std.math import ceildiv
from std.sys import size_of

from max.gpu.host import DeviceContext
from std.memory import dealloc
from std.memory.alloc import Layout as AllocLayout
from layout import Coord, CoordLike, Idx, TileTensor, row_major
from std.utils.index import IndexList
from internal_utils import assert_almost_equal, assert_with_measure
from std.random import rand
from internal_utils._measure import relative_difference
from std.collections import OptionalReg
from std.utils.index import IndexList

from ....utils import ElementwiseComputeFn, no_compute_fn
from ....utils_gpu import MatmulConfig
from ...vendor.blas import Backend
from ...vendor.blas import matmul as vendor_matmul
from ..tile_scheduler import MatmulSchedule
from .matmul import warp_specialize_gemm_with_multicasting


def test_matmul_sm90[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    ComputeFnType: ElementwiseComputeFn = type_of(no_compute_fn),
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    cluster_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    num_consumer: Int = 1,
    num_pipeline_stages: Int = 4,
    transpose_b: Bool = True,
    partitioned_multicast: Bool = False,
    grid_shape: OptionalReg[IndexList[2]] = None,
    use_tma_store: Bool = False,
    schedule: MatmulSchedule = MatmulSchedule.NONE,
    default_epilogue: Bool = False,
    has_compute_fn: Bool = False,
    measure_threshold: Optional[Float64] = None,
    backend: Backend = Backend.CUBLAS,
    k_group_size: Int = 1,
](
    ctx: DeviceContext,
    m: MType,
    n: NType,
    k: KType,
    compute_fn: ComputeFnType = no_compute_fn,
) raises:
    """Runs an SM90 warp-specialized matmul and validates the result against a vendor reference.

    Allocates host and device buffers for the operands, initializes them with
    random data, launches the warp-specialized GEMM with multicasting, and compares
    the output against a vendor (cuBLAS) reference using an elementwise tolerance
    check.

    Parameters:
        MType: Coordinate-like type encoding the M dimension extent (inferred).
        NType: Coordinate-like type encoding the N dimension extent (inferred).
        KType: Coordinate-like type encoding the K dimension extent (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        a_type: Element dtype of the left-hand operand.
        b_type: Element dtype of the right-hand operand.
        c_type: Element dtype of the output matrix.
        cluster_shape: Thread block cluster shape as `(N, M, K)`.
        block_tile_shape: Block tile shape as `(BM, BN, BK)`.
        wgmma_shape: WGMMA instruction shape as `(M, N, K)`.
        num_consumer: Number of consumer warps in the warp-specialized pipeline.
        num_pipeline_stages: Number of software pipeline stages.
        transpose_b: Whether the right-hand operand is stored transposed.
        partitioned_multicast: Whether to use partitioned instead of broadcast multicast.
        grid_shape: Optional explicit grid shape override.
        use_tma_store: Whether to use TMA store for the epilogue.
        schedule: Tile scheduler hint controlling launch order.
        default_epilogue: Whether to store the output through an epilogue
            closure instead of the kernel's own store.
        has_compute_fn: Whether the kernel stores `compute_fn(idx, a @ b)`;
            takes precedence over `default_epilogue`.
        measure_threshold: Optional relative-difference threshold for a measured assertion.
        backend: Vendor backend used for the reference matmul.
        k_group_size: Group size used for grouped matmul along the K dimension.

    Args:
        ctx: Device context used for device allocations and kernel dispatch.
        m: Number of rows of the output matrix (M dimension).
        n: Number of columns of the output matrix (N dimension).
        k: Contraction dimension shared by the operands (K dimension).
        compute_fn: Maps each output index and value to the value to store
            when `has_compute_fn` is set.
    """
    var M = Int(m)
    var N = Int(n)
    var K = Int(k)

    comptime CLUSTER_N = cluster_shape[0]
    comptime CLUSTER_M = cluster_shape[1]

    # Calculate sizes
    var a_size = M * K
    var b_size = N * K if transpose_b else K * N
    var c_size = M * N

    # Host allocations
    var a_host_alloc = alloc(
        AllocLayout[Scalar[a_type]](length=a_size)
    ).into_managed()
    var a_host = a_host_alloc.unsafe_ptr()
    var b_host_alloc = alloc(
        AllocLayout[Scalar[b_type]](length=b_size)
    ).into_managed()
    var b_host = b_host_alloc.unsafe_ptr()
    var c_host_alloc = alloc(
        AllocLayout[Scalar[c_type]](length=c_size)
    ).into_managed()
    var c_host = c_host_alloc.unsafe_ptr()
    var c_host_ref_alloc = alloc(
        AllocLayout[Scalar[c_type]](length=c_size)
    ).into_managed()
    var c_host_ref: MutPointer[
        Scalar[c_type], origin_of(c_host_ref_alloc)
    ] = c_host_ref_alloc.unsafe_ptr()

    # Device allocations
    var a_dev_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var c_dev_ref_buffer = ctx.enqueue_create_buffer[c_type](c_size)

    # Construct TileTensors for device buffers
    var a_tensor = TileTensor(a_dev_buffer, row_major(m, k)).as_imm()
    var b_tensor = TileTensor(
        b_dev_buffer,
        row_major(
            Coord(
                Idx[NType.static_value if transpose_b else KType.static_value],
                Idx[KType.static_value if transpose_b else NType.static_value],
            ),
        ),
    ).as_imm()
    var c_tensor = TileTensor(c_dev_buffer, row_major(m, n))
    var c_ref_tensor = TileTensor(c_dev_ref_buffer, row_major(m, n))

    # Initialize matmul operands
    rand(a_host, a_size)
    rand(b_host, b_size)

    # Move operands to the Device
    ctx.enqueue_copy(a_dev_buffer, a_host)
    ctx.enqueue_copy(b_dev_buffer, b_host)

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    print(
        "wgmma_shape",
        wgmma_shape,
        a_type,
        "x",
        b_type,
        "x",
        c_type,
        " : PROBLEM SHAPE (M,N,K): (",
        M,
        "x",
        N,
        "x",
        K,
        ") - ",
        "BLOCKS SHAPE (BM,BN,BK): (",
        BM,
        "x",
        BN,
        "x",
        BK,
        ") - ",
        "CLUSTER DIMS (M,N): (",
        CLUSTER_M,
        "x",
        CLUSTER_N,
        ") NUM CONSUMERS: ",
        num_consumer,
        " NUM PIPELINE STAGES: ",
        num_pipeline_stages,
        " MULTICAST MODE: ",
        "PARTITIONED" if partitioned_multicast else "BROADCAST",
        "USE TMA STORE: ",
        use_tma_store,
    )

    assert (ceildiv(M, BM) % (CLUSTER_M)) == 0, String(
        "Number of blocks on M axis should be multiple of cluster dim. M",
        "(M // BM=",
        String(M // BM),
        ") CLUSTER SIZE:",
        String(CLUSTER_M),
    )

    assert (ceildiv(N, BN) % (CLUSTER_N)) == 0, String(
        "Number of blocks on M axis should be multiple of cluster dim. N",
        "N // BN=(",
        String(N // BN),
        ") CLUSTER SIZE:",
        String(CLUSTER_N),
    )

    @inline(.always)
    def epilogue_fn[
        _dtype: DType, width: SIMDLength, *, alignment: Int
    ](idx: Coord, val: SIMD[_dtype, width]) {var c_tensor}:
        comptime assert idx.rank == 2, "the matmul epilogue contract is rank 2"
        c_tensor.store[alignment=alignment * size_of[c_type]()](
            idx, rebind[SIMD[c_type, width]](val)
        )

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=block_tile_shape,
        mma_shape=wgmma_shape,
        cluster_shape=cluster_shape,
        num_pipeline_stages=num_pipeline_stages,
        num_consumer=num_consumer,
        partitioned_multicast=partitioned_multicast,
        k_group_size=k_group_size,
    )

    comptime if has_compute_fn:
        warp_specialize_gemm_with_multicasting[
            transpose_b=transpose_b,
            config=matmul_config,
            schedule=schedule,
            grid_shape=grid_shape,
            use_tma_store=use_tma_store,
        ](c_tensor, a_tensor, b_tensor, compute_fn, ctx)
    elif default_epilogue:
        warp_specialize_gemm_with_multicasting[
            transpose_b=transpose_b,
            config=matmul_config,
            schedule=schedule,
            grid_shape=grid_shape,
            use_tma_store=use_tma_store,
        ](IndexList[2](M, N), a_tensor, b_tensor, epilogue_fn, ctx)
    else:
        warp_specialize_gemm_with_multicasting[
            transpose_b=transpose_b,
            config=matmul_config,
            schedule=schedule,
            grid_shape=grid_shape,
            use_tma_store=use_tma_store,
        ](c_tensor, a_tensor, b_tensor, ctx)

    comptime assert a_type != .float8_e4m3fn or transpose_b, (
        "Testing is only supported for transposed_b==True when"
        " a_type==float8_e4m3fn. Add the non-transposed case if needed."
    )

    vendor_matmul(
        ctx,
        c_ref_tensor,
        a_tensor,
        b_tensor,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.enqueue_copy(c_host, c_dev_buffer)
    ctx.enqueue_copy(c_host_ref, c_dev_ref_buffer)
    ctx.synchronize()

    comptime if has_compute_fn:
        for i in range(M):
            for j in range(N):
                c_host_ref[i * N + j] = compute_fn[c_type, 1, alignment=1](
                    (i, j), c_host_ref[i * N + j]
                )

    comptime if case .Some(threshold) = measure_threshold:
        assert_with_measure[relative_difference](
            c_host,
            c_host_ref,
            c_size,
            threshold=threshold,
        )

    comptime rtol = 1e-2
    assert_almost_equal(
        c_host,
        c_host_ref,
        c_size,
        atol=0.0001,
        rtol=rtol,
    )

    # Cleanup host pointers
    dealloc(a_host_alloc^)
    dealloc(b_host_alloc^)
    dealloc(c_host_alloc^)
    dealloc(c_host_ref_alloc^)
