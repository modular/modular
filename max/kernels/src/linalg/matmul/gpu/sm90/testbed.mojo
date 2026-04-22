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

from std.math import ceildiv
from std.sys import align_of

from std.gpu.host import DeviceContext
from layout import Coord, CoordLike, Idx, TileTensor, row_major
from std.utils.index import IndexList
from internal_utils import assert_almost_equal, assert_with_measure
from std.random import rand
from internal_utils._measure import relative_difference
from std.collections import OptionalReg
from std.utils.index import IndexList

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ....utils_gpu import MatmulConfig
from ...vendor.blas import Backend
from ...vendor.blas import matmul as vendor_matmul
from ..tile_scheduler import MatmulSchedule
from .matmul import warp_specialize_gemm_with_multicasting


def test_matmul_sm90[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
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
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    measure_threshold: Optional[Float64] = None,
    backend: Backend = Backend.CUBLAS,
    k_group_size: Int = 1,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    var M = m.value()
    var N = n.value()
    var K = k.value()

    comptime CLUSTER_N = cluster_shape[0]
    comptime CLUSTER_M = cluster_shape[1]

    # Calculate sizes
    var a_size = M * K
    var b_size = N * K if transpose_b else K * N
    var c_size = M * N

    # Host allocations
    var a_host_ptr = alloc[Scalar[a_type]](a_size)
    var b_host_ptr = alloc[Scalar[b_type]](b_size)
    var c_host_ptr = alloc[Scalar[c_type]](c_size)
    var c_host_ref_ptr = alloc[Scalar[c_type]](c_size)

    # Device allocations
    var a_dev_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var c_dev_ref_buffer = ctx.enqueue_create_buffer[c_type](c_size)

    # Construct TileTensors for device buffers
    var a_tensor = TileTensor(a_dev_buffer, row_major(Coord(m, k))).as_immut()
    var b_tensor = TileTensor(
        b_dev_buffer,
        row_major(
            Coord(
                Idx[
                    NType.static_value if transpose_b else KType.static_value
                ](),
                Idx[
                    KType.static_value if transpose_b else NType.static_value
                ](),
            ),
        ),
    ).as_immut()
    var c_tensor = TileTensor(c_dev_buffer, row_major(Coord(m, n)))
    var c_ref_tensor = TileTensor(c_dev_ref_buffer, row_major(Coord(m, n)))

    # Initialize matmul operands
    rand(a_host_ptr, a_size)
    rand(b_host_ptr, b_size)

    # Move operands to the Device
    ctx.enqueue_copy(a_dev_buffer, a_host_ptr)
    ctx.enqueue_copy(b_dev_buffer, b_host_ptr)

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

    @parameter
    @always_inline
    @__copy_capture(c_tensor)
    def epilogue_fn[
        _dtype: DType,
        width: SIMDSize,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        c_tensor.store_linear[alignment=alignment](
            idx, rebind[SIMD[c_type, width]](val)
        )

    comptime elf = Optional[elementwise_epilogue_type](
        epilogue_fn
    ) if default_epilogue and elementwise_compute_lambda_fn is None else None

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=block_tile_shape,
        mma_shape=wgmma_shape,
        cluster_shape=cluster_shape,
        num_pipeline_stages=num_pipeline_stages,
        num_consumer=num_consumer,
        partitioned_multicast=partitioned_multicast,
        k_group_size=k_group_size,
    )

    warp_specialize_gemm_with_multicasting[
        transpose_b=transpose_b,
        config=matmul_config,
        schedule=schedule,
        grid_shape=grid_shape,
        use_tma_store=use_tma_store,
        elementwise_lambda_fn=elf,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        ctx,
    )

    comptime assert a_type != DType.float8_e4m3fn or transpose_b, (
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

    ctx.enqueue_copy(c_host_ptr, c_dev_buffer)
    ctx.enqueue_copy(c_host_ref_ptr, c_dev_ref_buffer)
    ctx.synchronize()

    comptime if elementwise_compute_lambda_fn:
        # Apply the compute lambda directly on the reference tensor
        comptime compute_lambda = elementwise_compute_lambda_fn.value()
        for i in range(M):
            for j in range(N):
                c_host_ref_ptr[i * N + j] = compute_lambda(
                    IndexList[2](i, j),
                    c_host_ref_ptr[i * N + j],
                )

    comptime if measure_threshold:
        assert_with_measure[relative_difference](
            c_host_ptr,
            c_host_ref_ptr,
            c_size,
            threshold=measure_threshold.value(),
        )

    comptime rtol = 1e-2
    assert_almost_equal(
        c_host_ptr,
        c_host_ref_ptr,
        c_size,
        atol=0.0001,
        rtol=rtol,
    )

    # Cleanup host pointers
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
