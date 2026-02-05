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
from collections import Optional
from math import align_up, ceildiv
from sys import (
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    simd_width_of,
)
from layout.tma_async import create_tensor_tile
import gpu.primitives.warp as warp
from algorithm.reduction import _reduce_generator
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    global_idx,
    lane_id,
    thread_idx,
)
from gpu import warp_id as get_warp_id
from gpu.host import (
    DeviceAttribute,
    DeviceBuffer,
    DeviceContext,
    LaunchAttribute,
    get_gpu_target,
    FuncAttribute,
)
from gpu.host.launch_attribute import AccessPolicyWindow, AccessProperty
from gpu.memory import load
from gpu.primitives.grid_controls import (
    PDLLevel,
    pdl_launch_attributes,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from gpu.host.info import H100
from gpu.memory import (
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
    AddressSpace,
)
from layout.layout_tensor import LayoutTensorIter
from gpu.sync import (
    named_barrier,
    named_barrier_arrive,
    syncwarp,
    mbarrier_arrive,
)
from layout.swizzle import make_swizzle

# layout imports
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
)
from layout._ndbuffer_stub import from_ndbuffer_row_major
from logger import Logger
from memory import LegacyUnsafePointer, stack_allocation
from gpu.host.nvidia.tma import TensorMapSwizzle
from layout.tma_async import SharedMemBarrier, TMATensorTile
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

from utils import IndexList
from utils.index import Index
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from linalg.gemv import GEMVAlgorithm, reverse_idx, log_shape
from linalg.utils import GemmShape, elementwise_epilogue_type
from linalg.fp4_utils import (
    get_scale_factor,
    SF_MN_GROUP_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    SF_ATOM_M,
    SF_ATOM_K,
    cast_f4e2m1x16_to_fp16x16,
    cast_f4e2m1x2_to_fp16x2,
    nvfp4_scaled_tile_multiply_accumulate,
)
from memory import bitcast
from gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_pipeline import (
    ProducerConsumerPipeline,
)

comptime logger = Logger()


@always_inline
fn block_scaled_gemv[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    sfa_type: DType,
    sfb_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    //,
    umma_kind: UMMAKind,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c_tensor: LayoutTensor[c_type, c_layout, MutAnyOrigin],  # 1 * n
    a_tensor: LayoutTensor[a_type, a_layout, MutAnyOrigin],  # 1 * k
    b_tensor: LayoutTensor[b_type, b_layout, MutAnyOrigin],  # n * k
    sfa_tensor: LayoutTensor[sfa_type, sfa_layout, MutAnyOrigin],
    sfb_tensor: LayoutTensor[sfb_type, sfb_layout, MutAnyOrigin],
    alpha: Float32,
    ctx: DeviceContext,
) raises:
    __comptime_assert (
        ctx.default_device_info.compute == B200.compute
    ), "Only support B200 for block scaled gemv"
    __comptime_assert (
        transpose_b == True
    ), "Only support transpose_b == True for block scaled gemv"
    __comptime_assert (
        a_type == b_type
    ), "Only support same input type for block scaled gemv"
    __comptime_assert (
        sfa_type == sfb_type
    ), "Only support same scales type for block scaled gemv"
    __comptime_assert (
        umma_kind == UMMAKind.KIND_MXF4NVF4
        and a_type == DType.uint8
        and sfa_type == NVFP4_SF_DTYPE
        and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
    ), "Only support MXF4NVF4 scaling kind for block scaled gemv"

    comptime WARPS_PER_BLOCK = 1024 // WARP_SIZE
    # comptime WARPS_PER_BLOCK = B200.threads_per_multiprocessor // WARP_SIZE

    # on B200 we can use 32-bit SIMD width for loads
    comptime simd_width = 32 if umma_kind == UMMAKind.KIND_MXF4NVF4 else simd_width_of[
        a_type, target = get_gpu_target()
    ]()

    var m = c_tensor.dim[0]()
    var n = c_tensor.dim[1]()
    var k = a_tensor.dim[1]()

    comptime static_K = a_layout.shape[1].value()
    comptime static_N = c_layout.shape[1].value()

    comptime K_PACK = 2 if umma_kind == UMMAKind.KIND_MXF4NVF4 else 1  # each uint8 element has two Float4-E2M1 values

    @parameter
    if static_K % simd_width == 0:
        __comptime_assert (
            simd_width * K_PACK == SF_VECTOR_SIZE * SF_ATOM_K
        ), "simd_width * K_PACK must be equal to SF_VECTOR_SIZE * SF_ATOM_K"

        fn get_parameters(N: Int, K: Int) -> IndexList[2]:
            if K >= 7168:
                return Index(16, 128)
            else:
                return Index(4, 64)

        comptime tile_n = get_parameters(static_N, static_K)[0]
        comptime num_threads = get_parameters(static_N, static_K)[1]

        if (
            ceildiv(n, tile_n)
            <= ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_Y)
            and static_N % tile_n == 0
        ):
            logger.info("Executing: BLOCK SCALED GEMV SPLIT_K kernel")
            comptime tile_m = 1
            comptime check_bounds = static_N % tile_n != 0

            comptime kernel = block_scaled_gemv_split_k[
                c_type,
                a_type,
                b_type,
                sfa_type,
                sfb_type,
                c_layout,
                a_layout,
                b_layout,
                sfa_layout,
                sfb_layout,
                simd_width = UInt(simd_width),
                tile_m = UInt(tile_m),
                tile_n = UInt(tile_n),
                num_threads = UInt(num_threads),
                umma_kind=umma_kind,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                elementwise_lambda_fn=elementwise_lambda_fn,
                check_bounds=check_bounds,
                pdl_level=pdl_level,
            ]
            ctx.enqueue_function[kernel, kernel, dump_asm=False](
                c_tensor,
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                alpha,
                m,
                n,
                k,
                grid_dim=(ceildiv(m, tile_m), ceildiv(n, tile_n)),
                block_dim=num_threads,
                attributes=pdl_launch_attributes(pdl_level),
            )
        else:
            logger.info("Executing: BLOCK SCALED GEMV VECTORIZED kernel")

            var block_dim = min(
                align_up(k // simd_width, WARP_SIZE),
                WARP_SIZE * WARPS_PER_BLOCK,
            )

            comptime kernel = block_scaled_gemv_vectorized_kernel[
                c_type,
                b_type,
                a_type,
                sfb_type,
                sfa_type,
                c_layout,
                b_layout,
                a_layout,
                sfb_layout,
                sfa_layout,
                umma_kind=umma_kind,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                simd_width = UInt(simd_width),
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                pdl_level=pdl_level,
            ]
            ctx.enqueue_function[kernel, kernel](
                c_tensor,
                b_tensor,
                a_tensor,
                sfb_tensor,
                sfa_tensor,
                alpha,
                n,
                m,
                k,
                grid_dim=ceildiv(n, block_dim // WARP_SIZE),
                block_dim=block_dim,
                attributes=pdl_launch_attributes(pdl_level),
            )
    else:
        logger.info("Executing: BLOCK SCALED GEMV kernel")

        comptime kernel = block_scaled_gemv_kernel[
            c_type,
            b_type,
            a_type,
            sfb_type,
            sfa_type,
            c_layout,
            b_layout,
            a_layout,
            sfb_layout,
            sfa_layout,
            umma_kind=umma_kind,
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            pdl_level=pdl_level,
        ]
        ctx.enqueue_function[kernel, kernel](
            c_tensor,
            b_tensor,
            a_tensor,
            sfb_tensor,
            sfa_tensor,
            alpha,
            n,
            m,
            k,
            grid_dim=ceildiv(n, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
            attributes=pdl_launch_attributes(pdl_level),
        )


# NVFP4 Matrix-Column Vector Multiplication using vectorized instructions
fn block_scaled_gemv_vectorized_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    sfa_type: DType,
    sfb_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    *,
    simd_width: UInt,
    umma_kind: UMMAKind,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
    pdl_level: PDLLevel = PDLLevel(),
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],  # m
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],  # m * k
    b: LayoutTensor[b_type, b_layout, MutAnyOrigin],  # 1 * k
    sfa: LayoutTensor[sfa_type, sfa_layout, MutAnyOrigin],
    sfb: LayoutTensor[sfb_type, sfb_layout, MutAnyOrigin],
    alpha: Float32,
    m: Int,
    n: Int,
    k: Int,
):
    var tid = global_idx.x
    var warp_id = Int(warp.broadcast(tid // UInt(WARP_SIZE)))
    comptime step = WARP_SIZE * Int(simd_width)

    var idx = lane_id() * simd_width

    if warp_id >= m:
        return

    # Every warp processes a single row of the resultant vector
    var local_accum = Scalar[accum_type](0)
    comptime local_accum_type = type_of(local_accum)

    comptime K_PACK = 2 if umma_kind == UMMAKind.KIND_MXF4NVF4 else 1  # each uint8 element has two Float4-E2M1 values
    comptime ELEMENTS_PER_SF_VECTOR = SF_VECTOR_SIZE // K_PACK

    @parameter
    if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    # Each uint8 element has two Float4-E2M1 values, so each iteration processes 2 * simd_width elements (64 for NVFP4)
    for i in range(ceildiv(k // Int(simd_width), WARP_SIZE)):
        var a_tile = a.tile[1, WARP_SIZE * Int(simd_width)](warp_id, i)
        var b_tile = b.tile[1, WARP_SIZE * Int(simd_width)](0, i)

        if idx >= UInt(k):
            continue

        var a_vec = a_tile.vectorize[1, Int(simd_width)]()[0, Int(lane_id())]
        var b_vec = b_tile.vectorize[1, Int(simd_width)]()[0, Int(lane_id())]

        var sfa_vec = get_scale_factor[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE, width=SF_ATOM_K
        ](sfa, warp_id, Int(idx * K_PACK))
        var sfb_vec = get_scale_factor[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE, width=SF_ATOM_K
        ](sfb, 0, Int(idx * K_PACK))

        var temp_accum = nvfp4_scaled_tile_multiply_accumulate(
            bitcast[DType.uint32, 8](a_vec),
            bitcast[DType.uint32, 8](b_vec),
            bitcast[DType.uint32, 1](sfa_vec),
            bitcast[DType.uint32, 1](sfb_vec),
            local_accum.cast[DType.float32](),
        )
        local_accum = rebind[Scalar[accum_type]](temp_accum.cast[accum_type]())

        idx += UInt(step)

    var accum = warp.sum(local_accum)
    accum = accum * alpha.cast[accum_type]()

    if lane_id() == 0:

        @parameter
        if elementwise_lambda_fn:
            comptime elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                reverse_idx[transpose_b](warp_id, 0),
                accum.cast[c_type](),
            )
        else:

            @parameter
            if transpose_b:
                c[0, warp_id] = accum.cast[c_type]()
            else:
                c[warp_id, 0] = accum.cast[c_type]()

    @parameter
    if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()


fn block_scaled_gemv_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    sfa_type: DType,
    sfb_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    *,
    umma_kind: UMMAKind,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
    pdl_level: PDLLevel = PDLLevel(),
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],  # m
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],  # m * k
    b: LayoutTensor[b_type, b_layout, MutAnyOrigin],  # 1 * k
    sfa: LayoutTensor[sfa_type, sfa_layout, MutAnyOrigin],
    sfb: LayoutTensor[sfb_type, sfb_layout, MutAnyOrigin],
    alpha: Float32,
    m: Int,
    n: Int,
    k: Int,
):
    comptime K_PACK = 2 if umma_kind == UMMAKind.KIND_MXF4NVF4 else 1  # each uint8 element has two Float4-E2M1 values

    var tid = global_idx.x
    var warp_id = warp.broadcast(tid // UInt(WARP_SIZE))

    if warp_id >= UInt(m):
        return

    var accum = Scalar[accum_type](0)

    @parameter
    if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    # Every warp processes a single row of the resultant vector
    for i in range(ceildiv(k, WARP_SIZE)):
        var idx = i * WARP_SIZE + Int(lane_id())

        if idx < k:
            # each uint8 element has two Float4-E2M1 values,
            var a_val_fp16x2 = cast_f4e2m1x2_to_fp16x2(
                rebind[UInt8](a[warp_id, idx])
            ).cast[accum_type]()
            var b_val_fp16x2 = cast_f4e2m1x2_to_fp16x2(
                rebind[UInt8](b[0, idx])
            ).cast[accum_type]()

            var a_scale = get_scale_factor[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE, width=1
            ](sfa, Int(warp_id), Int(idx * K_PACK))
            var b_scale = get_scale_factor[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE, width=1
            ](sfb, Int(0), Int(idx * K_PACK))

            @parameter
            for k_idx in range(K_PACK):
                var a_val = rebind[Scalar[accum_type]](a_val_fp16x2[k_idx])
                var b_val = rebind[Scalar[accum_type]](b_val_fp16x2[k_idx])
                var a_scale_val = a_scale.cast[accum_type]()
                var b_scale_val = b_scale.cast[accum_type]()

                accum += a_val * b_val * a_scale_val * b_scale_val

    accum = warp.sum(accum)
    accum *= alpha.cast[accum_type]()

    if lane_id() == 0:

        @parameter
        if elementwise_lambda_fn:
            comptime elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                reverse_idx[transpose_b](Int(warp_id), 0),
                accum.cast[c_type](),
            )
        else:

            @parameter
            if transpose_b:
                c[0, warp_id] = accum.cast[c_type]()
            else:
                c[warp_id, 0] = accum.cast[c_type]()

    @parameter
    if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
fn block_scaled_gemv_split_k[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    sfa_type: DType,
    sfb_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    *,
    simd_width: UInt,
    tile_m: UInt,
    tile_n: UInt,
    num_threads: UInt,
    umma_kind: UMMAKind,
    SF_VECTOR_SIZE: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
    check_bounds: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
](
    output: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    act: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    weight: LayoutTensor[b_type, b_layout, MutAnyOrigin],
    sfa: LayoutTensor[sfa_type, sfa_layout, MutAnyOrigin],
    sfb: LayoutTensor[sfb_type, sfb_layout, MutAnyOrigin],
    alpha: Float32,
    m: Int,
    n: Int,
    k: Int,
):
    """Block scaled GEMV with tiling in K dimension.
    Assuming the B (weight) matrix is transposed i.e. row major N x K, this kernel
    implements a vector (1 x K) times a matrix (N x K).
    The impl can actually handle M > 1 but it's only optimal for tiny M. We use
    it for M = 1 only.
    """
    # tile_m represents how many rows each thread will process of the output activation matrix
    # tile_n represents how many rows each thread will process of the weight matrix.
    # Nvidia vectorized load is 16B.

    __comptime_assert (
        Int(num_threads).is_power_of_two() and num_threads % WARP_SIZE == 0
    ), "num_threads must be a power of two and divisible by WARP_SIZE"
    __comptime_assert tile_m == 1, "tile_m must be 1"
    __comptime_assert Int(
        tile_n
    ).is_power_of_two(), "tile_n must be a power of two"

    # each uint8 element has two Float4-E2M1 values
    comptime K_PACK = 2 if umma_kind == UMMAKind.KIND_MXF4NVF4 else 1
    comptime ELEMENTS_PER_SF_VECTOR = SF_VECTOR_SIZE // K_PACK

    comptime tile_k = simd_width * num_threads
    # which rows of the activation matrix each thread will process
    var tile_id_m = block_idx.x * tile_m
    # which rows of the weight matrix each thread will process
    var tile_id_n = block_idx.y * tile_n
    var tid = thread_idx.x
    var tile_w = LayoutTensor[
        b_type,
        Layout.row_major(Int(tile_n), Int(simd_width)),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()
    # these are the partial accumlations for each thread this a matrix of values
    # since each thread will process a tile_m x tile_n partials of the output vector
    var acc = (
        LayoutTensor[
            accum_type,
            Layout.row_major(Int(tile_m), Int(tile_n)),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )
    var output_idx = tile_id_m * UInt(n) + tile_id_n
    var iteration = 0
    comptime WeightVecType = SIMD[b_type, Int(simd_width)]

    @parameter
    if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    # Each thread sums local data in K.
    for k_idx in range(tid * simd_width, k, tile_k):
        var weight_tile = weight.tile[Int(tile_n), Int(tile_k)](
            Int(block_idx.y), iteration
        )
        var act_tile = act.tile[Int(tile_m), Int(tile_k)](
            Int(block_idx.x), iteration
        )

        @parameter
        for i in range(tile_n):
            # Here we load data @ thread_idx.x from the weight matrix
            # and store it into tile_w. We skip this if if the current
            # row we are reading from (i + tile_id_n) is greater than the number
            # of rows in the weight matrix.
            @parameter
            if check_bounds:
                if i + tile_id_n >= UInt(n):
                    continue
            var b_vec = weight_tile.vectorize[1, Int(simd_width)]()[
                i, thread_idx.x
            ]
            tile_w.store[Int(simd_width)](
                Int(i), 0, rebind[WeightVecType](b_vec)
            )

        @parameter
        for i in range(tile_m):
            # Here we load data @ thread_idx.x from the activation matrix
            # and store it into tile_a. We skip this if if the current
            # row we are reading from (i + tile_id_m) is greater than the number
            # of rows in the activation matrix. This should never be the case if
            # tile_m is 1.
            @parameter
            if check_bounds:
                if i + tile_id_m >= UInt(m):
                    continue
            var act_vec = act_tile.vectorize[1, Int(simd_width)]()[
                i, thread_idx.x
            ]

            # Now we multiply tile_a by tile_w and store the partials
            # in acc
            @parameter
            for j in range(tile_n):
                var weight_vec = tile_w.vectorize[1, Int(simd_width)]()[j, 0]

                var local_accum = rebind[Scalar[accum_type]](acc[i, j])

                var sfa_vec = get_scale_factor[
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE, width=SF_ATOM_K
                ](sfa, Int(tile_id_m + i), Int(k_idx * K_PACK))
                var sfb_vec = get_scale_factor[
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE, width=SF_ATOM_K
                ](sfb, Int(tile_id_n + j), Int(k_idx * K_PACK))

                var temp_accum = nvfp4_scaled_tile_multiply_accumulate(
                    bitcast[DType.uint32, 8](weight_vec),
                    bitcast[DType.uint32, 8](act_vec),
                    bitcast[DType.uint32, 1](sfa_vec),
                    bitcast[DType.uint32, 1](sfb_vec),
                    local_accum.cast[DType.float32](),
                )
                local_accum = rebind[Scalar[accum_type]](
                    temp_accum.cast[accum_type]()
                )

                acc.store[1](Int(i), Int(j), local_accum)

        iteration += 1

    # Warps are arranged along K.
    comptime k_warp_num = num_threads // UInt(WARP_SIZE)
    var warp_id = warp.broadcast(tid // UInt(WARP_SIZE))
    var shmem = LayoutTensor[
        accum_type,
        Layout.row_major(1, Int(tile_m * tile_n * k_warp_num)),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var smem_results = SIMD[accum_type, Int(tile_m * tile_n)](0)

    @parameter
    for mi in range(tile_m):

        @parameter
        for ni in range(tile_n):
            var val = warp.sum(acc[mi, ni])
            smem_results[Int(mi * tile_n + ni)] = val

    if lane_id() == 0:
        shmem.store[
            width = Int(tile_m * tile_n),
            store_alignment = align_of[type_of(smem_results)](),
        ](0, Int(warp_id * tile_m * tile_n), smem_results)
    barrier()

    # Sum across warps' results in shared memory then output.
    # TODO: should be able to vectorize and maybe use larger tile_n.
    for ii in range(tid, tile_m * tile_n, num_threads):
        var mid = ii // tile_n
        var nid = ii % tile_n
        var val = Scalar[accum_type]()
        comptime ValType = type_of(val)

        @parameter
        for jj in range(k_warp_num):
            val += rebind[ValType](shmem[0, jj * tile_m * tile_n + ii])

        val *= alpha.cast[accum_type]()

        @parameter
        if elementwise_lambda_fn:
            comptime elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                Index(0, output_idx + mid * UInt(n) + nid), val.cast[c_type]()
            )
        else:
            var idx = output_idx + mid * UInt(n) + nid

            @parameter
            if check_bounds:
                if idx >= UInt(n):
                    continue
            output[0, idx] = val.cast[c_type]()

    @parameter
    if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()
