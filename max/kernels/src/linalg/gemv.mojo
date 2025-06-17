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
from collections import OptionalReg
from math import align_up, ceildiv
from sys import (
    alignof,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    simdwidthof,
)

import gpu.warp as warp
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
from gpu.host import DeviceAttribute, DeviceContext, LaunchAttribute
from gpu.host._compile import get_gpu_target
from gpu.host.launch_attribute import AccessPolicyWindow, AccessProperty
from gpu.memory import AddressSpace, CacheOperation, load
from gpu.tensor_ops import tc_reduce_gevm_8x
from memory import memset_zero, stack_allocation

from utils import IndexList
from utils.index import Index
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from .matmul_gpu import matmul_kernel_naive
from .utils import GemmShape, elementwise_epilogue_type


@fieldwise_init
struct GEMVAlgorithm(Copyable, Movable):
    var _value: Int

    alias GEMV_KERNEL = Self(0)
    alias GEMV_KERNEL_VECTOR = Self(1)
    alias GEMV_SPLIT_K = Self(2)
    alias GEVM_KERNEL_VECTOR = Self(3)
    alias GEVM_KERNEL = Self(4)
    alias MATMUL_NAIVE = Self(5)

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other


@always_inline
fn reverse_idx[transpose: Bool](x: Int, y: Int) -> IndexList[2]:
    return Index(y, x) if transpose else Index(x, y)


# Matrix-Column Vector Multiplication using scalar arithmetic
fn gemv_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    *,
    reduction_method: warp.ReductionMethod,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
](
    c: UnsafePointer[Scalar[c_type]],
    a: UnsafePointer[Scalar[a_type]],
    b: UnsafePointer[Scalar[b_type]],
    m: Int,
    n: Int,
    k: Int,
):
    var tid = global_idx.x
    var warp_id = warp.broadcast(tid // WARP_SIZE)

    if warp_id >= m:
        return

    var accum = Scalar[s_type](0)

    # Every warp processes a single row of the resultant vector
    for i in range(ceildiv(k, WARP_SIZE)):
        var idx = i * WARP_SIZE + lane_id()
        if idx < k:
            accum += (
                a.load(warp_id * k + idx).cast[s_type]()
                * b.load(idx).cast[s_type]()
            )

    accum = warp.sum[
        a_type, reduction_method=reduction_method, output_type=s_type
    ](accum)

    if lane_id() == 0:

        @parameter
        if elementwise_lambda_fn:
            alias elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                reverse_idx[transpose_b](warp_id, 0),
                accum.cast[c_type](),
            )
        else:
            c[warp_id] = accum.cast[c_type]()


# Matrix-Column Vector Multiplication using vectorized instructions
fn gemv_kernel_vector[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    *,
    reduction_method: warp.ReductionMethod,
    simd_width: UInt,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
](
    c: NDBuffer[mut=True, c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b: NDBuffer[b_type, 2, MutableAnyOrigin, b_shape],
    m: UInt,
    n: UInt,
    k: UInt,
):
    alias align_a = alignof[SIMD[a_type, simd_width]]()
    alias align_b = alignof[SIMD[b_type, simd_width]]()
    var tid = global_idx.x
    var warp_id = warp.broadcast(tid // WARP_SIZE)

    var a_ptr = (
        a.data + (warp_id * a.dim[1]()) + lane_id() * simd_width
    ).as_noalias_ptr()
    var idx = lane_id() * simd_width
    alias step = WARP_SIZE * simd_width

    if warp_id >= m:
        return

    # Every warp processes a single row of the resultant vector
    var local_accum = SIMD[s_type, simd_width](0)
    for _ in range(ceildiv(k // simd_width, WARP_SIZE)):
        if idx >= k:
            continue
        var ax = load[
            width=simd_width,
            prefetch_size=128,
            alignment=align_a,
            cache_policy = CacheOperation.LAST_USE,
        ](a_ptr).cast[s_type]()
        var bx = load[
            width=simd_width,
            prefetch_size=128,
            alignment=align_b,
            cache_policy = CacheOperation.ALWAYS,
        ](b._offset(reverse_idx[transpose_b](idx, 0))).cast[s_type]()

        # Do simd vector loads in ax,bx to multiply element wise for matrix
        # row and vector column
        local_accum += ax * bx

        idx += step
        a_ptr += step

    var accum = warp.sum[
        a_type, reduction_method=reduction_method, output_type=s_type
    ](local_accum)

    if lane_id() == 0:

        @parameter
        if elementwise_lambda_fn:
            alias elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                reverse_idx[transpose_b](warp_id, 0),
                accum.cast[c_type](),
            )
        else:
            c.store(
                reverse_idx[transpose_b](warp_id, 0),
                accum.cast[c_type](),
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn gemv_split_k[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    simd_width: UInt,
    tile_m: UInt,
    tile_n: UInt,
    num_threads: UInt,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
](
    output: NDBuffer[mut=True, c_type, 2, MutableAnyOrigin, c_shape],
    act: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    weight: NDBuffer[b_type, 2, MutableAnyOrigin, b_shape],
    m: UInt,
    n: UInt,
    k: UInt,
):
    """GEMV with tiling in K dimension.
    Assuming the B (weight) matrix is transposed i.e. row major N x K, this kernel
    implements a vector (1 x K) times a matrix (N x K).

    The impl can actually handle M > 1 but it's only optimal fro tiny M. We use
    it for M = 1 only.
    """
    # Nvidia vectorized load is 16B.
    alias tile_k = simd_width * num_threads
    var tile_id_m = block_idx.x * tile_m
    var tile_id_n = block_idx.y * tile_n
    var tid = thread_idx.x
    var tile_a = stack_allocation[
        simd_width, a_type, address_space = AddressSpace.LOCAL
    ]()
    var tile_w = stack_allocation[
        tile_n * simd_width, b_type, address_space = AddressSpace.LOCAL
    ]()
    var acc = stack_allocation[
        tile_m * tile_n, s_type, address_space = AddressSpace.LOCAL
    ]()

    alias align_act = alignof[SIMD[a_type, simd_width]]()
    alias align_weight = alignof[SIMD[b_type, simd_width]]()

    memset_zero[count = tile_m * tile_n](acc)

    var act_idx = tile_id_m * k
    var weight_idx = tile_id_n * k
    var output_idx = tile_id_m * n + tile_id_n

    # Each thread sums local data in K.
    for idxK in range(tid * simd_width, k, tile_k):

        @parameter
        for i in range(tile_n):
            var b_vec = weight.data.load[
                width=simd_width, alignment=align_weight
            ](weight_idx + i * k + idxK)

            tile_w.store[alignment=align_weight](i * simd_width, b_vec)

        @parameter
        for i in range(tile_m):
            var a_vec = act.data.load[width=simd_width, alignment=align_act](
                act_idx + i * k + idxK
            )

            tile_a.store[alignment=align_act](i * simd_width, a_vec)

            @parameter
            for j in range(tile_n):

                @parameter
                for l in range(simd_width):
                    acc[i * tile_n + j] += (
                        tile_a[l].cast[s_type]()
                        * tile_w[j * simd_width + l].cast[s_type]()
                    )

    # Warps are arranged along K.
    alias k_warp_num = num_threads // WARP_SIZE
    var warp_id = warp.broadcast(tid // WARP_SIZE)
    var lane_id = tid % WARP_SIZE
    var shmem = stack_allocation[
        k_warp_num * tile_m * tile_n,
        s_type,
        address_space = AddressSpace.SHARED,
    ]()

    # Each warp sums across its threads and stages results in shared memory.
    # Shared memory data is row mojor (num_warps, tile_m, tile_n) stored in 1D.
    @parameter
    for mi in range(tile_m):

        @parameter
        for ni in range(tile_n):
            var val = warp.sum(acc[mi * tile_n + ni])
            if lane_id == 0:
                shmem[mi * tile_n + ni + warp_id * tile_m * tile_n] = val

    barrier()

    # Sum across warps' results in shared memory then output.
    # TODO: should be able to vectorize and maybe use larger tile_n.
    for ii in range(tid, tile_m * tile_n, num_threads):
        var mid = ii // tile_n
        var nid = ii % tile_n
        var val = Scalar[s_type]()

        @parameter
        for jj in range(k_warp_num):
            val += shmem[jj * tile_m * tile_n + ii]

        @parameter
        if elementwise_lambda_fn:
            alias elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                Index(0, output_idx + mid * n + nid), val.cast[c_type]()
            )
        else:
            output.data.store(output_idx + mid * n + nid, val.cast[c_type]())


# Row Vector-Matrix multiplication
fn gevm_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    *,
    tile_size: Int,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
](
    c: UnsafePointer[Scalar[c_type]],
    a: UnsafePointer[Scalar[a_type]],
    b: UnsafePointer[Scalar[b_type]],
    m: Int,
    n: Int,
    k: Int,
):
    var warps_per_block = block_dim.x // WARP_SIZE
    var warp_id = get_warp_id()
    var accum = Scalar[s_type]()
    var col = block_idx.x * WARP_SIZE + lane_id()
    var tid = global_idx.x
    var global_warp_id = tid // WARP_SIZE

    var x_shared = stack_allocation[
        tile_size,
        s_type,
        address_space = AddressSpace.SHARED,
    ]()

    # Every block computes warp size length of output values
    for i in range(ceildiv(UInt(k), warps_per_block)):
        var row = i * warps_per_block + warp_id
        var lhs = a.load(row)
        var rhs = b.load(row * n + col)
        accum += lhs.cast[s_type]() * rhs.cast[s_type]()

    x_shared[lane_id() * WARP_SIZE + warp_id] = accum
    barrier()

    var total = x_shared.load(thread_idx.x).cast[s_type]()
    total = warp.sum(total)

    if lane_id() == 0:

        @parameter
        if elementwise_lambda_fn:
            alias elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                Index(0, global_warp_id), total.cast[c_type]()
            )
        else:
            c[global_warp_id] = total.cast[c_type]()


fn gevm_tc_kernel_vector_8x[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    tile_size: Int,
    simd_width: Int,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
](
    c: NDBuffer[c_type, 2, MutableAnyOrigin],
    a: NDBuffer[a_type, 2, MutableAnyOrigin],
    b: NDBuffer[b_type, 2, MutableAnyOrigin],
    m: UInt,
    n: UInt,
    k: UInt,
):
    alias align_b = alignof[SIMD[b_type, simd_width]]()
    alias align_x = alignof[SIMD[s_type, simd_width]]()

    var warps_per_block = block_dim.x // WARP_SIZE
    var warp_id = get_warp_id()
    var accum = SIMD[s_type, simd_width]()
    var col = block_idx.x * WARP_SIZE * simd_width + lane_id() * simd_width
    var tid = global_idx.x
    var global_warp_id = warp.broadcast(tid // WARP_SIZE)

    var x_shared = stack_allocation[
        tile_size,
        a_type,
        address_space = AddressSpace.SHARED,
    ]()

    # Every block computes warp size * simd_width length of output values
    for i in range(ceildiv(k, warps_per_block)):
        var row = i * warps_per_block + warp_id
        if row < k and col < n:
            var lhs = a.load(Index(0, row))
            var rhs = b.load[width=simd_width, alignment=align_b](
                Index(row, col)
            )
            accum += lhs.cast[s_type]() * rhs.cast[s_type]()

    var xs = warp_id * WARP_SIZE * simd_width + lane_id() * simd_width

    @parameter
    for x in range(simd_width):
        x_shared[xs + x] = accum[x].cast[a_type]()

    barrier()

    var val1 = SIMD[s_type, simd_width // 2]()
    var val2 = SIMD[s_type, simd_width // 2]()

    # indexing to fetch correctly from shared memory
    var stride = UInt(256)
    var mma_tile_width = UInt(8)
    var mma_col_elem_width = UInt(4)
    var target_row = (lane_id() % mma_col_elem_width) * mma_col_elem_width
    var target_col = warp_id * mma_tile_width + (
        lane_id() // mma_col_elem_width
    )

    @parameter
    for i in range(simd_width // 2):
        val1[i] = x_shared[(target_row + i) * stride + target_col].cast[
            s_type
        ]()
        val2[i] = x_shared[(target_row + 16 + i) * stride + target_col].cast[
            s_type
        ]()

    # Doing tensor core reduction to get final results in first row
    var res = tc_reduce_gevm_8x[s_type, a_type, simd_width // 2](
        val1.cast[a_type](), val2.cast[a_type]()
    )

    if lane_id() < 4:
        var final = res.split()

        @parameter
        if elementwise_lambda_fn:
            alias elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, (simd_width // 2) // 2](
                Index(0, global_warp_id * simd_width + lane_id() * 2),
                final[0].cast[c_type](),
            )
        else:
            c.store[
                width = (simd_width // 2) // 2,
                alignment = alignof[SIMD[c_type, (simd_width // 2) // 2]](),
            ](
                Index(0, global_warp_id * simd_width + lane_id() * 2),
                final[0].cast[c_type](),
            )


@always_inline
fn gemv_gpu_dispatch[
    transpose_b: Bool = False,
    reduction_method: warp.ReductionMethod = warp.ReductionMethod.WARP,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    kernel_func: GEMVAlgorithm,
    c: NDBuffer[rank=2, *_, **_],
    a: NDBuffer[rank=2, *_, **_],
    b: NDBuffer[rank=2, *_, **_],
    ctx: DeviceContext,
) raises:
    var shape = GemmShape.get[transpose_b=False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K
    alias WARPS_PER_BLOCK = 1024 // WARP_SIZE
    alias simd_width = simdwidthof[a.type, target = get_gpu_target()]()

    if kernel_func is GEMVAlgorithm.GEMV_SPLIT_K:
        alias num_threads = 128
        alias tile_m = 1
        alias tile_n = 2
        alias kernel = gemv_split_k[
            c.type,
            c.shape,
            a.type,
            a.shape,
            b.type,
            b.shape,
            simd_width=simd_width,
            tile_m=tile_m,
            tile_n=tile_n,
            num_threads=num_threads,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ]
        ctx.enqueue_function[kernel](
            c,
            a,
            b,
            m,
            n,
            k,
            grid_dim=(ceildiv(m, tile_m), ceildiv(n, tile_n)),
            block_dim=num_threads,
        )

    elif kernel_func is GEMVAlgorithm.GEMV_KERNEL_VECTOR:
        if transpose_b == False:
            var block_dim = min(
                align_up(k // simd_width, WARP_SIZE),
                WARP_SIZE * WARPS_PER_BLOCK,
            )

            @parameter
            if has_nvidia_gpu_accelerator():
                var max_access_policy_window_size = ctx.get_attribute(
                    DeviceAttribute.MAX_ACCESS_POLICY_WINDOW_SIZE
                )
                var launch_attributes = List[LaunchAttribute](
                    AccessPolicyWindow(
                        base_ptr=a.data,
                        count=min(a.size(), max_access_policy_window_size),
                        hit_ratio=1,
                        hit_prop=AccessProperty.PERSISTING,
                        miss_prop=AccessProperty.STREAMING,
                    ),
                )
                alias kernel = gemv_kernel_vector[
                    c.type,
                    c.shape,
                    a.type,
                    a.shape,
                    b.type,
                    b.shape,
                    simd_width=simd_width,
                    reduction_method = warp.ReductionMethod.WARP,
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ]
                ctx.enqueue_function[kernel](
                    c,
                    a,
                    b,
                    UInt(m),
                    UInt(n),
                    UInt(k),
                    grid_dim=ceildiv(m, block_dim // WARP_SIZE),
                    block_dim=block_dim,
                    attributes=launch_attributes,
                )
            else:
                alias kernel = gemv_kernel_vector[
                    c.type,
                    c.shape,
                    a.type,
                    a.shape,
                    b.type,
                    b.shape,
                    simd_width=simd_width,
                    reduction_method = warp.ReductionMethod.WARP,
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ]
                ctx.enqueue_function[kernel](
                    c,
                    a,
                    b,
                    UInt(m),
                    UInt(n),
                    UInt(k),
                    grid_dim=ceildiv(m, block_dim // WARP_SIZE),
                    block_dim=block_dim,
                )
        else:
            var block_dim = min(
                align_up(k // simd_width, WARP_SIZE),
                WARP_SIZE * WARPS_PER_BLOCK,
            )
            alias kernel = gemv_kernel_vector[
                c.type,
                c.shape,
                b.type,
                b.shape,
                a.type,
                a.shape,
                simd_width=simd_width,
                reduction_method=reduction_method,
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ]
            ctx.enqueue_function[kernel](
                c,
                b,
                a,
                UInt(n),
                UInt(m),
                UInt(k),
                grid_dim=ceildiv(n, block_dim // WARP_SIZE),
                block_dim=block_dim,
            )

    elif kernel_func is GEMVAlgorithm.GEMV_KERNEL and transpose_b == False:
        ctx.enqueue_function[
            gemv_kernel[
                c.type,
                a.type,
                b.type,
                reduction_method = warp.ReductionMethod.WARP,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ]
        ](
            c.data,
            a.data,
            b.data,
            m,
            n,
            k,
            grid_dim=ceildiv(m, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
        )

    elif kernel_func is GEMVAlgorithm.GEMV_KERNEL and transpose_b == True:
        ctx.enqueue_function[
            gemv_kernel[
                c.type,
                b.type,
                a.type,
                reduction_method = warp.ReductionMethod.WARP,
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ]
        ](
            c.data,
            b.data,
            a.data,
            n,
            m,
            k,
            grid_dim=ceildiv(n, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
        )
    elif kernel_func is GEMVAlgorithm.GEVM_KERNEL:
        ctx.enqueue_function[
            gevm_kernel[
                c.type,
                a.type,
                b.type,
                tile_size = WARP_SIZE * WARPS_PER_BLOCK,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ]
        ](
            c.data,
            a.data,
            b.data,
            m,
            n,
            k,
            grid_dim=ceildiv(n, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
        )

    elif kernel_func is GEMVAlgorithm.MATMUL_NAIVE:
        alias BLOCK_DIM = 16
        ctx.enqueue_function[
            matmul_kernel_naive[
                c.type,
                a.type,
                b.type,
                BLOCK_DIM,
                transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ]
        ](
            c.data,
            a.data,
            b.data,
            m,
            n,
            k,
            grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )
    else:
        print("Gemv Kernel selection mismatch")
        return


@always_inline
fn gemv_gpu[
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: NDBuffer[rank=2, *_, **_],
    a: NDBuffer[rank=2, *_, **_],
    b: NDBuffer[rank=2, *_, **_],
    ctx: DeviceContext,
) raises:
    var shape = GemmShape.get[transpose_b=False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K
    alias simd_width = simdwidthof[a.type, target = get_gpu_target()]()

    # Kernel selection
    var kernel_func = GEMVAlgorithm.GEMV_KERNEL_VECTOR

    if n == 1:

        @parameter
        if a.type is DType.bfloat16:
            if k % simd_width == 0:
                kernel_func = GEMVAlgorithm.GEMV_KERNEL_VECTOR
            else:
                kernel_func = GEMVAlgorithm.GEMV_KERNEL
        else:
            kernel_func = GEMVAlgorithm.GEMV_KERNEL

    elif m == 1 and transpose_b == True:

        @parameter
        if a.type is DType.bfloat16:
            if k % simd_width == 0:
                if ceildiv(n, 2) <= ctx.get_attribute(
                    DeviceAttribute.MAX_GRID_DIM_Y
                ):
                    kernel_func = GEMVAlgorithm.GEMV_SPLIT_K
                else:
                    kernel_func = GEMVAlgorithm.GEMV_KERNEL_VECTOR
            else:
                kernel_func = GEMVAlgorithm.GEMV_KERNEL
        else:
            kernel_func = GEMVAlgorithm.GEMV_KERNEL

    elif m == 1 and n % WARP_SIZE == 0 and k % WARP_SIZE == 0:

        @parameter
        if a.type is DType.bfloat16 and has_nvidia_gpu_accelerator():
            if (
                k >= 4096
                and n >= 4096
                and k % simd_width == 0
                and n % simd_width == 0
            ):
                alias WARPS_PER_BLOCK = 32
                alias kernel = gevm_tc_kernel_vector_8x[
                    c.type,
                    a.type,
                    b.type,
                    WARP_SIZE * WARPS_PER_BLOCK * simd_width,
                    simd_width,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ]
                ctx.enqueue_function[kernel](
                    c,
                    a,
                    b,
                    m,
                    n,
                    k,
                    grid_dim=ceildiv(n, WARP_SIZE * simd_width),
                    block_dim=WARP_SIZE * WARPS_PER_BLOCK,
                )

            else:
                kernel_func = GEMVAlgorithm.GEVM_KERNEL
        else:
            kernel_func = GEMVAlgorithm.GEVM_KERNEL

        # GEVM_KERNEL does not work with AMDGPU yet
        @parameter
        if has_amd_gpu_accelerator():
            kernel_func = GEMVAlgorithm.MATMUL_NAIVE

    else:
        kernel_func = GEMVAlgorithm.MATMUL_NAIVE

    # default reduction method
    alias reduction_method = warp.ReductionMethod.WARP

    gemv_gpu_dispatch[
        transpose_b=transpose_b,
        reduction_method=reduction_method,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](kernel_func, c, a, b, ctx)


# Parallelized version of Gemv


@always_inline
fn gemv[
    parallelize: Bool,
    c_size: Dim,
    c_type: DType,
    a_shape: DimList,
    a_type: DType,
    b_size: Dim,
    b_type: DType,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c_buf: NDBuffer[mut=True, c_type, 1, _, c_size],
    a_buf: NDBuffer[a_type, 2, _, a_shape],
    b_buf: NDBuffer[b_type, 1, _, b_size],
) raises:
    alias simd_width = simdwidthof[c_type]()

    var M = a_buf.dim[0]()
    var K = a_buf.dim[1]()

    @always_inline
    @parameter
    fn input_fn[
        type: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[type, width]:
        return (
            a_buf.load[width=width](Index(idx[0], idx[1])).cast[type]()
            * b_buf.load[width=width](idx[1]).cast[type]()
        ).cast[type]()

    @always_inline
    @parameter
    fn output_fn[
        out_type: DType, width: Int, rank: Int
    ](idx: IndexList[rank], value: SIMD[out_type, width]):
        @parameter
        if elementwise_lambda_fn:
            alias func = elementwise_lambda_fn.value()

            @parameter
            for i in range(width):
                func[out_type, 1]((idx[0] + i, 0), value[i])
        else:
            c_buf.store[width=width](idx[0], value.cast[c_type]())

    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 + v2

    _reduce_generator[
        input_fn,
        output_fn,
        reduce_impl,
        single_thread_blocking_override = not parallelize,
    ](
        Index(M, K),
        init=Scalar[c_type](0),
        reduce_dim=1,
    )


fn naive_gemv[
    c_size: Dim,
    a_shape: DimList,
    b_size: Dim,
    type: DType,
](
    c_buf: NDBuffer[mut=True, type, 1, _, c_size],
    a_buf: NDBuffer[type, 2, _, a_shape],
    b_buf: NDBuffer[type, 1, _, b_size],
):
    var M = a_buf.dim[0]()
    var K = a_buf.dim[1]()

    c_buf.zero()
    for k in range(K):
        var b_val = b_buf[k]
        for m in range(M):
            var a_val = a_buf[m, k]
            c_buf[m] += a_val * b_val
