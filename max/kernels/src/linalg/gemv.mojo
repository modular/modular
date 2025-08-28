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
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    simd_width_of,
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
from gpu.host import get_gpu_target, DeviceBuffer
from gpu.host.launch_attribute import AccessPolicyWindow, AccessProperty
from gpu.memory import AddressSpace, load
from logger import Logger
from memory import stack_allocation

from utils import IndexList
from utils.index import Index
from io.write import Writable, Writer
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from .matmul_gpu import matmul_kernel_naive
from .utils import GemmShape, elementwise_epilogue_type

# layout imports
from layout import (
    LayoutTensor,
    Layout,
    UNKNOWN_VALUE,
    RuntimeLayout,
    RuntimeTuple,
)
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.tensor_builder import LayoutTensorBuild as tb


@fieldwise_init
struct GEMVAlgorithm(Copyable, Movable, Stringable, Writable):
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

    fn __str__(self) -> String:
        """Returns the string representation of this algorithm.

        Returns:
            String: A human-readable string representation of the algorithm.
        """
        if self is Self.GEMV_KERNEL:
            return "GEMV_KERNEL"
        elif self is Self.GEMV_KERNEL_VECTOR:
            return "GEMV_KERNEL_VECTOR"
        elif self is Self.GEMV_SPLIT_K:
            return "GEMV_SPLIT_K"
        elif self is Self.GEVM_KERNEL_VECTOR:
            return "GEVM_KERNEL_VECTOR"
        elif self is Self.GEVM_KERNEL:
            return "GEVM_KERNEL"
        elif self is Self.MATMUL_NAIVE:
            return "MATMUL_NAIVE"
        else:
            return String("UNKNOWN_GEMV_ALGORITHM(", self._value, ")")

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(String(self))


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
                reverse_idx[transpose_b](Int(warp_id), 0),
                accum.cast[c_type](),
            )
        else:
            c[warp_id] = accum.cast[c_type]()


# Matrix-Column Vector Multiplication using vectorized instructions
fn gemv_kernel_vector[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    *,
    reduction_method: warp.ReductionMethod,
    simd_width: UInt,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
](
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],  # m
    a: LayoutTensor[a_type, a_layout, MutableAnyOrigin],  # m * k
    b: LayoutTensor[b_type, b_layout, MutableAnyOrigin],  # 1 * k
    m: Int,
    n: Int,
    k: Int,
):
    var tid = global_idx.x
    var warp_id = Int(warp.broadcast(tid // WARP_SIZE))
    alias step = WARP_SIZE * simd_width

    var idx = lane_id() * simd_width

    if warp_id >= m:
        return

    # Every warp processes a single row of the resultant vector
    var local_accum = SIMD[s_type, Int(simd_width)](0)

    alias local_accum_type = __type_of(local_accum)

    for i in range(Int(ceildiv(k // simd_width, WARP_SIZE))):
        var a_tile = a.tile[1, Int(WARP_SIZE * simd_width)](warp_id, i)
        var b_tile = b.tile[1, Int(WARP_SIZE * simd_width)](0, i)

        if idx >= k:
            continue

        var a_vec = a_tile.vectorize[1, Int(simd_width)]()[0, Int(lane_id())]
        var b_vec = b_tile.vectorize[1, Int(simd_width)]()[0, Int(lane_id())]
        local_accum += rebind[local_accum_type](a_vec.cast[s_type]()) * rebind[
            local_accum_type
        ](b_vec.cast[s_type]())

        idx += step

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

            @parameter
            if transpose_b:
                c[0, warp_id] = accum.cast[c_type]()
            else:
                c[warp_id, 0] = accum.cast[c_type]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn gemv_split_k[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    simd_width: UInt,
    tile_m: UInt,
    tile_n: UInt,
    num_threads: UInt,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
    check_bounds: Bool = True,
](
    output: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    act: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
    weight: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    """GEMV with tiling in K dimension.
    Assuming the B (weight) matrix is transposed i.e. row major N x K, this kernel
    implements a vector (1 x K) times a matrix (N x K).
    The impl can actually handle M > 1 but it's only optimal fro tiny M. We use
    it for M = 1 only.
    """
    # tile_m represents how many rows each thread will process of the output activation matrix
    # tile_n represents how many rows each thread will process of the weight matrix.
    # Nvidia vectorized load is 16B.
    alias tile_k = simd_width * num_threads
    # which rows of the activation matrix each thread will process
    var tile_id_m = block_idx.x * tile_m
    # which rows of the weight matrix each thread will process
    var tile_id_n = block_idx.y * tile_n
    var tid = thread_idx.x
    var tile_w = tb[b_type]().row_major[tile_n, simd_width]().local().alloc()
    # these are the partial accumlations for each thread this a matrix of values
    # since each thread will process a tile_m x tile_n partials of the output vector
    var acc = tb[s_type]().row_major[tile_m, tile_n]().local().alloc().fill(0)
    var output_idx = tile_id_m * n + tile_id_n
    var iteration = 0
    alias WeightVecType = SIMD[b_type, simd_width]
    # Each thread sums local data in K.
    for _ in range(tid * simd_width, k, tile_k):
        var weight_tile = weight.tile[tile_n, tile_k](block_idx.y, iteration)
        var act_tile = act.tile[tile_m, tile_k](block_idx.x, iteration)

        @parameter
        for i in range(tile_n):
            # Here we load data @ thread_idx.x from the weight matrix
            # and store it into tile_w. We skip this if if the current
            # row we are reading from (i + tile_id_n) is greater than the number
            # of rows in the weight matrix.
            @parameter
            if check_bounds:
                if i + tile_id_n >= n:
                    continue
            var b_vec = weight_tile.vectorize[1, simd_width]()[i, thread_idx.x]
            tile_w.store[simd_width](i, 0, rebind[WeightVecType](b_vec))

        @parameter
        for i in range(tile_m):
            # Here we load data @ thread_idx.x from the activation matrix
            # and store it into tile_a. We skip this if if the current
            # row we are reading from (i + tile_id_m) is greater than the number
            # of rows in the activation matrix. This should never be the case if
            # tile_m is 1.
            @parameter
            if check_bounds:
                if i + tile_id_m >= m:
                    continue
            var act_vec = act_tile.vectorize[1, simd_width]()[i, thread_idx.x]

            # Now we multiply tile_a by tile_w and store the partials
            # in acc
            @parameter
            for j in range(tile_n):
                var weight_vec = tile_w.vectorize[1, simd_width]()[j, 0]

                var local_accum = rebind[Scalar[s_type]](acc[i, j])

                @parameter
                for l in range(simd_width):
                    local_accum += (
                        act_vec[l].cast[s_type]() * weight_vec[l].cast[s_type]()
                    )

                acc.store[1](i, j, local_accum)

        iteration += 1

    # Warps are arranged along K.
    alias k_warp_num = num_threads // WARP_SIZE
    var warp_id = warp.broadcast(tid // WARP_SIZE)
    var shmem = (
        tb[s_type]()
        .row_major[1, tile_m * tile_n * k_warp_num]()
        .shared()
        .alloc()
    )

    # Each warp sums across its threads and stages results in shared memory.
    # Shared memory data is row mojor (num_warps, tile_m, tile_n) stored in 1D.
    @parameter
    for mi in range(tile_m):

        @parameter
        for ni in range(tile_n):
            var val = warp.sum(acc[mi, ni])
            if lane_id() == 0:
                shmem[0, mi * tile_n + ni + warp_id * tile_m * tile_n] = val
    barrier()
    # Sum across warps' results in shared memory then output.
    # TODO: should be able to vectorize and maybe use larger tile_n.
    for ii in range(tid, tile_m * tile_n, num_threads):
        var mid = ii // tile_n
        var nid = ii % tile_n
        var val = Scalar[s_type]()
        alias ValType = __type_of(val)

        @parameter
        for jj in range(k_warp_num):
            val += rebind[ValType](shmem[0, jj * tile_m * tile_n + ii])

        @parameter
        if elementwise_lambda_fn:
            alias elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                Index(0, output_idx + mid * n + nid), val.cast[c_type]()
            )
        else:
            var idx = output_idx + mid * n + nid

            @parameter
            if check_bounds:
                if idx >= n:
                    continue
            output[0, idx] = val.cast[c_type]()


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
    logger: Logger,
) raises:
    var shape = GemmShape.get[transpose_b=False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K

    alias WARPS_PER_BLOCK = 1024 // WARP_SIZE
    alias simd_width = simd_width_of[a.type, target = get_gpu_target()]()

    var c_tensor = from_ndbuffer_row_major(c)
    var b_tensor = from_ndbuffer_row_major(b)
    var a_tensor = from_ndbuffer_row_major(a)

    var a_buffer = DeviceBuffer[a.type](
        ctx,
        rebind[UnsafePointer[Scalar[a.type]]](a.data),
        a.size(),
        owning=False,
    )
    var b_buffer = DeviceBuffer[b.type](
        ctx,
        rebind[UnsafePointer[Scalar[b.type]]](b.data),
        b.size(),
        owning=False,
    )
    var c_buffer = DeviceBuffer[c.type](
        ctx,
        rebind[UnsafePointer[Scalar[c.type]]](c.data),
        c.size(),
        owning=False,
    )

    alias has_N = c.shape.has_value[1]()
    alias static_N = c.shape.get[1]() if has_N else UNKNOWN_VALUE

    if kernel_func is GEMVAlgorithm.GEMV_SPLIT_K:
        logger.info("Executing: GEMV_SPLIT_K kernel")
        alias num_threads = 128
        alias tile_m = 1
        alias tile_n = 2
        alias check_bounds = static_N % tile_n != 0

        alias kernel = gemv_split_k[
            c.type,
            a.type,
            b.type,
            c_tensor.layout,
            a_tensor.layout,
            b_tensor.layout,
            simd_width=simd_width,
            tile_m=tile_m,
            tile_n=tile_n,
            num_threads=num_threads,
            elementwise_lambda_fn=elementwise_lambda_fn,
            check_bounds=check_bounds,
        ]
        ctx.enqueue_function_checked[kernel, kernel](
            c_tensor,
            a_tensor,
            b_tensor,
            m,
            n,
            k,
            grid_dim=(ceildiv(m, tile_m), ceildiv(n, tile_n)),
            block_dim=num_threads,
        )

    elif kernel_func is GEMVAlgorithm.GEMV_KERNEL_VECTOR:
        logger.info("Executing: GEMV_KERNEL_VECTOR kernel")

        var block_dim = min(
            align_up(k // simd_width, WARP_SIZE),
            WARP_SIZE * WARPS_PER_BLOCK,
        )
        if n == 1:

            @parameter
            if transpose_b:
                alias kernel = gemv_kernel_vector[
                    c.type,
                    a.type,
                    b.type,
                    c_tensor.layout,
                    a_tensor.layout,
                    b_tensor.layout,
                    simd_width=simd_width,
                    reduction_method = warp.ReductionMethod.WARP,
                    transpose_b=False,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ]
                ctx.enqueue_function_checked[kernel, kernel](
                    c_tensor,
                    a_tensor,
                    b_tensor,
                    m,
                    n,
                    k,
                    grid_dim=ceildiv(m, block_dim // WARP_SIZE),
                    block_dim=block_dim,
                )
            else:
                # runtime transpose since layout_tensor.transpose requires static shape
                alias b_alignment = b.alignment
                var aligned_b = b.data.static_alignment_cast[b_alignment]()

                alias has_K = a.shape.has_value[1]()
                alias static_K = a.shape.get[1]() if has_K else UNKNOWN_VALUE
                alias b_layout_template = Layout.row_major(static_N, static_K)

                var b_runtime_shape = RuntimeTuple[
                    b_layout_template.shape, element_type = DType.int32
                ](n, k)

                var b_runtime_stride = RuntimeTuple[
                    b_layout_template.stride, element_type = DType.int32
                ](k, 1)

                var b_runtime_layout = RuntimeLayout[
                    b_layout_template,
                    element_type = DType.int32,
                    linear_idx_type = DType.int32,
                ](b_runtime_shape, b_runtime_stride)

                var b_tensor_n_major = LayoutTensor[
                    b.type,
                    b_layout_template,
                    MutableAnyOrigin,
                    alignment = aligned_b.alignment,
                    address_space = aligned_b.address_space,
                ](aligned_b, b_runtime_layout)

                @parameter
                if has_nvidia_gpu_accelerator():
                    var max_access_policy_window_size = ctx.get_attribute(
                        DeviceAttribute.MAX_ACCESS_POLICY_WINDOW_SIZE
                    )
                    var launch_attributes = List[LaunchAttribute](
                        LaunchAttribute(
                            AccessPolicyWindow(
                                base_ptr=a.data,
                                count=min(
                                    a.size(), max_access_policy_window_size
                                ),
                                hit_ratio=1,
                                hit_prop=AccessProperty.PERSISTING,
                                miss_prop=AccessProperty.STREAMING,
                            )
                        ),
                    )
                    alias kernel = gemv_kernel_vector[
                        c.type,
                        a.type,
                        b.type,
                        c_tensor.layout,
                        a_tensor.layout,
                        b_tensor_n_major.layout,
                        simd_width=simd_width,
                        reduction_method = warp.ReductionMethod.WARP,
                        transpose_b=transpose_b,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                    ]
                    ctx.enqueue_function[kernel](
                        c_tensor,
                        a_tensor,
                        b_tensor_n_major,
                        m,
                        n,
                        k,
                        grid_dim=ceildiv(m, block_dim // WARP_SIZE),
                        block_dim=block_dim,
                        attributes=launch_attributes,
                    )
                else:
                    alias kernel = gemv_kernel_vector[
                        c.type,
                        a.type,
                        b.type,
                        c_tensor.layout,
                        a_tensor.layout,
                        b_layout_template,
                        simd_width=simd_width,
                        reduction_method = warp.ReductionMethod.WARP,
                        transpose_b=transpose_b,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                    ]
                    ctx.enqueue_function[kernel](
                        c_tensor,
                        a_tensor,
                        b_tensor_n_major,
                        m,
                        n,
                        k,
                        grid_dim=ceildiv(m, block_dim // WARP_SIZE),
                        block_dim=block_dim,
                    )
        elif m == 1:
            alias kernel = gemv_kernel_vector[
                c.type,
                b.type,
                a.type,
                c_tensor.layout,
                b_tensor.layout,
                a_tensor.layout,
                simd_width=simd_width,
                reduction_method=reduction_method,
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ]
            ctx.enqueue_function_checked[kernel, kernel](
                c_tensor,
                b_tensor,
                a_tensor,
                n,
                m,
                k,
                grid_dim=ceildiv(n, block_dim // WARP_SIZE),
                block_dim=block_dim,
            )

    elif kernel_func is GEMVAlgorithm.GEMV_KERNEL and transpose_b == False:
        logger.info("Executing: GEMV_KERNEL (no transpose)")

        alias kernel = gemv_kernel[
            c.type,
            a.type,
            b.type,
            reduction_method = warp.ReductionMethod.WARP,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ]

        ctx.enqueue_function_checked[kernel, kernel](
            c_buffer,
            a_buffer,
            b_buffer,
            m,
            n,
            k,
            grid_dim=ceildiv(m, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
        )

    elif kernel_func is GEMVAlgorithm.GEMV_KERNEL and transpose_b == True:
        logger.info("Executing: GEMV_KERNEL (with transpose)")

        alias kernel = gemv_kernel[
            c.type,
            b.type,
            a.type,
            reduction_method = warp.ReductionMethod.WARP,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ]
        ctx.enqueue_function_checked[kernel, kernel](
            c_buffer,
            b_buffer,
            a_buffer,
            n,
            m,
            k,
            grid_dim=ceildiv(n, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
        )
    elif kernel_func is GEMVAlgorithm.GEVM_KERNEL:
        logger.info("Executing: GEVM_KERNEL")
        alias kernel = gevm_kernel[
            c.type,
            a.type,
            b.type,
            tile_size = WARP_SIZE * WARPS_PER_BLOCK,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ]
        ctx.enqueue_function_checked[kernel, kernel](
            c_buffer,
            a_buffer,
            b_buffer,
            m,
            n,
            k,
            grid_dim=ceildiv(n, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
        )

    else:
        logger.info("Executing: MATMUL_NAIVE kernel")
        alias BLOCK_DIM = 16

        alias kernel = matmul_kernel_naive[
            c.type,
            a.type,
            b.type,
            c_tensor.layout,
            a_tensor.layout,
            b_tensor.layout,
            BLOCK_DIM,
            transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ]
        ctx.enqueue_function_checked[kernel, kernel](
            c_tensor,
            a_tensor,
            b_tensor,
            m,
            n,
            k,
            grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )


fn log_shape[
    has_mode_1: Bool, has_mode_2: Bool, name: String
](logger: Logger, mode_1: Int, mode_2: Int,) -> None:
    logger.info(
        name,
        ": (",
        "_" if has_mode_1 else "",
        mode_1,
        ", ",
        "_" if has_mode_2 else "",
        mode_2,
        ")",
    )


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
    var logger = Logger()

    var shape = GemmShape.get[transpose_b=False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K
    alias simd_width = simd_width_of[a.type, target = get_gpu_target()]()

    alias has_M = c.shape.has_value[0]()
    alias has_N = c.shape.has_value[1]()
    alias has_K = a.shape.has_value[1]()

    logger.info("------ Dispatching to GEMV ------")

    # Log dimension static/dynamic status
    log_shape[has_M, has_K, "A"](logger, m, k)
    log_shape[has_K, has_N, "B"](logger, k, n)
    log_shape[has_M, has_N, "C"](logger, m, n)

    # Kernel selection
    var kernel_func: GEMVAlgorithm

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
    ](kernel_func, c, a, b, ctx, logger)


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
    alias simd_width = simd_width_of[c_type]()

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
