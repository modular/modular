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
from collections import Optional
from math import align_up, ceildiv
from sys import (
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    simd_width_of,
)

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
    warp_id,
)
from gpu.host import (
    DeviceAttribute,
    DeviceBuffer,
    DeviceContext,
    LaunchAttribute,
    get_gpu_target,
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

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

from utils import IndexList
from utils.index import Index
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from .matmul.gpu import matmul_kernel_naive
from .utils import GemmShape, elementwise_epilogue_type

comptime logger = Logger()


@fieldwise_init
struct GEMVAlgorithm(ImplicitlyCopyable, Stringable, Writable):
    var _value: Int

    comptime GEMV_KERNEL = Self(0)
    comptime GEMV_KERNEL_VECTOR = Self(1)
    comptime GEMV_SPLIT_K = Self(2)
    comptime GEVM_KERNEL_VECTOR = Self(3)
    comptime GEVM_KERNEL = Self(4)
    comptime MATMUL_NAIVE = Self(5)

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
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
    pdl_level: PDLLevel = PDLLevel(),
](
    c: UnsafePointer[Scalar[c_type]],
    a: UnsafePointer[Scalar[a_type]],
    b: UnsafePointer[Scalar[b_type]],
    m: Int,
    n: Int,
    k: Int,
):
    var tid = global_idx.x
    var global_warp_id = warp.broadcast(tid // UInt(WARP_SIZE))
    var lane_id = lane_id()

    if global_warp_id >= UInt(m):
        return

    var accum = Scalar[s_type](0)

    @parameter
    if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    # Every warp processes a single row of the resultant vector
    for i in range(ceildiv(k, WARP_SIZE)):
        var idx = i * WARP_SIZE + Int(lane_id)
        if idx < k:
            accum += (
                a.load(global_warp_id * UInt(k) + UInt(idx)).cast[s_type]()
                * b.load(idx).cast[s_type]()
            )

    accum = warp.sum(accum)

    if lane_id == 0:

        @parameter
        if elementwise_lambda_fn:
            comptime elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                reverse_idx[transpose_b](Int(global_warp_id), 0),
                accum.cast[c_type](),
            )
        else:
            c[global_warp_id] = accum.cast[c_type]()

    @parameter
    if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()


# Matrix-Column Vector Multiplication using vectorized instructions
fn gemv_kernel_vector[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    *,
    simd_width: UInt,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
    pdl_level: PDLLevel = PDLLevel(),
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],  # m
    a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],  # m * k
    b: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],  # 1 * k
    m: Int,
    n: Int,
    k: Int,
):
    var tid = global_idx.x
    var global_warp_id = Int(warp.broadcast(tid // UInt(WARP_SIZE)))
    var lane_id = lane_id()
    comptime step = WARP_SIZE * Int(simd_width)

    var idx = lane_id * simd_width

    if global_warp_id >= m:
        return

    # Every warp processes a single row of the resultant vector
    var local_accum = SIMD[s_type, Int(simd_width)](0)

    comptime local_accum_type = type_of(local_accum)

    @parameter
    if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    for i in range(ceildiv(k // Int(simd_width), WARP_SIZE)):
        var a_tile = a.tile[1, WARP_SIZE * Int(simd_width)](global_warp_id, i)
        var b_tile = b.tile[1, WARP_SIZE * Int(simd_width)](0, i)

        if idx >= UInt(k):
            continue

        var a_vec = a_tile.vectorize[1, Int(simd_width)]()[0, Int(lane_id)]
        var b_vec = b_tile.vectorize[1, Int(simd_width)]()[0, Int(lane_id)]
        local_accum += rebind[local_accum_type](a_vec.cast[s_type]()) * rebind[
            local_accum_type
        ](b_vec.cast[s_type]())

        idx += UInt(step)

    var accum = warp.sum(local_accum)

    if lane_id == 0:

        @parameter
        if elementwise_lambda_fn:
            comptime elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                reverse_idx[transpose_b](global_warp_id, 0),
                accum.cast[c_type](),
            )
        else:

            @parameter
            if transpose_b:
                c[0, global_warp_id] = accum.cast[c_type]()
            else:
                c[global_warp_id, 0] = accum.cast[c_type]()

    @parameter
    if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
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
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
    check_bounds: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
](
    output: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    act: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
    weight: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    """GEMV with tiling in K dimension.
    Assuming the B (weight) matrix is transposed i.e. row major N x K, this kernel
    implements a vector (1 x K) times a matrix (N x K).
    The impl can actually handle M > 1 but it's only optimal for tiny M. We use
    it for M = 1 only.
    """
    # tile_m represents how many rows each thread will process of the output activation matrix
    # tile_n represents how many rows each thread will process of the weight matrix.
    # Nvidia vectorized load is 16B.
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
            s_type,
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
    for _ in range(tid * simd_width, k, tile_k):
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

                var local_accum = rebind[Scalar[s_type]](acc[i, j])

                @parameter
                for l in range(simd_width):
                    local_accum += (
                        act_vec[Int(l)].cast[s_type]()
                        * weight_vec[Int(l)].cast[s_type]()
                    )

                acc.store[1](Int(i), Int(j), local_accum)

        iteration += 1

    # Warps are arranged along K.
    comptime k_warp_num = num_threads // UInt(WARP_SIZE)
    var warp_id = warp_id()
    var lane_id = lane_id()
    var shmem = LayoutTensor[
        s_type,
        Layout.row_major(1, Int(tile_m * tile_n * k_warp_num)),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Each warp sums across its threads and stages results in shared memory.
    # Shared memory data is row mojor (num_warps, tile_m, tile_n) stored in 1D.
    @parameter
    for mi in range(tile_m):

        @parameter
        for ni in range(tile_n):
            var val = warp.sum(acc[mi, ni])
            if lane_id == 0:
                shmem[0, mi * tile_n + ni + warp_id * tile_m * tile_n] = val
    barrier()
    # Sum across warps' results in shared memory then output.
    # TODO: should be able to vectorize and maybe use larger tile_n.
    for ii in range(tid, tile_m * tile_n, num_threads):
        var mid = ii // tile_n
        var nid = ii % tile_n
        var val = Scalar[s_type]()
        comptime ValType = type_of(val)

        @parameter
        for jj in range(k_warp_num):
            val += rebind[ValType](shmem[0, jj * tile_m * tile_n + ii])

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


# Row Vector-Matrix multiplication
fn gevm_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    *,
    tile_size: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
    pdl_level: PDLLevel = PDLLevel(),
](
    c: UnsafePointer[Scalar[c_type]],
    a: UnsafePointer[Scalar[a_type]],
    b: UnsafePointer[Scalar[b_type]],
    m: Int,
    n: Int,
    k: Int,
):
    var warps_per_block = block_dim.x // UInt(WARP_SIZE)
    var warp_id = warp_id()
    var lane_id = lane_id()
    var accum = Scalar[s_type]()
    var col = block_idx.x * UInt(WARP_SIZE) + lane_id
    var tid = global_idx.x
    var global_warp_id = tid // UInt(WARP_SIZE)

    var x_shared = stack_allocation[
        tile_size,
        s_type,
        address_space = AddressSpace.SHARED,
    ]()

    @parameter
    if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    # Every block computes warp size length of output values
    for i in range(ceildiv(UInt(k), warps_per_block)):
        var row = i * warps_per_block + warp_id
        var lhs = a.load(row)
        var rhs = b.load(row * UInt(n) + col)
        accum += lhs.cast[s_type]() * rhs.cast[s_type]()

    x_shared[lane_id * UInt(WARP_SIZE) + warp_id] = accum
    barrier()

    var total = x_shared.load(thread_idx.x).cast[s_type]()
    total = warp.sum(total)

    if lane_id == 0:

        @parameter
        if elementwise_lambda_fn:
            comptime elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                Index(0, global_warp_id), total.cast[c_type]()
            )
        else:
            c[global_warp_id] = total.cast[c_type]()

    @parameter
    if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()


@always_inline
fn gemv_gpu_dispatch[
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    kernel_func: GEMVAlgorithm,
    c: NDBuffer[rank=2, ...],
    a: NDBuffer[rank=2, ...],
    b: NDBuffer[rank=2, ...],
    ctx: DeviceContext,
) raises:
    var shape = GemmShape.get[transpose_b=False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K

    comptime WARPS_PER_BLOCK = 1024 // WARP_SIZE
    comptime simd_width = simd_width_of[a.type, target = get_gpu_target()]()

    var c_tensor = from_ndbuffer_row_major(c)
    var b_tensor = from_ndbuffer_row_major(b)
    var a_tensor = from_ndbuffer_row_major(a)

    comptime has_N = c.shape.has_value[1]()
    comptime static_N = c.shape.get[1]() if has_N else UNKNOWN_VALUE

    if kernel_func is GEMVAlgorithm.GEMV_SPLIT_K:
        logger.info("Executing: GEMV_SPLIT_K kernel")
        comptime num_threads = 128
        comptime tile_m = 1
        comptime tile_n = 2
        comptime check_bounds = static_N % tile_n != 0

        comptime kernel = gemv_split_k[
            c.type,
            a.type,
            b.type,
            c_tensor.layout,
            a_tensor.layout,
            b_tensor.layout,
            simd_width = UInt(simd_width),
            tile_m = UInt(tile_m),
            tile_n = UInt(tile_n),
            num_threads = UInt(num_threads),
            elementwise_lambda_fn=elementwise_lambda_fn,
            check_bounds=check_bounds,
            pdl_level=pdl_level,
        ]
        ctx.enqueue_function[kernel, kernel](
            c_tensor,
            a_tensor,
            b_tensor,
            m,
            n,
            k,
            grid_dim=(ceildiv(m, tile_m), ceildiv(n, tile_n)),
            block_dim=num_threads,
            attributes=pdl_launch_attributes(pdl_level),
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
                comptime kernel = gemv_kernel_vector[
                    c.type,
                    a.type,
                    b.type,
                    c_tensor.layout,
                    a_tensor.layout,
                    b_tensor.layout,
                    simd_width = UInt(simd_width),
                    transpose_b=False,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    pdl_level=pdl_level,
                ]
                ctx.enqueue_function[kernel, kernel](
                    c_tensor,
                    a_tensor,
                    b_tensor,
                    m,
                    n,
                    k,
                    grid_dim=ceildiv(m, block_dim // WARP_SIZE),
                    block_dim=block_dim,
                    attributes=pdl_launch_attributes(pdl_level),
                )
            else:
                # runtime transpose since layout_tensor.transpose requires static shape
                var aligned_b = b.data

                comptime has_K = a.shape.has_value[1]()
                comptime static_K = a.shape.get[1]() if has_K else UNKNOWN_VALUE
                comptime b_layout_template = Layout.row_major(
                    static_N, static_K
                )

                var b_runtime_shape = RuntimeTuple[b_layout_template.shape](
                    n, k
                )

                var b_runtime_stride = RuntimeTuple[b_layout_template.stride](
                    k, 1
                )

                var b_runtime_layout = RuntimeLayout[b_layout_template](
                    b_runtime_shape, b_runtime_stride
                )

                var b_tensor_n_major = LayoutTensor[
                    b.type,
                    b_layout_template,
                    MutAnyOrigin,
                    address_space = aligned_b.address_space,
                ](aligned_b, b_runtime_layout)

                @parameter
                if has_nvidia_gpu_accelerator():
                    var max_access_policy_window_size = ctx.get_attribute(
                        DeviceAttribute.MAX_ACCESS_POLICY_WINDOW_SIZE
                    )

                    var launch_attributes: List[LaunchAttribute] = [
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
                    ]

                    comptime pdl_attribute_list = pdl_launch_attributes(
                        pdl_level
                    )

                    @parameter
                    if len(pdl_attribute_list) > 0:
                        comptime pdl_attribute = pdl_attribute_list[0]
                        launch_attributes.append(pdl_attribute)

                    comptime kernel = gemv_kernel_vector[
                        c.type,
                        a.type,
                        b.type,
                        c_tensor.layout,
                        a_tensor.layout,
                        b_layout_template,
                        simd_width = UInt(simd_width),
                        transpose_b=transpose_b,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                        pdl_level=pdl_level,
                    ]
                    ctx.enqueue_function[kernel, kernel](
                        c_tensor,
                        a_tensor,
                        b_tensor_n_major,
                        m,
                        n,
                        k,
                        grid_dim=ceildiv(m, block_dim // WARP_SIZE),
                        block_dim=block_dim,
                        attributes=launch_attributes^,
                    )
                else:
                    comptime kernel = gemv_kernel_vector[
                        c.type,
                        a.type,
                        b.type,
                        c_tensor.layout,
                        a_tensor.layout,
                        b_layout_template,
                        simd_width = UInt(simd_width),
                        transpose_b=transpose_b,
                        elementwise_lambda_fn=elementwise_lambda_fn,
                        pdl_level=pdl_level,
                    ]
                    ctx.enqueue_function[kernel, kernel](
                        c_tensor,
                        a_tensor,
                        b_tensor_n_major,
                        m,
                        n,
                        k,
                        grid_dim=ceildiv(m, block_dim // WARP_SIZE),
                        block_dim=block_dim,
                        attributes=pdl_launch_attributes(pdl_level),
                    )
        elif m == 1:
            comptime kernel = gemv_kernel_vector[
                c.type,
                b.type,
                a.type,
                c_tensor.layout,
                b_tensor.layout,
                a_tensor.layout,
                simd_width = UInt(simd_width),
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                pdl_level=pdl_level,
            ]
            ctx.enqueue_function[kernel, kernel](
                c_tensor,
                b_tensor,
                a_tensor,
                n,
                m,
                k,
                grid_dim=ceildiv(n, block_dim // WARP_SIZE),
                block_dim=block_dim,
                attributes=pdl_launch_attributes(pdl_level),
            )

    elif kernel_func is GEMVAlgorithm.GEMV_KERNEL and transpose_b == False:
        logger.info("Executing: GEMV_KERNEL (no transpose)")

        comptime kernel = gemv_kernel[
            c.type,
            a.type,
            b.type,
            elementwise_lambda_fn=elementwise_lambda_fn,
            pdl_level=pdl_level,
        ]

        ctx.enqueue_function[kernel, kernel](
            c_tensor.to_device_buffer(ctx),
            a_tensor.to_device_buffer(ctx),
            b_tensor.to_device_buffer(ctx),
            m,
            n,
            k,
            grid_dim=ceildiv(m, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
            attributes=pdl_launch_attributes(pdl_level),
        )

    elif kernel_func is GEMVAlgorithm.GEMV_KERNEL and transpose_b == True:
        logger.info("Executing: GEMV_KERNEL (with transpose)")

        comptime kernel = gemv_kernel[
            c.type,
            b.type,
            a.type,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            pdl_level=pdl_level,
        ]
        ctx.enqueue_function[kernel, kernel](
            c_tensor.to_device_buffer(ctx),
            b_tensor.to_device_buffer(ctx),
            a_tensor.to_device_buffer(ctx),
            n,
            m,
            k,
            grid_dim=ceildiv(n, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
            attributes=pdl_launch_attributes(pdl_level),
        )
    elif kernel_func is GEMVAlgorithm.GEVM_KERNEL:
        logger.info("Executing: GEVM_KERNEL")
        comptime kernel = gevm_kernel[
            c.type,
            a.type,
            b.type,
            tile_size = WARP_SIZE * WARPS_PER_BLOCK,
            elementwise_lambda_fn=elementwise_lambda_fn,
            pdl_level=pdl_level,
        ]
        ctx.enqueue_function[kernel, kernel](
            c_tensor.to_device_buffer(ctx),
            a_tensor.to_device_buffer(ctx),
            b_tensor.to_device_buffer(ctx),
            m,
            n,
            k,
            grid_dim=ceildiv(n, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
            attributes=pdl_launch_attributes(pdl_level),
        )

    else:
        logger.info("Executing: MATMUL_NAIVE kernel")
        comptime BLOCK_DIM = 16

        comptime kernel = matmul_kernel_naive[
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
        ctx.enqueue_function[kernel, kernel](
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
](mode_1: Int, mode_2: Int,) -> None:
    logger.info(
        name,
        ": (",
        "_" if has_mode_1 else "",
        mode_1,
        ", ",
        "_" if has_mode_2 else "",
        mode_2,
        ")",
        sep="",
    )


@always_inline
fn gemv_gpu[
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: NDBuffer[rank=2, ...],
    a: NDBuffer[rank=2, ...],
    b: NDBuffer[rank=2, ...],
    ctx: DeviceContext,
) raises:
    var shape = GemmShape.get[transpose_b=False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K
    comptime simd_width = simd_width_of[a.type, target = get_gpu_target()]()

    comptime has_M = c.shape.has_value[0]()
    comptime has_N = c.shape.has_value[1]()
    comptime has_K = a.shape.has_value[1]()

    logger.info("------ Dispatching to GEMV ------")

    # Log dimension static/dynamic status
    log_shape[has_M, has_K, "A"](m, k)
    log_shape[has_K, has_N, "B"](k, n)
    log_shape[has_M, has_N, "C"](m, n)

    # Kernel selection
    var kernel_func: GEMVAlgorithm

    if n == 1:

        @parameter
        if a.type == DType.bfloat16:
            if k % simd_width == 0:
                kernel_func = GEMVAlgorithm.GEMV_KERNEL_VECTOR
            else:
                kernel_func = GEMVAlgorithm.GEMV_KERNEL
        else:
            kernel_func = GEMVAlgorithm.GEMV_KERNEL

    elif m == 1 and transpose_b == True:

        @parameter
        if a.type in (DType.bfloat16, DType.float8_e4m3fn):
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

    gemv_gpu_dispatch[
        transpose_b=transpose_b,
        elementwise_lambda_fn=elementwise_lambda_fn,
        pdl_level=pdl_level,
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
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_buf: NDBuffer[mut=True, c_type, 1, _, c_size],
    a_buf: NDBuffer[a_type, 2, _, a_shape],
    b_buf: NDBuffer[b_type, 1, _, b_size],
) raises:
    comptime simd_width = simd_width_of[c_type]()

    var M = a_buf.dim[0]()
    var K = a_buf.dim[1]()

    @always_inline
    @parameter
    fn input_fn[
        dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[dtype, width]:
        return (
            a_buf.load[width=width](Index(idx[0], idx[1])).cast[dtype]()
            * b_buf.load[width=width](idx[1]).cast[dtype]()
        ).cast[dtype]()

    @always_inline
    @parameter
    fn output_fn[
        out_type: DType, width: Int, rank: Int
    ](idx: IndexList[rank], value: SIMD[out_type, width]):
        @parameter
        if elementwise_lambda_fn:
            comptime func = elementwise_lambda_fn.value()

            @parameter
            for i in range(width):
                func[out_type, 1]((idx[0] + i, 0), value[i])
        else:
            c_buf.store[width=width](IndexList[1](idx[0]), value.cast[c_type]())

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
    dtype: DType,
](
    c_buf: NDBuffer[mut=True, dtype, 1, _, c_size],
    a_buf: NDBuffer[dtype, 2, _, a_shape],
    b_buf: NDBuffer[dtype, 1, _, b_size],
):
    var M = a_buf.dim[0]()
    var K = a_buf.dim[1]()

    c_buf.zero()
    for k in range(K):
        var b_val = b_buf[k]
        for m in range(M):
            var a_val = a_buf[m, k]
            c_buf[m] += a_val * b_val
