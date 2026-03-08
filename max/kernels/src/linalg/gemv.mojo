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
from std.collections import Optional
from std.math import align_down, align_up, ceildiv
from std.sys import (
    has_amd_gpu_accelerator,
    is_amd_gpu,
    llvm_intrinsic,
    simd_width_of,
)
from std.sys.info import _accelerator_arch


import std.gpu.primitives.warp as warp
from std.algorithm.reduction import _reduce_generator
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx_int as block_idx,
    global_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from std.gpu.host import (
    DeviceAttribute,
    DeviceBuffer,
    DeviceContext,
    get_gpu_target,
)
from std.gpu.primitives.grid_controls import (
    PDLLevel,
    pdl_launch_attributes,
    launch_dependent_grids,
    wait_on_dependent_grids,
)

# layout imports
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    TileTensor,
)
from std.logger import Logger
from std.memory import stack_allocation
from std.utils import IndexList
from std.utils.index import Index
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

from .matmul.gpu import matmul_kernel_naive
from .utils import GemmShape, elementwise_epilogue_type

comptime logger = Logger()


@fieldwise_init
struct GEMVAlgorithm(ImplicitlyCopyable, Writable):
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

    @deprecated("Stringable is deprecated. Use Writable instead.")
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
            return t"UNKNOWN_GEMV_ALGORITHM({self._value})"

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
    accum_type: DType = get_accum_type[c_type](),
    pdl_level: PDLLevel = PDLLevel(),
](
    c: UnsafePointer[Scalar[c_type], AnyOrigin[mut=True]],
    a: UnsafePointer[Scalar[a_type], AnyOrigin[mut=False]],
    b: UnsafePointer[Scalar[b_type], AnyOrigin[mut=False]],
    m: Int,
    n: Int,
    k: Int,
):
    var tid = global_idx.x
    var global_warp_id = warp.broadcast(tid // UInt(WARP_SIZE))
    var lane_id = lane_id()

    if global_warp_id >= UInt(m):
        return

    var accum = Scalar[accum_type](0)

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    # Every warp processes a single row of the resultant vector
    for i in range(ceildiv(k, WARP_SIZE)):
        var idx = i * WARP_SIZE + Int(lane_id)
        if idx < k:
            accum += (
                a.load(global_warp_id * UInt(k) + UInt(idx)).cast[accum_type]()
                * b.load(idx).cast[accum_type]()
            )

    accum = warp.sum(accum)

    if lane_id == 0:
        comptime if elementwise_lambda_fn:
            comptime elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                reverse_idx[transpose_b](Int(global_warp_id), 0),
                accum.cast[c_type](),
            )
        else:
            c[global_warp_id] = accum.cast[c_type]()

    comptime if pdl_level > PDLLevel.OFF:
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
    accum_type: DType = get_accum_type[c_type](),
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
    var local_accum = SIMD[accum_type, Int(simd_width)](0)

    comptime local_accum_type = type_of(local_accum)

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    for i in range(ceildiv(k // Int(simd_width), WARP_SIZE)):
        var a_tile = a.tile[1, WARP_SIZE * Int(simd_width)](global_warp_id, i)
        var b_tile = b.tile[1, WARP_SIZE * Int(simd_width)](0, i)

        if idx >= UInt(k):
            continue

        var a_vec = a_tile.vectorize[1, Int(simd_width)]()[0, Int(lane_id)]
        var b_vec = b_tile.vectorize[1, Int(simd_width)]()[0, Int(lane_id)]
        local_accum += rebind[local_accum_type](
            a_vec.cast[accum_type]()
        ) * rebind[local_accum_type](b_vec.cast[accum_type]())

        idx += UInt(step)

    var accum = warp.sum(local_accum)

    if lane_id == 0:
        comptime if elementwise_lambda_fn:
            comptime elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                reverse_idx[transpose_b](global_warp_id, 0),
                accum.cast[c_type](),
            )
        else:
            comptime if transpose_b:
                c[0, global_warp_id] = accum.cast[c_type]()
            else:
                c[global_warp_id, 0] = accum.cast[c_type]()

    comptime if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()


@always_inline
fn _dot_accum[
    in_type: DType,
    accum_type: DType,
    width: Int,
](
    a: SIMD[in_type, width], b: SIMD[in_type, width], acc: Scalar[accum_type]
) -> Scalar[accum_type]:
    """Compute dot(a, b) + acc with fused bf16→f32 dot product on AMD.

    On AMD GPUs except gfx90a, bf16 inputs with an f32 accumulator use
    v_dot2_f32_bf16 to avoid explicit bf16→f32 conversion
    (120 v_perm/v_bfi instructions). On other targets or types, this
    falls back to cast-then-multiply.
    """
    var result = acc

    comptime if (
        is_amd_gpu() and not _is_amd_mi250x
        and in_type == DType.bfloat16
        and accum_type == DType.float32
    ):
        # v_dot2_f32_bf16: D.f32 = S0.bf16[0]*S1.bf16[0] + S0.bf16[1]*S1.bf16[1] + S2.f32
        comptime for p in range(width // 2):
            var a_pair = rebind[SIMD[DType.bfloat16, 2]](
                a.slice[2, offset=p * 2]()
            )
            var b_pair = rebind[SIMD[DType.bfloat16, 2]](
                b.slice[2, offset=p * 2]()
            )
            result = rebind[Scalar[accum_type]](
                llvm_intrinsic[
                    "llvm.amdgcn.fdot2.f32.bf16",
                    Scalar[DType.float32],
                ](
                    a_pair,
                    b_pair,
                    rebind[Scalar[DType.float32]](result),
                    False,
                )
            )

        comptime if width % 2 != 0:
            result += (
                a[width - 1].cast[accum_type]()
                * b[width - 1].cast[accum_type]()
            )
    elif is_amd_gpu():
        # AMD non-BF16 (e.g. FP8): vector multiply + horizontal reduce.
        result += (a.cast[accum_type]() * b.cast[accum_type]()).reduce_add()
    else:
        # NVIDIA/generic: scalar element-wise loop. reduce_add() generates
        # wider intermediates that increase NVIDIA register pressure vs
        # sequential FMA chains (13% regression on small-K shapes).
        comptime for l in range(width):
            result += a[l].cast[accum_type]() * b[l].cast[accum_type]()

    return result


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
    simd_width: Int,
    tile_m: Int,
    tile_n: Int,
    num_threads: Int,
    unroll_factor: Int = 2,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
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
    var tid = Int(thread_idx.x)
    var tile_w = LayoutTensor[
        b_type,
        Layout.row_major(tile_n, simd_width),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ].stack_allocation()
    # these are the partial accumlations for each thread this a matrix of values
    # since each thread will process a tile_m x tile_n partials of the output vector
    var acc = (
        LayoutTensor[
            accum_type,
            Layout.row_major(tile_m, tile_n),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )
    var output_idx = tile_id_m * n + tile_id_n
    var iteration = 0
    comptime WeightVecType = SIMD[b_type, simd_width]

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    # Each thread sums local data in K.
    @parameter
    @always_inline
    fn _k_iter_body():
        """Single K-iteration: load weights, load activations, accumulate."""
        var weight_tile = weight.tile[tile_n, tile_k](block_idx.y, iteration)
        var act_tile = act.tile[tile_m, tile_k](block_idx.x, iteration)

        # Load weights with non-temporal hints on AMD to avoid L1/L2
        # cache pollution (weights are read exactly once).
        comptime for i in range(tile_n):
            comptime if check_bounds:
                if i + tile_id_n >= n:
                    continue
            comptime if is_amd_gpu():
                var b_vec = weight_tile.load[simd_width, non_temporal=True](
                    i, Int(thread_idx.x) * simd_width
                )
                tile_w.store(i, 0, rebind[WeightVecType](b_vec))
            else:
                var vec_weight_tile = weight_tile.vectorize[1, simd_width]()
                var b_vec = vec_weight_tile[i, thread_idx.x]
                tile_w.store(i, 0, rebind[WeightVecType](b_vec))

        # Load activations and accumulate dot products.
        comptime for i in range(tile_m):
            comptime if check_bounds:
                if i + tile_id_m >= m:
                    continue
            var act_vec = act_tile.vectorize[1, simd_width]()[i, thread_idx.x]

            comptime NativeVecType = SIMD[a_type, simd_width]
            var act_native = rebind[NativeVecType](act_vec)
            comptime for j in range(tile_n):
                var weight_native = rebind[NativeVecType](
                    tile_w.vectorize[1, simd_width]()[j, 0]
                )
                var local_accum = rebind[Scalar[accum_type]](acc[i, j])
                local_accum = _dot_accum(act_native, weight_native, local_accum)
                acc.store(i, j, local_accum)

        iteration += 1

    comptime if unroll_factor == 1:
        # Simple loop — no ceildiv, no main_iters/remainder split.
        # Produces minimal PTX with fewest registers on NVIDIA.
        for _ in range(tid * simd_width, k, tile_k):
            _k_iter_body()
    else:
        # Unrolled loop for ILP — comptime for duplicates the body.
        var k_start = tid * simd_width
        var num_k_iters = ceildiv(k - k_start, tile_k) if k > k_start else 0
        var main_iters = align_down(num_k_iters, unroll_factor)

        # Main unrolled loop.
        for _outer in range(0, main_iters, unroll_factor):
            comptime for _u in range(unroll_factor):
                _k_iter_body()

        # Remainder iterations (at most unroll_factor - 1).
        for _rem in range(main_iters, num_k_iters):
            _k_iter_body()

    # Warps are arranged along K.
    comptime k_warp_num = num_threads // WARP_SIZE
    var warp_id = Int(warp_id())
    var lane_id = lane_id()
    var shmem = LayoutTensor[
        accum_type,
        Layout.row_major(1, tile_m * tile_n * k_warp_num),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # Each warp sums across its threads and stages results in shared memory.
    # Shared memory data is row mojor (num_warps, tile_m, tile_n) stored in 1D.
    comptime for mi in range(tile_m):
        comptime for ni in range(tile_n):
            var val = warp.sum(acc[mi, ni])
            if lane_id == 0:
                shmem[0, mi * tile_n + ni + warp_id * tile_m * tile_n] = val
    barrier()
    # Sum across warps' results in shared memory then output.
    # TODO: should be able to vectorize and maybe use larger tile_n.
    for ii in range(tid, tile_m * tile_n, num_threads):
        var mid = ii // tile_n
        var nid = ii % tile_n
        var val = Scalar[accum_type]()
        comptime ValType = type_of(val)

        comptime for jj in range(k_warp_num):
            val += rebind[ValType](shmem[0, jj * tile_m * tile_n + ii])

        var idx = output_idx + mid * n + nid

        comptime if check_bounds:
            if idx >= n:
                continue

        comptime if elementwise_lambda_fn:
            comptime elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda(Index(0, idx), val.cast[c_type]())
        else:
            output[0, idx] = val.cast[c_type]()

    comptime if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()


# Row Vector-Matrix multiplication
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(tile_size))
)
fn gevm_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    *,
    tile_size: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
    pdl_level: PDLLevel = PDLLevel(),
](
    c: UnsafePointer[Scalar[c_type], AnyOrigin[mut=True]],
    a: UnsafePointer[Scalar[a_type], AnyOrigin[mut=False]],
    b: UnsafePointer[Scalar[b_type], AnyOrigin[mut=False]],
    m: Int,
    n: Int,
    k: Int,
):
    comptime warps_per_block = tile_size // WARP_SIZE

    var warp_id = Int(warp_id())
    var lane_id = Int(lane_id())
    var col = block_idx.x * WARP_SIZE + lane_id
    var global_warp_id = Int(global_idx.x) // warps_per_block

    var x_shared = stack_allocation[
        tile_size,
        accum_type,
        address_space=AddressSpace.SHARED,
    ]()

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    var accum = Scalar[accum_type]()

    # Every block computes warp size length of output values
    for i in range(ceildiv(k, warps_per_block)):
        var row = i * warps_per_block + warp_id
        var lhs = a[row]
        var rhs = b[row * n + col]
        accum += lhs.cast[accum_type]() * rhs.cast[accum_type]()

    x_shared[lane_id * warps_per_block + warp_id] = accum
    barrier()

    var total = warp.lane_group_sum[num_lanes=warps_per_block](
        x_shared[thread_idx.x]
    )

    if lane_id % warps_per_block == 0:
        comptime if elementwise_lambda_fn:
            comptime elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda(Index(0, global_warp_id), total.cast[c_type]())
        else:
            c[global_warp_id] = total.cast[c_type]()

    comptime if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()


fn _amd_gemv_config[
    simd_width: Int,
    max_thread_block_size: Int,
    static_K: Int,
    has_N: Bool,
    static_N: Int,
]() -> IndexList[3]:
    """Compute GEMV split-K dispatch config for AMD GPUs.

    Returns (num_threads, tile_n, unroll_factor).

    Works for both FP8 (simd_width=16) and BF16 (simd_width=8) — all
    thresholds derive from simd_width, WARP_SIZE, and
    max_thread_block_size.

    Thread count: pick from {64, 128, 256} to balance wave parallelism
    vs K-iteration count (tile_k = num_threads × simd_width). Single warp
    (64T) for K≤2048 (both BF16 and FP8) to avoid LDS sync. 256T when K
    provides ≥2 clean iterations or exactly 1 clean iteration. 128T for
    mid-K with bad fractional iterations at 256T.

    tile_n: when there are enough K-iterations (≥3 for BF16, ≥4 for FP8),
    pick the largest tile_n from {4,2,1} that gives ≥ min_waves_per_simd
    waves/SIMD. Otherwise default to tile_n=2 (grid parallelism >
    loads-per-iter).
    """
    comptime tile_k_256 = 256 * simd_width
    # BF16 (sw=8) has 2× more K-iterations than FP8 (sw=16) for the same K,
    # so each wave keeps the SIMD busy longer and fewer waves/SIMD suffice.
    # FP8 needs ≥10 waves/SIMD (Exp Q showed tile_n=1 optimal for small N).
    # BF16 needs ≥5 waves/SIMD to hide L2 latency on K=16384 shapes.
    comptime min_waves_per_simd = 5 if simd_width <= 8 else 10

    # --- Thread count ---
    # Single warp (64T) avoids LDS cross-warp reduction overhead.
    # BF16: K≤1024 (tile_k=512, 2 iters), FP8: K≤2048 (tile_k=1024, 2 iters).
    var num_threads: Int
    if static_K <= 2 * WARP_SIZE * simd_width:
        num_threads = 64
    elif static_K >= 2 * tile_k_256 or static_K % tile_k_256 == 0:
        # ≥2 clean iterations, or exactly 1 clean iteration at 256T.
        num_threads = 256
    else:
        # Mid-K with fractional iters at 256T. 128T halves tile_k,
        # giving more iterations with better pipelining.
        num_threads = 128

    # --- tile_n ---
    # With ≥4 K-iterations per wave, there's enough work to tolerate
    # fewer grid blocks — pick largest tile_n with sufficient waves/SIMD.
    # With <4 iterations, grid parallelism matters more — keep tile_n=2.
    var tile_n = 2
    var k_iters = static_K // (num_threads * simd_width)
    # BF16 has NT loads + fdot2 doing more work per iteration, so tile_n=4
    # is profitable at fewer K-iterations (≥3 vs ≥4 for FP8).
    comptime min_k_iters_for_tile_n = 3 if simd_width <= 8 else 4
    if k_iters >= min_k_iters_for_tile_n and has_N:
        var wavefront_capacity = static_N * (num_threads // WARP_SIZE)
        if wavefront_capacity >= min_waves_per_simd * max_thread_block_size * 4:
            tile_n = 4
        elif (
            wavefront_capacity >= min_waves_per_simd * max_thread_block_size * 2
        ):
            tile_n = 2
        else:
            # tile_n=1 only benefits FP8 (more grid parallelism needed).
            # BF16 has more work per iteration, so tile_n=2 is the floor.
            tile_n = 1 if simd_width > 8 else 2

    # unroll=4 when there are enough K-iterations and tile_n is small enough
    # to avoid register pressure (tile_n=4 + unroll=4 hurts large-N shapes).
    var unroll = 4 if k_iters >= 8 and tile_n <= 2 else 2
    return IndexList[3](num_threads, tile_n, unroll)


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
    comptime simd_width = simd_width_of[a.type, target=get_gpu_target()]()

    var c_tensor = TileTensor(c).to_layout_tensor()
    var b_tensor = TileTensor(b).to_layout_tensor()
    var a_tensor = TileTensor(a).to_layout_tensor()

    comptime has_N = c.shape.has_value[1]()
    comptime static_N = c.shape.get[1]() if has_N else UNKNOWN_VALUE
    comptime static_K = a.shape.get[1]()

    if kernel_func is GEMVAlgorithm.GEMV_SPLIT_K:
        logger.info("Executing: GEMV_SPLIT_K kernel")
        comptime tile_m = 1

        @parameter
        fn _gemv_split_k_dispatch[
            num_threads: Int,
            tile_n: Int,
            unroll_factor: Int = 2,
        ]() raises:
            comptime check_bounds = static_N % tile_n != 0
            comptime kernel = gemv_split_k[
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
                unroll_factor=unroll_factor,
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

        comptime if has_amd_gpu_accelerator():
            comptime config = _amd_gemv_config[
                simd_width,
                ctx.default_device_info.max_thread_block_size,
                static_K,
                has_N,
                static_N,
            ]()
            _gemv_split_k_dispatch[
                config[0],
                config[1],
                config[2],
            ]()
        else:
            # NVIDIA/generic: uniform 128T/tile_n=2/unroll=1.
            # unroll_factor=1 uses the simple loop path (no ceildiv,
            # no remainder loop) — produces cleaner NVIDIA PTX with
            # fewer registers. Thread count sweep on B200 showed all
            # configs within ±2% noise for both FP8 and BF16.
            _gemv_split_k_dispatch[128, tile_n=2, unroll_factor=1]()

    elif kernel_func is GEMVAlgorithm.GEMV_KERNEL_VECTOR:
        logger.info("Executing: GEMV_KERNEL_VECTOR kernel")

        var block_dim = min(
            align_up(k // simd_width, WARP_SIZE),
            WARP_SIZE * WARPS_PER_BLOCK,
        )
        if n == 1:
            comptime if transpose_b:
                comptime kernel = gemv_kernel_vector[
                    c.type,
                    a.type,
                    b.type,
                    c_tensor.layout,
                    a_tensor.layout,
                    b_tensor.layout,
                    simd_width=UInt(simd_width),
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
                    b.origin,
                    address_space=aligned_b.address_space,
                ](aligned_b, b_runtime_layout)

                comptime kernel = gemv_kernel_vector[
                    c.type,
                    a.type,
                    b.type,
                    c_tensor.layout,
                    a_tensor.layout,
                    b_layout_template,
                    simd_width=UInt(simd_width),
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
                simd_width=UInt(simd_width),
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
            tile_size=WARP_SIZE * WARPS_PER_BLOCK,
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

        var c_tt = TileTensor(c)
        var a_tt = TileTensor(a)
        var b_tt = TileTensor(b)

        comptime kernel = matmul_kernel_naive[
            c.type,
            a.type,
            b.type,
            type_of(c_tt).LayoutType,
            type_of(a_tt).LayoutType,
            type_of(b_tt).LayoutType,
            BLOCK_DIM,
            transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ]
        ctx.enqueue_function[kernel, kernel](
            c_tt,
            a_tt,
            b_tt,
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
    comptime simd_width = simd_width_of[a.type, target=get_gpu_target()]()

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
        comptime if a.type == DType.bfloat16:
            if k % simd_width == 0:
                kernel_func = GEMVAlgorithm.GEMV_KERNEL_VECTOR
            else:
                kernel_func = GEMVAlgorithm.GEMV_KERNEL
        else:
            kernel_func = GEMVAlgorithm.GEMV_KERNEL

    elif m == 1 and transpose_b == True:
        comptime if a.type in (DType.bfloat16, DType.float8_e4m3fn):
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
    c_size: DimList,
    c_type: DType,
    a_shape: DimList,
    a_type: DType,
    b_size: DimList,
    b_type: DType,
    //,
    parallelize: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_buf: NDBuffer[mut=True, c_type, 1, _, c_size],
    a_buf: NDBuffer[mut=False, a_type, 2, _, a_shape],
    b_buf: NDBuffer[mut=False, b_type, 1, _, b_size],
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
        comptime if elementwise_lambda_fn:
            comptime func = elementwise_lambda_fn.value()

            comptime for i in range(width):
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
        single_thread_blocking_override=not parallelize,
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
