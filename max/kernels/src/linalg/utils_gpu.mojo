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

from hashlib.hasher import Hasher
from math import ceildiv
from sys import (
    env_get_int,
    has_nvidia_gpu_accelerator,
    sizeof,
)
from sys.ffi import external_call

from gpu import WARP_SIZE
from gpu.grid_controls import PDLLevel
from gpu.host import DeviceContext
from gpu.host.info import A100
from layout.tensor_core import get_mma_shape

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from gpu.host.device_context import DeviceBuffer

# ===------------------------------------------------------------------===#
# GPU Matmul Block Swizzling
# ===------------------------------------------------------------------===#


fn block_swizzle(
    block_idx: IndexList[2, **_], grid_dim: __type_of(block_idx)
) -> __type_of(block_idx):
    return _block_swizzle_by_scale[3](block_idx, grid_dim)


@always_inline
fn _block_swizzle_by_scale[
    scale0: UInt
](block_idx: IndexList[2, **_], grid_dim: __type_of(block_idx)) -> __type_of(
    block_idx
):
    """
    Block swizzling based on https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/threadblock_swizzle.h

    This version tries to partition the N dim (M x N matrix) into 2^scale partitions.
    If N can't be divided evenly then it reduces scale till 0, which means not swizzling.

    E.g. linearized block id for two partitions is

        B0 B1 | B4 B5    .vs. B0 B1 B2 B3
        B2 B3 | B6 B7         B4 B5 B6 B7

    This helps when N is very large e.g. 1024 x 32768 x 3072 in Replit 3B.
    """
    var scale = scale0
    # basically num_partitions = 2^3 = 8
    var num_partitions = 1 << scale
    # while griddim_x not divisible by num_partitions, reduce scale till scale is 0
    while (grid_dim.data[0] & (num_partitions - 1)) and scale > 0:
        scale -= 1
        num_partitions = 1 << scale

    # bx is the x coordinate of the block
    # by is the y coordinate of the block
    # bx = block_idx.data[0] >> scale
    var bx = block_idx.data[0] >> scale
    var by = (block_idx.data[1] << scale) + (
        (block_idx.data[0]) & ((1 << scale) - 1)
    )

    # for the number of rows of overflow, we want to move to next stripe
    # So if one overflow occurs and a stripe is six blocks wide, we slide bx six places to the right.
    # where a stripe is determined by remaining blocks of x
    # bx is now 5 + 1 * rows in a stripe or width of stripe
    bx = bx + by // grid_dim.data[1] * (grid_dim.data[0] >> scale)
    by = by % grid_dim.data[1]

    return __type_of(block_idx)(Int(bx), Int(by))


# ===------------------------------------------------------------------===#
# GPU Matmul Configuration
# ===------------------------------------------------------------------===#


@register_passable("trivial")
struct MatmulConfig[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = False,
](Copyable, Movable, Stringable, Writable):
    """Static configuration of GPU matmul."""

    var block_tile_shape: IndexList[3]

    var warp_tile_shape: IndexList[3]

    var mma_shape: IndexList[3]

    var num_pipeline_stages: UInt

    var num_k_partitions: UInt

    var k_group_size: UInt

    var num_warp_k_partitions: UInt

    var cluster_shape: IndexList[3]

    var num_consumer: UInt

    var partitioned_multicast: Bool

    var scheduler_hint: IndexList[3]

    var _pdl_level: PDLLevel

    alias accum_type = get_accum_type[a_type]()  # TODO: factor b_type

    # MMA is typically accumulated in FP32. The reduction over partitions may be
    # done in lower precision to reduce traffic to intermediate buffer. This is
    # acceptable since the number of partitions is small, typically < 8.
    # We see some discrepancy between BF16 and FP32 in KERN-933 and use FP32
    # by default to be safe. TODO: set via env var KERN-1002.

    alias split_k_reduction_scheme = env_get_int["SPLITK_REDUCTION_SCHEME", 2]()

    alias OUTPUT_PRECISION = 2

    alias ACCUM_PRECISION = 1

    # TODO: output precision will break the integration test.
    alias split_k_reduction_type = c_type if Self.OUTPUT_PRECISION == Self.split_k_reduction_scheme else Self.accum_type

    fn __init__(
        out self,
        *,
        block_tile_shape: IndexList[3] = Index(128, 128, 32),
        warp_tile_shape: IndexList[3] = Index(64, 64, 32),
        mma_shape: IndexList[3] = get_mma_shape[a_type, Self.accum_type](),
        cluster_shape: IndexList[3] = Index(1, 1, 1),
        num_pipeline_stages: UInt = 4,
        num_k_partitions: UInt = 1,
        k_group_size: UInt = 1,
        num_warp_k_partitions: UInt = 1,
        num_consumer: UInt = 1,
        partitioned_multicast: Bool = False,
        scheduler_hint: IndexList[3] = Index(2, 2, 2),
        pdl_level: PDLLevel = PDLLevel(),
    ):
        self.block_tile_shape = block_tile_shape
        self.warp_tile_shape = warp_tile_shape
        self.mma_shape = mma_shape
        self.num_pipeline_stages = num_pipeline_stages
        self.num_k_partitions = num_k_partitions
        self.k_group_size = k_group_size
        self.num_warp_k_partitions = num_warp_k_partitions
        self.cluster_shape = cluster_shape
        self.num_consumer = num_consumer
        self.partitioned_multicast = partitioned_multicast
        self.scheduler_hint = scheduler_hint
        self._pdl_level = pdl_level

    fn num_warps_m(self) -> UInt:
        return self.block_tile_shape[0] // self.warp_tile_shape[0]

    fn num_warps_n(self) -> UInt:
        return self.block_tile_shape[1] // self.warp_tile_shape[1]

    fn num_threads(self) -> UInt:
        return (
            self.num_warps_m()
            * self.num_warps_n()
            * self.num_warp_k_partitions
            * WARP_SIZE
        )

    fn shared_mem_usage(self) -> Int:
        return Int(
            _shared_memory_usage[a_type, b_type, c_type](
                self.block_tile_shape,
                Int(self.num_pipeline_stages),
                Int(self.num_warp_k_partitions),
            )
        )

    fn grid_dim(self, m: UInt, n: UInt) -> IndexList[3]:
        return Index(
            Int(ceildiv(n, self.block_tile_shape[1])),
            Int(ceildiv(m, self.block_tile_shape[0])),
            Int(self.num_k_partitions),
        )

    fn block_dim(self) -> IndexList[3]:
        return Index(Int(self.num_threads()), 1, 1)

    fn work_space_size(self, M: UInt, N: UInt) -> UInt:
        return M * N * (self.num_k_partitions - 1)

    fn pdl_level(self) -> PDLLevel:
        return self._pdl_level

    fn __eq__(self, rhs: MatmulConfig) -> Bool:
        alias static_info_match = a_type == rhs.a_type and b_type == rhs.b_type and c_type == rhs.c_type and transpose_b == rhs.transpose_b

        @parameter
        if static_info_match:
            return (
                self.block_tile_shape == rhs.block_tile_shape
                and self.num_pipeline_stages == rhs.num_pipeline_stages
            )
        else:
            return False

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("kernel_")
        writer.write(a_type, "_")
        writer.write(c_type, "_")
        # Use BNxBM to match cublas
        writer.write(
            self.block_tile_shape[1], "x", self.block_tile_shape[0], "_"
        )
        writer.write(self.num_pipeline_stages, "_")
        if self.num_k_partitions > 1:
            writer.write("k", self.num_k_partitions, "_")
        if self.num_warp_k_partitions > 1:
            writer.write("warp_k", self.num_warp_k_partitions, "_")
        # transpose A
        writer.write("N")
        # transpose B
        writer.write("T" if transpose_b else "N")

    fn __repr__(self) -> String:
        return String.write(self)

    fn __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with the underlying bytes.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher.update(a_type)
        hasher.update(b_type)
        hasher.update(c_type)
        hasher.update(transpose_b)
        hasher.update(self.block_tile_shape)
        hasher.update(self.warp_tile_shape)
        hasher.update(self.cluster_shape)
        hasher.update(self.num_pipeline_stages)
        hasher.update(self.num_k_partitions)
        hasher.update(self.num_warp_k_partitions)
        hasher.update(self.k_group_size)
        hasher.update(self.split_k_reduction_scheme)
        hasher.update(self.num_consumer)
        hasher.update(self.partitioned_multicast)


# Helper for choosing the base of BK based on type.
# Actual BK should be multiple of BK_base.
fn _bk_base[type: DType, amd_kernel: Bool = False]() -> Int:
    if type.is_float8():

        @parameter
        if amd_kernel:
            return 128
        else:
            return 64
    elif type.is_half_float():

        @parameter
        if amd_kernel:
            return 64
        else:
            return 32
    else:
        return 16


@always_inline
fn _shared_memory_usage[
    a_type: DType, b_type: DType, c_type: DType
](block_mnk: IndexList[3], num_pipeline_stages: Int, slice_k: Int = 1) -> UInt:
    # fmt: off
    var a_usage = slice_k * block_mnk[0] * block_mnk[2] * num_pipeline_stages * sizeof[a_type]()
    var b_usage = slice_k * block_mnk[1] * block_mnk[2] * num_pipeline_stages * sizeof[b_type]()
    # reduction within thread blocks is done with fp32
    var slice_k_reduction = block_mnk[0] * block_mnk[1] * (slice_k // 2) * sizeof[DType.float32]()
    var c_usage = block_mnk[0] * block_mnk[1] * \
                  sizeof[c_type]() if c_type.is_half_float() else 0
    # fmt: on
    return max(max(a_usage + b_usage, c_usage), slice_k_reduction)


@fieldwise_init
@register_passable("trivial")
struct MatmulKernels[
    a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool = False
](Copyable, Movable):
    """Supported matmul kernels.

    The configurations are named as: <arch>_<BNxBM>_<stages>.
    BK, mma shape, and warp tile shape are decided internally.
    """

    alias hopper_128x128_4 = MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=Index(128, 128, _bk_base[a_type]()),
        warp_tile_shape=Index(64, 64, _bk_base[a_type]()),
        num_pipeline_stages=4,
    )

    alias ampere_128x128_4 = MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=Index(128, 128, _bk_base[a_type]()),
        warp_tile_shape=Index(64, 64, _bk_base[a_type]()),
        num_pipeline_stages=4,
    )

    alias ampere_256x64_4 = MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=Index(64, 256, _bk_base[a_type]()),
        warp_tile_shape=Index(64, 64, _bk_base[a_type]()),
        num_pipeline_stages=4,
    )

    alias ampere_256x128_3 = MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=Index(128, 256, 2 * _bk_base[a_type]()),
        warp_tile_shape=Index(64, 64, 2 * _bk_base[a_type]()),
        num_pipeline_stages=3,
    )

    alias tuning_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=Index(
            env_get_int["TUNE_BM", 128](),
            env_get_int["TUNE_BN", 128](),
            env_get_int["TUNE_BK", 32](),
        ),
        warp_tile_shape=Index(
            env_get_int["TUNE_WM", 64](),
            env_get_int["TUNE_WN", 64](),
            env_get_int["TUNE_BK", 32](),
        ),
        num_pipeline_stages=env_get_int["TUNE_NUM_STAGES", 4](),
        num_k_partitions=env_get_int["TUNE_NUM_K_PARTITIONS", 1](),
        num_warp_k_partitions=env_get_int["TUNE_NUM_WARP_K_PARTITIONS", 1](),
    )


fn select_config[
    a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool = False
](M: Int, N: Int, K: Int, ctx: DeviceContext) -> MatmulConfig[
    a_type, b_type, c_type, transpose_b
]:
    # Select an optimal matmul config by heuristic.
    # The heuristic is to choose the parameters leading to min workload per SM.
    # The work load is estimated as

    #     work_per_SM = BM * BN * k_partition * num_waves.

    # * BM, BN are the thread block's M and N. Here we assume single block per SM,
    #   which is valid compute-bound gemm in practice.
    # * k_partition is the K dim for one partition in split-k, which equals to the
    #   original K if split-k is not used.
    # * num_waves is the maximum thread blocks that are dispatched to a SM.
    #   E.g. 128 blocks to A100's 108 SMs, one SM at most computes two blocks.

    alias gpu_info = ctx.default_device_info

    # TODO(KERN-1310): This disables split-k for AMD, enable it after fixing KERN-1310.
    alias max_num_k_partitions = 8 if has_nvidia_gpu_accelerator() else 1
    alias min_k_partition = 1024

    # Initial values overwritten in loop
    var best_bmnk = Index(128, 128, _bk_base[a_type]())
    var best_num_k_partitions = 1
    var best_num_stages = 4
    var min_num_waves = Int.MAX
    var min_work_per_SM = Int.MAX

    alias _128x128_4 = Index(128, 128, _bk_base[a_type](), 4)
    alias _256x64_4 = Index(64, 256, _bk_base[a_type](), 4)
    # Only enable this when the target is exactly A100. We use A100 properties
    # for target="gpu" (default) on A10, L4. This avoids breaking tests there.
    # The tile is skipped in the loop for exceeding shared memory capacity when
    # sm_80 is present in target.
    alias _256x128_3 = Index(
        128, 256, 2 * _bk_base[a_type](), 3
    ) if gpu_info is A100 else Index(1024, 1024, 1024, 1024)

    alias opt_list = [_128x128_4, _256x64_4, _256x128_3]

    for bmnk_stage in opt_list:
        var bm = bmnk_stage[0]
        var bn = bmnk_stage[1]
        var bk = bmnk_stage[2]
        var num_stages = bmnk_stage[3]
        var num_blocks = ceildiv(M, bm) * ceildiv(N, bn)
        var num_waves_base = ceildiv(num_blocks, A100.sm_count)

        # Skip if it requires more shared memory than the GPU supports.
        if (
            _shared_memory_usage[a_type, b_type, c_type](
                Index(bm, bn, bk), num_stages
            )
            > gpu_info.shared_memory_per_multiprocessor
        ):
            continue

        var allowed_num_k_partitions = (
            1 if num_waves_base > 3 else max_num_k_partitions
        )

        # Traverse split-k possibilities to find the min work per SM.
        for num_k_partitions in range(1, allowed_num_k_partitions + 1):
            # Skip if partition becomes too small.
            var k_partition = K // num_k_partitions
            if k_partition < min_k_partition:
                break

            # Skip non-divisible K, TODO: generalize e.g. 4, 4 3
            if K < num_k_partitions * bk:
                break

            # Skip pipeline stages = 3 for non-split-k cases since default
            # 4 stage kernel seems faster on A100.
            # TODO: shouldn't hardcode this way, needs a long-term solution.
            if num_k_partitions == 1 and num_stages != 4:
                continue

            var num_waves = ceildiv(
                num_k_partitions * num_blocks, A100.sm_count
            )
            var work_per_SM = bm * bn * k_partition * num_waves

            # Minimize work per SM but intuitively waves shouldn't increase too much.
            if num_waves <= 2 * num_waves_base and (
                work_per_SM < min_work_per_SM
                or (
                    work_per_SM == min_work_per_SM and num_waves < min_num_waves
                )
            ):
                best_bmnk[0] = bm
                best_bmnk[1] = bn
                best_bmnk[2] = bk
                best_num_stages = num_stages
                best_num_k_partitions = num_k_partitions

                min_work_per_SM = work_per_SM
                min_num_waves = num_waves

    return MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=best_bmnk,
        warp_tile_shape=Index(64, 64, best_bmnk[2]),
        num_pipeline_stages=best_num_stages,
        num_k_partitions=best_num_k_partitions,
    )


fn create_hilbert_lut(
    ctx: DeviceContext, grid_x: Int, grid_y: Int
) raises -> DeviceBuffer[DType.uint32]:
    """Precompute Hilbert-curve block swizzle lookup-table for a rectangular grid.

    The returned device pointer refers to a 1-D UInt32 array of length
        grid_x * grid_y.
    For linear (row-major) block id `id`, the packed value at `lut[id]`
    encodes the swizzled coordinates:  upper 16-bits = y, lower 16-bits = x.
    """
    var num_blocks = grid_x * grid_y
    # Allocate temporary host buffer.
    var host_ptr = UnsafePointer[Scalar[DType.uint32]].alloc(num_blocks)

    # Next power-of-two square dimension enclosing the rectangle.
    var dim_pow2 = 1
    while dim_pow2 < grid_x or dim_pow2 < grid_y:
        dim_pow2 <<= 1

    var seen: Int = 0
    var d: UInt32 = 0
    while seen < num_blocks:
        # Decode Hilbert distance d to (hx,hy).
        var hx: UInt32 = 0
        var hy: UInt32 = 0
        var t: UInt32 = d
        var s: UInt32 = 1
        while s < UInt32(dim_pow2):
            var rx = (t >> 1) & 1
            var ry = (t ^ rx) & 1
            if ry == 0:
                if rx == 1:
                    hx = UInt32(s) - 1 - hx
                    hy = UInt32(s) - 1 - hy
                # rotate
                var tmp = hx
                hx = hy
                hy = tmp
            hx += UInt32(s) * rx
            hy += UInt32(s) * ry
            t >>= 2
            s <<= 1

        if hx < UInt32(grid_x) and hy < UInt32(grid_y):
            host_ptr[seen] = Scalar[DType.uint32]((hy << 16) | hx)  # pack (y,x)
            seen += 1
        d += 1

    # Allocate device buffer and copy.
    var device_buf = ctx.enqueue_create_buffer[DType.uint32](num_blocks)
    ctx.enqueue_copy(device_buf, host_ptr)
    host_ptr.free()
    return device_buf


fn get_hilbert_lut_with_cache(
    ctx: DeviceContext, grid_x: Int, grid_y: Int
) raises -> DeviceBuffer[DType.uint32]:
    """Get Hilbert lookup table using global cache (no struct needed)."""
    var key_str = String("hilbert_lut_", grid_x, "_", grid_y)

    # use runtime lookup since key is computed at runtime
    var cached_ptr = external_call[
        "KGEN_CompilerRT_GetGlobalOrNull", OpaquePointer
    ](key_str.unsafe_cstr_ptr(), key_str.byte_length())

    if cached_ptr:
        var device_ptr = cached_ptr.bitcast[Scalar[DType.uint32]]()
        var num_blocks = grid_x * grid_y
        # the cached buffer stays alive as long as the program runs
        return DeviceBuffer[DType.uint32](
            ctx, device_ptr, num_blocks, owning=False
        )

    # not in cache :(
    var buf = create_hilbert_lut(ctx, grid_x, grid_y)
    var device_ptr = buf._unsafe_ptr()
    var num_blocks = grid_x * grid_y

    # store the device pointer directly in global cache
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(key_str),
        device_ptr.bitcast[NoneType](),
    )

    # the buffer will live for the duration of the program
    _ = buf.take_ptr()

    return DeviceBuffer[DType.uint32](ctx, device_ptr, num_blocks, owning=False)
