# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import ceildiv
from sys import sizeof

from gpu import WARP_SIZE
from gpu.host.info import A100
from layout.tensor_core import (
    TensorCore,
    get_accum_type,
    get_fragment_size,
    get_mma_shape,
)
from utils.index import Index, StaticIntTuple

# ===------------------------------------------------------------------===#
# GPU Matmul Block Swizzling
# ===------------------------------------------------------------------===#


@always_inline
fn block_swizzle(
    block_idx: StaticIntTuple[2], grid_dim: StaticIntTuple[2]
) -> StaticIntTuple[2]:
    return _block_swizzle_by_scale[3](block_idx, grid_dim)


@always_inline
fn _block_swizzle_by_scale[
    scale0: Int
](block_idx: StaticIntTuple[2], grid_dim: StaticIntTuple[2]) -> StaticIntTuple[
    2
]:
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
    var num_partitions = (1 << scale)
    while (grid_dim[0] & (num_partitions - 1)) and scale > 0:
        scale -= 1
        num_partitions = 1 << scale

    var bx = block_idx[0] >> scale
    var by = (block_idx[1] << scale) + ((block_idx[0]) & ((1 << scale) - 1))
    bx = bx + by // grid_dim[1] * (grid_dim[0] >> scale)
    by = by % grid_dim[1]

    return Index(bx, by)


# ===------------------------------------------------------------------===#
# GPU Matmul Cofiguration
# ===------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct MatmulConfig[
    a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool = False
](Stringable, Formattable):
    """Static configuration of GPU matmul."""

    var block_tile_shape: StaticIntTuple[3]

    var warp_tile_shape: StaticIntTuple[3]

    var num_pipeline_stages: UInt

    var num_k_partitions: UInt

    alias accum_type = get_accum_type[a_type]()  # TODO: factor b_type
    alias mma_shape = get_mma_shape[a_type, get_accum_type[a_type]()]()

    # MMA is typically accumulated in FP32. The reduction over partitions may be
    # done in lower precision to reduce traffic to intermediate buffer. This is
    # acceptible since the number of partitions is small, typically < 8. In case
    # it overflows, we should bump precision to fp32.
    alias split_k_reduction_type = c_type

    fn __init__(
        inout self,
        block_tile_shape: StaticIntTuple[3] = Index(128, 128, 32),
        warp_tile_shape: StaticIntTuple[3] = Index(64, 64, 32),
        num_pipeline_stages: UInt = 4,
        num_k_partitions: UInt = 1,
    ):
        self.block_tile_shape = block_tile_shape
        self.warp_tile_shape = warp_tile_shape
        self.num_pipeline_stages = num_pipeline_stages
        self.num_k_partitions = num_k_partitions

    fn num_warps_m(self) -> UInt:
        return self.block_tile_shape[0] // self.warp_tile_shape[0]

    fn num_warps_n(self) -> UInt:
        return self.block_tile_shape[1] // self.warp_tile_shape[1]

    fn num_threads(self) -> UInt:
        return self.num_warps_m() * self.num_warps_n() * WARP_SIZE

    fn shared_mem_usage(self) -> UInt:
        # fmt: off
        var a_usage = self.block_tile_shape[0] * self.block_tile_shape[2] * \
                      self.num_pipeline_stages * sizeof[a_type]()
        var b_usage = self.block_tile_shape[1] * self.block_tile_shape[2] * \
                      self.num_pipeline_stages * sizeof[b_type]()
        var c_usage = self.block_tile_shape[0] * self.block_tile_shape[1] * \
                      sizeof[Self.accum_type]() if c_type.is_half_float() else 0
        # fmt: on

        return max(a_usage + b_usage, c_usage)

    fn grid_dim(self, m: UInt, n: UInt) -> StaticIntTuple[3]:
        return Index(
            int(ceildiv(n, self.block_tile_shape[1])),
            int(ceildiv(m, self.block_tile_shape[0])),
            int(self.num_k_partitions),
        )

    fn block_dim(self) -> StaticIntTuple[3]:
        return Index(int(self.num_threads()), 1, 1)

    fn work_space_size(self, M: UInt, N: UInt) -> UInt:
        return M * N * (self.num_k_partitions - 1)

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
        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        writer.write("ampere_")
        writer.write(a_type, "_")
        writer.write(c_type, "_")
        # Use BNxBM to match cublas
        writer.write(
            self.block_tile_shape[1], "x", self.block_tile_shape[0], "_"
        )
        writer.write(self.num_pipeline_stages, "_")
        if self.num_k_partitions > 1:
            writer.write("k", self.num_k_partitions, "_")
        # transpose A
        writer.write("N")
        # transpose B
        writer.write("T" if transpose_b else "N")


# Helper for choosing the base of BK based on type.
# Actual BK should be multiple of BK_base.
fn _bk_base[type: DType]() -> Int:
    return 32 if type in (DType.float16, DType.bfloat16) else 16


@value
@register_passable("trivial")
struct MatmulKernels[
    a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool = False
]:
    """Supported matmul kernels.

    The configurations are named as: <arch>_<BNxBM>_<stages>.
    BK, mma shape, and warp tile shape are decided internally.
    """

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


fn select_config[
    a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool = False
](M: Int, N: Int, K: Int) -> MatmulConfig[a_type, b_type, c_type, transpose_b]:
    # A super simple heuristic is to just choose the tile shapes that lead to
    # min waves. Only support two shapes for now.

    alias max_k_partitions = 8

    var best_bmnk = Index(128, 128, _bk_base[a_type]())
    var min_num_waves = 1000
    # var min_num_waves = ceildiv(
    #     ceildiv(M, best_bmnk[0]) * ceildiv(N, best_bmnk[1]), A100.sm_count
    # )
    var best_num_k_partitions = 1

    for bm_bn in List(Index(128, 128), Index(64, 256)):
        var bm = bm_bn[][0]
        var bn = bm_bn[][1]
        var num_blocks = ceildiv(M, bm) * ceildiv(N, bn)

        # Enable split K when only < 50% of SMs are used.
        var num_k_partitions = 1
        if num_blocks < A100.sm_count // 2 and K % 32 == 0:
            num_k_partitions = min(
                max_k_partitions, ceildiv(A100.sm_count, num_blocks)
            )
            while K % (num_k_partitions * 32) != 0:
                num_k_partitions -= 1

        var num_waves = ceildiv(num_blocks * num_k_partitions, A100.sm_count)
        if num_waves < min_num_waves:
            best_bmnk[0] = bm
            best_bmnk[1] = bn
            min_num_waves = num_waves
            best_num_k_partitions = num_k_partitions

    return MatmulConfig[a_type, b_type, c_type, transpose_b](
        block_tile_shape=best_bmnk,
        warp_tile_shape=Index(64, 64, best_bmnk[2]),
        num_pipeline_stages=4,
        num_k_partitions=best_num_k_partitions,
    )
