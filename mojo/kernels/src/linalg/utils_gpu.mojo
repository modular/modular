# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

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
