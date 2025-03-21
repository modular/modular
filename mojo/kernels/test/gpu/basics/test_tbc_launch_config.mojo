# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug %s | FileCheck %s


from collections import OptionalReg

from gpu.host import DeviceContext, Dim, FuncAttribute
from gpu.host._compile import _get_gpu_target
from gpu.id import block_idx, cluster_idx
from gpu.cluster import block_rank_in_cluster


fn test_thread_block_cluster():
    var rank = block_rank_in_cluster()
    print(
        "cluster_ids=(",
        cluster_idx.x,
        ",",
        cluster_idx.y,
        ",",
        cluster_idx.z,
        ")",
        "block_ids=(",
        block_idx.x,
        ",",
        block_idx.y,
        ")",
        "block_rank=(",
        rank,
        ")",
    )


# CHECK-LABEL: test_tbc_launch_config_2x1x1
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 1 , 0 ) block_rank=( 1 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 0 , 0 ) block_rank=( 0 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 0 , 1 ) block_rank=( 0 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 1 , 1 ) block_rank=( 1 )
fn test_tbc_launch_config_2x1x1(ctx: DeviceContext) raises:
    print("== test_tbc_launch_config_2x1x1")
    ctx.enqueue_function[test_thread_block_cluster](
        grid_dim=(2, 2),
        block_dim=(1),
        cluster_dim=OptionalReg[Dim]((2, 1, 1)),
    )
    ctx.synchronize()


# CHECK-LABEL: test_tbc_launch_config_1x2x1
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 0 , 1 ) block_rank=( 1 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 0 , 0 ) block_rank=( 0 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 1 , 1 ) block_rank=( 1 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 1 , 0 ) block_rank=( 0 )
fn test_tbc_launch_config_1x2x1(ctx: DeviceContext) raises:
    print("== test_tbc_launch_config_1x2x1")
    ctx.enqueue_function[test_thread_block_cluster](
        grid_dim=(2, 2),
        block_dim=(1),
        cluster_dim=OptionalReg[Dim]((1, 2, 1)),
    )
    ctx.synchronize()


# CHECK-LABEL: test_tbc_launch_config_2x2x2
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 2 , 1 ) block_rank=( 6 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 3 , 1 ) block_rank=( 7 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 2 , 1 ) block_rank=( 2 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 2 , 0 ) block_rank=( 0 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 3 , 1 ) block_rank=( 3 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 0 , 3 ) block_rank=( 6 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 3 , 0 ) block_rank=( 1 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 1 , 3 ) block_rank=( 7 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 0 , 3 ) block_rank=( 2 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 1 , 3 ) block_rank=( 3 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 3 , 2 ) block_rank=( 5 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 0 , 1 ) block_rank=( 6 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 1 , 1 ) block_rank=( 7 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 0 , 0 ) block_rank=( 0 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 0 , 2 ) block_rank=( 4 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 1 , 2 ) block_rank=( 1 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 0 , 2 ) block_rank=( 0 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 1 , 2 ) block_rank=( 5 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 3 , 0 ) block_rank=( 5 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 2 , 0 ) block_rank=( 4 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 2 , 3 ) block_rank=( 6 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 3 , 3 ) block_rank=( 7 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 2 , 2 ) block_rank=( 0 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 3 , 2 ) block_rank=( 1 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 2 , 3 ) block_rank=( 2 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 1 , 1 ) block_rank=( 3 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 0 , 1 ) block_rank=( 2 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 1 , 0 ) block_rank=( 1 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 1 , 0 ) block_rank=( 5 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 0 , 0 ) block_rank=( 4 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 3 , 3 ) block_rank=( 3 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 2 , 2 ) block_rank=( 4 )
fn test_tbc_launch_config_2x2x2(ctx: DeviceContext) raises:
    print("== test_tbc_launch_config_2x2x2")
    ctx.enqueue_function[test_thread_block_cluster](
        grid_dim=(4, 4, 2),
        block_dim=(1),
        cluster_dim=OptionalReg[Dim]((2, 2, 2)),
    )
    ctx.synchronize()


def main():
    with DeviceContext() as ctx:
        test_tbc_launch_config_2x1x1(ctx)
        test_tbc_launch_config_1x2x1(ctx)
        test_tbc_launch_config_2x2x2(ctx)
