# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: H100-GPU
# RUN: %mojo-no-debug %s

from gpu.cluster import block_rank_in_cluster, cluster_sync
from gpu.host import DeviceContext, Dim
from gpu.host._compile import _get_gpu_target
from gpu.id import block_idx, cluster_dim, cluster_idx


fn test_cluster_sync_kernel():
    var block_rank = block_rank_in_cluster()
    var num_blocks_in_cluster = cluster_dim.x * cluster_dim.y * cluster_dim.z

    for i in range(num_blocks_in_cluster):
        if block_rank == i:
            print(block_rank)
        cluster_sync()


# CHECK-LABEL: test_cluster_sync
# CHECK: 0
# CHECK: 1
# CHECK: 2
# CHECK: 3
# CHECK: 4
# CHECK: 5
# CHECK: 6
# CHECK: 7
fn test_cluster_sync(ctx: DeviceContext) raises:
    print("== test_cluster_sync")
    ctx.enqueue_function[test_cluster_sync_kernel](
        grid_dim=(2, 2, 2),
        block_dim=(1),
        cluster_dim=Dim((2, 2, 2)),
    )
    ctx.synchronize()


def main():
    with DeviceContext() as ctx:
        test_cluster_sync(ctx)
