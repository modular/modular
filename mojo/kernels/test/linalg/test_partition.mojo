# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from MatmulUtils import get_partitioned_matmul_mojo, SubMatmulConfig


# CHECK-LABEL: test_partition
fn test_partition():
    print("== test_partition")
    # Matmul dimensions
    let M = 4
    let N = 768
    let K = 3072

    let num_tasks = 8

    var config: SubMatmulConfig

    config = get_partitioned_matmul_mojo[6, 64](M, N, K, 0, num_tasks)
    # CHECK: (0, 0, 0)
    print(config.offset)
    # CHECK: (4, 192, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[6, 64](M, N, K, 1, num_tasks)
    # CHECK: (0, 192, 0)
    print(config.offset)
    # CHECK: (4, 192, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[6, 64](M, N, K, 2, num_tasks)
    # CHECK: (0, 384, 0)
    print(config.offset)
    # CHECK: (4, 192, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[6, 64](M, N, K, 3, num_tasks)
    # CHECK: (0, 576, 0)
    print(config.offset)
    # CHECK: (4, 192, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[6, 64](M, N, K, 4, num_tasks)
    # CHECK: (6, 0, 0)
    print(config.offset)
    # CHECK: (-2, 192, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[6, 64](M, N, K, 5, num_tasks)
    # CHECK: (6, 192, 0)
    print(config.offset)
    # CHECK: (-2, 192, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[6, 64](M, N, K, 6, num_tasks)
    # CHECK: (6, 384, 0)
    print(config.offset)
    # CHECK: (-2, 192, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[6, 64](M, N, K, 7, num_tasks)
    # CHECK: (6, 576, 0)
    print(config.offset)
    # CHECK: (-2, 192, 3072)
    print(config.shape)


fn main():
    test_partition()
