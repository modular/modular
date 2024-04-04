# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from LinAlg.MatmulUtils import SubMatmulConfig, get_partitioned_matmul_mojo

alias kernel_height = 6
alias kernel_width = 64
alias a_type = DType.float32
alias b_type = DType.float32
alias c_type = DType.float32


# CHECK-LABEL: test_partition
fn test_partition():
    print("== test_partition")
    # Matmul dimensions
    var M = 4
    var N = 768
    var K = 3072

    var num_tasks = 8

    var config: SubMatmulConfig

    config = get_partitioned_matmul_mojo[
        a_type, b_type, c_type, kernel_height, kernel_width
    ](M, N, K, 0, num_tasks)
    # CHECK: (0, 0, 0)
    print(config.offset)
    # CHECK: (4, 128, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[
        a_type, b_type, c_type, kernel_height, kernel_width
    ](M, N, K, 1, num_tasks)
    # CHECK: (0, 128, 0)
    print(config.offset)
    # CHECK: (4, 128, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[
        a_type, b_type, c_type, kernel_height, kernel_width
    ](M, N, K, 2, num_tasks)
    # CHECK: (0, 256, 0)
    print(config.offset)
    # CHECK: (4, 128, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[
        a_type, b_type, c_type, kernel_height, kernel_width
    ](M, N, K, 3, num_tasks)
    # CHECK: (0, 384, 0)
    print(config.offset)
    # CHECK: (4, 128, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[
        a_type, b_type, c_type, kernel_height, kernel_width
    ](M, N, K, 4, num_tasks)
    # CHECK: (0, 512, 0)
    print(config.offset)
    # CHECK: (4, 64, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[
        a_type, b_type, c_type, kernel_height, kernel_width
    ](M, N, K, 5, num_tasks)
    # CHECK: (0, 576, 0)
    print(config.offset)
    # CHECK: (4, 64, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[
        a_type, b_type, c_type, kernel_height, kernel_width
    ](M, N, K, 6, num_tasks)
    # CHECK: (0, 640, 0)
    print(config.offset)
    # CHECK: (4, 64, 3072)
    print(config.shape)

    config = get_partitioned_matmul_mojo[
        a_type, b_type, c_type, kernel_height, kernel_width
    ](M, N, K, 7, num_tasks)
    # CHECK: (0, 704, 0)
    print(config.offset)
    # CHECK: (4, 64, 3072)
    print(config.shape)


fn main():
    test_partition()
