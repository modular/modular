# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from linalg.utils_gpu import select_config


# CHECK-LABEL: test_matmul_selection
fn test_matmul_selection():
    print("=== test_matmul_selection")

    var config0 = select_config[DType.bfloat16, DType.bfloat16, DType.bfloat16](
        1024, 1024, 1024
    )
    # CHECK: ampere_bfloat16_bfloat16_128x128_4_NN
    # CHECK: (128, 128, 32)
    # CHECK: (64, 64, 32)
    # CHECK: 4
    # CHECK: 1
    print(config0)
    print(config0.block_tile_shape)
    print(config0.warp_tile_shape)
    print(config0.num_pipeline_stages)
    print(config0.num_k_partitions)

    var config1 = select_config[
        DType.bfloat16, DType.bfloat16, DType.bfloat16, transpose_b=True
    ](528, 3072, 3072)
    # CHECK: ampere_bfloat16_bfloat16_256x64_4_NT
    # CHECK: (64, 256, 32)
    # CHECK: (64, 64, 32)
    # CHECK: 4
    # CHECK: 1
    print(config1)
    print(config1.block_tile_shape)
    print(config1.warp_tile_shape)
    print(config1.num_pipeline_stages)
    print(config1.num_k_partitions)

    var config2 = select_config[
        DType.bfloat16, DType.bfloat16, DType.bfloat16, transpose_b=True
    ](16, 4096, 14336)
    # CHECK: ampere_bfloat16_bfloat16_256x64_4_k4_NT
    # CHECK: (64, 256, 32)
    # CHECK: (64, 64, 32)
    # CHECK: 4
    # CHECK: 4
    # CHECK: 196608
    print(config2)
    print(config2.block_tile_shape)
    print(config2.warp_tile_shape)
    print(config2.num_pipeline_stages)
    print(config2.num_k_partitions)
    print(config2.work_space_size(16, 4096))


fn main():
    test_matmul_selection()
