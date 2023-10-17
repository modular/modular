# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -parsing-stdlib -debug-level full %s | FileCheck %s

from sys.info import simdwidthof

from ConvUtils import ConvShape, get_conv_num_partitions
from MatmulUtils import partition_work

from utils.index import Index, StaticIntTuple

# CHECK-LABEL: test_conv_partition
fn test_partition():
    print("== test_conv_partition")
    alias micro_kernel_height = 6
    alias micro_kernel_width = 4
    alias simd_size = 16
    alias micro_kernel_f_size = micro_kernel_width * simd_size
    alias num_threads = 8

    let conv_shape = ConvShape {
        n: 1,
        h: 56,
        w: 56,
        c: 64,
        out_h: 56,
        out_w: 56,
        f: 64,
        r: 3,
        s: 3,
        stride: Index(1, 1),
        dilation: Index(1, 1),
        pad_h: Index(1, 1),
        pad_w: Index(1, 1),
        num_groups: 1,
    }

    # Matmul dimensions
    let num_partitions = get_conv_num_partitions[
        micro_kernel_height, micro_kernel_f_size
    ](num_threads, conv_shape)

    # CHECK: (1, 1, 1, 8)
    print(num_partitions)

    print("n partitions")
    # CHECK: (0, 1)
    for i in range(num_partitions[0]):
        let n_range = partition_work(i, num_partitions[0], conv_shape.n, 1)
        print(n_range)

    print("c partitions")
    # CHECK: (0, 64)
    for i in range(num_partitions[1]):
        let c_range = partition_work(i, num_partitions[1], conv_shape.c, 1)
        print(c_range)

    print("f partitions")
    # CHECK: (0, 64)
    for i in range(num_partitions[2]):
        let f_range = partition_work(
            i, num_partitions[2], conv_shape.f, micro_kernel_f_size
        )
        print(f_range)

    print("ho partitions")
    # CHECK: (0, 7)
    # CHECK: (7, 7)
    # CHECK: (14, 7)
    # CHECK: (21, 7)
    # CHECK: (28, 7)
    # CHECK: (35, 7)
    # CHECK: (42, 7)
    # CHECK: (49, 7)
    for i in range(num_partitions[3]):
        let ho_range = partition_work(i, num_partitions[3], conv_shape.out_h, 1)
        print(ho_range)


fn main():
    test_partition()
