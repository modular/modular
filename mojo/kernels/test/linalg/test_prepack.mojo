# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from List import DimList, Dim
from Matmul import pack_b
from Math import div_ceil
from DType import DType
from Range import range
from Buffer import Buffer, NDBuffer
from IO import print
from Index import StaticIntTuple


# CHECK-LABEL: test_prepack
fn test_prepack():
    print("== test_prepack")

    alias k = 10
    alias tile_k = 4
    alias simd_size = 2
    alias inner_size = 2 * simd_size
    alias n = 12
    alias tile_n = 8
    alias type = DType.f32.value

    alias k_padded = div_ceil(k, tile_k) * tile_k
    alias n_padded = div_ceil(n, tile_n) * tile_n

    alias src_shape_dyn = DimList.create_unknown[2]()
    alias dst_shape_dyn = DimList.create_unknown[2]()
    alias src_shape_static = DimList(k, n)
    alias dst_shape_static = DimList(k_padded, n_padded)

    let src_storage = Buffer[Dim(n * k), type].aligned_stack_allocation[64]()
    src_storage.fill(0)
    let dst_storage = Buffer[
        Dim(n_padded * k_padded), type
    ].aligned_stack_allocation[64]()
    dst_storage.fill(0)

    let src_buf = NDBuffer[2, src_shape_dyn, type](
        src_storage.data, src_shape_static, type
    )
    let dst_buf = NDBuffer[2, dst_shape_dyn, type](
        dst_storage.data, dst_shape_static, type
    )

    for i in range(src_storage.__len__()):
        src_storage[i] = i

    pack_b[False, simd_size, inner_size, type, src_shape_dyn, dst_shape_dyn](
        dst_buf,
        src_buf,
        tile_n,
        tile_k,
    )
    # CHECK: 0.000000
    # CHECK-NEXT: 1.000000
    # CHECK-NEXT: 2.000000
    # CHECK-NEXT: 3.000000
    # CHECK-NEXT: 12.000000
    # CHECK-NEXT: 13.000000
    # CHECK-NEXT: 14.000000
    # CHECK-NEXT: 15.000000
    # CHECK-NEXT: 24.000000
    # CHECK-NEXT: 25.000000
    # CHECK-NEXT: 26.000000
    # CHECK-NEXT: 27.000000
    # CHECK-NEXT: 36.000000
    # CHECK-NEXT: 37.000000
    # CHECK-NEXT: 38.000000
    # CHECK-NEXT: 39.000000
    # CHECK-NEXT: 4.000000
    # CHECK-NEXT: 5.000000
    # CHECK-NEXT: 6.000000
    # CHECK-NEXT: 7.000000
    # CHECK-NEXT: 16.000000
    # CHECK-NEXT: 17.000000
    # CHECK-NEXT: 18.000000
    # CHECK-NEXT: 19.000000
    # CHECK-NEXT: 28.000000
    # CHECK-NEXT: 29.000000
    # CHECK-NEXT: 30.000000
    # CHECK-NEXT: 31.000000
    # CHECK-NEXT: 40.000000
    # CHECK-NEXT: 41.000000
    # CHECK-NEXT: 42.000000
    # CHECK-NEXT: 43.000000
    # CHECK-NEXT: 8.000000
    # CHECK-NEXT: 9.000000
    # CHECK-NEXT: 10.000000
    # CHECK-NEXT: 11.000000
    # CHECK-NEXT: 20.000000
    # CHECK-NEXT: 21.000000
    # CHECK-NEXT: 22.000000
    # CHECK-NEXT: 23.000000
    # CHECK-NEXT: 32.000000
    # CHECK-NEXT: 33.000000
    # CHECK-NEXT: 34.000000
    # CHECK-NEXT: 35.000000
    # CHECK-NEXT: 44.000000
    # CHECK-NEXT: 45.000000
    # CHECK-NEXT: 46.000000
    # CHECK-NEXT: 47.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 48.000000
    # CHECK-NEXT: 49.000000
    # CHECK-NEXT: 50.000000
    # CHECK-NEXT: 51.000000
    # CHECK-NEXT: 60.000000
    # CHECK-NEXT: 61.000000
    # CHECK-NEXT: 62.000000
    # CHECK-NEXT: 63.000000
    # CHECK-NEXT: 72.000000
    # CHECK-NEXT: 73.000000
    # CHECK-NEXT: 74.000000
    # CHECK-NEXT: 75.000000
    # CHECK-NEXT: 84.000000
    # CHECK-NEXT: 85.000000
    # CHECK-NEXT: 86.000000
    # CHECK-NEXT: 87.000000
    # CHECK-NEXT: 52.000000
    # CHECK-NEXT: 53.000000
    # CHECK-NEXT: 54.000000
    # CHECK-NEXT: 55.000000
    # CHECK-NEXT: 64.000000
    # CHECK-NEXT: 65.000000
    # CHECK-NEXT: 66.000000
    # CHECK-NEXT: 67.000000
    # CHECK-NEXT: 76.000000
    # CHECK-NEXT: 77.000000
    # CHECK-NEXT: 78.000000
    # CHECK-NEXT: 79.000000
    # CHECK-NEXT: 88.000000
    # CHECK-NEXT: 89.000000
    # CHECK-NEXT: 90.000000
    # CHECK-NEXT: 91.000000
    # CHECK-NEXT: 56.000000
    # CHECK-NEXT: 57.000000
    # CHECK-NEXT: 58.000000
    # CHECK-NEXT: 59.000000
    # CHECK-NEXT: 68.000000
    # CHECK-NEXT: 69.000000
    # CHECK-NEXT: 70.000000
    # CHECK-NEXT: 71.000000
    # CHECK-NEXT: 80.000000
    # CHECK-NEXT: 81.000000
    # CHECK-NEXT: 82.000000
    # CHECK-NEXT: 83.000000
    # CHECK-NEXT: 92.000000
    # CHECK-NEXT: 93.000000
    # CHECK-NEXT: 94.000000
    # CHECK-NEXT: 95.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 96.000000
    # CHECK-NEXT: 97.000000
    # CHECK-NEXT: 98.000000
    # CHECK-NEXT: 99.000000
    # CHECK-NEXT: 108.000000
    # CHECK-NEXT: 109.000000
    # CHECK-NEXT: 110.000000
    # CHECK-NEXT: 111.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 100.000000
    # CHECK-NEXT: 101.000000
    # CHECK-NEXT: 102.000000
    # CHECK-NEXT: 103.000000
    # CHECK-NEXT: 112.000000
    # CHECK-NEXT: 113.000000
    # CHECK-NEXT: 114.000000
    # CHECK-NEXT: 115.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 104.000000
    # CHECK-NEXT: 105.000000
    # CHECK-NEXT: 106.000000
    # CHECK-NEXT: 107.000000
    # CHECK-NEXT: 116.000000
    # CHECK-NEXT: 117.000000
    # CHECK-NEXT: 118.000000
    # CHECK-NEXT: 119.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000
    # CHECK-NEXT: 0.000000

    for ii in range(dst_buf.dim[0]()):
        for jj in range(dst_buf.dim[1]()):
            print(dst_buf[StaticIntTuple[2](ii, jj)])


fn main():
    test_prepack()
