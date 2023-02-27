# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file only tests the Apple AMX matmul functionality which is defined as a
# A^T.B where A and B are 16x16 f32 matrices.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: apple-m1
# RUN: lit %s | FileCheck %s

from AppleAMX import amx_detail
from Buffer import NDBuffer
from DType import DType
from F32 import F32
from Int import Int
from Index import StaticIntTuple
from IO import print
from List import create_kgen_list
from Range import range
from SIMD import SIMD
from TargetInfo import is_apple_m1, sizeof


fn fill_a(
    buf: NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](16, 16),
        DType.f32.value,
    ]
):
    # Fills the A matrix with the following values row + 2*col
    let rows = 16
    let cols = 16
    for i in range(rows):
        for j in range(cols):
            let val = F32(i + 2 * j)
            buf.__setitem__(StaticIntTuple[2](i, j), val.value)


fn fill_b(
    buf: NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](16, 16),
        DType.f32.value,
    ]
):
    # Fills the A matrix with the following values row/(col + 1) + col
    let rows = 16
    let cols = 16
    for i in range(rows):
        for j in range(cols):
            let val = F32(i // (j + 1) + j)
            buf.__setitem__(StaticIntTuple[2](i, j), val.value)


fn clear_c(
    buf: NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](16, 16),
        DType.f32.value,
    ]
):
    buf.zero()


fn print_matrix(
    buf: NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](16, 16),
        DType.f32.value,
    ]
):
    # Fills the A matrix with the following values row/(col + 1) + col + 3
    let rows = 16
    let cols = 16
    for i in range(rows):
        for j in range(cols):
            print(buf[i, j])


# CHECK-LABEL: test_amx_matmul
fn test_amx_matmul():
    if is_apple_m1():
        print("== test_amx_matmul\n")

    var a_matrix = NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](16, 16),
        DType.f32.value,
    ].stack_allocation()
    var b_matrix = NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](16, 16),
        DType.f32.value,
    ].stack_allocation()
    var c_matrix = NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](16, 16),
        DType.f32.value,
    ].stack_allocation()

    fill_a(a_matrix)
    fill_b(b_matrix)
    clear_c(c_matrix)

    amx_detail.dot_at_b(c_matrix, a_matrix, b_matrix)

    # CHECK: 1240.000000
    # CHECK: 708.000000
    # CHECK: 615.000000
    # CHECK: 620.000000
    # CHECK: 680.000000
    # CHECK: 759.000000
    # CHECK: 848.000000
    # CHECK: 932.000000
    # CHECK: 1044.000000
    # CHECK: 1155.000000
    # CHECK: 1265.000000
    # CHECK: 1374.000000
    # CHECK: 1482.000000
    # CHECK: 1589.000000
    # CHECK: 1695.000000
    # CHECK: 1800.000000
    # CHECK: 1480.000000
    # CHECK: 852.000000
    # CHECK: 749.000000
    # CHECK: 764.000000
    # CHECK: 844.000000
    # CHECK: 947.000000
    # CHECK: 1062.000000
    # CHECK: 1172.000000
    # CHECK: 1314.000000
    # CHECK: 1455.000000
    # CHECK: 1595.000000
    # CHECK: 1734.000000
    # CHECK: 1872.000000
    # CHECK: 2009.000000
    # CHECK: 2145.000000
    # CHECK: 2280.000000
    # CHECK: 1720.000000
    # CHECK: 996.000000
    # CHECK: 883.000000
    # CHECK: 908.000000
    # CHECK: 1008.000000
    # CHECK: 1135.000000
    # CHECK: 1276.000000
    # CHECK: 1412.000000
    # CHECK: 1584.000000
    # CHECK: 1755.000000
    # CHECK: 1925.000000
    # CHECK: 2094.000000
    # CHECK: 2262.000000
    # CHECK: 2429.000000
    # CHECK: 2595.000000
    # CHECK: 2760.000000
    # CHECK: 1960.000000
    # CHECK: 1140.000000
    # CHECK: 1017.000000
    # CHECK: 1052.000000
    # CHECK: 1172.000000
    # CHECK: 1323.000000
    # CHECK: 1490.000000
    # CHECK: 1652.000000
    # CHECK: 1854.000000
    # CHECK: 2055.000000
    # CHECK: 2255.000000
    # CHECK: 2454.000000
    # CHECK: 2652.000000
    # CHECK: 2849.000000
    # CHECK: 3045.000000
    # CHECK: 3240.000000
    # CHECK: 2200.000000
    # CHECK: 1284.000000
    # CHECK: 1151.000000
    # CHECK: 1196.000000
    # CHECK: 1336.000000
    # CHECK: 1511.000000
    # CHECK: 1704.000000
    # CHECK: 1892.000000
    # CHECK: 2124.000000
    # CHECK: 2355.000000
    # CHECK: 2585.000000
    # CHECK: 2814.000000
    # CHECK: 3042.000000
    # CHECK: 3269.000000
    # CHECK: 3495.000000
    # CHECK: 3720.000000
    # CHECK: 2440.000000
    # CHECK: 1428.000000
    # CHECK: 1285.000000
    # CHECK: 1340.000000
    # CHECK: 1500.000000
    # CHECK: 1699.000000
    # CHECK: 1918.000000
    # CHECK: 2132.000000
    # CHECK: 2394.000000
    # CHECK: 2655.000000
    # CHECK: 2915.000000
    # CHECK: 3174.000000
    # CHECK: 3432.000000
    # CHECK: 3689.000000
    # CHECK: 3945.000000
    # CHECK: 4200.000000
    # CHECK: 2680.000000
    # CHECK: 1572.000000
    # CHECK: 1419.000000
    # CHECK: 1484.000000
    # CHECK: 1664.000000
    # CHECK: 1887.000000
    # CHECK: 2132.000000
    # CHECK: 2372.000000
    # CHECK: 2664.000000
    # CHECK: 2955.000000
    # CHECK: 3245.000000
    # CHECK: 3534.000000
    # CHECK: 3822.000000
    # CHECK: 4109.000000
    # CHECK: 4395.000000
    # CHECK: 4680.000000
    # CHECK: 2920.000000
    # CHECK: 1716.000000
    # CHECK: 1553.000000
    # CHECK: 1628.000000
    # CHECK: 1828.000000
    # CHECK: 2075.000000
    # CHECK: 2346.000000
    # CHECK: 2612.000000
    # CHECK: 2934.000000
    # CHECK: 3255.000000
    # CHECK: 3575.000000
    # CHECK: 3894.000000
    # CHECK: 4212.000000
    # CHECK: 4529.000000
    # CHECK: 4845.000000
    # CHECK: 5160.000000
    # CHECK: 3160.000000
    # CHECK: 1860.000000
    # CHECK: 1687.000000
    # CHECK: 1772.000000
    # CHECK: 1992.000000
    # CHECK: 2263.000000
    # CHECK: 2560.000000
    # CHECK: 2852.000000
    # CHECK: 3204.000000
    # CHECK: 3555.000000
    # CHECK: 3905.000000
    # CHECK: 4254.000000
    # CHECK: 4602.000000
    # CHECK: 4949.000000
    # CHECK: 5295.000000
    # CHECK: 5640.000000
    # CHECK: 3400.000000
    # CHECK: 2004.000000
    # CHECK: 1821.000000
    # CHECK: 1916.000000
    # CHECK: 2156.000000
    # CHECK: 2451.000000
    # CHECK: 2774.000000
    # CHECK: 3092.000000
    # CHECK: 3474.000000
    # CHECK: 3855.000000
    # CHECK: 4235.000000
    # CHECK: 4614.000000
    # CHECK: 4992.000000
    # CHECK: 5369.000000
    # CHECK: 5745.000000
    # CHECK: 6120.000000
    # CHECK: 3640.000000
    # CHECK: 2148.000000
    # CHECK: 1955.000000
    # CHECK: 2060.000000
    # CHECK: 2320.000000
    # CHECK: 2639.000000
    # CHECK: 2988.000000
    # CHECK: 3332.000000
    # CHECK: 3744.000000
    # CHECK: 4155.000000
    # CHECK: 4565.000000
    # CHECK: 4974.000000
    # CHECK: 5382.000000
    # CHECK: 5789.000000
    # CHECK: 6195.000000
    # CHECK: 6600.000000
    # CHECK: 3880.000000
    # CHECK: 2292.000000
    # CHECK: 2089.000000
    # CHECK: 2204.000000
    # CHECK: 2484.000000
    # CHECK: 2827.000000
    # CHECK: 3202.000000
    # CHECK: 3572.000000
    # CHECK: 4014.000000
    # CHECK: 4455.000000
    # CHECK: 4895.000000
    # CHECK: 5334.000000
    # CHECK: 5772.000000
    # CHECK: 6209.000000
    # CHECK: 6645.000000
    # CHECK: 7080.000000
    # CHECK: 4120.000000
    # CHECK: 2436.000000
    # CHECK: 2223.000000
    # CHECK: 2348.000000
    # CHECK: 2648.000000
    # CHECK: 3015.000000
    # CHECK: 3416.000000
    # CHECK: 3812.000000
    # CHECK: 4284.000000
    # CHECK: 4755.000000
    # CHECK: 5225.000000
    # CHECK: 5694.000000
    # CHECK: 6162.000000
    # CHECK: 6629.000000
    # CHECK: 7095.000000
    # CHECK: 7560.000000
    # CHECK: 4360.000000
    # CHECK: 2580.000000
    # CHECK: 2357.000000
    # CHECK: 2492.000000
    # CHECK: 2812.000000
    # CHECK: 3203.000000
    # CHECK: 3630.000000
    # CHECK: 4052.000000
    # CHECK: 4554.000000
    # CHECK: 5055.000000
    # CHECK: 5555.000000
    # CHECK: 6054.000000
    # CHECK: 6552.000000
    # CHECK: 7049.000000
    # CHECK: 7545.000000
    # CHECK: 8040.000000
    # CHECK: 4600.000000
    # CHECK: 2724.000000
    # CHECK: 2491.000000
    # CHECK: 2636.000000
    # CHECK: 2976.000000
    # CHECK: 3391.000000
    # CHECK: 3844.000000
    # CHECK: 4292.000000
    # CHECK: 4824.000000
    # CHECK: 5355.000000
    # CHECK: 5885.000000
    # CHECK: 6414.000000
    # CHECK: 6942.000000
    # CHECK: 7469.000000
    # CHECK: 7995.000000
    # CHECK: 8520.000000
    # CHECK: 4840.000000
    # CHECK: 2868.000000
    # CHECK: 2625.000000
    # CHECK: 2780.000000
    # CHECK: 3140.000000
    # CHECK: 3579.000000
    # CHECK: 4058.000000
    # CHECK: 4532.000000
    # CHECK: 5094.000000
    # CHECK: 5655.000000
    # CHECK: 6215.000000
    # CHECK: 6774.000000
    # CHECK: 7332.000000
    # CHECK: 7889.000000
    # CHECK: 8445.000000
    # CHECK: 9000.000000
    print_matrix(c_matrix)


fn main():
    test_amx_matmul()
