# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file only tests the Apple AMX matmul functionality which is defined as a
# A^T.B where A and B are 16x16 Float32 matrices.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: apple-silicone
# RUN: %mojo -debug-level full %s | FileCheck %s

from sys.info import is_apple_silicone, sizeof

from AppleAMX import amx_detail
from memory.buffer import NDBuffer

from utils.index import StaticIntTuple
from utils.list import DimList


fn fill_a(
    buf: NDBuffer[
        DType.float32,
        2,
        DimList(16, 16),
    ]
):
    # Fills the A matrix with the following values row + 2*col
    var rows = 16
    var cols = 16
    for i in range(rows):
        for j in range(cols):
            var val = Float32(i + 2 * j)
            buf[StaticIntTuple[2](i, j)] = val.value


fn fill_b(
    buf: NDBuffer[
        DType.float32,
        2,
        DimList(16, 16),
    ]
):
    # Fills the A matrix with the following values row/(col + 1) + col
    var rows = 16
    var cols = 16
    for i in range(rows):
        for j in range(cols):
            var val = Float32(i // (j + 1) + j)
            buf[StaticIntTuple[2](i, j)] = val.value


fn clear_c(
    buf: NDBuffer[
        DType.float32,
        2,
        DimList(16, 16),
    ]
):
    buf.zero()


fn print_matrix(
    buf: NDBuffer[
        DType.float32,
        2,
        DimList(16, 16),
    ]
):
    # Fills the A matrix with the following values row/(col + 1) + col + 3
    var rows = 16
    var cols = 16
    for i in range(rows):
        for j in range(cols):
            print(buf[i, j])


# CHECK-LABEL: test_amx_matmul
fn test_amx_matmul():
    if is_apple_silicone():
        print("== test_amx_matmul")

    var a_matrix = NDBuffer[
        DType.float32,
        2,
        DimList(16, 16),
    ].stack_allocation()
    var b_matrix = NDBuffer[
        DType.float32,
        2,
        DimList(16, 16),
    ].stack_allocation()
    var c_matrix = NDBuffer[
        DType.float32,
        2,
        DimList(16, 16),
    ].stack_allocation()

    fill_a(a_matrix)
    fill_b(b_matrix)
    clear_c(c_matrix)

    amx_detail.dot_at_b(c_matrix, a_matrix, b_matrix)

    # CHECK: 1240.0
    # CHECK: 708.0
    # CHECK: 615.0
    # CHECK: 620.0
    # CHECK: 680.0
    # CHECK: 759.0
    # CHECK: 848.0
    # CHECK: 932.0
    # CHECK: 1044.0
    # CHECK: 1155.0
    # CHECK: 1265.0
    # CHECK: 1374.0
    # CHECK: 1482.0
    # CHECK: 1589.0
    # CHECK: 1695.0
    # CHECK: 1800.0
    # CHECK: 1480.0
    # CHECK: 852.0
    # CHECK: 749.0
    # CHECK: 764.0
    # CHECK: 844.0
    # CHECK: 947.0
    # CHECK: 1062.0
    # CHECK: 1172.0
    # CHECK: 1314.0
    # CHECK: 1455.0
    # CHECK: 1595.0
    # CHECK: 1734.0
    # CHECK: 1872.0
    # CHECK: 2009.0
    # CHECK: 2145.0
    # CHECK: 2280.0
    # CHECK: 1720.0
    # CHECK: 996.0
    # CHECK: 883.0
    # CHECK: 908.0
    # CHECK: 1008.0
    # CHECK: 1135.0
    # CHECK: 1276.0
    # CHECK: 1412.0
    # CHECK: 1584.0
    # CHECK: 1755.0
    # CHECK: 1925.0
    # CHECK: 2094.0
    # CHECK: 2262.0
    # CHECK: 2429.0
    # CHECK: 2595.0
    # CHECK: 2760.0
    # CHECK: 1960.0
    # CHECK: 1140.0
    # CHECK: 1017.0
    # CHECK: 1052.0
    # CHECK: 1172.0
    # CHECK: 1323.0
    # CHECK: 1490.0
    # CHECK: 1652.0
    # CHECK: 1854.0
    # CHECK: 2055.0
    # CHECK: 2255.0
    # CHECK: 2454.0
    # CHECK: 2652.0
    # CHECK: 2849.0
    # CHECK: 3045.0
    # CHECK: 3240.0
    # CHECK: 2200.0
    # CHECK: 1284.0
    # CHECK: 1151.0
    # CHECK: 1196.0
    # CHECK: 1336.0
    # CHECK: 1511.0
    # CHECK: 1704.0
    # CHECK: 1892.0
    # CHECK: 2124.0
    # CHECK: 2355.0
    # CHECK: 2585.0
    # CHECK: 2814.0
    # CHECK: 3042.0
    # CHECK: 3269.0
    # CHECK: 3495.0
    # CHECK: 3720.0
    # CHECK: 2440.0
    # CHECK: 1428.0
    # CHECK: 1285.0
    # CHECK: 1340.0
    # CHECK: 1500.0
    # CHECK: 1699.0
    # CHECK: 1918.0
    # CHECK: 2132.0
    # CHECK: 2394.0
    # CHECK: 2655.0
    # CHECK: 2915.0
    # CHECK: 3174.0
    # CHECK: 3432.0
    # CHECK: 3689.0
    # CHECK: 3945.0
    # CHECK: 4200.0
    # CHECK: 2680.0
    # CHECK: 1572.0
    # CHECK: 1419.0
    # CHECK: 1484.0
    # CHECK: 1664.0
    # CHECK: 1887.0
    # CHECK: 2132.0
    # CHECK: 2372.0
    # CHECK: 2664.0
    # CHECK: 2955.0
    # CHECK: 3245.0
    # CHECK: 3534.0
    # CHECK: 3822.0
    # CHECK: 4109.0
    # CHECK: 4395.0
    # CHECK: 4680.0
    # CHECK: 2920.0
    # CHECK: 1716.0
    # CHECK: 1553.0
    # CHECK: 1628.0
    # CHECK: 1828.0
    # CHECK: 2075.0
    # CHECK: 2346.0
    # CHECK: 2612.0
    # CHECK: 2934.0
    # CHECK: 3255.0
    # CHECK: 3575.0
    # CHECK: 3894.0
    # CHECK: 4212.0
    # CHECK: 4529.0
    # CHECK: 4845.0
    # CHECK: 5160.0
    # CHECK: 3160.0
    # CHECK: 1860.0
    # CHECK: 1687.0
    # CHECK: 1772.0
    # CHECK: 1992.0
    # CHECK: 2263.0
    # CHECK: 2560.0
    # CHECK: 2852.0
    # CHECK: 3204.0
    # CHECK: 3555.0
    # CHECK: 3905.0
    # CHECK: 4254.0
    # CHECK: 4602.0
    # CHECK: 4949.0
    # CHECK: 5295.0
    # CHECK: 5640.0
    # CHECK: 3400.0
    # CHECK: 2004.0
    # CHECK: 1821.0
    # CHECK: 1916.0
    # CHECK: 2156.0
    # CHECK: 2451.0
    # CHECK: 2774.0
    # CHECK: 3092.0
    # CHECK: 3474.0
    # CHECK: 3855.0
    # CHECK: 4235.0
    # CHECK: 4614.0
    # CHECK: 4992.0
    # CHECK: 5369.0
    # CHECK: 5745.0
    # CHECK: 6120.0
    # CHECK: 3640.0
    # CHECK: 2148.0
    # CHECK: 1955.0
    # CHECK: 2060.0
    # CHECK: 2320.0
    # CHECK: 2639.0
    # CHECK: 2988.0
    # CHECK: 3332.0
    # CHECK: 3744.0
    # CHECK: 4155.0
    # CHECK: 4565.0
    # CHECK: 4974.0
    # CHECK: 5382.0
    # CHECK: 5789.0
    # CHECK: 6195.0
    # CHECK: 6600.0
    # CHECK: 3880.0
    # CHECK: 2292.0
    # CHECK: 2089.0
    # CHECK: 2204.0
    # CHECK: 2484.0
    # CHECK: 2827.0
    # CHECK: 3202.0
    # CHECK: 3572.0
    # CHECK: 4014.0
    # CHECK: 4455.0
    # CHECK: 4895.0
    # CHECK: 5334.0
    # CHECK: 5772.0
    # CHECK: 6209.0
    # CHECK: 6645.0
    # CHECK: 7080.0
    # CHECK: 4120.0
    # CHECK: 2436.0
    # CHECK: 2223.0
    # CHECK: 2348.0
    # CHECK: 2648.0
    # CHECK: 3015.0
    # CHECK: 3416.0
    # CHECK: 3812.0
    # CHECK: 4284.0
    # CHECK: 4755.0
    # CHECK: 5225.0
    # CHECK: 5694.0
    # CHECK: 6162.0
    # CHECK: 6629.0
    # CHECK: 7095.0
    # CHECK: 7560.0
    # CHECK: 4360.0
    # CHECK: 2580.0
    # CHECK: 2357.0
    # CHECK: 2492.0
    # CHECK: 2812.0
    # CHECK: 3203.0
    # CHECK: 3630.0
    # CHECK: 4052.0
    # CHECK: 4554.0
    # CHECK: 5055.0
    # CHECK: 5555.0
    # CHECK: 6054.0
    # CHECK: 6552.0
    # CHECK: 7049.0
    # CHECK: 7545.0
    # CHECK: 8040.0
    # CHECK: 4600.0
    # CHECK: 2724.0
    # CHECK: 2491.0
    # CHECK: 2636.0
    # CHECK: 2976.0
    # CHECK: 3391.0
    # CHECK: 3844.0
    # CHECK: 4292.0
    # CHECK: 4824.0
    # CHECK: 5355.0
    # CHECK: 5885.0
    # CHECK: 6414.0
    # CHECK: 6942.0
    # CHECK: 7469.0
    # CHECK: 7995.0
    # CHECK: 8520.0
    # CHECK: 4840.0
    # CHECK: 2868.0
    # CHECK: 2625.0
    # CHECK: 2780.0
    # CHECK: 3140.0
    # CHECK: 3579.0
    # CHECK: 4058.0
    # CHECK: 4532.0
    # CHECK: 5094.0
    # CHECK: 5655.0
    # CHECK: 6215.0
    # CHECK: 6774.0
    # CHECK: 7332.0
    # CHECK: 7889.0
    # CHECK: 8445.0
    # CHECK: 9000.0
    print_matrix(c_matrix)


fn main():
    @parameter
    if is_apple_silicone():
        test_amx_matmul()
