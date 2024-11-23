# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: AMD-GPU
# RUN: %mojo-no-debug %s | FileCheck %s

from builtin.io import _printf
from gpu import barrier, lane_id
from gpu.host import DeviceContext
from gpu.id import ThreadIdx
from gpu.memory import _GPUAddressSpace as AddressSpace
from layout import Layout, LayoutTensor
from layout._utils import (
    load_to_simd,
)
from layout.fillers import arange
from layout.layout_tensor import copy_dram_to_sram
from layout.tensor_core import TensorCore

from utils.index import Index, IndexList
from gpu import WARP_SIZE
from memory import UnsafePointer
from layout.tensor_builder import LayoutTensorBuild as tb
from testing import assert_almost_equal


fn test_load_a[
    dst_dtype: DType, dtype: DType, layout: Layout, inst_shape: IndexList[3]
](
    a: LayoutTensor[dtype, layout],
    a_lane: LayoutTensor[dtype, Layout(WARP_SIZE)],
):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, False]()
    var a_reg_tile = mma.load_a(a)
    # only storing 0th element for result
    a_lane[lane_id()] = a_reg_tile[0, 0]


fn test_load_b[
    dst_dtype: DType, dtype: DType, layout: Layout, inst_shape: IndexList[3]
](
    b: LayoutTensor[dtype, layout],
    b_lane: LayoutTensor[dtype, Layout(WARP_SIZE)],
):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, False]()
    var b_reg_tile = mma.load_b(b)
    # only storing 0th element for result
    b_lane[lane_id()] = b_reg_tile[0, 0]


fn test_load_c[
    dst_dtype: DType, dtype: DType, layout: Layout, inst_shape: IndexList[3]
](
    c: LayoutTensor[dst_dtype, layout],
    c_lane: LayoutTensor[dst_dtype, Layout.row_major(WARP_SIZE, 4)],
):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, False]()
    var c_reg_tile = mma.load_c(c)
    for i in range(4):
        c_lane[lane_id(), i] = c_reg_tile[0, i]


fn test_store_d[
    dst_dtype: DType, dtype: DType, layout: Layout, inst_shape: IndexList[3]
](d: LayoutTensor[dst_dtype, layout]):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, False]()
    var src = __type_of(mma).c_reg_tile_type.stack_allocation().fill(lane_id())
    mma.store_d(d, src)


fn test_mma_op[
    dst_dtype: DType,
    dtype: DType,
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout,
    inst_shape: IndexList[3],
](
    a: LayoutTensor[dtype, layout_a],
    b: LayoutTensor[dtype, layout_b],
    c: LayoutTensor[dst_dtype, layout_c],
    d: LayoutTensor[dst_dtype, layout_c],
):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, False]()
    var a_reg = mma.load_a(a)
    var b_reg = mma.load_b(b)
    var c_reg = mma.load_c(c)
    var d_reg = mma.mma_op(a_reg, b_reg, c_reg)
    mma.store_d(d, d_reg)


def test_load_and_mma_and_multiply_operands[
    dst_dtype: DType,
    dtype: DType,
    shape: IndexList[3],
    transpose_b: Bool = False,
](ctx: DeviceContext):
    alias M = shape[0]
    alias N = shape[1]
    alias K = shape[2]

    var a_host_ptr = UnsafePointer[Scalar[dtype]].alloc(M * K)
    var b_host_ptr = UnsafePointer[Scalar[dtype]].alloc(K * N)
    var c_host_ptr = UnsafePointer[Scalar[dst_dtype]].alloc(M * N)
    var d_host_ptr = UnsafePointer[Scalar[dst_dtype]].alloc(M * N)
    var d_ref_ptr = UnsafePointer[Scalar[dst_dtype]].alloc(M * N)

    var a_lane_host_ptr = UnsafePointer[Scalar[dtype]].alloc(WARP_SIZE)
    var b_lane_host_ptr = UnsafePointer[Scalar[dtype]].alloc(WARP_SIZE)
    var c_lane_host_ptr = UnsafePointer[Scalar[dst_dtype]].alloc(WARP_SIZE * 4)

    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[dtype](K * N)
    var c_device = ctx.enqueue_create_buffer[dst_dtype](M * N)
    var d_device = ctx.enqueue_create_buffer[dst_dtype](M * N)

    var d_device_mma = ctx.enqueue_create_buffer[dst_dtype](M * N)

    var a_lane_device = ctx.enqueue_create_buffer[dtype](WARP_SIZE)
    var b_lane_device = ctx.enqueue_create_buffer[dtype](WARP_SIZE)
    var c_lane_device = ctx.enqueue_create_buffer[dst_dtype](WARP_SIZE * 4)

    var a_host = tb[dtype]().row_major[M, K]().view(a_host_ptr)
    var a_dev = tb[dtype]().row_major[M, K]().view(a_device.ptr)

    var b_host = tb[dtype]().row_major[K, N]().view(b_host_ptr)
    var b_dev = tb[dtype]().row_major[K, N]().view(b_device.ptr)

    var c_host = tb[dst_dtype]().row_major[M, N]().view(c_host_ptr).fill(0)
    var c_dev = tb[dst_dtype]().row_major[M, N]().view(c_device.ptr)

    var d_host = tb[dst_dtype]().row_major[M, N]().view(d_host_ptr).fill(0)
    var d_ref = tb[dst_dtype]().row_major[M, N]().view(d_ref_ptr).fill(0)
    var d_dev = tb[dst_dtype]().row_major[M, N]().view(d_device.ptr)
    var d_dev_mma = tb[dst_dtype]().row_major[M, N]().view(d_device_mma.ptr)

    var a_lane_host = tb[dtype]().layout[WARP_SIZE]().view(a_lane_host_ptr)
    var a_lane_dev = tb[dtype]().layout[WARP_SIZE]().view(a_lane_device.ptr)
    var b_lane_host = tb[dtype]().layout[WARP_SIZE]().view(b_lane_host_ptr)
    var b_lane_dev = tb[dtype]().layout[WARP_SIZE]().view(b_lane_device.ptr)

    var c_lane_host = tb[dst_dtype]().row_major[WARP_SIZE, 4]().view(
        c_lane_host_ptr
    )
    var c_lane_dev = tb[dst_dtype]().row_major[WARP_SIZE, 4]().view(
        c_lane_device.ptr
    )

    arange(a_host)
    arange(b_host)
    arange(c_host)
    ctx.enqueue_copy_to_device(a_device, a_host_ptr)
    ctx.enqueue_copy_to_device(b_device, b_host_ptr)
    ctx.enqueue_copy_to_device(c_device, c_host_ptr)

    var func_load_a = ctx.compile_function[
        test_load_a[
            dst_dtype,
            dtype,
            a_dev.layout,
            shape,
        ],
    ]()

    ctx.enqueue_function(
        func_load_a, a_dev, a_lane_dev, grid_dim=(1, 1), block_dim=(WARP_SIZE)
    )

    var func_load_b = ctx.compile_function[
        test_load_b[
            dst_dtype,
            dtype,
            b_dev.layout,
            shape,
        ],
    ]()

    ctx.enqueue_function(
        func_load_b, b_dev, b_lane_dev, grid_dim=(1, 1), block_dim=(WARP_SIZE)
    )

    var func_load_c = ctx.compile_function[
        test_load_c[
            dst_dtype,
            dtype,
            c_dev.layout,
            shape,
        ],
    ]()

    ctx.enqueue_function(
        func_load_c, c_dev, c_lane_dev, grid_dim=(1, 1), block_dim=(WARP_SIZE)
    )

    var func_store_d = ctx.compile_function[
        test_store_d[
            dst_dtype,
            dtype,
            c_dev.layout,
            shape,
        ],
    ]()

    ctx.enqueue_function(
        func_store_d, d_dev, grid_dim=(1, 1), block_dim=(WARP_SIZE)
    )

    var func_mma_op = ctx.compile_function[
        test_mma_op[
            dst_dtype,
            dtype,
            a_dev.layout,
            b_dev.layout,
            c_dev.layout,
            shape,
        ],
    ]()

    ctx.enqueue_function(
        func_mma_op,
        a_dev,
        b_dev,
        c_dev,
        d_dev_mma,
        grid_dim=(1, 1),
        block_dim=(WARP_SIZE),
    )

    ctx.enqueue_copy_from_device(a_lane_host_ptr, a_lane_device)
    ctx.enqueue_copy_from_device(b_lane_host_ptr, b_lane_device)
    ctx.enqueue_copy_from_device(c_lane_host_ptr, c_lane_device)
    ctx.enqueue_copy_from_device(d_host_ptr, d_device)
    ctx.synchronize()

    print("== test_load_a")
    print(a_lane_host)

    print("== test_load_b")
    print(b_lane_host)

    print("== test_load_c")
    print(c_lane_host)

    print("== test_load_d")
    print(d_host)

    ctx.enqueue_copy_from_device(d_host_ptr, d_device_mma)
    ctx.synchronize()

    print("== test_mma")
    print(d_host)
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = d_device^
    _ = a_lane_device^
    _ = b_lane_device^
    _ = c_lane_device^
    _ = d_device_mma^

    _ = a_host_ptr
    _ = b_host_ptr
    _ = c_host_ptr
    _ = d_host_ptr
    _ = a_lane_host_ptr
    _ = b_lane_host_ptr
    _ = c_lane_host_ptr
    _ = d_ref_ptr


# CHECK-LABEL: test_load_and_mma_f32_f32_16x16x4
# CHECK-LABEL: test_load_a
# CHECK: 0.0 4.0 8.0 12.0 16.0 20.0 24.0 28.0 32.0 36.0 40.0 44.0 48.0 52.0 56.0 60.0 1.0 5.0 9.0 13.0 17.0 21.0 25.0 29.0 33.0 37.0 41.0 45.0 49.0 53.0 57.0 61.0 2.0 6.0 10.0 14.0 18.0 22.0 26.0 30.0 34.0 38.0 42.0 46.0 50.0 54.0 58.0 62.0 3.0 7.0 11.0 15.0 19.0 23.0 27.0 31.0 35.0 39.0 43.0 47.0 51.0 55.0 59.0 63.0
# CHECK-LABEL: test_load_b
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK-LABEL: test_load_c
# CHECK: 0.0 16.0 32.0 48.0
# CHECK: 1.0 17.0 33.0 49.0
# CHECK: 2.0 18.0 34.0 50.0
# CHECK: 3.0 19.0 35.0 51.0
# CHECK: 4.0 20.0 36.0 52.0
# CHECK: 5.0 21.0 37.0 53.0
# CHECK: 6.0 22.0 38.0 54.0
# CHECK: 7.0 23.0 39.0 55.0
# CHECK: 8.0 24.0 40.0 56.0
# CHECK: 9.0 25.0 41.0 57.0
# CHECK: 10.0 26.0 42.0 58.0
# CHECK: 11.0 27.0 43.0 59.0
# CHECK: 12.0 28.0 44.0 60.0
# CHECK: 13.0 29.0 45.0 61.0
# CHECK: 14.0 30.0 46.0 62.0
# CHECK: 15.0 31.0 47.0 63.0
# CHECK: 64.0 80.0 96.0 112.0
# CHECK: 65.0 81.0 97.0 113.0
# CHECK: 66.0 82.0 98.0 114.0
# CHECK: 67.0 83.0 99.0 115.0
# CHECK: 68.0 84.0 100.0 116.0
# CHECK: 69.0 85.0 101.0 117.0
# CHECK: 70.0 86.0 102.0 118.0
# CHECK: 71.0 87.0 103.0 119.0
# CHECK: 72.0 88.0 104.0 120.0
# CHECK: 73.0 89.0 105.0 121.0
# CHECK: 74.0 90.0 106.0 122.0
# CHECK: 75.0 91.0 107.0 123.0
# CHECK: 76.0 92.0 108.0 124.0
# CHECK: 77.0 93.0 109.0 125.0
# CHECK: 78.0 94.0 110.0 126.0
# CHECK: 79.0 95.0 111.0 127.0
# CHECK: 128.0 144.0 160.0 176.0
# CHECK: 129.0 145.0 161.0 177.0
# CHECK: 130.0 146.0 162.0 178.0
# CHECK: 131.0 147.0 163.0 179.0
# CHECK: 132.0 148.0 164.0 180.0
# CHECK: 133.0 149.0 165.0 181.0
# CHECK: 134.0 150.0 166.0 182.0
# CHECK: 135.0 151.0 167.0 183.0
# CHECK: 136.0 152.0 168.0 184.0
# CHECK: 137.0 153.0 169.0 185.0
# CHECK: 138.0 154.0 170.0 186.0
# CHECK: 139.0 155.0 171.0 187.0
# CHECK: 140.0 156.0 172.0 188.0
# CHECK: 141.0 157.0 173.0 189.0
# CHECK: 142.0 158.0 174.0 190.0
# CHECK: 143.0 159.0 175.0 191.0
# CHECK: 192.0 208.0 224.0 240.0
# CHECK: 193.0 209.0 225.0 241.0
# CHECK: 194.0 210.0 226.0 242.0
# CHECK: 195.0 211.0 227.0 243.0
# CHECK: 196.0 212.0 228.0 244.0
# CHECK: 197.0 213.0 229.0 245.0
# CHECK: 198.0 214.0 230.0 246.0
# CHECK: 199.0 215.0 231.0 247.0
# CHECK: 200.0 216.0 232.0 248.0
# CHECK: 201.0 217.0 233.0 249.0
# CHECK: 202.0 218.0 234.0 250.0
# CHECK: 203.0 219.0 235.0 251.0
# CHECK: 204.0 220.0 236.0 252.0
# CHECK: 205.0 221.0 237.0 253.0
# CHECK: 206.0 222.0 238.0 254.0
# CHECK: 207.0 223.0 239.0 255.0
# CHECK-LABEL: test_load_d
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK-LABEL: test_mma
# CHECK: 224.0 231.0 238.0 245.0 252.0 259.0 266.0 273.0 280.0 287.0 294.0 301.0 308.0 315.0 322.0 329.0
# CHECK: 624.0 647.0 670.0 693.0 716.0 739.0 762.0 785.0 808.0 831.0 854.0 877.0 900.0 923.0 946.0 969.0
# CHECK: 1024.0 1063.0 1102.0 1141.0 1180.0 1219.0 1258.0 1297.0 1336.0 1375.0 1414.0 1453.0 1492.0 1531.0 1570.0 1609.0
# CHECK: 1424.0 1479.0 1534.0 1589.0 1644.0 1699.0 1754.0 1809.0 1864.0 1919.0 1974.0 2029.0 2084.0 2139.0 2194.0 2249.0
# CHECK: 1824.0 1895.0 1966.0 2037.0 2108.0 2179.0 2250.0 2321.0 2392.0 2463.0 2534.0 2605.0 2676.0 2747.0 2818.0 2889.0
# CHECK: 2224.0 2311.0 2398.0 2485.0 2572.0 2659.0 2746.0 2833.0 2920.0 3007.0 3094.0 3181.0 3268.0 3355.0 3442.0 3529.0
# CHECK: 2624.0 2727.0 2830.0 2933.0 3036.0 3139.0 3242.0 3345.0 3448.0 3551.0 3654.0 3757.0 3860.0 3963.0 4066.0 4169.0
# CHECK: 3024.0 3143.0 3262.0 3381.0 3500.0 3619.0 3738.0 3857.0 3976.0 4095.0 4214.0 4333.0 4452.0 4571.0 4690.0 4809.0
# CHECK: 3424.0 3559.0 3694.0 3829.0 3964.0 4099.0 4234.0 4369.0 4504.0 4639.0 4774.0 4909.0 5044.0 5179.0 5314.0 5449.0
# CHECK: 3824.0 3975.0 4126.0 4277.0 4428.0 4579.0 4730.0 4881.0 5032.0 5183.0 5334.0 5485.0 5636.0 5787.0 5938.0 6089.0
# CHECK: 4224.0 4391.0 4558.0 4725.0 4892.0 5059.0 5226.0 5393.0 5560.0 5727.0 5894.0 6061.0 6228.0 6395.0 6562.0 6729.0
# CHECK: 4624.0 4807.0 4990.0 5173.0 5356.0 5539.0 5722.0 5905.0 6088.0 6271.0 6454.0 6637.0 6820.0 7003.0 7186.0 7369.0
# CHECK: 5024.0 5223.0 5422.0 5621.0 5820.0 6019.0 6218.0 6417.0 6616.0 6815.0 7014.0 7213.0 7412.0 7611.0 7810.0 8009.0
# CHECK: 5424.0 5639.0 5854.0 6069.0 6284.0 6499.0 6714.0 6929.0 7144.0 7359.0 7574.0 7789.0 8004.0 8219.0 8434.0 8649.0
# CHECK: 5824.0 6055.0 6286.0 6517.0 6748.0 6979.0 7210.0 7441.0 7672.0 7903.0 8134.0 8365.0 8596.0 8827.0 9058.0 9289.0
# CHECK: 6224.0 6471.0 6718.0 6965.0 7212.0 7459.0 7706.0 7953.0 8200.0 8447.0 8694.0 8941.0 9188.0 9435.0 9682.0 9929.0


def test_load_and_mma_f32_f32_16x16x4(ctx: DeviceContext):
    print("== test_load_and_mma_f32_f32_16x16x4")
    test_load_and_mma_and_multiply_operands[
        DType.float32, DType.float32, Index(16, 16, 4)
    ](ctx)


# CHECK-LABEL: test_load_and_mma_f32_bf16_16x16x16
# CHECK-LABEL: test_load_a
# CHECK: 0.0 16.0 32.0 48.0 64.0 80.0 96.0 112.0 128.0 144.0 160.0 176.0 192.0 208.0 224.0 240.0 1.0 17.0 33.0 49.0 65.0 81.0 97.0 113.0 129.0 145.0 161.0 177.0 193.0 209.0 225.0 241.0 2.0 18.0 34.0 50.0 66.0 82.0 98.0 114.0 130.0 146.0 162.0 178.0 194.0 210.0 226.0 242.0 3.0 19.0 35.0 51.0 67.0 83.0 99.0 115.0 131.0 147.0 163.0 179.0 195.0 211.0 227.0 243.0
# CHECK-LABEL: test_load_b
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK-LABEL: test_load_c
# CHECK: 0.0 16.0 32.0 48.0
# CHECK: 1.0 17.0 33.0 49.0
# CHECK: 2.0 18.0 34.0 50.0
# CHECK: 3.0 19.0 35.0 51.0
# CHECK: 4.0 20.0 36.0 52.0
# CHECK: 5.0 21.0 37.0 53.0
# CHECK: 6.0 22.0 38.0 54.0
# CHECK: 7.0 23.0 39.0 55.0
# CHECK: 8.0 24.0 40.0 56.0
# CHECK: 9.0 25.0 41.0 57.0
# CHECK: 10.0 26.0 42.0 58.0
# CHECK: 11.0 27.0 43.0 59.0
# CHECK: 12.0 28.0 44.0 60.0
# CHECK: 13.0 29.0 45.0 61.0
# CHECK: 14.0 30.0 46.0 62.0
# CHECK: 15.0 31.0 47.0 63.0
# CHECK: 64.0 80.0 96.0 112.0
# CHECK: 65.0 81.0 97.0 113.0
# CHECK: 66.0 82.0 98.0 114.0
# CHECK: 67.0 83.0 99.0 115.0
# CHECK: 68.0 84.0 100.0 116.0
# CHECK: 69.0 85.0 101.0 117.0
# CHECK: 70.0 86.0 102.0 118.0
# CHECK: 71.0 87.0 103.0 119.0
# CHECK: 72.0 88.0 104.0 120.0
# CHECK: 73.0 89.0 105.0 121.0
# CHECK: 74.0 90.0 106.0 122.0
# CHECK: 75.0 91.0 107.0 123.0
# CHECK: 76.0 92.0 108.0 124.0
# CHECK: 77.0 93.0 109.0 125.0
# CHECK: 78.0 94.0 110.0 126.0
# CHECK: 79.0 95.0 111.0 127.0
# CHECK: 128.0 144.0 160.0 176.0
# CHECK: 129.0 145.0 161.0 177.0
# CHECK: 130.0 146.0 162.0 178.0
# CHECK: 131.0 147.0 163.0 179.0
# CHECK: 132.0 148.0 164.0 180.0
# CHECK: 133.0 149.0 165.0 181.0
# CHECK: 134.0 150.0 166.0 182.0
# CHECK: 135.0 151.0 167.0 183.0
# CHECK: 136.0 152.0 168.0 184.0
# CHECK: 137.0 153.0 169.0 185.0
# CHECK: 138.0 154.0 170.0 186.0
# CHECK: 139.0 155.0 171.0 187.0
# CHECK: 140.0 156.0 172.0 188.0
# CHECK: 141.0 157.0 173.0 189.0
# CHECK: 142.0 158.0 174.0 190.0
# CHECK: 143.0 159.0 175.0 191.0
# CHECK: 192.0 208.0 224.0 240.0
# CHECK: 193.0 209.0 225.0 241.0
# CHECK: 194.0 210.0 226.0 242.0
# CHECK: 195.0 211.0 227.0 243.0
# CHECK: 196.0 212.0 228.0 244.0
# CHECK: 197.0 213.0 229.0 245.0
# CHECK: 198.0 214.0 230.0 246.0
# CHECK: 199.0 215.0 231.0 247.0
# CHECK: 200.0 216.0 232.0 248.0
# CHECK: 201.0 217.0 233.0 249.0
# CHECK: 202.0 218.0 234.0 250.0
# CHECK: 203.0 219.0 235.0 251.0
# CHECK: 204.0 220.0 236.0 252.0
# CHECK: 205.0 221.0 237.0 253.0
# CHECK: 206.0 222.0 238.0 254.0
# CHECK: 207.0 223.0 239.0 255.0
# CHECK-LABEL: test_load_d
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK-LABEL: test_mma
# CHECK: 19840.0 19961.0 20082.0 20203.0 20324.0 20445.0 20566.0 20687.0 20808.0 20929.0 21050.0 21171.0 21292.0 21413.0 21534.0 21655.0
# CHECK: 50576.0 50953.0 51330.0 51707.0 52084.0 52461.0 52838.0 53215.0 53592.0 53969.0 54346.0 54723.0 55100.0 55477.0 55854.0 56231.0
# CHECK: 81312.0 81945.0 82578.0 83211.0 83844.0 84477.0 85110.0 85743.0 86376.0 87009.0 87642.0 88275.0 88908.0 89541.0 90174.0 90807.0
# CHECK: 112048.0 112937.0 113826.0 114715.0 115604.0 116493.0 117382.0 118271.0 119160.0 120049.0 120938.0 121827.0 122716.0 123605.0 124494.0 125383.0
# CHECK: 142784.0 143929.0 145074.0 146219.0 147364.0 148509.0 149654.0 150799.0 151944.0 153089.0 154234.0 155379.0 156524.0 157669.0 158814.0 159959.0
# CHECK: 173520.0 174921.0 176322.0 177723.0 179124.0 180525.0 181926.0 183327.0 184728.0 186129.0 187530.0 188931.0 190332.0 191733.0 193134.0 194535.0
# CHECK: 204256.0 205913.0 207570.0 209227.0 210884.0 212541.0 214198.0 215855.0 217512.0 219169.0 220826.0 222483.0 224140.0 225797.0 227454.0 229111.0
# CHECK: 234992.0 236905.0 238818.0 240731.0 242644.0 244557.0 246470.0 248383.0 250296.0 252209.0 254122.0 256035.0 257948.0 259861.0 261774.0 263687.0
# CHECK: 265728.0 267897.0 270066.0 272235.0 274404.0 276573.0 278742.0 280911.0 283080.0 285249.0 287418.0 289587.0 291756.0 293925.0 296094.0 298263.0
# CHECK: 296464.0 298889.0 301314.0 303739.0 306164.0 308589.0 311014.0 313439.0 315864.0 318289.0 320714.0 323139.0 325564.0 327989.0 330414.0 332839.0
# CHECK: 327200.0 329881.0 332562.0 335243.0 337924.0 340605.0 343286.0 345967.0 348648.0 351329.0 354010.0 356691.0 359372.0 362053.0 364734.0 367415.0
# CHECK: 357936.0 360873.0 363810.0 366747.0 369684.0 372621.0 375558.0 378495.0 381432.0 384369.0 387306.0 390243.0 393180.0 396117.0 399054.0 401991.0
# CHECK: 388672.0 391865.0 395058.0 398251.0 401444.0 404637.0 407830.0 411023.0 414216.0 417409.0 420602.0 423795.0 426988.0 430181.0 433374.0 436567.0
# CHECK: 419408.0 422857.0 426306.0 429755.0 433204.0 436653.0 440102.0 443551.0 447000.0 450449.0 453898.0 457347.0 460796.0 464245.0 467694.0 471143.0
# CHECK: 450144.0 453849.0 457554.0 461259.0 464964.0 468669.0 472374.0 476079.0 479784.0 483489.0 487194.0 490899.0 494604.0 498309.0 502014.0 505719.0
# CHECK: 480880.0 484841.0 488802.0 492763.0 496724.0 500685.0 504646.0 508607.0 512568.0 516529.0 520490.0 524451.0 528412.0 532373.0 536334.0 540295.0
def test_load_and_mma_f32_bf16_16x16x16(ctx: DeviceContext):
    print("== test_load_and_mma_f32_bf16_16x16x16")
    test_load_and_mma_and_multiply_operands[
        DType.float32, DType.bfloat16, Index(16, 16, 16)
    ](ctx)


# CHECK-LABEL: test_load_and_mma_f32_f16_16x16x16
# CHECK-LABEL: test_load_a
# CHECK: 0.0 16.0 32.0 48.0 64.0 80.0 96.0 112.0 128.0 144.0 160.0 176.0 192.0 208.0 224.0 240.0 1.0 17.0 33.0 49.0 65.0 81.0 97.0 113.0 129.0 145.0 161.0 177.0 193.0 209.0 225.0 241.0 2.0 18.0 34.0 50.0 66.0 82.0 98.0 114.0 130.0 146.0 162.0 178.0 194.0 210.0 226.0 242.0 3.0 19.0 35.0 51.0 67.0 83.0 99.0 115.0 131.0 147.0 163.0 179.0 195.0 211.0 227.0 243.0
# CHECK-LABEL: test_load_b
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK-LABEL: test_load_c
# CHECK: 0.0 16.0 32.0 48.0
# CHECK: 1.0 17.0 33.0 49.0
# CHECK: 2.0 18.0 34.0 50.0
# CHECK: 3.0 19.0 35.0 51.0
# CHECK: 4.0 20.0 36.0 52.0
# CHECK: 5.0 21.0 37.0 53.0
# CHECK: 6.0 22.0 38.0 54.0
# CHECK: 7.0 23.0 39.0 55.0
# CHECK: 8.0 24.0 40.0 56.0
# CHECK: 9.0 25.0 41.0 57.0
# CHECK: 10.0 26.0 42.0 58.0
# CHECK: 11.0 27.0 43.0 59.0
# CHECK: 12.0 28.0 44.0 60.0
# CHECK: 13.0 29.0 45.0 61.0
# CHECK: 14.0 30.0 46.0 62.0
# CHECK: 15.0 31.0 47.0 63.0
# CHECK: 64.0 80.0 96.0 112.0
# CHECK: 65.0 81.0 97.0 113.0
# CHECK: 66.0 82.0 98.0 114.0
# CHECK: 67.0 83.0 99.0 115.0
# CHECK: 68.0 84.0 100.0 116.0
# CHECK: 69.0 85.0 101.0 117.0
# CHECK: 70.0 86.0 102.0 118.0
# CHECK: 71.0 87.0 103.0 119.0
# CHECK: 72.0 88.0 104.0 120.0
# CHECK: 73.0 89.0 105.0 121.0
# CHECK: 74.0 90.0 106.0 122.0
# CHECK: 75.0 91.0 107.0 123.0
# CHECK: 76.0 92.0 108.0 124.0
# CHECK: 77.0 93.0 109.0 125.0
# CHECK: 78.0 94.0 110.0 126.0
# CHECK: 79.0 95.0 111.0 127.0
# CHECK: 128.0 144.0 160.0 176.0
# CHECK: 129.0 145.0 161.0 177.0
# CHECK: 130.0 146.0 162.0 178.0
# CHECK: 131.0 147.0 163.0 179.0
# CHECK: 132.0 148.0 164.0 180.0
# CHECK: 133.0 149.0 165.0 181.0
# CHECK: 134.0 150.0 166.0 182.0
# CHECK: 135.0 151.0 167.0 183.0
# CHECK: 136.0 152.0 168.0 184.0
# CHECK: 137.0 153.0 169.0 185.0
# CHECK: 138.0 154.0 170.0 186.0
# CHECK: 139.0 155.0 171.0 187.0
# CHECK: 140.0 156.0 172.0 188.0
# CHECK: 141.0 157.0 173.0 189.0
# CHECK: 142.0 158.0 174.0 190.0
# CHECK: 143.0 159.0 175.0 191.0
# CHECK: 192.0 208.0 224.0 240.0
# CHECK: 193.0 209.0 225.0 241.0
# CHECK: 194.0 210.0 226.0 242.0
# CHECK: 195.0 211.0 227.0 243.0
# CHECK: 196.0 212.0 228.0 244.0
# CHECK: 197.0 213.0 229.0 245.0
# CHECK: 198.0 214.0 230.0 246.0
# CHECK: 199.0 215.0 231.0 247.0
# CHECK: 200.0 216.0 232.0 248.0
# CHECK: 201.0 217.0 233.0 249.0
# CHECK: 202.0 218.0 234.0 250.0
# CHECK: 203.0 219.0 235.0 251.0
# CHECK: 204.0 220.0 236.0 252.0
# CHECK: 205.0 221.0 237.0 253.0
# CHECK: 206.0 222.0 238.0 254.0
# CHECK: 207.0 223.0 239.0 255.0
# CHECK-LABEL: test_load_d
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
# CHECK-LABEL: test_mma
# CHECK: 19840.0 19961.0 20082.0 20203.0 20324.0 20445.0 20566.0 20687.0 20808.0 20929.0 21050.0 21171.0 21292.0 21413.0 21534.0 21655.0
# CHECK: 50576.0 50953.0 51330.0 51707.0 52084.0 52461.0 52838.0 53215.0 53592.0 53969.0 54346.0 54723.0 55100.0 55477.0 55854.0 56231.0
# CHECK: 81312.0 81945.0 82578.0 83211.0 83844.0 84477.0 85110.0 85743.0 86376.0 87009.0 87642.0 88275.0 88908.0 89541.0 90174.0 90807.0
# CHECK: 112048.0 112937.0 113826.0 114715.0 115604.0 116493.0 117382.0 118271.0 119160.0 120049.0 120938.0 121827.0 122716.0 123605.0 124494.0 125383.0
# CHECK: 142784.0 143929.0 145074.0 146219.0 147364.0 148509.0 149654.0 150799.0 151944.0 153089.0 154234.0 155379.0 156524.0 157669.0 158814.0 159959.0
# CHECK: 173520.0 174921.0 176322.0 177723.0 179124.0 180525.0 181926.0 183327.0 184728.0 186129.0 187530.0 188931.0 190332.0 191733.0 193134.0 194535.0
# CHECK: 204256.0 205913.0 207570.0 209227.0 210884.0 212541.0 214198.0 215855.0 217512.0 219169.0 220826.0 222483.0 224140.0 225797.0 227454.0 229111.0
# CHECK: 234992.0 236905.0 238818.0 240731.0 242644.0 244557.0 246470.0 248383.0 250296.0 252209.0 254122.0 256035.0 257948.0 259861.0 261774.0 263687.0
# CHECK: 265728.0 267897.0 270066.0 272235.0 274404.0 276573.0 278742.0 280911.0 283080.0 285249.0 287418.0 289587.0 291756.0 293925.0 296094.0 298263.0
# CHECK: 296464.0 298889.0 301314.0 303739.0 306164.0 308589.0 311014.0 313439.0 315864.0 318289.0 320714.0 323139.0 325564.0 327989.0 330414.0 332839.0
# CHECK: 327200.0 329881.0 332562.0 335243.0 337924.0 340605.0 343286.0 345967.0 348648.0 351329.0 354010.0 356691.0 359372.0 362053.0 364734.0 367415.0
# CHECK: 357936.0 360873.0 363810.0 366747.0 369684.0 372621.0 375558.0 378495.0 381432.0 384369.0 387306.0 390243.0 393180.0 396117.0 399054.0 401991.0
# CHECK: 388672.0 391865.0 395058.0 398251.0 401444.0 404637.0 407830.0 411023.0 414216.0 417409.0 420602.0 423795.0 426988.0 430181.0 433374.0 436567.0
# CHECK: 419408.0 422857.0 426306.0 429755.0 433204.0 436653.0 440102.0 443551.0 447000.0 450449.0 453898.0 457347.0 460796.0 464245.0 467694.0 471143.0
# CHECK: 450144.0 453849.0 457554.0 461259.0 464964.0 468669.0 472374.0 476079.0 479784.0 483489.0 487194.0 490899.0 494604.0 498309.0 502014.0 505719.0
# CHECK: 480880.0 484841.0 488802.0 492763.0 496724.0 500685.0 504646.0 508607.0 512568.0 516529.0 520490.0 524451.0 528412.0 532373.0 536334.0 540295.0


def test_load_and_mma_f32_f16_16x16x16(ctx: DeviceContext):
    print("== test_load_and_mma_f32_f16_16x16x16")
    test_load_and_mma_and_multiply_operands[
        DType.float32, DType.float16, Index(16, 16, 16)
    ](ctx)


def main():
    with DeviceContext() as ctx:
        test_load_and_mma_f32_f32_16x16x4(ctx)
        test_load_and_mma_f32_bf16_16x16x16(ctx)
        test_load_and_mma_f32_f16_16x16x16(ctx)
