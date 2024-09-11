# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from builtin.io import _printf

from layout import LayoutTensor, Layout
from layout.layout_tensor import copy_dram_to_sram
from layout.tensor_core import TensorCore
from layout._utils import ManagedLayoutTensor, gpu_free, gpu_managed_alloc
from layout.fillers import arange
from gpu.host import DeviceContext
from gpu.id import ThreadIdx
from gpu import barrier, lane_id
from gpu.memory import _GPUAddressSpace as AddressSpace
from utils.index import StaticIntTuple, Index


fn mma_load_and_print_operands_kernel[
    dst_dtype: DType,
    dtype: DType,
    lhs_layout: Layout,
    rhs_layout: Layout,
    inst_shape: StaticIntTuple[3],
    transpose_b: Bool = False,
](lhs: LayoutTensor[dtype, lhs_layout], rhs: LayoutTensor[dtype, rhs_layout]):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, transpose_b]()
    var a_frags = mma.load_a(lhs).cast[DType.float64]()
    var b_frags = mma.load_b(rhs).cast[DType.float64]()

    @parameter
    if a_frags.size == 4 and b_frags.size == 2:
        _printf["thread %u a_vals=[%g %g %g %g], b_vals=[%g %g]\n"](
            ThreadIdx.x(),
            a_frags[0],
            a_frags[1],
            a_frags[2],
            a_frags[3],
            b_frags[0],
            b_frags[1],
        )
    elif a_frags.size == 2 and b_frags.size == 1:
        _printf["thread %u a_vals=[%g %g], b_vals=[%g]\n"](
            ThreadIdx.x(),
            a_frags[0],
            a_frags[1],
            b_frags[0],
        )


fn mma_write_operand_kernel[
    dst_dtype: DType,
    dtype: DType,
    layout: Layout,
    inst_shape: StaticIntTuple[3],
](out: LayoutTensor[dst_dtype, layout]):
    var mma = TensorCore[dst_dtype, dtype, inst_shape]()
    var thread_val = mma.c_reg_type(ThreadIdx.x())
    mma.store_d(out, thread_val)


def test_load_operands[
    dst_dtype: DType,
    dtype: DType,
    shape: StaticIntTuple[3],
    transpose_b: Bool = False,
](ctx: DeviceContext):
    alias M = shape[0]
    alias N = shape[1]
    alias K = shape[2]

    var lhs = ManagedLayoutTensor[
        dtype, Layout.row_major(M, K), gpu_managed_alloc, gpu_free
    ]()
    arange(lhs.tensor)
    var rhs = ManagedLayoutTensor[
        dtype, Layout.row_major(K, N), gpu_managed_alloc, gpu_free
    ]()
    arange(rhs.tensor)
    alias mma_load_and_print_kernel_fn = mma_load_and_print_operands_kernel[
        dst_dtype, dtype, lhs.layout, rhs.layout, shape, transpose_b
    ]
    var func = ctx.compile_function[mma_load_and_print_kernel_fn]()
    ctx.enqueue_function(
        func, lhs.tensor, rhs.tensor, grid_dim=(1, 1), block_dim=(32)
    )
    ctx.synchronize()


def test_write_res_operand[
    dst_dtype: DType, dtype: DType, shape: StaticIntTuple[3]
](ctx: DeviceContext):
    alias M = shape[0]
    alias N = shape[1]
    alias K = shape[2]

    var dst = ManagedLayoutTensor[
        dst_dtype, Layout.row_major(M, N), gpu_managed_alloc, gpu_free
    ]()
    _ = dst.tensor.fill(0)

    alias mma_load_and_print_kernel_fn = mma_write_operand_kernel[
        dst_dtype, dtype, dst.layout, shape
    ]
    var func = ctx.compile_function[mma_load_and_print_kernel_fn]()
    ctx.enqueue_function(func, dst.tensor, grid_dim=(1, 1), block_dim=(32))
    ctx.synchronize()
    print(dst.tensor)


# CHECK-LABEL: test_load_f32_f32_16x8x8
# CHECK-DAG: thread 0 a_vals=[0 64 4 68], b_vals=[0 32]
# CHECK-DAG: thread 1 a_vals=[1 65 5 69], b_vals=[8 40]
# CHECK-DAG: thread 2 a_vals=[2 66 6 70], b_vals=[16 48]
# CHECK-DAG: thread 3 a_vals=[3 67 7 71], b_vals=[24 56]
# CHECK-DAG: thread 4 a_vals=[8 72 12 76], b_vals=[1 33]
# CHECK-DAG: thread 5 a_vals=[9 73 13 77], b_vals=[9 41]
# CHECK-DAG: thread 6 a_vals=[10 74 14 78], b_vals=[17 49]
# CHECK-DAG: thread 7 a_vals=[11 75 15 79], b_vals=[25 57]
# CHECK-DAG: thread 8 a_vals=[16 80 20 84], b_vals=[2 34]
# CHECK-DAG: thread 9 a_vals=[17 81 21 85], b_vals=[10 42]
# CHECK-DAG: thread 10 a_vals=[18 82 22 86], b_vals=[18 50]
# CHECK-DAG: thread 11 a_vals=[19 83 23 87], b_vals=[26 58]
# CHECK-DAG: thread 12 a_vals=[24 88 28 92], b_vals=[3 35]
# CHECK-DAG: thread 13 a_vals=[25 89 29 93], b_vals=[11 43]
# CHECK-DAG: thread 14 a_vals=[26 90 30 94], b_vals=[19 51]
# CHECK-DAG: thread 15 a_vals=[27 91 31 95], b_vals=[27 59]
# CHECK-DAG: thread 16 a_vals=[32 96 36 100], b_vals=[4 36]
# CHECK-DAG: thread 17 a_vals=[33 97 37 101], b_vals=[12 44]
# CHECK-DAG: thread 18 a_vals=[34 98 38 102], b_vals=[20 52]
# CHECK-DAG: thread 19 a_vals=[35 99 39 103], b_vals=[28 60]
# CHECK-DAG: thread 20 a_vals=[40 104 44 108], b_vals=[5 37]
# CHECK-DAG: thread 21 a_vals=[41 105 45 109], b_vals=[13 45]
# CHECK-DAG: thread 22 a_vals=[42 106 46 110], b_vals=[21 53]
# CHECK-DAG: thread 23 a_vals=[43 107 47 111], b_vals=[29 61]
# CHECK-DAG: thread 24 a_vals=[48 112 52 116], b_vals=[6 38]
# CHECK-DAG: thread 25 a_vals=[49 113 53 117], b_vals=[14 46]
# CHECK-DAG: thread 26 a_vals=[50 114 54 118], b_vals=[22 54]
# CHECK-DAG: thread 27 a_vals=[51 115 55 119], b_vals=[30 62]
# CHECK-DAG: thread 28 a_vals=[56 120 60 124], b_vals=[7 39]
# CHECK-DAG: thread 29 a_vals=[57 121 61 125], b_vals=[15 47]
# CHECK-DAG: thread 30 a_vals=[58 122 62 126], b_vals=[23 55]
# CHECK-DAG: thread 31 a_vals=[59 123 63 127], b_vals=[31 63]
def test_load_f32_f32_16x8x8(ctx: DeviceContext):
    print("== test_load_f32_f32_16x8x8")
    test_load_operands[DType.float32, DType.float32, Index(16, 8, 8)](ctx)


# CHECK-LABEL: test_load_f32_f32_16x8x8_b_transpose
# CHECK-DAG: thread 0 a_vals=[0 64 4 68], b_vals=[0 4]
# CHECK-DAG: thread 1 a_vals=[1 65 5 69], b_vals=[1 5]
# CHECK-DAG: thread 2 a_vals=[2 66 6 70], b_vals=[2 6]
# CHECK-DAG: thread 3 a_vals=[3 67 7 71], b_vals=[3 7]
# CHECK-DAG: thread 4 a_vals=[8 72 12 76], b_vals=[8 12]
# CHECK-DAG: thread 5 a_vals=[9 73 13 77], b_vals=[9 13]
# CHECK-DAG: thread 6 a_vals=[10 74 14 78], b_vals=[10 14]
# CHECK-DAG: thread 7 a_vals=[11 75 15 79], b_vals=[11 15]
# CHECK-DAG: thread 8 a_vals=[16 80 20 84], b_vals=[16 20]
# CHECK-DAG: thread 9 a_vals=[17 81 21 85], b_vals=[17 21]
# CHECK-DAG: thread 10 a_vals=[18 82 22 86], b_vals=[18 22]
# CHECK-DAG: thread 11 a_vals=[19 83 23 87], b_vals=[19 23]
# CHECK-DAG: thread 12 a_vals=[24 88 28 92], b_vals=[24 28]
# CHECK-DAG: thread 13 a_vals=[25 89 29 93], b_vals=[25 29]
# CHECK-DAG: thread 14 a_vals=[26 90 30 94], b_vals=[26 30]
# CHECK-DAG: thread 15 a_vals=[27 91 31 95], b_vals=[27 31]
# CHECK-DAG: thread 16 a_vals=[32 96 36 100], b_vals=[32 36]
# CHECK-DAG: thread 17 a_vals=[33 97 37 101], b_vals=[33 37]
# CHECK-DAG: thread 18 a_vals=[34 98 38 102], b_vals=[34 38]
# CHECK-DAG: thread 19 a_vals=[35 99 39 103], b_vals=[35 39]
# CHECK-DAG: thread 20 a_vals=[40 104 44 108], b_vals=[40 44]
# CHECK-DAG: thread 21 a_vals=[41 105 45 109], b_vals=[41 45]
# CHECK-DAG: thread 22 a_vals=[42 106 46 110], b_vals=[42 46]
# CHECK-DAG: thread 23 a_vals=[43 107 47 111], b_vals=[43 47]
# CHECK-DAG: thread 24 a_vals=[48 112 52 116], b_vals=[48 52]
# CHECK-DAG: thread 25 a_vals=[49 113 53 117], b_vals=[49 53]
# CHECK-DAG: thread 26 a_vals=[50 114 54 118], b_vals=[50 54]
# CHECK-DAG: thread 27 a_vals=[51 115 55 119], b_vals=[51 55]
# CHECK-DAG: thread 28 a_vals=[56 120 60 124], b_vals=[56 60]
# CHECK-DAG: thread 29 a_vals=[57 121 61 125], b_vals=[57 61]
# CHECK-DAG: thread 30 a_vals=[58 122 62 126], b_vals=[58 62]
# CHECK-DAG: thread 31 a_vals=[59 123 63 127], b_vals=[59 63]
def test_load_f32_f32_16x8x8_b_transpose(ctx: DeviceContext):
    print("== test_load_f32_f32_16x8x8_b_transpose")
    test_load_operands[
        DType.float32, DType.float32, Index(16, 8, 8), transpose_b=True
    ](ctx)


# CHECK-LABEL: test_load_f32_f32_16x8x4
# CHECK-DAG: thread 0 a_vals=[0 32], b_vals=[0]
# CHECK-DAG: thread 1 a_vals=[1 33], b_vals=[8]
# CHECK-DAG: thread 2 a_vals=[2 34], b_vals=[16]
# CHECK-DAG: thread 3 a_vals=[3 35], b_vals=[24]
# CHECK-DAG: thread 4 a_vals=[4 36], b_vals=[1]
# CHECK-DAG: thread 5 a_vals=[5 37], b_vals=[9]
# CHECK-DAG: thread 6 a_vals=[6 38], b_vals=[17]
# CHECK-DAG: thread 7 a_vals=[7 39], b_vals=[25]
# CHECK-DAG: thread 8 a_vals=[8 40], b_vals=[2]
# CHECK-DAG: thread 9 a_vals=[9 41], b_vals=[10]
# CHECK-DAG: thread 10 a_vals=[10 42], b_vals=[18]
# CHECK-DAG: thread 11 a_vals=[11 43], b_vals=[26]
# CHECK-DAG: thread 12 a_vals=[12 44], b_vals=[3]
# CHECK-DAG: thread 13 a_vals=[13 45], b_vals=[11]
# CHECK-DAG: thread 14 a_vals=[14 46], b_vals=[19]
# CHECK-DAG: thread 15 a_vals=[15 47], b_vals=[27]
# CHECK-DAG: thread 16 a_vals=[16 48], b_vals=[4]
# CHECK-DAG: thread 17 a_vals=[17 49], b_vals=[12]
# CHECK-DAG: thread 18 a_vals=[18 50], b_vals=[20]
# CHECK-DAG: thread 19 a_vals=[19 51], b_vals=[28]
# CHECK-DAG: thread 20 a_vals=[20 52], b_vals=[5]
# CHECK-DAG: thread 21 a_vals=[21 53], b_vals=[13]
# CHECK-DAG: thread 22 a_vals=[22 54], b_vals=[21]
# CHECK-DAG: thread 23 a_vals=[23 55], b_vals=[29]
# CHECK-DAG: thread 24 a_vals=[24 56], b_vals=[6]
# CHECK-DAG: thread 25 a_vals=[25 57], b_vals=[14]
# CHECK-DAG: thread 26 a_vals=[26 58], b_vals=[22]
# CHECK-DAG: thread 27 a_vals=[27 59], b_vals=[30]
# CHECK-DAG: thread 28 a_vals=[28 60], b_vals=[7]
# CHECK-DAG: thread 29 a_vals=[29 61], b_vals=[15]
# CHECK-DAG: thread 30 a_vals=[30 62], b_vals=[23]
# CHECK-DAG: thread 31 a_vals=[31 63], b_vals=[31]
def test_load_f32_f32_16x8x4(ctx: DeviceContext):
    print("== test_load_f32_f32_16x8x4")
    test_load_operands[DType.float32, DType.float32, Index(16, 8, 4)](ctx)


# CHECK-LABEL: test_write_f32_f32_16x8x8
# CHECK: 0.0 0.0 1.0 1.0 2.0 2.0 3.0 3.0
# CHECK: 4.0 4.0 5.0 5.0 6.0 6.0 7.0 7.0
# CHECK: 8.0 8.0 9.0 9.0 10.0 10.0 11.0 11.0
# CHECK: 12.0 12.0 13.0 13.0 14.0 14.0 15.0 15.0
# CHECK: 16.0 16.0 17.0 17.0 18.0 18.0 19.0 19.0
# CHECK: 20.0 20.0 21.0 21.0 22.0 22.0 23.0 23.0
# CHECK: 24.0 24.0 25.0 25.0 26.0 26.0 27.0 27.0
# CHECK: 28.0 28.0 29.0 29.0 30.0 30.0 31.0 31.0
# CHECK: 0.0 0.0 1.0 1.0 2.0 2.0 3.0 3.0
# CHECK: 4.0 4.0 5.0 5.0 6.0 6.0 7.0 7.0
# CHECK: 8.0 8.0 9.0 9.0 10.0 10.0 11.0 11.0
# CHECK: 12.0 12.0 13.0 13.0 14.0 14.0 15.0 15.0
# CHECK: 16.0 16.0 17.0 17.0 18.0 18.0 19.0 19.0
# CHECK: 20.0 20.0 21.0 21.0 22.0 22.0 23.0 23.0
# CHECK: 24.0 24.0 25.0 25.0 26.0 26.0 27.0 27.0
# CHECK: 28.0 28.0 29.0 29.0 30.0 30.0 31.0 31.0
def test_write_f32_f32_16x8x8(ctx: DeviceContext):
    print("== test_write_f32_f32_16x8x8")
    test_write_res_operand[DType.float32, DType.float32, Index(16, 8, 8)](ctx)


# CHECK-LABEL: test_write_f32_f32_16x8x4
# CHECK: 0.0 0.0 1.0 1.0 2.0 2.0 3.0 3.0
# CHECK: 4.0 4.0 5.0 5.0 6.0 6.0 7.0 7.0
# CHECK: 8.0 8.0 9.0 9.0 10.0 10.0 11.0 11.0
# CHECK: 12.0 12.0 13.0 13.0 14.0 14.0 15.0 15.0
# CHECK: 16.0 16.0 17.0 17.0 18.0 18.0 19.0 19.0
# CHECK: 20.0 20.0 21.0 21.0 22.0 22.0 23.0 23.0
# CHECK: 24.0 24.0 25.0 25.0 26.0 26.0 27.0 27.0
# CHECK: 28.0 28.0 29.0 29.0 30.0 30.0 31.0 31.0
# CHECK: 0.0 0.0 1.0 1.0 2.0 2.0 3.0 3.0
# CHECK: 4.0 4.0 5.0 5.0 6.0 6.0 7.0 7.0
# CHECK: 8.0 8.0 9.0 9.0 10.0 10.0 11.0 11.0
# CHECK: 12.0 12.0 13.0 13.0 14.0 14.0 15.0 15.0
# CHECK: 16.0 16.0 17.0 17.0 18.0 18.0 19.0 19.0
# CHECK: 20.0 20.0 21.0 21.0 22.0 22.0 23.0 23.0
# CHECK: 24.0 24.0 25.0 25.0 26.0 26.0 27.0 27.0
# CHECK: 28.0 28.0 29.0 29.0 30.0 30.0 31.0 31.0
def test_write_f32_f32_16x8x4(ctx: DeviceContext):
    print("== test_write_f32_f32_16x8x4")
    test_write_res_operand[DType.float32, DType.float32, Index(16, 8, 4)](ctx)


fn mma_load_and_print_operands_kernel_ldmatrix[
    dst_dtype: DType,
    dtype: DType,
    lhs_layout: Layout,
    rhs_layout: Layout,
    inst_shape: StaticIntTuple[3],
    transpose_b: Bool = False,
](lhs: LayoutTensor[dtype, lhs_layout], rhs: LayoutTensor[dtype, rhs_layout]):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, transpose_b]()
    var a_smem = LayoutTensor[
        dtype,
        lhs.layout,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var b_smem = LayoutTensor[
        dtype,
        rhs.layout,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    alias thread_layout = Layout.row_major(8, 4)
    copy_dram_to_sram[thread_layout=thread_layout](a_smem, lhs)
    copy_dram_to_sram[thread_layout=thread_layout](b_smem, rhs)
    barrier()

    alias a_simd_width = mma.a_reg_type.size
    alias b_simd_width = mma.b_reg_type.size
    var a_reg_tile = LayoutTensor[
        dtype,
        Layout.row_major(1, a_simd_width),
        address_space = AddressSpace.LOCAL,
    ].stack_allocation().vectorize[1, a_simd_width]()

    var b_reg_tile = LayoutTensor[
        dtype,
        Layout.row_major(1, b_simd_width),
        address_space = AddressSpace.LOCAL,
    ].stack_allocation().vectorize[1, b_simd_width]()

    mma.load_a(a_smem, a_reg_tile)
    mma.load_b(b_smem, b_reg_tile)

    var a_frags = a_reg_tile[0, 0].cast[DType.float64]()
    var b_frags = b_reg_tile[0, 0].cast[DType.float64]()

    @parameter
    if a_frags.size == 4 and b_frags.size == 2:
        _printf["thread %u a_vals=[%g %g %g %g], b_vals=[%g %g]\n"](
            ThreadIdx.x(),
            a_frags[0],
            a_frags[1],
            a_frags[2],
            a_frags[3],
            b_frags[0],
            b_frags[1],
        )
    elif a_frags.size == 8 and b_frags.size == 4:
        _printf[
            "thread %u a_vals=[%g %g %g %g %g %g %g %g], b_vals=[%g %g %g %g]\n"
        ](
            ThreadIdx.x(),
            a_frags[0],
            a_frags[1],
            a_frags[2],
            a_frags[3],
            a_frags[4],
            a_frags[5],
            a_frags[6],
            a_frags[7],
            b_frags[0],
            b_frags[1],
            b_frags[2],
            b_frags[3],
        )


def test_load_operands_ldmatrix[
    dst_dtype: DType,
    dtype: DType,
    shape: StaticIntTuple[3],
    transpose_b: Bool = False,
](ctx: DeviceContext):
    alias M = shape[0]
    alias N = shape[1]
    alias K = shape[2]

    var lhs = ManagedLayoutTensor[
        dtype, Layout.row_major(M, K), gpu_managed_alloc, gpu_free
    ]()
    arange(lhs.tensor)
    var rhs = ManagedLayoutTensor[
        dtype, Layout.row_major(K, N), gpu_managed_alloc, gpu_free
    ]()
    arange(rhs.tensor)
    alias mma_load_and_print_kernel_fn = mma_load_and_print_operands_kernel_ldmatrix[
        dst_dtype, dtype, lhs.layout, rhs.layout, shape, transpose_b
    ]
    var func = ctx.compile_function[
        mma_load_and_print_kernel_fn, dump_ptx=False
    ]()
    ctx.enqueue_function(
        func, lhs.tensor, rhs.tensor, grid_dim=(1, 1), block_dim=(32)
    )
    ctx.synchronize()
    _ = func^
    _ = lhs^
    _ = rhs^


# CHECK-LABEL: test_load_f32_bf16_16x8x16_ldmatrix
# CHECK-DAG thread 0 a_vals=[0 1 128 129 8 9 136 137], b_vals=[0 8 64 72]
# CHECK-DAG thread 1 a_vals=[2 3 130 131 10 11 138 139], b_vals=[16 24 80 88]
# CHECK-DAG thread 2 a_vals=[4 5 132 133 12 13 140 141], b_vals=[32 40 96 104]
# CHECK-DAG thread 3 a_vals=[6 7 134 135 14 15 142 143], b_vals=[48 56 112 120]
# CHECK-DAG thread 4 a_vals=[16 17 144 145 24 25 152 153], b_vals=[1 9 65 73]
# CHECK-DAG thread 5 a_vals=[18 19 146 147 26 27 154 155], b_vals=[17 25 81 89]
# CHECK-DAG thread 6 a_vals=[20 21 148 149 28 29 156 157], b_vals=[33 41 97 105]
# CHECK-DAG thread 7 a_vals=[22 23 150 151 30 31 158 159], b_vals=[49 57 113 121]
# CHECK-DAG thread 8 a_vals=[32 33 160 161 40 41 168 169], b_vals=[2 10 66 74]
# CHECK-DAG thread 9 a_vals=[34 35 162 163 42 43 170 171], b_vals=[18 26 82 90]
# CHECK-DAG thread 10 a_vals=[36 37 164 165 44 45 172 173], b_vals=[34 42 98 106]
# CHECK-DAG thread 11 a_vals=[38 39 166 167 46 47 174 175], b_vals=[50 58 114 122]
# CHECK-DAG thread 12 a_vals=[48 49 176 177 56 57 184 185], b_vals=[3 11 67 75]
# CHECK-DAG thread 13 a_vals=[50 51 178 179 58 59 186 187], b_vals=[19 27 83 91]
# CHECK-DAG thread 14 a_vals=[52 53 180 181 60 61 188 189], b_vals=[35 43 99 107]
# CHECK-DAG thread 15 a_vals=[54 55 182 183 62 63 190 191], b_vals=[51 59 115 123]
# CHECK-DAG thread 16 a_vals=[72 73 200 201 64 65 192 193], b_vals=[4 12 68 76]
# CHECK-DAG thread 17 a_vals=[74 75 202 203 66 67 194 195], b_vals=[20 28 84 92]
# CHECK-DAG thread 18 a_vals=[76 77 204 205 68 69 196 197], b_vals=[36 44 100 108]
# CHECK-DAG thread 19 a_vals=[78 79 206 207 70 71 198 199], b_vals=[52 60 116 124]
# CHECK-DAG thread 20 a_vals=[88 89 216 217 80 81 208 209], b_vals=[5 13 69 77]
# CHECK-DAG thread 21 a_vals=[90 91 218 219 82 83 210 211], b_vals=[21 29 85 93]
# CHECK-DAG thread 22 a_vals=[92 93 220 221 84 85 212 213], b_vals=[37 45 101 109]
# CHECK-DAG thread 23 a_vals=[94 95 222 223 86 87 214 215], b_vals=[53 61 117 125]
# CHECK-DAG thread 24 a_vals=[104 105 232 233 96 97 224 225], b_vals=[6 14 70 78]
# CHECK-DAG thread 25 a_vals=[106 107 234 235 98 99 226 227], b_vals=[22 30 86 94]
# CHECK-DAG thread 26 a_vals=[108 109 236 237 100 101 228 229], b_vals=[38 46 102 110]
# CHECK-DAG thread 27 a_vals=[110 111 238 239 102 103 230 231], b_vals=[54 62 118 126]
# CHECK-DAG thread 28 a_vals=[120 121 248 249 112 113 240 241], b_vals=[7 15 71 79]
# CHECK-DAG thread 29 a_vals=[122 123 250 251 114 115 242 243], b_vals=[23 31 87 95]
# CHECK-DAG thread 30 a_vals=[124 125 252 253 116 117 244 245], b_vals=[39 47 103 111]
# CHECK-DAG thread 31 a_vals=[126 127 254 255 118 119 246 247], b_vals=[55 63 119 127]


def test_load_f32_bf16_16x8x16_ldmatrix(ctx: DeviceContext):
    print("== test_load_f32_bf16_16x8x16_ldmatrix")
    test_load_operands_ldmatrix[
        DType.float32, DType.bfloat16, Index(16, 8, 16)
    ](ctx)


# CHECK-LABEL: test_load_f32_f32_16x8x8_ldmatrix
# CHECK-DAG thread 0 a_vals=[0 64 4 68], b_vals=[0 32]
# CHECK-DAG thread 1 a_vals=[1 65 5 69], b_vals=[8 40]
# CHECK-DAG thread 2 a_vals=[2 66 6 70], b_vals=[16 48]
# CHECK-DAG thread 3 a_vals=[3 67 7 71], b_vals=[24 56]
# CHECK-DAG thread 4 a_vals=[8 72 12 76], b_vals=[1 33]
# CHECK-DAG thread 5 a_vals=[9 73 13 77], b_vals=[9 41]
# CHECK-DAG thread 6 a_vals=[10 74 14 78], b_vals=[17 49]
# CHECK-DAG thread 7 a_vals=[11 75 15 79], b_vals=[25 57]
# CHECK-DAG thread 8 a_vals=[16 80 20 84], b_vals=[2 34]
# CHECK-DAG thread 9 a_vals=[17 81 21 85], b_vals=[10 42]
# CHECK-DAG thread 10 a_vals=[18 82 22 86], b_vals=[18 50]
# CHECK-DAG thread 11 a_vals=[19 83 23 87], b_vals=[26 58]
# CHECK-DAG thread 12 a_vals=[24 88 28 92], b_vals=[3 35]
# CHECK-DAG thread 13 a_vals=[25 89 29 93], b_vals=[11 43]
# CHECK-DAG thread 14 a_vals=[26 90 30 94], b_vals=[19 51]
# CHECK-DAG thread 15 a_vals=[27 91 31 95], b_vals=[27 59]
# CHECK-DAG thread 16 a_vals=[36 100 32 96], b_vals=[4 36]
# CHECK-DAG thread 17 a_vals=[37 101 33 97], b_vals=[12 44]
# CHECK-DAG thread 18 a_vals=[38 102 34 98], b_vals=[20 52]
# CHECK-DAG thread 19 a_vals=[39 103 35 99], b_vals=[28 60]
# CHECK-DAG thread 20 a_vals=[44 108 40 104], b_vals=[5 37]
# CHECK-DAG thread 21 a_vals=[45 109 41 105], b_vals=[13 45]
# CHECK-DAG thread 22 a_vals=[46 110 42 106], b_vals=[21 53]
# CHECK-DAG thread 23 a_vals=[47 111 43 107], b_vals=[29 61]
# CHECK-DAG thread 24 a_vals=[52 116 48 112], b_vals=[6 38]
# CHECK-DAG thread 25 a_vals=[53 117 49 113], b_vals=[14 46]
# CHECK-DAG thread 26 a_vals=[54 118 50 114], b_vals=[22 54]
# CHECK-DAG thread 27 a_vals=[55 119 51 115], b_vals=[30 62]
# CHECK-DAG thread 28 a_vals=[60 124 56 120], b_vals=[7 39]
# CHECK-DAG thread 29 a_vals=[61 125 57 121], b_vals=[15 47]
# CHECK-DAG thread 30 a_vals=[62 126 58 122], b_vals=[23 55]
# CHECK-DAG thread 31 a_vals=[63 127 59 123], b_vals=[31 63]
def test_load_f32_f32_16x8x8_ldmatrix(ctx: DeviceContext):
    print("== test_load_f32_f32_16x8x8_ldmatrix")
    test_load_operands_ldmatrix[DType.float32, DType.float32, Index(16, 8, 8)](
        ctx
    )


def main():
    with DeviceContext() as ctx:
        test_load_f32_f32_16x8x8(ctx)
        test_load_f32_f32_16x8x8_b_transpose(ctx)
        test_load_f32_f32_16x8x4(ctx)
        test_write_f32_f32_16x8x8(ctx)
        test_write_f32_f32_16x8x4(ctx)
        # ldmatrix
        test_load_f32_bf16_16x8x16_ldmatrix(ctx)
        test_load_f32_f32_16x8x8_ldmatrix(ctx)
