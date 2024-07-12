# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug -I%S/../ %s | FileCheck %s

from math import ceildiv

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import BlockDim, BlockIdx, ThreadIdx, barrier
from gpu.host.device_context import DeviceContext, DeviceBuffer
from linalg.matmul_gpu import _matmul_gpu, matmul_kernel_naive
from memory import memset_zero, stack_allocation
from memory.reference import _GPUAddressSpace as GPUAddressSpace

from utils.index import Index
from internal_utils import (
    HostNDBuffer,
    DeviceNDBuffer,
    fill,
    zero,
    linspace,
    random,
    assert_equal,
    assert_almost_equal,
)
from buffer.dimlist import _make_tuple
from testing import assert_equal as assert_equal_val


alias TILE_SZ_A = 128
alias TILE_SZ_B = 16
alias TILE_SZ_RATIO = TILE_SZ_A // TILE_SZ_B


fn matmul(
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    c_ptr: DTypePointer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
):
    var a = NDBuffer[DType.float32, 2](a_ptr, Index(m, k))
    var b = NDBuffer[DType.float32, 2](b_ptr, Index(k, n))
    var c = NDBuffer[DType.float32, 2](c_ptr, Index(m, n))

    # Compute C = A x B
    #   where A is a (m x k) matrix
    #   where B is a (k x n) matrix
    #   where C is a (m x n) matrix
    #
    # Use register and shared memory tiling and thread coarsening
    #
    # NOTE: A and C are column major, B is row major.

    # Allocate B array into shared memory for tiling.
    var b_shared = stack_allocation[
        TILE_SZ_RATIO * TILE_SZ_B,
        DType.float32,
        address_space = GPUAddressSpace.SHARED,
    ]()

    # Thread indexing offsets.
    var row: UInt = BlockIdx.x() * BlockDim.x() + ThreadIdx.x()
    var col: UInt = BlockIdx.y() * TILE_SZ_B

    # Privatization of the C matrix.
    var c_reg = stack_allocation[TILE_SZ_B, DType.float32]()

    memset_zero(c_reg, TILE_SZ_B)

    # Loop over each input tile.
    for tile_idx in range((k - 1) // TILE_SZ_RATIO + 1):
        var i: UInt = ThreadIdx.x() // TILE_SZ_B
        var j: UInt = ThreadIdx.x() % TILE_SZ_B

        # Load the B matrix into shared memory.
        var b_val: Float32
        if tile_idx * TILE_SZ_RATIO + i < k and col + j < n:
            b_val = b[
                (tile_idx * TILE_SZ_RATIO + i),
                (col + j),
            ]
        else:
            b_val = 0
        b_shared[i * TILE_SZ_B + j] = b_val

        barrier()

        # Loop within the tile.
        for idx in range(TILE_SZ_RATIO):
            # Load the A tile into the register.
            var a_reg: Float32
            if row < m and tile_idx * TILE_SZ_RATIO + idx < k:
                a_reg = a[row, (tile_idx * TILE_SZ_RATIO + idx)]
            else:
                a_reg = 0

            # Compute the output element for each thread.
            for out_idx in range(TILE_SZ_B):
                c_reg[out_idx] += (
                    a_reg * b_shared[idx * TILE_SZ_RATIO + out_idx]
                )
        barrier()

    # Store the values into the output matrix.
    for out_idx in range(TILE_SZ_B):
        if row < m and out_idx + col < n:
            c[Index(row, col + out_idx)] = c_reg[out_idx]


struct run_matmul[m: Int = 512, n: Int = 512, k: Int = 512]:
    var ctx: DeviceContext

    var a_host: HostNDBuffer[DType.float32, 2, DimList(m, k)]
    var b_host: HostNDBuffer[DType.float32, 2, DimList(k, m)]
    var c_host: HostNDBuffer[DType.float32, 2, DimList(m, n)]

    var a_device: DeviceBuffer[DType.float32]
    var b_device: DeviceBuffer[DType.float32]
    var c_device: DeviceBuffer[DType.float32]

    fn __init__(inout self, ctx: DeviceContext) raises:
        self.ctx = ctx

        self.a_host = HostNDBuffer[DType.float32, 2, DimList(m, k)]()
        self.b_host = HostNDBuffer[DType.float32, 2, DimList(k, m)]()
        self.c_host = HostNDBuffer[DType.float32, 2, DimList(m, n)]()

        self.a_device = ctx.create_buffer[DType.float32](m * k)
        self.b_device = ctx.create_buffer[DType.float32](k * n)
        self.c_device = ctx.create_buffer[DType.float32](m * n)

        fill(self.a_host.tensor, 1.0)
        fill(self.b_host.tensor, 1.0)
        zero(self.c_host.tensor)

    # CHECK-LABEL: run_matmul
    fn run_test(self) raises:
        print("== run_matmul")

        var ctx = self.ctx

        ctx.enqueue_copy_to_device(self.a_device, self.a_host.tensor.data)
        ctx.enqueue_copy_to_device(self.b_device, self.b_host.tensor.data)

        var func_matmul = ctx.compile_function[matmul]()

        ctx.enqueue_function(
            func_matmul,
            self.a_device,
            self.b_device,
            self.c_device,
            m,
            n,
            k,
            grid_dim=(ceildiv(m, TILE_SZ_A), ceildiv(n, TILE_SZ_B)),
            block_dim=(TILE_SZ_A, 1),
        )
        ctx.enqueue_copy_from_device(self.c_host.tensor.data, self.c_device)

        ctx.synchronize()

        for i in range(10):
            for j in range(10):
                print(
                    "at index = [",
                    i,
                    ",",
                    j,
                    "] the value is",
                    self.c_host.tensor[i, j],
                )


struct test_matmul[
    type: DType,
    static_shape: DimList = DimList.create_unknown[2](),
    transpose_b: Bool = False,
]:
    var ctx: DeviceContext

    var M: Int
    var N: Int
    var K: Int

    var low_precision: Bool

    var a_host: HostNDBuffer[type, 2]
    var b_host: HostNDBuffer[type, 2, static_shape]
    var c_host: HostNDBuffer[type, 2]
    var c_host_ref: HostNDBuffer[type, 2]

    var a_device: DeviceNDBuffer[type, 2]
    var b_device: DeviceNDBuffer[type, 2, static_shape]
    var c_device: DeviceNDBuffer[type, 2]
    var c_device_ref: DeviceNDBuffer[type, 2]

    fn __init__(
        inout self,
        ctx: DeviceContext,
        shape: Tuple[Int, Int, Int],
        low_precision: Bool = False,
    ) raises:
        self.ctx = ctx

        self.M = shape[0]
        self.K = shape[1]
        self.N = shape[2]

        alias kK = static_shape.get[1]() if transpose_b else static_shape.get[
            0
        ]()
        alias kN = static_shape.get[0]() if transpose_b else static_shape.get[
            1
        ]()

        @parameter
        if static_shape.all_known[2]():
            assert_equal_val(shape[1], kK)
            assert_equal_val(shape[2], kN)

        self.low_precision = low_precision

        var a_shape = DimList(self.M, self.K)
        var b_shape = DimList(self.K, self.N)
        var c_shape = DimList(self.M, self.N)

        self.a_host = HostNDBuffer[type, 2](a_shape)
        self.b_host = HostNDBuffer[type, 2, static_shape](b_shape)
        self.c_host = HostNDBuffer[type, 2](c_shape)
        self.c_host_ref = HostNDBuffer[type, 2](c_shape)

        self.a_device = DeviceNDBuffer[type, 2](a_shape, ctx=ctx)
        self.b_device = DeviceNDBuffer[type, 2, static_shape](b_shape, ctx=ctx)
        self.c_device = DeviceNDBuffer[type, 2](c_shape, ctx=ctx)
        self.c_device_ref = DeviceNDBuffer[type, 2](c_shape, ctx=ctx)

        if low_precision:
            linspace(self.a_host.tensor)
            linspace(self.b_host.tensor)
        else:
            random(self.a_host.tensor)
            random(self.b_host.tensor)

        zero(self.c_host.tensor)
        zero(self.c_host_ref.tensor)

    fn run_test[test_function: fn (Self) raises capturing -> None](self) raises:
        var ctx = self.ctx

        ctx.enqueue_copy_to_device(
            self.a_device.buffer, self.a_host.tensor.data
        )
        ctx.enqueue_copy_to_device(
            self.b_device.buffer, self.b_host.tensor.data
        )
        ctx.enqueue_copy_to_device(
            self.c_device.buffer, self.c_host.tensor.data
        )
        ctx.enqueue_copy_to_device(
            self.c_device_ref.buffer, self.c_host_ref.tensor.data
        )

        test_function(self)

        ctx.enqueue_copy_from_device(
            self.c_host.tensor.data, self.c_device.buffer
        )
        ctx.enqueue_copy_from_device(
            self.c_host_ref.tensor.data, self.c_device_ref.buffer
        )
        ctx.synchronize()

        if self.low_precision:
            assert_almost_equal(
                self.c_host_ref.tensor, self.c_host.tensor, rtol=0.01
            )
        else:
            assert_equal(self.c_host_ref.tensor, self.c_host.tensor)


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with DeviceContext() as ctx:
            run_matmul(ctx).run_test()

            @parameter
            fn basic_test[
                type: DType,
                /,
                *,
                shape: DimList = DimList.create_unknown[2](),
                transpose_b: Bool = False,
                use_tensor_core: Bool = False,
            ](test_ctx: test_matmul[type, shape, transpose_b]) raises:
                var M = test_ctx.M
                var K = test_ctx.K
                var N = test_ctx.N

                _matmul_gpu[use_tensor_core=use_tensor_core](
                    test_ctx.c_device.tensor,
                    test_ctx.a_device.tensor,
                    test_ctx.b_device.tensor,
                    ctx,
                )

                alias BLOCK_DIM = 16
                var gemm_naive = ctx.compile_function[
                    matmul_kernel_naive[type, type, type, BLOCK_DIM]
                ](threads_per_block=256)

                ctx.enqueue_function(
                    gemm_naive,
                    test_ctx.c_device_ref.buffer,
                    test_ctx.a_device.buffer,
                    test_ctx.b_device.buffer,
                    M,
                    N,
                    K,
                    grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
                    block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
                )

            alias small_test = (1024, 3072, 5120)
            alias large_test = (1024, 12288, 3072)

            test_matmul[DType.float32](ctx, small_test).run_test[
                basic_test[DType.float32]
            ]()
            test_matmul[DType.float32](ctx, large_test).run_test[
                basic_test[DType.float32]
            ]()

            # Low precision test (use_tensor_core)
            test_matmul[DType.float32](ctx, small_test, True).run_test[
                basic_test[DType.float32, use_tensor_core=True]
            ]()
            test_matmul[DType.bfloat16](ctx, small_test, True).run_test[
                basic_test[DType.bfloat16, use_tensor_core=True]
            ]()

            test_matmul[DType.float32, DimList(3072, 5120)](
                ctx, small_test, True
            ).run_test[
                basic_test[
                    DType.float32,
                    shape = DimList(3072, 5120),
                    transpose_b=False,
                    use_tensor_core=True,
                ]
            ]()

            @parameter
            fn epilogue_test[
                type: DType, use_tensor_core: Bool = False
            ](test_ctx: test_matmul[type]) raises:
                var M = test_ctx.M
                var K = test_ctx.K
                var N = test_ctx.N

                alias some_constant = 20

                var c_tensor = test_ctx.c_device.tensor

                @parameter
                @always_inline
                @__copy_capture(c_tensor)
                fn epilogue_fn[
                    _type: DType, width: Int
                ](
                    idx: StaticIntTuple[2], val: SIMD[_type, width]
                ) capturing -> None:
                    c_tensor.store(
                        idx, rebind[SIMD[type, width]](val + some_constant)
                    )

                var c_ref_tensor = test_ctx.c_device_ref.tensor

                @parameter
                @always_inline
                @__copy_capture(c_ref_tensor)
                fn naive_epilogue_fn[
                    _type: DType, width: Int
                ](
                    idx: StaticIntTuple[2], val: SIMD[_type, width]
                ) capturing -> None:
                    c_ref_tensor.store(
                        idx, rebind[SIMD[type, width]](val + some_constant)
                    )

                _matmul_gpu[
                    use_tensor_core=use_tensor_core,
                    transpose_b=False,
                    elementwise_lambda_fn=epilogue_fn,
                ](
                    test_ctx.c_device.tensor,
                    test_ctx.a_device.tensor,
                    test_ctx.b_device.tensor,
                    ctx,
                )

                alias BLOCK_DIM = 16
                var gemm_naive = ctx.compile_function[
                    matmul_kernel_naive[
                        type,
                        type,
                        type,
                        BLOCK_DIM,
                        elementwise_lambda_fn=naive_epilogue_fn,
                    ]
                ]()

                ctx.enqueue_function(
                    gemm_naive,
                    test_ctx.c_device_ref.buffer,
                    test_ctx.a_device.buffer,
                    test_ctx.b_device.buffer,
                    M,
                    N,
                    K,
                    grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM, BLOCK_DIM),
                )

            test_matmul[DType.float32](ctx, small_test).run_test[
                epilogue_test[DType.float32]
            ]()
            test_matmul[DType.float32](ctx, large_test).run_test[
                epilogue_test[DType.float32]
            ]()

            # Low precision test (use_tensor_core)
            test_matmul[DType.float32](ctx, small_test, True).run_test[
                epilogue_test[DType.float32, True]
            ]()
            test_matmul[DType.bfloat16](ctx, small_test, True).run_test[
                epilogue_test[DType.bfloat16, True]
            ]()
    except e:
        print("CUDA_ERROR:", e)
