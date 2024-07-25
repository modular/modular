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
from gpu.host.memory import _memset
from gpu.host.device_context import DeviceContext, DeviceBuffer
from linalg.matmul_gpu import _matmul_gpu, matmul_kernel_naive
from memory import memset_zero, stack_allocation
from memory.reference import _GPUAddressSpace as GPUAddressSpace
from gpu.cublas.cublas import (
    check_cublas_error,
    cublasContext,
    cublasCreate,
    cublasDestroy,
)

from linalg.cublas import cublas_matmul
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
from gpu.cublas.cublas import (
    check_cublas_error,
    cublasContext,
    cublasCreate,
    cublasDestroy,
)
from buffer.dimlist import _make_tuple
from linalg.cublas import cublas_matmul
from testing import assert_equal as assert_equal_val


struct test_matmul[
    type: DType,
    static_b_shape: DimList = DimList.create_unknown[2](),
    transpose_b: Bool = False,
]:
    var ctx: DeviceContext

    var M: Int
    var N: Int
    var K: Int

    var low_precision: Bool

    var a_host: HostNDBuffer[type, 2]
    var b_host: HostNDBuffer[type, 2, static_b_shape]
    var c_host: HostNDBuffer[type, 2]
    var c_host_ref: HostNDBuffer[type, 2]

    var a_device: DeviceNDBuffer[type, 2]
    var b_device: DeviceNDBuffer[type, 2, static_b_shape]
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
        self.N = shape[1]
        self.K = shape[2]

        @parameter
        if static_b_shape.all_known[2]():
            alias b_k_dim = 1 if transpose_b else 0
            alias b_n_dim = 0 if transpose_b else 1
            assert_equal_val(self.K, static_b_shape.get[b_k_dim]())
            assert_equal_val(self.N, static_b_shape.get[b_n_dim]())

        self.low_precision = low_precision

        var a_shape = DimList(self.M, self.K)
        var b_shape = DimList(self.K, self.N)
        var c_shape = DimList(self.M, self.N)

        self.a_host = HostNDBuffer[type, 2](a_shape)
        self.b_host = HostNDBuffer[type, 2, static_b_shape](b_shape)
        self.c_host = HostNDBuffer[type, 2](c_shape)
        self.c_host_ref = HostNDBuffer[type, 2](c_shape)

        self.a_device = DeviceNDBuffer[type, 2](a_shape, ctx=ctx)
        self.b_device = DeviceNDBuffer[type, 2, static_b_shape](
            b_shape, ctx=ctx
        )
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
        print("=== test_matmul")

        var ctx = self.ctx

        ctx.enqueue_copy_to_device(
            self.a_device.buffer, self.a_host.tensor.data
        )
        ctx.enqueue_copy_to_device(
            self.b_device.buffer, self.b_host.tensor.data
        )
        _memset(self.c_device.buffer.ptr, 0, self.M * self.N)
        _memset(self.c_device_ref.buffer.ptr, 0, self.M * self.N)

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
                self.c_host_ref.tensor,
                self.c_host.tensor,
                atol=0.0001,
                rtol=0.01,
            )
        else:
            assert_equal(self.c_host_ref.tensor, self.c_host.tensor)


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with DeviceContext() as ctx:

            @parameter
            fn basic_test[
                type: DType,
                /,
                *,
                shape: DimList = DimList.create_unknown[2](),
                transpose_b: Bool = False,
                use_tensor_core: Bool = True,
            ](test_ctx: test_matmul[type, shape, transpose_b]) raises:
                _matmul_gpu[use_tensor_core=use_tensor_core](
                    test_ctx.c_device.tensor,
                    test_ctx.a_device.tensor,
                    test_ctx.b_device.tensor,
                    ctx,
                    True,
                )

                var handle = UnsafePointer[cublasContext]()
                check_cublas_error(
                    cublasCreate(UnsafePointer.address_of(handle))
                )
                check_cublas_error(
                    cublas_matmul(
                        handle,
                        test_ctx.c_device_ref.tensor,
                        test_ctx.a_device.tensor,
                        test_ctx.b_device.tensor,
                        c_row_major=True,
                        transpose_b=transpose_b,
                    )
                )
                check_cublas_error(cublasDestroy(handle))

            test_matmul[DType.float32, DimList(128, 384)](
                ctx, (256, 384, 128), low_precision=True
            ).run_test[
                basic_test[
                    DType.float32,
                    shape = DimList(128, 384),
                    use_tensor_core=True,
                ]
            ]()

            # TODO: re-enable after KERN-702
            # test_matmul[DType.float32](ctx, (111, 133, 157)).run_test[
            #     basic_test[DType.float32]
            # ]()

            test_matmul[DType.float32, DimList(4096, 4096)](
                ctx, (256, 4096, 4096), low_precision=True
            ).run_test[
                basic_test[
                    DType.float32,
                    shape = DimList(4096, 4096),
                    use_tensor_core=True,
                ]
            ]()

            test_matmul[DType.bfloat16, DimList(3072, 5120)](
                ctx, (1024, 5120, 3072), low_precision=True
            ).run_test[
                basic_test[
                    DType.bfloat16,
                    shape = DimList(3072, 5120),
                    use_tensor_core=True,
                ]
            ]()

            test_matmul[DType.bfloat16, DimList(32768, 3072)](
                ctx, (1024, 3072, 32768), low_precision=True
            ).run_test[
                basic_test[
                    DType.bfloat16,
                    shape = DimList(32768, 3072),
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

            test_matmul[DType.float32](
                ctx, (128, 256, 512), low_precision=True
            ).run_test[epilogue_test[DType.float32]]()
            # TODO: change back to K = 255 after KERN-702
            test_matmul[DType.float32](
                ctx, (192, 128, 256), low_precision=True
            ).run_test[epilogue_test[DType.float32]]()

    except e:
        print("CUDA_ERROR:", e)
