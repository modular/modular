# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s

from collections.optional import Optional
from math import ceildiv
from sys import simdwidthof

from algorithm.functional import elementwise
from buffer import NDBuffer
from buffer.dimlist import DimList, Dim, _make_tuple
from gpu import BlockDim, BlockIdx, ThreadIdx, barrier
from gpu.cublas.cublas import (
    check_cublas_error,
    cublasContext,
    cublasCreate,
    cublasDestroy,
)
from gpu.host._compile import _get_nvptx_target
from gpu.host.device_context import DeviceBuffer, DeviceContext
from gpu.host.memory import _memset
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    assert_equal,
    fill,
    linspace,
    random,
    zero,
)
from linalg.cublas import cublas_matmul
from linalg.matmul_gpu import _matmul_gpu, matmul_kernel_naive
from memory import memset_zero, stack_allocation
from memory.reference import _GPUAddressSpace as GPUAddressSpace
from testing import assert_equal as assert_equal_val

from utils import StaticIntTuple
from utils.index import Index

alias init_fn_type = fn (buff: NDBuffer) -> None


struct test_matmul[
    type: DType,
    static_KN: DimList = DimList.create_unknown[2](),
    transpose_b: Bool = False,
    init_a: Optional[init_fn_type] = None,
    init_b: Optional[init_fn_type] = None,
]:
    var ctx: DeviceContext

    var M: Int
    var N: Int
    var K: Int

    # fmt: off
    alias dim_K = static_KN.at[0]()
    alias dim_N = static_KN.at[1]()
    alias static_a_shape = DimList(Dim(), Self.dim_K)
    alias static_b_shape = DimList(Self.dim_N, Self.dim_K) if transpose_b else DimList(Self.dim_K, Self.dim_N)
    alias static_c_shape = DimList(Dim(), Self.dim_N)
    # fmt: on

    var a_host: HostNDBuffer[type, 2, Self.static_a_shape]
    var b_host: HostNDBuffer[type, 2, Self.static_b_shape]
    var c_host: HostNDBuffer[type, 2, Self.static_c_shape]
    var c_host_ref: HostNDBuffer[type, 2, Self.static_c_shape]

    var a_device: DeviceNDBuffer[type, 2, Self.static_a_shape]
    var b_device: DeviceNDBuffer[type, 2, Self.static_b_shape]
    var c_device: DeviceNDBuffer[type, 2, Self.static_c_shape]
    var c_device_ref: DeviceNDBuffer[type, 2, Self.static_c_shape]

    fn __init__(
        inout self,
        ctx: DeviceContext,
        shape: Tuple[Int, Int, Int],
    ) raises:
        self.ctx = ctx

        self.M = shape[0]
        self.N = shape[1]
        self.K = shape[2]

        @parameter
        if static_KN.all_known[2]():
            assert_equal_val(self.K, static_KN.get[0]())
            assert_equal_val(self.N, static_KN.get[1]())

        var dynamic_a_shape = DimList(self.M, self.K)
        var dynamic_b_shape = DimList(
            self.N, self.K
        ) if transpose_b else DimList(self.K, self.N)
        var dynamic_c_shape = DimList(self.M, self.N)

        self.a_host = HostNDBuffer[type, 2, self.static_a_shape](
            dynamic_a_shape
        )
        self.b_host = HostNDBuffer[type, 2, self.static_b_shape](
            dynamic_b_shape
        )
        self.c_host = HostNDBuffer[type, 2, self.static_c_shape](
            dynamic_c_shape
        )
        self.c_host_ref = HostNDBuffer[type, 2, self.static_c_shape](
            dynamic_c_shape
        )

        self.a_device = DeviceNDBuffer[type, 2, self.static_a_shape](
            dynamic_a_shape, ctx=ctx
        )
        self.b_device = DeviceNDBuffer[type, 2, self.static_b_shape](
            dynamic_b_shape, ctx=ctx
        )
        self.c_device = DeviceNDBuffer[type, 2, self.static_c_shape](
            dynamic_c_shape, ctx=ctx
        )
        self.c_device_ref = DeviceNDBuffer[type, 2, self.static_c_shape](
            dynamic_c_shape, ctx=ctx
        )

        @parameter
        if init_a:
            alias init_a_fn = init_a.value()
            init_a_fn(self.a_host.tensor)
        else:
            random(self.a_host.tensor)

        @parameter
        if init_b:
            alias init_b_fn = init_b.value()
            init_b_fn(self.b_host.tensor)
        else:
            random(self.b_host.tensor)

        zero(self.c_host.tensor)
        zero(self.c_host_ref.tensor)

    fn run_test[test_function: fn (Self) raises capturing -> None](self) raises:
        print(
            "test_matmul",
            self.M,
            "x",
            self.N,
            "x",
            self.K,
            "transpose_b" if transpose_b else "",
        )

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

        assert_almost_equal(
            self.c_host.tensor,
            self.c_host_ref.tensor,
            atol=0.0001,
            rtol=0.02,
        )


def main():
    with DeviceContext() as ctx:

        @parameter
        fn basic_test[
            type: DType,
            /,
            *,
            shape: DimList = DimList.create_unknown[2](),
            transpose_b: Bool = False,
            use_tensor_core: Bool = True,
            init_a: Optional[init_fn_type] = None,
            init_b: Optional[init_fn_type] = None,
        ](
            test_ctx: test_matmul[
                type,
                shape,
                transpose_b,
                init_a,
                init_b,
            ]
        ) raises:
            _matmul_gpu[
                use_tensor_core=use_tensor_core, transpose_b=transpose_b
            ](
                test_ctx.c_device.tensor,
                test_ctx.a_device.tensor,
                test_ctx.b_device.tensor,
                ctx,
            )

            var handle = UnsafePointer[cublasContext]()
            check_cublas_error(cublasCreate(UnsafePointer.address_of(handle)))
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

        print("===> tfloat32-float32 mma")

        test_matmul[DType.float32, DimList(128, 384)](
            ctx, (256, 384, 128)
        ).run_test[
            basic_test[
                DType.float32,
                shape = DimList(128, 384),
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.float32, DimList(4096, 4096)](
            ctx, (128, 4096, 4096)
        ).run_test[
            basic_test[
                DType.float32,
                shape = DimList(4096, 4096),
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.float32, DimList(4096, 12288)](
            ctx, (512, 12288, 4096)
        ).run_test[
            basic_test[
                DType.float32,
                shape = DimList(4096, 12288),
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.float32, DimList(11008, 4096)](
            ctx, (23, 4096, 11008)
        ).run_test[
            basic_test[
                DType.float32,
                shape = DimList(11008, 4096),
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.float32, DimList(12288, 4096)](
            ctx, (67, 4096, 12288)
        ).run_test[
            basic_test[
                DType.float32,
                shape = DimList(12288, 4096),
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.float32, DimList(4096, 4096)](
            ctx, (555, 4096, 4096)
        ).run_test[
            basic_test[
                DType.float32,
                shape = DimList(4096, 4096),
                use_tensor_core=True,
            ]
        ]()

        print("===> bfloat16-float32 mma")

        test_matmul[DType.bfloat16, DimList(12288, 3072)](
            ctx, (1024, 3072, 12288)
        ).run_test[
            basic_test[
                DType.bfloat16,
                shape = DimList(12288, 3072),
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.bfloat16, DimList(3072, 12288)](
            ctx, (1024, 12288, 3072)
        ).run_test[
            basic_test[
                DType.bfloat16,
                shape = DimList(3072, 12288),
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.bfloat16, DimList(3072, 5120)](
            ctx, (1024, 5120, 3072)
        ).run_test[
            basic_test[
                DType.bfloat16,
                shape = DimList(3072, 5120),
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.bfloat16, DimList(32768, 3072)](
            ctx, (1024, 3072, 32768)
        ).run_test[
            basic_test[
                DType.bfloat16,
                shape = DimList(32768, 3072),
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.bfloat16, DimList(3072, 3072)](
            ctx, (1024, 3072, 3072)
        ).run_test[
            basic_test[
                DType.bfloat16,
                shape = DimList(3072, 3072),
                use_tensor_core=True,
            ]
        ]()

        @parameter
        fn epilogue_test[
            type: DType,
            /,
            *,
            shape: DimList = DimList.create_unknown[2](),
            transpose_b: Bool = False,
            use_tensor_core: Bool = False,
        ](test_ctx: test_matmul[type, shape, transpose_b]) raises:
            var M = test_ctx.M
            var N = test_ctx.N

            var c_tensor = test_ctx.c_device.tensor
            var ctx = test_ctx.ctx
            var epilogue_shape = Index(M, N)
            var epilogue_host = HostNDBuffer[type, 2](epilogue_shape)
            var epilogue_device = DeviceNDBuffer[type, 2](
                epilogue_shape, ctx=ctx
            )
            random(epilogue_host.tensor, 0.5, 1.5)
            ctx.enqueue_copy_to_device(
                epilogue_device.buffer, epilogue_host.tensor.data
            )
            var epilogue_buff = epilogue_device.tensor

            @parameter
            @always_inline
            @__copy_capture(c_tensor, epilogue_buff)
            fn epilogue_fn[
                _type: DType, width: Int, *, alignment: Int = 1
            ](
                idx: StaticIntTuple[2], val: SIMD[_type, width]
            ) capturing -> None:
                var another_val = rebind[SIMD[_type, width]](
                    epilogue_buff.load[width=width](idx)
                )
                c_tensor.store(
                    idx, rebind[SIMD[type, width]](val * another_val)
                )

            alias pack_size = simdwidthof[type, target = _get_nvptx_target()]()

            _matmul_gpu[
                use_tensor_core=use_tensor_core,
                transpose_b=transpose_b,
                elementwise_lambda_fn=epilogue_fn,
            ](
                test_ctx.c_device.tensor,
                test_ctx.a_device.tensor,
                test_ctx.b_device.tensor,
                ctx,
            )

            var handle = UnsafePointer[cublasContext]()
            check_cublas_error(cublasCreate(UnsafePointer.address_of(handle)))
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
            var c_ref_tensor = test_ctx.c_device_ref.tensor

            @always_inline
            @__copy_capture(c_ref_tensor, epilogue_buff)
            @parameter
            fn func[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
                var idx = rebind[StaticIntTuple[2]](idx0)
                var another_val = epilogue_buff.load[width=simd_width](idx)

                c_ref_tensor.store(
                    idx,
                    c_ref_tensor.load[width=simd_width](idx) * another_val,
                )

            elementwise[func, pack_size, target="cuda"](
                StaticIntTuple[2](M, N),
                ctx,
            )
            _ = epilogue_host^
            _ = epilogue_device^

        print("===> tfloat32-float32 mma with epilogue")

        test_matmul[DType.float32, DimList(3072, 3072)](
            ctx, (999, 3072, 3072)
        ).run_test[
            epilogue_test[
                DType.float32,
                shape = DimList(3072, 3072),
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.float32, DimList(2048, 12288)](
            ctx, (777, 12288, 2048)
        ).run_test[
            epilogue_test[
                DType.float32,
                shape = DimList(2048, 12288),
                use_tensor_core=True,
            ]
        ]()

        print("===> bfloat16-float32 mma with epilogue")

        test_matmul[DType.bfloat16, DimList(12288, 3072), transpose_b=True](
            ctx, (14, 3072, 12288)
        ).run_test[
            epilogue_test[
                DType.bfloat16,
                shape = DimList(12288, 3072),
                transpose_b=True,
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.bfloat16, DimList(3072, 12288), transpose_b=True](
            ctx, (33, 12288, 3072)
        ).run_test[
            epilogue_test[
                DType.bfloat16,
                shape = DimList(3072, 12288),
                transpose_b=True,
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.bfloat16, DimList(3072, 5120), transpose_b=True](
            ctx, (101, 5120, 3072)
        ).run_test[
            epilogue_test[
                DType.bfloat16,
                shape = DimList(3072, 5120),
                transpose_b=True,
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.bfloat16, DimList(32768, 3072), transpose_b=True](
            ctx, (400, 3072, 32768)
        ).run_test[
            epilogue_test[
                DType.bfloat16,
                shape = DimList(32768, 3072),
                transpose_b=True,
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.bfloat16, DimList(3072, 3072), transpose_b=True](
            ctx, (910, 3072, 3072)
        ).run_test[
            epilogue_test[
                DType.bfloat16,
                shape = DimList(3072, 3072),
                transpose_b=True,
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.bfloat16, DimList(4096, 6144), transpose_b=True](
            ctx, (50, 6144, 4096)
        ).run_test[
            epilogue_test[
                DType.bfloat16,
                shape = DimList(4096, 6144),
                transpose_b=True,
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.bfloat16, DimList(4096, 4096), transpose_b=True](
            ctx, (22, 4096, 4096)
        ).run_test[
            epilogue_test[
                DType.bfloat16,
                shape = DimList(4096, 4096),
                transpose_b=True,
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.bfloat16, DimList(4096, 28672), transpose_b=True](
            ctx, (88, 28672, 4096)
        ).run_test[
            epilogue_test[
                DType.bfloat16,
                shape = DimList(4096, 28672),
                transpose_b=True,
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.bfloat16, DimList(14336, 4096), transpose_b=True](
            ctx, (100, 4096, 14336)
        ).run_test[
            epilogue_test[
                DType.bfloat16,
                shape = DimList(14336, 4096),
                transpose_b=True,
                use_tensor_core=True,
            ]
        ]()

        test_matmul[DType.bfloat16, DimList(4096, 128256), transpose_b=True](
            ctx, (600, 128256, 4096)
        ).run_test[
            epilogue_test[
                DType.bfloat16,
                shape = DimList(4096, 128256),
                transpose_b=True,
                use_tensor_core=True,
            ]
        ]()
