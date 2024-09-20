# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
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
from memory import memset_zero, stack_allocation, UnsafePointer
from memory.reference import _GPUAddressSpace as GPUAddressSpace
from testing import assert_equal as assert_equal_val

from utils import StaticIntTuple
from internal_utils import linspace
from linalg.utils import elementwise_epilogue_type
from utils.index import Index

alias init_fn_type = fn (buff: NDBuffer) capturing -> None

alias epilogue_func_type = fn[type: DType, width: Int, *, alignment: Int = 1] (
    StaticIntTuple[2], StaticIntTuple[2], SIMD[type, width]
) capturing -> SIMD[type, width]


@parameter
@always_inline
fn epilogue_test_fn[
    type: DType, width: Int, *, alignment: Int = 1
](
    idx: StaticIntTuple[2],
    dim_space: StaticIntTuple[2],
    val: SIMD[type, width],
) -> SIMD[type, width]:
    var bias = SIMD[type, width](0)

    @parameter
    for i in range(width):
        bias[i] = (
            0.5
            + ((idx[0] + idx[1] + i) / (dim_space[0] + dim_space[1])).cast[
                type
            ]()
        )

    return val + bias


fn test[
    type: DType,
    static_MNK: DimList = DimList.create_unknown[3](),
    /,
    *,
    transpose_b: Bool = False,
    init_a: Optional[init_fn_type] = None,
    init_b: Optional[init_fn_type] = None,
    lambda_fn: Optional[epilogue_func_type] = None,
](ctx: DeviceContext, dim3: StaticIntTuple[3]) raises:
    constrained[
        static_MNK.has_value[1]() and static_MNK.has_value[2](),
        "This test currently requires static N and K.",
    ]()

    @parameter
    if static_MNK.all_known[3]():
        assert_equal_val(static_MNK.at[0](), dim3[0])
        assert_equal_val(static_MNK.at[1](), dim3[1])
        assert_equal_val(static_MNK.at[2](), dim3[2])
    else:
        assert_equal_val(static_MNK.at[1](), dim3[1])
        assert_equal_val(static_MNK.at[2](), dim3[2])

    var M = dim3[0]
    alias N = static_MNK.at[1]()
    alias K = static_MNK.at[2]()

    alias static_a_shape = DimList(Dim(), K)
    alias static_b_shape = DimList(N, K) if transpose_b else DimList(K, N)
    alias static_c_shape = DimList(Dim(), N)
    var dynamic_a_shape = DimList(M, K)
    var dynamic_b_shape = DimList(N, K) if transpose_b else DimList(K, N)
    var dynamic_c_shape = DimList(M, N)

    var a_host = HostNDBuffer[type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[type, 2, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    # Initialize matmul operands
    @parameter
    if init_a:
        alias init_a_fn = init_a.value()
        init_a_fn(a_host.tensor)
    else:
        random(a_host.tensor)

    @parameter
    if init_b:
        alias init_b_fn = init_b.value()
        init_b_fn(b_host.tensor)
    else:
        random(b_host.tensor)

    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    # Move operands to the Device

    ctx.enqueue_copy_to_device(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy_to_device(b_device.buffer, b_host.tensor.data)

    ctx.enqueue_copy_to_device(c_device.buffer, c_host.tensor.data)
    ctx.enqueue_copy_to_device(c_device_ref.buffer, c_host_ref.tensor.data)

    var c_tensor = c_device.tensor

    @parameter
    @always_inline
    @__copy_capture(c_tensor, M)
    fn epilogue_fn[
        _type: DType, width: Int, *, alignment: Int = 1
    ](idx: StaticIntTuple[2], val: SIMD[_type, width]) capturing -> None:
        var update_val: SIMD[_type, width] = val

        @parameter
        if lambda_fn:
            alias func = lambda_fn.value()
            update_val = func(idx, (M, int(N)), update_val)
        c_tensor.store(idx, rebind[SIMD[type, width]](update_val))

    @parameter
    if lambda_fn:
        _matmul_gpu[
            use_tensor_core=True,
            transpose_b=transpose_b,
            elementwise_lambda_fn=epilogue_fn,
        ](
            c_device.tensor,
            a_device.tensor,
            b_device.tensor,
            ctx,
        )
    else:
        _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b,](
            c_device.tensor,
            a_device.tensor,
            b_device.tensor,
            ctx,
        )

    ctx.synchronize()

    var handle = UnsafePointer[cublasContext]()
    check_cublas_error(cublasCreate(UnsafePointer.address_of(handle)))
    check_cublas_error(
        cublas_matmul(
            handle,
            c_device_ref.tensor,
            a_device.tensor,
            b_device.tensor,
            c_row_major=True,
            transpose_b=transpose_b,
        )
    )
    check_cublas_error(cublasDestroy(handle))

    var c_ref_tensor = c_device_ref.tensor
    alias pack_size = simdwidthof[type, target = _get_nvptx_target()]()

    @always_inline
    @__copy_capture(c_ref_tensor, M)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: StaticIntTuple[rank]):
        var idx = rebind[StaticIntTuple[2]](idx0)

        var val = c_ref_tensor.load[width=simd_width](idx)

        var update_val = val

        @parameter
        if lambda_fn:
            alias element_lambda = lambda_fn.value()
            update_val = element_lambda(idx, (M, int(N)), val)

        c_ref_tensor.store(
            idx,
            update_val,
        )

    @parameter
    if lambda_fn:
        elementwise[func, pack_size, target="cuda"](
            StaticIntTuple[2](M, int(N)),
            ctx,
        )
    ctx.synchronize()

    ctx.enqueue_copy_from_device(c_host.tensor.data, c_device.buffer)
    ctx.enqueue_copy_from_device(c_host_ref.tensor.data, c_device_ref.buffer)
    ctx.synchronize()

    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=0.0001,
        rtol=0.02,
    )
    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device


def main():
    with DeviceContext() as ctx:
        print("===> tfloat32-float32 mma")
        test[DType.float32, DimList(Dim(), 384, 128), init_a=linspace](
            ctx, (256, 384, 128)
        )
        test[DType.float32, DimList(Dim(), 4096, 4096), init_b=linspace](
            ctx, (128, 4096, 4096)
        )
        test[
            DType.float32,
            DimList(Dim(), 12288, 4096),
            init_a=linspace,
            init_b=linspace,
        ](ctx, (512, 12288, 4096))
        test[DType.float32, DimList(Dim(), 4096, 11008)](ctx, (23, 4096, 11008))
        test[DType.float32, DimList(Dim(), 4096, 12288)](ctx, (67, 4096, 12288))
        test[DType.float32, DimList(Dim(), 4096, 4096)](ctx, (555, 4096, 4096))

        print("===> bfloat16-float32 mma")
        test[DType.bfloat16, DimList(Dim(), 3072, 12288), init_a=linspace](
            ctx, (1024, 3072, 12288)
        )
        test[DType.bfloat16, DimList(Dim(), 12288, 3072), init_b=linspace](
            ctx, (11379, 12288, 3072)
        )
        test[
            DType.bfloat16,
            DimList(Dim(), 5120, 3072),
            init_a=linspace,
            init_b=linspace,
        ](ctx, (9127, 5120, 3072))
        test[DType.bfloat16, DimList(Dim(), 3072, 32768)](
            ctx, (1171, 3072, 32768)
        )
        test[DType.bfloat16, DimList(Dim(), 3072, 3072)](
            ctx, (16315, 3072, 3072)
        )

        print("===> tfloat32-float32 mma with epilogue")
        test[
            DType.float32,
            DimList(Dim(), 3072, 3072),
            lambda_fn=epilogue_test_fn,
        ](ctx, (999, 3072, 3072))
        test[
            DType.float32,
            DimList(Dim(), 12288, 2048),
            lambda_fn=epilogue_test_fn,
        ](ctx, (777, 12288, 2048))
        print("===> bfloat16-float32 mma with epilogue")
        test[
            DType.bfloat16,
            DimList(Dim(), 3072, 12288),
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, (14, 3072, 12288))
        test[
            DType.bfloat16,
            DimList(Dim(), 12288, 3072),
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, (33, 12288, 3072))
        test[
            DType.bfloat16,
            DimList(Dim(), 5120, 3072),
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, (101, 5120, 3072))
        test[
            DType.bfloat16,
            DimList(Dim(), 3072, 32768),
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, (400, 3072, 32768))
        test[
            DType.bfloat16,
            DimList(Dim(), 3072, 3072),
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, (910, 3072, 3072))
        test[
            DType.bfloat16,
            DimList(Dim(), 6144, 4096),
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, (50, 6144, 4096))
        test[
            DType.bfloat16,
            DimList(Dim(), 4096, 4096),
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, (22, 4096, 4096))
        test[
            DType.bfloat16,
            DimList(Dim(), 28672, 4096),
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, (88, 28672, 4096))
        test[
            DType.bfloat16,
            DimList(Dim(), 4096, 14336),
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, (100, 4096, 14336))
        test[
            DType.bfloat16,
            DimList(Dim(), 128256, 4096),
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
        ](ctx, (600, 128256, 4096))
