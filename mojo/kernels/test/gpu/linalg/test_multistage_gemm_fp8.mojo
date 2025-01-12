# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1480
# UNSUPPORTED: NVIDIA-GPU
# RUN: %mojo-no-debug %s

from collections.optional import OptionalReg
from math import ceildiv, isclose
from pathlib import Path
from random import rand
from sys import alignof, argv, simdwidthof
from buffer import NDBuffer
from buffer.dimlist import DimList, Dim
from gpu import WARP_SIZE, block_idx, grid_dim, thread_idx, barrier, lane_id
from gpu.host import DeviceContext, FuncAttribute
from gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_group,
    external_memory,
)
from gpu.mma import ld_matrix, mma
from layout.int_tuple import IntTuple
from layout.layout import *
from layout import RuntimeLayout
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    _swizzle_signature,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_sram,
    copy_sram_to_dram,
    copy_local_to_local,
)
from layout.tensor_core import (
    TensorCore,
    get_accum_type,
    get_fragment_size,
    get_mma_shape,
)
from linalg.utils_gpu import block_swizzle
from memory import UnsafePointer
import linalg.vendor_blas
from utils.index import Index, IndexList
from internal_utils._utils import ValOrDim, dynamic, static

from layout.nd_buffer_stub import from_ndbuffer_row_major
from layout.swizzle import Swizzle, make_swizzle
from linalg.utils_gpu import (
    block_swizzle,
    MatmulConfig,
    MatmulKernels,
    select_config,
)
from layout.tensor_builder import LayoutTensorBuild as tb

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

from linalg._multistage_gemm_gpu import multistage_gemm_kernel
from gpu.host._compile import _get_gpu_target


fn test_fp8_multistage_gemm[
    type: DType,
    M: Int,
    N: Int,
    K: Int,
    /,
    *,
    transpose_b: Bool = False,
](ctx: DeviceContext, handle: vendor_blas.Handle) raises:
    print("test fp8 multistage matmul")

    alias static_a_shape = DimList(M, K)
    alias static_b_shape = DimList(N, K) if transpose_b else DimList(K, N)
    alias static_c_shape = DimList(M, N)

    var a_host = HostNDBuffer[type, 2, static_a_shape]()
    var b_host = HostNDBuffer[type, 2, static_b_shape]()
    var c_host = HostNDBuffer[DType.float32, 2, static_c_shape]()
    var c_host_ref = HostNDBuffer[DType.float32, 2, static_c_shape]()

    @parameter
    for i in range(M):

        @parameter
        for j in range(K):
            a_host.tensor[i, j] = i + j

    @parameter
    for i in range(static_b_shape.get[0]()):

        @parameter
        for j in range(static_b_shape.get[1]()):
            b_host.tensor[i, j] = i + j

    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    var a_device = DeviceNDBuffer[type, 2, static_a_shape](ctx=ctx)
    var b_device = DeviceNDBuffer[type, 2, static_b_shape](ctx=ctx)
    var c_device = DeviceNDBuffer[DType.float32, 2, static_c_shape](ctx=ctx)
    var c_device_ref = DeviceNDBuffer[DType.float32, 2, static_c_shape](ctx=ctx)

    ctx.enqueue_copy_to_device(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy_to_device(b_device.buffer, b_host.tensor.data)

    var c_tensor = from_ndbuffer_row_major(c_device.tensor)
    var a_tensor = from_ndbuffer_row_major(a_device.tensor)
    var b_tensor = from_ndbuffer_row_major(b_device.tensor)

    alias kernels = MatmulKernels[type, type, DType.float32, transpose_b]()
    alias config = kernels.hopper_128x128_4

    alias multistage_gemm_kernel_fn = multistage_gemm_kernel[
        DType.float32,  # c_type
        c_tensor.layout,
        type,  # a_type
        a_tensor.layout,
        type,  # b_type
        b_tensor.layout,
        transpose_b,
        config,
    ]

    var multistage_gemm_kernel_compiled = ctx.compile_function[
        multistage_gemm_kernel_fn,
        _target = _get_gpu_target["sm_90"](),
        # dump_llvm=Path("./pipeline-gemm.ir"),
        # dump_asm=Path("./pipeline-gemm.ptx"),
    ](
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            config.shared_mem_usage()
        ),
    )

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]

    ctx.enqueue_function(
        multistage_gemm_kernel_compiled,
        c_tensor,
        a_tensor,
        b_tensor,
        UnsafePointer[Scalar[DType.int32]](),
        grid_dim=config.grid_dim(M, N),
        block_dim=config.block_dim(),
        shared_mem_bytes=config.shared_mem_usage(),
    )

    ctx.enqueue_copy_from_device(c_host.tensor.data, c_device.buffer)

    if transpose_b:
        vendor_blas.matmul(
            ctx,
            handle,
            c_device_ref.tensor,
            a_device.tensor,
            b_device.tensor,
            c_row_major=True,
        )

    else:
        # TODO: Matrix B should always be in col-major layout for cublasLt to work
        var b_host_col_major = HostNDBuffer[type, 2, DimList(N, K)]()

        for i in range(N):
            for j in range(K):
                b_host_col_major.tensor[i, j] = b_host.tensor[j, i]

        var b_device_col_major = DeviceNDBuffer[type, 2, DimList(N, K)](ctx=ctx)
        ctx.enqueue_copy_to_device(
            b_device_col_major.buffer, b_host_col_major.tensor.data
        )

        vendor_blas.matmul(
            ctx,
            handle,
            c_device_ref.tensor,
            a_device.tensor,
            b_device_col_major.tensor,
            c_row_major=True,
        )

    ctx.enqueue_copy_from_device(c_host_ref.tensor.data, c_device_ref.buffer)

    ctx.synchronize()

    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=0.0001,
        rtol=0.01,
    )

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device

    _ = a_tensor
    _ = b_tensor
    _ = c_tensor

    _ = multistage_gemm_kernel_compiled^


def main():
    with DeviceContext() as ctx:
        with vendor_blas.Handle[vendor_blas.Backend.CUBLASLT]() as handle:
            test_fp8_multistage_gemm[
                DType.float8e4m3, 128, 128, 64, transpose_b=True
            ](ctx, handle)
            test_fp8_multistage_gemm[
                DType.float8e4m3, 128, 128, 128, transpose_b=True
            ](ctx, handle)
            test_fp8_multistage_gemm[
                DType.float8e4m3, 128, 128, 64, transpose_b=False
            ](ctx, handle)
            test_fp8_multistage_gemm[
                DType.float8e4m3, 128, 128, 128, transpose_b=False
            ](ctx, handle)
