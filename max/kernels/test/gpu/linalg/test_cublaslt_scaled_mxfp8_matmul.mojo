# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from math import ceildiv

from buffer import DimList
from gpu.host import DeviceContext
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    random,
    zero,
    fill,
)
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from linalg.matmul.vendor.blas import Backend, Handle, matmul
from internal_utils._utils import ValOrDim, dynamic, static
from buffer import Dim
from _cublas.cublaslt import cublasLtGetVersion, cublasLtMatmulMatrixScale_t
from collections import OptionalReg
from buffer import NDBuffer
from builtin.simd import _convert_f32_to_float8_ue8m0
from layout import Layout, LayoutTensor, IntTuple
from layout._ndbuffer_stub import from_ndbuffer_row_major
from sys import argv
from utils import Index


fn _convert_ref_scales_to_mxfp8_format[
    ref_scales_type: DType,
    mxfp8_scales_type: DType,
    *,
    REF_BLOCK_SCALE: Int,
](
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    ref_a_scales: NDBuffer[ref_scales_type, 2, *_],
    ref_b_scales: NDBuffer[ref_scales_type, 2, *_],
    a_scales: NDBuffer[mut=True, mxfp8_scales_type, 5, *_],
    b_scales: NDBuffer[mut=True, mxfp8_scales_type, 5, *_],
):
    constrained[
        ref_scales_type == DType.float32,
        "Only support float32 reference scales",
    ]()
    constrained[
        mxfp8_scales_type == DType.float8_e8m0fnu,
        "Only support float8_e8m0fnu scales",
    ]()

    var M = m.value
    var N = n.value
    var K = k.value

    comptime SF_VECTOR_SIZE = 32
    comptime atom_m = (32, 4)
    comptime atom_k = 4
    comptime MN_SCALE = atom_m[0] * atom_m[1]
    comptime K_SCALE = SF_VECTOR_SIZE * atom_k

    # initialize a_scales_tensor and b_scales_tensor based on reference scales
    for m in range(M):
        for k in range(K):
            a_scales[
                m // MN_SCALE,
                k // K_SCALE,
                m % atom_m[0],
                (m % MN_SCALE) // atom_m[0],
                k % atom_k,
            ] = rebind[Scalar[mxfp8_scales_type]](
                _convert_f32_to_float8_ue8m0[DType.float8_e8m0fnu](
                    ref_a_scales[k // REF_BLOCK_SCALE, m]
                )
            )

    for n in range(N):
        for k in range(K):
            b_scales[
                n // MN_SCALE,
                k // K_SCALE,
                n % atom_m[0],
                (n % MN_SCALE) // atom_m[0],
                k % atom_k,
            ] = rebind[Scalar[mxfp8_scales_type]](
                _convert_f32_to_float8_ue8m0[DType.float8_e8m0fnu](
                    ref_b_scales[n // REF_BLOCK_SCALE, k // REF_BLOCK_SCALE]
                )
            )


fn test_scaled_mxfp8_cublaslt[
    input_type: DType,
    output_type: DType,
    transpose_b: Bool,
    a_scaling_mode: cublasLtMatmulMatrixScale_t,
    b_scaling_mode: cublasLtMatmulMatrixScale_t,
](
    ctx: DeviceContext,
    handle: Handle,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
) raises:
    constrained[
        transpose_b == True,
        "Only transpose_b = True is supported for scaled FP8 matmul",
    ]()

    var M = m.value
    var N = n.value
    var K = k.value

    var cublaslt_version = cublasLtGetVersion()

    if cublaslt_version < 120901:
        raise Error(
            "This test needs cublasLt version 120901 or higher",
            " cublasLt version: ",
            cublaslt_version,
        )

    if (
        a_scaling_mode == cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0
        and b_scaling_mode
        == cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0
    ):
        print(
            "== test_cublaslt_matmul_scaled_mxfp8 (fp8_scalers)",
            input_type,
            "x",
            M,
            "x",
            N,
            "x",
            K,
            " -- cublasLt version: ",
            cublaslt_version,
        )
    else:
        raise Error("Unknown scaling mode")

    comptime scales_type = DType.float8_e8m0fnu
    comptime ref_scales_type = DType.float32

    # Initialize reference scales
    comptime REF_BLOCK_SCALE = 128
    comptime static_ref_a_scales_shape = DimList(
        ceildiv(Int(k.dim), REF_BLOCK_SCALE), m.dim
    )
    comptime static_ref_b_scales_shape = DimList(
        ceildiv(Int(n.dim), REF_BLOCK_SCALE),
        ceildiv(Int(k.dim), REF_BLOCK_SCALE),
    )

    var dynamic_ref_a_scales_shape = DimList(
        ceildiv(k.value, REF_BLOCK_SCALE), m.value
    )
    var dynamic_ref_b_scales_shape = DimList(
        ceildiv(n.value, REF_BLOCK_SCALE), ceildiv(k.value, REF_BLOCK_SCALE)
    )

    var a_scales_host_ref = HostNDBuffer[
        ref_scales_type, 2, static_ref_a_scales_shape
    ](dynamic_ref_a_scales_shape)
    var b_scales_host_ref = HostNDBuffer[
        ref_scales_type, 2, static_ref_b_scales_shape
    ](dynamic_ref_b_scales_shape)

    var a_scales_device_ref = DeviceNDBuffer[
        ref_scales_type, 2, static_ref_a_scales_shape
    ](dynamic_ref_a_scales_shape, ctx=ctx)
    var b_scales_device_ref = DeviceNDBuffer[
        ref_scales_type, 2, static_ref_b_scales_shape
    ](dynamic_ref_b_scales_shape, ctx=ctx)

    fill(a_scales_host_ref.tensor, Scalar[ref_scales_type](1.0))
    fill(b_scales_host_ref.tensor, Scalar[ref_scales_type](1.0))

    # NOTE: We can't initialize this scales randomly as our naive kernel cannot handle mxfp8 style scaling.
    for i in range(a_scales_host_ref.tensor.dim(0)):
        for j in range(a_scales_host_ref.tensor.dim(1) // 32):
            for k in range(32):
                a_scales_host_ref.tensor[i, j * 32 + k] = 1 << (j % 4)

    for i in range(b_scales_host_ref.tensor.dim(0)):
        for j in range(b_scales_host_ref.tensor.dim(1)):
            b_scales_host_ref.tensor[i, j] = 1 << j

    comptime static_a_shape = DimList(m.dim, k.dim)
    comptime static_b_shape = DimList(n.dim, k.dim)
    comptime static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value)
    var dynamic_c_shape = DimList(m.value, n.value)

    comptime SF_VECTOR_SIZE = 32
    comptime atom_m = (32, 4)
    comptime atom_k = 4
    comptime sf_k = ceildiv(k.dim, SF_VECTOR_SIZE)
    comptime static_a_scales_shape = DimList(
        ceildiv(m.dim, atom_m[0] * atom_m[1]),
        ceildiv(sf_k, atom_k),
        Dim(atom_m[0]),
        Dim(atom_m[1]),
        Dim(atom_k),
    )
    comptime static_b_scales_shape = DimList(
        ceildiv(n.dim, atom_m[0] * atom_m[1]),
        ceildiv(sf_k, atom_k),
        Dim(atom_m[0]),
        Dim(atom_m[1]),
        Dim(atom_k),
    )

    var dynamic_a_scales_shape = DimList(
        ceildiv(m.value, atom_m[0] * atom_m[1]),
        ceildiv(sf_k, atom_k),
        Dim(atom_m[0]),
        Dim(atom_m[1]),
        Dim(atom_k),
    )
    var dynamic_b_scales_shape = DimList(
        ceildiv(n.value, atom_m[0] * atom_m[1]),
        ceildiv(sf_k, atom_k),
        Dim(atom_m[0]),
        Dim(atom_m[1]),
        Dim(atom_k),
    )

    var a_scales_host = HostNDBuffer[scales_type, 5, static_a_scales_shape](
        dynamic_a_scales_shape
    )
    var b_scales_host = HostNDBuffer[scales_type, 5, static_b_scales_shape](
        dynamic_b_scales_shape
    )

    var a_scales_device = DeviceNDBuffer[scales_type, 5, static_a_scales_shape](
        dynamic_a_scales_shape, ctx=ctx
    )
    var b_scales_device = DeviceNDBuffer[scales_type, 5, static_b_scales_shape](
        dynamic_b_scales_shape, ctx=ctx
    )

    var a_host = HostNDBuffer[input_type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[input_type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[output_type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[output_type, 2, static_c_shape](
        dynamic_c_shape
    )

    var a_device = DeviceNDBuffer[input_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[input_type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[output_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[output_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    _convert_ref_scales_to_mxfp8_format[REF_BLOCK_SCALE=REF_BLOCK_SCALE](
        m,
        n,
        k,
        a_scales_host_ref.tensor,
        b_scales_host_ref.tensor,
        a_scales_host.tensor,
        b_scales_host.tensor,
    )

    random(a_host.tensor)
    random(b_host.tensor)

    # Move operands to the Device
    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)
    ctx.enqueue_copy(a_scales_device_ref.buffer, a_scales_host_ref.tensor.data)
    ctx.enqueue_copy(b_scales_device_ref.buffer, b_scales_host_ref.tensor.data)
    ctx.enqueue_copy(a_scales_device.buffer, a_scales_host.tensor.data)
    ctx.enqueue_copy(b_scales_device.buffer, b_scales_host.tensor.data)

    var a = from_ndbuffer_row_major(a_device.tensor)
    var b = from_ndbuffer_row_major(b_device.tensor)
    var c = from_ndbuffer_row_major(c_device.tensor)
    var a_scales = from_ndbuffer_row_major(a_scales_device.tensor)
    var b_scales = from_ndbuffer_row_major(b_scales_device.tensor)

    matmul[scales_type=scales_type](
        ctx,
        c,
        a,
        b,
        a_scales=a_scales,
        b_scales=b_scales,
        transpose_b=True,
        c_row_major=True,
    )

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)

    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=transpose_b,
        scales_granularity_mnk = Index(1, REF_BLOCK_SCALE, REF_BLOCK_SCALE),
    ](
        c_device_ref.tensor,
        a_device.tensor,
        b_device.tensor,
        a_scales_device_ref.tensor,
        b_scales_device_ref.tensor,
        ctx,
    )

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)

    ctx.synchronize()

    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=0.01,
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
    _ = a_scales
    _ = b_scales
    _ = a_scales_host
    _ = b_scales_host
    _ = a_scales_device
    _ = b_scales_device
    _ = a_scales_host_ref
    _ = b_scales_host_ref


fn main() raises:
    with DeviceContext() as ctx, Handle[Backend.CUBLASLT]() as handle:
        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
        ](ctx, handle, dynamic(128), static[128](), static[128]())

        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
        ](ctx, handle, dynamic(256), static[256](), static[256]())

        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
        ](ctx, handle, dynamic(128), static[3 * 128](), static[256]())

        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
        ](ctx, handle, dynamic(3 * 128), static[128](), static[3 * 128]())

        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
        ](ctx, handle, dynamic(2560), static[4096](), static[1024]())

        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
        ](ctx, handle, dynamic(1000), static[4096](), static[1024]())

        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
        ](ctx, handle, dynamic(1000), static[4096 + 64](), static[1024]())

        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
            cublasLtMatmulMatrixScale_t.MATRIX_SCALE_VEC32_UE8M0,
        ](ctx, handle, dynamic(1000), static[4096 + 64](), static[1024 + 64]())
