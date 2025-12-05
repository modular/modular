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

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from memory import LegacyUnsafePointer as UnsafePointer
from internal_utils import assert_almost_equal, random, zero, fill
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from linalg.matmul.vendor.blas import Backend, Handle, matmul
from internal_utils._utils import ValOrDim, dynamic, static
from _cublas.cublaslt import cublasLtGetVersion, cublasLtMatmulMatrixScale_t
from collections import OptionalReg
from builtin.simd import _convert_f32_to_float8_ue8m0
from layout import Layout, LayoutTensor, IntTuple
from layout._ndbuffer_stub import from_ndbuffer_row_major
from sys import argv
from utils import Index
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
)
from linalg.fp4_quantization import naive_block_scaled_nvfp4_matmul


fn test_block_scaled_nvfp4_cublaslt[
    out_dtype: DType,
    in_dtype: DType,
    transpose_b: Bool,
](
    ctx: DeviceContext,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    tensor_sf: Float32 = 1.0,
) raises:
    constrained[
        transpose_b == True,
        "Only transpose_b = True is supported for scaled NVFP4 matmul",
    ]()

    constrained[
        in_dtype == DType.float4_e2m1fn
        and out_dtype in (DType.float32, DType.bfloat16),
        (
            "Only float4-e2m1fn input type and float32 or bfloat16 output type"
            " are supported for NVFP4."
        ),
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

    alias scales_dtype = NVFP4_SF_DTYPE
    # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
    # Replace this with float4-e2m1fn when GENAI-337 is fixed.
    alias input_dtype = DType.uint8

    alias static_a_shape = DimList(m.dim, k.dim // 2)
    alias static_b_shape = DimList(n.dim, k.dim // 2)
    alias static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value // 2)
    var dynamic_b_shape = DimList(n.value, k.value // 2)
    var dynamic_c_shape = DimList(m.value, n.value)

    alias static_a_scales_shape = DimList(
        ceildiv(m.dim, SF_MN_GROUP_SIZE),
        ceildiv(k.dim, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )
    alias static_b_scales_shape = DimList(
        ceildiv(n.dim, SF_MN_GROUP_SIZE),
        ceildiv(k.dim, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var dynamic_a_scales_shape = DimList(
        ceildiv(m.value, SF_MN_GROUP_SIZE),
        ceildiv(k.value, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )
    var dynamic_b_scales_shape = DimList(
        ceildiv(n.value, SF_MN_GROUP_SIZE),
        ceildiv(k.value, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var a_scales_size = (
        ceildiv(m.value, SF_MN_GROUP_SIZE)
        * ceildiv(k.value, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )
    var b_scales_size = (
        ceildiv(n.value, SF_MN_GROUP_SIZE)
        * ceildiv(k.value, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )

    var a_scales_host_ptr = UnsafePointer[Scalar[scales_dtype]].alloc(
        a_scales_size
    )
    var a_scales_host = NDBuffer[scales_dtype, 5, _, static_a_scales_shape](
        a_scales_host_ptr, dynamic_a_scales_shape
    )
    var b_scales_host_ptr = UnsafePointer[Scalar[scales_dtype]].alloc(
        b_scales_size
    )
    var b_scales_host = NDBuffer[scales_dtype, 5, _, static_b_scales_shape](
        b_scales_host_ptr, dynamic_b_scales_shape
    )

    random(a_scales_host)
    random(b_scales_host)

    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](a_scales_size)
    var a_scales_device_nd = NDBuffer[
        scales_dtype, 5, _, static_a_scales_shape
    ](a_scales_device.unsafe_ptr(), dynamic_a_scales_shape)
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](b_scales_size)
    var b_scales_device_nd = NDBuffer[
        scales_dtype, 5, _, static_b_scales_shape
    ](b_scales_device.unsafe_ptr(), dynamic_b_scales_shape)

    var a_size = m.value * (k.value // 2)
    var b_size = n.value * (k.value // 2)
    var c_size = m.value * n.value

    var a_host_ptr = UnsafePointer[Scalar[input_dtype]].alloc(a_size)
    var a_host = NDBuffer[input_dtype, 2, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host_ptr = UnsafePointer[Scalar[input_dtype]].alloc(b_size)
    var b_host = NDBuffer[input_dtype, 2, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )
    var c_host_ptr = UnsafePointer[Scalar[out_dtype]].alloc(c_size)
    var c_host = NDBuffer[out_dtype, 2, _, static_c_shape](
        c_host_ptr, dynamic_c_shape
    )
    var c_host_ref_ptr = UnsafePointer[Scalar[out_dtype]].alloc(c_size)
    var c_host_ref = NDBuffer[out_dtype, 2, _, static_c_shape](
        c_host_ref_ptr, dynamic_c_shape
    )

    var a_device = ctx.enqueue_create_buffer[input_dtype](a_size)
    var a_device_nd = NDBuffer[input_dtype, 2, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var b_device = ctx.enqueue_create_buffer[input_dtype](b_size)
    var b_device_nd = NDBuffer[input_dtype, 2, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var c_device = ctx.enqueue_create_buffer[out_dtype](c_size)
    var c_device_nd = NDBuffer[out_dtype, 2, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )
    var c_device_ref = ctx.enqueue_create_buffer[out_dtype](c_size)
    var c_device_ref_nd = NDBuffer[out_dtype, 2, _, static_c_shape](
        c_device_ref.unsafe_ptr(), dynamic_c_shape
    )

    random(a_host, min=0, max=255)
    random(b_host, min=0, max=255)

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    var a = from_ndbuffer_row_major(a_device_nd)
    var b = from_ndbuffer_row_major(b_device_nd)
    var c = from_ndbuffer_row_major(c_device_nd)
    var a_scales = from_ndbuffer_row_major(a_scales_device_nd)
    var b_scales = from_ndbuffer_row_major(b_scales_device_nd)

    matmul[scales_type=scales_dtype](
        ctx,
        c,
        a,
        b,
        a_scales=a_scales,
        b_scales=b_scales,
        transpose_b=True,
        c_row_major=True,
    )

    var c_ref = from_ndbuffer_row_major(c_device_ref_nd)
    naive_block_scaled_nvfp4_matmul[
        SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
        transpose_b=transpose_b,
    ](
        c_ref,
        a,
        b,
        a_scales,
        b_scales,
        ctx,
    )

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)

    ctx.synchronize()

    assert_almost_equal(
        c_host,
        c_host_ref,
        atol=0.01,
        rtol=0.01,
    )

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^

    _ = a_scales
    _ = b_scales


fn main() raises:
    with DeviceContext() as ctx:
        test_block_scaled_nvfp4_cublaslt[
            DType.bfloat16,
            DType.float4_e2m1fn,
            True,
        ](ctx, dynamic(128), static[128](), static[64]())

        test_block_scaled_nvfp4_cublaslt[
            DType.bfloat16,
            DType.float4_e2m1fn,
            True,
        ](ctx, dynamic(256), static[256](), static[64 - 32]())

        test_block_scaled_nvfp4_cublaslt[
            DType.bfloat16,
            DType.float4_e2m1fn,
            True,
        ](ctx, dynamic(128), static[3 * 128](), static[256 + 32]())

        test_block_scaled_nvfp4_cublaslt[
            DType.bfloat16,
            DType.float4_e2m1fn,
            True,
        ](ctx, dynamic(3 * 128), static[128](), static[3 * 64]())

        test_block_scaled_nvfp4_cublaslt[
            DType.bfloat16,
            DType.float4_e2m1fn,
            True,
        ](ctx, dynamic(2560), static[4096](), static[1024]())

        test_block_scaled_nvfp4_cublaslt[
            DType.bfloat16,
            DType.float4_e2m1fn,
            True,
        ](ctx, dynamic(1000), static[4096](), static[1024]())

        test_block_scaled_nvfp4_cublaslt[
            DType.bfloat16,
            DType.float4_e2m1fn,
            True,
        ](ctx, dynamic(1000), static[4096 + 64](), static[1024]())

        test_block_scaled_nvfp4_cublaslt[
            DType.bfloat16,
            DType.float4_e2m1fn,
            True,
        ](ctx, dynamic(1000), static[4096 + 64](), static[1024 + 64]())
