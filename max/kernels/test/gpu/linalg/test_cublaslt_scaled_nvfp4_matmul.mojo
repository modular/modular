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

    var a_scales_host = HostNDBuffer[scales_dtype, 5, static_a_scales_shape](
        dynamic_a_scales_shape
    )
    var b_scales_host = HostNDBuffer[scales_dtype, 5, static_b_scales_shape](
        dynamic_b_scales_shape
    )

    random(
        a_scales_host.tensor,
    )
    random(
        b_scales_host.tensor,
    )

    var a_scales_device = DeviceNDBuffer[
        scales_dtype, 5, static_a_scales_shape
    ](dynamic_a_scales_shape, ctx=ctx)
    var b_scales_device = DeviceNDBuffer[
        scales_dtype, 5, static_b_scales_shape
    ](dynamic_b_scales_shape, ctx=ctx)

    var a_host = HostNDBuffer[input_dtype, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[input_dtype, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[out_dtype, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[out_dtype, 2, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[input_dtype, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[input_dtype, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[out_dtype, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[out_dtype, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    random(a_host.tensor, min=0, max=255)
    random(b_host.tensor, min=0, max=255)

    # Move operands to the Device
    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)
    ctx.enqueue_copy(a_scales_device.buffer, a_scales_host.tensor.data)
    ctx.enqueue_copy(b_scales_device.buffer, b_scales_host.tensor.data)

    var a = from_ndbuffer_row_major(a_device.tensor)
    var b = from_ndbuffer_row_major(b_device.tensor)
    var c = from_ndbuffer_row_major(c_device.tensor)
    var a_scales = from_ndbuffer_row_major(a_scales_device.tensor)
    var b_scales = from_ndbuffer_row_major(b_scales_device.tensor)

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

    var c_ref = from_ndbuffer_row_major(c_device_ref.tensor)
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
