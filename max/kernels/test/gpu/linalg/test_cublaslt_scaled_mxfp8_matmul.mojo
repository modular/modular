# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from math import ceildiv, align_up
from random import random_ui64
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from internal_utils import assert_almost_equal
from random import rand
from linalg.matmul.vendor.blas import Backend, Handle, matmul
from internal_utils._utils import ValOrDim, dynamic, static
from _cublas.cublaslt import cublasLtGetVersion, cublasLtMatmulMatrixScale_t
from collections import OptionalReg
from layout import Layout, LayoutTensor, IntTuple
from layout._ndbuffer_stub import from_ndbuffer_row_major
from sys import argv
from utils import Index
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
    # convert_ref_scales_to_mxfp8_format,
    set_scale_factor,
)
from linalg.fp4_quantization import naive_block_scaled_matmul
from gpu.compute.arch.mma_nvidia_sm100 import UMMAKind


fn test_scaled_mxfp8_cublaslt[
    input_type: DType,
    output_type: DType,
    transpose_b: Bool,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim,) raises:
    comptime assert (
        transpose_b == True
    ), "Only transpose_b = True is supported for scaled FP8 matmul"

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

    comptime scales_type = MXFP8_SF_DTYPE

    print(
        String(
            "in/out dtypes=(",
            input_type,
            ", ",
            output_type,
            ", ",
            scales_type,
            ") ",
            " problem shape=(",
            M,
            ", ",
            N,
            ", ",
            K,
            ") ",
        )
    )

    comptime static_a_shape = DimList(m.dim, k.dim)
    comptime static_b_shape = DimList(n.dim, k.dim)
    comptime static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value)
    var dynamic_c_shape = DimList(m.value, n.value)

    comptime static_a_scales_shape = DimList(
        ceildiv(m.dim, SF_MN_GROUP_SIZE),
        ceildiv(k.dim, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )
    comptime static_b_scales_shape = DimList(
        ceildiv(n.dim, SF_MN_GROUP_SIZE),
        ceildiv(k.dim, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var dynamic_a_scales_shape = DimList(
        ceildiv(m.value, SF_MN_GROUP_SIZE),
        ceildiv(k.value, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )
    var dynamic_b_scales_shape = DimList(
        ceildiv(n.value, SF_MN_GROUP_SIZE),
        ceildiv(k.value, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var a_scales_size = (
        ceildiv(m.value, SF_MN_GROUP_SIZE)
        * ceildiv(k.value, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )
    var b_scales_size = (
        ceildiv(n.value, SF_MN_GROUP_SIZE)
        * ceildiv(k.value, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )

    var a_scales_host_ptr = UnsafePointer[Scalar[scales_type]].alloc(
        a_scales_size
    )
    var a_scales_host = NDBuffer[scales_type, 5, _, static_a_scales_shape](
        a_scales_host_ptr, dynamic_a_scales_shape
    )
    var b_scales_host_ptr = UnsafePointer[Scalar[scales_type]].alloc(
        b_scales_size
    )
    var b_scales_host = NDBuffer[scales_type, 5, _, static_b_scales_shape](
        b_scales_host_ptr, dynamic_b_scales_shape
    )

    var a_scales_device = ctx.enqueue_create_buffer[scales_type](a_scales_size)
    var a_scales_device_nd = NDBuffer[scales_type, 5, _, static_a_scales_shape](
        a_scales_device.unsafe_ptr(), dynamic_a_scales_shape
    )
    var b_scales_device = ctx.enqueue_create_buffer[scales_type](b_scales_size)
    var b_scales_device_nd = NDBuffer[scales_type, 5, _, static_b_scales_shape](
        b_scales_device.unsafe_ptr(), dynamic_b_scales_shape
    )

    var a_size = m.value * k.value
    var b_size = n.value * k.value
    var c_size = m.value * n.value

    var a_host_ptr = UnsafePointer[Scalar[input_type]].alloc(a_size)
    var a_host = NDBuffer[input_type, 2, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host_ptr = UnsafePointer[Scalar[input_type]].alloc(b_size)
    var b_host = NDBuffer[input_type, 2, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )
    var c_host_ptr = UnsafePointer[Scalar[output_type]].alloc(c_size)
    var c_host = NDBuffer[output_type, 2, _, static_c_shape](
        c_host_ptr, dynamic_c_shape
    )
    var c_host_ref_ptr = UnsafePointer[Scalar[output_type]].alloc(c_size)
    var c_host_ref = NDBuffer[output_type, 2, _, static_c_shape](
        c_host_ref_ptr, dynamic_c_shape
    )

    var a_device = ctx.enqueue_create_buffer[input_type](a_size)
    var a_device_nd = NDBuffer[input_type, 2, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var b_device = ctx.enqueue_create_buffer[input_type](b_size)
    var b_device_nd = NDBuffer[input_type, 2, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var c_device = ctx.enqueue_create_buffer[output_type](c_size)
    var c_device_nd = NDBuffer[output_type, 2, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )
    var c_device_ref = ctx.enqueue_create_buffer[output_type](c_size)
    var c_device_ref_nd = NDBuffer[output_type, 2, _, static_c_shape](
        c_device_ref.unsafe_ptr(), dynamic_c_shape
    )

    var a_scales_tensor_host = from_ndbuffer_row_major(
        a_scales_host
    ).as_any_origin()
    # NOTE: It is very important that we set unused scales to 0.0 otherwise we will hit accuracy issues
    for idx0 in range(align_up(m.value, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0,
            align_up(k.value, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K),
            MXFP8_SF_VECTOR_SIZE,
        ):
            if idx0 < m.value and idx1 < k.value:
                var scale_value = (
                    (1 << random_ui64(0, 3))
                    .cast[DType.float32]()
                    .cast[scales_type]()
                )
                set_scale_factor[SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE](
                    a_scales_tensor_host, idx0, idx1, scale_value
                )
            else:
                set_scale_factor[SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE](
                    a_scales_tensor_host, idx0, idx1, Scalar[scales_type](0.0)
                )
    var b_scales_tensor_host = from_ndbuffer_row_major(
        b_scales_host
    ).as_any_origin()
    for idx0 in range(align_up(n.value, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0,
            align_up(k.value, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K),
            MXFP8_SF_VECTOR_SIZE,
        ):
            if idx0 < n.value and idx1 < k.value:
                var scale_value = (
                    (1 << random_ui64(0, 3))
                    .cast[DType.float32]()
                    .cast[scales_type]()
                )
                set_scale_factor[SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE](
                    b_scales_tensor_host, idx0, idx1, scale_value
                )
            else:
                set_scale_factor[SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE](
                    b_scales_tensor_host, idx0, idx1, Scalar[scales_type](0.0)
                )

    rand(a_host.data, a_host.num_elements())
    rand(b_host.data, b_host.num_elements())

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

    matmul(
        ctx,
        c,
        a,
        b,
        a_scales=a_scales.get_immutable(),
        b_scales=b_scales.get_immutable(),
        transpose_b=True,
        c_row_major=True,
    )

    ctx.enqueue_copy(c_host_ptr, c_device)

    var c_ref = from_ndbuffer_row_major(c_device_ref_nd)
    naive_block_scaled_matmul[
        scaling_kind = UMMAKind.KIND_MXF8F6F4,
        SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE,
        transpose_b=transpose_b,
    ](
        c_ref,
        a,
        b,
        a_scales,
        b_scales,
        ctx,
    )

    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)

    ctx.synchronize()

    assert_almost_equal(
        c_host.data,
        c_host_ref.data,
        c_host.num_elements(),
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
        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
        ](ctx, dynamic(128), static[128](), static[128]())

        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
        ](ctx, dynamic(256), static[256](), static[256]())

        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
        ](ctx, dynamic(128), static[3 * 128](), static[256]())

        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
        ](ctx, dynamic(3 * 128), static[128](), static[3 * 128]())

        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
        ](ctx, dynamic(2560), static[4096](), static[1024]())

        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
        ](ctx, dynamic(1000), static[4096](), static[1024]())

        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
        ](ctx, dynamic(1000), static[4096 + 64](), static[1024]())

        test_scaled_mxfp8_cublaslt[
            DType.float8_e4m3fn,
            DType.bfloat16,
            True,
        ](ctx, dynamic(1000), static[4096 + 64](), static[1024 + 64]())
