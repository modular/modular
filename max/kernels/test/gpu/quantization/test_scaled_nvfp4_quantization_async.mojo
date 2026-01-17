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

from sys import size_of, align_of
from math import ceildiv
from gpu import barrier
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu import block_idx, grid_dim, thread_idx, lane_id
from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout, RuntimeTuple
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.swizzle import make_swizzle
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tma_tile
from memory import stack_allocation
from testing import assert_equal
from buffer.dimlist import DimList, Dim
from utils.index import Index, IndexList
from memory import LegacyUnsafePointer
from buffer.buffer import NDBuffer
from layout._ndbuffer_stub import from_ndbuffer_row_major
from gpu.memory import (
    AddressSpace,
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from math import recip

comptime UnsafePointer = LegacyUnsafePointer[mut=True, *_, **_]
from layout.layout_tensor import LayoutTensorIter
from gpu.host import DeviceContext, FuncAttribute
from utils.static_tuple import StaticTuple
from linalg.matmul.gpu.sm100.pipeline import ProducerConsumerPipeline
from gpu.sync import named_barrier
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    NVFP4_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    cast_fp32_to_fp4e2m1,
    cast_uint_to_fp4e2m1,
)
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.fp4_quantization import (
    quantize_dynamic_scaled_fp4,
    quantize_dynamic_scaled_fp4_async,
)
from layout._fillers import random
from testing import assert_equal


def test_nvfp4_quantization[
    dtype: DType,
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
](
    ctx: DeviceContext,
    batch: ValOrDim,
    m: ValOrDim,
    n: ValOrDim,
    tensor_sf: Float32,
):
    comptime out_dtype = DType.uint8

    var B = batch.value
    var M = m.value
    var N = n.value

    comptime input_static_shape = DimList(m.dim, n.dim)
    var input_dynamic_shape = DimList(m.value, n.value)

    var host_ptr = UnsafePointer[Scalar[dtype]].alloc(M * N)
    var host_buffer = NDBuffer[dtype, 2, _, input_static_shape](
        host_ptr, input_dynamic_shape
    )

    comptime output_static_shape = DimList(m.dim, ceildiv(n.dim, 2))
    var output_dynamic_shape = DimList(m.value, ceildiv(n.value, 2))

    var host_ptr_output = UnsafePointer[Scalar[out_dtype]].alloc(
        M * ceildiv(N, 2)
    )
    var host_buffer_output = NDBuffer[out_dtype, 2, _, output_static_shape](
        host_ptr_output, output_dynamic_shape
    )

    var host_ptr_output_ref = UnsafePointer[Scalar[out_dtype]].alloc(
        M * ceildiv(N, 2)
    )
    var host_buffer_output_ref = NDBuffer[out_dtype, 2, _, output_static_shape](
        host_ptr_output_ref, output_dynamic_shape
    )

    var device_buffer = ctx.enqueue_create_buffer[dtype](M * N)
    var device_nd_buffer = NDBuffer[dtype, 2, _, input_static_shape](
        device_buffer.unsafe_ptr(), input_dynamic_shape
    )

    var device_buffer_output = ctx.enqueue_create_buffer[out_dtype](
        M * ceildiv(N, 2)
    )
    var device_nd_buffer_output = NDBuffer[
        out_dtype, 2, _, output_static_shape
    ](device_buffer_output.unsafe_ptr(), output_dynamic_shape)

    var device_buffer_output_ref = ctx.enqueue_create_buffer[out_dtype](
        M * ceildiv(N, 2)
    )
    var device_nd_buffer_output_ref = NDBuffer[
        out_dtype, 2, _, output_static_shape
    ](device_buffer_output_ref.unsafe_ptr(), output_dynamic_shape)

    var host_tensor = from_ndbuffer_row_major(host_buffer)
    random(host_tensor, min=-1.0, max=1.0)

    ctx.enqueue_copy(device_buffer, host_ptr)

    comptime static_scales_shape = DimList(
        ceildiv(m.dim, SF_MN_GROUP_SIZE),
        ceildiv(n.dim, SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var dynamic_scales_shape = DimList(
        ceildiv(m.value, SF_MN_GROUP_SIZE),
        ceildiv(n.value, SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var scales_total = (
        Int(ceildiv(m.value, SF_MN_GROUP_SIZE))
        * Int(ceildiv(n.value, SF_VECTOR_SIZE * SF_ATOM_K))
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )

    var scales_host_ptr = UnsafePointer[Scalar[scales_dtype]].alloc(
        scales_total
    )
    var scales_host = NDBuffer[scales_dtype, 5, _, static_scales_shape](
        scales_host_ptr, dynamic_scales_shape
    )

    var scales_host_ptr_ref = UnsafePointer[Scalar[scales_dtype]].alloc(
        scales_total
    )
    var scales_host_ref = NDBuffer[scales_dtype, 5, _, static_scales_shape](
        scales_host_ptr_ref, dynamic_scales_shape
    )

    var scales_device = ctx.enqueue_create_buffer[scales_dtype](scales_total)
    var scales_device_nd = NDBuffer[scales_dtype, 5, _, static_scales_shape](
        scales_device.unsafe_ptr(), dynamic_scales_shape
    )

    var scales_device_ref = ctx.enqueue_create_buffer[scales_dtype](
        scales_total
    )
    var scales_device_nd_ref = NDBuffer[
        scales_dtype, 5, _, static_scales_shape
    ](scales_device_ref.unsafe_ptr(), dynamic_scales_shape)

    var input_tensor = from_ndbuffer_row_major(device_nd_buffer)
    var output_tensor = from_ndbuffer_row_major(device_nd_buffer_output)
    var scales_tensor = from_ndbuffer_row_major(scales_device_nd)
    var output_tensor_ref = from_ndbuffer_row_major(device_nd_buffer_output_ref)
    var scales_tensor_ref = from_ndbuffer_row_major(scales_device_nd_ref)

    quantize_dynamic_scaled_fp4_async[SF_VECTOR_SIZE=SF_VECTOR_SIZE,](
        ctx,
        output_tensor,
        scales_tensor,
        input_tensor,
        tensor_sf,
    )

    quantize_dynamic_scaled_fp4[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
        ctx,
        output_tensor_ref,
        scales_tensor_ref,
        input_tensor,
        num_cols=N,
        num_cols_padded=N,
        tensor_sf=tensor_sf,
    )

    ctx.enqueue_copy(host_ptr_output, device_buffer_output)
    ctx.enqueue_copy(host_ptr_output_ref, device_buffer_output_ref)

    ctx.enqueue_copy(scales_host_ptr, scales_device)
    ctx.enqueue_copy(scales_host_ptr_ref, scales_device_ref)

    ctx.synchronize()

    var scales_tensor_host_ref = from_ndbuffer_row_major(scales_host_ref)
    var scales_tensor_host = from_ndbuffer_row_major(scales_host)

    # check scalers
    for i in range(scales_tensor_host_ref.dim(0)):
        for j in range(scales_tensor_host_ref.dim(1)):
            for k in range(scales_tensor_host_ref.dim(2)):
                for l in range(scales_tensor_host_ref.dim(3)):
                    for m in range(scales_tensor_host_ref.dim(4)):
                        assert_equal(
                            scales_tensor_host_ref[i, j, k, l, m].cast[
                                DType.float64
                            ](),
                            scales_tensor_host[i, j, k, l, m].cast[
                                DType.float64
                            ](),
                        )

    var output_tensor_host_ref = from_ndbuffer_row_major(host_buffer_output_ref)
    var output_tensor_host = from_ndbuffer_row_major(host_buffer_output)

    # check output
    for row_idx in range(0, M):
        for col_idx in range(0, N // 2, SF_VECTOR_SIZE // 2):
            var output_vector = output_tensor_host.load[SF_VECTOR_SIZE // 2](
                row_idx, col_idx
            )
            var output_vector_ref = output_tensor_host_ref.load[
                SF_VECTOR_SIZE // 2
            ](row_idx, col_idx)

            var output_fp32 = cast_uint_to_fp4e2m1[
                out_dtype = DType.float32,
                out_width=SF_VECTOR_SIZE,
            ](output_vector)
            var output_fp32_ref = cast_uint_to_fp4e2m1[
                out_dtype = DType.float32,
                out_width=SF_VECTOR_SIZE,
            ](output_vector_ref)

            assert_equal(
                output_fp32,
                output_fp32_ref,
            )

    host_ptr.free()
    host_ptr_output.free()
    scales_host_ptr.free()


def main():
    with DeviceContext() as ctx:
        test_nvfp4_quantization[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
        ](
            ctx,
            dynamic(1),
            dynamic(2 * 128),
            static[11 * 64](),
            tensor_sf=1.0,
        )
        test_nvfp4_quantization[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
        ](
            ctx,
            dynamic(1),
            dynamic(999),
            static[576](),
            tensor_sf=1.0,
        )
        test_nvfp4_quantization[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
        ](
            ctx,
            dynamic(1),
            dynamic(129),
            static[23 * 128](),
            tensor_sf=1.0,
        )
        test_nvfp4_quantization[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
        ](
            ctx,
            dynamic(1),
            dynamic(27 * 128),
            static[23 * 128](),
            tensor_sf=0.43,
        )
        test_nvfp4_quantization[
            DType.bfloat16,
            NVFP4_SF_DTYPE,
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
        ](
            ctx,
            dynamic(1),
            dynamic(13),
            static[17 * 128](),
            tensor_sf=0.5,
        )
