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

from buffer.dimlist import DimList
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, random, zero
from internal_utils._utils import ValOrDim, dynamic, static
from linalg.fp4_quantization import (
    quantize_dynamic_scaled_fp4,
)
from testing import assert_equal, assert_almost_equal
from math import ceildiv, recip
from utils.numerics import max_finite, min_finite
from buffer import Dim
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.fp4_utils import (
    cast_fp_to_fp4e2m1,
    cast_uint32_to_fp4e2m1,
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    _get_scale_factor,
)


fn test_dynamic_fp4_quant[
    in_dtype: DType,
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
](
    ctx: DeviceContext, m: ValOrDim, n: ValOrDim, tensor_sf: Float32 = 1.0
) raises:
    if n.value % (SF_VECTOR_SIZE // 2) != 0:
        raise Error(
            "n must be a multiple of (SF_VECTOR_SIZE // 2) due to kernel"
            " constraints"
        )

    alias out_dtype = DType.uint32

    alias input_static_shape = DimList(m.dim, n.dim)
    var input_dynamic_shape = DimList(m.value, n.value)

    var in_host = HostNDBuffer[in_dtype, 2, input_static_shape](
        input_dynamic_shape
    )
    var in_device = DeviceNDBuffer[in_dtype, 2, input_static_shape](
        input_dynamic_shape, ctx=ctx
    )

    alias output_static_shape = DimList(
        m.dim, ceildiv(n.dim, SF_VECTOR_SIZE // 2)
    )
    var output_dynamic_shape = DimList(
        m.value, ceildiv(n.value, SF_VECTOR_SIZE // 2)
    )

    var out_host = HostNDBuffer[out_dtype, 2, output_static_shape](
        output_dynamic_shape
    )
    var out_device = DeviceNDBuffer[out_dtype, 2, output_static_shape](
        output_dynamic_shape, ctx=ctx
    )

    random(in_host.tensor)
    zero(out_host.tensor)

    comptime scales_shape_static = DimList(
        ceildiv(m.dim, SF_MN_GROUP_SIZE),
        ceildiv(n.dim, SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var scales_shape_dynamic = DimList(
        ceildiv(m.value, SF_MN_GROUP_SIZE),
        ceildiv(n.value, SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var scales_host = HostNDBuffer[scales_dtype, 5, scales_shape_static](
        scales_shape_dynamic
    )
    var scales_device = DeviceNDBuffer[scales_dtype, 5, scales_shape_static](
        scales_shape_dynamic, ctx=ctx
    )

    var scales_tensor = from_ndbuffer_row_major(scales_device.tensor)
    var input_tensor = from_ndbuffer_row_major(in_device.tensor)
    var output_tensor = from_ndbuffer_row_major(out_device.tensor)

    ctx.enqueue_copy(in_device.buffer, in_host.tensor.data)
    ctx.enqueue_copy(out_device.buffer, out_host.tensor.data)

    quantize_dynamic_scaled_fp4[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
        ctx,
        output_tensor,
        scales_tensor,
        input_tensor,
        num_cols=n.value,
        num_cols_padded=n.value,
        tensor_sf=tensor_sf,
    )

    ctx.enqueue_copy(out_host.tensor.data, out_device.buffer)
    ctx.enqueue_copy(scales_host.tensor.data, scales_device.buffer)
    ctx.synchronize()

    var scales_tensor_host = from_ndbuffer_row_major(scales_host.tensor)
    var input_tensor_host = from_ndbuffer_row_major(in_host.tensor)
    var output_tensor_host = from_ndbuffer_row_major(out_host.tensor)

    for row_idx in range(0, m.value):
        for col_idx in range(0, n.value, SF_VECTOR_SIZE):
            var vec_max = Scalar[DType.float32](0.0)
            # kernel support N shapes that are multiples of (SF_VECTOR_SIZE // 2).
            # Here we handle the oob case by loading only the first half of the SF_VECTOR_SIZE.
            if (n.value % SF_VECTOR_SIZE != 0) and (
                col_idx + SF_VECTOR_SIZE > n.value
            ):
                var input_vector = input_tensor_host.load[SF_VECTOR_SIZE // 2](
                    row_idx, col_idx
                )
                vec_max = abs(input_vector).reduce_max().cast[DType.float32]()
            else:
                var input_vector = input_tensor_host.load[SF_VECTOR_SIZE](
                    row_idx, col_idx
                )
                vec_max = abs(input_vector).reduce_max().cast[DType.float32]()

            var sf_value = tensor_sf * (vec_max * recip(Float32(6.0)))
            var ref_fp8_sf = sf_value.cast[scales_dtype]()

            var fp8_sf = _get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                scales_tensor_host, row_idx, col_idx
            )

            # verify the scale factors
            assert_almost_equal(
                ref_fp8_sf.cast[DType.float64](),
                rebind[Scalar[scales_dtype]](fp8_sf).cast[DType.float64](),
                rtol=1e-1,
                atol=1e-1,
            )

            var output_scale = Float32(0.0)
            if vec_max != 0:
                output_scale = recip(
                    ref_fp8_sf.cast[DType.float32]() * recip(tensor_sf)
                )

            # verify the output values
            if (n.value % SF_VECTOR_SIZE != 0) and (
                col_idx + SF_VECTOR_SIZE > n.value
            ):
                var input_f32 = (
                    input_tensor_host.load[SF_VECTOR_SIZE // 2](
                        row_idx, col_idx
                    ).cast[DType.float32]()
                    * output_scale
                )
                var ref_output_e2m1 = cast_fp_to_fp4e2m1(input_f32)
                var output_e2m1 = cast_uint32_to_fp4e2m1[
                    out_dtype = DType.float32, out_width = SF_VECTOR_SIZE // 2
                ](
                    output_tensor_host.load[1](
                        row_idx, col_idx // (SF_VECTOR_SIZE // 2)
                    )
                )
                assert_almost_equal(
                    ref_output_e2m1,
                    output_e2m1,
                    rtol=1e0,
                    atol=1e-1,
                )
            else:
                var input_f32 = (
                    input_tensor_host.load[SF_VECTOR_SIZE](
                        row_idx, col_idx
                    ).cast[DType.float32]()
                    * output_scale
                )
                var ref_output_e2m1 = cast_fp_to_fp4e2m1(input_f32)
                var output_e2m1 = cast_uint32_to_fp4e2m1[
                    out_dtype = DType.float32, out_width=SF_VECTOR_SIZE
                ](
                    output_tensor_host.load[2](
                        row_idx, col_idx // (SF_VECTOR_SIZE // 2)
                    )
                )
                assert_almost_equal(
                    ref_output_e2m1,
                    output_e2m1,
                    rtol=1e0,
                    atol=1e-1,
                )


def main():
    with DeviceContext() as ctx:
        test_dynamic_fp4_quant[
            DType.bfloat16, NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE
        ](ctx, dynamic(256), static[128]())
        test_dynamic_fp4_quant[
            DType.bfloat16, NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE
        ](ctx, dynamic(258), static[128 + 8]())
        test_dynamic_fp4_quant[
            DType.bfloat16, NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE
        ](ctx, dynamic(258), static[128 + 64 - 8]())
        test_dynamic_fp4_quant[
            DType.bfloat16, NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE
        ](ctx, dynamic(1000), static[8192 + 8](), tensor_sf=0.43)
        test_dynamic_fp4_quant[
            DType.bfloat16, NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE
        ](ctx, dynamic(2048), static[16384 + 8](), tensor_sf=0.5)
