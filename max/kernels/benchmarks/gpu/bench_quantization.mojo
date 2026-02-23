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

from sys import env_get_dtype, env_get_int, size_of, env_get_bool
from math import ceildiv
from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    ThroughputMeasure,
    BenchMetric,
)
from gpu.host import DeviceContext
from internal_utils import env_get_shape, int_list_to_tuple
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
)
from memory import LegacyUnsafePointer
from layout._fillers import random
from gpu.host.info import B200

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from utils.index import IndexList
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
)
from linalg.fp4_quantization import (
    quantize_dynamic_scaled_fp4fp8,
    quantize_dynamic_scaled_fp4_async,
)
from internal_utils import arg_parse


fn bench_1d1d_quantization[
    in_dtype: DType, cols: Int, use_async: Bool, is_fp4: Bool
](ctx: DeviceContext, mut b: Bench, fn_name: String, rows: Int) raises:
    comptime out_dtype = DType.uint8 if is_fp4 else DType.float8_e4m3fn
    comptime scales_dtype = NVFP4_SF_DTYPE if is_fp4 else MXFP8_SF_DTYPE
    comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE if is_fp4 else MXFP8_SF_VECTOR_SIZE

    comptime input_static_shape = Layout.row_major(UNKNOWN_VALUE, cols)
    var input_dynamic_shape = IndexList[2](rows, cols)
    var input_runtime_layout = RuntimeLayout[input_static_shape].row_major(
        input_dynamic_shape
    )
    var in_device = ctx.enqueue_create_buffer[in_dtype](
        input_dynamic_shape.flattened_length()
    )
    var input_tensor = LayoutTensor[in_dtype, input_static_shape](
        in_device, input_runtime_layout
    )

    # Output tensor layout and buffer
    comptime output_static_shape = Layout.row_major(
        UNKNOWN_VALUE,
        ceildiv(cols, 2),
    )
    var output_dynamic_shape = IndexList[2](rows, ceildiv(cols, 2))
    var output_runtime_layout = RuntimeLayout[output_static_shape].row_major(
        output_dynamic_shape
    )
    var out_device = ctx.enqueue_create_buffer[out_dtype](
        output_dynamic_shape.flattened_length()
    )
    var output_tensor = LayoutTensor[out_dtype, output_static_shape](
        out_device, output_runtime_layout
    )

    # Scales tensor layout and buffer
    var scales_shape = IndexList[5](
        ceildiv(rows, SF_MN_GROUP_SIZE),
        ceildiv(cols, SF_VECTOR_SIZE * SF_ATOM_K),
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )
    comptime scales_static_layout = Layout.row_major(
        UNKNOWN_VALUE,
        ceildiv(cols, SF_VECTOR_SIZE * SF_ATOM_K),
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    )
    var scales_runtime_layout = RuntimeLayout[scales_static_layout].row_major(
        scales_shape
    )
    var scales_device = ctx.enqueue_create_buffer[scales_dtype](
        scales_shape.flattened_length()
    )
    var scales_tensor = LayoutTensor[scales_dtype, scales_static_layout](
        scales_device, scales_runtime_layout
    )

    # Initialize input with random data and output with zeros on host
    with in_device.map_to_host() as in_host:
        var in_host_tensor = LayoutTensor[in_dtype, input_static_shape](
            in_host, input_runtime_layout
        )
        random(in_host_tensor)

    @always_inline
    @__copy_capture(input_tensor, output_tensor, scales_tensor)
    @parameter
    fn bench_fn(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            # Run the quantization kernel
            comptime if use_async:
                quantize_dynamic_scaled_fp4_async[
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE
                ](
                    ctx,
                    output_tensor.as_any_origin(),
                    scales_tensor.as_any_origin(),
                    input_tensor.as_any_origin(),
                )
            else:
                quantize_dynamic_scaled_fp4fp8[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    ctx,
                    output_tensor.as_any_origin(),
                    scales_tensor.as_any_origin(),
                    input_tensor.as_any_origin(),
                    num_cols=cols,
                    num_cols_padded=cols,
                )

        b.iter_custom[kernel_launch](ctx)

    var bytes = ThroughputMeasure(
        BenchMetric.bytes,
        (rows * cols) * size_of[in_dtype]()
        + (rows * cols // (2 if is_fp4 else 1)) * size_of[out_dtype]()
        + (
            ceildiv(rows, SF_MN_GROUP_SIZE)
            * ceildiv(cols, SF_VECTOR_SIZE * SF_ATOM_K)
        )
        * (SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K)
        * size_of[scales_dtype](),
    )

    b.bench_function[bench_fn](
        BenchId(
            "1d1d_quantization",
            input_id=String(
                fn_name,
                in_dtype,
                out_dtype,
                scales_dtype,
                SF_VECTOR_SIZE,
                rows,
                cols,
                sep="/",
            ),
        ),
        [bytes],
        # fixed_iterations=1,
    )

    ctx.synchronize()


def main():
    comptime in_dtype = env_get_dtype["dtype", DType.bfloat16]()

    var rows = Int(arg_parse("M", 1))
    comptime cols = env_get_int["N", 1024]()
    comptime use_async = env_get_bool["use_async", True]()
    comptime is_fp4 = env_get_bool["is_fp4", True]()

    with DeviceContext() as ctx:
        comptime if ctx.default_device_info.compute == B200.compute:
            var m = Bench(BenchConfig(num_repetitions=1))
            bench_1d1d_quantization[in_dtype, cols, use_async, is_fp4](
                ctx, m, "1d1d_quantization", rows
            )
            m.dump_report()
        else:
            print("this benchmark is only supported on NVIDIA SM100")
