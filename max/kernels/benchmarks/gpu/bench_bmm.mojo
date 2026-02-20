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

from math import align_up
from sys import (
    env_get_bool,
    env_get_dtype,
    env_get_int,
    has_nvidia_gpu_accelerator,
    simd_width_of,
    size_of,
)
from sys.info import has_amd_gpu_accelerator

from layout._tile_tensor import TileTensor
import linalg.matmul.vendor.blas as vendor_blas
from algorithm.functional import elementwise
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext, get_gpu_target
from internal_utils import arg_parse
from internal_utils._utils import (
    InitializationType,
    init_vector_launch,
)
from linalg.bmm import _batched_matmul_gpu

from utils import Index, IndexList

comptime to_dim[value: Optional[Int]] = value.value() if value else Dim()


fn _get_run_name[
    dtype: DType,
    *,
    transpose_b: Bool,
    use_vendor_blas: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
    B: Optional[Int] = None,
    M: Optional[Int] = None,
    N: Optional[Int] = None,
    K: Optional[Int] = None,
](b: Int, m: Int, n: Int, k: Int) -> String:
    var vendor_str = "vendor_bmm" if use_vendor_blas else "bmm"
    var type_str = String("(", dtype, ") : ")
    # B
    var b_str = String(b, "" if B else "_dynamic")
    # M
    var m_str = String(m, "" if M else "_dynamic")
    # N
    var n_str = String(n, "" if N else "_dynamic")
    # K
    var k_str = String(k, "" if K else "_dynamic")

    var transpose_b_str = String("/transpose_b=", transpose_b)

    return String(
        vendor_str,
        type_str,
        b_str,
        " x ",
        m_str,
        " x ",
        n_str,
        " x ",
        k_str,
        transpose_b_str,
    )


comptime epilogue_func_type = fn[
    dtype: DType, width: Int, *, alignment: Int = 1
](SIMD[dtype, width]) capturing -> SIMD[dtype, width]


fn _row_major_shapes_to_strides[shapes_dim: DimList]() -> DimList:
    """Compute the strides for a 3D shape. Assuming row-major layout."""

    comptime if shapes_dim.has_value[2]():
        comptime if shapes_dim.has_value[1]():
            return DimList(
                shapes_dim.get[1]() * shapes_dim.get[2](),
                shapes_dim.get[2](),
                1,
            )
        else:
            return DimList(Dim(), shapes_dim.get[2](), 1)
    else:
        return DimList(Dim(), Dim(), 1)


@always_inline
@parameter
fn elementwise_epilogue_fn[
    dtype: DType,
    width: Int,
    *,
    alignment: Int = 1,
](val: SIMD[dtype, width],) -> SIMD[dtype, width]:
    return val + 2


fn bench_bmm[
    dtype: DType,
    /,
    *,
    use_vendor_blas: Bool = False,
    transpose_b: Bool = False,
    lambda_fn: Optional[epilogue_func_type] = None,
    B: Optional[Int] = None,
    M: Optional[Int] = None,
    N: Optional[Int] = None,
    K: Optional[Int] = None,
](
    ctx: DeviceContext,
    mut bench: Bench,
    b: Int,
    m: Int,
    n: Int,
    k: Int,
    init_type: InitializationType,
) raises:
    comptime batch_static_a_shape = DimList(to_dim[B], to_dim[M], to_dim[K])
    comptime batch_static_b_shape = DimList(
        to_dim[B], to_dim[N], to_dim[K]
    ) if transpose_b else DimList(to_dim[B], to_dim[K], to_dim[N])
    comptime batch_static_c_shape = DimList(to_dim[B], to_dim[M], to_dim[N])

    comptime batch_static_a_strides = _row_major_shapes_to_strides[
        batch_static_a_shape
    ]()
    comptime batch_static_b_strides = _row_major_shapes_to_strides[
        batch_static_b_shape
    ]()
    comptime batch_static_c_strides = _row_major_shapes_to_strides[
        batch_static_c_shape
    ]()

    comptime static_a_shape = DimList(to_dim[M], to_dim[K])
    comptime static_b_shape = DimList(
        to_dim[N], to_dim[K]
    ) if transpose_b else DimList(to_dim[K], to_dim[N])
    comptime static_c_shape = DimList(to_dim[M], to_dim[N])

    var batch_dynamic_a_shape = IndexList[3](
        B.or_else(b), M.or_else(m), K.or_else(k)
    )
    var batch_dynamic_b_shape = IndexList[3](
        B.or_else(b), N.or_else(n), K.or_else(k)
    ) if transpose_b else IndexList[3](B.or_else(b), K.or_else(k), N.or_else(n))

    var batch_dynamic_c_shape = IndexList[3](
        B.or_else(b), M.or_else(m), N.or_else(n)
    )

    var dynamic_a_shape = IndexList[2](M.or_else(m), K.or_else(k))
    var dynamic_b_shape = IndexList[2](
        N.or_else(n), K.or_else(k)
    ) if transpose_b else IndexList[2](K.or_else(k), N.or_else(n))

    var dynamic_c_shape = IndexList[2](M.or_else(m), N.or_else(n))

    var a_size = b * m * k
    var b_size = b * n * k if transpose_b else b * k * n
    var c_size = b * m * n

    var a_device_buffer = ctx.enqueue_create_buffer[dtype](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[dtype](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[dtype](c_size)

    var a_device = NDBuffer[
        dtype, 3, MutAnyOrigin, batch_static_a_shape, batch_static_a_strides
    ](a_device_buffer.unsafe_ptr(), batch_dynamic_a_shape)
    var b_device = NDBuffer[
        dtype, 3, MutAnyOrigin, batch_static_b_shape, batch_static_b_strides
    ](b_device_buffer.unsafe_ptr(), batch_dynamic_b_shape)
    var c_device = NDBuffer[
        dtype, 3, MutAnyOrigin, batch_static_c_shape, batch_static_c_strides
    ](c_device_buffer.unsafe_ptr(), batch_dynamic_c_shape)

    # Initialize data on the device
    init_vector_launch[dtype](a_device_buffer, a_size, init_type, ctx)
    init_vector_launch[dtype](b_device_buffer, b_size, init_type, ctx)

    @parameter
    @always_inline
    @__copy_capture(c_device)
    fn epilogue_fn[
        dtype: DType,
        width: Int,
        rank: Int,
        *,
        alignment: Int = 1,
    ](idx: IndexList[rank], val: SIMD[dtype, width],) capturing -> None:
        comptime func = lambda_fn.value()
        var update_val = func(val)
        c_device.store(
            Index(idx[0], idx[1], idx[2]), update_val.cast[c_device.type]()
        )

    comptime pack_size = simd_width_of[dtype, target = get_gpu_target()]()

    @always_inline
    @__copy_capture(c_device, b, m, n)
    @parameter
    fn func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx0: IndexList[rank]):
        var idx = rebind[IndexList[3]](idx0)
        var val = c_device.load[width=simd_width](idx)
        comptime element_lambda = lambda_fn.value()
        var update_val = element_lambda(val)

        c_device.store(
            idx,
            update_val,
        )

    @parameter
    @__copy_capture(a_device, b_device, c_device)
    @always_inline
    fn bench_func(mut bench: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            comptime if use_vendor_blas:
                comptime if has_amd_gpu_accelerator():
                    var c_buffer = NDBuffer[dtype, 2, _, static_c_shape](
                        c_device.data, dynamic_c_shape
                    )
                    var a_buffer = NDBuffer[dtype, 2, _, static_a_shape](
                        a_device.data, dynamic_a_shape
                    )
                    var b_buffer = NDBuffer[dtype, 2, _, static_b_shape](
                        b_device.data, dynamic_b_shape
                    )

                    vendor_blas.matmul(
                        ctx,
                        c_buffer,
                        a_buffer,
                        b_buffer,
                        c_row_major=True,
                        transpose_b=transpose_b,
                        batch_size=b,
                    )
                else:
                    # Fallback vendor BMM for non-AMD GPUs or when AMD GPU acceleration is not available
                    for i in range(b):
                        var c_ptr = c_device.data + (i * m * n)
                        var a_ptr = a_device.data + (i * m * k)
                        var b_ptr = b_device.data + (i * k * n)

                        var c_buffer = NDBuffer[dtype, 2, _, static_c_shape](
                            c_ptr, dynamic_c_shape
                        )
                        var a_buffer = NDBuffer[dtype, 2, _, static_a_shape](
                            a_ptr, dynamic_a_shape
                        )
                        var b_buffer = NDBuffer[dtype, 2, _, static_b_shape](
                            b_ptr, dynamic_b_shape
                        )

                        vendor_blas.matmul(
                            ctx,
                            c_buffer,
                            a_buffer,
                            b_buffer,
                            c_row_major=True,
                            transpose_b=transpose_b,
                        )
                ctx.synchronize()

                # Epilogue
                comptime if lambda_fn:
                    elementwise[func, pack_size, target="gpu"](
                        IndexList[3](b, m, Int(N.value())),
                        ctx,
                    )
            else:
                comptime if lambda_fn:
                    _batched_matmul_gpu[
                        transpose_b=transpose_b,
                        elementwise_epilogue_fn=epilogue_fn,
                    ](
                        TileTensor(c_device),
                        TileTensor(a_device),
                        TileTensor(b_device),
                        ctx,
                    )
                else:
                    _batched_matmul_gpu[transpose_b=transpose_b](
                        TileTensor(c_device),
                        TileTensor(a_device),
                        TileTensor(b_device),
                        ctx,
                    )

        bench.iter_custom[kernel_launch](ctx)

    bench.bench_function[bench_func](
        BenchId(
            _get_run_name[
                dtype,
                transpose_b=transpose_b,
                use_vendor_blas=use_vendor_blas,
                lambda_fn=lambda_fn,
                B=B,
                M=M,
                N=N,
                K=K,
            ](b, m, n, k)
        ),
        # TODO: Pick relevant benchmetric
        [
            ThroughputMeasure(
                BenchMetric.flops,
                2 * b * m * n * k,
            )
        ],
    )

    # Retain our buffers till the end.
    _ = a_device_buffer^
    _ = b_device_buffer^
    _ = c_device_buffer^


fn create_bmm_bench[
    dtype: DType,
    *,
    transpose_b: Bool,
    use_vendor_blas: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
    B: Optional[Int] = None,
    M: Optional[Int] = None,
    N: Optional[Int] = None,
    K: Optional[Int] = None,
](
    ctx: DeviceContext,
    mut bench: Bench,
    b: Int,
    m: Int,
    n: Int,
    k: Int,
    init_type: InitializationType,
) raises:
    bench_bmm[
        dtype,
        transpose_b=transpose_b,
        use_vendor_blas=use_vendor_blas,
        lambda_fn=lambda_fn,
        B=B,
        M=M,
        N=N,
        K=K,
    ](
        ctx,
        bench,
        b,
        m,
        n,
        k,
        init_type,
    )


def main():
    comptime dtype = env_get_dtype["dtype", DType.bfloat16]()

    var b = Int(arg_parse("B", 1))
    var m = Int(arg_parse("M", 1))
    comptime N = env_get_int["N", 1]()
    comptime K = env_get_int["K", 1]()
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    comptime transpose_b = False
    comptime use_vendor_blas = env_get_bool["use_vendor_blas", False]()

    var bench = Bench()
    with DeviceContext() as ctx:
        create_bmm_bench[
            dtype,
            transpose_b=transpose_b,
            use_vendor_blas=use_vendor_blas,
            N = Int(N),
            K = Int(K),
        ](
            ctx,
            bench,
            b,
            m,
            N,
            K,
            init_type,
        )

    bench.dump_report()
