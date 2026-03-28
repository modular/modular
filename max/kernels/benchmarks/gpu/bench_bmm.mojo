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

from std.sys import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    simd_width_of,
)
from std.sys.info import has_amd_gpu_accelerator

from layout import Coord, RuntimeInt, TileTensor, row_major
import linalg.matmul.vendor.blas as vendor_blas
from std.algorithm.functional import elementwise
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu.host import DeviceContext, get_gpu_target
from internal_utils import arg_parse
from internal_utils._utils import (
    InitializationType,
    init_vector_launch,
)
from linalg.bmm import _batched_matmul_gpu

from std.utils import IndexList


def _ri(v: Int) -> RuntimeInt[DType.int64]:
    return RuntimeInt[DType.int64](Int64(v))


def _get_run_name[
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


comptime epilogue_func_type = def[
    dtype: DType, width: Int, *, alignment: Int = 1
](SIMD[dtype, width]) capturing -> SIMD[dtype, width]


@always_inline
@parameter
def elementwise_epilogue_fn[
    dtype: DType,
    width: Int,
    *,
    alignment: Int = 1,
](val: SIMD[dtype, width],) -> SIMD[dtype, width]:
    return val + 2


def bench_bmm[
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
    var a_size = b * m * k
    var b_size = b * n * k if transpose_b else b * k * n
    var c_size = b * m * n

    var a_device_buffer = ctx.enqueue_create_buffer[dtype](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[dtype](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[dtype](c_size)

    var a_device = TileTensor(
        a_device_buffer.unsafe_ptr(),
        row_major(Coord(_ri(b), _ri(m), _ri(k))),
    ).as_any_origin()
    var b_device = TileTensor(
        b_device_buffer.unsafe_ptr(),
        row_major(Coord(_ri(b), _ri(n), _ri(k))) if transpose_b else row_major(
            Coord(_ri(b), _ri(k), _ri(n))
        ),
    ).as_any_origin()
    var c_device = TileTensor(
        c_device_buffer.unsafe_ptr(),
        row_major(Coord(_ri(b), _ri(m), _ri(n))),
    ).as_any_origin()

    # Initialize data on the device
    init_vector_launch[dtype](a_device_buffer, a_size, init_type, ctx)
    init_vector_launch[dtype](b_device_buffer, b_size, init_type, ctx)

    @parameter
    @always_inline
    @__copy_capture(c_device)
    def epilogue_fn[
        dtype: DType,
        width: Int,
        rank: Int,
        *,
        alignment: Int = 1,
    ](idx: IndexList[rank], val: SIMD[dtype, width],) capturing -> None:
        comptime func = lambda_fn.value()
        var update_val = func(val)
        c_device.store_linear(idx, update_val.cast[c_device.dtype]())

    comptime pack_size = simd_width_of[dtype, target=get_gpu_target()]()

    @always_inline
    @__copy_capture(c_device, b, m, n)
    @parameter
    def func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx0: IndexList[rank]):
        var idx = rebind[IndexList[3]](idx0)
        var val = c_device.load_linear[width=simd_width](idx)
        comptime element_lambda = lambda_fn.value()
        var update_val = element_lambda(val)

        c_device.store_linear(
            idx,
            update_val,
        )

    @parameter
    @__copy_capture(a_device, b_device, c_device)
    @always_inline
    def bench_func(mut bench: Bencher):
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            comptime if use_vendor_blas:
                comptime if has_amd_gpu_accelerator():
                    var c_buffer = TileTensor(
                        c_device.ptr, row_major(Coord(_ri(m), _ri(n)))
                    )
                    var a_buffer = TileTensor(
                        a_device.ptr, row_major(Coord(_ri(m), _ri(k)))
                    )
                    var b_buffer = TileTensor(
                        b_device.ptr,
                        row_major(
                            Coord(_ri(n), _ri(k))
                        ) if transpose_b else row_major(Coord(_ri(k), _ri(n))),
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
                    # Fallback vendor BMM for non-AMD GPUs
                    for i in range(b):
                        var c_ptr = c_device.ptr + (i * m * n)
                        var a_ptr = a_device.ptr + (i * m * k)
                        var b_ptr = b_device.ptr + (i * k * n)

                        var c_buffer = TileTensor(
                            c_ptr, row_major(Coord(_ri(m), _ri(n)))
                        )
                        var a_buffer = TileTensor(
                            a_ptr, row_major(Coord(_ri(m), _ri(k)))
                        )
                        var b_buffer = TileTensor(
                            b_ptr,
                            row_major(
                                Coord(_ri(n), _ri(k))
                            ) if transpose_b else row_major(
                                Coord(_ri(k), _ri(n))
                            ),
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
                        c_device,
                        a_device,
                        b_device,
                        ctx,
                    )
                else:
                    _batched_matmul_gpu[transpose_b=transpose_b](
                        c_device,
                        a_device,
                        b_device,
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


def create_bmm_bench[
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


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()

    var b = Int(arg_parse("B", 1))
    var m = Int(arg_parse("M", 1))
    comptime N = get_defined_int["N", 1]()
    comptime K = get_defined_int["K", 1]()
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    comptime transpose_b = False
    comptime use_vendor_blas = get_defined_bool["use_vendor_blas", False]()

    var bench = Bench()
    with DeviceContext() as ctx:
        create_bmm_bench[
            dtype,
            transpose_b=transpose_b,
            use_vendor_blas=use_vendor_blas,
            N=Int(N),
            K=Int(K),
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
