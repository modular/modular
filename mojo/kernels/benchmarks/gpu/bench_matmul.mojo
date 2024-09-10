# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-build-no-debug %s


from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList, NDBuffer
from buffer.dimlist import _make_tuple
from gpu.host.device_context import DeviceBuffer, DeviceContext
from linalg.matmul_gpu import _matmul_gpu
from internal_utils import DeviceNDBuffer, bench_compile_time, env_get_dtype
from utils import StaticIntTuple
from sys import env_get_int


fn _get_run_name[
    type: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
    transpose_b: Bool = False,
](
    shape_c_dim: StaticIntTuple[2],
    shape_a_dim: StaticIntTuple[2],
    shape_b_dim: StaticIntTuple[2],
) -> String:
    var str = String("matmul(")
    str += type.__str__()
    str += ") : "
    # M
    str += shape_c_dim[0].__str__()
    # N
    str += (
        "_dynamic"
        + " x "
        + shape_c_dim[1].__str__() if shape_c.at[0]().is_dynamic() else " x "
        + shape_c_dim[1].__str__()
    )
    # K
    str += (
        "_dynamic"
        + " x "
        + shape_a_dim[1].__str__() if shape_c.at[1]().is_dynamic() else " x "
        + shape_a_dim[1].__str__()
    )
    str += "_dynamic" if shape_a.at[1]().is_dynamic() else ""
    str += " transpose_b" if transpose_b else ""
    str += "\t"
    return str


fn bench_matmul[
    dtype: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
    transpose_b: Bool = False,
](
    ctx: DeviceContext,
    inout b: Bench,
    shape_c_dim: StaticIntTuple[2],
    shape_a_dim: StaticIntTuple[2],
    shape_b_dim: StaticIntTuple[2],
) raises:
    var mat_c = DeviceNDBuffer[dtype, 2, shape_c](shape_c_dim, ctx=ctx)
    var mat_a = DeviceNDBuffer[dtype, 2, shape_a](shape_a_dim, ctx=ctx)
    var mat_b = DeviceNDBuffer[dtype, 2, shape_b](shape_b_dim, ctx=ctx)

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
                mat_c.tensor, mat_a.tensor, mat_b.tensor, ctx
            )

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_func](
        BenchId(
            _get_run_name[dtype, shape_c, shape_a, shape_b, transpose_b](
                shape_c_dim, shape_a_dim, shape_b_dim
            )
        ),
        # TODO: Pick relevant benchmetric
        ThroughputMeasure(
            BenchMetric.elements,
            # Flop: 2*M*N*K. Use A and C shapes since they're not transposed.
            2 * shape_c_dim[0] * shape_c_dim[1] * shape_a_dim[1],
        ),
    )

    # Retain our buffers till the end.
    _ = mat_c^
    _ = mat_a^
    _ = mat_b^


struct ValOrDim[dim: Dim = Dim()]:
    var value: Int

    fn __init__(inout self):
        constrained[
            not dim.is_dynamic(),
            "Can't construct a dynamic dim with no runtime value",
        ]()
        self.value = dim.get()

    fn __init__(inout self, v: Int):
        self.value = v


fn static[d: Int]() -> ValOrDim[d]:
    return ValOrDim[d]()


fn dynamic(d: Int) -> ValOrDim:
    return ValOrDim(d)


fn create_matmul_bench[
    dtype: DType,
    transpose_b: Bool = False,
](
    ctx: DeviceContext, inout b: Bench, m: ValOrDim, n: ValOrDim, k: ValOrDim
) raises:
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    var dynamic_b_shape = (n.value, k.value) if transpose_b else (
        k.value,
        n.value,
    )

    bench_matmul[
        dtype,
        DimList(m.dim, n.dim),
        DimList(m.dim, k.dim),
        static_b_shape,
        transpose_b,
    ](ctx, b, (m.value, n.value), (m.value, k.value), dynamic_b_shape)


fn compile_matmul_bench[
    dtype: DType
](
    ctx: DeviceContext, inout b: Bench, m: ValOrDim, n: ValOrDim, k: ValOrDim
) raises:
    var s: String = "type=" + str(dtype) + "/m=" + str(m.value) + ", n=" + str(
        n.value
    ) + ", k=" + str(k.value)
    # Note: important to pass list of BenchMetric's used by the computational benchmark (in this case, BenchMetric.elements)
    bench_compile_time[
        bench_matmul[
            dtype,
            DimList(m.dim, n.dim),
            DimList(m.dim, k.dim),
            DimList(k.dim, n.dim),
        ]
    ](b, "matmul/" + s)


fn main() raises:
    alias dtype = env_get_dtype["dtype", DType.bfloat16]()

    alias M = env_get_int["M", 1]()
    alias N = env_get_int["N", 1]()
    alias K = env_get_int["K", 1]()

    var m = Bench()
    try:
        with DeviceContext() as ctx:
            # benchmarking matmul
            create_matmul_bench[dtype, transpose_b=True](
                ctx,
                m,
                dynamic(M),
                static[N](),
                static[K](),
            )

            # benchmarking compilation time of matmul
            compile_matmul_bench[dtype](
                ctx,
                m,
                dynamic(M),
                static[N](),
                static[K](),
            )

    except e:
        print("CUDA_ERROR:", e)

    m.dump_report()
