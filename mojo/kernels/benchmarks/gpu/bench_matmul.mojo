# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s -t | FileCheck %s
# CHECK: Benchmark results

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from benchmark._cuda import time_async_cuda_kernel
from gpu.host import Context, Stream, CUDADeviceStream
from gpu.host.memory import _malloc, _free

from LinAlg.MatmulGPU import _matmul_gpu

from buffer import NDBuffer, DimList, Dim


fn _size[rank: Int](dims: StaticIntTuple[rank]) -> Int:
    var size = 1

    @parameter
    for i in range(rank):
        size *= dims[i]
    return size


fn _create_device_buffer[
    dtype: DType, rank: Int, shape: DimList
](dynamic_shape: StaticIntTuple[rank]) raises -> NDBuffer[dtype, rank, shape]:
    var storage_ptr = _malloc[dtype](_size(dynamic_shape))
    return NDBuffer[dtype, rank, shape](
        storage_ptr, dynamic_shape=dynamic_shape
    )


fn _get_run_name[
    type: DType, shape_c: DimList, shape_a: DimList, shape_b: DimList
](
    shape_c_dim: StaticIntTuple[2],
    shape_a_dim: StaticIntTuple[2],
    shape_b_dim: StaticIntTuple[2],
) -> String:
    var str = String("matmul(")
    str += type.__str__()
    str += ") : "
    str += shape_c_dim[0].__str__()
    str += (
        "_dynamic"
        + " x "
        + shape_b_dim[1].__str__() if shape_c.at[0]().is_dynamic() else " x "
        + shape_b_dim[1].__str__()
    )
    str += (
        "_dynamic"
        + " x "
        + shape_a_dim[1].__str__() if shape_b.at[1]().is_dynamic() else " x "
        + shape_a_dim[1].__str__()
    )
    str += "_dynamic" if shape_a.at[1]().is_dynamic() else ""
    str += "\t"
    return str


fn bench_matmul[
    dtype: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
](
    inout b: Bench,
    shape_c_dim: StaticIntTuple[2],
    shape_a_dim: StaticIntTuple[2],
    shape_b_dim: StaticIntTuple[2],
) raises:
    var mat_c = _create_device_buffer[dtype, 2, shape_c](shape_c_dim)
    var mat_a = _create_device_buffer[dtype, 2, shape_a](shape_a_dim)
    var mat_b = _create_device_buffer[dtype, 2, shape_b](shape_b_dim)

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(stream: Stream) raises:
            _matmul_gpu(mat_c, mat_a, mat_b, CUDADeviceStream(stream))

        b.iter_custom[time_async_cuda_kernel[kernel_launch]]()

    b.bench_function[bench_func](
        BenchId(
            _get_run_name[dtype, shape_c, shape_a, shape_b](
                shape_c_dim, shape_a_dim, shape_b_dim
            )
        ),
        # TODO: Pick relevant benchmetric
        ThroughputMeasure(
            BenchMetric.elements,
            2 * shape_c_dim[0] * shape_c_dim[1] * shape_b_dim[0],
        ),
    )

    _free(mat_c.data)
    _free(mat_a.data)
    _free(mat_b.data)


struct ValOrDim[dim: Dim = Dim()]:
    var value: Int

    fn __init__(inout self):
        constrained[
            not dim.is_dynamic(),
            "Can't construct a dynamic dim with no runtime value",
        ]()
        self.value = dim.value.value()

    fn __init__(inout self, v: Int):
        self.value = v


fn static[d: Int]() -> ValOrDim[d]:
    return ValOrDim[d]()


fn dynamic(d: Int) -> ValOrDim:
    return ValOrDim(d)


fn create_matmul_bench[
    dtype: DType
](inout b: Bench, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises:
    bench_matmul[
        dtype,
        DimList(m.dim, n.dim),
        DimList(m.dim, k.dim),
        DimList(k.dim, n.dim),
    ](b, (m.value, n.value), (m.value, k.value), (k.value, n.value))


fn main() raises:
    var b = Bench()
    with Context() as ctx:
        # Lama2 shapes CE.
        create_matmul_bench[DType.float32](
            b, dynamic(256), static[22016](), static[4096]()
        )
        create_matmul_bench[DType.float32](
            b, dynamic(256), static[12288](), static[4096]()
        )
        create_matmul_bench[DType.float32](
            b, dynamic(256), static[4096](), static[11008]()
        )
        create_matmul_bench[DType.float32](
            b, dynamic(256), static[12288](), static[4096]()
        )
        create_matmul_bench[DType.float32](
            b, dynamic(1), static[32000](), static[4096]()
        )
        create_matmul_bench[DType.float32](
            b, dynamic(256), static[4096](), static[4096]()
        )
        # Lama2 shapes LPTG.
        create_matmul_bench[DType.float32](
            b, dynamic(1), static[12288](), static[3072]()
        )
        create_matmul_bench[DType.float32](
            b, dynamic(1), static[3072](), static[12288]()
        )
        create_matmul_bench[DType.float32](
            b, dynamic(1), static[5120](), static[3072]()
        )
        create_matmul_bench[DType.float32](
            b, dynamic(1), static[3072](), static[3072]()
        )

    b.dump_report()
