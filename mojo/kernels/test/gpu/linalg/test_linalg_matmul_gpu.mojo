# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s
# CHECK-NOT: Failed

from math import isclose

from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceBuffer, DeviceContext
from gpu.host.info import DEFAULT_GPU_ARCH
from linalg.matmul import matmul
from linalg.matmul_gpu import _matmul_gpu
from memory import UnsafePointer
from testing import assert_almost_equal

from utils import IndexList


fn _size[rank: Int](dims: IndexList[rank, **_]) -> Int:
    var size = 1

    @parameter
    for i in range(rank):
        size *= dims[i]
    return size


fn _create_device_buffer[
    dtype: DType, rank: Int, shape: DimList
](ctx: DeviceContext, dynamic_shape: IndexList[rank]) raises -> Tuple[
    DeviceBuffer[dtype], NDBuffer[dtype, rank, MutableAnyOrigin, shape]
]:
    var storage = ctx.enqueue_create_buffer[dtype](_size(dynamic_shape))
    return (
        storage,
        NDBuffer[dtype, rank, _, shape](
            storage.unsafe_ptr(), dynamic_shape=dynamic_shape
        ),
    )


fn _create_host_buffer[
    dtype: DType, rank: Int, shape: DimList
](dynamic_shape: IndexList[rank, **_]) raises -> NDBuffer[
    dtype, rank, MutableAnyOrigin, shape
]:
    var storage_ptr = UnsafePointer[Scalar[dtype]].alloc(_size(dynamic_shape))
    return NDBuffer[dtype, rank, _, shape](
        storage_ptr, dynamic_shape=dynamic_shape
    )


fn _linspace_fill[
    dtype: DType, rank: Int, shape: DimList
](mut buff: NDBuffer[mut=True, dtype, rank, _, shape]):
    for i in range(buff.size()):
        buff.data[i] = i


fn _create_host_buffer_like[
    dtype: DType, rank: Int, shape: DimList
](buff: NDBuffer[dtype, rank, _, shape]) raises -> NDBuffer[
    dtype, rank, MutableAnyOrigin, shape
]:
    return _create_host_buffer[dtype, rank, shape](buff.dynamic_shape)


fn _get_test_name[
    type: DType, shape_c: DimList, shape_a: DimList, shape_b: DimList
](
    shape_c_dim: IndexList[2],
    shape_a_dim: IndexList[2],
    shape_b_dim: IndexList[2],
) -> String:
    var str = String("test-case(")
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
    str += ", ... "
    return str


fn matmul_test_case[
    dtype: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
](
    shape_c_dim: IndexList[2],
    shape_a_dim: IndexList[2],
    shape_b_dim: IndexList[2],
    ctx: DeviceContext,
) raises:
    print(
        _get_test_name[dtype, shape_c, shape_a, shape_b](
            shape_c_dim, shape_a_dim, shape_b_dim
        ),
        end=" ",
    )

    var mat_a_dev = _create_device_buffer[dtype, 2, shape_a](ctx, shape_a_dim)
    var mat_b_dev = _create_device_buffer[dtype, 2, shape_b](ctx, shape_b_dim)
    var mat_a_host = _create_host_buffer_like(mat_a_dev[1])
    var mat_b_host = _create_host_buffer_like(mat_b_dev[1])
    var mat_c_dev = _create_device_buffer[dtype, 2, shape_c](ctx, shape_c_dim)
    var mat_c_host = _create_host_buffer_like(mat_c_dev[1])
    var mat_c_ref_host = _create_host_buffer_like(mat_c_host)

    _linspace_fill(mat_a_host)
    _linspace_fill(mat_b_host)

    ctx.enqueue_copy(mat_a_dev[0], mat_a_host.data)
    ctx.enqueue_copy(mat_b_dev[0], mat_b_host.data)

    _matmul_gpu[use_tensor_core=True](
        mat_c_dev[1], mat_a_dev[1], mat_b_dev[1], ctx
    )

    ctx.enqueue_copy(mat_c_host.data, mat_c_dev[0])
    ctx.synchronize()

    # FIXME: We should run a reference gpu matmul, the reference should also
    # support applying the epilogue on the final result.
    matmul(
        mat_c_ref_host,
        mat_a_host,
        mat_b_host,
    )

    for m in range(shape_c_dim[0]):
        for n in range(shape_c_dim[1]):
            assert_almost_equal(mat_c_ref_host[m, n], mat_c_host[m, n])

    mat_a_host.data.free()
    mat_b_host.data.free()
    _ = mat_a_dev^
    _ = mat_b_dev^
    _ = mat_c_dev^


struct ValOrDim[dim: Dim = Dim()]:
    var value: Int

    fn __init__(out self):
        constrained[
            not dim.is_dynamic(),
            "Can't construct a dynamic dim with no runtime value",
        ]()
        self.value = dim.get()

    @implicit
    fn __init__(out self, v: Int):
        self.value = v


fn static[d: Int]() -> ValOrDim[d]:
    return ValOrDim[d]()


fn dynamic(d: Int) -> ValOrDim:
    return ValOrDim(d)


fn create_matmul_test_case[
    dtype: DType
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim) raises:
    matmul_test_case[
        DType.float32,
        DimList(m.dim, n.dim),
        DimList(m.dim, k.dim),
        DimList(k.dim, n.dim),
    ]((m.value, n.value), (m.value, k.value), (k.value, n.value), ctx)


def main():
    with DeviceContext() as ctx:
        create_matmul_test_case[DType.float32](
            ctx, dynamic(8), static[8](), static[4]()
        )
        create_matmul_test_case[DType.float32](
            ctx, dynamic(16), static[16](), static[8]()
        )
