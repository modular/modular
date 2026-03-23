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

from std.algorithm.functional import stencil, stencil_gpu
from buffer import NDBuffer
from buffer.dimlist import DimList
from std.gpu.host import DeviceContext
from layout import Layout
from layout._utils import ManagedLayoutTensor
from std.testing import assert_almost_equal

from std.utils import IndexList
from std.utils.numerics import min_or_neg_inf

comptime _map_fn_type = def[rank: Int](IndexList[rank]) capturing -> Tuple[
    IndexList[rank],
    IndexList[rank],
]
comptime load_fn_type = def[dtype: DType, rank: Int, simd_width: Int](
    IndexList[rank]
) capturing -> SIMD[dtype, simd_width]


def fill_buffer[
    dtype: DType, rank: Int, shape: DimList
](buf: NDBuffer[mut=True, rank=rank, dtype, _, shape]):
    var s: Int = 1
    for i in range(buf.get_rank()):
        s *= buf.dim(i)

    for j in range(s):
        buf.flatten()[j] = Scalar[dtype](j) + 1


def assert_allclose[
    dtype: DType, rank: Int, shape: DimList
](
    h_output_ref: NDBuffer[rank=rank, dtype, _, shape],
    h_output_gpu: NDBuffer[rank=rank, dtype, _, shape],
) raises:
    var shape_ = h_output_ref.get_shape()
    for i in range(shape_.flattened_length()):
        assert_almost_equal(h_output_ref.data[i], h_output_gpu.data[i])


def test_stencil_avg_pool(ctx: DeviceContext) raises:
    print("== test_stencil_avg_pool")
    comptime rank = 4
    comptime stencil_rank = 2
    comptime dtype = DType.float32
    comptime simd_width = 1

    comptime input_width = 5
    comptime input_height = 5

    comptime stride = 1
    comptime pool_window_h = 3
    comptime pool_window_w = 3
    comptime dilation = 1

    comptime input_shape = DimList[1, input_height, input_width, 1]()
    var input_shape_dyn = IndexList[4](1, input_height, input_width, 1)
    comptime output_height = input_height - pool_window_h + 1
    comptime output_width = input_width - pool_window_w + 1
    comptime output_shape = DimList[1, output_height, output_width, 1]()
    var output_shape_dyn = IndexList[4](1, output_height, output_width, 1)

    var d_input_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, input_height, input_width, 1)
    ](ctx)
    var d_output_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, output_height, output_width, 1)
    ](ctx)

    var h_input = NDBuffer[rank=rank, dtype, MutAnyOrigin, input_shape](
        d_input_managed.tensor[update=False]().ptr, input_shape_dyn
    )
    var h_output = NDBuffer[rank=rank, dtype, MutAnyOrigin, output_shape](
        d_output_managed.tensor[update=False]().ptr, output_shape_dyn
    )
    var h_output_ref = NDBuffer[
        rank=rank, dtype, MutAnyOrigin, output_shape
    ].stack_allocation()

    fill_buffer(h_input)
    h_output.fill(0)
    h_output_ref.fill(0)

    var d_input = NDBuffer[rank=rank, dtype](
        d_input_managed.device_tensor().ptr, input_shape_dyn
    )
    var d_output = NDBuffer[rank=rank, dtype](
        d_output_managed.device_tensor().ptr, output_shape_dyn
    )

    @parameter
    def map_fn[
        rank: Int
    ](point: IndexList[stencil_rank, ...]) -> Tuple[
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ]:
        var lower_bound = IndexList[stencil_rank](point[0], point[1])
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h, point[1] + pool_window_w
        )
        return lower_bound, upper_bound

    @always_inline
    @__copy_capture(d_input)
    @parameter
    def load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load[width=simd_width](point)
        )

    @always_inline
    @parameter
    def avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    @parameter
    def avg_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    @parameter
    def dilation_fn(dim: Int) -> Int:
        return 1

    @always_inline
    @__copy_capture(d_output)
    @parameter
    def avg_pool_compute_finalize_gpu[
        simd_width: Int
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        d_output.store(point, res)

    comptime stencil_axis = IndexList[stencil_rank](1, 2)
    stencil_gpu[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn_gpu,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize_gpu,
    ](ctx, d_output.get_shape(), d_input.get_shape())

    # Refresh host view; tensor() handles device-to-host transfer and sync.
    h_output = NDBuffer[rank=rank, dtype, MutAnyOrigin, output_shape](
        d_output_managed.tensor().ptr, output_shape_dyn
    )

    # Reference implementation on CPU (unified closures for CPU stencil)
    def map_fn_cpu(
        point: IndexList[stencil_rank, ...],
    ) unified {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](point[0], point[1])
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h, point[1] + pool_window_w
        )
        return lower_bound, upper_bound

    def dilation_fn_cpu(dim: Int) unified {} -> Int:
        return 1

    @always_inline
    def load_fn_ref[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) unified {
        var h_input,
    } -> SIMD[
        dtype, simd_width
    ]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load[width=simd_width](point)
        )

    def avg_pool_compute_init_cpu[
        simd_width: Int
    ]() unified {} -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    def avg_pool_compute_cpu[
        simd_width: Int
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) unified {} -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    def avg_pool_compute_finalize_ref[
        simd_width: Int
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) unified {
        var h_output_ref,
    }:
        var res = val / (pool_window_h * pool_window_w)
        h_output_ref.store(point, res)

    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
    ](
        h_output_ref.get_shape(),
        h_input.get_shape(),
        map_fn_cpu,
        dilation_fn_cpu,
        load_fn_ref,
        avg_pool_compute_init_cpu,
        avg_pool_compute_cpu,
        avg_pool_compute_finalize_ref,
    )

    assert_allclose(h_output_ref, h_output)

    for i in range(0, output_height):
        for j in range(0, output_width):
            print(h_output[0, i, j, 0], "\t", end="")
        print("")


def test_stencil_avg_pool_padded(ctx: DeviceContext) raises:
    print("== test_stencil_avg_pool_padded")
    comptime rank = 4
    comptime stencil_rank = 2
    comptime dtype = DType.float32
    comptime simd_width = 1

    comptime input_width = 5
    comptime input_height = 5

    comptime stride = 1
    comptime pool_window_h = 5
    comptime pool_window_w = 5
    comptime dilation = 1
    comptime pad_h = 2
    comptime pad_w = 2

    comptime input_shape = DimList[1, input_height, input_width, 1]()
    var input_shape_dyn = IndexList[4](1, input_height, input_width, 1)
    comptime output_height = input_height - pool_window_h + pad_h * 2 + 1
    comptime output_width = input_width - pool_window_w + pad_w * 2 + 1
    comptime output_shape = DimList[1, output_height, output_width, 1]()
    var output_shape_dyn = IndexList[4](1, output_height, output_width, 1)

    var d_input_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, input_height, input_width, 1)
    ](ctx)
    var d_output_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, output_height, output_width, 1)
    ](ctx)

    var h_input = NDBuffer[rank=rank, dtype, MutAnyOrigin, input_shape](
        d_input_managed.tensor[update=False]().ptr, input_shape_dyn
    )
    var h_output = NDBuffer[rank=rank, dtype, MutAnyOrigin, output_shape](
        d_output_managed.tensor[update=False]().ptr, output_shape_dyn
    )
    var h_output_ref = NDBuffer[
        rank=rank, dtype, MutAnyOrigin, output_shape
    ].stack_allocation()
    h_output_ref.fill(0)

    fill_buffer(h_input)
    h_output.fill(0)

    var d_input = NDBuffer[rank=rank, dtype](
        d_input_managed.device_tensor().ptr, input_shape_dyn
    )
    var d_output = NDBuffer[rank=rank, dtype](
        d_output_managed.device_tensor().ptr, output_shape_dyn
    )

    @parameter
    def map_fn[
        rank: Int
    ](point: IndexList[stencil_rank, ...]) -> Tuple[
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ]:
        var lower_bound = IndexList[stencil_rank](
            point[0] - pad_h, point[1] - pad_w
        )
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h - pad_h, point[1] + pool_window_w - pad_w
        )
        return lower_bound, upper_bound

    @always_inline
    @__copy_capture(d_input)
    @parameter
    def load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load[width=simd_width](point)
        )

    @always_inline
    @parameter
    def avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    @parameter
    def avg_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    @parameter
    def dilation_fn(dim: Int) -> Int:
        return 1

    @always_inline
    @__copy_capture(d_output)
    @parameter
    def avg_pool_compute_finalize_gpu[
        simd_width: Int
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        d_output.store(point, res)

    comptime stencil_axis = IndexList[stencil_rank](1, 2)
    stencil_gpu[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn_gpu,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize_gpu,
    ](ctx, d_output.get_shape(), d_input.get_shape())

    # Refresh host view; tensor() handles device-to-host transfer and sync.
    h_output = NDBuffer[rank=rank, dtype, MutAnyOrigin, output_shape](
        d_output_managed.tensor().ptr, output_shape_dyn
    )

    # Reference implementation on CPU (unified closures for CPU stencil)
    def map_fn_cpu(
        point: IndexList[stencil_rank, ...],
    ) unified {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] - pad_h, point[1] - pad_w
        )
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h - pad_h, point[1] + pool_window_w - pad_w
        )
        return lower_bound, upper_bound

    def dilation_fn_cpu(dim: Int) unified {} -> Int:
        return 1

    @always_inline
    def load_fn_ref[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) unified {
        var h_input,
    } -> SIMD[
        dtype, simd_width
    ]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load[width=simd_width](point)
        )

    def avg_pool_compute_init_cpu[
        simd_width: Int
    ]() unified {} -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    def avg_pool_compute_cpu[
        simd_width: Int
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) unified {} -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    def avg_pool_compute_finalize_ref[
        simd_width: Int
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) unified {
        var h_output_ref,
    }:
        var res = val / (pool_window_h * pool_window_w)
        h_output_ref.store(point, res)

    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
    ](
        h_output_ref.get_shape(),
        h_input.get_shape(),
        map_fn_cpu,
        dilation_fn_cpu,
        load_fn_ref,
        avg_pool_compute_init_cpu,
        avg_pool_compute_cpu,
        avg_pool_compute_finalize_ref,
    )

    # Ensure results match expected
    assert_allclose(h_output_ref, h_output)

    for i in range(0, output_height):
        for j in range(0, output_width):
            print(h_output[0, i, j, 0], "\t", end="")
        print("")


def test_stencil_avg_pool_stride_2(ctx: DeviceContext) raises:
    print("== test_stencil_avg_pool_stride_2")
    comptime rank = 4
    comptime stencil_rank = 2
    comptime dtype = DType.float32
    comptime simd_width = 1

    comptime input_width = 7
    comptime input_height = 7

    comptime stride = 2
    comptime pool_window_h = 3
    comptime pool_window_w = 3
    comptime dilation = 1

    comptime input_shape = DimList[1, input_height, input_width, 1]()
    var input_shape_dyn = IndexList[4](1, input_height, input_width, 1)
    comptime output_height = (input_height - pool_window_h) // stride + 1
    comptime output_width = (input_width - pool_window_w) // stride + 1
    comptime output_shape = DimList[1, output_height, output_width, 1]()
    var output_shape_dyn = IndexList[4](1, output_height, output_width, 1)

    var d_input_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, input_height, input_width, 1)
    ](ctx)
    var d_output_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, output_height, output_width, 1)
    ](ctx)

    var h_input = NDBuffer[rank=rank, dtype, MutAnyOrigin, input_shape](
        d_input_managed.tensor[update=False]().ptr, input_shape_dyn
    )
    var h_output = NDBuffer[rank=rank, dtype, MutAnyOrigin, output_shape](
        d_output_managed.tensor[update=False]().ptr, output_shape_dyn
    )
    var h_output_ref = NDBuffer[
        rank=rank, dtype, MutAnyOrigin, output_shape
    ].stack_allocation()
    h_output_ref.fill(0)

    fill_buffer(h_input)
    h_output.fill(0)

    var d_input = NDBuffer[rank=rank, dtype](
        d_input_managed.device_tensor().ptr, input_shape_dyn
    )
    var d_output = NDBuffer[rank=rank, dtype](
        d_output_managed.device_tensor().ptr, output_shape_dyn
    )

    @parameter
    def map_fn[
        rank: Int
    ](point: IndexList[stencil_rank, ...]) -> Tuple[
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ]:
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride, point[1] * stride
        )
        var upper_bound = IndexList[stencil_rank](
            point[0] * stride + pool_window_h,
            point[1] * stride + pool_window_w,
        )
        return lower_bound, upper_bound

    @always_inline
    @__copy_capture(d_input)
    @parameter
    def load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load[width=simd_width](point)
        )

    @always_inline
    @parameter
    def avg_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    @always_inline
    @parameter
    def avg_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    @parameter
    def dilation_fn(dim: Int) -> Int:
        return 1

    @always_inline
    @__copy_capture(d_output)
    @parameter
    def avg_pool_compute_finalize_gpu[
        simd_width: Int
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]):
        var res = val / (pool_window_h * pool_window_w)
        d_output.store(point, res)

    comptime stencil_axis = IndexList[stencil_rank](1, 2)
    stencil_gpu[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn_gpu,
        avg_pool_compute_init,
        avg_pool_compute,
        avg_pool_compute_finalize_gpu,
    ](ctx, d_output.get_shape(), d_input.get_shape())

    # Refresh host view; tensor() handles device-to-host transfer and sync.
    h_output = NDBuffer[rank=rank, dtype, MutAnyOrigin, output_shape](
        d_output_managed.tensor().ptr, output_shape_dyn
    )

    # Reference implementation on CPU (unified closures for CPU stencil)
    def map_fn_cpu(
        point: IndexList[stencil_rank, ...],
    ) unified {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride, point[1] * stride
        )
        var upper_bound = IndexList[stencil_rank](
            point[0] * stride + pool_window_h,
            point[1] * stride + pool_window_w,
        )
        return lower_bound, upper_bound

    def dilation_fn_cpu(dim: Int) unified {} -> Int:
        return 1

    @always_inline
    def load_fn_ref[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) unified {
        var h_input,
    } -> SIMD[
        dtype, simd_width
    ]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load[width=simd_width](point)
        )

    def avg_pool_compute_init_cpu[
        simd_width: Int
    ]() unified {} -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    def avg_pool_compute_cpu[
        simd_width: Int
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) unified {} -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    def avg_pool_compute_finalize_ref[
        simd_width: Int
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) unified {
        var h_output_ref,
    }:
        var res = val / (pool_window_h * pool_window_w)
        h_output_ref.store(point, res)

    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
    ](
        h_output_ref.get_shape(),
        h_input.get_shape(),
        map_fn_cpu,
        dilation_fn_cpu,
        load_fn_ref,
        avg_pool_compute_init_cpu,
        avg_pool_compute_cpu,
        avg_pool_compute_finalize_ref,
    )

    # Ensure results match expected
    assert_allclose(h_output_ref, h_output)

    for i in range(0, output_height):
        for j in range(0, output_width):
            print(h_output[0, i, j, 0], "\t", end="")
        print("")


def test_stencil_gpu_max_pool(ctx: DeviceContext) raises:
    print("== test_stencil_gpu_max_pool")
    comptime rank = 4
    comptime stencil_rank = 2
    comptime dtype = DType.float32
    comptime simd_width = 1

    comptime input_width = 7
    comptime input_height = 7

    comptime stride = 1
    comptime pool_window_h = 3
    comptime pool_window_w = 3
    comptime dilation = 1

    comptime input_shape = DimList[1, input_height, input_width, 1]()
    var input_shape_dyn = IndexList[4](1, input_height, input_width, 1)
    comptime output_height = (
        input_height - pool_window_h - (pool_window_h - 1) * (dilation - 1)
    ) // stride + 1
    comptime output_width = (
        input_width - pool_window_w - (pool_window_w - 1) * (dilation - 1)
    ) // stride + 1

    comptime output_shape = DimList[1, output_height, output_width, 1]()
    var output_shape_dyn = IndexList[4](1, output_height, output_width, 1)

    var pad_value = 0

    var d_input_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, input_height, input_width, 1)
    ](ctx)
    var d_output_managed = ManagedLayoutTensor[
        dtype, Layout.row_major(1, output_height, output_width, 1)
    ](ctx)

    var h_input = NDBuffer[rank=rank, dtype, MutAnyOrigin, input_shape](
        d_input_managed.tensor[update=False]().ptr, input_shape_dyn
    )
    var h_output = NDBuffer[rank=rank, dtype, MutAnyOrigin, output_shape](
        d_output_managed.tensor[update=False]().ptr, output_shape_dyn
    )
    var h_output_ref = NDBuffer[
        rank=rank, dtype, MutAnyOrigin, output_shape
    ].stack_allocation()
    h_output_ref.fill(0)

    fill_buffer(h_input)
    h_output.fill(0)

    var d_input = NDBuffer[rank=rank, dtype](
        d_input_managed.device_tensor().ptr, input_shape_dyn
    )
    var d_output = NDBuffer[rank=rank, dtype](
        d_output_managed.device_tensor().ptr, output_shape_dyn
    )

    @parameter
    def map_fn[
        rank: Int
    ](point: IndexList[stencil_rank, ...]) -> Tuple[
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ]:
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride, point[1] * stride
        )
        var upper_bound = IndexList[stencil_rank](
            (point[0] * stride + pool_window_h * dilation),
            (point[1] * stride + pool_window_w * dilation),
        )
        return lower_bound, upper_bound

    @always_inline
    @__copy_capture(d_input)
    @parameter
    def load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load[width=simd_width](point)
        )

    @always_inline
    @parameter
    def max_pool_compute_init[simd_width: Int]() -> SIMD[dtype, simd_width]:
        return min_or_neg_inf[dtype]()

    @always_inline
    @parameter
    def max_pool_compute[
        simd_width: Int
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return max(val, result)

    @always_inline
    @__copy_capture(d_output)
    @parameter
    def max_pool_compute_finalize[
        simd_width: Int
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]):
        d_output.store(point, val)

    @always_inline
    @parameter
    def dilation_fn(dim: Int) -> Int:
        return dilation

    comptime stencil_axis = IndexList[stencil_rank](1, 2)
    stencil_gpu[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
        map_fn[stencil_rank],
        dilation_fn,
        load_fn_gpu,
        max_pool_compute_init,
        max_pool_compute,
        max_pool_compute_finalize,
    ](ctx, d_output.get_shape(), d_input.get_shape())

    # Refresh host view; tensor() handles device-to-host transfer and sync.
    h_output = NDBuffer[rank=rank, dtype, MutAnyOrigin, output_shape](
        d_output_managed.tensor().ptr, output_shape_dyn
    )
    # ctx.enqueue_copy(h_input.data, d_input_buf)

    # Reference implementation on CPU (unified closures for CPU stencil)
    def map_fn_cpu(
        point: IndexList[stencil_rank, ...],
    ) unified {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] * stride, point[1] * stride
        )
        var upper_bound = IndexList[stencil_rank](
            (point[0] * stride + pool_window_h * dilation),
            (point[1] * stride + pool_window_w * dilation),
        )
        return lower_bound, upper_bound

    def dilation_fn_cpu(dim: Int) unified {} -> Int:
        return dilation

    @always_inline
    def load_fn_ref[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) unified {
        var h_input,
    } -> SIMD[
        dtype, simd_width
    ]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load[width=simd_width](point)
        )

    def max_pool_compute_init_cpu[
        simd_width: Int
    ]() unified {} -> SIMD[dtype, simd_width]:
        return min_or_neg_inf[dtype]()

    def max_pool_compute_cpu[
        simd_width: Int
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) unified {} -> SIMD[dtype, simd_width]:
        return max(val, result)

    @always_inline
    def max_pool_compute_finalize_ref[
        simd_width: Int
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) unified {
        var h_output_ref,
    }:
        h_output_ref.store(point, val)

    stencil[
        rank,
        stencil_rank,
        stencil_axis,
        simd_width,
        dtype,
    ](
        h_output_ref.get_shape(),
        h_input.get_shape(),
        map_fn_cpu,
        dilation_fn_cpu,
        load_fn_ref,
        max_pool_compute_init_cpu,
        max_pool_compute_cpu,
        max_pool_compute_finalize_ref,
    )

    # Ensure results match expected
    assert_allclose(h_output_ref, h_output)

    for i in range(0, output_height):
        for j in range(0, output_width):
            print(h_output[0, i, j, 0], "\t", end="")
        print("")


def main() raises:
    with DeviceContext() as ctx:
        test_stencil_avg_pool(ctx)
        test_stencil_avg_pool_padded(ctx)
        test_stencil_avg_pool_stride_2(ctx)
        test_stencil_gpu_max_pool(ctx)
