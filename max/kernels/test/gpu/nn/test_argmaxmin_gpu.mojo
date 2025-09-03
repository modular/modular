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

from random import random_float64

from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer
from layout import (
    LayoutTensor,
    Layout,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout.int_tuple import fill_like
from nn.argmaxmin import argmax, argmin
from nn.argmaxmin_gpu import argmax_gpu, argmin_gpu
from testing import assert_equal
from utils.index import IndexList


fn test_argmaxmin_gpu[
    dtype: DType,
    output_type: DType,
    fill_fn: fn[rank: Int, dtype: DType] (
        LayoutTensor[mut=True, dtype, **_]
    ) capturing [_] -> None,
    largest: Bool = True,
    rank: Int = 2,
](
    ctx: DeviceContext, N: Int, batch_size: Int = 12, num_batches: Int = 6
) raises:
    # Instantiate data in host memory
    var in_shape: IndexList[rank]
    var out_shape: IndexList[rank]

    @parameter
    if rank == 1:
        out_shape = IndexList[rank](1)
        in_shape = IndexList[rank](N)
    elif rank == 2:
        out_shape = IndexList[rank](batch_size, 1)
        in_shape = IndexList[rank](batch_size, N)
    elif rank == 3:
        out_shape = IndexList[rank](num_batches, batch_size, 1)
        in_shape = IndexList[rank](num_batches, batch_size, N)
    else:
        raise Error("Test case doesn't support rank above 3 (just add it)")

    var in_buffer = HostNDBuffer[dtype, rank](in_shape)
    var out_idxs = HostNDBuffer[output_type, rank](out_shape)

    # Fill the buffer with consecutive values
    fill_fn[rank](in_buffer.to_layout_tensor())

    var device_in = DeviceNDBuffer[dtype, rank](in_shape, ctx=ctx)
    var device_out_idxs = DeviceNDBuffer[output_type, rank](out_shape, ctx=ctx)

    ctx.enqueue_copy(device_in.buffer, in_buffer.tensor.data)

    @parameter
    if largest:
        argmax_gpu(
            ctx,
            device_in.to_layout_tensor(),
            device_out_idxs.to_layout_tensor(),
        )
    else:
        argmin_gpu(
            ctx,
            device_in.to_layout_tensor(),
            device_out_idxs.to_layout_tensor(),
        )

    ctx.enqueue_copy(out_idxs.to_layout_tensor().ptr, device_out_idxs.buffer)
    ctx.synchronize()

    # Test for correctness against CPU reference
    var out_idxs_cpu = HostNDBuffer[DType.int64, rank](out_shape)

    @parameter
    if largest:
        argmax(
            in_buffer.to_layout_tensor(),
            rank - 1,
            out_idxs_cpu.to_layout_tensor(),
        )
    else:
        argmin(
            in_buffer.to_layout_tensor(),
            rank - 1,
            out_idxs_cpu.to_layout_tensor(),
        )

    for i in range(out_idxs_cpu.tensor.num_elements()):
        assert_equal(
            out_idxs.tensor.data[i],
            out_idxs_cpu.tensor.data[i].cast[output_type](),
        )

    _ = device_in
    _ = device_out_idxs


fn _test_argmaxmin_gpu_helper_2[
    idx_type: DType,
    fill_fn: fn[rank: Int, dtype: DType] (
        LayoutTensor[mut=True, dtype, **_]
    ) capturing [_] -> None,
    largest: Bool,
](ctx: DeviceContext) raises:
    test_argmaxmin_gpu[
        DType.float32, idx_type, fill_fn, largest=largest, rank=1
    ](ctx, N=102_400)
    test_argmaxmin_gpu[
        DType.float32, idx_type, fill_fn, largest=largest, rank=2
    ](ctx, N=16_384, batch_size=32)
    test_argmaxmin_gpu[
        DType.float32, idx_type, fill_fn, largest=largest, rank=3
    ](ctx, N=1024, batch_size=12, num_batches=10)


fn test_argmaxmin_gpu_helper[
    idx_type: DType,
    fill_fn: fn[rank: Int, dtype: DType] (
        LayoutTensor[mut=True, dtype, **_]
    ) capturing [_] -> None,
](ctx: DeviceContext) raises:
    # argmax
    _test_argmaxmin_gpu_helper_2[idx_type, fill_fn, largest=True](ctx)

    # argmin
    _test_argmaxmin_gpu_helper_2[idx_type, fill_fn, largest=False](ctx)


def main():
    @parameter
    fn fill_random[
        rank: Int, dtype: DType
    ](buffer: LayoutTensor[mut=True, dtype, **_]):
        alias min_val = -1e9
        alias max_val = 1e9
        var total_elements = buffer.size()
        for i in range(total_elements):
            var random_value = random_float64(min_val, max_val)
            buffer.ptr[i] = random_value.cast[dtype]()

    with DeviceContext() as ctx:  # argmax tests
        # index
        test_argmaxmin_gpu_helper[DType.index, fill_random](ctx)

        # int64
        test_argmaxmin_gpu_helper[DType.int64, fill_random](ctx)

        # int32
        test_argmaxmin_gpu_helper[DType.int32, fill_random](ctx)

        # uint64
        test_argmaxmin_gpu_helper[DType.uint64, fill_random](ctx)
