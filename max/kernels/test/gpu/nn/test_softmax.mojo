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

from math import isclose
from random import rand, random_float64, seed
from sys import has_amd_gpu_accelerator

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE
from gpu.host import DeviceContext
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
from nn.softmax import _online_softmax_kernel, _softmax_cpu, _softmax_gpu
from testing import assert_almost_equal, assert_true

from utils import IndexList


fn test_gpu_softmax(ctx: DeviceContext) raises:
    print("== test_gpu_softmax")

    alias type = DType.float32
    alias rank = 3
    var shape = IndexList[rank](3, 5, 515)
    var in_host_ptr = UnsafePointer[Scalar[type]].alloc(
        shape.flattened_length()
    )
    var in_device_ptr = ctx.enqueue_create_buffer[type](
        shape.flattened_length()
    )
    var in_host = NDBuffer[type, rank](in_host_ptr, shape)
    var in_device = NDBuffer[type, rank](in_device_ptr.unsafe_ptr(), shape)
    var out_host_ptr = UnsafePointer[Scalar[type]].alloc(
        shape.flattened_length()
    )
    var out_ref_ptr = UnsafePointer[Scalar[type]].alloc(
        shape.flattened_length()
    )
    var out_device_ptr = ctx.enqueue_create_buffer[type](
        shape.flattened_length()
    )
    var out_host = NDBuffer[type, rank](out_host_ptr, shape)
    var out_ref = NDBuffer[type, rank](out_ref_ptr, shape)
    var out_device = NDBuffer[type, rank](out_device_ptr.unsafe_ptr(), shape)

    rand[type](in_host_ptr, shape.flattened_length())
    ctx.enqueue_copy(in_device_ptr, in_host_ptr)

    @parameter
    @__copy_capture(in_device)
    fn input_fn_device[
        _simd_width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[type, _simd_width]:
        return in_device.load[width=_simd_width](
            rebind[IndexList[rank]](coords)
        )

    @parameter
    @__copy_capture(in_host)
    fn input_fn_host[
        _simd_width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[type, _simd_width]:
        return in_host.load[width=_simd_width](rebind[IndexList[rank]](coords))

    _softmax_gpu[
        type, 1, rank, DimList.create_unknown[rank](), input_fn_device
    ](shape, out_device, rank - 1, ctx)

    _softmax_cpu[
        type,
        1,
        rank,
        DimList.create_unknown[rank](),
        __origin_of(),
        input_fn_host,
    ](shape, out_ref, rank - 1)

    ctx.synchronize()
    ctx.enqueue_copy(out_host_ptr, out_device_ptr)

    for i in range(shape.flattened_length()):
        if not isclose(
            out_ref.flatten()[i],
            out_host.flatten()[i],
            atol=1e-4,
            rtol=1e-5,
        ):
            print("ERROR. Mismatch at flattened idx:", i)
            assert_true(False)

    in_host_ptr.free()
    out_host_ptr.free()
    out_ref_ptr.free()

    _ = in_device
    _ = in_host
    _ = in_device_ptr
    _ = out_device_ptr


def test_gpu_softmax_half[test_type: DType](ctx: DeviceContext):
    print("== test_gpu_softmax_half")
    alias seed_val = 42
    seed(seed_val)

    alias ref_type = DType.float32
    alias rank = 3

    var shape = IndexList[rank](3, 5, 515)
    var length = shape.flattened_length()

    var in_host_ref_ptr = UnsafePointer[Scalar[ref_type]].alloc(length)
    var in_device_ref_ptr = ctx.enqueue_create_buffer[ref_type](length)
    var in_host_test_ptr = UnsafePointer[Scalar[test_type]].alloc(length)
    var in_device_test_ptr = ctx.enqueue_create_buffer[test_type](length)
    var in_device_ref = NDBuffer[ref_type, rank](
        in_device_ref_ptr.unsafe_ptr(), shape
    )
    var in_device_test = NDBuffer[test_type, rank](
        in_device_test_ptr.unsafe_ptr(), shape
    )

    var out_host_ref_ptr = UnsafePointer[Scalar[ref_type]].alloc(length)
    var out_device_ref_ptr = ctx.enqueue_create_buffer[ref_type](length)
    var out_host_test_ptr = UnsafePointer[Scalar[test_type]].alloc(length)
    var out_device_test_ptr = ctx.enqueue_create_buffer[test_type](length)

    var out_device_ref = NDBuffer[ref_type, rank](
        out_device_ref_ptr.unsafe_ptr(), shape
    )
    var out_device_test = NDBuffer[test_type, rank](
        out_device_test_ptr.unsafe_ptr(), shape
    )

    # first fill BF16 pointer with random values, then cast to FP32 to
    # circumvent precision loss on casting of input. Skew the values to simulate
    # precision loss
    for i in range(length):
        # TODO use randn when GCC Float64 -> Float16 truncation is fixed #33932
        in_host_test_ptr[i] = (
            random_float64(1, 10).cast[DType.float32]().cast[test_type]()
        )
        in_host_ref_ptr[i] = in_host_test_ptr[i].cast[ref_type]()

    ctx.enqueue_copy(in_device_test_ptr, in_host_test_ptr)
    ctx.enqueue_copy(in_device_ref_ptr, in_host_ref_ptr)

    @parameter
    @__copy_capture(in_device_ref)
    fn input_fn_ref[
        _simd_width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[ref_type, _simd_width]:
        return in_device_ref[rebind[IndexList[rank]](coords)]

    @parameter
    @__copy_capture(in_device_test)
    fn input_fn_test[
        _simd_width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[test_type, _simd_width]:
        return in_device_test[rebind[IndexList[rank]](coords)]

    _softmax_gpu[
        ref_type,
        1,
        rank,
        DimList.create_unknown[rank](),
        input_fn_ref,
    ](shape, out_device_ref, rank - 1, ctx)

    _softmax_gpu[
        test_type,
        1,
        rank,
        DimList.create_unknown[rank](),
        input_fn_test,
    ](shape, out_device_test, rank - 1, ctx)

    ctx.synchronize()
    ctx.enqueue_copy(out_host_ref_ptr, out_device_ref_ptr)
    ctx.enqueue_copy(out_host_test_ptr, out_device_test_ptr)

    for i in range(length):
        var ref_val = out_host_ref_ptr[i]
        var test_val = out_host_test_ptr[i].cast[ref_type]()
        assert_almost_equal(ref_val, test_val, atol=1e-2)

    _ = in_device_ref_ptr
    _ = in_device_test_ptr
    _ = in_device_test
    _ = in_device_ref


fn test_gpu_online_softmax[
    WM: Int, WN: Int, transpose_fragments: Bool
](ctx: DeviceContext) raises:
    print("== test_online_softmax")

    alias type = DType.float32
    alias rank = 3
    alias seqlen = 256

    # For testing purpose, call online softmax twice and each time updates half
    # seq_len. Limit to WM rows and arrange warps in N dim.
    alias shape = IndexList[rank](1, WM, seqlen)
    alias num_warps = seqlen // (2 * WN)
    alias num_threads = num_warps * WARP_SIZE

    var in_host_ptr = UnsafePointer[Scalar[type]].alloc(
        shape.flattened_length()
    )
    var out_host_ptr = UnsafePointer[Scalar[type]].alloc(
        shape.flattened_length()
    )
    var out_ref_ptr = UnsafePointer[Scalar[type]].alloc(
        shape.flattened_length()
    )

    var in_host = NDBuffer[type, rank](in_host_ptr, shape)
    var out_ref = NDBuffer[type, rank](out_ref_ptr, shape)

    var in_device_ptr = ctx.enqueue_create_buffer[type](
        shape.flattened_length()
    )
    var out_device_ptr = ctx.enqueue_create_buffer[type](
        shape.flattened_length()
    )

    var in_device = LayoutTensor[type, Layout.row_major(shape[1], shape[2])](
        in_device_ptr
    )
    var out_device = LayoutTensor[type, Layout.row_major(shape[1], shape[2])](
        out_device_ptr
    )

    rand[type](in_host_ptr, shape.flattened_length())

    ctx.enqueue_copy(in_device_ptr, in_host_ptr)

    ctx.enqueue_function[
        _online_softmax_kernel[
            WM,
            WN,
            DType.float32,
            Layout.row_major(shape[1], shape[2]),
            transpose_fragments,
        ],
    ](
        in_device,
        out_device,
        grid_dim=1,
        block_dim=num_threads,
    )

    @parameter
    @__copy_capture(in_host)
    fn input_fn_host[
        _simd_width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[type, _simd_width]:
        return in_host.load[width=_simd_width](rebind[IndexList[rank]](coords))

    _softmax_cpu[
        type,
        1,
        rank,
        DimList.create_unknown[rank](),
        __origin_of(),
        input_fn_host,
    ](shape, out_ref, rank - 1)

    ctx.synchronize()
    ctx.enqueue_copy(out_host_ptr, out_device_ptr)

    for i in range(shape.flattened_length()):
        assert_almost_equal(
            out_host_ptr[i], out_ref_ptr[i], atol=1e-4, rtol=1e-5
        )

    in_host_ptr.free()
    out_host_ptr.free()
    out_ref_ptr.free()

    _ = in_device_ptr
    _ = out_device_ptr


def main():
    with DeviceContext() as ctx:
        test_gpu_softmax(ctx)
        test_gpu_softmax_half[DType.bfloat16](ctx)
        test_gpu_softmax_half[DType.float16](ctx)
        # Test general online-softmax, communicating data via shared memory.

        test_gpu_online_softmax[32, 32, False](ctx)
        # Test covering entire row within one warp
        test_gpu_online_softmax[16, 128, False](ctx)

        @parameter
        if has_amd_gpu_accelerator():
            test_gpu_online_softmax[32, 32, True](ctx)
            # Test covering entire row within one warp
            test_gpu_online_softmax[16, 128, True](ctx)
