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

from math import ceildiv
from os.atomic import Atomic, Consistency

from gpu import *
from gpu.host import DeviceContext
from testing import assert_equal, TestSuite
from sys import is_apple_gpu, has_apple_gpu_accelerator


@fieldwise_init
struct FillStrategy(Equatable, ImplicitlyCopyable):
    var value: Int

    comptime LINSPACE = Self(0)
    comptime NEG_LINSPACE = Self(1)
    comptime SYMMETRIC_LINSPACE = Self(2)
    comptime ZEROS = Self(3)
    comptime ONES = Self(4)

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value


fn reduce_add(
    res_add: UnsafePointer[Float32, MutAnyOrigin],
    vec: UnsafePointer[Float32, MutAnyOrigin],
    len: Int,
):
    comptime ord = Consistency.RELEASE if is_apple_gpu() else Consistency.SEQUENTIAL
    var tid = global_idx.x

    if tid >= UInt(len):
        return

    _ = Atomic.fetch_add[ordering=ord](res_add, vec[tid])


fn reduce_min_max(
    res_min: UnsafePointer[Float32, MutAnyOrigin],
    res_max: UnsafePointer[Float32, MutAnyOrigin],
    vec: UnsafePointer[Float32, MutAnyOrigin],
    len: Int,
):
    comptime ord = Consistency.RELEASE if is_apple_gpu() else Consistency.SEQUENTIAL
    var tid = global_idx.x

    if tid >= UInt(len):
        return

    Atomic.min[ordering=ord](res_min, vec[tid])
    Atomic.max[ordering=ord](res_max, vec[tid])


fn run_reduce(fill_strategy: FillStrategy, ctx: DeviceContext) raises:
    comptime BLOCK_SIZE = 32
    comptime n = 1024
    comptime F32 = DType.float32

    var stack = InlineArray[Float32, n](fill=0)
    var vec_host = Span(stack)

    if fill_strategy == FillStrategy.LINSPACE:
        for i in range(n):
            vec_host[i] = Float32(i)
    elif fill_strategy == FillStrategy.NEG_LINSPACE:
        for i in range(n):
            vec_host[i] = Float32(-i)
    elif fill_strategy == FillStrategy.SYMMETRIC_LINSPACE:
        for i in range(n):
            vec_host[i] = Float32(i - n // 2)
    elif fill_strategy == FillStrategy.ZEROS:
        for i in range(n):
            vec_host[i] = 0
    elif fill_strategy == FillStrategy.ONES:
        for i in range(n):
            vec_host[i] = 1

    var vec_device = ctx.enqueue_create_buffer[F32](n)
    vec_device.enqueue_copy_from(vec_host.unsafe_ptr())

    var res_add_device = ctx.enqueue_create_buffer[F32](1)
    res_add_device.enqueue_fill(0)

    ctx.enqueue_function_experimental[reduce_add](
        res_add_device,
        vec_device,
        n,
        grid_dim=ceildiv(n, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )

    var res = Float32(0)
    res_add_device.enqueue_copy_to(UnsafePointer(to=res))

    var res_min = Float32(0)
    var res_max = Float32(0)

    @parameter
    if not has_apple_gpu_accelerator():
        var res_min_device = ctx.enqueue_create_buffer[F32](1)
        res_min_device.enqueue_fill(0)

        var res_max_device = ctx.enqueue_create_buffer[F32](1)
        res_max_device.enqueue_fill(0)
        ctx.enqueue_function_experimental[reduce_min_max](
            res_min_device,
            res_max_device,
            vec_device,
            n,
            grid_dim=ceildiv(n, BLOCK_SIZE),
            block_dim=BLOCK_SIZE,
        )

        res_min_device.enqueue_copy_to(UnsafePointer(to=res_min))
        res_max_device.enqueue_copy_to(UnsafePointer(to=res_max))

    ctx.synchronize()

    if fill_strategy == FillStrategy.LINSPACE:
        assert_equal(res, n * (n - 1) // 2)
        if not has_apple_gpu_accelerator():
            assert_equal(res_min, 0)
            assert_equal(res_max, n - 1)
    elif fill_strategy == FillStrategy.NEG_LINSPACE:
        assert_equal(res, -n * (n - 1) // 2)
        if not has_apple_gpu_accelerator():
            assert_equal(res_min, -n + 1)
            assert_equal(res_max, 0)
    elif fill_strategy == FillStrategy.SYMMETRIC_LINSPACE:
        assert_equal(res, -n // 2)
        if not has_apple_gpu_accelerator():
            assert_equal(res_min, -n // 2)
            assert_equal(res_max, (n - 1) // 2)
    elif fill_strategy == FillStrategy.ZEROS:
        assert_equal(res, 0)
        if not has_apple_gpu_accelerator():
            assert_equal(res_min, 0)
            assert_equal(res_max, 0)
    elif fill_strategy == FillStrategy.ONES:
        assert_equal(res, n)
        if not has_apple_gpu_accelerator():
            assert_equal(res_min, 0)
            assert_equal(res_max, 1)

    _ = vec_device


def test_reduce_atomic():
    with DeviceContext() as ctx:
        run_reduce(FillStrategy.LINSPACE, ctx)
        run_reduce(FillStrategy.NEG_LINSPACE, ctx)
        run_reduce(FillStrategy.SYMMETRIC_LINSPACE, ctx)
        run_reduce(FillStrategy.ZEROS, ctx)
        run_reduce(FillStrategy.ONES, ctx)


def main():
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    # TestSuite.discover_tests[__functions_in_module()]().run()
    var suite = TestSuite()

    suite.test[test_reduce_atomic]()

    suite^.run()
