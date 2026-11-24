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

# FP8 <-> float32 conversions on Apple Metal via the air.convert AIR
# intrinsics. e5m2 has inf/nan (overflow -> inf); e4m3fn is finite-only
# and non-saturating (overflow -> nan). FP8 conversion is M5-only.

# REQUIRES: APPLE-GPU
# RUN: %mojo %s | FileCheck %s

from std.ffi import external_call
from std.sys.info import _accelerator_arch

from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from std.memory import Pointer


@__name(t"_fp8e5m2_to_f32_kernel")
def _fp8e5m2_to_f32_kernel(
    in_ptr: Pointer[Float8_e5m2, MutAnyOrigin],
    out_ptr: Pointer[Float32, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    out_ptr[unsafe_offset=tid] = external_call[
        "air.convert.f.f32.f.f8e5m2",
        Float32,
    ](in_ptr[unsafe_offset=tid])


@__name(t"_f32_to_fp8e5m2_kernel")
def _f32_to_fp8e5m2_kernel(
    in_ptr: Pointer[Float32, MutAnyOrigin],
    out_ptr: Pointer[Float8_e5m2, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    out_ptr[unsafe_offset=tid] = external_call[
        "air.convert.f.f8e5m2.f.f32",
        Float8_e5m2,
    ](in_ptr[unsafe_offset=tid])


@__name(t"_e4m3fn_to_f32_kernel")
def _e4m3fn_to_f32_kernel(
    in_ptr: Pointer[Float8_e4m3fn, MutAnyOrigin],
    out_ptr: Pointer[Float32, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    out_ptr[unsafe_offset=tid] = external_call[
        "air.convert.f.f32.f.f8e4m3fn",
        Float32,
    ](in_ptr[unsafe_offset=tid])


@__name(t"_f32_to_e4m3fn_kernel")
def _f32_to_e4m3fn_kernel(
    in_ptr: Pointer[Float32, MutAnyOrigin],
    out_ptr: Pointer[Float8_e4m3fn, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    out_ptr[unsafe_offset=tid] = external_call[
        "air.convert.f.f8e4m3fn.f.f32",
        Float8_e4m3fn,
    ](in_ptr[unsafe_offset=tid])


def _skip(name: String):
    print("==", name)
    print("SKIP: requires Apple M5 + Metal 4")


def _skip_all():
    _skip("fp8e5m2_to_f32")
    _skip("f32_to_fp8e5m2")
    _skip("f32_to_fp8e5m2_to_f32")
    _skip("f32_to_e4m3fn")
    _skip("f32_to_e4m3fn_to_f32")


def main() raises:
    comptime N = 4

    comptime if "metal" not in _accelerator_arch():
        _skip_all()
        return

    var ctx = DeviceContext()
    if ctx.compute_capability() < 5:
        _skip_all()
        return

    var fp8_in = ctx.enqueue_create_buffer[.float8_e5m2](N)
    var f32_out = ctx.enqueue_create_buffer[.float32](N)

    with fp8_in.map_to_host() as h:
        h[0] = Float8_e5m2(0.0)
        h[1] = Float8_e5m2(1.0)
        h[2] = Float8_e5m2(2.0)
        h[3] = Float8_e5m2(0.5)

    ctx.enqueue_function[_fp8e5m2_to_f32_kernel](
        fp8_in.unsafe_ptr(),
        f32_out.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(N, 1, 1),
    )

    var result_f32 = List[Float32](length=N, fill=Float32(0))
    ctx.enqueue_copy(dst_ptr=result_f32.unsafe_ptr(), src_buf=f32_out)
    ctx.synchronize()

    # CHECK-LABEL: fp8e5m2_to_f32
    # CHECK: {{0.0 1.0 2.0 0.5|SKIP}}
    print("== fp8e5m2_to_f32")
    print(result_f32[0], result_f32[1], result_f32[2], result_f32[3])

    var f32_in = ctx.enqueue_create_buffer[.float32](N)
    var fp8_out = ctx.enqueue_create_buffer[.float8_e5m2](N)

    with f32_in.map_to_host() as h:
        h[0] = Float32(0.0)
        h[1] = Float32(1024.0)
        h[2] = Float32(16384.0)
        h[3] = Float32(65536.5)  # overflow -> inf

    ctx.enqueue_function[_f32_to_fp8e5m2_kernel](
        f32_in.unsafe_ptr(),
        fp8_out.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(N, 1, 1),
    )

    var result_fp8 = List[Float8_e5m2](length=N, fill=Float8_e5m2(0))
    ctx.enqueue_copy(dst_ptr=result_fp8.unsafe_ptr(), src_buf=fp8_out)
    ctx.synchronize()

    # CHECK-LABEL: f32_to_fp8e5m2
    # CHECK: {{0.0 1024.0 16384.0 inf|SKIP}}
    print("== f32_to_fp8e5m2")
    print(
        result_fp8[0].cast[.float32](),
        result_fp8[1].cast[.float32](),
        result_fp8[2].cast[.float32](),
        result_fp8[3].cast[.float32](),
    )

    var f32_rt = ctx.enqueue_create_buffer[.float32](N)

    ctx.enqueue_function[_fp8e5m2_to_f32_kernel](
        fp8_out.unsafe_ptr(),
        f32_rt.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(N, 1, 1),
    )

    var result_rt = List[Float32](length=N, fill=Float32(0))
    ctx.enqueue_copy(dst_ptr=result_rt.unsafe_ptr(), src_buf=f32_rt)
    ctx.synchronize()

    # CHECK-LABEL: f32_to_fp8e5m2_to_f32
    # CHECK: {{0.0 1024.0 16384.0 inf|SKIP}}
    print("== f32_to_fp8e5m2_to_f32")
    print(result_rt[0], result_rt[1], result_rt[2], result_rt[3])

    var f32_in_e4 = ctx.enqueue_create_buffer[.float32](N)
    var e4m3fn_out = ctx.enqueue_create_buffer[.float8_e4m3fn](N)

    with f32_in_e4.map_to_host() as h:
        h[0] = Float32(0.5)
        h[1] = Float32(256.0)
        h[2] = Float32(448.0)  # max finite e4m3fn
        h[3] = Float32(1000.0)  # overflow -> nan

    ctx.enqueue_function[_f32_to_e4m3fn_kernel](
        f32_in_e4.unsafe_ptr(),
        e4m3fn_out.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(N, 1, 1),
    )

    var result_e4m3fn = List[Float8_e4m3fn](length=N, fill=Float8_e4m3fn(0))
    ctx.enqueue_copy(dst_ptr=result_e4m3fn.unsafe_ptr(), src_buf=e4m3fn_out)
    ctx.synchronize()

    # CHECK-LABEL: f32_to_e4m3fn
    # CHECK: {{0.5 256.0 448.0 nan|SKIP}}
    print("== f32_to_e4m3fn")
    print(
        result_e4m3fn[0].cast[.float32](),
        result_e4m3fn[1].cast[.float32](),
        result_e4m3fn[2].cast[.float32](),
        result_e4m3fn[3].cast[.float32](),
    )

    var f32_rt_e4 = ctx.enqueue_create_buffer[.float32](N)

    ctx.enqueue_function[_e4m3fn_to_f32_kernel](
        e4m3fn_out.unsafe_ptr(),
        f32_rt_e4.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(N, 1, 1),
    )

    var result_rt_e4 = List[Float32](length=N, fill=Float32(0))
    ctx.enqueue_copy(dst_ptr=result_rt_e4.unsafe_ptr(), src_buf=f32_rt_e4)
    ctx.synchronize()

    # CHECK-LABEL: f32_to_e4m3fn_to_f32
    # CHECK: {{0.5 256.0 448.0 nan|SKIP}}
    print("== f32_to_e4m3fn_to_f32")
    print(result_rt_e4[0], result_rt_e4[1], result_rt_e4[2], result_rt_e4[3])
