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

from asyncrt_test_utils import create_test_device_context
from max.gpu import global_idx
from std.sys import size_of
from std.memory import Layout, alloc
from std.testing import TestSuite, assert_equal, assert_true

# Metal aliases whole VM pages, so a wrapped range must be page-aligned and a
# page multiple, and `Layout` carries its alignment as a parameter rather than
# an argument. 16 KiB is the largest page macOS reports and every smaller page
# size divides it, so one constant is page-exact on every host. CUDA and HIP
# accept any range.
comptime _PAGE_BYTES = 16 * 1024
comptime _PageLayout = Layout[Float32, alignment=.of_bytes[_PAGE_BYTES]()]
comptime _COUNT = _PAGE_BYTES // size_of[Float32]()


def test_wrap_host_memory_round_trip() raises:
    var ctx = create_test_device_context()

    var src_alloc = alloc(_PageLayout(count=_COUNT)).into_managed()
    var dst_alloc = alloc(_PageLayout(count=_COUNT)).into_managed()
    var src = src_alloc.unsafe_ptr()
    var dst = dst_alloc.unsafe_ptr()

    for i in range(_COUNT):
        src.unsafe_offset(i).unsafe_store(Float32(i))
        dst.unsafe_offset(i).unsafe_store(Float32(0))

    var wrapped_src = ctx.wrap_host_memory[.float32](src, _COUNT)
    var wrapped_dst = ctx.wrap_host_memory[.float32](dst, _COUNT)
    assert_equal(len(wrapped_src), _COUNT)

    # Without this the copies below still pass by falling back to a memcpy of
    # the unwrapped host pointer, proving nothing about the wrap.
    # Only CUDA maps a registered range back at the host address.
    if ctx.api() == "cuda":
        assert_true(wrapped_src.unsafe_ptr() == src)
    else:
        assert_true(wrapped_src.unsafe_ptr() != src)

    var dev = ctx.enqueue_create_buffer[.float32](_COUNT)
    wrapped_src.enqueue_copy_to(dev)
    dev.enqueue_copy_to(wrapped_dst)
    ctx.synchronize()

    for i in range(_COUNT):
        assert_equal(dst.unsafe_offset(i).unsafe_load(), Float32(i))

    # The wrap is non-owning, so these must outlive the copies above: `src`'s
    # last use is the assert, which precedes the DMA.
    _ = src_alloc^
    _ = dst_alloc^
    print("Done")


def test_wrap_host_memory_leaves_pages_to_caller() raises:
    var ctx = create_test_device_context()

    var allocation = alloc(_PageLayout(count=_COUNT)).into_managed()
    var ptr = allocation.unsafe_ptr()
    for i in range(_COUNT):
        ptr.unsafe_offset(i).unsafe_store(Float32(i))

    # The wrap grants access only, so dropping it must leave the pages intact
    # and still the caller's to free.
    var wrapped = ctx.wrap_host_memory[.float32](ptr, _COUNT)
    _ = wrapped^
    ctx.synchronize()

    for i in range(_COUNT):
        assert_equal(ptr.unsafe_offset(i).unsafe_load(), Float32(i))

    _ = allocation^
    print("Done")


def _vec_add(
    in0: Pointer[Float32, MutAnyOrigin],
    in1: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    len_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var length = Int(len_dev)
    var tid = global_idx.x
    if tid >= length:
        return
    output[unsafe_offset=tid] = in0[unsafe_offset=tid] + in1[unsafe_offset=tid]


def test_wrap_host_memory_feeds_kernel_via_device_pointer() raises:
    """A wrapped range is a kernel operand, reached through a `DevicePointer`.

    The kernel writes through it and the caller reads the result from its own
    pages, with no copy back.
    """
    var ctx = create_test_device_context()

    comptime block_dim = 32
    var out_alloc = alloc(_PageLayout(count=_COUNT)).into_managed()
    var out_ptr = out_alloc.unsafe_ptr()
    for i in range(_COUNT):
        out_ptr.unsafe_offset(i).unsafe_store(Float32(0))

    var in0 = ctx.enqueue_create_buffer[.float32](_COUNT)
    var in1 = ctx.enqueue_create_buffer[.float32](_COUNT)
    with in0.map_to_host() as in0_host, in1.map_to_host() as in1_host:
        for i in range(_COUNT):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(2 * i + 1)

    var wrapped_out = ctx.wrap_host_memory[.float32](out_ptr, _COUNT)

    var kernel = ctx.compile_function[_vec_add]()
    ctx.enqueue_function(
        kernel,
        in0.device_ptr(),
        in1.device_ptr(),
        wrapped_out.device_ptr(),
        Int32(_COUNT),
        grid_dim=(_COUNT // block_dim),
        block_dim=block_dim,
    )
    ctx.synchronize()

    for i in range(_COUNT):
        assert_equal(
            out_ptr.unsafe_offset(i).unsafe_load(),
            Float32(i) + Float32(2 * i + 1),
        )

    # `DevicePointer` is a non-owning view, so the buffers it borrows must
    # outlive the enqueued kernel.
    _ = in0^
    _ = in1^
    _ = wrapped_out^
    _ = out_alloc^
    print("Done")


def main() raises:
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    var suite = TestSuite()

    suite.test[test_wrap_host_memory_round_trip]()
    suite.test[test_wrap_host_memory_leaves_pages_to_caller]()
    suite.test[test_wrap_host_memory_feeds_kernel_via_device_pointer]()

    suite^.run()
