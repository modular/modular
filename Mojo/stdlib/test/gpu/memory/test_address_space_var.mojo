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
"""Tests `var` declarations in shared memory with `@__address_space`."""

from std.testing import assert_equal

from max.gpu import barrier, thread_idx
from max.gpu.host import DeviceContext

comptime _THREADS = 32


# Every thread runs the initializer, so a shared variable starts uninitialized
# and only thread 0 writes it.
def _broadcast_kernel(output: MutPointer[Int32, MutUntrackedOrigin]):
    @__address_space(.SHARED)
    var value = Array[Int32, 1](uninitialized=True)
    if thread_idx.x == 0:
        value[0] = 42
    barrier()
    output.unsafe_offset(Int(thread_idx.x)).unsafe_store(value[0])


def _broadcast_parametric_kernel[
    address_space: AddressSpace
](output: MutPointer[Int32, MutUntrackedOrigin]):
    @__address_space(address_space)
    var value = Array[Int32, 1](uninitialized=True)
    if thread_idx.x == 0:
        value[0] = 7
    barrier()
    output.unsafe_offset(Int(thread_idx.x)).unsafe_store(value[0])


# Each thread has its own copy of a variable in the default address space, so
# each thread can initialize it.
def _private_parametric_kernel[
    address_space: AddressSpace
](output: MutPointer[Int32, MutUntrackedOrigin]):
    @__address_space(address_space)
    var value = Int32(thread_idx.x)
    output.unsafe_offset(Int(thread_idx.x)).unsafe_store(value)


def test_shared_var_is_visible_to_all_threads() raises:
    with DeviceContext() as ctx:
        var out_device = ctx.enqueue_create_buffer[.int32](_THREADS)
        ctx.enqueue_function[_broadcast_kernel](
            out_device, grid_dim=1, block_dim=_THREADS
        )
        ctx.synchronize()
        with out_device.map_to_host() as out_host:
            for i in range(_THREADS):
                assert_equal(out_host[i], 42, msg=String("thread ", i))


def test_parametric_shared_var() raises:
    with DeviceContext() as ctx:
        var out_device = ctx.enqueue_create_buffer[.int32](_THREADS)
        ctx.enqueue_function[_broadcast_parametric_kernel[.SHARED]](
            out_device, grid_dim=1, block_dim=_THREADS
        )
        ctx.synchronize()
        with out_device.map_to_host() as out_host:
            for i in range(_THREADS):
                assert_equal(out_host[i], 7, msg=String("thread ", i))


def test_parametric_generic_var() raises:
    with DeviceContext() as ctx:
        var out_device = ctx.enqueue_create_buffer[.int32](_THREADS)
        ctx.enqueue_function[_private_parametric_kernel[.GENERIC]](
            out_device, grid_dim=1, block_dim=_THREADS
        )
        ctx.synchronize()
        with out_device.map_to_host() as out_host:
            for i in range(_THREADS):
                assert_equal(out_host[i], Int32(i), msg=String("thread ", i))


# The kernels only build for a GPU, so the tests are listed here rather than
# discovered from every function in the module.
def main() raises:
    test_shared_var_is_visible_to_all_threads()
    test_parametric_shared_var()
    test_parametric_generic_var()
