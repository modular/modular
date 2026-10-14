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
# Multi-device `DeviceContext` contract, shared by the AsyncRT and HAL backends.

from std.collections import Array, Optional
from std.ffi import c_int, external_call

from max.gpu.host import DeviceContext
from std.testing import assert_equal, assert_false, assert_raises, assert_true


def test_device_ids_are_distinct() raises:
    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)
    assert_equal(ctx0.id(), 0)
    assert_equal(ctx1.id(), 1)


def test_device_id_out_of_range() raises:
    # AsyncRT and HAL word the error differently; only the raise is contract.
    var count = DeviceContext.number_of_devices()
    with assert_raises():
        _ = DeviceContext(device_id=count)
    with assert_raises():
        _ = DeviceContext(device_id=-1)


def test_cross_device_copy_round_trip() raises:
    comptime N = 1024
    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)
    var src = ctx0.enqueue_create_buffer[.int32](N)
    with src.map_to_host() as host:
        for i in range(N):
            host[i] = Int32(i * 3)
    var dst = ctx1.enqueue_create_buffer[.int32](N)
    dst.enqueue_fill(0)
    # The copy runs on ctx1, so order it after ctx0's writes.
    ctx1.enqueue_wait_for(ctx0)
    ctx1.enqueue_copy(dst, src)
    ctx1.synchronize()
    with dst.map_to_host() as host:
        for i in range(N):
            assert_equal(host[i], Int32(i * 3))


def test_cross_device_copy_wrong_destination_raises() raises:
    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)
    var on0 = ctx0.enqueue_create_buffer[.int32](16)
    var on1 = ctx1.enqueue_create_buffer[.int32](16)
    ctx0.synchronize()
    ctx1.synchronize()
    with assert_raises():
        ctx0.enqueue_copy(on1, on0)
    ctx0.synchronize()


def test_can_access_self_is_false() raises:
    var ctx0 = DeviceContext(device_id=0)
    assert_false(ctx0.can_access(ctx0))


def test_can_access_is_symmetric() raises:
    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)
    assert_equal(ctx0.can_access(ctx1), ctx1.can_access(ctx0))


def test_all_peer_access_enabled_transitions() raises:
    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)
    if not ctx0.can_access(ctx1):
        print("pair has no peer access; skipping")
        return
    # Runs after `test_enable_all_concurrent`, so these calls pin idempotence.
    DeviceContext.enable_all_peer_access()
    assert_true(DeviceContext.all_peer_access_enabled())
    DeviceContext.enable_all_peer_access()
    assert_true(DeviceContext.all_peer_access_enabled())


def test_cpu_peer_is_not_accessible() raises:
    var gpu1 = DeviceContext(device_id=1)
    var cpu = DeviceContext(api="cpu")
    var accessible: Bool
    try:
        accessible = gpu1.can_access(cpu)
    except:
        # AsyncRT raises for a CPU peer; HAL answers False.
        accessible = False
    assert_false(accessible)
    with assert_raises():
        gpu1.enable_peer_access(cpu)


def _enable_all_worker(
    error_slot: Pointer[NoneType, MutUnsafeAnyOrigin]
) -> Pointer[NoneType, MutUnsafeAnyOrigin]:
    try:
        DeviceContext.enable_all_peer_access()
    except e:
        error_slot.unsafe_bitcast[Optional[Error]]()[] = e^
    return Pointer[NoneType, MutUnsafeAnyOrigin].unsafe_dangling()


def test_enable_all_concurrent() raises:
    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)
    if not ctx0.can_access(ctx1):
        print("pair has no peer access; skipping")
        return
    # Runs before any enable, so every worker races the full enable.
    assert_false(DeviceContext.all_peer_access_enabled())
    comptime num_callers = 4
    var errors = Array[Optional[Error], num_callers](fill=Optional[Error]())
    var threads = Array[UInt64, num_callers](fill=0)

    # `parallelize` is serial on the HAL backend, and capturing `__async`
    # workers fail to compile (MOCO-5065), so use OS threads.
    for i in range(num_callers):
        var rc = external_call["pthread_create", c_int](
            Pointer(to=threads[i]),
            Optional[Pointer[NoneType, MutUnsafeAnyOrigin]](),
            _enable_all_worker,
            Pointer(to=errors[i]).unsafe_bitcast[NoneType](),
        )
        assert_equal(rc, 0, "pthread_create failed")
    var retval = Pointer[NoneType, MutUnsafeAnyOrigin].unsafe_dangling()
    for i in range(num_callers):
        var rc = external_call["pthread_join", c_int](
            threads[i], Pointer(to=retval)
        )
        assert_equal(rc, 0, "pthread_join failed")

    for i in range(num_callers):
        if errors[i]:
            raise errors[i].take()
    assert_true(DeviceContext.all_peer_access_enabled())


def _write_peer(ptr: MutPointer[Int32, MutAnyOrigin], value: Int32):
    ptr[] = value


def test_kernel_writes_through_peer_pointer() raises:
    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)
    if not ctx0.can_access(ctx1):
        print("pair has no peer access; skipping")
        return
    DeviceContext.enable_all_peer_access()
    var target = ctx0.enqueue_create_buffer[.int32](1)
    target.enqueue_fill(0)
    ctx0.synchronize()
    # Only peer access makes device 0's address valid on device 1.
    ctx1.enqueue_function[_write_peer](
        target, Int32(42), grid_dim=1, block_dim=1
    )
    ctx1.synchronize()
    with target.map_to_host() as host:
        assert_equal(host[0], Int32(42))


def main() raises:
    assert_true(DeviceContext.number_of_devices() > 1, "test requires two GPUs")
    test_device_ids_are_distinct()
    test_device_id_out_of_range()
    test_cross_device_copy_round_trip()
    test_cross_device_copy_wrong_destination_raises()
    test_can_access_self_is_false()
    test_can_access_is_symmetric()
    test_cpu_peer_is_not_accessible()
    test_enable_all_concurrent()
    test_all_peer_access_enabled_transitions()
    test_kernel_writes_through_peer_pointer()
