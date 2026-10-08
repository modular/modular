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
"""Driver-level tests for the symmetric multicast pool.

The pool lives in a process-wide, create-once registry, so these tests share a
single module-scoped allocation rather than allocating per test. They run on
two devices: the four-rank floor is serving policy (``capacity_bytes``), not a
driver constraint.
"""

import gc

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, SymmetricPool
from max.dtype import DType
from max.nn.comm import MulticastPool

# Deliberately unaligned so the driver's rounding is exercised.
POOL_BYTES = (1 << 20) + 4096


@pytest.fixture(scope="module")
def devices() -> list[Accelerator]:
    return [Accelerator(id=0), Accelerator(id=1)]


@pytest.fixture(scope="module")
def pool(devices: list[Accelerator]) -> SymmetricPool:
    try:
        return SymmetricPool.allocate(devices, POOL_BYTES)
    except RuntimeError as exc:
        pytest.skip(f"multicast unsupported on this node: {exc}")


def _ptr(pool: SymmetricPool, device: Accelerator) -> int:
    return pool.unicast_view(device)._data_ptr()


def test_lookup_misses_unregistered_device_set(
    devices: list[Accelerator],
) -> None:
    """A device set nobody allocated for has no pool."""
    assert SymmetricPool.lookup([devices[1]]) is None


def test_sizes(pool: SymmetricPool) -> None:
    """byte_size is what was asked for, not the driver's rounded reservation."""
    assert pool.byte_size == POOL_BYTES


def test_allocate_twice_is_rejected(
    pool: SymmetricPool, devices: list[Accelerator]
) -> None:
    """Create-once: a second allocation for the same set must not silently
    replace the first, since the kernel resolves the pool by device set."""
    with pytest.raises(RuntimeError):
        SymmetricPool.allocate(devices, POOL_BYTES)


def test_lookup_returns_the_same_allocation_in_any_order(
    pool: SymmetricPool, devices: list[Accelerator]
) -> None:
    """Two handles must name one allocation, compared by device address --
    equal pointers is the only evidence that both went through one registry.
    The key is the device set, so listing order must not matter."""
    for order in (devices, list(reversed(devices))):
        found = SymmetricPool.lookup(order)
        assert found is not None
        for device in devices:
            assert _ptr(found, device) == _ptr(pool, device)


def test_dropping_a_handle_does_not_free(
    pool: SymmetricPool, devices: list[Accelerator]
) -> None:
    """The driver owns the memory, not the Python handle.

    The module fixture necessarily keeps one handle alive, so this shows that
    an extra handle is not what holds the allocation up -- not that the last
    handle is safe to drop, which the registry guarantees by construction.
    """
    extra = SymmetricPool.lookup(devices)
    assert extra is not None
    before = _ptr(extra, devices[0])
    del extra
    gc.collect()

    after = SymmetricPool.lookup(devices)
    assert after is not None
    assert _ptr(after, devices[0]) == before


def test_views_are_distinct_and_correctly_sized(
    pool: SymmetricPool, devices: list[Accelerator]
) -> None:
    """The views stop at byte_size, not the padded reservation, and the
    multicast alias is a separate mapping of the same bytes."""
    for device in devices:
        unicast = pool.unicast_view(device)
        multicast = pool.multicast_view(device)
        assert unicast.dtype == DType.uint8
        assert unicast.shape == (POOL_BYTES,)
        assert multicast.shape == (POOL_BYTES,)
        assert unicast._data_ptr() != multicast._data_ptr()
    assert _ptr(pool, devices[0]) != _ptr(pool, devices[1])


def test_unicast_view_round_trips_on_every_device(
    pool: SymmetricPool, devices: list[Accelerator]
) -> None:
    """The unicast view is ordinary device memory: writable and readable.

    Uses a sub-region, since a payload is always smaller than the pool and
    `Buffer.view` matches byte counts exactly -- retyping the full capacity
    would not exercise the shape a kernel actually sees. Each device gets its
    own data so a view aliased to the wrong device would be caught.
    """
    count = 256
    payload_bytes = count * DType.float32.size_in_bytes
    typed = [
        pool.unicast_view(device)[:payload_bytes].view(DType.float32, [count])
        for device in devices
    ]
    expected = [
        np.arange(count, dtype=np.float32) + 1000 * (i + 1)
        for i in range(len(devices))
    ]
    for view, data in zip(typed, expected, strict=True):
        view.inplace_copy_from(Buffer.from_numpy(data))
    for device in devices:
        device.synchronize()
    for view, data in zip(typed, expected, strict=True):
        np.testing.assert_array_equal(view.to_numpy(), data)


def test_multicast_pool_allocate_reuses_the_registered_pool(
    pool: SymmetricPool, devices: list[Accelerator]
) -> None:
    """Serving's wrapper finds the registered pool rather than tripping the
    create-once rule, whatever size it asks for."""
    reused = MulticastPool.allocate(devices, POOL_BYTES)
    assert reused is not None
    assert _ptr(reused, devices[0]) == _ptr(pool, devices[0])
    again = MulticastPool.allocate(devices, 2 * POOL_BYTES)
    assert again is not None
    assert _ptr(again, devices[0]) == _ptr(pool, devices[0])


def test_multicast_pool_allocate_zero_bytes_is_none(
    devices: list[Accelerator],
) -> None:
    """A zero charge means no pool, even where one could be made."""
    assert MulticastPool.allocate(devices, 0) is None
