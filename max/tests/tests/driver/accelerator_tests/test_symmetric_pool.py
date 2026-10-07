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
single module-scoped allocation rather than allocating per test.
"""

import gc

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, SymmetricPool
from max.dtype import DType

POOL_BYTES = 1 << 20


@pytest.fixture(scope="module")
def devices() -> list[Accelerator]:
    return [Accelerator(id=0), Accelerator(id=1)]


@pytest.fixture(scope="module")
def pool(devices: list[Accelerator]) -> SymmetricPool:
    try:
        return SymmetricPool.allocate(devices, POOL_BYTES)
    except RuntimeError as exc:
        pytest.skip(f"multicast unsupported on this node: {exc}")


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


def test_lookup_returns_the_same_allocation(
    pool: SymmetricPool, devices: list[Accelerator]
) -> None:
    """Two handles must name one allocation, compared by device address --
    equal pointers is the only evidence that both went through one registry."""
    found = SymmetricPool.lookup(devices)
    assert found is not None
    assert found.unicast_view(devices[0])._data_ptr() == (
        pool.unicast_view(devices[0])._data_ptr()
    )


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
    before = extra.unicast_view(devices[0])._data_ptr()
    del extra
    gc.collect()

    after = SymmetricPool.lookup(devices)
    assert after is not None
    assert after.unicast_view(devices[0])._data_ptr() == before


def test_views_are_distinct_and_correctly_sized(
    pool: SymmetricPool, devices: list[Accelerator]
) -> None:
    """The multicast alias is a separate mapping of the same bytes."""
    for device in devices:
        unicast = pool.unicast_view(device)
        multicast = pool.multicast_view(device)
        assert unicast.dtype == DType.uint8
        assert unicast.shape == (POOL_BYTES,)
        assert multicast.shape == (POOL_BYTES,)
        assert unicast._data_ptr() != multicast._data_ptr()


def test_unicast_view_round_trips(
    pool: SymmetricPool, devices: list[Accelerator]
) -> None:
    """The unicast view is ordinary device memory: writable and readable.

    Uses a sub-region, since a payload is always smaller than the pool and
    `Buffer.view` matches byte counts exactly -- retyping the full capacity
    would not exercise the shape a kernel actually sees.
    """
    count = 256
    payload_bytes = count * DType.float32.size_in_bytes
    typed = pool.unicast_view(devices[0])[:payload_bytes].view(
        DType.float32, [count]
    )
    data = np.arange(count, dtype=np.float32)
    typed.inplace_copy_from(Buffer.from_numpy(data))
    devices[0].synchronize()
    np.testing.assert_array_equal(typed.to_numpy(), data)
