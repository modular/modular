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
"""Tests the `Device` view returned by `DeviceContext.device()`."""

from max.gpu.host import DeviceAttribute, DeviceContext
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


def _multiprocessor_count(ctx: DeviceContext) raises -> Int:
    return ctx.device().multiprocessor_count()


def test_device_queries() raises:
    with DeviceContext() as ctx:
        var device = ctx.device()
        assert_equal(device.id, 0)
        assert_true(device.get_name().byte_length() > 0)
        assert_true(
            device.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT) > 0
        )
        _ = device.is_host_unified()
        assert_true(device.get_arch().byte_length() > 0)


def _try_query[
    query: def(DeviceContext) thin raises -> Int
](ctx: DeviceContext) -> Optional[Int]:
    try:
        return query(ctx)
    except:
        return None


def _raw[attr: DeviceAttribute](ctx: DeviceContext) raises -> Int:
    return ctx.device().get_attribute(attr)


def _max_threads_per_block(ctx: DeviceContext) raises -> Int:
    return ctx.device().max_threads_per_block()


def _max_shared_memory_per_block(ctx: DeviceContext) raises -> Int:
    return ctx.device().max_shared_memory_per_block()


def _max_grid_dim_x(ctx: DeviceContext) raises -> Int:
    return ctx.device().max_grid_dim_x()


def _max_grid_dim_y(ctx: DeviceContext) raises -> Int:
    return ctx.device().max_grid_dim_y()


def _max_grid_dim_z(ctx: DeviceContext) raises -> Int:
    return ctx.device().max_grid_dim_z()


def _assert_matches_raw(named: Optional[Int], raw: Optional[Int]) raises:
    # A plugin may not map every CUDA-numbered attribute, so the named query
    # has to raise exactly where `get_attribute` does.
    assert_equal(Bool(named), Bool(raw))
    if raw:
        assert_equal(named.value(), raw.value())
        assert_true(raw.value() > 0)


def test_named_queries_match_get_attribute() raises:
    with DeviceContext() as ctx:
        _assert_matches_raw(
            _try_query[_multiprocessor_count](ctx),
            _try_query[_raw[DeviceAttribute.MULTIPROCESSOR_COUNT]](ctx),
        )
        _assert_matches_raw(
            _try_query[_max_threads_per_block](ctx),
            _try_query[_raw[DeviceAttribute.MAX_THREADS_PER_BLOCK]](ctx),
        )
        _assert_matches_raw(
            _try_query[_max_shared_memory_per_block](ctx),
            _try_query[_raw[DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK]](ctx),
        )
        _assert_matches_raw(
            _try_query[_max_grid_dim_x](ctx),
            _try_query[_raw[DeviceAttribute.MAX_GRID_DIM_X]](ctx),
        )
        _assert_matches_raw(
            _try_query[_max_grid_dim_y](ctx),
            _try_query[_raw[DeviceAttribute.MAX_GRID_DIM_Y]](ctx),
        )
        _assert_matches_raw(
            _try_query[_max_grid_dim_z](ctx),
            _try_query[_raw[DeviceAttribute.MAX_GRID_DIM_Z]](ctx),
        )


def test_max_grid_dim_matches_per_dimension_queries() raises:
    with DeviceContext() as ctx:
        var device = ctx.device()
        var x = _try_query[_max_grid_dim_x](ctx)
        var y = _try_query[_max_grid_dim_y](ctx)
        var z = _try_query[_max_grid_dim_z](ctx)
        # A plugin that maps only some dimensions makes `max_grid_dim()`
        # raise; the legacy backend answers all three.
        if not (x and y and z):
            with assert_raises():
                _ = device.max_grid_dim()
            return
        var dims = device.max_grid_dim()
        assert_equal(dims.x(), x.value())
        assert_equal(dims.y(), y.value())
        assert_equal(dims.z(), z.value())


def test_named_queries_on_cpu() raises:
    # A CPU driver answers these however it likes, or not at all; the call
    # must only come back as an `Int` or an error.
    var cpu = DeviceContext.cpu()
    _ = _try_query[_multiprocessor_count](cpu)
    _ = _try_query[_max_threads_per_block](cpu)
    _ = _try_query[_max_shared_memory_per_block](cpu)
    _ = _try_query[_max_grid_dim_x](cpu)
    _ = _try_query[_max_grid_dim_y](cpu)
    _ = _try_query[_max_grid_dim_z](cpu)


def test_device_cannot_peer_itself() raises:
    with DeviceContext() as ctx:
        assert_false(ctx.device().can_access_peer(ctx.device()))


def test_device_cannot_peer_cpu() raises:
    var cpu = DeviceContext.cpu()
    with DeviceContext() as ctx:
        assert_false(ctx.device().can_access_peer(cpu.device()))


def test_device_through_borrowed_context() raises:
    with DeviceContext() as ctx:
        assert_true(_multiprocessor_count(ctx) > 0)


def test_device_from_temporary_context() raises:
    var maybe_ctx = Optional(DeviceContext())
    assert_true(maybe_ctx.value().device().multiprocessor_count() > 0)


def test_device_id_follows_context() raises:
    if DeviceContext.number_of_devices() < 2:
        return
    with DeviceContext(1) as ctx:
        assert_equal(ctx.device().id, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
