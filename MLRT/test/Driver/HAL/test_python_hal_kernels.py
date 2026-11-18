# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Kernel compile and launch tests for max.sys._hal.

Compiles the ``vec_add`` Mojo kernel and launches it through the real HAL plugin
selected by the build. Kept apart from ``test_python_hal.py`` so a backend that
cannot build this kernel still runs the kernel-free suite.
"""

from __future__ import annotations

import ctypes

import mojo.importer  # noqa: F401  # registers Mojo .so importer hook
import pytest
from max_private.sys._hal import Buffer, Bundle, Context, Driver, Function
from vec_add import vec_add  # type: ignore[import-not-found]


# Share one driver and context across the module, as test_python_hal.py
# does: creating either can take a minute on a busy GPU host.
@pytest.fixture(scope="module")
def driver() -> Driver:
    return Driver()


@pytest.fixture(scope="module")
def ctx(driver: Driver) -> Context:
    return driver.get_device(0).get_context()


# ----------------------------------------------------------------------
# Mojo kernel -> Python compile path
# ----------------------------------------------------------------------


def test_compile_returns_bundle_with_name(ctx: Context) -> None:
    bundle = ctx.compile(vec_add)
    assert isinstance(bundle, Bundle)
    assert isinstance(bundle.function_name, str)
    assert len(bundle.function_name) > 0


def test_load_function_returns_function(ctx: Context) -> None:
    bundle = ctx.compile(vec_add)
    func = ctx.load_function(bundle, bundle.function_name)
    assert isinstance(func, Function)


# ----------------------------------------------------------------------
# Kernel launch via Queue / Stream
# ----------------------------------------------------------------------


def _expected_vec_add(i: int) -> float:
    return float((i % 2) * -i * 2 + i)


def _prepare_vec_add_inputs(
    ctx: Context,
    length: int,
) -> tuple[
    Buffer,
    Buffer,
    Buffer,
    ctypes.Array[ctypes.c_float],
    ctypes.Array[ctypes.c_float],
    int,
]:
    buf_size = length * ctypes.sizeof(ctypes.c_float)
    a_host = (ctypes.c_float * length)(*(float(i) for i in range(length)))
    b_host = (ctypes.c_float * length)(
        *(float((i % 2) * -i * 2) for i in range(length))
    )
    a_dev = ctx.alloc_sync(buf_size)
    b_dev = ctx.alloc_sync(buf_size)
    out_dev = ctx.alloc_sync(buf_size)
    return a_dev, b_dev, out_dev, a_host, b_host, buf_size


def _vec_add_kernel_args(
    ctx: Context,
    a_dev: Buffer,
    b_dev: Buffer,
    out_dev: Buffer,
    length: int,
) -> tuple[list[int], list[int], list[object]]:
    a_ptr = ctypes.c_uint64(ctx.memory_get_address(a_dev))
    b_ptr = ctypes.c_uint64(ctx.memory_get_address(b_dev))
    out_ptr = ctypes.c_uint64(ctx.memory_get_address(out_dev))
    n_val = ctypes.c_int64(length)
    addrs = [
        ctypes.addressof(a_ptr),
        ctypes.addressof(b_ptr),
        ctypes.addressof(out_ptr),
        ctypes.addressof(n_val),
    ]
    sizes = [
        ctypes.sizeof(a_ptr),
        ctypes.sizeof(b_ptr),
        ctypes.sizeof(out_ptr),
        ctypes.sizeof(n_val),
    ]
    return addrs, sizes, [a_ptr, b_ptr, out_ptr, n_val]


def test_queue_execute_vec_add(ctx: Context) -> None:
    bundle = ctx.compile(vec_add)
    func = ctx.load_function(bundle, bundle.function_name)
    queue = ctx.create_queue()

    length = 1024
    a_dev, b_dev, out_dev, a_host, b_host, _buf_size = _prepare_vec_add_inputs(
        ctx, length
    )
    out_host = (ctypes.c_float * length)()

    queue.copy_to_device(a_dev.view(), ctypes.addressof(a_host))
    queue.copy_to_device(b_dev.view(), ctypes.addressof(b_host))
    queue.synchronize()

    args, arg_sizes, _keepalive = _vec_add_kernel_args(
        ctx, a_dev, b_dev, out_dev, length
    )

    block_dim = 256
    grid_dim = length // block_dim
    queue.execute(
        func,
        grid=(grid_dim, 1, 1),
        block=(block_dim, 1, 1),
        args=args,
        arg_sizes=arg_sizes,
    )
    queue.synchronize()

    queue.copy_from_device(ctypes.addressof(out_host), out_dev.view())
    queue.synchronize()

    for i in range(length):
        assert out_host[i] == _expected_vec_add(i), (
            f"out_host[{i}] = {out_host[i]} != {_expected_vec_add(i)}"
        )


def test_stream_execute_vec_add(ctx: Context) -> None:
    bundle = ctx.compile(vec_add)
    func = ctx.load_function(bundle, bundle.function_name)
    stream = ctx.create_stream()

    length = 1024
    a_dev, b_dev, out_dev, a_host, b_host, _buf_size = _prepare_vec_add_inputs(
        ctx, length
    )
    out_host = (ctypes.c_float * length)()

    stream.copy_to_device(a_dev.view(), ctypes.addressof(a_host))
    stream.copy_to_device(b_dev.view(), ctypes.addressof(b_host))

    args, arg_sizes, _keepalive = _vec_add_kernel_args(
        ctx, a_dev, b_dev, out_dev, length
    )

    block_dim = 256
    grid_dim = length // block_dim
    stream.execute(
        func,
        grid=(grid_dim, 1, 1),
        block=(block_dim, 1, 1),
        args=args,
        arg_sizes=arg_sizes,
    )

    stream.copy_from_device(ctypes.addressof(out_host), out_dev.view())
    stream.synchronize()

    for i in range(length):
        assert out_host[i] == _expected_vec_add(i), (
            f"out_host[{i}] = {out_host[i]} != {_expected_vec_add(i)}"
        )


def test_queue_execute_validates_shapes(ctx: Context) -> None:
    bundle = ctx.compile(vec_add)
    func = ctx.load_function(bundle, bundle.function_name)
    queue = ctx.create_queue()

    with pytest.raises(ValueError, match="grid"):
        queue.execute(func, grid=(1, 1), block=(1, 1, 1), args=[], arg_sizes=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="block"):
        queue.execute(func, grid=(1, 1, 1), block=(1, 1), args=[], arg_sizes=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same length"):
        queue.execute(
            func, grid=(1, 1, 1), block=(1, 1, 1), args=[0], arg_sizes=[]
        )
