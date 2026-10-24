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
"""Lifecycle smoke tests for max.sys._hal.

Exercises the Python -> Mojo HAL FFI path against the real HAL plugin
selected by the build.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import mojo.importer  # noqa: F401  # registers Mojo .so importer hook
import numpy as np
import pytest
from max_private.sys._hal import (
    Array,
    Buffer,
    Bundle,
    Context,
    Driver,
    DType,
    Event,
    Function,
    copy,
    fill,
    set_memory,
)
from vec_add import vec_add  # type: ignore[import-not-found]

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_host_buf(
    data: bytes | bytearray,
) -> tuple[ctypes.Array[ctypes.c_uint8], int]:
    n = len(data)
    buf = (ctypes.c_uint8 * n)(*data)
    return buf, ctypes.addressof(buf)


def _make_empty_host_buf(
    n: int,
) -> tuple[ctypes.Array[ctypes.c_uint8], int]:
    buf = (ctypes.c_uint8 * n)()
    return buf, ctypes.addressof(buf)


def _bytes_of(buf: ctypes.Array[ctypes.c_uint8]) -> bytes:
    return bytes(buf)


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


# Share one driver and context across the module. Each new driver loads and
# starts the GPU driver library, and each new context makes the GPU primary
# context. On a busy GPU host either step can take a minute, so doing them for
# every test runs past the test timeout.
@pytest.fixture(scope="module")
def driver() -> Driver:
    return Driver()


@pytest.fixture(scope="module")
def ctx(driver: Driver) -> Context:
    return driver.get_device(0).get_context()


def test_driver_load(driver: Driver) -> None:
    assert isinstance(driver.name, str) and driver.name != ""
    assert driver.device_count >= 1


def test_queue_record_event_and_synchronize(ctx: Context) -> None:
    queue = ctx.create_queue()
    event = queue.record_event()
    assert isinstance(event, Event)
    event.synchronize()
    assert event.is_ready() is True


# ----------------------------------------------------------------------
# Buffer / memory allocation
# ----------------------------------------------------------------------


def test_context_alloc_sync_returns_buffer(ctx: Context) -> None:
    buf = ctx.alloc_sync(64)
    assert isinstance(buf, Buffer)
    assert buf.byte_size == 64
    assert ctx.memory_get_address(buf) != 0


def test_context_alloc_host_pinned_returns_buffer(ctx: Context) -> None:
    buf = ctx.alloc_host_pinned(64)
    assert isinstance(buf, Buffer)
    assert buf.byte_size == 64


def test_context_wrap_unwrap(ctx: Context) -> None:
    # A zero address is rejected by every plugin rather than minting a
    # handle that would segfault on first use.
    with pytest.raises(Exception):
        ctx.wrap_memory(0, 64)

    # A pinned buffer cannot round-trip — a wrapped handle always frees
    # through the device path — so unwrap rejects it up front.
    with pytest.raises(Exception, match="pinned"):
        ctx.unwrap_memory(ctx.alloc_host_pinned(32))

    # A non-owning wrap over an existing allocation reports the same
    # address; dropping the wrapper releases only the HAL bookkeeping, so
    # the original allocation stays valid.
    orig = ctx.alloc_sync(64)
    addr = ctx.memory_get_address(orig)
    wrapped = ctx.wrap_memory(addr, 64)
    assert ctx.memory_get_address(wrapped) == addr
    del wrapped
    assert ctx.memory_get_address(orig) == addr

    victim = ctx.alloc_sync(64)
    victim_addr = ctx.memory_get_address(victim)
    assert ctx.unwrap_memory(victim) == victim_addr
    adopted = ctx.wrap_memory(victim_addr, 64, owning=True)
    del victim
    del adopted


# ----------------------------------------------------------------------
# Memory copies via Queue
# ----------------------------------------------------------------------


def test_queue_host_to_device_to_host_roundtrip(ctx: Context) -> None:
    queue = ctx.create_queue()

    n = 256
    src_data = bytes(range(n))
    _src, src_addr = _make_host_buf(src_data)
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    queue.copy_to_device(dev.view(), src_addr)
    queue.copy_from_device(dst_addr, dev.view())
    queue.synchronize()

    assert _bytes_of(dst) == src_data


def test_queuecopy_intra_device(ctx: Context) -> None:
    queue = ctx.create_queue()

    n = 256
    src_data = bytes((i * 3) % 256 for i in range(n))
    _src, src_addr = _make_host_buf(src_data)
    dst, dst_addr = _make_empty_host_buf(n)
    dev_a = ctx.alloc_sync(n)
    dev_b = ctx.alloc_sync(n)

    queue.copy_to_device(dev_a.view(), src_addr)
    queue.copy_intra_device(dev_b.view(), dev_a.view())
    queue.copy_from_device(dst_addr, dev_b.view())
    queue.synchronize()

    assert _bytes_of(dst) == src_data


def test_unified_copy_device_to_device(ctx: Context) -> None:
    queue = ctx.create_queue()

    n = 128
    src_data = bytes((i * 5) % 256 for i in range(n))
    _src, src_addr = _make_host_buf(src_data)
    dst, dst_addr = _make_empty_host_buf(n)
    dev_a = ctx.alloc_sync(n)
    dev_b = ctx.alloc_sync(n)
    dev_c = ctx.alloc_sync(n)

    queue.copy_to_device(dev_a.view(), src_addr)
    queue.synchronize()

    # Unified copy on the queue (device -> device).
    queue.copy(dst=dev_b, src=dev_a)
    queue.synchronize()

    # Module-level synchronous copy (device -> device via its own queue).
    copy(dst=dev_c, src=dev_b)

    queue.copy_from_device(dst_addr, dev_c.view())
    queue.synchronize()
    assert _bytes_of(dst) == src_data


def test_context_sync_copy_roundtrip(ctx: Context) -> None:
    n = 256
    src_data = bytes((i * 11) % 256 for i in range(n))
    _src, src_addr = _make_host_buf(src_data)
    dst, dst_addr = _make_empty_host_buf(n)
    dev_a = ctx.alloc_sync(n)
    dev_b = ctx.alloc_sync(n)

    ctx.copy_to_device(dev_a.view(), src_addr)
    ctx.copy_intra_device(dev_b.view(), dev_a.view())
    ctx.copy_from_device(dst_addr, dev_b.view())

    assert _bytes_of(dst) == src_data


# ----------------------------------------------------------------------
# Memory set / fill via Queue
# ----------------------------------------------------------------------


def test_queue_set_memory(ctx: Context) -> None:
    queue = ctx.create_queue()

    n = 128
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    queue.set_memory(dev.view(), 0x5A)
    queue.copy_from_device(dst_addr, dev.view())
    queue.synchronize()

    assert _bytes_of(dst) == bytes([0x5A]) * n


def test_queue_fill(ctx: Context) -> None:
    queue = ctx.create_queue()

    n = 128
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    # 32-bit pattern, little-endian repeating across the buffer.
    queue.fill(dev.view(), 0xDEADBEEF, 4)
    queue.copy_from_device(dst_addr, dev.view())
    queue.synchronize()

    assert _bytes_of(dst) == (0xDEADBEEF).to_bytes(4, "little") * (n // 4)


# ----------------------------------------------------------------------
# Module-level synchronous set_memory / fill (blocking, no queue)
# ----------------------------------------------------------------------


def test_set_memory_sync(ctx: Context) -> None:
    queue = ctx.create_queue()

    n = 128
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    # Blocking, queue-less: fills all of `dev` and returns once complete.
    set_memory(dst=dev, value=0x5A)

    queue.copy_from_device(dst_addr, dev.view())
    queue.synchronize()
    assert _bytes_of(dst) == bytes([0x5A]) * n


def test_fill_sync(ctx: Context) -> None:
    queue = ctx.create_queue()

    n = 128
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    # Blocking, queue-less: fills all of `dev` with a 32-bit pattern.
    fill(dst=dev, value=0xDEADBEEF, value_size=4)

    queue.copy_from_device(dst_addr, dev.view())
    queue.synchronize()
    assert _bytes_of(dst) == (0xDEADBEEF).to_bytes(4, "little") * (n // 4)


# ----------------------------------------------------------------------
# Memory copies via Stream
# ----------------------------------------------------------------------


def test_stream_host_to_device_to_host_roundtrip(ctx: Context) -> None:
    stream = ctx.create_stream()

    n = 256
    src_data = bytes(range(n))
    _src, src_addr = _make_host_buf(src_data)
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    stream.copy_to_device(dev.view(), src_addr)
    stream.copy_from_device(dst_addr, dev.view())
    stream.synchronize()

    assert _bytes_of(dst) == src_data


def test_streamcopy_intra_device(ctx: Context) -> None:
    stream = ctx.create_stream()

    n = 256
    src_data = bytes((i * 7) % 256 for i in range(n))
    _src, src_addr = _make_host_buf(src_data)
    dst, dst_addr = _make_empty_host_buf(n)
    dev_a = ctx.alloc_sync(n)
    dev_b = ctx.alloc_sync(n)

    stream.copy_to_device(dev_a.view(), src_addr)
    stream.copy_intra_device(dev_b.view(), dev_a.view())
    stream.copy_from_device(dst_addr, dev_b.view())
    stream.synchronize()

    assert _bytes_of(dst) == src_data


def test_stream_set_memory(ctx: Context) -> None:
    stream = ctx.create_stream()

    n = 128
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    stream.set_memory(dev.view(), 0x5A)
    stream.copy_from_device(dst_addr, dev.view())
    stream.synchronize()

    assert _bytes_of(dst) == bytes([0x5A]) * n


def test_stream_fill(ctx: Context) -> None:
    stream = ctx.create_stream()

    n = 128
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    stream.fill(dev.view(), 0xDEADBEEF, 4)
    stream.copy_from_device(dst_addr, dev.view())
    stream.synchronize()

    assert _bytes_of(dst) == (0xDEADBEEF).to_bytes(4, "little") * (n // 4)


# ----------------------------------------------------------------------
# BufferView — accessors and sub-range copies
# ----------------------------------------------------------------------


def test_queue_partial_copy_roundtrip(ctx: Context) -> None:
    queue = ctx.create_queue()

    n = 256
    src_data = bytes(range(n))
    _src, src_addr = _make_host_buf(src_data)
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    # Upload the whole buffer, then read back only the middle [64, 192).
    queue.copy_to_device(dev.view(), src_addr)
    mid = dev.view(byte_offset=64, byte_size=128)
    queue.copy_from_device(dst_addr + 64, mid)
    queue.synchronize()

    assert _bytes_of(dst)[64:192] == src_data[64:192]
    assert _bytes_of(dst)[:64] == bytes(64)
    assert _bytes_of(dst)[192:] == bytes(64)


def test_stream_partial_copy_roundtrip(ctx: Context) -> None:
    stream = ctx.create_stream()

    n = 256
    src_data = bytes(range(n))
    _src, src_addr = _make_host_buf(src_data)
    dst, dst_addr = _make_empty_host_buf(n)
    dev = ctx.alloc_sync(n)

    stream.copy_to_device(dev.view(), src_addr)
    mid = dev.view(byte_offset=64, byte_size=128)
    stream.copy_from_device(dst_addr + 64, mid)
    stream.synchronize()

    assert _bytes_of(dst)[64:192] == src_data[64:192]
    assert _bytes_of(dst)[:64] == bytes(64)
    assert _bytes_of(dst)[192:] == bytes(64)


# ----------------------------------------------------------------------
# Cross-queue / cross-stream wait_for_events
# ----------------------------------------------------------------------


def test_queue_wait_for_events_single(ctx: Context) -> None:
    producer = ctx.create_queue()
    consumer = ctx.create_queue()
    event = producer.record_event()
    consumer.wait_for_events(event)
    consumer.synchronize()


def test_stream_wait_for_events_single(ctx: Context) -> None:
    producer = ctx.create_stream()
    consumer = ctx.create_stream()
    event = producer.record_event()
    consumer.wait_for_events(event)
    consumer.synchronize()


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


# ----------------------------------------------------------------------
# Array — typed view over a HAL Buffer
# ----------------------------------------------------------------------


def test_array_construction_and_fill(ctx: Context, tmp_path: Path) -> None:
    # Host -> device -> host roundtrip preserves data and metadata.
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    arr = Array.from_numpy(ctx, src)
    assert arr.shape == (3, 4)
    assert arr.dtype == DType.float32
    assert arr.strides == (4, 1)
    assert arr.is_contiguous
    np.testing.assert_array_equal(arr.as_numpy(), src)

    # from_list infers shape/dtype from Python values; explicit dtype wins.
    typed = Array.from_list(ctx, [[1, 2], [3, 4]], dtype=DType.int32)
    assert typed.shape == (2, 2)
    np.testing.assert_array_equal(
        typed.as_numpy(), np.array([[1, 2], [3, 4]], np.int32)
    )

    # from_file copies a file's packed bytes to the device, honoring offset;
    # a file too short for offset + dtype*shape is rejected.
    data = np.arange(6, dtype=np.float32)
    path = tmp_path / "weights.bin"
    path.write_bytes(b"\xaa" * 3 + data.tobytes())
    loaded = Array.from_file(ctx, path, DType.float32, (2, 3), offset=3)
    np.testing.assert_array_equal(loaded.as_numpy(), data.reshape(2, 3))
    with pytest.raises(ValueError, match="bytes"):
        Array.from_file(ctx, path, DType.float32, (100,), offset=3)

    # full/fill: byte-uniform zero, a multi-byte pattern, and -0.0 must
    # keep its IEEE sign bit.
    np.testing.assert_array_equal(
        Array.full(ctx, DType.int64, (4,), 0).as_numpy(), np.zeros(4, np.int64)
    )
    np.testing.assert_array_equal(
        Array.full(ctx, DType.int32, (4,), 0x01020304).as_numpy(),
        np.full(4, 0x01020304, np.int32),
    )
    assert np.signbit(
        Array.full(ctx, DType.float32, (2,), -0.0).as_numpy()
    ).all()


def test_array_copy_false_adoption(ctx: Context, tmp_path: Path) -> None:
    # copy=False adopts host memory zero-copy, which only works when the source
    # is already on the context's device. On a host (CPU) context a numpy
    # array is on-device and is adopted; on a device context it is not, so
    # adoption is refused rather than silently copied. Whichever context this
    # suite runs against, one branch below exercises the routing.
    src = np.arange(6, dtype=np.float32)
    path = tmp_path / "raw.bin"
    path.write_bytes(src.tobytes())
    on_device = ctx.get_dlpack_device(False) == tuple(src.__dlpack_device__())
    if on_device:
        adopted = Array.from_numpy(ctx, src, copy=False)
        np.testing.assert_array_equal(adopted.as_numpy(), src)
        np.testing.assert_array_equal(adopted.as_numpy(copy=False), src)
        mapped = Array.from_file(ctx, path, DType.float32, (6,), copy=False)
        np.testing.assert_array_equal(mapped.as_numpy(), src)
    else:
        with pytest.raises(ValueError, match="host"):
            Array.from_numpy(ctx, src, copy=False)
        with pytest.raises(ValueError, match="host"):
            Array.from_file(ctx, path, DType.float32, (6,), copy=False)


def test_array_slicing_and_views(ctx: Context) -> None:
    src = np.arange(24, dtype=np.int32).reshape(4, 6)
    arr = Array.from_numpy(ctx, src)

    # Slicing follows numpy view semantics: strided, shares storage.
    sub = arr[:, 1:3]  # as_numpy materializes via contiguous()
    assert sub.shape == (4, 2)
    assert not sub.is_contiguous
    np.testing.assert_array_equal(sub.as_numpy(), src[:, 1:3])
    assert arr[1, 2].shape == ()  # full integer index -> rank-0

    arr[0].fill(9)  # writing through a slice hits the base
    assert (arr.as_numpy()[0] == 9).all()
    arr[1, 2] = 5  # element assignment also goes through to the base
    assert arr.as_numpy()[1, 2] == 5

    # view reinterprets the bytes as another dtype, sharing the buffer.
    floats = np.array([1.0, 2.0, -3.5], dtype=np.float32)
    np.testing.assert_array_equal(
        Array.from_numpy(ctx, floats).view(DType.int32).as_numpy(),
        floats.view(np.int32),
    )

    # A pure reshape needs no dtype, and a -1 entry is inferred.
    fresh = Array.from_numpy(ctx, src)
    reshaped = fresh.view(shape=(2, 12))
    assert reshaped.dtype == fresh.dtype
    np.testing.assert_array_equal(reshaped.as_numpy(), src.reshape(2, 12))
    np.testing.assert_array_equal(
        fresh.view(shape=(-1, 8)).as_numpy(), src.reshape(-1, 8)
    )
    np.testing.assert_array_equal(
        fresh.reshape((-1, 12)).as_numpy(), src.reshape(-1, 12)
    )
    with pytest.raises(ValueError, match="-1"):
        fresh.view(shape=(-1, -1))

    with pytest.raises(IndexError):
        arr[10]
    with pytest.raises(TypeError):
        arr[None]


def test_array_fill_slice_stays_in_range(ctx: Context) -> None:
    # A sliced fill must touch only the slice's bytes, not the whole backing
    # buffer: exercises the view offset+size through the synchronous Context
    # fill path (a middle row has a non-zero byte offset).
    arr = Array.full(ctx, DType.int32, (4, 6), 0)
    arr[1].fill(7)
    out = arr.as_numpy()
    assert (out[1] == 7).all()
    assert (out[0] == 0).all()
    assert (out[2] == 0).all()
    assert (out[3] == 0).all()


def test_array_copy_and_subbyte_packing(ctx: Context) -> None:
    # Device-to-device copy into existing storage.
    src = Array.from_numpy(ctx, np.arange(12, dtype=np.int32).reshape(3, 4))
    dst = Array.empty(ctx, DType.int32, (3, 4))
    Array.copy(src, dst)
    np.testing.assert_array_equal(dst.as_numpy(), src.as_numpy())

    # Sub-byte dtypes pack densely like the legacy buffer: N 4-bit values
    # take ceil(N/2) bytes, and fill(0) zeroes exactly the packed bytes.
    n = 8
    arr = Array.empty(ctx, DType.float4_e2m1fn, (n,))
    assert arr.element_size == 1  # per-element size rounds up
    assert arr.byte_size == n // 2
    arr.fill(0)
    np.testing.assert_array_equal(
        arr.view(DType.uint8, (arr.byte_size,)).as_numpy(),
        np.zeros(arr.byte_size, np.uint8),
    )
    # A byte-aligned sub-byte slice is fine; a bit-unaligned one cannot be
    # a byte-addressed view and must raise.
    assert arr[2:].byte_size == (n - 2) // 2
    with pytest.raises(ValueError, match="byte"):
        arr[1:]


# ----------------------------------------------------------------------
# Array — DLPack protocol
# ----------------------------------------------------------------------


def test_array_dlpack_export(ctx: Context) -> None:
    arr = Array.from_numpy(ctx, np.arange(6, dtype=np.float32).reshape(2, 3))

    # The capsule pins the owning array; releasing it restores the balance.
    base = sys.getrefcount(arr)
    capsule = arr.__dlpack__()
    assert sys.getrefcount(arr) == base + 1
    del capsule
    assert sys.getrefcount(arr) == base

    arr.__dlpack__(max_version=(1, 0))  # versioned capsule, same lifetime
    assert sys.getrefcount(arr) == base

    with pytest.raises(BufferError, match="negative strides"):
        Array.from_numpy(ctx, np.arange(4, dtype=np.int32))[::-1].__dlpack__()

    # numpy cannot read device memory: its error must propagate intact and
    # the aborted export must still be released.
    with pytest.raises(RuntimeError):
        np.from_dlpack(arr)
    assert sys.getrefcount(arr) == base

    # Pinned memory is host-accessible, so numpy consumes the
    # exported capsule zero-copy — a round-trip through a real consumer.
    if ctx.driver_name in ("CUDA", "HIP"):
        pinned = Array.empty(ctx, DType.float32, (4,), pinned=True)
        host_view = np.from_dlpack(pinned)
        host_view[:] = 7.0
        np.testing.assert_array_equal(
            pinned.as_numpy(), np.full(4, 7.0, np.float32)
        )


class _DelegatingProducer:
    """A non-Array producer, so imports take the real capsule path."""

    def __init__(self, arr: Array) -> None:
        self._arr = arr

    def __dlpack_device__(self) -> tuple[int, int]:
        return self._arr.__dlpack_device__()

    def __dlpack__(self, *, stream: int | None = None) -> object:
        return self._arr.__dlpack__(stream=stream)


class _ForeignDeviceProducer:
    """A producer on a device (kDLVulkan) no HAL plugin can import from."""

    def __dlpack_device__(self) -> tuple[int, int]:
        return (7, 0)

    def __dlpack__(self, *, stream: int | None = None) -> object:
        raise AssertionError("capsule must not be requested")


def test_array_dlpack_import(ctx: Context) -> None:
    src = np.arange(6, dtype=np.float32).reshape(2, 3)
    with pytest.raises(ValueError, match="host"):
        Array.from_dlpack(ctx, src)

    # Build an on-device array to exercise the adopt paths below.
    arr = Array.from_numpy(ctx, src)

    # An Array input short-circuits; a foreign device is rejected up front.
    assert Array.from_dlpack(ctx, arr) is arr
    with pytest.raises(ValueError, match="device"):
        Array.from_dlpack(ctx, _ForeignDeviceProducer())

    producer = _DelegatingProducer(arr)

    # A same-device producer is adopted zero-copy; dropping the adopted
    # array runs the producer's deleter and releases its reference.
    base = sys.getrefcount(arr)
    adopted = Array.from_dlpack(ctx, producer)
    assert adopted.data_ptr == arr.data_ptr
    arr.fill(3)
    np.testing.assert_array_equal(
        adopted.as_numpy(), np.full((2, 3), 3, np.float32)
    )
    del adopted
    assert sys.getrefcount(arr) == base

    # The pinned-host variant adopts zero-copy too. The adopted array carries
    # the pinned flag only when the plugin distinguishes a pinned-host DLPack
    # device from its device one (CUDA's kDLCUDAHost vs kDLCUDA); Metal reports
    # the same device for both, so the flag does not survive the round-trip.
    keeps_pinned = ctx.get_dlpack_device(True) != ctx.get_dlpack_device(False)
    pinned = Array.empty(ctx, DType.float32, (4,), pinned=True)
    pinned.fill(5)
    adopted = Array.from_dlpack(ctx, _DelegatingProducer(pinned))
    assert adopted.data_ptr == pinned.data_ptr
    assert adopted.pinned == keeps_pinned
    np.testing.assert_array_equal(adopted.as_numpy(), np.full(4, 5, np.float32))


# ----------------------------------------------------------------------
# Array — async stream operations
# ----------------------------------------------------------------------


def test_array_async_ops(ctx: Context) -> None:
    stream = ctx.create_stream()

    # array_full / array_fill / array_copy enqueue WITHOUT waiting; the caller
    # orders them (here: submission order on one stream, then synchronize).
    a = stream.array_full(DType.float32, (4,), 2.0)
    b = Array.empty(ctx, DType.float32, (4,))
    stream.array_fill(b, 5.0)
    stream.array_copy(a, b)  # src=a, dst=b  ->  b := a (2.0)
    stream.synchronize()
    np.testing.assert_array_equal(b.as_numpy(), np.full(4, 2.0, np.float32))

    # A strided source is rejected — the fire-and-forget path cannot keep the
    # internal contiguous temp alive.
    strided = Array.full(ctx, DType.float32, (4,), 1.0)[::2]
    with pytest.raises(ValueError, match="contiguous"):
        stream.array_copy(strided, Array.empty(ctx, DType.float32, (2,)))

    host = np.arange(6, dtype=np.float32).reshape(2, 3)
    dev = stream.array_from_numpy(host)
    stream.synchronize()
    np.testing.assert_array_equal(dev.as_numpy(), host)
    with pytest.raises(ValueError, match="contiguous"):
        stream.array_from_numpy(host[:, ::2])

    # The same async ops are available on a Queue (identical primitive surface
    # to Stream). Keep the numpy source alive until synchronize().
    queue = ctx.create_queue()
    qa = queue.array_full(DType.float32, (3,), 7.0)
    qsrc = np.arange(3, dtype=np.float32)
    qn = queue.array_from_numpy(qsrc)
    queue.synchronize()
    np.testing.assert_array_equal(qa.as_numpy(), np.full(3, 7.0, np.float32))
    np.testing.assert_array_equal(qn.as_numpy(), qsrc)


def test_array_async_dlpack(ctx: Context) -> None:
    stream = ctx.create_stream()
    src = Array.from_numpy(ctx, np.arange(6, dtype=np.float32).reshape(2, 3))

    # array_from_dlpack host-blocks, so the adopted (zero-copy, same address)
    # array is safe to read immediately.
    adopted = stream.array_from_dlpack(_DelegatingProducer(src))
    assert adopted.data_ptr == src.data_ptr
    np.testing.assert_array_equal(
        adopted.as_numpy(), np.arange(6, dtype=np.float32).reshape(2, 3)
    )

    # Export returns a capsule (unversioned, versioned, and with the -1 no-sync
    # sentinel). The caller must have ordered any async writes beforehand.
    exported = Array.from_numpy(ctx, np.arange(4, dtype=np.float32))
    assert exported.__dlpack__() is not None
    assert exported.__dlpack__(max_version=(1, 0)) is not None
    assert exported.__dlpack__(stream=-1) is not None
