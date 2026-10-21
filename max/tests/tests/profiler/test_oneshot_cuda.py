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

"""Tests for the CUDA driver profiler bracketing in the one-shot profiler."""

from __future__ import annotations

import contextlib
import unittest.mock
from collections.abc import Callable
from typing import Any

import pytest
from max.profiler.oneshot import cuda_profiler_region


def _out(ref: Any) -> Any:
    """Return the ctypes object behind a ``ctypes.byref`` out-parameter."""
    return ref._obj


class _FakeDriverFn:
    """A ctypes-function stand-in that records calls and returns a status."""

    def __init__(
        self,
        name: str,
        calls: list[str],
        status: int,
        effect: Callable[..., None] | None,
    ) -> None:
        self.name = name
        self.calls = calls
        self.status = status
        self.effect = effect
        self.argtypes: list[object] = []
        self.restype: object = None

    def __call__(self, *args: object) -> int:
        self.calls.append(self.name)
        if self.status == 0 and self.effect is not None:
            self.effect(*args)
        return self.status


class _FakeDriver:
    """Records the order of driver calls; any entry point can be made to fail.

    Devices ``0..device_count-1`` exist, and those in ``active`` already have
    an active primary context.
    """

    def __init__(
        self,
        failing: frozenset[str] = frozenset(),
        device_count: int = 1,
        active: frozenset[int] = frozenset({0}),
    ) -> None:
        self.calls: list[str] = []
        self.retained: list[int] = []

        def get_count(count: object) -> None:
            _out(count).value = device_count

        def get_device(device: object, ordinal: int) -> None:
            _out(device).value = ordinal

        def get_state(device: Any, flags: object, is_active: object) -> None:
            _out(is_active).value = int(device.value in active)

        def retain(ctx: object, device: Any) -> None:
            self.retained.append(device.value)

        effects: dict[str, Callable[..., None]] = {
            "cuDeviceGetCount": get_count,
            "cuDeviceGet": get_device,
            "cuDevicePrimaryCtxGetState": get_state,
            "cuDevicePrimaryCtxRetain": retain,
        }
        for name in (
            "cuInit",
            "cuDeviceGetCount",
            "cuDeviceGet",
            "cuDevicePrimaryCtxGetState",
            "cuDevicePrimaryCtxRetain",
            "cuDevicePrimaryCtxRelease_v2",
            "cuCtxPushCurrent_v2",
            "cuCtxPopCurrent_v2",
            "cuProfilerStart",
            "cuProfilerStop",
        ):
            status = 1 if name in failing else 0
            fn = _FakeDriverFn(name, self.calls, status, effects.get(name))
            setattr(self, name, fn)


def _patch_driver(
    driver: _FakeDriver | None,
) -> contextlib.AbstractContextManager[object]:
    return unittest.mock.patch(
        "max.profiler.oneshot._cuda._load_cuda_driver", return_value=driver
    )


def test_region_brackets_with_primary_context_current() -> None:
    driver = _FakeDriver()
    with _patch_driver(driver):
        with cuda_profiler_region():
            driver.calls.append("body")
    assert driver.calls == [
        "cuInit",
        "cuDeviceGetCount",
        "cuDeviceGet",
        "cuDevicePrimaryCtxGetState",
        "cuDevicePrimaryCtxRetain",
        "cuCtxPushCurrent_v2",
        "cuProfilerStart",
        "cuCtxPopCurrent_v2",
        "body",
        "cuCtxPushCurrent_v2",
        "cuProfilerStop",
        "cuCtxPopCurrent_v2",
        "cuDevicePrimaryCtxRelease_v2",
    ]


def test_region_uses_the_active_primary_context() -> None:
    """MAX on gpu:1 has opened only device 1's context; device 0 is untouched."""
    driver = _FakeDriver(device_count=2, active=frozenset({1}))
    with _patch_driver(driver):
        with cuda_profiler_region():
            pass
    assert driver.retained == [1]


def test_region_falls_back_to_device_0_without_an_active_context() -> None:
    driver = _FakeDriver(device_count=2, active=frozenset())
    with _patch_driver(driver):
        with cuda_profiler_region():
            pass
    assert driver.retained == [0]


def test_region_stops_and_releases_when_body_raises() -> None:
    driver = _FakeDriver()
    with _patch_driver(driver), pytest.raises(RuntimeError, match="boom"):
        with cuda_profiler_region():
            raise RuntimeError("boom")
    assert driver.calls[-3:] == [
        "cuProfilerStop",
        "cuCtxPopCurrent_v2",
        "cuDevicePrimaryCtxRelease_v2",
    ]


def test_region_skips_profiler_calls_when_setup_fails() -> None:
    """A failed driver setup still runs the body, without bracketing it."""
    driver = _FakeDriver(failing=frozenset({"cuInit"}))
    with _patch_driver(driver):
        with cuda_profiler_region():
            driver.calls.append("body")
    assert driver.calls == ["cuInit", "body"]


def test_region_skips_call_when_context_push_fails() -> None:
    driver = _FakeDriver(failing=frozenset({"cuCtxPushCurrent_v2"}))
    with _patch_driver(driver):
        with cuda_profiler_region():
            pass
    assert "cuProfilerStart" not in driver.calls
    assert "cuProfilerStop" not in driver.calls
    assert "cuCtxPopCurrent_v2" not in driver.calls
    assert driver.calls[-1] == "cuDevicePrimaryCtxRelease_v2"


def test_region_raises_without_cuda_driver() -> None:
    with _patch_driver(None), pytest.raises(RuntimeError, match="libcuda"):
        with cuda_profiler_region():
            pass
