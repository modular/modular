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

"""In-process ``cuProfilerStart``/``Stop`` bracketing via ``ctypes``.

Used to bound the nsys capture window to the timed region only, so warmup
and graph-compile time are excluded from the report. Goes through the CUDA
driver API (``libcuda``) because MAX must not depend on the CUDA runtime.
"""

from __future__ import annotations

import ctypes
import logging
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger("max.profiler.oneshot")

_CUDA_DRIVER_LIB_CANDIDATES: tuple[str, ...] = (
    "libcuda.so.1",
    "libcuda.so",
)

_CUDA_SUCCESS = 0


def _load_cuda_driver() -> ctypes.CDLL | None:
    """Locate and dlopen the CUDA driver library, or return ``None``."""
    for name in _CUDA_DRIVER_LIB_CANDIDATES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def can_load_cuda_driver() -> bool:
    """Return True if libcuda can be dlopened on this host.

    Used by the backend gate to refuse the nsys re-exec when ``cuProfilerStart``
    cannot be called, since the nsys child is launched with
    ``--capture-range=cudaProfilerApi`` and would otherwise produce an empty
    ``.nsys-rep``.
    """
    return _load_cuda_driver() is not None


def _declare_signatures(driver: ctypes.CDLL) -> None:
    # The _v2 names are the symbols cuda.h maps the unsuffixed API names to.
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDevicePrimaryCtxGetState.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_int),
    ]
    driver.cuDevicePrimaryCtxRetain.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    driver.cuDevicePrimaryCtxRelease_v2.argtypes = [ctypes.c_int]
    driver.cuCtxPushCurrent_v2.argtypes = [ctypes.c_void_p]
    driver.cuCtxPopCurrent_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    driver.cuProfilerStart.argtypes = []
    driver.cuProfilerStop.argtypes = []
    for fn in (
        driver.cuInit,
        driver.cuDeviceGetCount,
        driver.cuDeviceGet,
        driver.cuDevicePrimaryCtxGetState,
        driver.cuDevicePrimaryCtxRetain,
        driver.cuDevicePrimaryCtxRelease_v2,
        driver.cuCtxPushCurrent_v2,
        driver.cuCtxPopCurrent_v2,
        driver.cuProfilerStart,
        driver.cuProfilerStop,
    ):
        fn.restype = ctypes.c_int


def _select_device(driver: ctypes.CDLL, device: ctypes.c_int) -> int:
    """Set ``device`` to the first device whose primary context is active.

    Falls back to ordinal 0 when no primary context is active yet.
    """
    count = ctypes.c_int()
    rc = driver.cuDeviceGetCount(ctypes.byref(count))
    if rc != _CUDA_SUCCESS:
        return rc
    flags = ctypes.c_uint()
    active = ctypes.c_int()
    for ordinal in range(count.value):
        rc = driver.cuDeviceGet(ctypes.byref(device), ordinal)
        if rc != _CUDA_SUCCESS:
            return rc
        rc = driver.cuDevicePrimaryCtxGetState(
            device, ctypes.byref(flags), ctypes.byref(active)
        )
        if rc != _CUDA_SUCCESS:
            return rc
        if active.value:
            return _CUDA_SUCCESS
    return driver.cuDeviceGet(ctypes.byref(device), 0)


def _call_in_context(
    driver: ctypes.CDLL, ctx: ctypes.c_void_p, name: str
) -> None:
    """Call a no-argument profiler entry point with ``ctx`` current."""
    rc = driver.cuCtxPushCurrent_v2(ctx)
    if rc != _CUDA_SUCCESS:
        logger.warning("cuCtxPushCurrent returned %d; skipping %s", rc, name)
        return
    try:
        rc = getattr(driver, name)()
        if rc != _CUDA_SUCCESS:
            logger.warning("%s returned %d", name, rc)
    finally:
        driver.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))


@contextmanager
def cuda_profiler_region() -> Iterator[None]:
    """Bracket a region with ``cuProfilerStart`` / ``cuProfilerStop``.

    Yields once with the CUDA profiler started and stops it on exit, so a
    profiler run with ``--capture-range=cudaProfilerApi`` (nsys) or
    ``--profile-from-start off`` (ncu) records only the wrapped region.

    The driver calls act on the current context, and the calling thread
    usually has none: MAX makes its context current only for the duration of
    each driver call. So both calls run with the first already-active primary
    context pushed, which is the one MAX opened for its device. Retaining an
    active context creates nothing; only when no context is active yet does
    this fall back to creating device 0's.

    Raises:
        RuntimeError: If libcuda cannot be loaded; ``cuProfilerStart``
            then cannot be called and the capture would be empty.
    """
    driver = _load_cuda_driver()
    if driver is None:
        raise RuntimeError(
            "libcuda not loadable inside nsys child; cuProfilerStart "
            "cannot be called and the trace would be empty"
        )
    _declare_signatures(driver)

    device = ctypes.c_int()
    ctx = ctypes.c_void_p()
    rc = driver.cuInit(0)
    if rc == _CUDA_SUCCESS:
        rc = _select_device(driver, device)
    if rc == _CUDA_SUCCESS:
        rc = driver.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), device)
    if rc != _CUDA_SUCCESS:
        logger.warning(
            "CUDA driver setup returned %d; the profiler capture will be empty",
            rc,
        )
        yield
        return

    try:
        _call_in_context(driver, ctx, "cuProfilerStart")
        try:
            yield
        finally:
            _call_in_context(driver, ctx, "cuProfilerStop")
    finally:
        driver.cuDevicePrimaryCtxRelease_v2(device)
