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
"""Out-of-process graph compilation.

Compiling multiple graphs in parallel in-process is currently
unsafe. :class:`ProcessCompilePool` compiles graphs in background worker
processes (a :class:`~concurrent.futures.ProcessPoolExecutor`).

Forking thread pools such as exist in Mojo and the graph compiler is not
typically safe. ProcessCompilePool works around this using ``forkserver``.
An initial process is created which imports `max.engine` to share import
overhead among workers, and then compile workers are forked from this process.

Graphs cross to a worker as module bytecode. Compiled models are saved
to MEF paths to avoid pickling large model binaries over pipes.
"""

from __future__ import annotations

import atexit
import ctypes
import multiprocessing
import os
import signal
import sys
import tempfile
import traceback
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import TypeVar

from max import driver, engine, mlir
from max._core import Operation
from max.graph import Graph, Module

_T = TypeVar("_T")
_R = TypeVar("_R")


def _map_future(future: Future[_T], f: Callable[[_T], _R]) -> Future[_R]:
    """Returns a future resolving to ``f(future.result())``.

    ``f`` runs on the thread that resolves ``future``.
    A failed or cancelled input resolves the returned future with the
    same exception.
    """
    result: Future[_R] = Future()

    def resolve(done: Future[_T]) -> None:
        if not result.set_running_or_notify_cancel():
            return
        try:
            result.set_result(f(done.result()))
        except BaseException as e:
            result.set_exception(e)

    future.add_done_callback(resolve)
    return result


class RemoteCompileError(RuntimeError):
    """A compile failed in the worker; the message embeds its traceback."""


def _pin_to_parent_death() -> None:
    """Asks Linux to SIGKILL this process when its parent dies uncleanly."""
    if sys.platform != "linux":
        return
    PR_SET_PDEATHSIG = 1
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)


# Worker-process state, created once per worker by _init_worker.
_WORKER_SESSION: engine.InferenceSession | None = None


def _init_worker(device_specs: Sequence[driver.DeviceSpec]) -> None:
    """Pins the worker's lifetime and builds its compile session."""
    global _WORKER_SESSION
    _pin_to_parent_death()
    _WORKER_SESSION = engine.InferenceSession(
        devices=driver.load_devices(device_specs)
    )


def _compile_to_mef(
    bytecode: bytes, extensions: Sequence[Path], mef_path: Path
) -> Path:
    assert _WORKER_SESSION is not None, "worker initializer did not run"

    try:
        module = Module(
            mlir_module=Operation.from_bytecode(bytecode, mlir.Context())
        )
        compiled = _WORKER_SESSION.compile(module, custom_extensions=extensions)
        compiled.export_mef(mef_path)
    except BaseException:
        # Tracebacks don't survive pickling.
        # Format the trace into the exception message.
        raise RemoteCompileError(
            f"failed to compile with error: {traceback.format_exc()}"
        ) from None

    return mef_path


class ProcessCompilePool:
    """Compiles graphs in background worker processes.

    Forking thread pools such as exist in Mojo and the graph compiler is not
    typically safe. ProcessCompilePool works around this using ``forkserver``.
    An initial process is created which imports `max.engine` to share import
    overhead among workers, and then compile workers are forked from this process.

    Graphs cross to a worker as module bytecode. Compiled models are saved
    to MEF paths to avoid pickling large model binaries over pipes.

    .. code-block:: python

        with ProcessCompilePool() as pool:
            future = pool.compile(graph)
            model = session.init(future.result())
    """

    def __init__(
        self,
        device_specs: Sequence[driver.DeviceSpec] | None = None,
        max_workers: int = os.cpu_count() or 1,
    ) -> None:
        """Creates a pool whose worker sessions use ``device_specs``.

        Args:
            device_specs: Devices for the workers' sessions. If None,
                defaults to all available accelerators plus the CPU.
            max_workers: Upper bound on concurrent compiles. Each worker
                holds an :class:`~max.engine.InferenceSession` for the
                given devices which reserves some device memory.
        """
        if device_specs is None:
            specs = driver.scan_available_devices()
            if (cpu := driver.DeviceSpec.cpu()) not in specs:
                specs.append(cpu)
            device_specs = specs

        # max import is O(seconds), do this pre-fork for each worker
        context = multiprocessing.get_context("forkserver")
        context.set_forkserver_preload(["max.engine"])
        self._executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(tuple(device_specs),),
        )

        self._mef_dir = tempfile.TemporaryDirectory(prefix="max-compile-pool-")
        self._closed = False

        # Cancel workers on parent exit
        atexit.register(self.close)

    def compile(self, graph: Graph) -> Future[engine.CompiledModel]:
        """Schedules ``graph`` for compilation.

        Returns:
            A future resolving to the compiled artifact, ready for
            :meth:`~max.engine.InferenceSession.init` on any session. The
            artifact is mmapped and owns its lifetime: it stays valid
            after the pool closes.

        Raises:
            RuntimeError: If the pool is closed.
            BrokenProcessPool: If a worker has already died.
        """
        if self._closed:
            raise RuntimeError("the compile pool is closed")

        mef_future = self._executor.submit(
            _compile_to_mef,
            graph._module.bytecode,
            graph.kernel_libraries_paths,
            Path(self._mef_dir.name) / f"{uuid.uuid4()}.mef",
        )

        def load(mef_path: Path) -> engine.CompiledModel:
            try:
                compiled = engine.read(mef_path)
            except Exception:
                # A compile that resolves while close() runs can load
                # against the deleted MEF dir.
                if self._closed:
                    raise CancelledError() from None
                raise
            mef_path.unlink(missing_ok=True)
            return compiled

        return _map_future(mef_future, load)

    def close(self) -> None:
        """Stops the pool, discarding queued and in-flight compiles.

        Unfinished futures are cancelled or raise
        :class:`~concurrent.futures.process.BrokenProcessPool`.
        """
        if self._closed:
            return

        self._closed = True
        atexit.unregister(self.close)
        # Terminate before shutdown(): worker death after shutdown
        # doesn't resolve futures.
        for process in (self._executor._processes or {}).values():
            process.terminate()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._mef_dir.cleanup()

    def __enter__(self) -> ProcessCompilePool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Waits for outstanding queue items and then closes the pool."""
        self._executor.__exit__(exc_type, exc_val, exc_tb)
        self.close()
