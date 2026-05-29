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
"""Cascade worker framework: base classes for workers, proxies, and runtimes."""

from __future__ import annotations

import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Mapping,
    Sequence,
)
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Concatenate,
    Generic,
    ParamSpec,
    TypeVar,
    cast,
)

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")


@dataclass
class Worker:
    """Base class for cascade workers with deployment metadata."""

    # hints to guide deployment in heterogenous runtime pools
    deploy_hints: list[str] = field(default_factory=list)
    deploy_timeout: float | None = None

    @asynccontextmanager
    async def open(self) -> AsyncIterator[Worker]:
        """Lifecycle context manager. Override to add setup/teardown."""
        yield self


WorkerType = TypeVar("WorkerType", bound=Worker)


# Note having Runtime as a member on Result is actually a feature
# When Runtime is a thin client proxy Runtime, it's serializable as a URL
# We can explicitly hook the rebuild of runtime to memoize by URL
# to enable efficient grpc connection sharing


class Runtime(ABC):
    """Transport-agnostic runtime for worker execution and result delivery."""

    @abstractmethod
    def open(self) -> AbstractAsyncContextManager[Runtime]:
        """Required lifecycle cleanup context manager."""
        ...

    @abstractmethod
    async def deploy_worker(self, worker: Worker) -> str:
        """Wire primitive: register an worker and return its stable id.

        This is the primitive that differs between local and remote runtimes.
        Most callers want :py:meth:`deploy` instead, which wraps the returned
        id in a client-side :py:class:`Proxy`.
        """
        ...

    async def deploy(self, worker: WorkerType) -> WorkerType:
        """Register an worker and return a client-side :py:class:`Proxy`.

        The runtime returns a :py:class:`Proxy` that is observationally
        equivalent to ``worker`` (same public method surface, same awaited
        result types, same async-iterator streaming surface). The return type
        is annotated as ``WorkerType`` so callers can type pipeline attributes
        with their worker classes (or :py:class:`typing.Protocol` interfaces)
        and get end-to-end type checking; the runtime cast is safe because
        :py:class:`Proxy` exposes a structurally compatible interface.
        """
        worker_id = await self.deploy_worker(worker)
        return cast(WorkerType, Proxy(self, worker_id, worker))

    @abstractmethod
    def call_method(
        self,
        actid: str,
        func: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> str:
        """Launch an worker method asynchronously and bind it to a result id."""
        ...

    @abstractmethod
    def get_result(self, resid: str) -> Awaitable[T]:
        """Await a single result future (may throw exception if call failed)."""
        ...

    @abstractmethod
    def next_result(self, resid: str) -> Awaitable[T]:
        """Await a single result future (may throw exception or StopAsyncIteration)."""
        ...

    @abstractmethod
    async def get_metrics(self) -> str:
        """Return Prometheus metrics summary."""
        ...


# ---------------------------------------------------------------------------
# ``Proxy`` and its result handles live below ``Runtime`` so the cross-class
# references resolve top-down, avoiding the cross-module circular import we
# hit when this lived in its own file.
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Result(Generic[T]):
    """Awaitable handle to a single result produced by a ``Runtime``."""

    runtime: Runtime
    result_id: str

    def __await__(self) -> Generator[Any, None, T]:
        return self.runtime.get_result(self.result_id).__await__()


@dataclass(slots=True, frozen=True)
class ResultIter(Generic[T]):
    """Async-iterable handle to a streamed result from a ``Runtime``."""

    runtime: Runtime
    result_id: str

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        # ``Runtime.next_result`` already raises ``StopAsyncIteration``
        # at end-of-stream, which terminates iteration cleanly.
        return await self.runtime.next_result(self.result_id)


class Proxy(Generic[T]):
    """Client-side handle to a deployed :py:class:`Worker`.

    Inspects ``type(worker)`` at construction time and attaches one bound
    method per non-underscore worker method. Each generated method calls
    ``runtime.call_method`` and wraps the returned ``result_id`` in a
    :py:class:`Result` (or :py:class:`ResultIter` for async-generator
    methods).
    """

    runtime: Runtime
    worker_id: str

    def __init__(
        self, runtime: Runtime, worker_id: str, worker: Worker
    ) -> None:
        self.runtime = runtime
        self.worker_id = worker_id
        worker_class = type(worker)
        for name in dir(worker_class):
            if name.startswith("_"):
                continue
            method = getattr(worker_class, name, None)
            if not callable(method):
                continue
            if inspect.isasyncgenfunction(method):
                setattr(self, name, self._bind_stream(name, method))
            else:
                setattr(self, name, self._bind_call(name, method))

    # ``method`` is the unbound class method, so its first parameter is ``self``;
    # we strip it via ``Concatenate[WorkerType, P]`` (a phantom TypeVar that
    # mypy unifies with the concrete worker subclass per call) and produce a
    # wrapper whose parameter list is ``P`` and whose return type tracks the
    # worker method's.
    def _bind_call(
        self,
        name: str,
        method: Callable[Concatenate[WorkerType, P], Coroutine[Any, Any, R]],
    ) -> Callable[P, Result[R]]:
        """Build a wrapper for a coroutine worker method."""
        runtime = self.runtime
        worker_id = self.worker_id

        @functools.wraps(method)
        def call(*args: P.args, **kwargs: P.kwargs) -> Result[R]:
            rid = runtime.call_method(worker_id, name, args, kwargs)
            return Result(runtime, rid)

        return call

    def _bind_stream(
        self,
        name: str,
        method: Callable[Concatenate[WorkerType, P], AsyncIterator[R]],
    ) -> Callable[P, ResultIter[R]]:
        """Build a wrapper for an async-generator worker method."""
        runtime = self.runtime
        worker_id = self.worker_id

        @functools.wraps(method)
        def call_stream(*args: P.args, **kwargs: P.kwargs) -> ResultIter[R]:
            rid = runtime.call_method(worker_id, name, args, kwargs)
            return ResultIter(runtime, rid)

        return call_stream
