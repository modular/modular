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
"""``worker_method`` decorator factory.

Usage::

    class MyWorker(Worker):
        @worker_method()
        async def encode(self, text: str) -> bytes: ...

The factory form is intentional so we can grow keyword args (timeout, retry
policy, telemetry tags, ...) without touching every call site.

A bounded set of ``__call__`` overloads on :py:class:`_WorkerMethodDecorator`
transforms each declared parameter ``T`` into ``MaybeAsync[T]`` for the
*exposed* signature, while the body keeps its original ``T`` typing. At
runtime the wrapper materializes awaitable args via :py:func:`_fetchall`
before invoking the body, so chained proxy calls and direct calls behave
identically.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Mapping,
    Sequence,
)
from typing import Any, Protocol, TypeAlias, TypeVar, cast, overload

from max.experimental.cascade.core.interfaces import Worker
from max.support.taskgroups import TaskGroup

T = TypeVar("T")

# Public alias used both for worker method annotations and helper signatures.
MaybeAsync: TypeAlias = T | Awaitable[T]


async def _await_if_needed(arg: MaybeAsync[T]) -> T:
    """Await ``arg`` if it is awaitable, otherwise return it as-is."""
    if inspect.isawaitable(arg):
        return await arg
    return cast(T, arg)


async def _fetchall(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """Resolve positional and keyword arguments concurrently."""
    async with TaskGroup() as tg:
        arg_tasks = [tg.create_task(_await_if_needed(v)) for v in args]
        kwarg_tasks = {
            k: tg.create_task(_await_if_needed(v)) for k, v in kwargs.items()
        }
    return (
        [t.result() for t in arg_tasks],
        {k: t.result() for k, t in kwarg_tasks.items()},
    )


S = TypeVar("S", bound=Worker)
R = TypeVar("R")
A1 = TypeVar("A1")
A2 = TypeVar("A2")
A3 = TypeVar("A3")
A4 = TypeVar("A4")
A5 = TypeVar("A5")
A6 = TypeVar("A6")


class _WorkerMethodDecorator(Protocol):
    """Type of the decorator returned by :py:func:`worker_method`.

    The ``__call__`` overloads carry the per-arity signature transform so
    mypy can match the worker method's declared parameter types and rewrite
    each one to ``MaybeAsync[T]`` on the exposed signature.
    """

    # Coroutine (``async def``) overloads.
    @overload
    def __call__(
        self, f: Callable[[S], Coroutine[Any, Any, R]], /
    ) -> Callable[[S], Coroutine[Any, Any, R]]: ...
    @overload
    def __call__(
        self, f: Callable[[S, A1], Coroutine[Any, Any, R]], /
    ) -> Callable[[S, MaybeAsync[A1]], Coroutine[Any, Any, R]]: ...
    @overload
    def __call__(
        self, f: Callable[[S, A1, A2], Coroutine[Any, Any, R]], /
    ) -> Callable[
        [S, MaybeAsync[A1], MaybeAsync[A2]], Coroutine[Any, Any, R]
    ]: ...
    @overload
    def __call__(
        self, f: Callable[[S, A1, A2, A3], Coroutine[Any, Any, R]], /
    ) -> Callable[
        [S, MaybeAsync[A1], MaybeAsync[A2], MaybeAsync[A3]],
        Coroutine[Any, Any, R],
    ]: ...
    @overload
    def __call__(
        self, f: Callable[[S, A1, A2, A3, A4], Coroutine[Any, Any, R]], /
    ) -> Callable[
        [S, MaybeAsync[A1], MaybeAsync[A2], MaybeAsync[A3], MaybeAsync[A4]],
        Coroutine[Any, Any, R],
    ]: ...
    @overload
    def __call__(
        self, f: Callable[[S, A1, A2, A3, A4, A5], Coroutine[Any, Any, R]], /
    ) -> Callable[
        [
            S,
            MaybeAsync[A1],
            MaybeAsync[A2],
            MaybeAsync[A3],
            MaybeAsync[A4],
            MaybeAsync[A5],
        ],
        Coroutine[Any, Any, R],
    ]: ...
    @overload
    def __call__(
        self,
        f: Callable[[S, A1, A2, A3, A4, A5, A6], Coroutine[Any, Any, R]],
        /,
    ) -> Callable[
        [
            S,
            MaybeAsync[A1],
            MaybeAsync[A2],
            MaybeAsync[A3],
            MaybeAsync[A4],
            MaybeAsync[A5],
            MaybeAsync[A6],
        ],
        Coroutine[Any, Any, R],
    ]: ...
    # Async-generator (``async def`` with ``yield``) overloads.
    @overload
    def __call__(
        self, f: Callable[[S], AsyncIterator[R]], /
    ) -> Callable[[S], AsyncIterator[R]]: ...
    @overload
    def __call__(
        self, f: Callable[[S, A1], AsyncIterator[R]], /
    ) -> Callable[[S, MaybeAsync[A1]], AsyncIterator[R]]: ...
    @overload
    def __call__(
        self, f: Callable[[S, A1, A2], AsyncIterator[R]], /
    ) -> Callable[[S, MaybeAsync[A1], MaybeAsync[A2]], AsyncIterator[R]]: ...
    @overload
    def __call__(
        self, f: Callable[[S, A1, A2, A3], AsyncIterator[R]], /
    ) -> Callable[
        [S, MaybeAsync[A1], MaybeAsync[A2], MaybeAsync[A3]], AsyncIterator[R]
    ]: ...
    @overload
    def __call__(
        self, f: Callable[[S, A1, A2, A3, A4], AsyncIterator[R]], /
    ) -> Callable[
        [S, MaybeAsync[A1], MaybeAsync[A2], MaybeAsync[A3], MaybeAsync[A4]],
        AsyncIterator[R],
    ]: ...
    @overload
    def __call__(
        self, f: Callable[[S, A1, A2, A3, A4, A5], AsyncIterator[R]], /
    ) -> Callable[
        [
            S,
            MaybeAsync[A1],
            MaybeAsync[A2],
            MaybeAsync[A3],
            MaybeAsync[A4],
            MaybeAsync[A5],
        ],
        AsyncIterator[R],
    ]: ...
    @overload
    def __call__(
        self, f: Callable[[S, A1, A2, A3, A4, A5, A6], AsyncIterator[R]], /
    ) -> Callable[
        [
            S,
            MaybeAsync[A1],
            MaybeAsync[A2],
            MaybeAsync[A3],
            MaybeAsync[A4],
            MaybeAsync[A5],
            MaybeAsync[A6],
        ],
        AsyncIterator[R],
    ]: ...
    # Fallback overloads for methods with more than 6 positional args. These
    # preserve the coroutine-vs-async-gen distinction (so :py:class:`Proxy`
    # dispatch stays correct) but lose the per-arg ``MaybeAsync[T]`` rewrite.
    # Add more arity-specific overloads above if precise typing is needed for a
    # wider method.
    @overload
    def __call__(
        self, f: Callable[..., Coroutine[Any, Any, R]], /
    ) -> Callable[..., Coroutine[Any, Any, R]]: ...
    @overload
    def __call__(
        self, f: Callable[..., AsyncIterator[R]], /
    ) -> Callable[..., AsyncIterator[R]]: ...


def worker_method() -> _WorkerMethodDecorator:
    """Build a decorator for an worker method.

    The factory form (``@worker_method()`` rather than ``@worker_method``)
    leaves room to grow keyword args (timeout, retry policy, telemetry tags,
    ...) without touching every call site.

    Each declared positional parameter ``T`` is exposed to callers as
    ``MaybeAsync[T]``; the wrapper resolves awaitables via :py:func:`_fetchall`
    before invoking the body, which keeps its original ``T`` typing.

    The decorator preserves the coroutine vs. async-generator nature of the
    underlying function so :py:class:`Proxy` continues to dispatch correctly.
    """
    return cast(_WorkerMethodDecorator, _wrap)


def _wrap(f: Callable[..., Any]) -> Callable[..., Any]:
    """Materialize ``MaybeAsync`` args before invoking ``f``.

    The signature is intentionally untyped (``Callable[..., Any]``): the
    public type-level contract for callers lives on
    :py:class:`_WorkerMethodDecorator`'s ``__call__`` overloads, and
    :py:func:`worker_method` casts ``_wrap`` to that protocol. Adding ParamSpec
    plumbing here would not propagate beyond the cast, so we keep this layer
    minimal and rely on the protocol to do the real type checking.
    """
    if inspect.isasyncgenfunction(f):

        @functools.wraps(f)
        async def gen_wrapper(
            self: Any, *args: Any, **kwargs: Any
        ) -> AsyncIterator[Any]:
            resolved_args, resolved_kwargs = await _fetchall(args, kwargs)
            async for item in f(self, *resolved_args, **resolved_kwargs):
                yield item

        return gen_wrapper

    @functools.wraps(f)
    async def coro_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        resolved_args, resolved_kwargs = await _fetchall(args, kwargs)
        return await f(self, *resolved_args, **resolved_kwargs)

    return coro_wrapper
