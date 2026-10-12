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
"""Python-side conveniences for :class:`max._core.mlrt.AsyncValue`.

The C++ binding (``max._core.mlrt.AsyncValue``) deliberately exposes only
the primitives (``done``, ``result``, ``exception``, ``set_result``,
``set_exception``, ``add_done_callback``, ``from_future``). The
``__await__`` integration with ``asyncio`` is composed from those
primitives here and monkeypatched onto the class at import time, since
nanobind can't synthesize Python generators ergonomically from C++.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any, TypeVar

from max._core.mlrt import AsyncValue as AsyncValue

_T = TypeVar("_T")


def _async_value_await(self: AsyncValue[_T]) -> Generator[Any, None, Any]:
    """``await async_value`` support.

    Fast path: if already done, return / raise immediately (no event loop
    interaction). Slow path: bridge resolution to an ``asyncio.Future`` on
    the running loop via ``call_soon_threadsafe`` and ``yield from`` it,
    so suspension and exception propagation use the standard machinery.
    """
    if self.done():
        if self.is_error():
            raise self.exception()  # type: ignore[misc]
        return self.result()

    loop = asyncio.get_running_loop()
    fut: asyncio.Future[Any] = loop.create_future()

    def _transfer(av: AsyncValue[Any]) -> None:
        # Runs on whatever thread MLRT resolves on (with the GIL held).
        # Hop to the loop thread before touching the Future.
        if loop.is_closed():
            return
        if av.is_error():
            exc = av.exception()
            assert exc is not None  # implied by is_error()
            loop.call_soon_threadsafe(fut.set_exception, exc)
        else:
            loop.call_soon_threadsafe(fut.set_result, av.result())

    self.add_done_callback(_transfer)
    return (yield from fut.__await__())


AsyncValue.__await__ = _async_value_await  # type: ignore[method-assign]
