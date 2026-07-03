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
"""``Result`` and ``ResultIter`` handles returned by proxy worker methods.

A handle carries a ``result_id`` plus the :py:class:`Runtime` that owns
the binding. The wire boundary keeps these handles uniform across
transports by relying on the runtime being picklable -- the local
in-process runtime pickles by reference (within a process), and remote
runtime proxies (e.g. :py:class:`HttpRuntimeProxy`) implement
``__getstate__`` / ``__setstate__`` to carry only their address.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

if TYPE_CHECKING:
    from max.experimental.cascade.core.interfaces import Runtime

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Result(Generic[T]):
    """Awaitable handle to a single worker-method result."""

    result_id: str
    runtime: Runtime

    def __await__(self) -> Generator[Any, None, T]:
        # ``Runtime.get_result`` is typed ``Awaitable[object]`` because
        # the runtime can't statically know the worker method's return
        # type; ``T`` is carried by the static :py:class:`Proxy` binding.
        return cast(
            "Generator[Any, None, T]",
            self.runtime.get_result(self.result_id).__await__(),
        )


@dataclass(frozen=True, slots=True)
class ResultIter(Generic[T]):
    """Async-iterable handle to a streamed worker-method result.

    Streaming counterpart of :py:class:`Result`; same picklability model.
    """

    result_id: str
    runtime: Runtime

    def __aiter__(self) -> AsyncIterator[T]:
        return cast(
            "AsyncIterator[T]",
            self.runtime.stream_result(self.result_id),
        )
