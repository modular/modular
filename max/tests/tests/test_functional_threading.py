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
"""Tests that functional ops work from non-main threads.

The MLIR ``Context.current`` is thread-local and only entered on the main
thread at import time. The ``functional`` wrapper must enter the default
MLIR context on background threads, otherwise ``Graph`` construction inside
``EagerRealizationContext.__init__`` fails with "No MLIR context active".
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable

from max.experimental import functional as F
from max.experimental import tensor as tensor_mod
from max.experimental.realization_context import EagerRealizationContext
from max.experimental.tensor import Tensor


def _run_simple_op() -> Tensor:
    a = Tensor.zeros((4, 8))
    b = Tensor.zeros((4, 8))
    return F.add(a, b)


def _run_in_thread(target: Callable[[], None]) -> list[BaseException]:
    errors: list[BaseException] = []

    def wrapper() -> None:
        try:
            target()
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=wrapper)
    t.start()
    t.join()
    return errors


def test_functional_op_in_background_thread() -> None:
    result: list[Tensor] = []
    errors: list[BaseException] = []

    def target() -> None:
        try:
            result.append(_run_simple_op())
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=target)
    t.start()
    t.join()

    assert not errors, f"op raised in background thread: {errors[0]!r}"
    assert len(result) == 1
    assert list(result[0].shape) == [4, 8]


def test_lazy_context_in_background_thread() -> None:
    result: list[Tensor] = []
    errors: list[BaseException] = []

    def target() -> None:
        try:
            with F.lazy():
                result.append(_run_simple_op())
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=target)
    t.start()
    t.join()

    assert not errors, f"op raised in background thread: {errors[0]!r}"
    assert len(result) == 1
    assert list(result[0].shape) == [4, 8]


def test_explicit_eager_context_in_background_thread() -> None:
    result: list[Tensor] = []
    errors: list[BaseException] = []

    def target() -> None:
        try:
            ctx = EagerRealizationContext()
            with ctx, tensor_mod.realization_context(ctx):
                result.append(_run_simple_op())
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=target)
    t.start()
    t.join()

    assert not errors, f"op raised in background thread: {errors[0]!r}"
    assert len(result) == 1
    assert list(result[0].shape) == [4, 8]


def test_lazy_realize_after_exit_in_background_thread() -> None:
    """The lazy path defers realize until awaited.

    After ``with F.lazy():`` exits on a background thread, ``mlir.Context``
    is no longer current on that thread. Realizing the deferred tensor
    must still re-enter the default MLIR context, otherwise
    ``finalize_graph()`` crashes inside ``mlir.FunctionType.get``.
    """
    result: list[Tensor] = []

    def target() -> None:
        with F.lazy():
            t = _run_simple_op()
        assert not t.real
        t._sync_realize()
        assert t.real
        result.append(t)

    errors = _run_in_thread(target)

    assert not errors, f"op raised in background thread: {errors[0]!r}"
    assert len(result) == 1
    assert result[0].real


def test_explicit_realize_all_in_background_thread() -> None:
    """Calling ``realize_all`` directly on a background thread must work.

    Some callers (and tests) drive realization via
    ``asyncio.run(ctx.realize_all())`` rather than letting ``__exit__`` do it.
    The MLIR context must be ensured by ``realize_all`` itself so this
    path doesn't depend on the caller's thread state.
    """

    def target() -> None:
        ctx = EagerRealizationContext()
        with tensor_mod.realization_context(ctx), ctx.graph:
            a = Tensor.zeros((4, 8))
            b = Tensor.zeros((4, 8))
            _ = F.add(a, b)
        # Explicitly drive realize_all on this thread without entering ctx.
        asyncio.run(ctx.realize_all())

    errors = _run_in_thread(target)

    assert not errors, f"op raised in background thread: {errors[0]!r}"
