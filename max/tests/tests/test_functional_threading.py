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

import threading

from max.experimental import functional as F
from max.experimental.tensor import Tensor


def _run_simple_op() -> Tensor:
    a = Tensor.zeros((4, 8))
    b = Tensor.zeros((4, 8))
    return F.add(a, b)


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
