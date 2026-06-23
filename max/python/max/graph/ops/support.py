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
"""Shared helpers for op implementations that take user-provided callables."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..type import Type
from ..value import TensorValue, TensorValueLike, Value


def as_iterable(
    x: Iterable[Value[Any] | TensorValueLike]
    | Value[Any]
    | TensorValueLike
    | None,
) -> Iterable[Value[Any] | TensorValueLike]:
    """Normalize a user-returned result into an iterable of values.

    Callables passed to control-flow ops (e.g., ``ops.cond``,
    ``ops.while_loop``) are allowed to return ``None``, a single value, or
    an iterable of values. Use this to flatten that variation away.
    """
    if x is None:
        return ()
    return x if isinstance(x, Iterable) else (x,)


def as_values(
    xs: Iterable[Value[Any] | TensorValueLike],
    expected_types: list[Type[Any]] | None = None,
) -> list[Value[Any]]:
    """Coerce each item to a :class:`~max.graph.Value`.

    :class:`~max.graph.TensorValueLike` items get wrapped with
    :class:`~max.graph.TensorValue`. If ``expected_types`` is given, raises
    :class:`TypeError` when the resulting value types don't match — useful
    for branch / body callables where the op signature dictates the result
    shape.
    """
    values = [x if isinstance(x, Value) else TensorValue(x) for x in xs]
    if expected_types is not None:
        result_types = [v.type for v in values]
        if result_types != expected_types:
            raise TypeError(
                "Results don't match expected types: \n"
                f"{result_types=}, \n{expected_types=}"
            )
    return values
