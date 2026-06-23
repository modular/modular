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
"""``Function`` — Python projection of HAL ``FunctionHandle``."""

from __future__ import annotations

from typing import Any


class Function:
    """A launchable function resolved from a Bundle.

    Obtained via
    ``context.load_function(bundle, name)``.

    Passed to ``Queue.execute(function, ...)`` to launch the kernel.

    """

    _inner: Any

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        raise TypeError(
            "Function is not directly constructible; use "
            "Context.load_function(bundle, name)."
        )

    @classmethod
    def _wrap(cls, inner: object) -> Function:
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    def __repr__(self) -> str:
        return "Function()"

    __str__ = __repr__
