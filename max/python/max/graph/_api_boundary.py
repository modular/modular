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

"""Marks the public entry points that create graph ops.

A source traceback shows where the caller's code created an op. The frames
below the caller's first call into the graph API are MAX's implementation of
that op, so :func:`callers_of_api` cuts the stack there::

    main()                  <- kept: the caller's code, whoever wrote it
      forward()             <- kept
        ops.mean(x)         <- boundary: cut here
          ops.sum(x)        <- boundary nested in it
            Graph._add_op   <- MAX's implementation

The boundary is the function, not where its file sits, so the cut holds
wherever MAX is installed or its modules move. A function is registered by
its code object, which costs nothing on the call itself.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Iterator, Mapping
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

_BOUNDARY_CODE: set[types.CodeType] = set()


def api_boundary(fn: _F) -> _F:
    """Marks ``fn`` as a public entry point that creates graph ops."""
    _BOUNDARY_CODE.add(fn.__code__)
    return fn


def _functions(value: object) -> Iterator[types.FunctionType]:
    """Yields the plain functions behind a namespace entry."""
    if isinstance(value, (staticmethod, classmethod)):
        value = value.__func__
    if isinstance(value, property):
        yield from (
            f
            for f in (value.fget, value.fset, value.fdel)
            if isinstance(f, types.FunctionType)
        )
    elif isinstance(value, types.FunctionType):
        yield value


def _is_public(name: str) -> bool:
    # Operators such as `__add__` are public API too.
    return not name.startswith("_") or (
        name.startswith("__") and name.endswith("__")
    )


def register_api_boundaries(
    namespace: Mapping[str, object], package: str
) -> None:
    """Marks every public function of ``namespace`` defined in ``package``.

    ``namespace`` is a module's or a class's ``__dict__``. Functions it
    imports from outside ``package`` are left alone.
    """
    for name, value in namespace.items():
        if not _is_public(name):
            continue
        for fn in _functions(value):
            if fn.__module__.startswith(package):
                _BOUNDARY_CODE.add(fn.__code__)


def is_api_boundary(fn: object) -> bool:
    """Returns whether ``fn`` is a registered API boundary."""
    code = getattr(fn, "__code__", None)
    return code is not None and code in _BOUNDARY_CODE


def callers_of_api(
    frame: types.FrameType | None,
) -> list[types.FrameType] | None:
    """Returns the frames that called into the graph API, outermost first.

    These are the frames above the outermost boundary on ``frame``'s stack.
    Returns ``None`` when the stack has no boundary, as when MAX builds an op
    through an internal path.
    """
    frames: list[types.FrameType] = []
    while frame is not None:
        frames.append(frame)
        frame = frame.f_back
    frames.reverse()
    for i, f in enumerate(frames):
        if f.f_code in _BOUNDARY_CODE:
            return frames[:i]
    return None
