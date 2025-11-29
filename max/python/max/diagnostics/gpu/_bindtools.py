# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Utilities for binding with ctypes using typing.Protocol for signatures."""

from __future__ import annotations

import ctypes
import inspect
import sys
from collections.abc import Callable
from typing import Annotated, Any, TypeVar, cast, get_args, get_origin

_T = TypeVar("_T")
_TCData = TypeVar("_TCData", bound="ctypes._CData")


def _get_protocol_members(protocol: type[_T]) -> frozenset[str]:
    if sys.version_info >= (3, 13):
        from typing import get_protocol_members

        return get_protocol_members(protocol)

    # Imperfect, but perhaps good enough
    return frozenset(
        key for key in protocol.__dict__ if not key.startswith("_")
    )


# Using Callable[[], _T] as a workaround to MyPy issue #4717.  See
# https://github.com/python/mypy/issues/4717#issuecomment-454609539.
def bind_protocol(dll: ctypes.CDLL, protocol: Callable[[], _T]) -> _T:
    if sys.version_info >= (3, 13):
        from typing import is_protocol

        if not is_protocol(protocol):
            raise TypeError("Protocol must actually be a protocol")
    assert isinstance(protocol, type)
    module_name = getattr(protocol, "__module__", None)
    if module_name is None or (module := sys.modules.get(module_name)) is None:
        raise TypeError("Unable to find protocol's module")
    if getattr(module, protocol.__name__, None) is not protocol:
        # This alone is not a problem but it's an indication our later
        # annotation evals might not work right.
        raise TypeError("Protocol not global in its module")
    members = _get_protocol_members(protocol)

    def resolve_annotation(annotation: Any) -> Any:
        if isinstance(annotation, inspect._empty):
            raise TypeError("Parameter/return must be annotated")
        if get_origin(annotation) is Annotated:
            # Retrieve the metadata, not the type itself, here.
            return resolve_annotation(get_args(annotation)[1])
        if not isinstance(annotation, str):
            return annotation
        # Oh boy, a string!  We'll have to eval this ourselves.
        if annotation.startswith("ctypes._Pointer[") and annotation.endswith(
            "]"
        ):
            # ctypes._Pointer barfs if you try to subscript it at runtime, so
            # intercept this (hackily) and try our best.
            inner = resolve_annotation(annotation[16:-1])
            return ctypes.POINTER(inner)
        return resolve_annotation(eval(annotation, module.__dict__))

    for member in members:
        cfunc = getattr(dll, member)
        # Can't use eval_str=True because (1) it's only available in Python
        # 3.10, and we're targeting 3.9+, and (2) we use ctypes._Pointer[...]
        # in annotations, and ctypes._Pointer isn't actually subscriptable at
        # runtime (which we do something very hacky about).
        signature = inspect.signature(getattr(protocol, member))
        cfunc.argtypes = [
            resolve_annotation(parameter.annotation)
            for parameter in list(signature.parameters.values())[1:]
        ]
        cfunc.restype = resolve_annotation(signature.return_annotation)
    assert isinstance(dll, protocol)
    # MyPy would understand this without cast if not for our #4717 workaround.
    return cast(_T, dll)


# Sort of a hack to get around typeshed's silly typing of ctypes.byref.
# (It mostly makes sense -- the result can only be used in arguments.
# But then it erases the type it points to, which is not that useful.)
def byref(value: _TCData) -> ctypes._Pointer[_TCData]:
    return cast("ctypes._Pointer[_TCData]", value)
