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
"""Private signature-rewriting infrastructure for ``max.experimental.functional``.

Graph ops are annotated with ``TensorValueLike`` / ``StrongTensorValueLike``
/ ``TensorValue``. The functional surface accepts ``Tensor``, so signatures
must show ``Tensor`` to match the public contract.

Aliases collapse as a unit: ``TensorValueLike`` is a ``UnionType`` over
``_Value`` / ``Shape`` / ``Dim`` / ``HasTensorValue`` / ``int`` / ``float`` /
etc., and exposing those internals to callers would mislead them.

In the parameter position, only the ``*Like`` aliases are substituted; bare
``TensorValue`` is left intact.

``BufferValueLike`` is intentionally excluded — buffer ops have hand-written
functional wrappers (``buffer_store``, ``allreduce_sum``, …) and never go
through ``functional()``.
"""

from __future__ import annotations

import functools
import inspect
import operator
import re
import types
import typing
from collections.abc import Callable
from typing import Any

from max.experimental.tensor import Tensor
from max.graph import TensorValue, TensorValueLike
from max.graph.value import StrongTensorValueLike

# Names listed explicitly: ``TensorValueLike`` / ``StrongTensorValueLike``
# are runtime ``UnionType`` aliases (PEP 604 ``|``) with no ``__name__``,
# so the regex source can't be auto-derived from the type objects.
_PARAM_ALIAS_NAMES = ("StrongTensorValueLike", "TensorValueLike")
TENSOR_VALUE_ALIAS_NAMES = (*_PARAM_ALIAS_NAMES, "TensorValue")

_PARAM_ALIASES = (TensorValueLike, StrongTensorValueLike)
_RETURN_ALIASES = (*_PARAM_ALIASES, TensorValue)
_PARAM_ALIAS_RE = re.compile(r"\b(" + "|".join(_PARAM_ALIAS_NAMES) + r")\b")
_RETURN_ALIAS_RE = re.compile(
    r"\b(" + "|".join(TENSOR_VALUE_ALIAS_NAMES) + r")\b"
)


def _substitute_tensor(annotation: object, *, in_return: bool) -> object:
    """Recursively replace tensor-value aliases with ``Tensor``."""
    aliases = _RETURN_ALIASES if in_return else _PARAM_ALIASES
    pattern = _RETURN_ALIAS_RE if in_return else _PARAM_ALIAS_RE
    if annotation is inspect.Parameter.empty:
        return annotation
    # Under ``from __future__ import annotations`` the annotation arrives
    # as a string; do a textual rewrite (containers like
    # ``Iterable[TensorValueLike]`` are handled by the same regex).
    if isinstance(annotation, str):
        return pattern.sub("Tensor", annotation)
    # Identity catches the common case; ``==`` catches union aliases that
    # were reconstructed (e.g. inside a generic container) and so have a
    # different identity than the imported ``TensorValueLike``.
    if any(annotation is a or annotation == a for a in aliases):
        return Tensor
    origin = typing.get_origin(annotation)
    if origin is None:
        return annotation
    args = typing.get_args(annotation)
    if not args:
        return annotation
    new_args = tuple(_substitute_tensor(a, in_return=in_return) for a in args)
    if new_args == args:
        return annotation
    if origin is typing.Union or origin is types.UnionType:
        # Dedup-by-equality (types like ``Annotated[...]`` aren't hashable,
        # so ``dict.fromkeys`` doesn't always work — keep the linear walk).
        unique: list[object] = []
        for a in new_args:
            if a not in unique:
                unique.append(a)
        return functools.reduce(operator.or_, unique)
    return origin[new_args]


def _rewritten_signature(
    fn: Callable[..., Any],
) -> inspect.Signature | None:
    """Signature for ``fn`` with the tensor-type substitution applied, or None."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return None
    return sig.replace(
        parameters=[
            p.replace(
                annotation=_substitute_tensor(p.annotation, in_return=False)
            )
            for p in sig.parameters.values()
        ],
        return_annotation=_substitute_tensor(
            sig.return_annotation, in_return=True
        ),
    )


def install_tensor_signature(fn: Callable[..., Any]) -> None:
    """Mutate ``fn.__signature__`` and ``fn.__annotations__`` to display
    ``Tensor`` in place of the graph-op wrapper types.

    Both attributes must be set: ``inspect.signature()`` consumers read
    ``__signature__``, while Sphinx autodoc reads ``__annotations__``
    directly for the Parameters / Return type sections.
    """
    rewritten = _rewritten_signature(fn)
    if rewritten is None:
        return
    fn.__signature__ = rewritten  # type: ignore[attr-defined]
    new_annotations: dict[str, object] = {
        name: param.annotation
        for name, param in rewritten.parameters.items()
        if param.annotation is not inspect.Parameter.empty
    }
    if rewritten.return_annotation is not inspect.Signature.empty:
        new_annotations["return"] = rewritten.return_annotation
    fn.__annotations__ = new_annotations
