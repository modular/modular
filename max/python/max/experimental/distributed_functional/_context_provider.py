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

"""Realization-context helpers and the ``functional`` op wrapper.

This module provides:

* :func:`functional` — decorator that wraps graph ops or dispatch functions
  so they work with :class:`~max.experimental.tensor.Tensor`.  It sets up a
  realization context and converts results back to Tensors.  Distributed
  (sharded) dispatch is the responsibility of each op individually.
* :func:`lazy` — context manager for deferred tensor realization.
* :func:`in_graph_context` — utility to check if inside a ``Graph`` block.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Generator, Iterable
from typing import Any, TypeAlias, TypeVar, overload

from max import driver
from max.experimental import realization_context as rc
from max.experimental import tensor
from max.graph import BufferValue, Graph, TensorValue
from typing_extensions import ParamSpec

Args = ParamSpec("Args")
Result = TypeVar("Result")
Op = Callable[Args, Result]


_ConvertibleToTensor: TypeAlias = (
    driver.Buffer | tensor.Tensor | TensorValue | BufferValue
)


def _to_tensor(value: _ConvertibleToTensor) -> tensor.Tensor:
    """Converts a tensor-like value to a Tensor."""
    if isinstance(value, tensor.Tensor):
        return value
    elif isinstance(value, driver.Buffer):
        return tensor.Tensor(storage=value)
    return tensor.Tensor.from_graph_value(value)


@overload
def _to_tensors(value: _ConvertibleToTensor, /) -> tensor.Tensor: ...


@overload
def _to_tensors(value: None, /) -> None: ...


@overload
def _to_tensors(
    values: Iterable[_ConvertibleToTensor],
) -> list[tensor.Tensor]: ...


def _to_tensors(values):
    """Converts one or more tensor-like values to Tensors."""
    if values is None:
        return None
    if isinstance(values, _ConvertibleToTensor):
        return _to_tensor(values)
    return [_to_tensor(value) for value in values]


def _return_tensors(op: Op[..., Any]) -> Op[..., Any]:
    """Decorator that converts operation results to Tensors."""

    @functools.wraps(op)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        results = op(*args, **kwargs)
        return _to_tensors(results)

    return wrapped


def in_graph_context() -> bool:
    """Checks whether the caller is inside a Graph context."""
    try:
        _ = Graph.current
    except LookupError:
        return False
    else:
        return True


def _ensure_context(stack: contextlib.ExitStack) -> None:
    """Push a distributed-capable realization context if none is active.

    If the active context is a non-distributed one (from the regular
    ``experimental/realization_context.py``), it is replaced with a
    distributed context that supports signal buffers for collectives.
    """
    current = tensor.current_realization_context(None)
    need_new = current is None or not isinstance(
        current, (rc.EagerRealizationContext, rc.GraphRealizationContext)
    )
    if need_new:
        ctx = (
            rc.GraphRealizationContext(Graph.current)
            if in_graph_context()
            else rc.EagerRealizationContext()
        )
        stack.enter_context(ctx)
        stack.enter_context(tensor.realization_context(ctx))


def _validate_distributed_args(
    args: tuple[Any, ...], kwargs: dict[str, Any], op_name: str
) -> None:
    """Validates distributed consistency of Tensor args.

    When any Tensor arg is distributed, ALL other Tensor args must also
    be distributed and on the same mesh.  Non-Tensor values (scalars,
    TensorValues, ints, floats) are ignored — they are used directly
    per-shard inside dispatch rules without any distribution.
    """
    tensors: list[tensor.Tensor] = []
    for a in args:
        if isinstance(a, tensor.Tensor):
            tensors.append(a)
        elif isinstance(a, (list, tuple)):
            tensors.extend(t for t in a if isinstance(t, tensor.Tensor))
    for v in kwargs.values():
        if isinstance(v, tensor.Tensor):
            tensors.append(v)

    if not tensors:
        return

    distributed = [t for t in tensors if t.is_distributed]
    if not distributed:
        return

    # At least one Tensor is distributed — all Tensors must be.
    non_distributed = [t for t in tensors if not t.is_distributed]
    if non_distributed:
        raise ValueError(
            f"{op_name}: got mixed distributed and non-distributed Tensor "
            f"operands. Use F.shard() to distribute all Tensors first."
        )

    # All distributed — must share the same mesh.
    ref_mesh = distributed[0].mesh
    for t in distributed[1:]:
        if t.mesh != ref_mesh:
            raise ValueError(
                f"{op_name}: all distributed Tensors must be on the same "
                f"DeviceMesh. Got {ref_mesh} and {t.mesh}."
            )


def functional(
    fn: Op[..., Any] | None = None,
    *,
    linear: bool | None = False,
) -> Op[..., Any]:
    """Wraps a graph op or dispatch function for use with Tensors.

    Sets up a realization context, validates that all distributed args
    share the same mesh, optionally resolves Partial placements, calls
    the function, and converts results to Tensors.

    Args:
        fn: The function to wrap.
        linear: Controls automatic Partial placement resolution.

            * ``False`` (default) — auto-reduce all Partial inputs before
              the op runs.  Correct for non-linear ops (relu, softmax, …)
              where ``f(a₁ + a₂) ≠ f(a₁) + f(a₂)``.
            * ``True`` — linear op: skip reduction when **all** distributed
              inputs are Partial (the op preserves the sum), auto-reduce
              when mixed.  Used by add, sub, negate, sum, mean.
            * ``None`` — skip Partial resolution entirely; the wrapped op
              has its own Partial logic.  Used by collectives (they *are*
              the reduce), matmul (``PxR -> P``), shape ops (Partial passes
              through), and creation ops (no tensor inputs).
    """
    if fn is None:
        return functools.partial(functional, linear=linear)

    wrapped_fn = _return_tensors(fn)

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with contextlib.ExitStack() as stack:
            _ensure_context(stack)
            _validate_distributed_args(args, kwargs, fn.__name__)
            if linear is not None:
                args = _auto_resolve_partials(args, linear=linear)
            return wrapped_fn(*args, **kwargs)

    return wrapped


def _auto_resolve_partials(
    args: tuple[Any, ...], *, linear: bool
) -> tuple[Any, ...]:
    """Resolves Partial placements on op args per the linear policy.

    Called by :func:`functional` before the wrapped op runs.
    Delegates to :func:`collectives._resolve_partials` which handles
    the ``auto_reduce_partial`` policy check internally.
    """
    from .collectives import _has_partial, _resolve_partials

    partials = [
        i
        for i, a in enumerate(args)
        if isinstance(a, tensor.Tensor) and a.is_distributed and _has_partial(a)
    ]
    if not partials:
        return args

    # Linear ops: passthrough when ALL distributed inputs are Partial.
    if linear and all(
        _has_partial(a)
        for a in args
        if isinstance(a, tensor.Tensor) and a.is_distributed
    ):
        return args

    resolved = list(args)
    for i in partials:
        resolved[i] = _resolve_partials(resolved[i])
    return tuple(resolved)


@contextlib.contextmanager
def lazy() -> Generator[None]:
    """Context manager for lazy tensor evaluation.

    Within this context, tensor operations are recorded but not executed.
    Tensors remain unrealized until explicitly awaited via ``await tensor.realize``
    or until their values are needed (e.g., by calling ``.item()``).

    Yields:
        None

    .. code-block:: python

        from max.experimental.distributed_functional import functional as F
        from max.experimental.tensor import Tensor

        with F.lazy():
            model = Linear(2, 3)

        print(model)  # Lazy weights not initialized
    """
    with rc.LazyRealizationContext() as ctx, tensor.realization_context(ctx):
        yield
