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

"""SPMD op dispatch — explicit per-op wiring of graph ops and sharding rules.

Each op is an explicit function that:

1. Calls the sharding rule (with ``tensor_to_layout()`` to convert Tensors to TensorLayouts).
2. Redistributes tensors to match the rule's suggestions.
3. Dispatches per-shard via ``per_shard_dispatch``.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from max.driver import CPU, Buffer
from max.experimental import tensor
from max.experimental.realization_context import ensure_context
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    PlacementMapping,
    Replicated,
    TensorLayout,
)
from max.experimental.sharding.per_shard_dim import global_dim
from max.experimental.tensor import Tensor
from max.graph import TensorValue, TensorValueLike, Type, ops
from max.graph.dim import Dim, DimLike, StaticDim
from max.graph.ops.slice_tensor import SliceIndices

from ..sharding import (
    ActionSet,
    PerShard,
    mode,
)
from ..sharding.mode import ShardingError
from ..sharding.rules import (
    argsort_rule,
    as_interleaved_complex_rule,
    band_part_rule,
    binary_rule,
    broadcast_to_rule,
    buffer_store_rule,
    buffer_store_slice_rule,
    chunk_rule,
    cond_rule,
    conv2d_rule,
    conv2d_transpose_rule,
    conv3d_rule,
    dequantize_rule,
    flatten_rule,
    fold_rule,
    gather_nd_rule,
    gather_rule,
    irfft_rule,
    layer_norm_rule,
    linear_binary_rule,
    linear_pool_rule,
    linear_reduce_rule,
    linear_unary_rule,
    masked_scatter_rule,
    matmul_rule,
    mean_rule,
    nonzero_rule,
    outer_rule,
    pad_rule,
    permute_rule,
    pool_rule,
    qmatmul_rule,
    rebind_rule,
    reduce_rule,
    repeat_interleave_rule,
    reshape_rule,
    resize_bicubic_rule,
    resize_linear_rule,
    resize_nearest_rule,
    resize_rule,
    rms_norm_rule,
    same_placement_multi_input_rule,
    scatter_add_rule,
    scatter_nd_add_rule,
    scatter_nd_rule,
    scatter_rule,
    slice_tensor_rule,
    softmax_rule,
    split_rule,
    squeeze_rule,
    stack_rule,
    ternary_rule,
    tile_rule,
    top_k_rule,
    transpose_rule,
    unary_rule,
    unsqueeze_rule,
    while_loop_rule,
)
from ._signatures import install_tensor_signature
from .creation_ops import full_like

# Re-exported; user-facing factory lives in ``max.experimental.sharding``.
__all__ = [
    "ShardingError",
    "any_distributed",
    "map_tensors",
    "mode",
    "to_tensors",
]


def to_tensors(values: Any) -> Any:
    """Converts graph op results to :class:`Tensor`, preserving container type.

    Recurses one level into ``list`` and ``tuple`` containers; unknown
    types pass through unchanged. Returns ``Tensor`` for ``Buffer`` and
    ``TensorValue`` leaves, and a same-shape container for list/tuple
    inputs (each leaf converted independently). ``Any`` reflects that
    leaves change type while the container type is preserved.
    """

    def _one(value: Any) -> Tensor | Any:
        if isinstance(value, Tensor):
            return value
        if isinstance(value, Buffer):
            return Tensor(storage=value)
        if isinstance(value, TensorValue):
            return Tensor.from_graph_value(value)
        return value

    if values is None:
        return None
    if isinstance(values, (Buffer, Tensor, TensorValue)):
        return _one(values)
    if isinstance(values, (list, tuple)):
        return type(values)(_one(v) for v in values)
    return values


def map_tensors(
    fn: Callable[[Tensor], Any], args: tuple[Any, ...]
) -> tuple[Any, ...]:
    """Applies ``fn`` to every :class:`Tensor` leaf in ``args``.

    Recurses into ``list`` and ``tuple`` containers; non-tensor leaves
    pass through unchanged.
    """

    def _walk(x: Any) -> Any:
        if isinstance(x, Tensor):
            return fn(x)
        if isinstance(x, list):
            return [_walk(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_walk(v) for v in x)
        return x

    return tuple(_walk(a) for a in args)


def tensor_to_layout(t: Tensor) -> TensorLayout:
    """Converts a :class:`Tensor` to a :class:`TensorLayout` for sharding-rule evaluation.

    ``t.shape`` already carries per-device cells on :class:`Sharded` axes
    (via :class:`PerShardDim`), so the rules that fold per-rank cells
    (notably ``reshape_rule``) can do the correct shape arithmetic
    directly. Non-distributed tensors fall back to a plain :class:`Shape`.
    """
    if t.is_distributed:
        return TensorLayout(
            t.dtype,
            t.shape,
            PlacementMapping(t.mesh, t.placements),
        )
    return TensorLayout(
        t.dtype,
        t.shape,
        PlacementMapping(DeviceMesh.single(t.device), (Replicated(),)),
    )


def any_distributed(args: tuple[object, ...]) -> bool:
    """True if any :class:`Tensor` in ``args`` is distributed (multi-device)."""
    for a in args:
        if isinstance(a, Tensor) and a.is_distributed:
            return True
        if isinstance(a, (list, tuple)):
            for item in a:
                if isinstance(item, Tensor) and item.is_distributed:
                    return True
    return False


def per_shard_dispatch(
    graph_op: Callable[..., Any],
    args: tuple[Any, ...],
    output_mappings: tuple[DeviceMapping, ...],
) -> Any:
    """Runs ``graph_op`` once per shard and reassembles distributed outputs.

    Args:
        graph_op: The per-rank graph op to run.
        args: Already-redistributed args.
        output_mappings: One :class:`DeviceMapping` per output.
    """
    mesh = output_mappings[0].mesh

    with ensure_context():
        per_shard = _run_per_shard(graph_op, args, mesh.num_devices)
        first = per_shard[0]
        if first is None:
            return None

        multi = isinstance(first, (list, tuple))
        num_out = len(first) if multi else 1
        outputs = [
            _reassemble_output(
                per_shard,
                j,
                output_mappings[builtins.min(j, len(output_mappings) - 1)],
                multi=multi,
            )
            for j in builtins.range(num_out)
        ]
        return type(first)(outputs) if multi else outputs[0]


def _run_per_shard(
    graph_op: Callable[..., Any],
    args: tuple[Any, ...],
    num_devices: int,
) -> list[Any]:
    """Calls ``graph_op`` once per shard with per-rank arg unwrapping."""
    per_shard: list[Any] = []
    for i in builtins.range(num_devices):

        def _per_rank(t: Tensor, _i: int = i) -> TensorValue:
            return (
                TensorValue(t.local_shards[_i])
                if t.is_distributed
                else TensorValue(t)
            )

        shard_args = map_tensors(_per_rank, args)
        shard_args = tuple(
            a[i] if isinstance(a, PerShard) else a for a in shard_args
        )
        per_shard.append(graph_op(*shard_args))
    return per_shard


def _reassemble_output(
    per_shard: Sequence[Any],
    j: int,
    out_mapping: DeviceMapping,
    *,
    multi: bool,
) -> Tensor:
    """Reassembles output ``j`` from per-shard results into one distributed Tensor."""
    tvs = [TensorValue(s[j] if multi else s) for s in per_shard]
    return Tensor.from_shard_values(tvs, out_mapping)


def functional(
    graph_op: Callable[..., Any],
    rule: Callable[..., ActionSet] | None = None,
) -> Callable[..., Any]:
    """Wraps a graph op as a distributed dispatch entry.

    Returns a callable that local-auto-shards when any argument is a
    distributed :class:`Tensor` (and a rule is bound), and otherwise
    forwards to the bare ``graph_op``. The returned wrapper carries
    ``graph_op`` and ``rule`` as attributes; reassign ``wrapper.rule``
    to swap the sharding rule at runtime without re-wrapping.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        active_rule = getattr(wrapper, "rule", None)
        if any_distributed(args) and active_rule is not None:
            return _local_dispatch(graph_op, active_rule, args, kwargs)
        with ensure_context():
            return to_tensors(graph_op(*args, **kwargs))

    # ``Any``-typed alias so attribute writes are dynamic;
    # ``functools.wraps`` types the closure as ``_Wrapped[...]`` which
    # rejects arbitrary attribute assignment under mypy.
    w: Any = wrapper
    w.__name__ = getattr(graph_op, "__name__", "wrapper")
    w.__qualname__ = getattr(graph_op, "__qualname__", w.__name__)
    w.__module__ = getattr(graph_op, "__module__", w.__module__)
    w.__wrapped__ = graph_op
    w.graph_op = graph_op
    w.rule = rule
    # Rewrite the wrapper's signature/annotations so inspect.signature and
    # Sphinx show ``Tensor`` instead of the graph op's ``TensorValueLike``
    # parameter types (reads graph_op via ``__wrapped__``). From #87216.
    install_tensor_signature(wrapper)
    return wrapper


def _local_dispatch(
    graph_op: Callable[..., Any],
    rule: Callable[..., ActionSet],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    """Picks one :class:`Action` for this call and applies it."""
    from max.experimental.sharding._diagnostics import report_reshard
    from max.experimental.sharding.mode import current_solver

    flat_args = _canonicalize_call(graph_op, args, kwargs)
    layout_args = map_tensors(tensor_to_layout, flat_args)
    in_layouts = _walk_tensor_layouts(layout_args)

    menu = rule(*layout_args)
    solver = current_solver()
    action = solver(menu, in_layouts)

    op_name = getattr(graph_op, "__name__", "<op>")
    report_reshard(solver, op_name, layout_args, menu, action)
    redistributed = _transfer_args(flat_args, action.inputs)

    if action.outputs:
        out_mappings = action.outputs
    else:
        out_mappings = (
            next(
                t.mapping for t in _walk_tensors(flat_args) if t.is_distributed
            ),
        )
    return per_shard_dispatch(
        graph_op,
        redistributed,
        out_mappings,
    )


def _canonicalize_call(
    graph_op: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Normalizes ``args`` + ``kwargs`` into a positional tuple.

    Binds against ``graph_op``'s signature so kwargs become positional.
    Falls back to ``args`` when the signature is uninspectable.
    """
    import inspect

    sig_source = getattr(graph_op, "graph_op", graph_op)
    try:
        bound = inspect.signature(sig_source).bind(*args, **kwargs)
        bound.apply_defaults()
        return tuple(bound.args)
    except (TypeError, NotImplementedError, ValueError):
        return args + tuple(kwargs.values())


def _walk_tensors(value: Any) -> Iterable[Tensor]:
    """Yields every :class:`Tensor` reachable through tuples/lists."""
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk_tensors(v)


def _walk_tensor_layouts(value: Any) -> list[Any]:
    """Flattens TensorLayout leaves out of arbitrary nested args."""
    out: list[Any] = []
    if isinstance(value, (list, tuple)):
        for v in value:
            out.extend(_walk_tensor_layouts(v))
        return out
    if hasattr(value, "mapping") and hasattr(value, "shape"):
        out.append(value)
    return out


def _transfer_args(
    args: tuple[Any, ...],
    suggested: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Reshards Tensor args to match the action's per-slot mappings."""
    from .collective_ops import transfer_to

    result: list[object] = []
    for orig, sugg in zip(args, suggested, strict=False):
        if isinstance(sugg, PerShard):
            result.append(sugg)
        elif isinstance(orig, Tensor) and isinstance(sugg, DeviceMapping):
            result.append(transfer_to(orig, sugg))
        elif isinstance(orig, (list, tuple)) and isinstance(
            sugg, (list, tuple)
        ):
            items = [
                transfer_to(o, s)
                if isinstance(o, Tensor) and isinstance(s, DeviceMapping)
                else s
                for o, s in zip(orig, sugg, strict=False)
            ]
            result.append(type(orig)(items))
        elif not isinstance(orig, Tensor) and sugg is not None:
            result.append(sugg)
        else:
            result.append(orig)
    if len(args) > len(suggested):
        result.extend(args[len(suggested) :])
    return tuple(result)


def _binary_with_scalar_promotion(
    inner: Callable[..., object],
) -> Callable[..., Tensor]:
    """Wraps a binary dispatch with scalar promotion.

    Scalar promotion is gated on ``any_distributed`` because the
    single-device graph-op path handles scalar + tensor natively. Rank
    differences are not equalized here: broadcasting is handled by the
    RMO dialect per shard, and the placement rules express trailing-axis
    alignment directly.
    """

    def wrapper(lhs: Tensor | int | float, rhs: Tensor | int | float) -> Tensor:
        if any_distributed((lhs, rhs)):
            if isinstance(lhs, (int, float)) and isinstance(rhs, Tensor):
                lhs = full_like(rhs, lhs)
            elif isinstance(rhs, (int, float)) and isinstance(lhs, Tensor):
                rhs = full_like(lhs, rhs)
        result = inner(lhs, rhs)
        assert isinstance(result, Tensor)
        return result

    wrapper.__module__ = getattr(inner, "__module__", wrapper.__module__)
    wrapper.__name__ = getattr(inner, "__name__", wrapper.__name__)
    wrapper.__qualname__ = getattr(inner, "__qualname__", wrapper.__qualname__)
    return wrapper


#: Adds two tensors element-wise with SPMD distribution support.
#: Scalars are promoted to tensors automatically.
#: See :func:`max.graph.ops.add` for details.
add = _binary_with_scalar_promotion(
    functional(ops.add, rule=linear_binary_rule)
)
add.__doc__ = """Adds two tensors element-wise.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([10.0, 20.0, 30.0])
    result = F.add(a, b)
    # result is [11.0, 22.0, 33.0]

    # Scalar is auto-promoted to a tensor.
    result = F.add(a, 0.5)
    # result is [1.5, 2.5, 3.5]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A tensor with the broadcast shape containing the element-wise sums.
"""

sub = _binary_with_scalar_promotion(
    functional(ops.sub, rule=linear_binary_rule)
)
sub.__doc__ = """Subtracts two tensors element-wise.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([10.0, 20.0, 30.0])
    b = Tensor([1.0, 2.0, 3.0])
    result = F.sub(a, b)
    # result is [9.0, 18.0, 27.0]

Args:
    lhs: The minuend (left-hand side) tensor or scalar.
    rhs: The subtrahend (right-hand side) tensor or scalar.

Returns:
    A tensor with the broadcast shape containing ``lhs - rhs`` element-wise.
"""

mul = _binary_with_scalar_promotion(functional(ops.mul, rule=binary_rule))
mul.__doc__ = """Multiplies two tensors element-wise.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    result = F.mul(a, b)
    # result is [4.0, 10.0, 18.0]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A tensor with the broadcast shape containing element-wise products.
"""

div = _binary_with_scalar_promotion(functional(ops.div, rule=binary_rule))
div.__doc__ = """Divides two tensors element-wise.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor. Integer
operands are promoted to floating point.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([10.0, 6.0, 3.0])
    b = Tensor([2.0, 3.0, 4.0])
    result = F.div(a, b)
    # result is [5.0, 2.0, 0.75]

Args:
    lhs: The numerator tensor or scalar.
    rhs: The denominator tensor or scalar.

Returns:
    A tensor with the broadcast shape containing ``lhs / rhs`` element-wise.
"""

pow = _binary_with_scalar_promotion(functional(ops.pow, rule=binary_rule))
pow.__doc__ = """Raises elements of one tensor to the power of another element-wise.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([2.0, 3.0, 4.0])
    b = Tensor([3.0, 2.0, 0.5])
    result = F.pow(a, b)
    # result is [8.0, 9.0, 2.0]

Args:
    lhs: The base tensor or scalar.
    rhs: The exponent tensor or scalar.

Returns:
    A tensor with the broadcast shape containing ``lhs ** rhs`` element-wise.
"""

mod = _binary_with_scalar_promotion(functional(ops.mod, rule=binary_rule))
mod.__doc__ = """Computes the element-wise modulus of two tensors.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([7.0, 10.0, 15.0])
    b = Tensor([3.0, 4.0, 6.0])
    result = F.mod(a, b)
    # result is [1.0, 2.0, 3.0]

Args:
    lhs: The dividend tensor or scalar.
    rhs: The divisor tensor or scalar.

Returns:
    A tensor with the broadcast shape containing ``lhs % rhs`` element-wise.
"""

#: Negates a tensor element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.negate` for details.
negate = functional(ops.negate, rule=linear_unary_rule)
negate.__doc__ = """Negates a tensor element-wise.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([-1.0, 0.0, 2.0])
    result = F.negate(x)
    # result is [1.0, 0.0, -2.0]

Args:
    x: The input tensor.

Returns:
    A tensor of the same shape and dtype with each element negated.
"""

relu = functional(ops.relu, rule=unary_rule)
relu.__doc__ = """Applies the ReLU activation function element-wise.

Computes ``max(0, x)``: negative values are set to zero while positive
values are unchanged.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]])
    result = F.relu(x)
    # result is [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]

Args:
    x: The input tensor.

Returns:
    A tensor of the same shape and dtype with negative values replaced by ``0``.
"""

abs = functional(ops.abs, rule=unary_rule)
abs.__doc__ = """Computes the absolute value of a tensor element-wise.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = F.abs(x)
    # result is [2.0, 1.0, 0.0, 1.0, 2.0]

Args:
    x: The input tensor.

Returns:
    A tensor of the same shape and dtype with each element replaced by
    its absolute value.
"""

exp = functional(ops.exp, rule=unary_rule)
exp.__doc__ = """Computes the exponential of a tensor element-wise.

Computes ``e ** x`` for each element, where ``e`` is Euler's number.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([0.0, 1.0, 2.0])
    result = F.exp(x)
    # result is approximately [1.0, 2.718, 7.389]

Args:
    x: The input tensor.

Returns:
    A tensor of the same shape and dtype with the exponential applied
    element-wise.
"""

log = functional(ops.log, rule=unary_rule)
log.__doc__ = """Computes the natural logarithm of a tensor element-wise.

``log(x)`` is undefined for ``x <= 0`` on real numbers.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.0, 2.718, 7.389, 20.0])
    result = F.log(x)
    # result is approximately [0.0, 1.0, 2.0, 2.996]

Args:
    x: The input tensor. Must contain positive values for real-valued results.

Returns:
    A tensor of the same shape and dtype with the natural logarithm applied
    element-wise.
"""

sqrt = functional(ops.sqrt, rule=unary_rule)
sqrt.__doc__ = """Computes the square root of a tensor element-wise.

Requires non-negative inputs for real-valued results.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.0, 4.0, 9.0, 16.0])
    result = F.sqrt(x)
    # result is [1.0, 2.0, 3.0, 4.0]

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A tensor of the same shape and dtype with the square root of each
    element.
"""

rsqrt = functional(ops.rsqrt, rule=unary_rule)
rsqrt.__doc__ = """Computes the reciprocal square root of a tensor element-wise.

Computes ``1 / sqrt(x)`` for each element.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.0, 4.0, 16.0])
    result = F.rsqrt(x)
    # result is [1.0, 0.5, 0.25]

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A tensor of the same shape and dtype with the reciprocal square root
    of each element.
"""

sigmoid = functional(ops.sigmoid, rule=unary_rule)
sigmoid.__doc__ = """Applies the sigmoid activation function element-wise.

Computes ``1 / (1 + exp(-x))`` for each element, mapping all values to
the range ``(0, 1)``.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]])
    result = F.sigmoid(x)
    # result is approximately:
    # [[0.119, 0.269, 0.5], [0.731, 0.881, 0.953]]

Args:
    x: The input tensor.

Returns:
    A tensor of the same shape and dtype with values in the range ``(0, 1)``.
"""

silu = functional(ops.silu, rule=unary_rule)
silu.__doc__ = """Applies the SiLU (Swish) activation function element-wise.

Computes ``x * sigmoid(x)`` for each element.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([-1.0, 0.0, 1.0, 2.0])
    result = F.silu(x)
    # result is approximately [-0.269, 0.0, 0.731, 1.762]

Args:
    x: The input tensor.

Returns:
    A tensor of the same shape and dtype with the SiLU activation applied
    element-wise.
"""

gelu = functional(ops.gelu, rule=unary_rule)
gelu.__doc__ = """Applies the GELU (Gaussian Error Linear Unit) activation element-wise.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([-1.0, 0.0, 1.0])
    result = F.gelu(x)
    # result is approximately [-0.159, 0.0, 0.841]

Args:
    x: The input tensor.
    approximate: The approximation method. Defaults to ``"none"`` (exact
        form using ``erf``). Use ``"tanh"`` for the tanh-based approximation
        or ``"quick"`` for the sigmoid-based approximation.

Returns:
    A tensor of the same shape and dtype with the GELU activation applied
    element-wise.
"""

tanh = functional(ops.tanh, rule=unary_rule)
tanh.__doc__ = """Computes the hyperbolic tangent of a tensor element-wise.

Maps all values to the range ``(-1, 1)``.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]])
    result = F.tanh(x)
    # result is approximately:
    # [[-0.964, -0.762, 0.0], [0.762, 0.964, 0.995]]

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A tensor of the same shape and dtype with values in the range ``(-1, 1)``.
"""

cos = functional(ops.cos, rule=unary_rule)
cos.__doc__ = """Computes the cosine of a tensor element-wise.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([0.0, 0.5, 1.0])
    result = F.cos(x)
    # result is approximately [1.0, 0.878, 0.540]

Args:
    x: The input tensor, interpreted as radians. Must have a floating-point
        dtype.

Returns:
    A tensor of the same shape and dtype with the cosine of each element.
"""

sin = functional(ops.sin, rule=unary_rule)
sin.__doc__ = """Computes the sine of a tensor element-wise.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([0.0, 0.5, 1.0])
    result = F.sin(x)
    # result is approximately [0.0, 0.479, 0.841]

Args:
    x: The input tensor, interpreted as radians. Must have a floating-point
        dtype.

Returns:
    A tensor of the same shape and dtype with the sine of each element.
"""

erf = functional(ops.erf, rule=unary_rule)
erf.__doc__ = """Computes the error function of a tensor element-wise.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([-1.0, 0.0, 1.0])
    result = F.erf(x)
    # result is approximately [-0.843, 0.0, 0.843]

Args:
    x: The input tensor.

Returns:
    A tensor of the same shape and dtype with the error function applied
    element-wise.
"""

ceil = functional(ops.ceil, rule=unary_rule)
ceil.__doc__ = """Computes the ceil of a tensor element-wise.

Rounds each element up toward positive infinity.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.5, 2.0, -1.5, -2.7])
    result = F.ceil(x)
    # result is [2.0, 2.0, -1.0, -2.0]

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A tensor of the same shape and dtype with each element rounded up.
"""

floor = functional(ops.floor, rule=unary_rule)
floor.__doc__ = """Computes the floor of a tensor element-wise.

Rounds each element down toward negative infinity.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.5, 2.0, -1.5, -2.7])
    result = F.floor(x)
    # result is [1.0, 2.0, -2.0, -3.0]

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A tensor of the same shape and dtype with each element rounded down.
"""

round = functional(ops.round, rule=unary_rule)
round.__doc__ = """Rounds a tensor to the nearest integer element-wise.

Ties round toward the nearest even number (banker's rounding).

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([0.5, 1.5, 2.5, -0.5])
    result = F.round(x)
    # Ties round to the nearest even integer:
    # result is [0.0, 2.0, 2.0, 0.0]

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A tensor of the same shape and dtype with each element rounded.
"""

trunc = functional(ops.trunc, rule=unary_rule)
trunc.__doc__ = """Truncates a tensor toward zero element-wise.

Discards the fractional part of each element.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.5, 2.7, -1.5, -2.7])
    result = F.trunc(x)
    # result is [1.0, 2.0, -1.0, -2.0]

Args:
    x: The input tensor. Must have a floating-point dtype.

Returns:
    A tensor of the same shape and dtype with the fractional part discarded.
"""


is_inf = functional(ops.is_inf, rule=unary_rule)
is_inf.__doc__ = """Tests element-wise whether a tensor contains infinite values.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.0, float("inf"), float("-inf"), float("nan")])
    result = F.is_inf(x)
    # result is [False, True, True, False]

Args:
    x: The input tensor.

Returns:
    A boolean tensor of the same shape, with ``True`` where the input is
    positive or negative infinity.
"""

is_nan = functional(ops.is_nan, rule=unary_rule)
is_nan.__doc__ = """Tests element-wise whether a tensor contains NaN values.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.0, float("inf"), float("nan"), 0.0])
    result = F.is_nan(x)
    # result is [False, False, True, False]

Args:
    x: The input tensor.

Returns:
    A boolean tensor of the same shape, with ``True`` where the input is NaN.
"""

logical_not = functional(ops.logical_not, rule=unary_rule)
logical_not.__doc__ = """Computes the element-wise logical NOT of a boolean tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([True, False, True])
    result = F.logical_not(x)
    # result is [False, True, False]

Args:
    x: The input boolean tensor.

Returns:
    A boolean tensor of the same shape with each element negated.
"""

log1p = functional(ops.log1p, rule=unary_rule)
log1p.__doc__ = """Computes ``log(1 + x)`` element-wise.

More numerically accurate than ``log(1 + x)`` when ``x`` is close to zero.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([0.0, 1e-7, 1.0])
    result = F.log1p(x)
    # result is approximately [0.0, 1e-7, 0.693]

Args:
    x: The input tensor.

Returns:
    A tensor of the same shape and dtype with ``log(1 + x)`` applied
    element-wise.
"""

atanh = functional(ops.atanh, rule=unary_rule)
atanh.__doc__ = """Computes the inverse hyperbolic tangent of a tensor element-wise.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([-0.5, 0.0, 0.5])
    result = F.atanh(x)
    # result is approximately [-0.549, 0.0, 0.549]

Args:
    x: The input tensor, with values in the range ``(-1, 1)``. Must have a
        floating-point dtype.

Returns:
    A tensor of the same shape and dtype with the inverse hyperbolic tangent
    of each element.
"""

acos = functional(ops.acos, rule=unary_rule)
#: Dequantizes a tensor. Distributed via SPMD.
#: See :func:`max.graph.ops.dequantize` for details.
dequantize = functional(ops.dequantize, rule=dequantize_rule)

equal = _binary_with_scalar_promotion(functional(ops.equal, rule=binary_rule))
equal.__doc__ = """Tests element-wise equality between two tensors.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([1.0, 5.0, 3.0])
    result = F.equal(a, b)
    # result is [True, False, True]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A boolean tensor that is ``True`` when
    ``lhs == rhs``.
"""

not_equal = _binary_with_scalar_promotion(
    functional(ops.not_equal, rule=binary_rule)
)
not_equal.__doc__ = """Tests element-wise inequality between two tensors.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([1.0, 5.0, 3.0])
    result = F.not_equal(a, b)
    # result is [False, True, False]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A boolean tensor that is ``True`` when
    ``lhs != rhs``.
"""

greater = _binary_with_scalar_promotion(
    functional(ops.greater, rule=binary_rule)
)
greater.__doc__ = """Tests element-wise whether one tensor is greater than another.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([1.0, 5.0, 3.0])
    b = Tensor([2.0, 3.0, 3.0])
    result = F.greater(a, b)
    # result is [False, True, False]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A boolean tensor that is ``True`` when
    ``lhs > rhs``.
"""

greater_equal = _binary_with_scalar_promotion(
    functional(ops.greater_equal, rule=binary_rule)
)
greater_equal.__doc__ = """Tests element-wise whether one tensor is greater than or equal to another.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([1.0, 5.0, 3.0])
    b = Tensor([2.0, 3.0, 3.0])
    result = F.greater_equal(a, b)
    # result is [False, True, True]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A boolean tensor that is ``True`` when
    ``lhs >= rhs``.
"""

logical_and = _binary_with_scalar_promotion(
    functional(ops.logical_and, rule=binary_rule)
)
logical_and.__doc__ = """Computes the element-wise logical AND of two boolean tensors.

Only supports boolean inputs.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([True, True, False])
    b = Tensor([True, False, False])
    result = F.logical_and(a, b)
    # result is [True, False, False]

Args:
    lhs: The left-hand side boolean tensor.
    rhs: The right-hand side boolean tensor.

Returns:
    A boolean tensor that is ``True`` when both
    inputs are ``True``.
"""

logical_or = _binary_with_scalar_promotion(
    functional(ops.logical_or, rule=binary_rule)
)
logical_or.__doc__ = """Computes the element-wise logical OR of two boolean tensors.

Only supports boolean inputs.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([True, True, False])
    b = Tensor([True, False, False])
    result = F.logical_or(a, b)
    # result is [True, True, False]

Args:
    lhs: The left-hand side boolean tensor.
    rhs: The right-hand side boolean tensor.

Returns:
    A boolean tensor that is ``True`` when at
    least one input is ``True``.
"""

logical_xor = _binary_with_scalar_promotion(
    functional(ops.logical_xor, rule=binary_rule)
)
logical_xor.__doc__ = """Computes the element-wise logical XOR of two boolean tensors.

Only supports boolean inputs.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([True, True, False])
    b = Tensor([True, False, False])
    result = F.logical_xor(a, b)
    # result is [False, True, False]

Args:
    lhs: The left-hand side boolean tensor.
    rhs: The right-hand side boolean tensor.

Returns:
    A boolean tensor that is ``True`` when exactly
    one input is ``True``.
"""

#: SPMD-distributed wrapper around :func:`max.graph.ops.where`.
_where_inner = functional(ops.where, rule=ternary_rule)


def where(
    cond: Tensor,
    x: Tensor | int | float,
    y: Tensor | int | float,
) -> Tensor:
    """Selects elements from two tensors based on a boolean condition.

    For each position, returns the corresponding element from ``x`` where
    ``cond`` is ``True`` and from ``y`` otherwise. Scalar ``x``/``y`` operands
    are promoted to tensors and all inputs are broadcast to a common shape.

    Args:
        cond: A boolean tensor controlling the selection.
        x: The tensor (or scalar) providing values where ``cond`` is ``True``.
        y: The tensor (or scalar) providing values where ``cond`` is ``False``.

    Returns:
        A tensor with the broadcast shape, with elements selected from ``x``
        or ``y`` according to ``cond``.
    """
    if isinstance(x, (int, float)) and isinstance(y, Tensor):
        x = full_like(y, x)
    elif isinstance(x, (int, float)):
        x = full_like(cond, x)
    if isinstance(y, (int, float)) and isinstance(x, Tensor):
        y = full_like(x, y)
    elif isinstance(y, (int, float)):
        y = full_like(cond, y)
    result = _where_inner(cond, x, y)
    assert isinstance(result, Tensor)
    return result


elementwise_min = _binary_with_scalar_promotion(
    functional(ops.elementwise.min, rule=binary_rule)
)
elementwise_min.__doc__ = """Computes the element-wise minimum of two tensors.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([3.0, 1.0, 4.0])
    b = Tensor([1.0, 5.0, 9.0])
    result = F.elementwise_min(a, b)
    # result is [1.0, 1.0, 4.0]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A tensor with the broadcast shape containing the smaller value at
    each position.
"""

elementwise_max = _binary_with_scalar_promotion(
    functional(ops.elementwise.max, rule=binary_rule)
)
elementwise_max.__doc__ = """Computes the element-wise maximum of two tensors.

Either operand may be a Python ``int`` or ``float`` scalar, which is
automatically promoted to a tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([3.0, 1.0, 4.0])
    b = Tensor([1.0, 5.0, 9.0])
    result = F.elementwise_max(a, b)
    # result is [3.0, 5.0, 9.0]

Args:
    lhs: The left-hand side tensor or scalar.
    rhs: The right-hand side tensor or scalar.

Returns:
    A tensor with the broadcast shape containing the larger value at each
    position.
"""

#: Casts a tensor to a different data type. Distributed via SPMD.
#: See :func:`max.graph.ops.cast` for details.
cast = functional(ops.cast, rule=unary_rule)
cast.__doc__ = """Casts a tensor to a different data type.

Values may change when the source and target types can't represent each
other exactly. Float-to-integer casts truncate toward zero; float-to-float
casts with lower precision round to the nearest representable value.

.. code-block:: python

    from max.dtype import DType
    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([1.7, -1.7, 2.5])  # float32 on CPU by default
    result = F.cast(x, DType.int32)
    # result has dtype int32 and values [1, -1, 2]

Args:
    x: The input tensor.
    dtype: The target data type.

Returns:
    A tensor with the same shape but the new dtype.
"""


#: Performs matrix multiplication. Distributed via SPMD.
#: See :func:`max.graph.ops.matmul` for details.
matmul = functional(ops.matmul, rule=matmul_rule)
matmul.__doc__ = """Performs matrix multiplication between two tensors.

Treats the innermost two dimensions of each input as a matrix: ``lhs``
of shape ``(..., M, K)`` and ``rhs`` of shape ``(..., K, N)`` produce
an output of shape ``(..., M, N)``. The ``K`` dimensions must match.
Any outer batch dimensions are broadcast.

When inputs are distributed across devices, the operation is sharded
according to the matmul sharding rule.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    result = F.matmul(a, b)
    # result has shape (2, 2):
    # [[19.0, 22.0], [43.0, 50.0]]

    # The ``@`` operator on Tensor also calls matmul.
    result = a @ b

Args:
    lhs: The left-hand side input tensor.
    rhs: The right-hand side input tensor.

Returns:
    A tensor representing the matrix product of ``lhs`` and ``rhs``.
"""

#: Applies layer normalization. Distributed via SPMD.
#: See :func:`max.graph.ops.layer_norm` for details.
layer_norm = functional(ops.layer_norm, rule=layer_norm_rule)
#: Performs quantized matrix multiplication. Distributed via SPMD.
#: See :func:`max.graph.ops.qmatmul` for details.
qmatmul = functional(ops.qmatmul, rule=qmatmul_rule)

avg_pool2d = functional(ops.avg_pool2d, rule=linear_pool_rule)
avg_pool2d.__doc__ = """Applies 2D average pooling to a tensor.

Slides a window of size ``kernel_size`` over the spatial dimensions and
replaces each window with the average of its values.

Args:
    input: The input tensor with shape ``(N, H, W, C)``.
    kernel_size: A tuple ``(kernel_h, kernel_w)`` giving the height and
        width of the sliding window.
    stride: The stride of the sliding window. Either a single ``int``
        applied to both spatial dimensions, or a tuple
        ``(stride_h, stride_w)``. Defaults to ``1``.
    dilation: The spacing between kernel elements. Either a single
        ``int`` applied to both spatial dimensions, or a tuple
        ``(dilation_h, dilation_w)``. Defaults to ``1``.
    padding: Zero-padding added to both sides of each spatial dimension.
        Either a single ``int`` applied to both spatial dimensions, or a
        tuple ``(pad_h, pad_w)``. Defaults to ``0``.
    ceil_mode: When ``True``, uses ceil instead of floor when computing
        the output spatial shape. Defaults to ``False``.
    count_boundary: When ``True``, includes padding elements in the
        divisor when computing each window's average. Defaults to
        ``True``.

Returns:
    A tensor with shape ``(N, H_out, W_out, C)`` containing the
    average-pooled values.
"""

max_pool2d = functional(ops.max_pool2d, rule=pool_rule)
max_pool2d.__doc__ = """Applies 2D max pooling to a tensor.

Slides a window of size ``kernel_size`` over the spatial dimensions and
replaces each window with its maximum value.

Args:
    input: The input tensor with shape ``(N, H, W, C)``.
    kernel_size: A tuple ``(kernel_h, kernel_w)`` giving the height and
        width of the sliding window.
    stride: The stride of the sliding window. Either a single ``int``
        applied to both spatial dimensions, or a tuple
        ``(stride_h, stride_w)``. Defaults to ``1``.
    dilation: The spacing between kernel elements. Either a single
        ``int`` applied to both spatial dimensions, or a tuple
        ``(dilation_h, dilation_w)``. Defaults to ``1``.
    padding: Zero-padding added to both sides of each spatial dimension.
        Either a single ``int`` applied to both spatial dimensions, or a
        tuple ``(pad_h, pad_w)``. Defaults to ``0``.
    ceil_mode: When ``True``, uses ceil instead of floor when computing
        the output spatial shape. Defaults to ``False``.

Returns:
    A tensor with shape ``(N, H_out, W_out, C)`` containing the
    max-pooled values.
"""

#: Permutes the dimensions of a tensor. Distributed via SPMD.
#: See :func:`max.graph.ops.permute` for details.
permute = functional(ops.permute, rule=permute_rule)
permute.__doc__ = """Permutes the dimensions of a tensor.

Args:
    x: The input tensor.
    dims: A list of dimension indices specifying the new ordering.

Returns:
    A tensor with its dimensions reordered according to ``dims``.
"""

transpose = functional(ops.transpose, rule=transpose_rule)
transpose.__doc__ = """Swaps two dimensions of a tensor.

Args:
    x: The input tensor.
    axis_1: The first axis to swap.
    axis_2: The second axis to swap.

Returns:
    A tensor with ``axis_1`` and ``axis_2`` swapped.
"""

unsqueeze = functional(ops.unsqueeze, rule=unsqueeze_rule)
unsqueeze.__doc__ = """Inserts a size-1 dimension into a tensor.

Args:
    x: The input tensor.
    axis: The position at which to insert the new size-1 dimension.
        Negative values count from the end.

Returns:
    A tensor of rank ``x.rank + 1`` with a size-1 dimension inserted at
    ``axis``.
"""

squeeze = functional(ops.squeeze, rule=squeeze_rule)
#: SPMD-distributed wrapper around :func:`max.graph.ops.reshape`.
reshape = functional(ops.reshape, rule=reshape_rule)
reshape.__doc__ = """Reshapes a tensor to a new shape.

Returns a tensor with the same data but a different shape; the total
number of elements must stay the same.

Args:
    x: The input tensor.
    shape: The desired output shape. The element count must match the
        input tensor.

Returns:
    A tensor with the requested shape.
"""
#: Flattens a tensor. Distributed via SPMD.
#: See :func:`max.graph.ops.flatten` for details.
flatten = functional(ops.flatten, rule=flatten_rule)
flatten.__doc__ = """Flattens a contiguous range of dimensions into one.

All dimensions from ``start_dim`` to ``end_dim`` (inclusive) are merged
into a single output dimension. The number and order of elements is
unchanged.

Args:
    x: The input tensor.
    start_dim: The first dimension to flatten. Negative values count
        from the end. Defaults to ``0``.
    end_dim: The last dimension to flatten (inclusive). Negative values
        count from the end. Defaults to ``-1``.

Returns:
    A tensor with the specified dimension range merged into a single
    dimension.
"""

tile = functional(ops.tile, rule=tile_rule)
tile.__doc__ = """Repeats a tensor along each dimension.

Args:
    x: The input tensor.
    repeats: An iterable of repeat counts, one per dimension of ``x``.
        All values must be positive and the length must equal the rank
        of ``x``.

Returns:
    A tensor whose ``i``-th dimension size equals
    ``x.shape[i] * repeats[i]``.
"""

pad = functional(ops.pad, rule=pad_rule)
#: SPMD-distributed wrapper around :func:`max.graph.ops.broadcast_to`.
broadcast_to = functional(ops.broadcast_to, rule=broadcast_to_rule)
broadcast_to.__doc__ = """Broadcasts a tensor to a target shape.

Follows NumPy broadcasting semantics: dimensions of size 1 in the input
expand to match larger dimensions in the target shape.

Args:
    x: The input tensor.
    shape: The target shape. Each dimension must match the input dimension
        or be broadcastable from size 1.

Returns:
    A tensor broadcast to the target shape.
"""
#: Repeats elements of a tensor. Distributed via SPMD.
#: See :func:`max.graph.ops.repeat_interleave` for details.
repeat_interleave = functional(
    ops.repeat_interleave, rule=repeat_interleave_rule
)
repeat_interleave.__doc__ = """Repeats elements of a tensor along a dimension.

Unlike :func:`tile`, which repeats whole blocks, this repeats each
element ``repeats`` times consecutively.

Args:
    x: The input tensor.
    repeats: The number of repetitions for each element. May be a single
        ``int`` (the same count applied to every element) or a 1-D
        :class:`~max.graph.TensorValue` giving a per-element count.
    axis: The dimension along which to repeat. When ``None`` (the
        default), the input is flattened to 1-D before repetition.
    out_dim: The output dimension size along ``axis``. Required when
        ``repeats`` is a :class:`~max.graph.TensorValue`, since the
        output size depends on values that aren't known at graph build
        time.

Returns:
    A tensor with elements repeated along ``axis``.
"""

slice_tensor = functional(ops.slice_tensor, rule=slice_tensor_rule)
slice_tensor.__doc__ = """Slices a subtensor view from a tensor using NumPy-style indexing.

Supports the usual NumPy index forms — integers, ``slice`` objects, an
``Ellipsis`` (``...``), and ``None`` (insert a new size-1 axis).

Args:
    x: The input tensor.
    indices: A sequence of slice specifications, one per dimension. May
        also use ``Ellipsis`` for omitted dimensions or ``None`` to
        insert a new axis.

Returns:
    A tensor view containing the selected slice.
"""

concat = functional(ops.concat, rule=same_placement_multi_input_rule)
concat.__doc__ = """Concatenates a sequence of tensors along an axis.

All input tensors must have the same dtype, the same rank, the same
device, and the same size in every dimension except ``axis``. The
sequence must contain at least one tensor.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])

    vertical = F.concat([a, b], axis=0)
    # vertical has shape (4, 2):
    # [[1, 2], [3, 4], [5, 6], [7, 8]]

    horizontal = F.concat([a, b], axis=1)
    # horizontal has shape (2, 4):
    # [[1, 2, 5, 6], [3, 4, 7, 8]]

Args:
    original_vals: The non-empty sequence of tensors to concatenate.
    axis: The dimension along which to concatenate. Negative values
        index relative to the end of the tensor shape. Defaults to ``0``.

Returns:
    A tensor with the same rank, dtype, and device as the inputs, whose
    size along ``axis`` is the sum of the inputs' sizes along that axis.

Raises:
    ValueError: If ``original_vals`` is empty, the inputs differ in rank,
        or the inputs differ in size along a non-``axis`` dimension.
    IndexError: If ``axis`` is out of range for the input rank.
"""

stack = functional(ops.stack, rule=stack_rule)
stack.__doc__ = """Stacks a sequence of tensors along a new dimension.

All input tensors must have the same shape.

Args:
    values: The sequence of tensors to stack.
    axis: The position at which to insert the new dimension. Defaults to
        ``0``.

Returns:
    A tensor of rank one greater than the inputs, with the new dimension
    at ``axis``.
"""

argsort = functional(ops.argsort, rule=argsort_rule)
argsort.__doc__ = """Returns the indices that would sort a 1-D tensor.

Currently only supports rank-1 inputs.

Args:
    x: The input tensor. Must have rank 1.
    ascending: When ``True`` (the default), sort in ascending order. When
        ``False``, sort in descending order.

Returns:
    An ``int64`` tensor of the same shape as ``x`` containing sort
    indices.
"""

nonzero = functional(ops.nonzero, rule=nonzero_rule)
nonzero.__doc__ = """Returns the indices of the non-zero elements of a tensor.

Indices are produced in row-major order.

Args:
    x: The input tensor. Must have rank at least 1 (scalars are not
        supported).
    out_dim: The symbolic dimension labeling the dynamically-sized
        first axis of the output. Sized at runtime to the number of
        non-zero elements in ``x``.

Returns:
    A 2-D ``int64`` tensor of shape ``(out_dim, rank(x))`` where each
    row is the multi-dimensional index of a non-zero element.

Raises:
    ValueError: If ``x`` is a scalar (rank 0).
"""

gather = functional(ops.gather, rule=gather_rule)
gather.__doc__ = """Gathers values from a tensor along an axis using indices.

When the gather axis is :class:`~max.experimental.sharding.Sharded`,
the dispatcher will :func:`allgather` the input to
:class:`~max.experimental.sharding.Replicated` first; the rule does
not emit an expert-parallel ``(Sharded(a_axis), R) → Partial(SUM)``
row because that's only correct when the caller masks indices per
rank. Models that genuinely want EP semantics override
``gather.rule`` with their own rule.

Args:
    input: The input tensor to gather from.
    indices: An integer tensor of indices.
    axis: The axis to gather along.

Returns:
    A tensor whose shape along ``axis`` matches ``indices``, with values
    pulled from ``input``.
"""

scatter = functional(ops.scatter, rule=scatter_rule)
scatter.__doc__ = """Writes values into a tensor at positions specified by indices.

When the scatter axis is :class:`~max.experimental.sharding.Sharded`,
the dispatcher will :func:`allgather` the input to
:class:`~max.experimental.sharding.Replicated` first; the rule does
not emit a per-rank-local ``(Sharded(a_axis), R, R) → Sharded(a_axis)``
row because that's only correct when the caller masks indices and
updates per rank. Models that genuinely want EP semantics override
``scatter.rule`` with their own rule.

Args:
    input: The destination tensor.
    updates: The values to write.
    indices: An integer tensor of positions to write to.
    axis: The axis to scatter along. Defaults to ``-1``.

Returns:
    A tensor matching ``input`` with the scattered values written in.
"""

scatter_add = functional(ops.scatter_add, rule=scatter_add_rule)
scatter_add.__doc__ = """Scatters values into a tensor, accumulating via addition.

Like :func:`scatter`, but when multiple updates target the same position
their sum is written.

Args:
    input: The destination tensor.
    updates: The values to add at each position.
    indices: An integer tensor of positions to write to.
    axis: The axis to scatter along. Defaults to ``-1``.

Returns:
    A tensor matching ``input`` with the accumulated values added in.
"""

scatter_max = functional(ops.scatter_max, rule=scatter_add_rule)
scatter_max.__doc__ = """Scatters values into a tensor, keeping the per-position maximum.

When multiple updates target the same position, the maximum is written.

Args:
    input: The destination tensor.
    updates: The candidate values.
    indices: An integer tensor of positions to write to.
    axis: The axis to scatter along. Defaults to ``-1``.

Returns:
    A tensor matching ``input`` with maximums written into the scattered
    positions.
"""

scatter_min = functional(ops.scatter_min, rule=scatter_add_rule)
scatter_min.__doc__ = """Scatters values into a tensor, keeping the per-position minimum.

When multiple updates target the same position, the minimum is written.

Args:
    input: The destination tensor.
    updates: The candidate values.
    indices: An integer tensor of positions to write to.
    axis: The axis to scatter along. Defaults to ``-1``.

Returns:
    A tensor matching ``input`` with minimums written into the scattered
    positions.
"""

scatter_mul = functional(ops.scatter_mul, rule=scatter_add_rule)
scatter_mul.__doc__ = """Scatters values into a tensor, accumulating via multiplication.

When multiple updates target the same position, their product is written.

Args:
    input: The destination tensor.
    updates: The values to multiply at each position.
    indices: An integer tensor of positions to write to.
    axis: The axis to scatter along. Defaults to ``-1``.

Returns:
    A tensor matching ``input`` with the product of the scattered values.
"""

scatter_nd = functional(ops.scatter_nd, rule=scatter_nd_rule)
scatter_nd.__doc__ = """Writes values into a tensor at multi-dimensional indices.

Args:
    input: The destination tensor.
    updates: The values to write.
    indices: A tensor of multi-dimensional indices.

Returns:
    A tensor matching ``input`` with the scattered values written in.
"""

scatter_nd_add = functional(ops.scatter_nd_add, rule=scatter_nd_add_rule)
scatter_nd_add.__doc__ = """Scatters values via multi-dimensional indices, accumulating via addition.

Args:
    input: The destination tensor.
    updates: The values to add at each position.
    indices: A tensor of multi-dimensional indices.

Returns:
    A tensor matching ``input`` with the accumulated values added in.
"""

scatter_nd_max = functional(ops.scatter_nd_max, rule=scatter_nd_add_rule)
scatter_nd_max.__doc__ = """Scatters values via multi-dimensional indices, keeping the per-position max.

Args:
    input: The destination tensor.
    updates: The candidate values.
    indices: A tensor of multi-dimensional indices.

Returns:
    A tensor matching ``input`` with maximums written into the scattered
    positions.
"""

scatter_nd_min = functional(ops.scatter_nd_min, rule=scatter_nd_add_rule)
scatter_nd_min.__doc__ = """Scatters values via multi-dimensional indices, keeping the per-position min.

Args:
    input: The destination tensor.
    updates: The candidate values.
    indices: A tensor of multi-dimensional indices.

Returns:
    A tensor matching ``input`` with minimums written into the scattered
    positions.
"""

scatter_nd_mul = functional(ops.scatter_nd_mul, rule=scatter_nd_add_rule)
scatter_nd_mul.__doc__ = """Scatters values via multi-dimensional indices, accumulating via multiplication.

Args:
    input: The destination tensor.
    updates: The values to multiply at each position.
    indices: A tensor of multi-dimensional indices.

Returns:
    A tensor matching ``input`` with the product of the scattered values.
"""

gather_nd = functional(ops.gather_nd, rule=gather_nd_rule)
gather_nd.__doc__ = """Selects elements from a tensor by N-dimensional index.

Unlike :func:`gather`, which indexes a single axis, ``gather_nd`` indexes
multiple dimensions at once. The trailing dimension of ``indices``
selects elements from ``input`` immediately after any ``batch_dims``
leading dimensions; remaining trailing dimensions of ``input`` are
sliced into the output.

Args:
    input: The input tensor to gather from.
    indices: An integer tensor of multi-dimensional indices. Its last
        dimension must be static and gives the size of the index vector.
    batch_dims: The number of leading batch dimensions shared between
        ``input`` and ``indices``. The shapes must match exactly along
        these leading dimensions. Defaults to ``0``.

Returns:
    A tensor with the same dtype as ``input``. Its shape is the
    concatenation of:

    - ``input.shape[:batch_dims]`` (the leading batch dimensions),
    - ``indices.shape[batch_dims:-1]`` (the index dimensions), and
    - ``input.shape[batch_dims + indices.shape[-1]:]`` (the trailing
      sliced dimensions).
"""

masked_scatter = functional(ops.masked_scatter, rule=masked_scatter_rule)
masked_scatter.__doc__ = """Replaces positions in a tensor where a boolean mask is ``True``.

Args:
    input: The destination tensor.
    mask: A boolean tensor of the same shape as ``input``.
    updates: The values to write into the masked positions.
    out_dim: The output dimension size for the number of replaced
        elements. Used to construct the symbolic output shape.

Returns:
    A tensor matching ``input`` with values from ``updates`` written
    wherever ``mask`` is ``True``.
"""

outer = functional(ops.outer, rule=outer_rule)
outer.__doc__ = """Computes the outer product of two 1-D tensors.

Args:
    lhs: The left-hand side 1-D tensor of length ``M``.
    rhs: The right-hand side 1-D tensor of length ``N``.

Returns:
    A 2-D tensor of shape ``(M, N)`` whose ``(i, j)`` element is
    ``lhs[i] * rhs[j]``.
"""

_split_impl = functional(ops.split, rule=split_rule)


def split(
    x: Tensor,
    split_size_or_sections: int | Sequence[DimLike],
    axis: int = 0,
) -> list[Tensor]:
    """Splits a tensor into chunks along an axis.

    An ``int`` ``split_size_or_sections`` produces equal chunks (the
    last may be smaller); a sequence specifies per-chunk sizes.
    """
    if isinstance(split_size_or_sections, int):
        # On a sharded axis ``x.shape[axis]`` is a PerShardDim carrying the
        # global size; ``global_dim`` recovers that static global (and is a
        # no-op on a plain dim).
        dim = global_dim(Dim(x.shape[axis]))
        if not isinstance(dim, StaticDim):
            raise TypeError(
                f"split(x, chunk_size={split_size_or_sections}, axis={axis}): "
                f"non-static dim {x.shape[axis]!r}; pass an explicit "
                "split_sizes list."
            )
        dim_size = dim.dim
        chunk_size = split_size_or_sections
        num_full, remainder = divmod(dim_size, chunk_size)
        split_sizes: list[DimLike] = [chunk_size] * num_full
        if remainder > 0:
            split_sizes.append(remainder)
    else:
        split_sizes = list(split_size_or_sections)
    return _split_impl(x, split_sizes, axis)


top_k = functional(ops.top_k, rule=top_k_rule)
top_k.__doc__ = """Returns the k largest elements (and their indices) along an axis.

Args:
    input: The input tensor.
    k: The number of largest elements to return.
    axis: The axis along which to find the top-k. Defaults to ``-1``.

Returns:
    A pair ``(values, indices)`` where ``values`` are the top-k entries
    and ``indices`` are their positions along ``axis``.
"""

bottom_k = functional(ops.bottom_k, rule=top_k_rule)
bottom_k.__doc__ = """Returns the k smallest elements (and their indices) along an axis.

Values are returned sorted in ascending order.

Args:
    input: The input tensor.
    k: The number of smallest elements to return.
    axis: The axis along which to find the bottom-k. Defaults to ``-1``.

Returns:
    A pair ``(values, indices)`` where ``values`` are the k smallest
    entries in ascending order and ``indices`` are their positions along
    ``axis``.
"""

chunk = functional(ops.chunk, rule=chunk_rule)
chunk.__doc__ = """Splits a tensor into a given number of equal-sized chunks along an axis.

``chunks`` must statically divide ``x.shape[axis]``; otherwise this
raises a :obj:`ValueError`. Splitting a scalar (rank-0) tensor is only
valid when ``chunks == 1``.

For example, splitting a length-6 vector into three chunks:

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor.arange(6)         # [0, 1, 2, 3, 4, 5]
    parts = F.chunk(x, 3)
    # parts[0] is [0, 1]
    # parts[1] is [2, 3]
    # parts[2] is [4, 5]

Args:
    x: The input tensor.
    chunks: The number of chunks to produce. Must evenly divide
        ``x.shape[axis]``.
    axis: The axis along which to split. Negative values count from the
        end. Defaults to ``0``.

Returns:
    A list of ``chunks`` tensors of equal size along ``axis``.
"""


def _reduce_op(
    graph_op: Callable[..., object],
    rule: Callable[..., ActionSet],
) -> Callable[..., Tensor]:
    """Builds a reduction wrapper.

    An integer ``axis`` delegates to the single-axis graph op;
    ``axis=None`` flattens to 1-D first.
    """
    single_axis = functional(graph_op, rule)

    def fn(
        x: Tensor,
        axis: int | None = -1,
    ) -> Tensor:
        assert isinstance(x, tensor.Tensor)
        if axis is None:
            x = reshape(x, [-1])
            axis = 0
        return single_axis(x, axis)

    return fn


def _reduce_elementwise_op(
    graph_op: Callable[..., object],
    rule: Callable[..., ActionSet],
    elementwise_fn: Callable[[Tensor, Tensor], Tensor],
) -> Callable[..., Tensor]:
    """Builds a function that reduces (1 arg) or runs elementwise (2 args)."""
    reduce_fn = _reduce_op(graph_op, rule)

    def fn(
        x: Tensor,
        y: Tensor | None = None,
        /,
        axis: int | None = -1,
    ) -> Tensor:
        if y is not None:
            return elementwise_fn(x, y)
        return reduce_fn(x, axis=axis)

    return fn


#: Computes the sum along one or more axes. Distributed via SPMD.
#: See :func:`max.graph.ops.sum` for details.
sum = _reduce_op(ops.sum, rule=linear_reduce_rule)
#: Computes the mean along one or more axes. Distributed via SPMD.
#: See :func:`max.graph.ops.mean` for details.
mean = _reduce_op(ops.mean, rule=mean_rule)
#: Computes the product along one or more axes. Distributed via SPMD.
#: See :func:`max.graph.ops.prod` for details.
prod = _reduce_op(ops.prod, rule=reduce_rule)
prod.__doc__ = """Computes the product of a tensor along an axis.

Args:
    x: The input tensor.
    axis: The axis along which to reduce. When ``None``, the tensor is
        flattened to 1-D and reduced. Defaults to ``-1``.

Returns:
    A tensor with the product computed along ``axis``.
"""

_argmax_impl = _reduce_op(ops.argmax, rule=reduce_rule)
_argmin_impl = _reduce_op(ops.argmin, rule=reduce_rule)


def argmax(
    x: Tensor,
    axis: int | None = -1,
) -> Tensor:
    """Returns the indices of the maximum values along an axis.

    Args:
        x: The input tensor.
        axis: The axis along which to find the maximum. When ``None``, the
            tensor is flattened to 1-D first. Defaults to ``-1``.

    Returns:
        An integer tensor of indices marking the positions of the maximum
        values along ``axis``.
    """
    return _argmax_impl(x, axis=axis)


def argmin(
    x: Tensor,
    axis: int | None = -1,
) -> Tensor:
    """Returns the indices of the minimum values along an axis.

    Args:
        x: The input tensor.
        axis: The axis along which to find the minimum. When ``None``, the
            tensor is flattened to 1-D first. Defaults to ``-1``.

    Returns:
        An integer tensor of indices marking the positions of the minimum
        values along ``axis``.
    """
    return _argmin_impl(x, axis=axis)


max = _reduce_elementwise_op(
    ops.reduction.max,
    rule=reduce_rule,
    elementwise_fn=elementwise_max,
)
max.__doc__ = """Computes the maximum of a tensor, or the element-wise maximum of two tensors.

Called with one argument, reduces ``x`` along ``axis``. Called with two
tensor arguments, returns their element-wise maximum.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]])

    row_max = F.max(x, axis=-1)
    # row_max has shape (2, 1): [[3.5], [4.2]]

    col_max = F.max(x, axis=0)
    # col_max has shape (1, 4): [[2.3, 3.5, 4.2, 3.1]]

    y = Tensor([[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
    element_wise = F.max(x, y)
    # element_wise: [[2.0, 3.5, 2.1, 2.0], [2.3, 2.0, 4.2, 3.1]]

Args:
    x: The input tensor.
    y: Optional second tensor. When provided, the result is the
        element-wise maximum of ``x`` and ``y``.
    axis: The axis to reduce along when ``y`` is omitted. When ``None``,
        the tensor is flattened to 1-D first. Defaults to ``-1``.

Returns:
    A tensor containing either the reduced maximum along ``axis`` or the
    element-wise maximum with the broadcast shape of the inputs.
"""

min = _reduce_elementwise_op(
    ops.reduction.min,
    rule=reduce_rule,
    elementwise_fn=elementwise_min,
)
min.__doc__ = """Computes the minimum of a tensor, or the element-wise minimum of two tensors.

Called with one argument, reduces ``x`` along ``axis``. Called with two
tensor arguments, returns their element-wise minimum.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([[1.2, 3.5, 2.1, 0.8], [2.3, 1.9, 4.2, 3.1]])

    row_min = F.min(x, axis=-1)
    # row_min has shape (2, 1): [[0.8], [1.9]]

    col_min = F.min(x, axis=0)
    # col_min has shape (1, 4): [[1.2, 1.9, 2.1, 0.8]]

    y = Tensor([[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
    element_wise = F.min(x, y)
    # element_wise: [[1.2, 2.0, 2.0, 0.8], [2.0, 1.9, 2.0, 2.0]]

Args:
    x: The input tensor.
    y: Optional second tensor. When provided, the result is the
        element-wise minimum of ``x`` and ``y``.
    axis: The axis to reduce along when ``y`` is omitted. When ``None``,
        the tensor is flattened to 1-D first. Defaults to ``-1``.

Returns:
    A tensor containing either the reduced minimum along ``axis`` or the
    element-wise minimum with the broadcast shape of the inputs.
"""

#: Applies the softmax function along an axis. Distributed via SPMD.
#: See :func:`max.graph.ops.softmax` for details.
softmax = functional(ops.softmax, rule=softmax_rule)
#: Applies the log softmax function along an axis. Distributed via SPMD.
#: See :func:`max.graph.ops.logsoftmax` for details.
logsoftmax = functional(ops.logsoftmax, rule=softmax_rule)
#: Computes the cumulative sum along an axis. Distributed via SPMD.
#: See :func:`max.graph.ops.cumsum` for details.
cumsum = functional(ops.cumsum, rule=linear_reduce_rule)
cumsum.__doc__ = """Computes the cumulative sum of a tensor along an axis.

Args:
    x: The input tensor.
    axis: The axis along which to compute the cumulative sum. Defaults to
        ``-1``.
    exclusive: When ``True``, the first output value is ``0`` and the
        final input element is excluded from the sum. Defaults to
        ``False``.
    reverse: When ``True``, computes the sum starting from the end of the
        axis. Defaults to ``False``.

Returns:
    A tensor of the same shape and dtype where each element is the sum of
    the corresponding input elements up to that position along ``axis``.
"""


#: Applies 2D convolution. Distributed via SPMD.
#: See :func:`max.graph.ops.conv2d` for details.
conv2d = functional(ops.conv2d, rule=conv2d_rule)
conv2d.__doc__ = """Applies a 2D convolution to a tensor.

Computes the 2-D convolution product of ``x`` with ``filter``, plus the
optional ``bias``. Currently supports strides and padding on the input
only.

Args:
    x: A rank-4 input tensor. With the default ``NHWC`` input layout,
        the shape is ``(N, H, W, C_in)``.
    filter: A rank-4 convolution kernel. With the default ``RSCF``
        filter layout, the shape is ``(H, W, C_in / groups, C_out)``.
    stride: The stride of the convolution, as ``(stride_h, stride_w)``.
        Defaults to ``(1, 1)``.
    dilation: The spacing between kernel elements, as ``(dilation_h,
        dilation_w)``. Defaults to ``(1, 1)``.
    padding: Zero-padding applied to the input, as
        ``(pad_h_before, pad_h_after, pad_w_before, pad_w_after)``.
        Defaults to ``(0, 0, 0, 0)``.
    groups: The number of groups for grouped convolution. Both ``C_in``
        and ``C_out`` must be divisible by ``groups``. Defaults to ``1``.
    bias: Optional rank-1 bias tensor of shape ``(C_out,)`` added to the
        convolution output.
    input_layout: The layout of the input tensor. Defaults to
        ``ConvInputLayout.NHWC``.
    filter_layout: The layout of the filter tensor. Defaults to
        ``FilterLayout.RSCF``.

Returns:
    The convolution result. With the default ``NHWC`` input layout, the
    shape is ``(N, H_out, W_out, C_out)``.

Raises:
    ValueError: If ``x`` is not rank 4, ``filter`` is not rank 4, or
        ``bias`` is provided and is not rank 1.
"""

conv3d = functional(ops.conv3d, rule=conv3d_rule)
conv3d.__doc__ = """Applies a 3D convolution to a tensor.

Computes the 3-D convolution product of ``x`` with ``filter``, plus the
optional ``bias``. Currently supports strides and padding on the input
only.

Args:
    x: A rank-5 input tensor. With the default channels-last (NDHWC)
        input layout, the shape is ``(N, D, H, W, C_in)``.
    filter: A rank-5 convolution kernel. With the default ``QRSCF``
        filter layout, the shape is ``(D, H, W, C_in / groups, C_out)``.
    stride: The stride of the convolution, as
        ``(stride_d, stride_h, stride_w)``. Defaults to ``(1, 1, 1)``.
    dilation: The spacing between kernel elements, as
        ``(dilation_d, dilation_h, dilation_w)``. Defaults to
        ``(1, 1, 1)``.
    padding: Zero-padding applied to the input, as
        ``(pad_d_before, pad_d_after, pad_h_before, pad_h_after,
        pad_w_before, pad_w_after)``. Defaults to ``(0, 0, 0, 0, 0, 0)``.
    groups: The number of groups for grouped convolution. Both ``C_in``
        and ``C_out`` must be divisible by ``groups``. Defaults to ``1``.
    bias: Optional rank-1 bias tensor of shape ``(C_out,)`` added to the
        convolution output.
    input_layout: The layout of the input tensor. Defaults to
        ``ConvInputLayout.NHWC`` (channels-last).
    filter_layout: The layout of the filter tensor. Defaults to
        ``FilterLayout.QRSCF``.

Returns:
    The convolution result. With the default channels-last input
    layout, the shape is ``(N, D, H_out, W_out, C_out)``.

Raises:
    ValueError: If ``x`` is not rank 5, ``filter`` is not rank 5, or
        ``bias`` is provided and is not rank 1.
"""

conv2d_transpose = functional(ops.conv2d_transpose, rule=conv2d_transpose_rule)
conv2d_transpose.__doc__ = """Applies a 2D transposed convolution to a tensor.

Also known as fractionally-strided or deconvolution. Computes the
gradient of a 2-D convolution with respect to its input, as if the
original convolution had the same filter and hyperparameters. Commonly
used to upsample feature maps.

Args:
    x: A rank-4 input tensor. With the default ``NHWC`` input layout,
        the shape is ``(N, H, W, C_in)``.
    filter: A rank-4 convolution kernel. With the default ``RSCF``
        filter layout, the shape is ``(H, W, C_out, C_in)``. Note that
        the channel order is reversed relative to :func:`conv2d`.
    stride: The stride of the transposed convolution, as
        ``(stride_h, stride_w)``. Defaults to ``(1, 1)``.
    dilation: The spacing between kernel elements, as
        ``(dilation_h, dilation_w)``. Defaults to ``(1, 1)``.
    padding: Zero-padding applied to the input, as
        ``(pad_h_before, pad_h_after, pad_w_before, pad_w_after)``.
        Defaults to ``(0, 0, 0, 0)``.
    output_paddings: Additional size added to one side of each spatial
        output dimension, as ``(out_pad_h, out_pad_w)``. Resolves the
        ambiguity in output shape when ``stride > 1``. Each value must be
        strictly less than the corresponding ``stride``. Currently only
        ``(0, 0)`` is supported. Defaults to ``(0, 0)``.
    bias: Optional rank-1 bias tensor of shape ``(C_out,)`` added to the
        transposed-convolution output.
    input_layout: The layout of the input tensor. Defaults to
        ``ConvInputLayout.NHWC``.
    filter_layout: The layout of the filter tensor. Defaults to
        ``FilterLayout.RSCF``.

Returns:
    The transposed-convolution result with shape
    ``(N, H_out, W_out, C_out)`` for the default ``NHWC`` input layout.

Raises:
    ValueError: If ``x`` is not rank 4, ``filter`` is not rank 4,
        ``bias`` is provided and is not rank 1, or any
        ``output_paddings`` value is greater than or equal to the
        corresponding ``stride``.
"""


#: Copies a tensor setting everything outside a central band to zero. Distributed via SPMD.
#: See :func:`max.graph.ops.band_part` for details.
band_part = functional(ops.band_part, rule=band_part_rule)
band_part.__doc__ = """Masks out everything except a diagonal band of an input matrix.

Operates on the last two axes of ``x`` (any earlier axes are treated as
batch dimensions). Elements outside the central diagonal band of each
sub-matrix are set to zero.

Args:
    x: The input tensor. Must have rank at least 2.
    num_lower: The number of subdiagonals to keep. Use :obj:`None` to
        keep the entire lower triangle.
    num_upper: The number of superdiagonals to keep. Use :obj:`None` to
        keep the entire upper triangle.
    exclude: When ``True``, inverts the selection — elements inside the
        band are zeroed and elements outside are kept. Defaults to
        ``False``.

Returns:
    A tensor of the same shape as ``x`` with elements outside the band
    set to zero.
"""

fold = functional(ops.fold, rule=fold_rule)
fold.__doc__ = """Combines an array of sliding local blocks into a larger containing tensor.

The inverse of an ``unfold`` operation.

The input tensor is rank 3 with shape ``(N, C * kernel_sizes, L)``,
where ``N`` is the batch dimension, ``C`` is the number of channels,
``kernel_sizes`` is the product ``kernel_size[0] * kernel_size[1]``, and
``L`` is the number of local blocks. The output is rank 4 with shape
``(N, C, output_size[0], output_size[1])``.

The number of blocks ``L`` must satisfy:

.. code-block:: text

    L = prod((output_size[d] + 2 * padding[d]
              - dilation[d] * (kernel_size[d] - 1) - 1) / stride[d] + 1)

where ``d`` ranges over the spatial dimensions.

Args:
    input: The 3-D input tensor of unfolded blocks with shape
        ``(N, C * kernel_sizes, L)``.
    output_size: The spatial dimensions of the output, as
        ``(out_h, out_w)``. Must be a tuple of two ints.
    kernel_size: The size of the sliding blocks, as
        ``(kernel_h, kernel_w)``. Must be a tuple of two ints.
    stride: The stride of the sliding blocks. Either a single ``int``
        applied to both spatial dimensions, or a tuple
        ``(stride_h, stride_w)``. Defaults to ``1``.
    dilation: The spacing between kernel elements. Either a single
        ``int`` applied to both spatial dimensions, or a tuple
        ``(dilation_h, dilation_w)``. Defaults to ``1``.
    padding: Zero-padding added to both sides of each spatial dimension.
        Either a single ``int`` applied to both spatial dimensions, or a
        tuple ``(pad_h, pad_w)``. Defaults to ``0``.

Returns:
    The folded 4-D tensor with shape
    ``(N, C, output_size[0], output_size[1])``.

Raises:
    ValueError: If dimension 1 of ``input`` is not a multiple of
        ``kernel_size[0] * kernel_size[1]``, or if dimension 2 of
        ``input`` doesn't match the computed number of blocks ``L``.
"""

as_interleaved_complex = functional(
    ops.complex.as_interleaved_complex,
    rule=as_interleaved_complex_rule,
)
as_interleaved_complex.__doc__ = """Reshapes a real tensor of alternating (real, imag) values into complex form.

Pulls each adjacent ``(real, imag)`` pair in the last dimension out into
a trailing pair of size 2.

Args:
    x: A real tensor representing complex numbers as alternating pairs of
        ``(real, imag)`` values. The last dimension must have an even
        size.

Returns:
    A tensor of shape ``(*x.shape[:-1], x.shape[-1] // 2, 2)``. All
    dimensions except the last are unchanged; the last dimension is
    halved, and a final dimension of size 2 is appended to hold the
    ``(real, imag)`` components.
"""

complex_mul = functional(ops.complex.mul, rule=binary_rule)
complex_mul.__doc__ = """Multiplies two complex-valued tensors element-wise.

Both inputs must use the interleaved complex representation (trailing
dimension of size 2).

Args:
    lhs: The left-hand side complex tensor.
    rhs: The right-hand side complex tensor.

Returns:
    A complex tensor with the broadcast shape containing element-wise
    products.
"""

resize = functional(ops.resize, rule=resize_rule)
resize.__doc__ = """Resizes a 4-D tensor to the given shape.

The input must be in NCHW layout — that is, a rank-4 tensor whose
dimensions represent ``(N, C, H, W)``: batch size, channels, height,
and width.

Dispatches to :func:`resize_nearest`, :func:`resize_linear`, or
:func:`resize_bicubic` based on ``interpolation``.

Args:
    input: The input tensor. Must have rank 4 in NCHW layout.
    shape: The full output shape of length 4 as ``(N, C, H, W)``.
    interpolation: The interpolation mode used to compute output values.
        Defaults to ``InterpolationMode.BILINEAR``.

Returns:
    A resized tensor with the given ``shape`` and the same dtype as
    ``input``.
"""

resize_linear = functional(ops.resize_linear, rule=resize_linear_rule)
#: Resizes a tensor using nearest-neighbor interpolation. Distributed via SPMD.
#: See :func:`max.graph.ops.resize_nearest` for details.
resize_nearest = functional(ops.resize_nearest, rule=resize_nearest_rule)
#: Resizes a tensor using bicubic interpolation. Distributed via SPMD.
#: See :func:`max.graph.ops.resize_bicubic` for details.
resize_bicubic = functional(ops.resize_bicubic, rule=resize_bicubic_rule)
#: Computes the inverse real FFT. Distributed via SPMD.
#: See :func:`max.graph.ops.irfft` for details.
irfft = functional(ops.irfft, rule=irfft_rule)
irfft.__doc__ = """Computes the inverse of the real-input FFT.

Args:
    input_tensor: The input tensor to compute the inverse real FFT of.
    n: The size of the output tensor. The input tensor is padded or
        truncated to ``n // 2 + 1`` along ``axis``.
    axis: The axis along which to compute the inverse FFT. Defaults to
        ``-1``.
    normalization: The normalization to apply to the output tensor. One of
        ``"backward"``, ``"ortho"``, or ``"forward"``. When ``"backward"``,
        the output is divided by ``n``. When ``"ortho"``, the output is
        divided by ``sqrt(n)``. When ``"forward"``, no normalization is
        applied.
    input_is_complex: Whether the input tensor is already interleaved
        complex. When ``True``, the last dimension of the input tensor must
        be 2, and is excluded from the dimension referred to by ``axis``.
    buffer_size_mb: The estimated size of a persistent buffer to use for
        storage of intermediate results. Needs to be the same across
        multiple calls to ``irfft`` within the same graph.

Returns:
    A real tensor that is the inverse FFT of the complex input. The shape
    matches the input shape, except along ``axis``, which is replaced by
    ``n``.
"""


def _cond_graph(
    pred: TensorValueLike,
    out_types: Iterable[Type[Any]] | None,
    then_fn: Callable[[], Iterable[TensorValueLike] | TensorValueLike | None],
    else_fn: Callable[[], Iterable[TensorValueLike] | TensorValueLike | None],
) -> list[TensorValue]:
    """``ops.cond`` requires a CPU predicate — inserts a transfer when needed."""
    pred = TensorValue(pred)
    if not pred.device.is_cpu():
        pred = ops.transfer_to(pred, CPU())
    return ops.cond(pred, out_types, then_fn, else_fn)


cond = functional(_cond_graph, rule=cond_rule)
cond.__doc__ = """Conditionally executes one of two branches based on a boolean predicate.

Both branches must return the same number and types of values as
specified by ``out_types``. The predicate is evaluated at runtime to
determine which branch executes. If ``pred`` lives on a non-CPU device,
it is transferred to CPU automatically.

.. code-block:: python

    from max.dtype import DType
    from max.experimental import Tensor
    from max.experimental import functional as F
    from max.graph import DeviceRef, TensorType

    def then_fn():
        return Tensor([1.0, 2.0])

    def else_fn():
        return Tensor([10.0, 20.0])

    pred = Tensor(True)
    out_types = [TensorType(DType.float32, [2], DeviceRef.CPU())]
    (result,) = F.cond(pred, out_types, then_fn, else_fn)
    # pred is True, so result is [1.0, 2.0]

Args:
    pred: A boolean scalar tensor of type :attr:`~max.dtype.DType.bool`
        determining which branch to execute.
    out_types: The expected output types for both branches. Use
        :obj:`None` for branches that don't return values (such as
        buffer mutations).
    then_fn: A callable executed when ``pred`` is ``True``.
    else_fn: A callable executed when ``pred`` is ``False``.

Returns:
    The output values from the executed branch, or an empty list when
    ``out_types`` is :obj:`None`.
"""


def _while_loop_graph(
    initial_values: Iterable[TensorValueLike] | TensorValueLike,
    predicate: Callable[..., Tensor],
    body: Callable[..., Tensor | Iterable[Tensor]],
) -> list[TensorValue]:
    """Wrap predicate/body so callbacks see :class:`Tensor`.

    ``ops.while_loop`` passes :class:`TensorValue` into its predicate/body
    and expects :class:`TensorValue` back. This wrapper wraps callback
    args as :class:`Tensor` and coerces callback returns back to
    :class:`TensorValue`. The outer ``functional()`` wrapper converts the
    returned :class:`TensorValue` list back to :class:`Tensor` for the
    public surface.
    """

    def _pred(*args: TensorValue) -> TensorValue:
        tensors = [Tensor.from_graph_value(a) for a in args]
        return TensorValue(predicate(*tensors))

    def _body(*args: TensorValue) -> list[TensorValue]:
        tensors = [Tensor.from_graph_value(a) for a in args]
        result = body(*tensors)
        if isinstance(result, Tensor):
            return [TensorValue(result)]
        return [TensorValue(t) for t in result]

    if isinstance(initial_values, Iterable):
        unwrapped = [TensorValue(v) for v in initial_values]
    else:
        unwrapped = [TensorValue(initial_values)]
    return ops.while_loop(unwrapped, _pred, _body)


while_loop = functional(_while_loop_graph, rule=while_loop_rule)
while_loop.__doc__ = """Repeatedly executes a body function while a predicate holds.

Both ``predicate`` and ``body`` receive and return :class:`Tensor`
values. They take the same number and types of arguments as the initial
values. The predicate must return a single boolean scalar tensor that
controls loop continuation; the body must return updated values matching
the types of ``initial_values``.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    def predicate(x):
        return x < 10

    def body(x):
        return x + 1

    x = Tensor(0)
    (result,) = F.while_loop(x, predicate, body)
    # Loop continues until ``x >= 10``; result is ``10``.

Args:
    initial_values: The initial values for the loop arguments. Must be
        non-empty.
    predicate: A callable that takes the loop arguments and returns a
        boolean scalar tensor of type :attr:`~max.dtype.DType.bool`.
    body: A callable that takes the loop arguments and returns updated
        values matching the types of ``initial_values``.

Returns:
    The output values from the final loop iteration.
"""


# Mutation ops: hand-rolled because they write in-place via __buffervalue__().


def _spmd_buffer_write(
    destination: Tensor,
    source: Tensor,
    write: Callable[[Any, Any], None],
) -> None:
    """Per-shard in-place write; flows each post-write value back to ``destination._state``."""
    shards = list(destination.local_shards)
    src_shards = list(source.local_shards) if source.is_distributed else None
    for i, dest_tensor in enumerate(shards):
        dest_shard = dest_tensor.__buffervalue__()
        src_shard = (
            src_shards[i].__tensorvalue__()
            if src_shards is not None
            else source.__tensorvalue__()
        )
        write(dest_shard, src_shard)
        if destination._state is not None and dest_tensor._state is not None:
            new_values = list(destination._state.values)
            new_values[i] = dest_tensor._state.value
            destination._state = type(destination._state)(
                tuple(new_values), destination._state.ctx
            )


def buffer_store(destination: Tensor, source: Tensor) -> None:
    """Stores values from a tensor into a tensor buffer.

    Args:
        destination: The destination buffer tensor.
        source: The source tensor whose values are written into
            ``destination``.
    """
    if destination.is_distributed:
        buffer_store_rule(
            tensor_to_layout(destination), tensor_to_layout(source)
        )

    with ensure_context():
        if destination.is_distributed:
            _spmd_buffer_write(destination, source, ops.buffer_store)
        else:
            ops.buffer_store(
                destination.__buffervalue__(), source.__tensorvalue__()
            )


def buffer_store_slice(
    destination: Tensor,
    source: Tensor,
    indices: SliceIndices,
) -> None:
    """Stores values into a slice of a tensor buffer.

    Args:
        destination: The destination buffer tensor.
        source: The source tensor whose values are written into the slice.
        indices: The slice specification within ``destination`` to write to.
    """
    if destination.is_distributed:
        buffer_store_slice_rule(
            tensor_to_layout(destination), tensor_to_layout(source), indices
        )

    with ensure_context():
        if destination.is_distributed:

            def _write(dest_buf: Any, src_tv: Any) -> None:
                dest_buf[indices] = src_tv

            _spmd_buffer_write(destination, source, _write)
        else:
            dest_buf = destination.__buffervalue__()
            source_tv = source.__tensorvalue__()
            dest_buf[indices] = source_tv


#: Applies group normalization.
#: See :func:`max.graph.ops.group_norm` for details.
group_norm = functional(ops.group_norm)
#: Applies RMS normalization.
#: See :func:`max.graph.ops.rms_norm` for details.
rms_norm = functional(ops.rms_norm, rule=rms_norm_rule)
#: Filters boxes with high intersection-over-union.
#: See :func:`max.graph.ops.non_maximum_suppression` for details.
non_maximum_suppression = functional(ops.non_maximum_suppression)
non_maximum_suppression.__doc__ = """Filters boxes by greedy non-maximum suppression per ``(batch, class)`` pair.

Object detectors often produce many overlapping bounding boxes around
the same object. Non-maximum suppression keeps only the
highest-scoring representative and discards lower-scoring boxes that
significantly overlap one already kept.

Overlap is measured by intersection-over-union (IoU): the area of the
intersection of two boxes divided by the area of their union. A value
of ``0`` means no overlap and a value of ``1`` means the boxes are
identical.

For each ``(batch, class)`` pair, the algorithm:

1. Drops boxes whose score is at or below ``score_threshold``.
2. Sorts the remaining boxes by score in descending order.
3. Walks the sorted list, keeping each box unless its IoU with an
   already-kept box exceeds ``iou_threshold`` (in which case it's
   suppressed).
4. Stops once ``max_output_boxes_per_class`` boxes have been kept.

Boxes are expressed in ``[y1, x1, y2, x2]`` corner format.

Args:
    boxes: A 3-D float tensor of shape ``[batch, num_boxes, 4]``.
    scores: A 3-D float tensor of per-class scores of shape
        ``[batch, num_classes, num_boxes]``. Must have the same dtype as
        ``boxes``.
    max_output_boxes_per_class: A scalar ``int64`` tensor giving the
        maximum number of boxes selected per ``(batch, class)`` pair.
    iou_threshold: A scalar float tensor giving the IoU suppression
        threshold.
    score_threshold: A scalar float tensor giving the minimum score
        required to keep a box.
    out_dim: The name of the symbolic output dimension representing the
        number of selected boxes. Defaults to ``"num_selected"``.

Returns:
    An ``int64`` tensor of shape ``[out_dim, 3]`` where each row is
    ``[batch_index, class_index, box_index]``.
"""

roi_align = functional(ops.roi_align)
roi_align.__doc__ = """Performs Region of Interest (ROI) align pooling on an NHWC tensor.

Extracts fixed-size feature maps from regions of interest in the input
tensor using bilinear interpolation.

Args:
    input: The input feature-map tensor of shape ``[N, H, W, C]``.
    rois: A tensor of regions of interest of shape ``[M, 5]``, where
        each row is ``[batch_index, x1, y1, x2, y2]``.
    output_height: The height of each pooled output feature map.
    output_width: The width of each pooled output feature map.
    spatial_scale: A multiplicative factor mapping ROI coordinates to
        input spatial coordinates. Defaults to ``1.0``.
    sampling_ratio: The number of sampling points per bin in each
        direction. ``0`` (the default) means adaptive
        (``ceil(bin_size)``).
    aligned: When ``True``, applies a half-pixel offset to ROI
        coordinates for more precise alignment. Defaults to ``False``.
    mode: The pooling mode applied to sampled values. One of ``"AVG"``
        or ``"MAX"``. Defaults to ``"AVG"``.

Returns:
    A tensor of shape ``[M, output_height, output_width, C]`` of pooled
    features.
"""


def clamp(
    x: Tensor,
    lower_bound: TensorValueLike,
    upper_bound: TensorValueLike,
) -> Tensor:
    """Clamps tensor values to ``[lower_bound, upper_bound]``."""
    return max(min(x, upper_bound), lower_bound)


clip = clamp
rebind = functional(ops.rebind, rule=rebind_rule)
rebind.__doc__ = """Rebinds the symbolic shape of a tensor.

Asserts at runtime that the tensor's dimensions match the new shape.
Useful for narrowing dynamic dimensions to specific sizes when you have
external knowledge of their values.

Args:
    x: The input tensor.
    shape: The new symbolic shape.
    message: A message included in the runtime assertion if the shapes
        don't match. Defaults to ``""``.
    layout: An optional filter layout to attach to the result. Defaults
        to :obj:`None`.

Returns:
    A tensor with the same data and the new symbolic shape.
"""

acos.__doc__ = """Computes the arccosine of a tensor element-wise.

.. code-block:: python

    from max.experimental import Tensor
    from max.experimental import functional as F

    x = Tensor([-1.0, 0.0, 1.0])
    result = F.acos(x)
    # result is approximately [3.1416, 1.5708, 0.0] or [pi, pi/2, 0]

Args:
    x: The input tensor, with values in the range ``[-1, 1]``. Values
        outside this domain are clamped. Must have a floating-point dtype.

Returns:
    A tensor of the same shape and dtype with values in the range
    ``[0, pi]`` (radians).
"""
dequantize.__doc__ = """Dequantizes a quantized tensor back to a floating-point representation.

Currently supports the ``Q4_0``, ``Q4_K``, and ``Q6_K`` encodings.

Args:
    encoding: The :class:`~max.graph.quantization.QuantizationEncoding`
        used to pack ``quantized``.
    quantized: The input quantized tensor.

Returns:
    A floating-point tensor with the values reconstructed from the
    quantized input.
"""
group_norm.__doc__ = """Applies group normalization over the channel axis of a tensor.

Splits the channel axis (axis 1) of ``input`` into ``num_groups``
groups, computes the mean and variance within each group, and
normalizes. ``gamma`` and ``beta`` then apply a per-channel affine
transform.

Args:
    input: The input tensor.
    gamma: The scale parameter tensor.
    beta: The shift parameter tensor.
    num_groups: The number of groups to split the channels into.
    epsilon: A small constant added to the variance for numerical
        stability.

Returns:
    A tensor of the same shape and dtype as ``input`` with group
    normalization applied.
"""
layer_norm.__doc__ = """Applies layer normalization over the last dimension of a tensor.

Computes ``gamma * (input - mean) / sqrt(var + epsilon) + beta``, where
``mean`` and ``var`` are reduced over the last axis of ``input`` and
broadcast back across the leading axes.

Args:
    input: The input tensor.
    gamma: The scale parameter tensor.
    beta: The shift parameter tensor.
    epsilon: A small constant added to the variance for numerical stability.

Returns:
    A tensor of the same shape and dtype as ``input`` with layer
    normalization applied.
"""
logsoftmax.__doc__ = """Computes ``log(softmax(x))`` along an axis.

Args:
    value: The input tensor.
    axis: The axis along which to compute the log-softmax. Defaults to the
        final axis (``-1``).

Returns:
    A tensor of the same shape and dtype with log-softmax applied along
    ``axis``.
"""
mean.__doc__ = """Computes the mean of a tensor along an axis.

Args:
    x: The input tensor.
    axis: The axis along which to reduce. When ``None``, the tensor is
        flattened to 1-D and reduced. Defaults to ``-1``.

Returns:
    A tensor with the mean computed along ``axis``.
"""
pad.__doc__ = """Pads a tensor along every dimension.

Args:
    input: The input tensor.
    paddings: A flat sequence of ``2 * rank(input)`` non-negative
        integers in the order
        ``[pad_before_dim0, pad_after_dim0, pad_before_dim1, pad_after_dim1, ...]``.
    mode: The padding mode. One of ``"constant"`` (fill with ``value``),
        ``"reflect"`` (reflect interior values about the edges, excluding
        the boundary), or ``"edge"`` (repeat the nearest boundary
        element). Defaults to ``"constant"``.
    value: The constant fill value used when ``mode == "constant"``.
        Defaults to ``0``.

Returns:
    A tensor with the same dtype as ``input`` padded along each
    dimension according to ``paddings``.
"""
qmatmul.__doc__ = """Performs matrix multiplication between a floating-point and a quantized tensor.

Computes ``dequantize(quantize(lhs) @ transpose(rhs))``: ``lhs`` is
quantized to match ``rhs``'s encoding, the matmul runs in the quantized
domain, then the result is dequantized back to floating point. ``rhs``
must be supplied in *transposed* form — for ``lhs`` of shape ``[M, K]``
and (transposed) ``rhs`` of shape ``[N, K]``, the output shape is
``[M, N]``. Currently supports the ``Q4_0``, ``Q4_K``, and ``Q6_K``
encodings.

Args:
    encoding: The quantization encoding used to pack ``rhs``.
    config: Optional quantization configuration. Required for some
        encodings (for example, ``GPTQ``); may be :obj:`None` otherwise.
    lhs: The left-hand side floating-point tensor.
    rhs: One or more packed and transposed quantized right-hand side
        tensors.

Returns:
    A floating-point tensor containing the dequantized matrix product.
"""
resize_bicubic.__doc__ = """Resizes a 4-D tensor using bicubic interpolation.

The input must be in NCHW layout — that is, a rank-4 tensor whose
dimensions represent ``(N, C, H, W)``: batch size, channels, height,
and width.

Uses a 4x4-pixel Catmull-Rom cubic filter with half-pixel coordinate
mapping.

Args:
    input: The input tensor. Must have rank 4 in NCHW layout.
    size: The full output shape of length 4 as ``(N, C, H, W)``.

Returns:
    A tensor with the given ``size`` and the same dtype as ``input``.
"""
resize_linear.__doc__ = """Resizes a tensor using linear (bilinear) interpolation.

Args:
    input: The input symbolic tensor to resize.
    size: The full output shape. Must have the same rank as ``input``.
    coordinate_transform_mode: How to map an output coordinate back to an
        input coordinate. One of ``0`` (``half_pixel``, the default),
        ``1`` (``align_corners``), ``2`` (``asymmetric``), or ``3``
        (``half_pixel_1D``).
    antialias: When ``True``, applies an antialiasing filter when the
        output is smaller than the input (downscaling). Has no effect
        when upscaling. Defaults to ``False``.

Returns:
    A tensor with the given ``size`` and the same dtype as ``input``.
"""
resize_nearest.__doc__ = """Resizes a tensor using nearest-neighbor interpolation.

Args:
    input: The input symbolic tensor to resize.
    size: The full output shape. Must have the same rank as ``input``.
    coordinate_transform_mode: How to map an output coordinate back to an
        input coordinate. One of ``0`` (``half_pixel``, the default),
        ``1`` (``align_corners``), ``2`` (``asymmetric``), or ``3``
        (``half_pixel_1D``).
    round_mode: How to round the mapped coordinate to select the nearest
        input sample. One of ``0`` (``HalfDown``, the default), ``1``
        (``HalfUp``), ``2`` (``Floor``), or ``3`` (``Ceil``).

Returns:
    A tensor with the given ``size`` and the same dtype as ``input``.
"""
rms_norm.__doc__ = """Applies RMS (root-mean-square) normalization over the last dimension of a tensor.

Computes ``input / rms(input) * (weight + weight_offset)`` where
``rms(x) = sqrt(mean(x ** 2) + epsilon)``. The reduction runs over the
last axis of ``input`` and is broadcast back across the leading axes.

Args:
    input: The input tensor.
    weight: The scale parameter tensor.
    epsilon: A small constant added to the mean-square for numerical
        stability.
    weight_offset: A constant added to ``weight`` before scaling. Defaults
        to ``0.0``.
    multiply_before_cast: When ``True``, multiplies by the scaled weight
        before casting the result back to the input dtype. Defaults to
        ``False``.

Returns:
    A tensor of the same shape and dtype as ``input`` with RMS
    normalization applied.
"""
softmax.__doc__ = """Applies the softmax function to a tensor along an axis.

Normalizes the values along ``axis`` so that they sum to ``1``.

Args:
    value: The input tensor.
    axis: The axis along which to compute the softmax. Defaults to the
        final axis (``-1``).

Returns:
    A tensor of the same shape and dtype with softmax applied along
    ``axis``.
"""
squeeze.__doc__ = """Removes a size-1 dimension from a tensor.

Args:
    x: The input tensor.
    axis: The dimension to remove. Must have size 1.

Returns:
    A tensor of rank ``x.rank - 1`` with the size-1 dimension at ``axis``
    removed.
"""
sum.__doc__ = """Computes the sum of a tensor along an axis.

Args:
    x: The input tensor.
    axis: The axis along which to reduce. When ``None``, the tensor is
        flattened to 1-D and reduced. Defaults to ``-1``.

Returns:
    A tensor with the sum computed along ``axis``.
"""
