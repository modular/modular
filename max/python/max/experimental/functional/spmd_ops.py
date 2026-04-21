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
3. Dispatches per-shard via ``spmd_dispatch``.

Rules return ``(suggested_args, output_mappings)`` — a 2-tuple with no kwargs.
"""

from __future__ import annotations

import builtins
import functools
import inspect
from collections.abc import Callable, Iterable
from typing import Any

from max.driver import CPU, Device
from max.experimental import tensor
from max.experimental.sharding import DeviceMapping
from max.experimental.tensor import Tensor
from max.graph import TensorValue, TensorValueLike, Type, ops
from max.graph.ops.slice_tensor import SliceIndices

from ..sharding.rules.conv import (
    conv2d_rule,
    conv2d_transpose_rule,
    conv3d_rule,
)
from ..sharding.rules.elementwise import (
    binary_rule,
    linear_binary_rule,
    linear_unary_rule,
    ternary_rule,
    unary_rule,
)
from ..sharding.rules.matmul import layer_norm_rule, matmul_rule
from ..sharding.rules.misc import (
    as_interleaved_complex_rule,
    band_part_rule,
    buffer_store_rule,
    buffer_store_slice_rule,
    cond_rule,
    fold_rule,
    irfft_rule,
    resize_linear_rule,
    resize_rule,
    while_loop_rule,
)
from ..sharding.rules.pooling import linear_pool_rule, pool_rule
from ..sharding.rules.reduction import linear_reduce_rule, reduce_rule
from ..sharding.rules.shape import (
    argsort_rule,
    broadcast_to_rule,
    chunk_rule,
    flatten_rule,
    gather_nd_rule,
    gather_rule,
    masked_scatter_rule,
    nonzero_rule,
    outer_rule,
    pad_rule,
    permute_rule,
    repeat_interleave_rule,
    reshape_rule,
    same_placement_multi_input_rule,
    scatter_add_rule,
    scatter_nd_add_rule,
    scatter_nd_rule,
    scatter_rule,
    slice_tensor_rule,
    split_rule,
    squeeze_rule,
    stack_rule,
    tile_rule,
    top_k_rule,
    transpose_rule,
    unsqueeze_rule,
)
from .collective_ops import transfer_to
from .creation_ops import full_like
from .utils import (
    any_distributed,
    ensure_context,
    map_tensors,
    tensor_to_layout,
    to_tensors,
)

# ═════════════════════════════════════════════════════════════════════════
#  spmd_dispatch — the per-shard dispatch engine
# ═════════════════════════════════════════════════════════════════════════


def spmd_dispatch(
    graph_op: Callable[..., Any],
    args: tuple[Any, ...],
    output_mappings: tuple[DeviceMapping, ...],
) -> Any:
    """Per-shard graph op dispatch for distributed tensors.

    Runs ``graph_op`` once per shard, extracting per-shard TensorValues
    from each distributed Tensor arg.  Reassembles the per-shard results
    into distributed Tensors.

    Also used by custom op dispatch (``_custom_dispatch.py``).
    """
    mesh = output_mappings[0].mesh
    n = mesh.num_devices

    with ensure_context():
        per_shard: list[Any] = []
        for i in builtins.range(n):

            def _get_shard(t: Tensor, _i: int = i) -> TensorValue:
                return t.local_shards[_i].__tensorvalue__()

            shard_args = map_tensors(_get_shard, args)
            per_shard.append(graph_op(*shard_args))

        first = per_shard[0]
        if first is None:
            return None

        if not isinstance(first, (list, tuple)):
            tvs = [TensorValue(s) for s in per_shard]
            return Tensor.from_shard_values(tvs, output_mappings[0])

        num_out = len(first)
        outputs: list[Tensor] = []
        for j in builtins.range(num_out):
            out_m = output_mappings[builtins.min(j, len(output_mappings) - 1)]
            tvs = [TensorValue(per_shard[i][j]) for i in builtins.range(n)]
            outputs.append(Tensor.from_shard_values(tvs, out_m))
        return type(first)(outputs)


def _transfer_args(
    args: tuple[Any, ...], suggested: tuple[Any, ...]
) -> tuple[Any, ...]:
    """Transfer Tensor args to match rule-suggested mappings.

    For Tensor args, calls ``transfer_to(tensor, mapping)``.
    For list/tuple args (e.g. concat's tensor list), walks into the
    container.  For non-Tensor args, uses the suggested value (which
    the rule may have modified, e.g. shape localization).
    """
    result: list[object] = []
    for orig, sugg in zip(args, suggested, strict=True):
        if isinstance(orig, Tensor):
            assert isinstance(sugg, (Device, DeviceMapping))
            result.append(transfer_to(orig, sugg))
        elif isinstance(orig, (list, tuple)):
            assert isinstance(sugg, (list, tuple))
            items = [
                transfer_to(o, s) if isinstance(o, Tensor) else s
                for o, s in zip(orig, sugg, strict=True)
            ]
            result.append(type(orig)(items))
        else:
            result.append(sugg)
    return tuple(result)


def functional(
    graph_op: Callable[..., Any],
    rule: Callable[..., tuple[tuple[Any, ...], tuple[DeviceMapping, ...]]]
    | None = None,
) -> Callable[..., Any]:
    """Wrap a graph op to work with Tensor inputs and optional SPMD sharding.

    Non-distributed path: calls ``graph_op`` directly inside an
    ``ensure_context()`` block.  Graph ops accept ``TensorValueLike``
    (which ``Tensor`` satisfies), so args pass through unchanged.
    Results are converted back to Tensors.

    Distributed path (when any input is distributed and a ``rule`` is
    provided): extracts TensorLayouts, calls the rule, transfers args
    to match, then dispatches per-shard via ``spmd_dispatch``.

    Uses ``functools.wraps`` so the wrapped function inherits the graph
    op's name and docstring.
    """
    # Pre-compute signature for canonicalizing kwargs → positional.
    sig = inspect.signature(graph_op)

    def _canonicalize(
        args: tuple[object, ...], kwargs: dict[str, object]
    ) -> tuple[object, ...]:
        """Bind args+kwargs into a single positional tuple with defaults."""
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return tuple(bound.args)

    @functools.wraps(graph_op)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if rule is not None and any_distributed(args):
            all_args = _canonicalize(args, kwargs)
            layout_args = map_tensors(tensor_to_layout, all_args)
            suggested, out_mappings = rule(*layout_args)
            redistributed = _transfer_args(all_args, suggested)
            return spmd_dispatch(graph_op, redistributed, out_mappings)

        with ensure_context():
            result = graph_op(*args, **kwargs)
            return to_tensors(result)

    return wrapped


# ═════════════════════════════════════════════════════════════════════════
#  Scalar promotion helper
# ═════════════════════════════════════════════════════════════════════════
# The underlying ``ops.add(tensor, 0.5)`` graph op accepts a Python number
# fine, but the SPMD rule-dispatch path in ``functional()`` expects every
# argument to be distributable (i.e. produce a ``TensorLayout`` via
# ``tensor_to_layout``). For binary ops that can take a scalar operand
# (``F.add(x, eps)`` inside RMSNorm, ``F.mul(x, 0.5)`` inside GELU, …) we
# promote the scalar to a ``full_like`` Tensor *before* dispatch so the
# rule always sees two layouts.


def _binary_with_scalar_promotion(
    inner: Callable[..., object],
) -> Callable[..., Tensor]:
    """Wrap a binary dispatch so that scalars are promoted to Tensors.

    Only the SPMD rule-dispatch path needs both args to be Tensors (so
    they produce a TensorLayout). The single-device graph-op path handles
    scalar + tensor natively — including int-dtype tensors, where eager
    ``full_like(int_tensor, 0.5)`` would bad_cast a float into an int
    constant. Gate the promotion on ``any_distributed`` to keep the
    native path intact.
    """

    def wrapper(lhs: Tensor | int | float, rhs: Tensor | int | float) -> Tensor:
        if any_distributed((lhs, rhs)):
            if isinstance(lhs, (int, float)) and isinstance(rhs, Tensor):
                lhs = full_like(rhs, float(lhs))
            elif isinstance(rhs, (int, float)) and isinstance(lhs, Tensor):
                rhs = full_like(lhs, float(rhs))
        result = inner(lhs, rhs)
        assert isinstance(result, Tensor)
        return result

    return wrapper


# ═════════════════════════════════════════════════════════════════════════
#  Elementwise — Binary (with scalar promotion)
# ═════════════════════════════════════════════════════════════════════════

#: Adds two tensors element-wise with SPMD distribution support.
#: Scalars are promoted to tensors automatically.
#: See :func:`max.graph.ops.add` for details.
add = _binary_with_scalar_promotion(
    functional(ops.add, rule=linear_binary_rule)
)
#: Subtracts two tensors element-wise with SPMD distribution support.
#: Scalars are promoted to tensors automatically.
#: See :func:`max.graph.ops.sub` for details.
sub = _binary_with_scalar_promotion(
    functional(ops.sub, rule=linear_binary_rule)
)
#: Multiplies two tensors element-wise with SPMD distribution support.
#: Scalars are promoted to tensors automatically.
#: See :func:`max.graph.ops.mul` for details.
mul = _binary_with_scalar_promotion(functional(ops.mul, rule=binary_rule))
#: Divides two tensors element-wise with SPMD distribution support.
#: Scalars are promoted to tensors automatically.
#: See :func:`max.graph.ops.div` for details.
div = _binary_with_scalar_promotion(functional(ops.div, rule=binary_rule))
#: Raises tensor elements to a power with SPMD distribution support.
#: Scalars are promoted to tensors automatically.
#: See :func:`max.graph.ops.pow` for details.
pow = _binary_with_scalar_promotion(functional(ops.pow, rule=binary_rule))
#: Computes the modulo operation element-wise with SPMD distribution support.
#: Scalars are promoted to tensors automatically.
#: See :func:`max.graph.ops.mod` for details.
mod = _binary_with_scalar_promotion(functional(ops.mod, rule=binary_rule))

# ═════════════════════════════════════════════════════════════════════════
#  Elementwise — Unary
# ═════════════════════════════════════════════════════════════════════════

#: Negates a tensor element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.negate` for details.
negate = functional(ops.negate, rule=linear_unary_rule)
#: Applies the ReLU activation function. Distributed via SPMD.
#: See :func:`max.graph.ops.relu` for details.
relu = functional(ops.relu, rule=unary_rule)
#: Computes the absolute value element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.abs` for details.
abs = functional(ops.abs, rule=unary_rule)
#: Computes the exponential element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.exp` for details.
exp = functional(ops.exp, rule=unary_rule)
#: Computes the natural logarithm element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.log` for details.
log = functional(ops.log, rule=unary_rule)
#: Computes the square root element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.sqrt` for details.
sqrt = functional(ops.sqrt, rule=unary_rule)
#: Computes the reciprocal square root element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.rsqrt` for details.
rsqrt = functional(ops.rsqrt, rule=unary_rule)
#: Applies the sigmoid activation function. Distributed via SPMD.
#: See :func:`max.graph.ops.sigmoid` for details.
sigmoid = functional(ops.sigmoid, rule=unary_rule)
#: Applies the SiLU (Swish) activation function. Distributed via SPMD.
#: See :func:`max.graph.ops.silu` for details.
silu = functional(ops.silu, rule=unary_rule)
#: Applies the GELU activation function. Distributed via SPMD.
#: See :func:`max.graph.ops.gelu` for details.
gelu = functional(ops.gelu, rule=unary_rule)
#: Computes the hyperbolic tangent element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.tanh` for details.
tanh = functional(ops.tanh, rule=unary_rule)
#: Computes the cosine element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.cos` for details.
cos = functional(ops.cos, rule=unary_rule)
#: Computes the sine element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.sin` for details.
sin = functional(ops.sin, rule=unary_rule)
#: Computes the error function element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.erf` for details.
erf = functional(ops.erf, rule=unary_rule)
#: Computes the floor element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.floor` for details.
floor = functional(ops.floor, rule=unary_rule)
#: Rounds tensor values element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.round` for details.
round = functional(ops.round, rule=unary_rule)
#: Truncates tensor values element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.trunc` for details.
trunc = functional(ops.trunc, rule=unary_rule)

#: Checks for infinite values element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.is_inf` for details.
is_inf = functional(ops.is_inf, rule=unary_rule)
#: Checks for NaN values element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.is_nan` for details.
is_nan = functional(ops.is_nan, rule=unary_rule)
#: Computes element-wise logical NOT. Distributed via SPMD.
#: See :func:`max.graph.ops.logical_not` for details.
logical_not = functional(ops.logical_not, rule=unary_rule)
#: Computes log(1 + x) element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.log1p` for details.
log1p = functional(ops.log1p, rule=unary_rule)
#: Computes the inverse hyperbolic tangent element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.atanh` for details.
atanh = functional(ops.atanh, rule=unary_rule)
#: Computes the inverse cosine element-wise. Distributed via SPMD.
#: See :func:`max.graph.ops.acos` for details.
acos = functional(ops.acos, rule=unary_rule)
#: Dequantizes a tensor. Distributed via SPMD.
#: See :func:`max.graph.ops.dequantize` for details.
dequantize = functional(ops.dequantize, rule=unary_rule)

# ═════════════════════════════════════════════════════════════════════════
#  Elementwise — Comparison / Logical
# ═════════════════════════════════════════════════════════════════════════

#: Computes element-wise equality comparison. Distributed via SPMD.
#: See :func:`max.graph.ops.equal` for details.
equal = _binary_with_scalar_promotion(functional(ops.equal, rule=binary_rule))
#: Computes element-wise inequality comparison. Distributed via SPMD.
#: See :func:`max.graph.ops.not_equal` for details.
not_equal = _binary_with_scalar_promotion(
    functional(ops.not_equal, rule=binary_rule)
)
#: Computes element-wise greater-than comparison. Distributed via SPMD.
#: See :func:`max.graph.ops.greater` for details.
greater = _binary_with_scalar_promotion(
    functional(ops.greater, rule=binary_rule)
)
#: Computes element-wise greater-than-or-equal comparison. Distributed via SPMD.
#: See :func:`max.graph.ops.greater_equal` for details.
greater_equal = _binary_with_scalar_promotion(
    functional(ops.greater_equal, rule=binary_rule)
)
#: Computes element-wise logical AND. Distributed via SPMD.
#: See :func:`max.graph.ops.logical_and` for details.
logical_and = _binary_with_scalar_promotion(
    functional(ops.logical_and, rule=binary_rule)
)
#: Computes element-wise logical OR. Distributed via SPMD.
#: See :func:`max.graph.ops.logical_or` for details.
logical_or = _binary_with_scalar_promotion(
    functional(ops.logical_or, rule=binary_rule)
)
#: Computes element-wise logical XOR. Distributed via SPMD.
#: See :func:`max.graph.ops.logical_xor` for details.
logical_xor = _binary_with_scalar_promotion(
    functional(ops.logical_xor, rule=binary_rule)
)

# ═════════════════════════════════════════════════════════════════════════
#  Elementwise — Ternary / Binary min-max
# ═════════════════════════════════════════════════════════════════════════

#: Selects elements from two tensors based on a condition. Distributed via SPMD.
#: See :func:`max.graph.ops.where` for details.
where = functional(ops.where, rule=ternary_rule)

#: Computes element-wise minimum of two tensors. Distributed via SPMD.
#: See ``max.graph.ops.elementwise.min`` for details.
elementwise_min = _binary_with_scalar_promotion(
    functional(ops.elementwise.min, rule=binary_rule)
)
#: Computes element-wise maximum of two tensors. Distributed via SPMD.
#: See ``max.graph.ops.elementwise.max`` for details.
elementwise_max = _binary_with_scalar_promotion(
    functional(ops.elementwise.max, rule=binary_rule)
)

# ═════════════════════════════════════════════════════════════════════════
#  Cast
# ═════════════════════════════════════════════════════════════════════════

#: Casts a tensor to a different data type. Distributed via SPMD.
#: See :func:`max.graph.ops.cast` for details.
cast = functional(ops.cast, rule=unary_rule)


# ═════════════════════════════════════════════════════════════════════════
#  Matmul / Linear Algebra
# ═════════════════════════════════════════════════════════════════════════

#: Performs matrix multiplication. Distributed via SPMD.
#: See :func:`max.graph.ops.matmul` for details.
matmul = functional(ops.matmul, rule=matmul_rule)

#: Applies layer normalization. Distributed via SPMD.
#: See :func:`max.graph.ops.layer_norm` for details.
layer_norm = functional(ops.layer_norm, rule=layer_norm_rule)
#: Performs quantized matrix multiplication. Distributed via SPMD.
#: See :func:`max.graph.ops.qmatmul` for details.
qmatmul = functional(ops.qmatmul, rule=matmul_rule)

# ═════════════════════════════════════════════════════════════════════════
#  Pooling
# ═════════════════════════════════════════════════════════════════════════

#: Applies 2D average pooling. Distributed via SPMD.
#: See :func:`max.graph.ops.avg_pool2d` for details.
avg_pool2d = functional(ops.avg_pool2d, rule=linear_pool_rule)
#: Applies 2D max pooling. Distributed via SPMD.
#: See :func:`max.graph.ops.max_pool2d` for details.
max_pool2d = functional(ops.max_pool2d, rule=pool_rule)

# ═════════════════════════════════════════════════════════════════════════
#  Shape
# ═════════════════════════════════════════════════════════════════════════

#: Permutes the dimensions of a tensor. Distributed via SPMD.
#: See :func:`max.graph.ops.permute` for details.
permute = functional(ops.permute, rule=permute_rule)
#: Transposes a tensor. Distributed via SPMD.
#: See :func:`max.graph.ops.transpose` for details.
transpose = functional(ops.transpose, rule=transpose_rule)
#: Adds a dimension of size 1. Distributed via SPMD.
#: See :func:`max.graph.ops.unsqueeze` for details.
unsqueeze = functional(ops.unsqueeze, rule=unsqueeze_rule)
#: Removes a dimension of size 1. Distributed via SPMD.
#: See :func:`max.graph.ops.squeeze` for details.
squeeze = functional(ops.squeeze, rule=squeeze_rule)
#: Reshapes a tensor to a new shape. Distributed via SPMD.
#: See :func:`max.graph.ops.reshape` for details.
reshape = functional(ops.reshape, rule=reshape_rule)
#: Flattens a tensor. Distributed via SPMD.
#: See :func:`max.graph.ops.flatten` for details.
flatten = functional(ops.flatten, rule=flatten_rule)
#: Tiles a tensor by repeating it. Distributed via SPMD.
#: See :func:`max.graph.ops.tile` for details.
tile = functional(ops.tile, rule=tile_rule)
#: Pads a tensor. Distributed via SPMD.
#: See :func:`max.graph.ops.pad` for details.
pad = functional(ops.pad, rule=pad_rule)
#: Broadcasts a tensor to a new shape. Distributed via SPMD.
#: See :func:`max.graph.ops.broadcast_to` for details.
broadcast_to = functional(ops.broadcast_to, rule=broadcast_to_rule)
#: Repeats elements of a tensor. Distributed via SPMD.
#: See :func:`max.graph.ops.repeat_interleave` for details.
repeat_interleave = functional(
    ops.repeat_interleave, rule=repeat_interleave_rule
)
#: Slices a tensor along specified dimensions. Distributed via SPMD.
#: See :func:`max.graph.ops.slice_tensor` for details.
slice_tensor = functional(ops.slice_tensor, rule=slice_tensor_rule)
#: Concatenates a list of tensors along an axis. Distributed via SPMD.
#: See :func:`max.graph.ops.concat` for details.
concat = functional(ops.concat, rule=same_placement_multi_input_rule)
#: Stacks tensors along a new dimension. Distributed via SPMD.
#: See :func:`max.graph.ops.stack` for details.
stack = functional(ops.stack, rule=stack_rule)
#: Returns the indices that would sort a tensor along an axis. Distributed via SPMD.
#: See :func:`max.graph.ops.argsort` for details.
argsort = functional(ops.argsort, rule=argsort_rule)
#: Returns the indices of non-zero elements. Distributed via SPMD.
#: See :func:`max.graph.ops.nonzero` for details.
nonzero = functional(ops.nonzero, rule=nonzero_rule)
#: Gathers values along an axis specified by indices. Distributed via SPMD.
#: See :func:`max.graph.ops.gather` for details.
gather = functional(ops.gather, rule=gather_rule)
#: Scatters values along an axis. Distributed via SPMD.
#: See :func:`max.graph.ops.scatter` for details.
scatter = functional(ops.scatter, rule=scatter_rule)
#: Scatters values and accumulates via addition. Distributed via SPMD.
#: See :func:`max.graph.ops.scatter_add` for details.
scatter_add = functional(ops.scatter_add, rule=scatter_add_rule)
#: Scatters values and accumulates via max. Distributed via SPMD.
#: See :func:`max.graph.ops.scatter_max` for details.
scatter_max = functional(ops.scatter_max, rule=scatter_add_rule)
#: Scatters values and accumulates via min. Distributed via SPMD.
#: See :func:`max.graph.ops.scatter_min` for details.
scatter_min = functional(ops.scatter_min, rule=scatter_add_rule)
#: Scatters values and accumulates via multiplication. Distributed via SPMD.
#: See :func:`max.graph.ops.scatter_mul` for details.
scatter_mul = functional(ops.scatter_mul, rule=scatter_add_rule)
#: Scatters values using multi-dimensional indices. Distributed via SPMD.
#: See :func:`max.graph.ops.scatter_nd` for details.
scatter_nd = functional(ops.scatter_nd, rule=scatter_nd_rule)
#: Scatters and accumulates via addition using multi-dimensional indices. Distributed via SPMD.
#: See :func:`max.graph.ops.scatter_nd_add` for details.
scatter_nd_add = functional(ops.scatter_nd_add, rule=scatter_nd_add_rule)
#: Scatters and accumulates via max using multi-dimensional indices. Distributed via SPMD.
#: See :func:`max.graph.ops.scatter_nd_max` for details.
scatter_nd_max = functional(ops.scatter_nd_max, rule=scatter_nd_add_rule)
#: Scatters and accumulates via min using multi-dimensional indices. Distributed via SPMD.
#: See :func:`max.graph.ops.scatter_nd_min` for details.
scatter_nd_min = functional(ops.scatter_nd_min, rule=scatter_nd_add_rule)
#: Scatters and accumulates via multiplication using multi-dimensional indices. Distributed via SPMD.
#: See :func:`max.graph.ops.scatter_nd_mul` for details.
scatter_nd_mul = functional(ops.scatter_nd_mul, rule=scatter_nd_add_rule)
#: Gathers values using multi-dimensional indices. Distributed via SPMD.
#: See :func:`max.graph.ops.gather_nd` for details.
gather_nd = functional(ops.gather_nd, rule=gather_nd_rule)
#: Scatters values according to a mask. Distributed via SPMD.
#: See :func:`max.graph.ops.masked_scatter` for details.
masked_scatter = functional(ops.masked_scatter, rule=masked_scatter_rule)
#: Computes the outer product of two vectors. Distributed via SPMD.
#: See :func:`max.graph.ops.outer` for details.
outer = functional(ops.outer, rule=outer_rule)

# Multi-output shape ops

_split_impl = functional(ops.split, rule=split_rule)


def split(
    x: Tensor,
    split_size_or_sections: int | list[int],
    axis: int = 0,
) -> list[Tensor]:
    """Split a tensor into chunks along an axis.

    When ``split_size_or_sections`` is an **int**, splits into equal chunks
    (last chunk may be smaller).  When it is a **list of ints**, splits
    into chunks with exactly those sizes.
    """
    if isinstance(split_size_or_sections, int):
        dim_size = int(x.shape[axis])
        chunk_size = split_size_or_sections
        num_full, remainder = divmod(dim_size, chunk_size)
        split_sizes: list[int] = [chunk_size] * num_full
        if remainder > 0:
            split_sizes.append(remainder)
    else:
        split_sizes = list(split_size_or_sections)
    return _split_impl(x, split_sizes, axis)


#: Returns the k largest elements along an axis. Distributed via SPMD.
#: See :func:`max.graph.ops.top_k` for details.
top_k = functional(ops.top_k, rule=top_k_rule)
#: Returns the k smallest elements along an axis. Distributed via SPMD.
#: See :func:`max.graph.ops.bottom_k` for details.
bottom_k = functional(ops.bottom_k, rule=top_k_rule)

#: Splits a tensor into chunks along a dimension. Distributed via SPMD.
#: See :func:`max.graph.ops.chunk` for details.
chunk = functional(ops.chunk, rule=chunk_rule)


# ═════════════════════════════════════════════════════════════════════════
#  Reduction helpers
# ═════════════════════════════════════════════════════════════════════════


def _reduce_op(
    graph_op: Callable[..., object],
    *,
    rule: Callable[..., tuple[tuple[object, ...], tuple[DeviceMapping, ...]]],
) -> Callable[..., Tensor]:
    """Build a reduction wrapper matching the old ``functional`` semantics.

    When ``axis`` is an ``int``, delegates directly to the single-axis
    graph op (via ``functional()``).

    When ``axis is None``, flattens the tensor to 1-D first, then reduces
    on axis 0 — producing shape ``[1]``.  This is pure syntactic sugar
    at the Tensor level; the graph op itself always reduces exactly one
    axis.
    """
    single_axis = functional(graph_op, rule=rule)

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
    *,
    rule: Callable[..., tuple[tuple[object, ...], tuple[DeviceMapping, ...]]],
    elementwise_fn: Callable[[Tensor, Tensor], Tensor],
) -> Callable[..., Tensor]:
    """Build a function that acts as reduction (1 arg) or elementwise (2 args).

    ``max(x)`` reduces along an axis; ``max(x, y)`` computes element-wise
    maximum.  The ``y`` argument disambiguates the two modes.

    Matches old ``functional`` semantics: ``axis=None`` flattens to 1-D
    then reduces on axis 0.
    """
    reduce_fn = _reduce_op(graph_op, rule=rule)

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


# ═════════════════════════════════════════════════════════════════════════
#  Reduction ops
# ═════════════════════════════════════════════════════════════════════════

#: Computes the sum along one or more axes. Distributed via SPMD.
#: See :func:`max.graph.ops.sum` for details.
sum = _reduce_op(ops.sum, rule=linear_reduce_rule)
#: Computes the mean along one or more axes. Distributed via SPMD.
#: See :func:`max.graph.ops.mean` for details.
mean = _reduce_op(ops.mean, rule=linear_reduce_rule)
#: Computes the product along one or more axes. Distributed via SPMD.
#: See :func:`max.graph.ops.prod` for details.
prod = _reduce_op(ops.prod, rule=reduce_rule)
_argmax_impl = _reduce_op(ops.argmax, rule=reduce_rule)
_argmin_impl = _reduce_op(ops.argmin, rule=reduce_rule)


def argmax(
    x: Tensor,
    axis: int | None = -1,
) -> Tensor:
    """Returns the indices of the maximum values along an axis.

    When ``axis is None``, flattens to 1-D first.
    Distributed via SPMD. See :func:`max.graph.ops.argmax` for details.
    """
    return _argmax_impl(x, axis=axis)


def argmin(
    x: Tensor,
    axis: int | None = -1,
) -> Tensor:
    """Returns the indices of the minimum values along an axis.

    When ``axis is None``, flattens to 1-D first.
    Distributed via SPMD. See :func:`max.graph.ops.argmin` for details.
    """
    return _argmin_impl(x, axis=axis)


#: Computes the maximum along one or more axes, or element-wise max of two tensors.
#: Distributed via SPMD. See ``max.graph.ops.reduction.max`` for details.
max = _reduce_elementwise_op(
    ops.reduction.max,
    rule=reduce_rule,
    elementwise_fn=elementwise_max,
)
#: Computes the minimum along one or more axes, or element-wise min of two tensors.
#: Distributed via SPMD. See ``max.graph.ops.reduction.min`` for details.
min = _reduce_elementwise_op(
    ops.reduction.min,
    rule=reduce_rule,
    elementwise_fn=elementwise_min,
)

#: Applies the softmax function along an axis. Distributed via SPMD.
#: See :func:`max.graph.ops.softmax` for details.
softmax = functional(ops.softmax, rule=reduce_rule)
#: Applies the log softmax function along an axis. Distributed via SPMD.
#: See :func:`max.graph.ops.logsoftmax` for details.
logsoftmax = functional(ops.logsoftmax, rule=reduce_rule)
#: Computes the cumulative sum along an axis. Distributed via SPMD.
#: See :func:`max.graph.ops.cumsum` for details.
cumsum = functional(ops.cumsum, rule=linear_reduce_rule)


# ═════════════════════════════════════════════════════════════════════════
#  Convolution
# ═════════════════════════════════════════════════════════════════════════


#: Applies 2D convolution. Distributed via SPMD.
#: See :func:`max.graph.ops.conv2d` for details.
conv2d = functional(ops.conv2d, rule=conv2d_rule)
#: Applies 3D convolution. Distributed via SPMD.
#: See :func:`max.graph.ops.conv3d` for details.
conv3d = functional(ops.conv3d, rule=conv3d_rule)
#: Applies 2D transposed convolution. Distributed via SPMD.
#: See :func:`max.graph.ops.conv2d_transpose` for details.
conv2d_transpose = functional(ops.conv2d_transpose, rule=conv2d_transpose_rule)


# ═════════════════════════════════════════════════════════════════════════
#  Misc
# ═════════════════════════════════════════════════════════════════════════

#: Copies a tensor setting everything outside a central band to zero. Distributed via SPMD.
#: See :func:`max.graph.ops.band_part` for details.
band_part = functional(ops.band_part, rule=band_part_rule)
#: Performs tensor folding operation. Distributed via SPMD.
#: See :func:`max.graph.ops.fold` for details.
fold = functional(ops.fold, rule=fold_rule)
#: Converts a tensor to interleaved complex representation. Distributed via SPMD.
#: See ``max.graph.ops.complex.as_interleaved_complex`` for details.
as_interleaved_complex = functional(
    ops.complex.as_interleaved_complex, rule=as_interleaved_complex_rule
)
#: Multiplies two complex-valued tensors. Distributed via SPMD.
#: See ``max.graph.ops.complex.mul`` for details.
complex_mul = functional(ops.complex.mul, rule=binary_rule)
#: Resizes a tensor. Distributed via SPMD.
#: See :func:`max.graph.ops.resize` for details.
resize = functional(ops.resize, rule=resize_rule)
#: Resizes a tensor using linear interpolation. Distributed via SPMD.
#: See :func:`max.graph.ops.resize_linear` for details.
resize_linear = functional(ops.resize_linear, rule=resize_linear_rule)
#: Resizes a tensor using nearest-neighbor interpolation. Distributed via SPMD.
#: See :func:`max.graph.ops.resize_nearest` for details.
resize_nearest = functional(ops.resize_nearest, rule=resize_rule)
#: Resizes a tensor using bicubic interpolation. Distributed via SPMD.
#: See :func:`max.graph.ops.resize_bicubic` for details.
resize_bicubic = functional(ops.resize_bicubic, rule=resize_rule)
#: Computes the inverse real FFT. Distributed via SPMD.
#: See :func:`max.graph.ops.irfft` for details.
irfft = functional(ops.irfft, rule=irfft_rule)


# ═════════════════════════════════════════════════════════════════════════
#  Control flow
# ═════════════════════════════════════════════════════════════════════════


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


#: Conditionally executes one of two branches based on a boolean predicate.
#: Distributed via SPMD. See :func:`max.graph.ops.cond` for details.
cond = functional(_cond_graph, rule=cond_rule)


def _while_loop_graph(
    initial_values: Iterable[TensorValueLike] | TensorValueLike,
    predicate: Callable[..., Tensor],
    body: Callable[..., Tensor | list[Tensor]],
) -> list[TensorValue]:
    """Wrap predicate/body so Tensor returns are unwrapped to TensorValue.

    ``ops.while_loop`` expects predicate and body functions that return
    :class:`TensorValue`, but our high-level API lets users return
    :class:`Tensor`.  This wrapper calls ``__tensorvalue__()`` on each
    returned Tensor before passing results to the graph op.
    """

    def _unwrap_list(
        vals: list[Tensor] | tuple[Tensor, ...],
    ) -> list[TensorValue]:
        return [v.__tensorvalue__() for v in vals]

    def _pred(*args: TensorValue) -> TensorValue:
        return predicate(*args).__tensorvalue__()

    def _body(*args: TensorValue) -> list[TensorValue]:
        result = body(*args)
        if isinstance(result, Tensor):
            return [result.__tensorvalue__()]
        return _unwrap_list(result)

    # ops.while_loop has no auto-coercion; coerce TensorValueLike (incl.
    # Tensor via __tensorvalue__) to TensorValue the same way _cond_graph
    # coerces its predicate.
    if isinstance(initial_values, Iterable):
        unwrapped = [TensorValue(v) for v in initial_values]
    else:
        unwrapped = [TensorValue(initial_values)]
    return ops.while_loop(unwrapped, _pred, _body)


#: Repeatedly executes a body function while a condition holds.
#: Distributed via SPMD. See :func:`max.graph.ops.while_loop` for details.
while_loop = functional(_while_loop_graph, rule=while_loop_rule)


# ═════════════════════════════════════════════════════════════════════════
#  Mutation ops
#
#  These are hand-rolled (not using ``functional()``) because they
#  mutate in-place via ``__buffervalue__()`` rather than returning a
#  new Tensor.
# ═════════════════════════════════════════════════════════════════════════


def buffer_store(destination: Tensor, source: Tensor) -> None:
    """Sets a tensor buffer to new values. Distributed via SPMD.

    See :func:`max.graph.ops.buffer_store` for details.
    """
    if destination.is_distributed:
        buffer_store_rule(
            tensor_to_layout(destination), tensor_to_layout(source)
        )

    with ensure_context():
        if destination.is_distributed:
            for i in builtins.range(len(destination.local_shards)):
                dest_shard = destination.local_shards[i].__buffervalue__()
                src_shard = (
                    source.local_shards[i].__tensorvalue__()
                    if source.is_distributed
                    else source.__tensorvalue__()
                )
                ops.buffer_store(dest_shard, src_shard)
        else:
            ops.buffer_store(
                destination.__buffervalue__(), source.__tensorvalue__()
            )


def buffer_store_slice(
    destination: Tensor,
    source: Tensor,
    indices: SliceIndices,
) -> None:
    """Sets a slice of a tensor buffer to new values. Distributed via SPMD.

    See :func:`max.graph.ops.buffer_store_slice` for details.
    """
    if destination.is_distributed:
        buffer_store_slice_rule(
            tensor_to_layout(destination), tensor_to_layout(source), indices
        )

    with ensure_context():
        if destination.is_distributed:
            for i in builtins.range(len(destination.local_shards)):
                dest_shard = destination.local_shards[i].__buffervalue__()
                src_shard = (
                    source.local_shards[i].__tensorvalue__()
                    if source.is_distributed
                    else source.__tensorvalue__()
                )
                dest_shard[indices] = src_shard
        else:
            dest_buf = destination.__buffervalue__()
            source_tv = source.__tensorvalue__()
            dest_buf[indices] = source_tv


# ═════════════════════════════════════════════════════════════════════════
#  Additional ops (no sharding rule — replicated-only for now)
# ═════════════════════════════════════════════════════════════════════════

#: Applies group normalization.
#: See :func:`max.graph.ops.group_norm` for details.
group_norm = functional(ops.group_norm)
#: Applies RMS normalization.
#: See :func:`max.graph.ops.rms_norm` for details.
rms_norm = functional(ops.rms_norm)
#: Filters boxes with high intersection-over-union.
#: See :func:`max.graph.ops.non_maximum_suppression` for details.
non_maximum_suppression = functional(ops.non_maximum_suppression)
#: Performs ROI Align pooling on an NHWC input tensor.
#: See :func:`max.graph.ops.roi_align` for details.
roi_align = functional(ops.roi_align)


def clamp(
    x: Tensor,
    lower_bound: TensorValueLike,
    upper_bound: TensorValueLike,
) -> Tensor:
    """Clamps tensor values to a specified range.

    Returns ``max(min(x, upper_bound), lower_bound)``.
    """
    return max(min(x, upper_bound), lower_bound)


#: Alias for :func:`clamp`.
clip = clamp
#: Rebinds the shape of a tensor, asserting dimension consistency at runtime.
#: See :func:`max.graph.ops.rebind` for details.
rebind = functional(ops.rebind)
