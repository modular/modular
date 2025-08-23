# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import functools
from typing import Callable, TypeVar

from typing_extensions import ParamSpec

from .. import driver
from ..graph import Graph, TensorValue, ops
from . import tensor

Args = ParamSpec("Args")
Result = TypeVar("Result")
Op = Callable[Args, Result]


def functional(op: Op) -> Op:
    """Convert an op on symbolic tensor values to one which
    additionally supports max.Tensor inputs."""

    def to_tensor(
        result: driver.Tensor | tensor.Tensor | TensorValue,
    ) -> tensor.Tensor:
        if isinstance(result, tensor.Tensor):
            return result
        if isinstance(result, driver.Tensor):
            return tensor.Tensor(storage=result)
        return tensor.Tensor.from_tensor_value(result)

    @functools.wraps(op)
    def wrapped(*args, **kwargs):
        # No-op for graph construction
        try:
            _ = Graph.current
        except LookupError:
            pass
        else:
            return op(*args, **kwargs)

        # No graph, use Tensor compute graph.
        with tensor.GRAPH.graph:
            results = op(*args, **kwargs)
        if isinstance(results, (driver.Tensor, tensor.Tensor, TensorValue)):
            return to_tensor(results)
        return [to_tensor(result) for result in results]

    return wrapped


abs = functional(ops.abs)
add = functional(ops.add)
argmax = functional(ops.argmax)
argmin = functional(ops.argmin)
argsort = functional(ops.argsort)
as_interleaved_complex = functional(ops.as_interleaved_complex)
atanh = functional(ops.atanh)
avg_pool2d = functional(ops.avg_pool2d)
band_part = functional(ops.band_part)
broadcast_to = functional(ops.broadcast_to)
cast = functional(ops.cast)
chunk = functional(ops.chunk)
constant = functional(ops.constant)
conv2d = functional(ops.conv2d)
conv2d_transpose = functional(ops.conv2d_transpose)
conv3d = functional(ops.conv3d)
cos = functional(ops.cos)
cumsum = functional(ops.cumsum)
custom = functional(ops.custom)
div = functional(ops.div)
equal = functional(ops.equal)
erf = functional(ops.erf)
exp = functional(ops.exp)
flatten = functional(ops.flatten)
floor = functional(ops.floor)
fold = functional(ops.fold)
gather = functional(ops.gather)
gather_nd = functional(ops.gather_nd)
gelu = functional(ops.gelu)
greater = functional(ops.greater)
greater_equal = functional(ops.greater_equal)
hann_window = functional(ops.hann_window)
irfft = functional(ops.irfft)
is_inf = functional(ops.is_inf)
is_nan = functional(ops.is_nan)
layer_norm = functional(ops.layer_norm)
log = functional(ops.log)
log1p = functional(ops.log1p)
logical_and = functional(ops.logical_and)
logical_not = functional(ops.logical_not)
logical_or = functional(ops.logical_or)
logical_xor = functional(ops.logical_xor)
logsoftmax = functional(ops.logsoftmax)
masked_scatter = functional(ops.masked_scatter)
matmul = functional(ops.matmul)
max = functional(ops.max)
max_pool2d = functional(ops.max_pool2d)
mean = functional(ops.mean)
min = functional(ops.min)
mod = functional(ops.mod)
mul = functional(ops.mul)
negate = functional(ops.negate)
nonzero = functional(ops.nonzero)
not_equal = functional(ops.not_equal)
outer = functional(ops.outer)
pad = functional(ops.pad)
permute = functional(ops.permute)
pow = functional(ops.pow)
range = functional(ops.range)
relu = functional(ops.relu)
repeat_interleave = functional(ops.repeat_interleave)
reshape = functional(ops.reshape)
round = functional(ops.round)
rsqrt = functional(ops.rsqrt)
scatter = functional(ops.scatter)
scatter_nd = functional(ops.scatter_nd)
sigmoid = functional(ops.sigmoid)
silu = functional(ops.silu)
sin = functional(ops.sin)
slice_tensor = functional(ops.slice_tensor)
softmax = functional(ops.softmax)
split = functional(ops.split)
sqrt = functional(ops.sqrt)
squeeze = functional(ops.squeeze)
stack = functional(ops.stack)
sub = functional(ops.sub)
sum = functional(ops.sum)
tanh = functional(ops.tanh)
tile = functional(ops.tile)
top_k = functional(ops.top_k)
transfer_to = functional(ops.transfer_to)
transpose = functional(ops.transpose)
trunc = functional(ops.trunc)
unsqueeze = functional(ops.unsqueeze)
where = functional(ops.where)
