# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Elementwise ops."""

from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, ValueLike

# ===----------------------------------------------------------------------=== #
# Binary Ops
# ===----------------------------------------------------------------------=== #
# Note: Keep alphabetized.


def _elementwise_binary(op):
    def elementwise_op(lhs: ValueLike, rhs: ValueLike) -> TensorValue:
        return Graph.current._add_op(op, TensorValue(lhs), TensorValue(rhs))[
            0
        ].tensor

    elementwise_op.__name__ = op.__name__
    return elementwise_op


add = _elementwise_binary(rmo.add)
div = _elementwise_binary(rmo.div)
max = _elementwise_binary(rmo.max)
min = _elementwise_binary(rmo.min)
mod = _elementwise_binary(rmo.mod)
mul = _elementwise_binary(rmo.mul)
pow = _elementwise_binary(rmo.pow)
sub = _elementwise_binary(rmo.sub)
equal = _elementwise_binary(rmo.equal)
greater = _elementwise_binary(rmo.greater)
greater_equal = _elementwise_binary(rmo.greater_equal)
not_equal = _elementwise_binary(rmo.not_equal)

# ===----------------------------------------------------------------------=== #
# Unary  Ops
# ===----------------------------------------------------------------------=== #
# Note: Keep alphabetized.


def _elementwise_unary(op):
    def elementwise_op(x: TensorValue) -> TensorValue:
        return Graph.current._add_op(op, x._mlir_value.type, x)[0].tensor

    elementwise_op.__name__ = op.__name__
    return elementwise_op


abs = _elementwise_unary(rmo.mo_abs)
exp = _elementwise_unary(rmo.mo_exp)
erf = _elementwise_unary(rmo.mo_erf)
gelu = _elementwise_unary(rmo.mo_gelu)
log = _elementwise_unary(rmo.mo_log)
log1p = _elementwise_unary(rmo.mo_log1p)
logsoftmax = _elementwise_unary(rmo.mo_logsoftmax)
relu = _elementwise_unary(rmo.mo_relu)
sigmoid = _elementwise_unary(rmo.sigmoid)


def silu(x: TensorValue):
    return mul(x, sigmoid(x))


softmax = _elementwise_unary(rmo.mo_softmax)
cos = _elementwise_unary(rmo.mo_cos)
floor = _elementwise_unary(rmo.mo_floor)
round = _elementwise_unary(rmo.mo_round)
roundeven = _elementwise_unary(rmo.mo_roundeven)
rsqrt = _elementwise_unary(rmo.mo_rsqrt)
sqrt = _elementwise_unary(rmo.mo_sqrt)
sin = _elementwise_unary(rmo.mo_sin)
tanh = _elementwise_unary(rmo.mo_tanh)
trunc = _elementwise_unary(rmo.mo_trunc)
is_nan = _elementwise_unary(rmo.mo_is_nan)
is_inf = _elementwise_unary(rmo.mo_is_inf)
logical_not = _elementwise_unary(rmo.mo_not)
negate = _elementwise_unary(rmo.mo_negative)
