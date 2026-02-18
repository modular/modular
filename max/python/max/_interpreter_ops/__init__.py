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

"""Python bindings for the MO interpreter ops.

This module defines the operation handler registry and the Mojo op bindings
for the MO graph interpreter.
"""

from collections.abc import Callable

import mojo.importer
from max import _core
from max._core.dialects import mo
from max._core.driver import Buffer

# Import op bindings from categorized Mojo modules
from . import (  # type: ignore[attr-defined]
    broadcast_ops,
    elementwise_ops,
    matmul_ops,
    misc_ops,
    reduce_ops,
    softmax_ops,
)

# Arithmetic binary ops: output dtype matches input dtype
# Dtype dispatch is handled in Mojo


BINARY_ELEMENTWISE: dict[
    type[_core.Operation], Callable[[Buffer, Buffer, Buffer, int], None]
] = {
    mo.AddOp: elementwise_ops.Add,
    mo.SubOp: elementwise_ops.Sub,
    mo.MulOp: elementwise_ops.Mul,
    mo.DivOp: elementwise_ops.Div,
    mo.ModOp: elementwise_ops.Mod,
    mo.MaxOp: elementwise_ops.Max,
    mo.MinOp: elementwise_ops.Min,
    mo.AndOp: elementwise_ops.And,
    mo.OrOp: elementwise_ops.Or,
    mo.XorOp: elementwise_ops.Xor,
    mo.PowOp: elementwise_ops.Pow,
}

# Comparison binary ops: output dtype is always bool
BINARY_ELEMENTWISE_COMPARISON: dict[
    type[_core.Operation], Callable[[Buffer, Buffer, Buffer, int], None]
] = {
    mo.EqualOp: elementwise_ops.Equal,
    mo.GreaterOp: elementwise_ops.Greater,
    mo.GreaterEqualOp: elementwise_ops.GreaterEqual,
    mo.NotEqualOp: elementwise_ops.NotEqual,
}

# Unary elementwise ops: output dtype matches input dtype
UNARY_ELEMENTWISE: dict[
    type[_core.Operation], Callable[[Buffer, Buffer, int], None]
] = {
    mo.NegativeOp: elementwise_ops.Negative,
    mo.AbsOp: elementwise_ops.Abs,
    mo.ReluOp: elementwise_ops.ReLU,
    mo.CeilOp: elementwise_ops.Ceil,
    mo.FloorOp: elementwise_ops.Floor,
    mo.RoundOp: elementwise_ops.Round,
    mo.ExpOp: elementwise_ops.Exp,
    mo.LogOp: elementwise_ops.Log,
    mo.Log1pOp: elementwise_ops.Log1p,
    mo.SqrtOp: elementwise_ops.Sqrt,
    mo.RsqrtOp: elementwise_ops.Rsqrt,
    mo.TanhOp: elementwise_ops.Tanh,
    mo.AtanhOp: elementwise_ops.ATanh,
    mo.TruncOp: elementwise_ops.Trunc,
    mo.SinOp: elementwise_ops.Sin,
    mo.CosOp: elementwise_ops.Cos,
    mo.ErfOp: elementwise_ops.Erf,
    mo.NotOp: elementwise_ops.Not,
}

# Reduce ops: reduce along an axis, output shape has reduced dim = 1
REDUCE: dict[
    type[_core.Operation], Callable[[Buffer, Buffer, int, int], None]
] = {
    mo.ReduceMaxOp: reduce_ops.ReduceMax,
    mo.ReduceMinOp: reduce_ops.ReduceMin,
    mo.ReduceAddOp: reduce_ops.ReduceAdd,
    mo.MeanOp: reduce_ops.Mean,
    mo.ReduceMulOp: reduce_ops.ReduceMul,
}

# Unary mixed-dtype ops: output dtype differs from input dtype
# IsNan, IsInf: float input -> bool output
# Cast: any dtype input -> any dtype output
UNARY_MIXED: dict[
    type[_core.Operation], Callable[[Buffer, Buffer, int], None]
] = {
    mo.CastOp: elementwise_ops.Cast,
    mo.IsNanOp: elementwise_ops.IsNan,
    mo.IsInfOp: elementwise_ops.IsInf,
}

# Softmax ops: output shape matches input, applied along an axis
SOFTMAX: dict[
    type[_core.Operation], Callable[[Buffer, Buffer, int, int], None]
] = {
    mo.SoftmaxOp: softmax_ops.Softmax,
    mo.LogsoftmaxOp: softmax_ops.LogSoftmax,
}

# Import handlers after defining kernels to avoid circular import issues.
# handlers.py uses the kernel dictionaries defined above.
from .handlers import _MO_OP_HANDLERS, lookup_handler, register_op_handler

__all__ = [
    "BINARY_ELEMENTWISE",
    "BINARY_ELEMENTWISE_COMPARISON",
    "REDUCE",
    "SOFTMAX",
    "UNARY_ELEMENTWISE",
    "UNARY_MIXED",
    "_MO_OP_HANDLERS",
    "lookup_handler",
    "register_op_handler",
]
