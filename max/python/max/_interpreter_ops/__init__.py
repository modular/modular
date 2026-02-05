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

# Import op bindings from Mojo module
from . import mojo_ops  # type: ignore[attr-defined]

# Arithmetic binary ops: output dtype matches input dtype
# Dtype dispatch is handled in Mojo


BINARY_ELEMENTWISE: dict[
    type[_core.Operation], Callable[[Buffer, Buffer, Buffer, int], None]
] = {
    mo.AddOp: mojo_ops.Add,
    mo.SubOp: mojo_ops.Sub,
    mo.MulOp: mojo_ops.Mul,
    mo.DivOp: mojo_ops.Div,
    mo.ModOp: mojo_ops.Mod,
    mo.MaxOp: mojo_ops.Max,
    mo.MinOp: mojo_ops.Min,
    mo.AndOp: mojo_ops.And,
    mo.OrOp: mojo_ops.Or,
    mo.XorOp: mojo_ops.Xor,
}

# Comparison binary ops: output dtype is always bool
BINARY_ELEMENTWISE_COMPARISON: dict[
    type[_core.Operation], Callable[[Buffer, Buffer, Buffer, int], None]
] = {
    mo.EqualOp: mojo_ops.Equal,
    mo.GreaterOp: mojo_ops.Greater,
    mo.GreaterEqualOp: mojo_ops.GreaterEqual,
    mo.NotEqualOp: mojo_ops.NotEqual,
}

# Unary elementwise ops: output dtype matches input dtype
UNARY_ELEMENTWISE: dict[
    type[_core.Operation], Callable[[Buffer, Buffer, int], None]
] = {
    mo.NegativeOp: mojo_ops.Negative,
    mo.AbsOp: mojo_ops.Abs,
    mo.ReluOp: mojo_ops.ReLU,
    mo.CeilOp: mojo_ops.Ceil,
    mo.FloorOp: mojo_ops.Floor,
    mo.RoundOp: mojo_ops.Round,
    mo.ExpOp: mojo_ops.Exp,
    mo.LogOp: mojo_ops.Log,
    mo.Log1pOp: mojo_ops.Log1p,
    mo.SqrtOp: mojo_ops.Sqrt,
    mo.RsqrtOp: mojo_ops.Rsqrt,
    mo.TanhOp: mojo_ops.Tanh,
    mo.AtanhOp: mojo_ops.ATanh,
    mo.TruncOp: mojo_ops.Trunc,
    mo.SinOp: mojo_ops.Sin,
    mo.CosOp: mojo_ops.Cos,
    mo.ErfOp: mojo_ops.Erf,
    mo.NotOp: mojo_ops.Not,
}

# Import handlers after defining kernels to avoid circular import issues.
# handlers.py uses the kernel dictionaries defined above.
from .handlers import _MO_OP_HANDLERS, lookup_handler, register_op_handler

__all__ = [
    "BINARY_ELEMENTWISE",
    "BINARY_ELEMENTWISE_COMPARISON",
    "UNARY_ELEMENTWISE",
    "_MO_OP_HANDLERS",
    "lookup_handler",
    "register_op_handler",
]
