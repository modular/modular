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
# matmul / unary-elementwise handlers are backed by graph-compiler models
# (compiled below), unlike the Mojo op bindings above.
from . import (  # type: ignore[attr-defined]
    argnonzero_ops,
    avg_pool_ops,
    band_part_ops,
    bottomk_ops,
    conv_ops,
    data_movement_ops,
    elementwise_binary_gc,
    elementwise_cast_ops,
    gather_scatter_ops,
    gc_compile,
    group_norm_ops,
    layer_norm_ops,
    matmul_gc,
    misc_ops,
    nms_ops,
    pooling_ops,
    reduce_axis_gc,
    resize_ops,
    rms_norm_ops,
    roi_align_ops,
    select_ops,
    shape_rearrange_gc,
    topk_ops,
    unary_elementwise_gc,
)

# Cast: any dtype input -> any dtype output. (IsNan/IsInf now route through the
# graph compiler; see unary_elementwise_gc.)
UNARY_MIXED: dict[
    type[_core.Operation], Callable[[Buffer, Buffer, int], None]
] = {
    mo.CastOp: elementwise_cast_ops.Cast,
}

# Import handlers after defining kernels to avoid circular import issues.
# handlers.py uses the kernel dictionaries defined above.
from .handlers import _MO_OP_HANDLERS, lookup_handler, register_op_handler


# Opt-in (MAX_EAGER_OP_PRECOMPILE=1) precompile of the full GC matrix; lazy
# per-dispatch otherwise (MXF-508). Wrapped in a function to defer the
# matmul_gc / unary_gc symbol access past their import cycle with this module.
def _precompile_gc_models() -> None:
    if gc_compile.should_precompile():
        matmul_gc.compile_matmul_sweep()
        unary_elementwise_gc.compile_unary_sweep()
        elementwise_binary_gc.compile_binary_sweep()
        reduce_axis_gc.compile_reduce_axis_sweep()
        shape_rearrange_gc.compile_shape_rearrange_sweep()


_precompile_gc_models()

__all__ = [
    "UNARY_MIXED",
    "_MO_OP_HANDLERS",
    "lookup_handler",
    "register_op_handler",
]
