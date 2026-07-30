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

import mojo.importer

# Import op bindings from categorized Mojo modules
# matmul / unary-elementwise handlers are backed by graph-compiler models
# (compiled below), unlike the Mojo op bindings above.
from . import (  # type: ignore[attr-defined]
    band_part_gc,
    cast_gc,
    conv_gc,
    data_movement_ops,
    elementwise_binary_gc,
    gather_scatter_ops,
    gc_compile,
    group_norm_gc,
    layer_norm_gc,
    matmul_gc,
    nms_gc,
    nonzero_gc,
    pooling_gc,
    random_gc,
    range_gc,
    reduce_axis_gc,
    resize_gc,
    rms_norm_gc,
    roi_align_gc,
    select_gc,
    shape_rearrange_gc,
    topk_gc,
    unary_elementwise_gc,
)

# Import handlers after the op modules to avoid circular import issues:
# handlers.py imports the op bindings above (via the package).
# Re-export the warm-adoption query (from gc_compile) so a consumer can assert
# the ops were force-loaded from the manifest rather than cold-compiled.
from .gc_compile import adopted_from_manifest
from .handlers import _MO_OP_HANDLERS, lookup_handler, register_op_handler

# Every warm path iterates this; each ``*_gc.py`` self-registers at import
# (MXF-533), so there's no hand-maintained list to drift.
GC_FAMILIES: tuple[gc_compile.GCOpFamily, ...] = (
    gc_compile.registered_families()
)


def compile_all_families() -> None:
    """Compile every registered GC family's full sweep into the cache."""
    for family in GC_FAMILIES:
        family.compile_sweep()


# Opt-in (MAX_EAGER_OP_PRECOMPILE=1) precompile of the full GC matrix; lazy
# per-dispatch otherwise (MXF-508).
def _precompile_gc_models() -> None:
    if gc_compile.should_precompile():
        compile_all_families()


_precompile_gc_models()

__all__ = [
    "GC_FAMILIES",
    "_MO_OP_HANDLERS",
    "adopted_from_manifest",
    "compile_all_families",
    "lookup_handler",
    "register_op_handler",
]
