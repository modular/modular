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
"""MLIR bindings used by :mod:`max.graph`, plus Modular dialect registration.

Re-exports the MLIR Python binding objects (``Context``, ``Operation``,
``Module``, and friends). :func:`site_initialize` registers MAX's custom
dialects with MLIR and runs when the module is imported.
"""

from .._mlir._mlir_libs._mlir.ir import (
    ArrayAttr,
    Attribute,
    Block,
    BoolAttr,
    Context,
    Diagnostic,
    DictAttr,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    Location,
    MLIRError,
    Module,
    Operation,
    OpView,
    StringAttr,
    Type,
    TypeAttr,
    Value,
)


def site_initialize() -> None:
    """Registers MAX's custom MLIR dialects with the global registry.

    Runs automatically when :mod:`max.mlir` is imported.
    """
    from max._core import graph  # type: ignore[import-not-found]

    # This ignore is needed externally but not internally
    from .._mlir._mlir_libs import (  # type: ignore[attr-defined, unused-ignore]
        get_dialect_registry,
    )

    graph.load_modular_dialects(get_dialect_registry())


site_initialize()
