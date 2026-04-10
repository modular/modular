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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

"""MAX Graph Python bindings."""

import os
import pathlib
from collections.abc import Sequence

import max._core
import max._core.dialects.builtin
import max._core.dialects.m
import max._core.dialects.mo
import max._core.driver
import max._core.dtype
import max._mlir.ir

def load_modular_dialects(arg: max._mlir.ir.DialectRegistry, /) -> None:
    """Registers all Modular MLIR dialects into the given registry."""

def array_attr(
    arg0: max._core.driver.Buffer, arg1: max._core.dialects.mo.TensorType, /
) -> max._core.dialects.m.ArrayElementsAttr:
    """Creates an array attribute from a buffer and tensor type."""

def _buffer_from_constant_attr(
    attr: max._core.dialects.builtin.ElementsAttr,
    dtype: max._core.dtype.DType,
    shape: Sequence[int],
    device: max._core.driver.Device,
) -> max._core.driver.Buffer:
    """Creates a Buffer from an MLIR constant attribute."""

def next_operation(arg: max._mlir.ir.Operation, /) -> max._mlir.ir.Operation:
    """Returns the next operation in the parent block, or None."""

def prev_operation(arg: max._mlir.ir.Operation, /) -> max._mlir.ir.Operation:
    """Returns the previous operation in the parent block, or None."""

def last_operation(arg: max._mlir.ir.Block, /) -> max._mlir.ir.Operation:
    """Returns the last operation in the given block, or None."""

def dtype_to_type(arg: max._core.dtype.DType, /) -> max._core.Type:
    """Converts a MAX DType to the corresponding MLIR type."""

def type_to_dtype(arg: max._core.Type, /) -> max._core.dtype.DType:
    """Converts an MLIR type to the corresponding MAX DType."""

def frame_loc(
    arg0: max._mlir.ir.Context, arg1: object, /
) -> max._mlir.ir.Location:
    """Creates an opaque MLIR location containing a Python stack frame."""

def _init_and_register_max_context(mlir_ctx: max._mlir.ir.Context) -> None:
    """
    Initializes a process-wide M::Context when none exists, then registers it with the given MLIR context so compiler code can use loadContext.

    Internal API; does not expose M::Context to Python. New contexts are
    created via Init::getOrCreateContext (same Init path the Engine uses before
    attaching devices).
    """

class Analysis:
    """
    Analyzes Mojo kernel libraries for custom operator integration.

    Loads one or more Mojo shared libraries and exposes their exported
    kernel operations for use in MAX graphs. Used internally to validate
    and register custom operations.
    """

    def __init__(
        self, context: object, paths: Sequence[str | os.PathLike]
    ) -> None:
        """
        Creates an analysis by loading the given Mojo libraries.

        Args:
            context: A Mojo library object or path.
            paths: Additional library search paths.
        """

    @property
    def symbol_names(self) -> list[str]:
        """Returns the list of exported kernel symbol names."""

    @property
    def library_paths(self) -> list[pathlib.Path]:
        """Returns the paths of all loaded libraries."""

    def kernel(self, arg: str, /) -> max._core.Operation:
        """Returns the MLIR operation for the named kernel symbol."""

    def verify_custom_op(self, arg: max._mlir.ir.Operation, /) -> None:
        """Verifies that an operation matches the expected kernel signature."""

    def add_path(self, arg: str | os.PathLike, /) -> None:
        """Adds a library search path to this analysis."""
